"""Random-initialization training with bounded resources, actual CSV curves and validation checkpoints."""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.ml.models import create_model,landmark_loss
from backend.ml.evaluation import errors,summarize
from training.data import LandmarkDataset


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)


@torch.inference_mode()
def validate(model,loader,device,dataset):
    model.eval()
    all_nme,all_px = [],[]
    total_loss = 0.0
    for image,target,side in loader:
        image,target = image.to(device),target.to(device)
        output = model(image)
        total_loss += float(landmark_loss(output,target))*len(image)
        nme,local = errors(output["coordinates"].cpu().numpy(),target.cpu().numpy(),dataset)
        all_nme.extend(nme)
        all_px.extend(local*side.numpy())
    metrics = summarize(np.array(all_nme),np.array(all_px))
    return total_loss/len(loader.dataset),metrics


def train(args):
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    train_set = LandmarkDataset(args.data,"train",augment=True)
    val_set = LandmarkDataset(args.data,"val")
    if not len(train_set) or not len(val_set):
        raise ValueError("Import train and validation data before training.")
    # Fixed generator for reproducible shuffle; workers=0 avoids Windows spawn and RNG surprises.
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=0,generator=generator)
    val_loader = DataLoader(val_set,batch_size=args.batch_size,num_workers=0)
    model = create_model(args.architecture,train_set.topology.count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sample = next(iter(train_loader))
    image,target = sample[0].to(device),sample[1].to(device)
    started = time.perf_counter()
    # A single forward/backward benchmark, without updating weights.
    landmark_loss(model(image),target).backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    batch_seconds = time.perf_counter()-started
    optimizer.zero_grad(set_to_none=True)
    estimate = {"architecture":args.architecture,"parameters":parameters,"device":str(device),"threads":args.threads,
                "batch_size":len(image),"benchmark_batch_seconds":batch_seconds,
                "approx_training_minutes":batch_seconds*len(train_loader)*args.epochs/60,
                "note":"Approximation includes one forward/backward batch, excludes decoding, validation and startup variation. No paid compute is used."}
    print(json.dumps(estimate,indent=2),flush=True)
    if args.estimate:
        return estimate
    args.out.mkdir(parents=True,exist_ok=True)
    metadata = train_set.metadata
    manifest = (args.data/"manifest.jsonl").read_bytes()
    if hashlib.sha256(manifest).hexdigest()!=metadata["manifest_sha256"]:
        raise ValueError("Manifest differs from imported fingerprint; import again.")
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (args.out/"config.json").write_text(json.dumps({**config,"estimate":estimate,"dataset":metadata},indent=2),encoding="utf-8")
    deadline = time.monotonic()+args.max_minutes*60
    best = float("inf")
    history = []
    training_steps = 0
    for epoch in range(1,args.epochs+1):
        if time.monotonic()>deadline:
            break
        model.train()
        loss_sum,samples = 0.0,0
        start = time.perf_counter()
        completed = True
        for image,target,_ in train_loader:
            if time.monotonic()>deadline:
                completed = False
                break
            image,target = image.to(device),target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = landmark_loss(model(image),target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; checkpoint not saved.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            training_steps += 1
            loss_sum += float(loss.detach())*len(image)
            samples += len(image)
        if not samples:
            break
        val_loss,metrics = validate(model,val_loader,device,metadata["dataset"])
        row = {"epoch":epoch,"train_loss":loss_sum/samples,"validation_loss":val_loss,
               "validation_nme":metrics["nme"],"seconds":time.perf_counter()-start,"complete_epoch":completed}
        history.append(row)
        with (args.out/"history.csv").open("w",newline="",encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row),flush=True)
        if metrics["nme"] < best:
            best = metrics["nme"]
            checkpoint = {"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                          "architecture":args.architecture,"dataset":metadata["dataset"],"landmarks":train_set.topology.count,
                          "input_size":96,"initialization":"random_from_scratch","training_steps":training_steps,"step_count_scope":"cumulative",
                          "epochs":epoch,"complete_epoch":completed,"seed":args.seed,"parameters":parameters,
                          "manifest_sha256":metadata["manifest_sha256"],"validation":metrics,"selection":"minimum validation NME",
                          "preprocessing":{"margin":0.12,"normalization":"RGB / 127.5 − 1","size":96},"torch_version":str(torch.__version__)}
            temporary = args.out/"best.tmp"
            torch.save(checkpoint,temporary)
            temporary.replace(args.out/"best.pt")
        if not completed:
            break
    if history:
        from training.visualize import plot_history
        plot_history(args.out/"history.csv",args.out/"loss.png")
    print(f"Finished. Completed {len(history)} validation passes. Best validation NME: {best:.6f}",flush=True)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("training/datasets/prepared"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--architecture",choices=["coordinates","heatmaps"],required=True)
    parser.add_argument("--epochs",type=int,default=30)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--lr",type=float,default=0.001)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--max-minutes",type=float,default=15)
    parser.add_argument("--estimate",action="store_true")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if min(args.epochs,args.batch_size,args.threads,args.max_minutes,args.lr)<=0:
        parser.error("Resource and training parameters must be positive.")
    try:
        train(args)
    except (FileNotFoundError,ValueError) as error:
        parser.exit(2,f"Training cannot start: {error}\n")

"""Train a face-framing classifier from random weights on WFLW crops and background/loose crops."""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader,TensorDataset

from backend.ml.regions import RegionCNN,RegionFinder,iou,padded_box,patch


def build_cache(directory,split,limit,seed,miner=None):
    metadata=json.loads((directory/"dataset.json").read_text(encoding="utf-8"))
    groups=defaultdict(list)
    for line in (directory/"manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row=json.loads(line)
        if row["split"]==split:
            groups[row["image_sha256"]].append(row)
    rng=np.random.default_rng(seed)
    names=sorted(groups)
    rng.shuffle(names)
    images,labels=[],[]
    used=[]
    for name in names:
        rows=groups[name]
        boxes=[]
        for row in rows:
            points=np.asarray(row["points"])
            lo,hi=points.min(0),points.max(0)
            boxes.append([*lo,*(hi-lo)])
        primary=max(boxes,key=lambda b:b[2]*b[3])
        if min(primary[2:])<40:
            continue
        with Image.open(Path(metadata["root"])/rows[0]["image"]) as image:
            rgb=np.asarray(image.convert("RGB")).copy()
        h,w=rgb.shape[:2]
        if primary[0]<0 or primary[1]<0 or primary[0]+primary[2]>w or primary[1]+primary[3]>h:
            continue
        face_window=padded_box(primary)
        images.append(patch(rgb,face_window)); labels.append(1.)
        if miner is not None:
            # Randomly shifted/scaled positive crops teach sliding-window tolerance.
            for _ in range(3):
                side=face_window[2]*float(rng.uniform(.9,1.12))
                cx=face_window[0]+face_window[2]/2+float(rng.uniform(-.06,.06))*side
                cy=face_window[1]+face_window[3]/2+float(rng.uniform(-.06,.06))*side
                images.append(patch(rgb,[cx-side/2,cy-side/2,side,side])); labels.append(1.)
            # Mine hard backgrounds using ONLY the previous model trained on train.
            candidates=[]
            for _ in range(80):
                side=float(rng.uniform(.16,.95)*min(h,w))
                candidate=[float(rng.uniform(0,max(1,w-side))),float(rng.uniform(0,max(1,h-side))),side,side]
                if max(iou(candidate,padded_box(b)) for b in boxes)<.2:
                    candidates.append(candidate)
            if candidates:
                scores=miner.score_boxes(rgb,candidates)
                for index in np.argsort(scores)[-12:]:
                    images.append(patch(rgb,candidates[index])); labels.append(0.)
        # Loose framing is deliberately negative, even when it contains a small face.
        loose=padded_box(primary,float(rng.uniform(2.4,3.5)))
        images.append(patch(rgb,loose)); labels.append(0.)
        for _ in range(40):
            side=float(rng.uniform(.15,.70)*min(h,w))
            candidate=[float(rng.uniform(0,max(1,w-side))),float(rng.uniform(0,max(1,h-side))),side,side]
            if max(iou(candidate,padded_box(b)) for b in boxes)<.08:
                images.append(patch(rgb,candidate)); labels.append(0.)
                break
        used.append(name)
        if len(used)%400==0:
            print(f"{split}: {len(used)} images, {len(images)} crops",flush=True)
        if len(used)>=limit:
            break
    if not used:
        raise ValueError("No usable face crops.")
    return TensorDataset(torch.from_numpy(np.stack(images).transpose(0,3,1,2).copy()),torch.tensor(labels)),used,metadata


@torch.inference_mode()
def evaluate(model,loader):
    model.eval()
    scores,labels=[],[]
    losses=[]
    for images,target in loader:
        logits=model(images.float()/127.5-1)
        losses.extend(torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction="none").numpy())
        scores.extend(logits.sigmoid().numpy()); labels.extend(target.numpy())
    s,y=np.array(scores),np.array(labels).astype(bool)
    positive,negative=s[y],s[~y]
    auc=float(((positive[:,None]>negative).mean()+.5*(positive[:,None]==negative).mean()))
    pred=s>=.90
    return {"samples":len(y),"loss":float(np.mean(losses)),"roc_auc":auc,"threshold":.90,
            "precision":float((pred&y).sum()/max(pred.sum(),1)),"recall":float((pred&y).sum()/max(y.sum(),1)),
            "accuracy":float((pred==y).mean()),"positive_crops":int(y.sum()),"negative_crops":int((~y).sum())}


def main(args):
    torch.set_num_threads(4); cv2.setNumThreads(1)
    random.seed(52); np.random.seed(52); torch.manual_seed(52)
    model=RegionCNN()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    x=torch.zeros(64,3,96,96); y=torch.zeros(64)
    start=time.perf_counter()
    torch.nn.functional.binary_cross_entropy_with_logits(model(x),y).backward()
    elapsed=time.perf_counter()-start
    optimizer.zero_grad(set_to_none=True)
    crops_per_image=18 if args.hard_negatives else 3
    print(json.dumps({"parameters":sum(p.numel() for p in model.parameters()),"benchmark_batch_seconds":elapsed,
                      "approx_training_minutes_excluding_cache":elapsed*(args.limit*crops_per_image/64)*args.epochs/60,
                      "cache_mb_approx":args.limit*crops_per_image*96*96*3/1024**2,"paid_compute":False,
                      "excludes":"image decoding, hard-negative mining, validation, test, report writing"}),flush=True)
    if args.estimate:
        return
    miner=None
    if args.hard_negatives:
        miner=RegionFinder(args.hard_negatives)
        if not miner.status()["ready"]:
            raise ValueError("Mining checkpoint is unavailable.")
    train,train_ids,metadata=build_cache(args.data,"train",args.limit,52,miner)
    val,val_ids,_=build_cache(args.data,"val",200,53,miner)
    assert not set(train_ids)&set(val_ids)
    train_loader=DataLoader(train,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(52))
    val_loader=DataLoader(val,batch_size=128)
    args.out.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+args.max_minutes*60
    best=float("inf"); history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for images,target in train_loader:
            if time.monotonic()>deadline:
                break
            image=images.float()/127.5-1
            if torch.rand(())<.5:
                image=image.flip(-1)
            image=(image*float(np.random.uniform(.8,1.2))+float(np.random.uniform(-.1,.1))).clamp(-1,1)
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(image),target)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        if not losses:
            break
        metric=evaluate(model,val_loader)
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"validation":metric})
        print(json.dumps(history[-1]),flush=True)
        if metric["loss"]<best:
            best=metric["loss"]
            ck={"state_dict":{k:v.detach().clone() for k,v in model.state_dict().items()},"epochs":epoch,
                "initialization":"random_from_scratch","manifest_sha256":metadata["manifest_sha256"],
                "validation":metric,"train_images":len(train_ids),"validation_images":len(val_ids),
                "hard_negative_mining":bool(miner),"training_crops":len(train),
                "purpose":"tight face framing vs background and excessively loose crops; not calibrated face presence probability"}
        if time.monotonic()>deadline:
            break
    model.load_state_dict(ck["state_dict"])
    test,test_ids,_=build_cache(args.data,"test",200,54,miner)
    assert not set(test_ids)&(set(train_ids)|set(val_ids))
    ck["test"]=evaluate(model,DataLoader(test,batch_size=128))
    args.export.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.export.with_suffix(".tmp"); torch.save(ck,temporary); temporary.replace(args.export)
    report={k:v for k,v in ck.items() if k!="state_dict"}
    report.update({"history":history,"test_images":len(test_ids),"limitations":"Crop classification test does not establish detection accuracy on full images. Unknown/unannotated faces may introduce negative-label noise."})
    (args.out/"evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report["test"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("training/datasets/prepared-wflw"))
    parser.add_argument("--out",type=Path,default=Path("training/experiments/regions"))
    parser.add_argument("--export",type=Path,default=Path("training/checkpoints/regions.pt"))
    parser.add_argument("--limit",type=int,default=1600)
    parser.add_argument("--epochs",type=int,default=18)
    parser.add_argument("--max-minutes",type=float,default=5)
    parser.add_argument("--estimate",action="store_true")
    parser.add_argument("--hard-negatives",type=Path,help="Own previously trained checkpoint, never external weights")
    main(parser.parse_args())

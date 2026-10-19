"""Select architecture using VALIDATION ONLY, then evaluate and export one model on the held-out test set."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.geometry.metrics import measure
from backend.ml.models import create_model
from backend.ml.evaluation import errors,summarize
from training.data import LandmarkDataset


@torch.inference_mode()
def evaluate(args):
    torch.set_num_threads(4)
    test = LandmarkDataset(args.data,"test")
    candidates = []
    for path in args.checkpoints:
        ck = torch.load(path,map_location="cpu",weights_only=True)
        if ck["manifest_sha256"]!=test.metadata["manifest_sha256"]:
            raise ValueError("Models and test data must share the same imported manifest.")
        candidates.append((float(ck["validation"]["nme"]),path,ck))
    _,selected,ck = min(candidates,key=lambda item:item[0])
    if ck.get("step_count_scope") != "cumulative":
        config = json.loads((selected.parent/"config.json").read_text(encoding="utf-8"))
        batches = math.ceil(test.metadata["counts"]["train"]/config["batch_size"])
        ck["training_steps"] = (ck["epochs"]-1)*batches+ck["training_steps"]
        ck["step_count_scope"] = "cumulative; reconstructed from completed epochs and last-epoch steps"
    model = create_model(ck["architecture"],ck["landmarks"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    predictions,targets,images,sides = [],[],[],[]
    for image,target,side in DataLoader(test,batch_size=32,num_workers=0):
        predictions.extend(model(image)["coordinates"].numpy())
        targets.extend(target.numpy())
        images.extend(image.numpy())
        sides.extend(side.numpy())
    pred,truth = np.asarray(predictions),np.asarray(targets)
    nme,local = errors(pred,truth,ck["dataset"])
    metrics = summarize(nme,local*np.array(sides))
    # A non-ML reference fitted only on training annotations, plus an image-use diagnostic.
    development = LandmarkDataset(args.data,"train")
    normalized_training = []
    for record in development.records:
        points = np.asarray(record["points"],float)
        lo,hi = points.min(0),points.max(0)
        side = max(hi-lo)*1.24
        origin = (lo+hi)/2-side/2
        normalized_training.append((points-origin)/side)
    mean_face = np.mean(normalized_training,axis=0)
    baseline_nme,baseline_px = errors(np.broadcast_to(mean_face,truth.shape),truth,ck["dataset"])
    shuffled_nme,_ = errors(np.roll(pred,len(pred)//2,axis=0),truth,ck["dataset"])
    baselines = {"mean_training_shape":summarize(baseline_nme,baseline_px*np.array(sides)),
                 "shuffled_image_prediction_nme":float(shuffled_nme.mean()),
                 "note":"Mean shape uses training annotations only. Shuffled predictions are a diagnostic of image dependence, not another trained model."}
    geometry_errors = {}
    for predicted,target in zip(pred,truth):
        try:
            measured = measure(predicted,ck["dataset"])["metrics"]
            reference = {m["id"]:m for m in measure(target,ck["dataset"])["metrics"]}
            for row in measured:
                actual = reference[row["id"]]["value"]
                if row["value"] is not None and actual is not None:
                    geometry_errors.setdefault(row["id"],[]).append(abs(row["value"]-actual))
        except ValueError:
            continue
    geometry = {key:{"mean_absolute_error":float(np.mean(values)),"p90_absolute_error":float(np.quantile(values,.9)),"samples":len(values)} for key,values in geometry_errors.items()}
    args.report.mkdir(parents=True,exist_ok=True)
    summary = {"status":"evaluated","selected_architecture":ck["architecture"],"selected_checkpoint":str(selected),
               "parameters":ck["parameters"],"dataset":test.metadata,"test":metrics,"baselines":baselines,"geometry_error_against_annotation":geometry,
               "normalization":"WFLW: pupil 96–97; 300-W: outer corners 36–45. NME is a fraction, not percent.",
               "selection":"Minimum validation NME only. Test not used for checkpoint or architecture selection.",
               "comparison":[{"architecture":c["architecture"],"validation":c["validation"],"parameters":c["parameters"],"checkpoint":str(path)} for _,path,c in candidates]}
    (args.report/"evaluation.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (args.report/"test_errors.json").write_text(json.dumps([{"image":r["image"],"nme":float(e),"localization_error_px":float(px*s)} for r,e,px,s in zip(test.records,nme,local,sides)],indent=2),encoding="utf-8")
    order = np.argsort(nme)
    chosen = list(order[:3])+list(order[-3:])
    fig,axes = plt.subplots(2,3,figsize=(10,7))
    for ax,index in zip(axes.flat,chosen):
        image = np.clip((np.asarray(images[index]).transpose(1,2,0)+1)/2,0,1)
        ax.imshow(image)
        a,b = truth[index]*95,pred[index]*95
        ax.scatter(a[:,0],a[:,1],s=4,c="#57dc98",label="Разметка")
        ax.scatter(b[:,0],b[:,1],s=4,c="#ef6789",label="Предсказание")
        ax.set_title(f"NME {nme[index]*100:.2f}%")
        ax.axis("off")
    axes[0,0].legend(fontsize=7)
    fig.suptitle("Три наименьшие / три наибольшие ошибки на тесте")
    fig.tight_layout()
    fig.savefig(args.report/"prediction_errors.png",dpi=150)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(6,4))
    ax.plot(np.sort(nme),np.arange(1,len(nme)+1)/len(nme))
    ax.axvline(.1,color="red",linestyle="--",label="Порог неудачи: NME > 0.10")
    ax.set(xlabel="NME (доля)",ylabel="Доля тестовых примеров",title="Кумулятивное распределение ошибок")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.report/"ced.png",dpi=160)
    plt.close(fig)
    ck["test"] = metrics
    ck["evaluation"] = {"selection":"validation_only","test_samples":len(test),"geometry_error":geometry}
    args.export.parent.mkdir(parents=True,exist_ok=True)
    temporary = args.export.with_suffix(".tmp")
    torch.save(ck,temporary)
    temporary.replace(args.export)
    digest = hashlib.sha256(args.export.read_bytes()).hexdigest()
    (args.export.with_suffix(".json")).write_text(json.dumps({"sha256":digest,**{k:v for k,v in summary.items() if k!="geometry_error_against_annotation"}},indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("training/datasets/prepared"))
    parser.add_argument("--checkpoints",type=Path,nargs="+",required=True)
    parser.add_argument("--report",type=Path,default=Path("reports/run"))
    parser.add_argument("--export",type=Path,default=Path("training/checkpoints/selected.pt"))
    args = parser.parse_args()
    evaluate(args)

"""Promote an independently compared candidate, preserving old weights.

Training exports a candidate, not the live checkpoint. Full-image validation
and test must refer to the same candidate bytes and current inference pipeline.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from backend.ml.face_detector import ARCHITECTURE, PIPELINE_VERSION


def promote(args):
    digest=hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    validation=json.loads(args.validation.read_text())
    test=json.loads(args.test.read_text())
    if validation["split"]!="val" or test["split"]!="test":
        raise ValueError("Need separate validation and test diagnostics.")
    v=validation["models"][args.candidate.name]
    t=test["models"][args.candidate.name]
    if any(report.get("checkpoint_sha256")!=digest or report.get("pipeline_version")!=PIPELINE_VERSION for report in [v,t]):
        raise ValueError("Diagnostics do not match the candidate checkpoint/current inference pipeline.")
    baseline=json.loads(args.baseline_validation.read_text())["models"][args.baseline_name]["summary"]
    if (v["summary"]["correct_complete"]<=baseline["correct_complete"]
        or v["summary"]["wrong_or_partial"]>baseline["wrong_or_partial"]):
        raise ValueError("Candidate did not improve the prespecified validation comparison.")
    checkpoint=torch.load(args.candidate,map_location="cpu",weights_only=True)
    if checkpoint.get("architecture")!=ARCHITECTURE or checkpoint.get("selection")!="validation_only" or not checkpoint.get("test"):
        raise ValueError("Candidate must be trained from scratch and assessed before promotion.")
    checkpoint["deployment_evaluation"]={"candidate_sha256":digest,"pipeline_version":PIPELINE_VERSION,
                                        "validation":v["summary"],"test":t["summary"],
                                        "selection":"validation_only; fixed .90 threshold; test reported after algorithm frozen",
                                        "limitations":test["limitations"]}
    args.export.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.export.with_suffix(".tmp")
    torch.save(checkpoint,temporary)
    temporary.replace(args.export)
    print(json.dumps({"export":str(args.export),"candidate_sha256":digest,"selected_epoch":checkpoint["epochs"],
                      "validation":v["summary"],"test":t["summary"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",type=Path,default=Path("training/checkpoints/complete-face.pt"))
    parser.add_argument("--validation",type=Path,default=Path("reports/qa/full-image-detection-new-val.json"))
    parser.add_argument("--test",type=Path,default=Path("reports/qa/full-image-detection-test.json"))
    parser.add_argument("--baseline-validation",type=Path,default=Path("reports/qa/full-image-detection-val.json"))
    parser.add_argument("--baseline-name",default="regions.pt")
    parser.add_argument("--export",type=Path,default=Path("training/checkpoints/regions-selected.pt"))
    promote(parser.parse_args())

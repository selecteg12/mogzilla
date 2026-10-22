"""Small, deterministic local technical audit; no human-rating accuracy claim.

Read-only source data are copied into this worktree before evaluation. Only
numeric results are committed; photographs remain under ignored experiments/.
"""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from backend.geometry.metrics import measure
from backend.ml.holistic import HolisticAnalyzer
from backend.ml.inference import Predictor
from backend.preprocessing.images import crop_image

ROOT = Path(__file__).resolve().parents[1]


def encode(image):
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=95)
    return out.getvalue()


def evaluate(prepared: Path, count: int):
    started = time.perf_counter()
    torch.set_num_threads(4)
    metadata = json.loads((prepared / "dataset.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (prepared / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    records = [r for r in records if r["split"] == "test" and np.ptp(np.array(r["points"]), axis=0).min() >= 110 and not r.get("out_of_frame_points")]
    records.sort(key=lambda r: hashlib.sha256((r["image"]+str(r["points"][0])).encode()).hexdigest())
    sample_dir = ROOT / "training/experiments/holistic/samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    analyzer, old = HolisticAnalyzer(), Predictor()
    status = analyzer.status()
    if not status["ready"]:
        raise RuntimeError(status["reason"])
    rows = []
    for i, record in enumerate(records[:count]):
        original = Image.open(Path(metadata["root"]) / record["image"]).convert("RGB")
        gt = np.array(record["points"])
        lo,hi = gt.min(0),gt.max(0)
        w,h = hi-lo
        left,top = max(0,int(lo[0]-.2*w)),max(0,int(lo[1]-.4*h))
        right,bottom = min(original.width,int(hi[0]+.2*w)),min(original.height,int(hi[1]+.15*h))
        portrait = original.crop((left,top,right,bottom))
        gt = gt-[left,top]
        data = encode(portrait)
        path = sample_dir / f"sample-{i+1:02d}.jpg"
        path.write_bytes(data)
        row = {"sample": path.name,"source_image": record["image"],"source_sha256": record.get("image_sha256"),"input_sha256":hashlib.sha256(data).hexdigest()}
        before = time.perf_counter()
        try:
            result = analyzer.analyze(data)
            analysis = result["analysis"]
            if analysis is None:
                row.update(status="abstained", reason=result["localization"]["message"])
            else:
                predicted = {m["id"]:m["value"] for m in analysis["geometry"]["metrics"]}
                truth = {m["id"]:m["value"] for m in measure(gt,"wflw")["metrics"]}
                common = ["eye_tilt_0","eye_tilt_1","eye_spacing","mouth_nose"]
                # Corners correspond closely; nose contour definitions differ.
                row.update(status="calculated", value=analysis["holistic"]["value"],raw=analysis["holistic"]["raw_value"],
                           inference_milliseconds=analysis["total_milliseconds"],
                           crop_stability=analysis["holistic"]["stability"], pose=analysis["quality"]["pose"],
                           warnings=analysis["quality"]["warnings"],
                           geometry_absolute_error={k:abs(predicted[k]-truth[k]) for k in common if predicted[k] is not None and truth[k] is not None})
                lo,hi = gt.min(0),gt.max(0)
                legacy = old.predict(*crop_image(np.asarray(portrait),[*lo,*(hi-lo)],allow_outside=True))
                legacy_values = {m["id"]:m["value"] for m in measure(legacy["points"],"wflw")["metrics"]}
                row["legacy_geometry_absolute_error"] = {k:abs(legacy_values[k]-truth[k]) for k in common if legacy_values[k] is not None and truth[k] is not None}
                if i < 8:
                    variants = {"brightness_085":ImageEnhance.Brightness(portrait).enhance(.85),
                                "brightness_115":ImageEnhance.Brightness(portrait).enhance(1.15),
                                "blur_08":portrait.filter(ImageFilter.GaussianBlur(.8)),
                                "mirror":portrait.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                                "rotate_5":portrait.rotate(5,resample=Image.Resampling.BICUBIC)}
                    row["synthetic_variants"] = {}
                    for name,img in variants.items():
                        try:
                            changed=analyzer.analyze(encode(img))["analysis"]
                            row["synthetic_variants"][name] = {"value":changed["holistic"]["value"],"delta":round(changed["holistic"]["value"]-row["value"],3)} if changed else {"abstained":True}
                        except ValueError as exc:
                            row["synthetic_variants"][name] = {"abstained":True,"reason":str(exc)}
        except ValueError as exc:
            row.update(status="abstained",reason=str(exc))
        row["milliseconds"] = round((time.perf_counter()-before)*1000,2)
        rows.append(row)
        print(f"{i+1}/{count}: {row['status']} {row.get('value','')} ({row['milliseconds']} ms)",flush=True)
    accepted = [r for r in rows if r["status"]=="calculated"]
    errors = {key:round(float(np.mean([r["geometry_absolute_error"][key] for r in accepted if key in r["geometry_absolute_error"]])),4)
              for key in ("eye_tilt_0","eye_tilt_1","eye_spacing","mouth_nose")}
    legacy_errors = {key:round(float(np.mean([r["legacy_geometry_absolute_error"][key] for r in accepted if key in r["legacy_geometry_absolute_error"]])),4) for key in errors}
    deltas = [abs(v["delta"]) for r in accepted for v in r.get("synthetic_variants",{}).values() if "delta" in v]
    report = {"protocol":"deterministic SHA-256 order; first eligible WFLW test faces >=110px; GT-selected portrait with padding; fresh detector inside portrait",
              "elapsed_seconds":round(time.perf_counter()-started,2),
              "accepted_inference_median_ms":round(float(np.median([r["inference_milliseconds"] for r in accepted])),2),
              "samples":len(rows),"calculated":len(accepted),"abstained":len(rows)-len(accepted),"model":status,
              "human_rating_agreement":None,"distinct_photos_per_identity":None,
              "geometry_mae_on_accepted":errors,"legacy_geometry_mae_on_same_accepted":legacy_errors,
              "geometry_caveat":"Native contour definitions differ; this small selected audit is not a WFLW benchmark or a guarantee of individual accuracy.",
              "synthetic_perturbations":{"comparisons":len(deltas),"mean_absolute_display_delta":round(float(np.mean(deltas)),3) if deltas else None,"max_absolute_display_delta":round(max(deltas),3) if deltas else None},
              "limitations":["No attractiveness labels; no claim of agreement with people.","No identity-labelled multi-photo set; synthetic transformations are not different photographs.","Possible source overlap with SCUT is not established.","Abstained faces excluded from geometry errors; compare coverage too."],
              "rows":rows}
    output=ROOT/"reports/holistic"
    output.mkdir(exist_ok=True)
    (output/"evaluation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--count",type=int,default=32)
    args=parser.parse_args()
    evaluate(args.prepared,args.count)

"""Recompute gallery through automatic photo analysis; GT checks outputs only."""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import HTTPException

from backend.app.main import localize, predictor, region_finder
from backend.geometry.rating import CATALOG, RatingRequest, calculate
from backend.geometry.reference_gallery import gallery_signature, gallery_is_current
from backend.ml.face_detector import inside_fraction
from backend.ml.regions import iou
from backend.preprocessing.images import crop_image

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/"reports/reference-values"


def scan():
    prepared=ROOT/"training/datasets/prepared-wflw"
    meta=json.loads((prepared/"dataset.json").read_text(encoding="utf-8"))
    root=Path(meta["root"])
    signature=gallery_signature(predictor.path,region_finder.path)
    if signature is None or not predictor.status()["ready"] or not region_finder.status()["ready"]:
        raise ValueError("Both active models must be trained and ready.")
    groups=defaultdict(list)
    for raw in (prepared/"manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row=json.loads(raw)
        if row["split"]=="test":
            groups[row["image_sha256"]].append(row)
    candidates=[]; counts=Counter()
    for digest,rows in groups.items():
        truths=[np.asarray(row["points"],dtype=float) for row in rows]
        if not any(min(points.min(0))>=0 and min(np.ptp(points,axis=0))>=90 for points in truths):
            continue
        counts["scanned_images"]+=1
        if counts["scanned_images"]%100==0:
            print(json.dumps({**counts,"filtered_candidates":len(candidates)}),flush=True)
        try:
            analysis=localize((root/rows[0]["image"]).read_bytes())["analysis"]
            if analysis is None:
                counts["automatic_abstentions"]+=1; continue
            counts["automatic_analyses"]+=1
            box=analysis["box"]
            if min(box[2:])<90:
                counts["small_detected_face"]+=1; continue
            boxes=[]
            for truth in truths:
                lo,hi=truth.min(0),truth.max(0)
                boxes.append([*lo.tolist(),*(hi-lo).tolist()])
            match=max(range(len(truths)),key=lambda index:iou(box,boxes[index]))
            truth=truths[match]; overlap=iou(box,boxes[match])
            window=[*analysis["transform"]["origin"],analysis["transform"]["side"],analysis["transform"]["side"]]
            coverage=inside_fraction(boxes[match],window)
            key=truth[[16,51,57,60,64,68,72,76,82]]
            complete=bool(((key[:,0]>=window[0]) & (key[:,0]<=window[0]+window[2])
                           & (key[:,1]>=window[1]) & (key[:,1]<=window[1]+window[3])).all())
            if overlap<.5 or coverage<.98 or not complete:
                counts["unmatched_or_incomplete_98_point_face"]+=1; continue
            points=np.asarray([[point["x"],point["y"]] for point in analysis["landmarks"]])
            nme=float(np.linalg.norm(points-truth,axis=1).mean()/max(np.linalg.norm(truth[96]-truth[97]),1e-6))
            flip=analysis["reliability"]["flip_disagreement_nme"]
            geometry=analysis["geometry"]
            if abs(geometry["roll_degrees"])>15 or geometry["pose_proxy"]>.15 or nme>.08 or flip>.08:
                counts["pose_or_landmark_error_filter"]+=1; continue
            rating=analysis["rating"]
            all_ten=rating.get("used_criteria")==5 and all(part["deviation"]["direction"]=="within" for part in rating["parts"])
            candidates.append({"image":rows[0]["image"],"image_sha256":digest,"box":box,"rating":rating,
                "nme":nme,"flip_disagreement_nme":flip,"points":points.tolist(),"all_ten":all_ten,
                "matched_face_iou":float(overlap),"face_coverage_in_input":float(coverage)})
        except (ValueError,RuntimeError,HTTPException) as exc:
            counts["processing_errors"]+=1
            print(json.dumps({"image":rows[0]["image"],"error":str(exc)}),flush=True)
    if gallery_signature(predictor.path,region_finder.path)!=signature:
        raise ValueError("Active pipeline changed during the scan; refusing mixed-version scores.")
    candidates.sort(key=lambda row:(-row["rating"]["value"],row["nme"]))
    TARGET.mkdir(parents=True,exist_ok=True)
    report={**signature,"manifest_sha256":meta["manifest_sha256"],"generated_at":datetime.now(timezone.utc).isoformat(),
            "scanned_images":counts["scanned_images"],"filtered_candidates":len(candidates),
            "all_ten_count":sum(row["all_ten"] for row in candidates),"counts":dict(counts),
            "quality_filters":{"minimum_face_pixels":90,"maximum_abs_roll_degrees":15,"maximum_pose_proxy":.15,
                "maximum_landmark_nme":.08,"maximum_flip_disagreement_nme":.08,"minimum_matched_face_iou":.5,
                "minimum_face_coverage_in_input":.98},
            "note":"Original images use the same automatic localize/analyze pipeline as uploads. GT is used after inference only for illustration quality checks. This selection is not a new benchmark or attractiveness validation.",
            "candidates":candidates}
    temporary=TARGET/"candidates.tmp"
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.replace(TARGET/"candidates.json")
    print(json.dumps({key:value for key,value in report.items() if key!="candidates"}),flush=True)
    print(json.dumps([{"index":index,"image":row["image"],"nme":row["nme"],"all_ten":row["all_ten"],
                       "score":row["rating"]["value"]} for index,row in enumerate(candidates[:20])]),flush=True)


def export_examples(indices):
    report=json.loads((TARGET/"candidates.json").read_text(encoding="utf-8"))
    if len(indices)!=3 or len(set(indices))!=3:
        raise ValueError("Select exactly three different examples.")
    if not gallery_is_current(report,predictor.path,region_finder.path):
        raise ValueError("Active pipeline changed; recompute candidates.")
    root=Path(json.loads((ROOT/"training/datasets/prepared-wflw/dataset.json").read_text(encoding="utf-8"))["root"])
    examples=[]
    for ordinal,index in enumerate(indices,1):
        row=report["candidates"][index]
        rating=calculate(RatingRequest(dataset="wflw",points=row["points"],rules=CATALOG))
        if rating["value"]!=row["rating"]["value"]:
            raise ValueError("Stored score does not reproduce.")
        with Image.open(root/row["image"]) as image:
            rgb=np.asarray(image.convert("RGB"))
        preview,_=crop_image(rgb,row["box"],size=384)
        version=report["generated_at"].replace(":","").replace("+","")
        Image.fromarray(preview).save(TARGET/f"example-{ordinal}.jpg",quality=92)
        examples.append({"id":ordinal,"title":f"Пример {ordinal}","image_url":f"/api/reference-values/photo/{ordinal}?v={version}",
            "source_image":row["image"],"box":row["box"],"dataset":"WFLW · тестовая выборка · автоматический поиск лица",
            "value":rating["value"],"all_ten":row["all_ten"],"parts":rating["parts"],"landmark_nme":row["nme"],
            "flip_disagreement_nme":row["flip_disagreement_nme"],"checkpoint_sha256":report["checkpoint_sha256"],
            "detector_checkpoint_sha256":report["detector_checkpoint_sha256"]})
    output={key:value for key,value in report.items() if key not in ["candidates","note"]}
    output["examples"]=examples
    output["selection_note"]=("Среди прошедших проверку снимков найдено совпадение всех пяти критериев с диапазонами 10/10. "
                              if report["all_ten_count"] else "Среди прошедших проверку снимков нет подтверждённого совпадения всех пяти критериев на 10/10. ")
    output["selection_note"]+="Ниже — фактические баллы обновлённого автоматического анализа. Они описывают выбранные геометрические предпочтения, а не объективную привлекательность."
    output["crop_note"]="Лицо найдено автоматически на исходном снимке обновлённым детектором. Точки и баллы получены тем же анализом, что используется при загрузке фото. Разметка WFLW применялась только после предсказания для проверки качества. Здесь показан фактический квадрат входа модели в большем разрешении; повторная загрузка обрезанного предпросмотра может дать другой результат."
    output["source_url"]="https://github.com/wywu/LAB"
    temporary=TARGET/"examples.tmp"
    temporary.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.replace(TARGET/"examples.json")
    print(json.dumps({"examples":len(examples),"scores":[row["value"] for row in examples],"all_ten_count":report["all_ten_count"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-indices",type=int,nargs="+")
    args=parser.parse_args()
    scan() if args.export_indices is None else export_examples(args.export_indices)

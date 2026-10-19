"""HTTP integration and a small full-image localization diagnostic, never attractiveness validation."""
import io
import json
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from backend.ml.regions import iou

ROOT=Path(__file__).resolve().parents[1]


def check():
    directory=ROOT/"training/datasets/prepared-wflw"
    meta=json.loads((directory/"dataset.json").read_text(encoding="utf-8"))
    root=Path(meta["root"])
    selected=[]; seen=set()
    for line in (directory/"manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row=json.loads(line)
        points=np.asarray(row["points"])
        lo,hi=points.min(0),points.max(0)
        if row["split"]=="test" and min(lo)>=0 and min(hi-lo)>=80 and row["image_sha256"] not in seen:
            selected.append(row); seen.add(row["image_sha256"])
        if len(selected)==12:
            break
    examples=[]
    with httpx.Client(base_url="http://127.0.0.1:8000",timeout=60,trust_env=False) as client:
        settings=client.get("/api/rating/settings").json()
        rules=[{k:r[k] for k in ["id","low","high","falloff","weight"]} for r in settings["criteria"]]
        assert client.get("/").status_code==200
        for row in selected:
            path=root/row["image"]
            response=client.post("/api/localize",content=path.read_bytes(),headers={"Content-Type":"image/jpeg"})
            response.raise_for_status()
            body=response.json(); analysis=body["analysis"]
            entry={"image":row["image"],"found":analysis is not None}
            if analysis:
                assert len(analysis["landmarks"])==98
                assert 0<=analysis["rating"]["value"]<=10
                assert analysis["rating"]["provisional"] is True
                assert all(metric["description"] for metric in analysis["geometry"]["metrics"])
                points=np.asarray(row["points"]); lo,hi=points.min(0),points.max(0)
                entry["roi_iou_against_selected_annotation"]=iou(analysis["box"],[*lo,*(hi-lo)])
                payload={"dataset":"wflw","points":[[p["x"],p["y"]] for p in analysis["landmarks"]],"rules":rules}
                assert client.post("/api/rating",json=payload).json()["value"]==analysis["rating"]["value"]
                payload["landmarks_reviewed"]=True
                rating=client.post("/api/rating",json=payload)
                rating.raise_for_status(); rated=rating.json()
                assert 0<=rated["value"]<=10
                assert abs(sum(p["contribution"] for p in rated["parts"])-rated["value"])<=.0051
                assert abs(sum(p["lost_contribution"] for p in rated["parts"])+rated["value"]-10)<=.0051
                assert all(part["annotations"] and part["description"] for part in rated["parts"])
                assert rated["sensitivity"] is not None
                entry["rating_formula_verified"]=True
                if entry["roi_iou_against_selected_annotation"]>=.3:
                    print("UI example: "+str(path),flush=True)
            examples.append(entry)
            print(json.dumps(entry),flush=True)
        negatives=[]
        for name,rgb in [("solid",np.full((320,320,3),127,dtype=np.uint8)),("noise",np.random.default_rng(5).integers(0,256,(320,320,3),dtype=np.uint8))]:
            stream=io.BytesIO(); Image.fromarray(rgb).save(stream,format="PNG")
            response=client.post("/api/localize",content=stream.getvalue(),headers={"Content-Type":"image/png"})
            response.raise_for_status()
            negatives.append({"image":name,"false_positive":response.json()["analysis"] is not None})
        summary={"http_and_formula":"passed","examples":examples,"synthetic_negative_checks":negatives,
                 "note":"12 first eligible held-out images (face box >=80px). Full-image proposals use pixels only, annotation is used AFTER prediction for IoU. Multiple annotated faces can reduce this single-annotation diagnostic. This is not a detector benchmark or validation of attractiveness scores."}
        (ROOT/"reports/qa/geometry-rating-smoke.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
        print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":
    check()

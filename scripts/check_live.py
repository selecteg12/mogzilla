"""Check the actual browser-facing HTTP boundary with licensed held-out images."""
import json
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def check():
    with httpx.Client(base_url="http://127.0.0.1:8000",timeout=60,trust_env=False) as client:
        status = client.get("/api/model").json()
        assert status["ready"],status
        dataset = "prepared-wflw" if status["metadata"]["dataset"]=="wflw" else "prepared-afw"
        directory = ROOT/"training/datasets"/dataset
        meta = json.loads((directory/"dataset.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (directory/"manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        root = Path(meta["root"])
        selected = []
        for row in records:
            points = np.asarray(row["points"])
            lo,hi = points.min(0),points.max(0)
            if row["split"]=="test" and min(lo)>=0 and min(hi-lo)>=100 and not any(r["image_sha256"]==row["image_sha256"] for r in selected):
                selected.append(row)
            if len(selected)==2:
                break
        outputs,normalized = [],[]
        for row in selected:
            points = np.asarray(row["points"])
            lo,hi = points.min(0),points.max(0)
            box = dict(zip(["x","y","width","height"],map(float,np.r_[lo,hi-lo])))
            path = root/row["image"]
            headers = {"Content-Type":"image/png" if path.suffix.lower()==".png" else "image/jpeg"}
            response = client.post("/api/analyze",params=box,content=path.read_bytes(),headers=headers)
            response.raise_for_status()
            data = response.json()
            assert len(data["landmarks"])==status["metadata"]["landmarks"]
            assert len(data["stages"])==7 and len(data["geometry"]["metrics"])>=19
            assert data["reliability"]["confidence_probability"] is None
            assert response.headers["Cache-Control"]=="no-store"
            predicted = np.array([[p["x"],p["y"]] for p in data["landmarks"]])
            normalized.append((predicted-data["transform"]["origin"])/data["transform"]["side"])
            outputs.append({"image":row["image"],"landmarks":len(data["landmarks"]),"metrics":len(data["geometry"]["metrics"]),
                            "stages":len(data["stages"]),"milliseconds":data["total_milliseconds"],"warnings":data["reliability"]["warnings"]})
            repeated = client.post("/api/analyze",params=box,content=path.read_bytes(),headers=headers).json()
            assert repeated["landmarks"]==data["landmarks"],"Inference must be deterministic."
        assert len(outputs)==2
        difference = float(np.abs(normalized[0]-normalized[1]).max())
        assert difference>1e-5,"Different images must not return a fixed normalized landmark template."
        summary = {"status":"passed","model":status["metadata"]["architecture"],"dataset":status["metadata"]["dataset"],
                   "examples":outputs,"maximum_normalized_coordinate_difference":difference,
                   "note":"ROI from test annotation used only as a crop initialization for this check, not as predicted landmarks."}
        target = ROOT/"reports/qa"
        target.mkdir(exist_ok=True,parents=True)
        (target/"api-smoke.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
        print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":
    check()

"""Reproduce published gallery scores through the live automatic upload API."""
import io
import json
import re
from pathlib import Path

import httpx
import numpy as np
from PIL import Image


def check():
    metadata=json.loads(Path("training/datasets/prepared-wflw/dataset.json").read_text())
    root=Path(metadata["root"])
    report={"status":"passed","photos":[],"browser_visual_verification":False}
    with httpx.Client(base_url="http://127.0.0.1:8000",timeout=60,trust_env=False) as client:
        response=client.get("/api/reference-values"); response.raise_for_status()
        data=response.json(); gallery=data["gallery"]
        assert data["reference"]["rating"]["value"]==10
        assert gallery["scan_mode"]=="automatic_full_image"
        assert gallery["detector_checkpoint_sha256"]
        assert len(gallery["examples"])==3
        report.update(reference_score=10,reference_criteria=5,searched_images=gallery["scanned_images"],
                      quality_filtered_candidates=gallery["filtered_candidates"],exact_all_ten_photos=gallery["all_ten_count"],
                      scan_mode=gallery["scan_mode"],generated_at=gallery["generated_at"])
        for example in gallery["examples"]:
            photo=client.get(example["image_url"]); photo.raise_for_status()
            assert Image.open(io.BytesIO(photo.content)).size==(384,384)
            result=client.post("/api/localize",content=(root/example["source_image"]).read_bytes(),
                               headers={"Content-Type":"image/jpeg"})
            result.raise_for_status()
            analysis=result.json()["analysis"]
            assert analysis is not None
            actual=analysis["rating"]
            assert actual["value"]==example["value"]
            assert np.allclose(analysis["box"],example["box"],rtol=0,atol=1e-5)
            assert [part["observed"] for part in actual["parts"]]==[part["observed"] for part in example["parts"]]
            assert [part["score"] for part in actual["parts"]]==[part["score"] for part in example["parts"]]
            all_ten=actual["used_criteria"]==5 and all(part["deviation"]["direction"]=="within" for part in actual["parts"])
            assert all_ten==example["all_ten"]
            report["photos"].append({"id":example["id"],"score":actual["value"],"photo_http":photo.status_code,
                                     "recomputed_automatically":True,"all_ten":all_ten})
        assert client.get("/api/reference-values/photo/1?v=outdated").status_code==404
        page=client.get("/"); page.raise_for_status()
        for asset in re.findall(r'(?:src|href)="(/assets/[^"]+)"',page.text):
            assert client.get(asset).status_code==200
    Path("reports/qa/reference-values-smoke.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    check()

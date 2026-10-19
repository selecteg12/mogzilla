"""Verify the live localhost API, optionally using private regression photos in RAM.

No photo or predicted face template is written. The report contains counts,
boxes and check outcomes only; private input file names are not persisted.
"""
import argparse
import io
import json
import re
from pathlib import Path

import httpx
from PIL import Image


def check(args):
    report={"browser_visual_verification":False,"checks":[],"private_photo_storage":False}
    with httpx.Client(base_url="http://127.0.0.1:8000",timeout=60,trust_env=False) as client:
        health=client.get("/api/health"); health.raise_for_status()
        model=client.get("/api/model").json()
        assert model["ready"] and model["region_finder"]["ready"]
        metadata=model["region_finder"]["metadata"]
        assert metadata["architecture"]=="complete_face_box_v1"
        assert metadata["initialization"]=="random_from_scratch"
        assert metadata["deployment_evaluation"]["test"]["images"]==100
        report["detector"]={"architecture":metadata["architecture"],"selected_epoch":metadata["epochs"],
                            "training_images":metadata["train"]["source_images"],"full_image_test":metadata["deployment_evaluation"]["test"]}
        index=client.get("/"); index.raise_for_status()
        assets=re.findall(r'(?:src|href)="(/assets/[^"]+)"',index.text)
        assert assets
        for asset in assets:
            assert client.get(asset).status_code==200
        report["checks"].append({"name":"HTML and built assets","passed":True})

        def positive(data,name):
            response=client.post("/api/localize",content=data,headers={"Content-Type":"image/jpeg"})
            response.raise_for_status()
            result=response.json()
            analysis=result["analysis"]
            assert analysis is not None,name
            assert len(analysis["landmarks"])==98
            assert len(analysis["stages"])==7
            assert analysis["localization"]["framing_check"]["complete"]
            assert analysis["reliability"]["flip_disagreement_nme"]<=.12
            assert 0<=analysis["rating"]["value"]<=10
            assert response.headers["Cache-Control"]=="no-store"
            report["checks"].append({"name":name,"passed":True,"box":analysis["box"],
                                     "actual_input_origin":analysis["transform"]["origin"],"actual_input_side":analysis["transform"]["side"],
                                     "flip_disagreement_nme":analysis["reliability"]["flip_disagreement_nme"],"points":98})
        # A held-out public image selected only after independent predictions.
        evaluation=json.loads(Path("reports/qa/full-image-detection-test.json").read_text())
        entry=next(item for item in evaluation["models"]["complete-face.pt"]["entries"] if item.get("correct"))
        root=Path(json.loads(Path("training/datasets/prepared-wflw/dataset.json").read_text())["root"])
        positive((root/entry["image"]).read_bytes(),"public held-out image")
        if args.portrait:
            positive(args.portrait.read_bytes(),"reported portrait")
            with Image.open(args.portrait) as source:
                # This optional regression case refers to the reported 736x920
                # portrait, not a generic face/eye locator or fabricated label.
                assert source.size==(736,920)
                stream=io.BytesIO()
                source.convert("RGB").crop((180,20,560,300)).save(stream,format="JPEG")
            response=client.post("/api/localize",content=stream.getvalue(),headers={"Content-Type":"image/jpeg"})
            response.raise_for_status()
            assert response.json()["analysis"] is None
            report["checks"].append({"name":"eyes-only crop abstains, no points or rating","passed":True})
            response=client.post("/api/analyze?x=198.4351&y=30.6271&width=255.2258&height=255.2258",
                                 content=args.portrait.read_bytes(),headers={"Content-Type":"image/jpeg"})
            assert response.status_code==422
            report["checks"].append({"name":"old forehead-only ROI rejected by analyze endpoint","passed":True})
        if args.full_body:
            positive(args.full_body.read_bytes(),"earlier full-body regression image")
    output=Path("reports/qa/face-detection-live.json")
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portrait",type=Path)
    parser.add_argument("--full-body",type=Path)
    check(parser.parse_args())

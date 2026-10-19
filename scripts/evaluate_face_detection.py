"""Bounded full-image comparison: pixels -> proposals -> flip check -> GT audit.

All WFLW annotations from a source image are considered. Correct detection means
IoU >= .5 AND the actual padded CNN input contains >=98% of the annotated face
box and all eye/nose/mouth/chin landmarks. These are task-specific diagnostics,
not official WIDER AP, not demographic or attractiveness validation.
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from backend.ml.regions import RegionFinder, padded_box, iou
from backend.ml.face_detector import inside_fraction, PIPELINE_VERSION
from backend.ml.inference import Predictor
from backend.preprocessing.images import crop_image
from training.train_face_detector import wider_annotations


def dataset_images(directory, split, limit):
    meta = json.loads((directory / "dataset.json").read_text())
    groups = defaultdict(list)
    for line in (directory / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["split"] == split:
            groups[row["image_sha256"]].append(row)
    names = sorted(groups)
    np.random.default_rng(921 if split == "val" else 922).shuffle(names)
    selected = []
    for name in names:
        rows = groups[name]
        faces = []
        for row in rows:
            points = np.asarray(row["points"])
            lo, hi = points.min(0), points.max(0)
            if min(hi - lo) >= 80 and min(lo) >= 0:
                faces.append({"box": [*lo, *(hi - lo)], "points": points})
        if faces:
            selected.append((Path(meta["root"]) / rows[0]["image"], rows[0]["image"], faces))
        if len(selected) >= limit:
            break
    return selected


def assess(box, faces):
    window = padded_box(box)
    wx, wy, ww, wh = window
    best = max(faces, key=lambda face: iou(box, face["box"]))
    key = best["points"][[16, 51, 57, 60, 64, 68, 72, 76, 82]]
    complete = bool(((key[:, 0] >= wx) & (key[:, 0] <= wx + ww)
                     & (key[:, 1] >= wy) & (key[:, 1] <= wy + wh)).all())
    overlap, coverage = iou(box, best["box"]), inside_fraction(best["box"], window)
    return {"iou": float(overlap), "face_box_coverage_in_actual_input": float(coverage),
            "eyes_nose_mouth_chin_in_input": complete, "correct": bool(overlap >= .5 and coverage >= .98 and complete)}


def main(args):
    torch.set_num_threads(4); cv2.setNumThreads(1)
    predictor = Predictor()
    dataset = dataset_images(args.data, args.split, args.limit)
    known=wider_annotations(Path("training/datasets/WIDER"))
    report = {"split": args.split, "selection": "fixed seed; held-out source images with WFLW face >=80px", "models": {},
              "limitations":"Strict complete-face criterion uses WFLW points. Other known WIDER faces are reported separately, not silently counted as correct 98-point crops. Identity and demographic independence not established."}
    for checkpoint in args.checkpoints:
        finder = RegionFinder(checkpoint)
        if not finder.status()["ready"]:
            raise ValueError(f"Checkpoint unavailable: {checkpoint}")
        entries = []
        for path, relative, faces in dataset:
            with Image.open(path) as source:
                rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
            start = time.perf_counter()
            detection = finder.find(rgb)
            selected = None
            for candidate in detection["candidates"]:
                framing=finder.verify(rgb,candidate["box"])
                if framing is not None and not framing["complete"]:
                    continue
                prediction = predictor.predict(*crop_image(rgb, candidate["box"]))
                if prediction["flip_disagreement_nme"] <= .12:
                    selected = candidate
                    break
            entry = {"image": relative, "found": selected is not None, "seconds": time.perf_counter() - start}
            if selected:
                entry.update(assess(selected["box"], faces), box=selected["box"])
                extra=known.get(relative.split("WFLW_images/")[-1],[])
                entry["best_wider_box_iou"]=float(max([iou(selected["box"],box) for box in extra] or [0]))
                entry["other_known_face"]=entry["iou"]<.1 and entry["best_wider_box_iou"]>=.5
            entries.append(entry)
            if len(entries) % 10 == 0:
                print(json.dumps({"model": checkpoint.name, "done": len(entries),
                                  "correct": sum(item.get("correct", False) for item in entries),
                                  "wrong": sum(item["found"] and not item.get("correct", False) for item in entries)}), flush=True)
        found = [entry for entry in entries if entry["found"]]
        negatives = []
        for name, rgb in [("solid", np.full((320, 320, 3), 127, np.uint8)),
                          ("noise", np.random.default_rng(92).integers(0, 256, (320, 320, 3), dtype=np.uint8))]:
            negatives.append({"name": name, "proposals": len(finder.find(rgb)["candidates"])})
        summary = {"images": len(entries), "found": len(found), "correct_complete": sum(item.get("correct", False) for item in entries),
                   "wrong_or_partial": sum(not item.get("correct", False) for item in found), "abstained": len(entries) - len(found),
                   "of_wrong_or_partial_other_known_wider_faces":sum(item.get("other_known_face",False) for item in entries),
                   "mean_seconds": float(np.mean([entry["seconds"] for entry in entries]))}
        report["models"][checkpoint.name] = {"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                             "pipeline_version":PIPELINE_VERSION,"summary": summary,
                                             "entries": entries, "synthetic_negatives": negatives}
        print(json.dumps({"model": checkpoint.name, **summary}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("training/datasets/prepared-wflw"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=[Path("training/checkpoints/regions.pt"), Path("training/checkpoints/regions-hard.pt")])
    parser.add_argument("--output", type=Path, default=Path("reports/qa/full-image-detection-val.json"))
    main(parser.parse_args())

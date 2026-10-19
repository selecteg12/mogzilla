"""Train complete-face classification + WFLW-box regression from random weights.

WIDER supplies additional face/background examples and exhaustive known boxes
for negative mining. Its different box convention is NOT a regression target:
regression is supervised only by WFLW landmark bounds. Held-out source images
and decoded-image hashes are excluded from extra training data.
"""
import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, TensorDataset

from backend.ml.face_detector import ARCHITECTURE, FaceBoxCNN, encode_box, decode_box, inside_fraction
from backend.ml.regions import RegionFinder, padded_box, patch, iou


def wider_annotations(directory, positives_only=False):
    result = {}
    for split in ["train", "val"]:
        path = directory / f"wider_face_split/wider_face_{split}_bbx_gt.txt"
        if not path.is_file():
            continue
        lines = iter(path.read_text().splitlines())
        for name in lines:
            count = int(next(lines))
            rows = [list(map(int, next(lines).split())) for _ in range(max(count, 1))]
            result[name] = [row[:4] for row in rows if row[2] > 0 and row[3] > 0
                            and (not positives_only or (row[7] == 0 and row[4] < 2 and row[8] < 2))]
    return result


def read_rgb(path):
    with Image.open(path) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert("RGB")).copy()


def build_cache(args, split, limit, extra_limit=0):
    metadata = json.loads((args.data / "dataset.json").read_text())
    records = [json.loads(line) for line in (args.data / "manifest.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for row in records:
        if row["split"] == split:
            groups[row["image_sha256"]].append(row)
    known = wider_annotations(args.wider)
    positive_wider = wider_annotations(args.wider, positives_only=True)
    miner = RegionFinder(args.miner)
    if not miner.status()["ready"]:
        miner = None
    rng = np.random.default_rng({"train": 714, "val": 715, "test": 716}[split])
    names = sorted(groups)
    rng.shuffle(names)
    images, labels, targets, masks, source_ids = [], [], [], [], []
    counts = defaultdict(int)

    def add(rgb, window, label, face, regress, kind):
        images.append(patch(rgb, window))
        labels.append(float(label))
        targets.append(encode_box(face, window) if label else np.zeros(4, dtype=np.float32))
        masks.append(float(label and regress))
        counts[kind] += 1

    def samples(rgb, faces, all_boxes, source_id, regress):
        h, w = rgb.shape[:2]
        eligible = [box for box in faces if min(box[2:]) >= 45 and inside_fraction(box, [0, 0, w, h]) >= .995]
        if not eligible:
            return False
        face = max(eligible, key=lambda box: box[2] * box[3])
        fx, fy, fw, fh = face
        side = max(fw, fh)
        cx, cy = fx + fw / 2, fy + fh / 2
        # Centered framing + varied scale/translation, always containing the
        # complete target. Identical crop generation policy for held-out sets.
        for index in range(5):
            s = side * (1.24 if index == 0 else float(rng.uniform(1.18, 1.95)))
            mx, my = max((s - fw) / 2 - side * .025, 0), max((s - fh) / 2 - side * .025, 0)
            dx, dy = (0, 0) if index == 0 else (rng.uniform(-mx, mx), rng.uniform(-my, my))
            add(rgb, [cx + dx - s / 2, cy + dy - s / 2, s, s], 1, face, regress, "positive")
        # Face fragments were missing from the previous training recipe.
        partials = []
        for sx, sy, scale in [(.5, .17, .63), (.5, .39, .64), (.5, .85, .63),
                               (.12, .52, .7), (.88, .52, .7), (.5, -.04, .8)]:
            s = side * scale
            partials.append([fx + fw * sx - s / 2, fy + fh * sy - s / 2, s, s])
        for window in partials:
            # A second annotated face inside this window must not become a negative.
            if not any(inside_fraction(other, window) > .90 and min(other[2:]) > window[2] * .25 for other in all_boxes):
                add(rgb, window, 0, face, False, "partial_face_negative")
        add(rgb, padded_box(face, 3.2), 0, face, False, "loose_negative")
        backgrounds = []
        for _ in range(65):
            s = float(rng.uniform(.15, 1.0) * min(h, w))
            window = [float(rng.uniform(0, max(1, w - s))), float(rng.uniform(0, max(1, h - s))), s, s]
            if all(inside_fraction(other, window) < .55 or min(other[2:]) < s * .18 for other in all_boxes):
                backgrounds.append(window)
        if backgrounds:
            scores = miner.score_boxes(rgb, backgrounds) if miner else rng.random(len(backgrounds))
            for index in np.argsort(scores)[-4:]:
                add(rgb, backgrounds[index], 0, face, False, "hard_background_negative")
        source_ids.append(source_id)
        return True

    for name in names:
        rows = groups[name]
        faces = []
        for row in rows:
            points = np.asarray(row["points"])
            lo, hi = points.min(0), points.max(0)
            faces.append([*lo, *(hi - lo)])
        image_name = rows[0]["image"].replace("\\", "/").split("WFLW_images/")[-1]
        rgb = read_rgb(Path(metadata["root"]) / rows[0]["image"])
        samples(rgb, faces, faces + known.get(image_name, []), name, True)
        if len(source_ids) >= limit:
            break
        if len(source_ids) and len(source_ids) % 400 == 0:
            print(f"{split}: {len(source_ids)} WFLW images, {len(images)} crops", flush=True)
    wflw_count = len(source_ids)
    if split == "train" and extra_limit:
        blocked_names = {row["image"].replace("\\", "/").split("WFLW_images/")[-1] for row in records}
        blocked_hashes = {row["image_sha256"] for row in records}
        extra_names = [name for name in sorted(known) if name not in blocked_names and (args.wider / "WIDER_train/images" / name).is_file()]
        rng.shuffle(extra_names)
        for name in extra_names:
            rgb = read_rgb(args.wider / "WIDER_train/images" / name)
            digest = hashlib.sha256(rgb.tobytes() + f"{rgb.shape[1]},{rgb.shape[0]}".encode()).hexdigest()
            if digest in blocked_hashes or digest in source_ids:
                continue
            samples(rgb, positive_wider[name], known[name], digest, False)
            if (len(source_ids)-wflw_count) and (len(source_ids)-wflw_count)%200==0:
                print(f"{split}: {len(source_ids)-wflw_count} additional WIDER images, {len(images)} crops",flush=True)
            if len(source_ids) - wflw_count >= extra_limit:
                break
    if not images:
        raise ValueError("No usable training images.")
    dataset = TensorDataset(torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2).copy()),
                            torch.tensor(labels), torch.tensor(np.stack(targets)), torch.tensor(masks))
    report = {"source_images": len(source_ids), "wflw_images": wflw_count, "additional_wider_images": len(source_ids) - wflw_count,
              "crops": len(images), "counts": dict(counts), "image_hashes": source_ids}
    print(json.dumps({"split": split, **{k: v for k, v in report.items() if k != "image_hashes"}}), flush=True)
    return dataset, report, metadata


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    scores, labels, losses, overlaps = [], [], [], []
    for images, target, boxes, mask in loader:
        logits, output = model(images.float() / 127.5 - 1)
        losses.extend(torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none").tolist())
        scores.extend(logits.sigmoid().tolist())
        labels.extend(target.tolist())
        for a, b in zip(output[mask.bool()].numpy(), boxes[mask.bool()].numpy()):
            overlaps.append(iou(decode_box(a, [0, 0, 1, 1]), decode_box(b, [0, 0, 1, 1])))
    score, truth = np.asarray(scores), np.asarray(labels, dtype=bool)
    pred = score >= .90
    return {"samples": len(truth), "classification_loss": float(np.mean(losses)),
            "threshold": .90, "precision": float((pred & truth).sum() / max(pred.sum(), 1)),
            "recall": float((pred & truth).sum() / max(truth.sum(), 1)),
            "mean_box_iou_on_positive_crops": float(np.mean(overlaps)),
            "partial_and_background_false_positives": int((pred & ~truth).sum()), "negative_crops": int((~truth).sum())}


def main(args):
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    random.seed(714); np.random.seed(714); torch.manual_seed(714)
    model = FaceBoxCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0002)
    start = time.perf_counter()
    logits, boxes = model(torch.zeros(64, 3, 96, 96))
    (logits.mean() + boxes.mean()).backward()
    optimizer.zero_grad(set_to_none=True)
    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()),
                      "batch64_backward_seconds": time.perf_counter() - start,
                      "initialization": "random_from_scratch", "paid_compute": False}), flush=True)
    if args.estimate:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "cache.pt"
    if args.reuse_cache and cache.is_file():
        cached = torch.load(cache, weights_only=False)
        train, val, train_info, val_info, metadata = cached
    else:
        train, train_info, metadata = build_cache(args, "train", args.limit, args.wider_limit)
        val, val_info, _ = build_cache(args, "val", args.val_limit)
        torch.save((train, val, train_info, val_info, metadata), cache)
    assert not set(train_info["image_hashes"]) & set(val_info["image_hashes"])
    train_loader = DataLoader(train, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(714))
    val_loader = DataLoader(val, batch_size=128)
    deadline = time.monotonic() + args.max_minutes * 60
    best, history, steps = float("inf"), [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, target, boxes, mask in train_loader:
            image = images.float() / 127.5 - 1
            boxes = boxes.clone()
            if torch.rand(()) < .5:
                image = image.flip(-1)
                boxes[:, 0] = -boxes[:, 0]
            image = (image * float(np.random.uniform(.75, 1.25)) + float(np.random.uniform(-.12, .12))).clamp(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            logits, output = model(image)
            classification = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            regression = torch.nn.functional.smooth_l1_loss(output[mask.bool()], boxes[mask.bool()], beta=.1) if mask.any() else output.sum() * 0
            loss = classification + 2 * regression
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
            if time.monotonic() > deadline:
                break
        validation = evaluate(model, val_loader)
        selection = validation["classification_loss"] + (1 - validation["mean_box_iou_on_positive_crops"])
        history.append({"epoch": epoch, "training_steps": steps, "train_loss": float(np.mean(losses)), "validation": validation})
        print(json.dumps(history[-1]), flush=True)
        if selection < best:
            best = selection
            checkpoint = {"state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
                          "architecture": ARCHITECTURE, "epochs": epoch, "training_steps": steps,
                          "initialization": "random_from_scratch", "manifest_sha256": metadata["manifest_sha256"],
                          "validation": validation, "selection": "validation_only", "threshold": .90,
                          "train": train_info, "validation_data": val_info,
                          "box_protocol": "WFLW landmark bounds only; WIDER classification samples have regression masked"}
            torch.save(checkpoint, args.out / "best-unassessed.pt")
        if time.monotonic() > deadline:
            break
    model.load_state_dict(checkpoint["state_dict"])
    test, test_info, _ = build_cache(args, "test", args.test_limit)
    assert not set(test_info["image_hashes"]) & (set(train_info["image_hashes"]) | set(val_info["image_hashes"]))
    checkpoint["test"] = evaluate(model, DataLoader(test, batch_size=128))
    checkpoint["test_data"] = test_info
    args.export.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.export.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(args.export)
    report = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    report["history"] = history
    report["limitations"] = "Crop metrics do not establish full-image detection accuracy. No verified gender labels; no demographic parity claim. No identity-disjoint guarantee."
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"export": str(args.export), "test": checkpoint["test"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("training/datasets/prepared-wflw"))
    parser.add_argument("--wider", type=Path, default=Path("training/datasets/WIDER"))
    parser.add_argument("--miner", type=Path, default=Path("training/checkpoints/regions-hard.pt"))
    parser.add_argument("--out", type=Path, default=Path("training/experiments/complete-face"))
    parser.add_argument("--export", type=Path, default=Path("training/checkpoints/complete-face.pt"))
    parser.add_argument("--limit", type=int, default=2400)
    parser.add_argument("--wider-limit", type=int, default=1200)
    parser.add_argument("--val-limit", type=int, default=180)
    parser.add_argument("--test-limit", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--max-minutes", type=float, default=12)
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true", help="Only reuse a cache created locally by this script.")
    main(parser.parse_args())

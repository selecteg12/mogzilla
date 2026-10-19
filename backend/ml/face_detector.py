"""Small, locally trained proposal classifier with supervised face-box regression.

No external weights. The classifier is taught to reject partial-face crops;
the second head refines the location of the complete annotated face.
"""
import numpy as np
import torch
from torch import nn

from backend.ml.regions import padded_box, patch, iou

ARCHITECTURE = "complete_face_box_v1"
PIPELINE_VERSION = "complete_face_box_v1_consensus3"


class FaceBoxCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a, b in zip([3, 16, 24, 48], [16, 24, 48, 64]):
            layers.extend([nn.Conv2d(a, b, 3, 2, 1, bias=False), nn.GroupNorm(8, b), nn.SiLU()])
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 6 * 6, 128), nn.SiLU(), nn.Linear(128, 5))

    def forward(self, images):
        output = self.head(self.features(images))
        return output[:, 0], output[:, 1:]


def encode_box(face, window):
    x, y, w, h = face
    wx, wy, ww, wh = window
    return np.asarray([(x + w / 2 - wx) / ww - .5, (y + h / 2 - wy) / wh - .5,
                       np.log(w / ww), np.log(h / wh)], dtype=np.float32)


def decode_box(deltas, window):
    wx, wy, ww, wh = window
    dx, dy, lw, lh = np.asarray(deltas, dtype=float)
    w, h = np.exp(np.clip([lw, lh], -3, 1)) * [ww, wh]
    cx, cy = wx + (dx + .5) * ww, wy + (dy + .5) * wh
    return [float(cx - w / 2), float(cy - h / 2), float(w), float(h)]


def inside_fraction(face, window):
    x, y = max(face[0], window[0]), max(face[1], window[1])
    right, bottom = min(face[0] + face[2], window[0] + window[2]), min(face[1] + face[3], window[1] + window[3])
    return max(0, right - x) * max(0, bottom - y) / max(face[2] * face[3], 1e-9)


@torch.inference_mode()
def predict_windows(model, rgb, windows):
    scores, boxes = [], []
    for start in range(0, len(windows), 96):
        batch = windows[start:start + 96]
        images = np.stack([patch(rgb, window) for window in batch])
        tensor = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).float() / 127.5 - 1
        logits, deltas = model(tensor)
        scores.extend(logits.sigmoid().tolist())
        boxes.extend(decode_box(delta, window) for delta, window in zip(deltas.numpy(), batch))
    return scores, boxes


def proposals(rgb):
    h, w = rgb.shape[:2]
    short = min(h, w)
    windows = []
    # A denser scale pyramid than the original classifier; include a context
    # window for portraits whose face occupies most of the photograph.
    for factor in [.18, .25, .35, .49, .68, .94, 1.25]:
        side = short * factor
        nx = max(1, int(np.ceil(max(w - side, 0) / (side * .32))) + 1)
        ny = max(1, int(np.ceil(max(h - side, 0) / (side * .32))) + 1)
        xs = np.linspace(min(0, (w - side) / 2), max(0, w - side), nx)
        ys = np.linspace(min(0, (h - side) / 2), max(0, h - side), ny)
        windows.extend([float(x), float(y), float(side), float(side)] for y in ys for x in xs)
    return windows


def find_faces(model, rgb, threshold=.9):
    h, w = rgb.shape[:2]
    windows = proposals(rgb)
    scores, boxes = predict_windows(model, rgb, windows)
    selected = []
    for index in np.argsort(scores)[::-1]:
        if scores[index] < threshold:
            break
        box = boxes[index]
        if not np.isfinite(box).all() or min(box[2:]) < 30:
            continue
        # Do not clip a partial face into a seemingly valid complete one.
        if inside_fraction(box, [0, 0, w, h]) < .98:
            continue
        x, y = max(box[0], 0.), max(box[1], 0.)
        box = [x, y, min(box[0] + box[2], w) - x, min(box[1] + box[3], h) - y]
        selected.append({"box": box, "window": windows[index], "proposal_score": float(scores[index])})
        if len(selected) >= 80:
            break
    if selected:
        # Verify each regressed box using the same centered, padded framing as
        # the landmark network. Partial proposals cannot directly reach it.
        centered = [padded_box(item["box"]) for item in selected]
        verification, refined = predict_windows(model, rgb, centered)
        for item, score, check in zip(selected, verification, refined):
            item.update(score=float(min(score, item["proposal_score"])),
                        centered_score=float(score), box_agreement_iou=float(iou(item["box"], check)),
                        score_kind="uncalibrated_complete_face_classifier")
        selected = [item for item in selected if item["score"] >= threshold and item["box_agreement_iou"] >= .55]
        # Suppress duplicates only AFTER centered-crop verification. Otherwise
        # a high-scoring fragment can suppress the complete face and then fail
        # verification itself, leaving no candidate (the reported portrait).
        selected.sort(key=lambda item: item["score"] * item["box_agreement_iou"], reverse=True)
        clusters=[]
        for item in selected:
            cluster=next((group for group in clusters if iou(item["box"],group[0]["box"])>.35),None)
            if cluster is None:
                clusters.append([item])
            else:
                cluster.append(item)
        distinct=[]
        for group in clusters:
            if len(group)<3:
                continue
            # Robust box voting reduces dependence on a single window offset.
            # Every coordinate is a trained model prediction, not a template.
            coordinates=np.asarray([item["box"] for item in group])
            fused=np.median(coordinates,axis=0).tolist()
            check=verify_box(model,rgb,fused,threshold)
            if check["complete"]:
                distinct.append({**group[0],"box":fused,"score":check["score"],
                                 "box_agreement_iou":check["box_agreement_iou"],"supporting_windows":len(group)})
        # The app analyzes one person: prefer the largest verified face rather
        # than a tiny background face with a saturated classifier score.
        selected=sorted(distinct,key=lambda item:item["box"][2]*item["box"][3],reverse=True)
    return {"status": "found" if selected else "not_found", "candidates": selected[:5],
            "windows_evaluated": len(windows), "threshold": threshold, "method": ARCHITECTURE, "pipeline_version": PIPELINE_VERSION,
            "message": "Найдена область лица с уточнёнными границами." if selected else
                       "Целое лицо не найдено достаточно надёжно. Загрузите чёткое фото анфас или выделите лицо вручную."}


def verify_box(model, rgb, box, threshold=.9):
    window = padded_box(box)
    scores, refined = predict_windows(model, rgb, [window])
    agreement = float(iou(box, refined[0]))
    coverage = float(inside_fraction(refined[0], window))
    return {"complete": scores[0] >= threshold and agreement >= .55 and coverage >= .98,
            "score": float(scores[0]), "box_agreement_iou": agreement,
            "predicted_face_coverage_in_input": coverage, "threshold": threshold,
            "score_kind": "uncalibrated_complete_face_classifier"}

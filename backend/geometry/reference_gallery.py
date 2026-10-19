"""Bind cached scores to active weights, detector protocol and rating rules."""
import hashlib
import json
from pathlib import Path

from backend.geometry.rating import CATALOG
from backend.ml.face_detector import PIPELINE_VERSION


def gallery_signature(landmark_path: Path, detector_path: Path):
    if not landmark_path.is_file() or not detector_path.is_file():
        return None
    rules=json.dumps(CATALOG,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")
    return {"checkpoint_sha256":hashlib.sha256(landmark_path.read_bytes()).hexdigest(),
            "detector_checkpoint_sha256":hashlib.sha256(detector_path.read_bytes()).hexdigest(),
            "detector_pipeline_version":PIPELINE_VERSION,
            "rating_rules_sha256":hashlib.sha256(rules).hexdigest(),"scan_mode":"automatic_full_image"}


def gallery_is_current(gallery, landmark_path: Path, detector_path: Path):
    signature=gallery_signature(landmark_path,detector_path)
    return signature is not None and all(gallery.get(key)==value for key,value in signature.items())

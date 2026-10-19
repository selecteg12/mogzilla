"""Serve a portable release only after checking both models and the built UI."""
import argparse
import json
import os
import re

from backend.app.main import DIST, ROOT, app, predictor, region_finder
from backend.geometry.reference_gallery import gallery_is_current


def validate_bundle():
    index = DIST / "index.html"
    if not index.is_file():
        raise RuntimeError("Built frontend is missing. Run pnpm build in frontend/.")
    assets = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', index.read_text(encoding="utf-8"))
    if not assets or any(not (DIST / name.lstrip("/")).is_file() for name in assets):
        raise RuntimeError("Built frontend assets are missing.")
    landmarks = predictor.status()
    detector = region_finder.status()
    if not landmarks["ready"]:
        raise RuntimeError("Landmark checkpoint is unavailable: " + str(landmarks["reason"]))
    if not detector["ready"]:
        raise RuntimeError("Face detector checkpoint is unavailable.")
    gallery_path = ROOT / "reports/reference-values/examples.json"
    if not gallery_path.is_file():
        raise RuntimeError("Reference gallery is missing.")
    gallery = json.loads(gallery_path.read_text(encoding="utf-8"))
    if not gallery_is_current(gallery, predictor.path, region_finder.path):
        raise RuntimeError("Reference examples do not match the active models and rating rules.")
    for example in gallery["examples"]:
        if not (ROOT / f"reports/reference-values/example-{example['id']}.jpg").is_file():
            raise RuntimeError("A reference gallery photo is missing.")
    return {"frontend": True, "landmarks": True, "face_detector": True,
            "reference_photos": len(gallery["examples"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without starting a server")
    args = parser.parse_args()
    try:
        status = validate_bundle()
        port = int(os.environ.get("PORT", "8000"))
        if not 1 <= port <= 65535:
            raise ValueError("PORT must be between 1 and 65535.")
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Release validation failed: {exc}\n")
    print(json.dumps(status), flush=True)
    if not args.check:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port, workers=1,
                    limit_concurrency=4, access_log=False)


if __name__ == "__main__":
    main()

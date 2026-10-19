"""Check a deployed release, including real inference on a gallery photo."""
import argparse
import json
import math
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def verify(base_url, wait_seconds=0):
    url = urlsplit(base_url)
    if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment:
        raise ValueError("Use an HTTP(S) application URL without query parameters or fragments.")
    base_url = base_url.rstrip("/")

    def read(path, data=None):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Expected a path on the application origin.")
        headers = {"User-Agent": "FaceMetrics-release-check"}
        if data is not None:
            headers["Content-Type"] = "image/jpeg"
        with urlopen(Request(base_url + path, data=data, headers=headers), timeout=60) as response:
            return response.read()

    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            health = json.loads(read("/api/health"))
            break
        except (URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, HTTPError) and exc.code in {401, 403}:
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(.5)
    if health.get("status") != "ok":
        raise RuntimeError("Health check failed.")
    models = json.loads(read("/api/model"))
    if not models.get("ready") or not models.get("region_finder", {}).get("ready"):
        raise RuntimeError("Both trained models must be available.")
    index = read("/").decode("utf-8")
    assets = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', index)
    if not assets or any(not read(asset) for asset in assets):
        raise RuntimeError("The frontend or its assets are missing.")
    reference = json.loads(read("/api/reference-values"))
    examples = reference.get("gallery", {}).get("examples", [])
    if len(examples) != 3:
        raise RuntimeError("Expected three current reference photos.")
    photos = [read(example["image_url"]) for example in examples]
    analysis = json.loads(read("/api/localize", photos[0])).get("analysis")
    if not analysis or len(analysis.get("landmarks", [])) != 98:
        raise RuntimeError("Face localization and landmark inference failed on the reference crop.")
    score = analysis.get("rating", {}).get("value")
    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 10:
        raise RuntimeError("A real geometry rating was not returned.")
    return {"url": base_url, "both_models_ready": True, "frontend_assets": len(assets),
            "reference_photos": len(photos), "landmarks": len(analysis["landmarks"]),
            "uploaded_reference_crop_score": score}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--wait-seconds", type=float, default=0)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.url, args.wait_seconds), ensure_ascii=False, indent=2))
    except (URLError, TimeoutError, ConnectionError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Release verification failed: {exc}\n")


if __name__ == "__main__":
    main()

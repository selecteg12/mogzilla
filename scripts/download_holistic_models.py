"""Download pinned research weights, verifying SHA-256 before atomic replacement."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "beauty_regressor.pt": (
        "https://huggingface.co/evanlyhf/scut-fbp5500-beauty/resolve/44c77683f3640a259aacff59a2d635875ec65729/beauty_regressor.pt",
        "54f55b2d03fe68b9bdb992d4992ea7d245b4dd25e0d376dca8f599463ff4a312",
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
}


def download():
    target = ROOT / "training/checkpoints/holistic"
    target.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in MODELS.items():
        path = target / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            print(f"Verified: {name}")
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        digest = hashlib.sha256()
        with urlopen(url, timeout=120) as source, temporary.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch: {name}")
        temporary.replace(path)
        print(f"Downloaded and verified: {name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    download()

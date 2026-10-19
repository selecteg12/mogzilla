"""Download original WIDER archives, verify official torchvision MD5, safely extract.

The CUHK-CSE mirror preserves the original archives byte for byte. No model
weights, executable scripts, test images or paid services are downloaded.
"""
import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

ARCHIVES = {
    "train": ("WIDER_train.zip", "3fedf70df600953d25982bcd13d91ba2", 1465602149),
    "val": ("WIDER_val.zip", "dfa7d7e790efa35df3788964cf0bbaea", 362752168),
}


def download(root, split):
    name, expected, expected_size = ARCHIVES[split]
    root.mkdir(parents=True, exist_ok=True)
    archive = root / name
    url = f"https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/{name}"
    temporary = archive.with_suffix(".part")
    if archive.is_file() and archive.stat().st_size < expected_size and not temporary.exists():
        archive.replace(temporary)
    if not archive.is_file():
        if shutil.disk_usage(root).free < 6 * 1024 ** 3:
            raise RuntimeError("At least 6 GB free disk space required before downloading WIDER.")
        start = time.monotonic()
        for attempt in range(4):
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset == expected_size:
                break
            request = urllib.request.Request(url + f"?download=true&offset={offset}",
                                             headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise ValueError("Server did not honor the resume range; refusing to corrupt the archive.")
                with temporary.open("ab" if offset else "wb") as output:
                    size, reported = offset, offset
                    while block := response.read(4 * 1024 ** 2):
                        output.write(block)
                        size += len(block)
                        if size - reported >= 100 * 1024 ** 2:
                            print(json.dumps({"split": split, "downloaded_mb": round(size / 1024 ** 2),
                                              "seconds": round(time.monotonic() - start)}), flush=True)
                            reported = size
        if temporary.stat().st_size != expected_size:
            raise ValueError("Incomplete download retained as .part; run again to resume.")
        temporary.replace(archive)
    checksum = hashlib.md5()
    with archive.open("rb") as stream:
        while block := stream.read(4 * 1024 ** 2):
            checksum.update(block)
    if checksum.hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {archive}; not extracting.")
    with zipfile.ZipFile(archive) as package:
        target = root.resolve()
        for member in package.infolist():
            destination = (target / member.filename).resolve()
            if not destination.is_relative_to(target):
                raise ValueError("Unsafe archive member.")
        package.extractall(target)
    report = {"split": split, "url": url, "bytes": archive.stat().st_size,
              "md5": checksum.hexdigest(), "official_md5_verified": True,
              "checksum_source": "https://github.com/pytorch/vision/blob/main/torchvision/datasets/widerface.py"}
    (root / f"download-{split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("training/datasets/WIDER"))
    parser.add_argument("--splits", nargs="+", choices=list(ARCHIVES), default=["train", "val"])
    arguments = parser.parse_args()
    for selected in arguments.splits:
        download(arguments.root, selected)

"""Import original annotations; split by source image before any augmentation.

python -m training.prepare --dataset 300w --root training/datasets/300w
python -m training.prepare --dataset wflw --root training/datasets/WFLW
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def read_pts(path: Path) -> np.ndarray:
    content = path.read_text(encoding="utf-8-sig")
    body = content.split("{",1)[1].split("}",1)[0]
    points = np.asarray([[float(v) for v in line.split()] for line in body.splitlines() if line.strip()],np.float32)
    if points.shape != (68,2):
        raise ValueError(f"Invalid 68-point annotation: {path}")
    return points-1  # iBUG explicitly uses MATLAB one-based coordinates.


def group_bucket(key: str, seed: int) -> float:
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:12],16)/16**12


def prepare(dataset: str, root: Path, output: Path, seed: int = 42, protocol: str = "official") -> dict:
    root = root.resolve()
    if protocol == "afw-only" and dataset != "300w":
        raise ValueError("afw-only protocol requires the 300w landmark convention.")
    records = []
    if dataset == "wflw":
        files = sorted(root.rglob("list_98pt_rect_attr_train_test/*.txt"))
        if not files:
            files = sorted(p for p in root.rglob("*.txt") if p.name in {"list_98pt_rect_attr_train.txt","list_98pt_rect_attr_test.txt"})
        files = [p for p in files if p.name in {"list_98pt_rect_attr_train.txt","list_98pt_rect_attr_test.txt"}]
        images = [p for p in root.rglob("WFLW_images") if p.is_dir()]
        image_root = images[0] if images else root
        if not files:
            jsons = [p for p in root.rglob("face_landmarks_wflw_*.json") if p.name in {"face_landmarks_wflw_train.json","face_landmarks_wflw_test.json"}]
            if len(jsons) != 2:
                raise ValueError("WFLW train/test annotations missing. Extract original TXT annotations or the OpenMMLab JSON archive.")
            if image_root == root and (root/"images").is_dir():
                image_root = root/"images"
            for annotation in jsons:
                data = json.loads(annotation.read_text(encoding="utf-8"))
                index = {row["id"]:row for row in data["images"]}
                is_test = annotation.stem.endswith("_test")
                for row in data["annotations"]:
                    image = index[row["image_id"]]
                    landmarks = np.asarray(row["keypoints"],float).reshape(-1,3)
                    if landmarks.shape != (98,3) or np.any(landmarks[:,2]<=0):
                        raise ValueError("Expected 98 annotated points per WFLW face.")
                    records.append({"image":(image_root/image["file_name"]).relative_to(root).as_posix(),
                                    "points":landmarks[:,:2].tolist(),"split":"test" if is_test else "train",
                                    "source":"wflw","annotation_source":"OpenMMLab COCO JSON; pixel coordinates used as supplied"})
        for annotation in files:
            is_test = annotation.stem.endswith("_test")
            for line in annotation.read_text().splitlines():
                cols = line.split()
                if not cols:
                    continue
                if len(cols) != 207:
                    raise ValueError(f"Expected 207 WFLW fields, got {len(cols)} in {annotation}")
                points = np.asarray(cols[:196],float).reshape(98,2)
                relative = cols[-1]
                split = "test" if is_test else ("val" if group_bucket(relative,seed)<0.15 else "train")
                records.append({"image":str((image_root/relative).relative_to(root)).replace("\\","/"),
                                "points":points.tolist(),"split":split,"source":"wflw",
                                "attributes":dict(zip(["pose","expression","illumination","makeup","occlusion","blur"],map(int,cols[200:206])))})
    else:
        for annotation in sorted(root.rglob("*.pts")):
            relative = annotation.relative_to(root).as_posix()
            parts = {part.lower() for part in annotation.relative_to(root).parts}
            source = next((s for s in ["lfpw","helen","afw","ibug"] if s in parts),None)
            if source is None:
                continue  # challenge's 600 images and other sets never silently enter training.
            image = next((annotation.with_suffix(ext) for ext in [".jpg",".png",".jpeg",".JPG",".PNG"] if annotation.with_suffix(ext).exists()),None)
            if image is None:
                raise ValueError(f"Missing image next to {annotation}")
            is_test = source == "ibug" or "testset" in parts or "test" in parts
            if source in {"lfpw","helen"} and not is_test and not {"trainset","train"}.intersection(parts):
                raise ValueError(f"Cannot infer official split for {relative}; preserve trainset/testset directories.")
            split = "test" if is_test else ("val" if group_bucket(relative,seed)<0.15 else "train")
            records.append({"image":image.relative_to(root).as_posix(),"points":read_pts(annotation).tolist(),"split":split,"source":source})
    if not records:
        raise ValueError("No usable annotations found. See reports/DATASETS.md for the required folder layout.")
    seen = {}
    source_counts = {}
    fingerprints = {}
    for number,record in enumerate(records):
        path = (root/record["image"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Dataset image path escapes the dataset root.")
        if not path.is_file():
            raise ValueError(f"Missing dataset image: {path}")
        points = np.asarray(record["points"])
        with Image.open(path) as image:
            if image.getexif().get(274,1) not in (1,None):
                raise ValueError(f"Nontrivial EXIF orientation in annotated dataset: {path}; normalize image AND annotations first.")
            w,h = image.size
            if not np.isfinite(points).all() or min(np.ptp(points,axis=0))<1:
                raise ValueError(f"Invalid or degenerate annotation: {path}")
            outside = (points[:,0]<0)|(points[:,1]<0)|(points[:,0]>w-1)|(points[:,1]>h-1)
            record["out_of_frame_points"] = int(outside.sum())
            # Some original WFLW labels extend beyond the photograph. Preserve them,
            # report them, and use the documented reflect crop rather than clipping labels.
            if record["image"] not in fingerprints:
                fingerprints[record["image"]] = hashlib.sha256(ImageOps.exif_transpose(image).convert("RGB").tobytes()+f"{w},{h}".encode()).hexdigest()
            digest = fingerprints[record["image"]]
        # Exact decoded image duplicates across official train/test are an error, not leaked evidence.
        original_role = "test" if record["split"] == "test" else "development"
        seen[digest] = original_role
        record["image_sha256"] = digest
        bucket = group_bucket(digest,seed)
        if protocol == "afw-only":
            if record["source"] != "afw":
                raise ValueError("afw-only input must contain only AFW records; use a separate root.")
            record["split"] = "train" if bucket < .70 else ("val" if bucket < .85 else "test")
        elif original_role != "test":
            record["split"] = "val" if bucket < .15 else "train"
        source_counts[record["source"]] = source_counts.get(record["source"],0)+1
        if number and number%1000==0:
            print(f"Validated {number}/{len(records)} annotated faces",flush=True)
    original_counts = {s:sum(r["split"]==s for r in records) for s in ["train","val","test"]}
    test_hashes = {r["image_sha256"] for r in records if r["split"]=="test"}
    excluded = [r for r in records if r["split"]!="test" and r["image_sha256"] in test_hashes]
    records = [r for r in records if r["split"]=="test" or r["image_sha256"] not in test_hashes]
    source_counts = {source:sum(r["source"]==source for r in records) for source in source_counts}
    counts = {s:sum(r["split"]==s for r in records) for s in ["train","val","test"]}
    if min(counts.values()) < 1:
        raise ValueError(f"All three splits are required; found {counts}.")
    output.mkdir(parents=True,exist_ok=True)
    content = "".join(json.dumps(r,separators=(",",":"))+"\n" for r in records)
    (output/"manifest.jsonl").write_bytes(content.encode("utf-8"))
    metadata = {"dataset":dataset,"root":str(root),"counts":counts,"sources":source_counts,"seed":seed,
                "manifest_sha256":hashlib.sha256(content.encode()).hexdigest(),"coordinate_origin":"zero-based",
                "protocol":protocol,"unique_images":len({r["image_sha256"] for r in records}),
                "excluded_development_faces_sharing_test_image":len(excluded),"original_counts_before_decontamination":original_counts,
                "faces_with_out_of_frame_labels":sum(r["out_of_frame_points"]>0 for r in records),
                "split_unit":"SHA-256 of decoded source image; all faces in the same image stay together",
                "split_policy":"70/15/15 by image hash; NOT the official 300-W benchmark" if protocol == "afw-only" else "Official test preserved; development faces sharing a decoded image with test excluded; 15% of remaining development image hashes reserved for validation. Modified training protocol, not directly comparable to published benchmarks.",
                "limitations":"Identity metadata unavailable: identity-independent splitting cannot be guaranteed. GT-derived crops approximate user-selected ROI."}
    (output/"dataset.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=["wflw","300w"],required=True)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path("training/datasets/prepared"))
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--protocol",choices=["official","afw-only"],default="official")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.dataset,args.root,args.out,args.seed,args.protocol),indent=2))
    except (ValueError,FileNotFoundError) as error:
        parser.exit(2,f"Dataset import failed: {error}\n")

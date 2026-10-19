"""Safely extract the official AFW zip into a dedicated project data directory."""
import argparse
import zipfile
from pathlib import Path


def extract(archive: Path, target: Path):
    target = target.resolve()
    with zipfile.ZipFile(archive) as source:
        files = [info for info in source.infolist() if not info.is_dir()]
        if len(files)>2000 or sum(i.file_size for i in files)>600*1024**2:
            raise ValueError("Unexpected archive size.")
        for item in files:
            relative = Path(item.filename)
            destination = (target/relative).resolve()
            if not destination.is_relative_to(target) or relative.suffix.lower() not in {".jpg",".jpeg",".png",".pts"}:
                raise ValueError(f"Unexpected archive member: {item.filename}")
        target.mkdir(parents=True,exist_ok=True)
        for item in files:
            destination = target/item.filename
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
            destination.write_bytes(source.read(item))
    print(f"Extracted {len(files)} files into {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--target",type=Path,default=Path("training/datasets/300w/afw"))
    args = parser.parse_args()
    extract(args.archive,args.target)

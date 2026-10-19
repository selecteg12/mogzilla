"""Extract WFLW images + OpenMMLab JSON annotations, with path and size validation."""
import argparse
import tarfile
from pathlib import Path


def extract(archive: Path,target: Path):
    target = target.resolve()
    with tarfile.open(archive,"r:*") as source:
        members = source.getmembers()
        if len(members)>100_000 or sum(m.size for m in members)>20*1024**3:
            raise ValueError("Unexpected archive size.")
        for member in members:
            path = (target/member.name).resolve()
            if not path.is_relative_to(target) or not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsafe archive entry: {member.name}")
            if member.isfile() and Path(member.name).suffix.lower() not in {".jpg",".jpeg",".png",".json",".txt"}:
                raise ValueError(f"Unexpected file: {member.name}")
        target.mkdir(parents=True,exist_ok=True)
        for member in members:
            path = target/member.name
            if member.isdir():
                path.mkdir(parents=True,exist_ok=True)
                continue
            if path.exists():
                if path.stat().st_size == member.size:
                    continue
                raise FileExistsError(f"Existing file has a different size: {path}")
            path.parent.mkdir(parents=True,exist_ok=True)
            with source.extractfile(member) as stream,path.open("wb") as output:
                while chunk := stream.read(1024**2):
                    output.write(chunk)
    print(f"Extracted {archive.name} into {target}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives",type=Path,nargs="+")
    parser.add_argument("--target",type=Path,default=Path("training/datasets/WFLW"))
    args = parser.parse_args()
    for archive in args.archives:
        extract(archive,args.target)

"""Render a labeled contact sheet for visual QA of public gallery candidates."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from backend.preprocessing.images import crop_image


if __name__=="__main__":
    directory=Path("reports/reference-values")
    report=json.loads((directory/"candidates.json").read_text(encoding="utf-8"))
    root=Path(json.loads(Path("training/datasets/prepared-wflw/dataset.json").read_text())["root"])
    rows=report["candidates"][:20]
    figure,axes=plt.subplots(5,4,figsize=(12,15))
    for index,(axis,row) in enumerate(zip(axes.flat,rows)):
        with Image.open(root/row["image"]) as source:
            crop,_=crop_image(np.asarray(source.convert("RGB")),row["box"],size=256)
        axis.imshow(crop)
        axis.set_title(f"#{index} · {row['rating']['value']:.2f}/10 · NME {row['nme']:.3f}",fontsize=10)
        axis.axis("off")
    for axis in list(axes.flat)[len(rows):]:
        axis.axis("off")
    figure.suptitle("Кандидаты после автоматического анализа · порядок по баллу",fontsize=13)
    figure.tight_layout()
    path=directory/"candidate-preview.jpg"
    figure.savefig(path,dpi=110)
    plt.close(figure)
    print(str(path.resolve()))

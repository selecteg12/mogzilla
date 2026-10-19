import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(source: Path, output: Path):
    with source.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fig,ax = plt.subplots(1,2,figsize=(10,4))
    epochs = [int(row["epoch"]) for row in rows]
    for key,label in [("train_loss","Обучение"),("validation_loss","Валидация")]:
        ax[0].plot(epochs,[float(r[key]) for r in rows],label=label)
    ax[0].set(xlabel="Эпоха",ylabel="Функция потерь архитектуры",title="Фактическая история обучения")
    ax[0].legend()
    ax[1].plot(epochs,[float(r["validation_nme"])*100 for r in rows])
    ax[1].set(xlabel="Эпоха",ylabel="NME валидации (%)",title="Локализация ориентиров")
    fig.tight_layout()
    fig.savefig(output,dpi=160)
    plt.close(fig)

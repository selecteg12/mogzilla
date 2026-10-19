import os
import threading
from pathlib import Path

import numpy as np
import torch

from backend.geometry.topology import make_topology
from backend.ml.models import create_model
from backend.preprocessing.images import image_points

ROOT = Path(__file__).resolve().parents[2]


class ModelUnavailable(Exception):
    pass


class Predictor:
    def __init__(self,path: Path | None = None):
        self.path = path or Path(os.environ.get("FACEMETRICS_CHECKPOINT",str(ROOT/"training/checkpoints/selected.pt")))
        self.lock = threading.Lock()
        self.model = None
        self.metadata = None
        self.signature = None
        self.problem = "Обученные веса не найдены. Импортируйте данные, обучите модель с нуля, оцените и экспортируйте её."
        torch.set_num_threads(min(4,os.cpu_count() or 1))

    def _load(self):
        if not self.path.is_file():
            self.model = None
            self.metadata = None
            self.problem = "Обученные веса не найдены. Импортируйте данные, обучите модель с нуля, оцените и экспортируйте её."
            return
        signature = (self.path.stat().st_mtime_ns,self.path.stat().st_size)
        if signature == self.signature:
            return
        self.signature = signature
        self.model = None
        self.metadata = None
        try:
            ck = torch.load(self.path,map_location="cpu",weights_only=True)
            topology = make_topology(ck["dataset"])
            if (ck["initialization"]!="random_from_scratch" or ck["epochs"]<1 or ck["training_steps"]<1
                or ck["landmarks"]!=topology.count or not ck.get("manifest_sha256")
                or ck.get("test",{}).get("samples",0)<1 or ck.get("evaluation",{}).get("selection")!="validation_only"
                or ck["input_size"]!=96 or ck.get("preprocessing",{}).get("margin")!=.12):
                raise ValueError("Checkpoint must be trained from scratch and independently evaluated with this preprocessing.")
            model = create_model(ck["architecture"],topology.count)
            model.load_state_dict(ck["state_dict"],strict=True)
            if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
                raise ValueError("Checkpoint has non-finite parameters.")
            self.model = model.eval()
            self.metadata = {k:v for k,v in ck.items() if k!="state_dict"}
            self.problem = None
        except Exception as exc:
            self.problem = f"Checkpoint unavailable or incompatible: {type(exc).__name__}: {exc}"

    def status(self) -> dict:
        with self.lock:
            self._load()
            return {"ready":self.model is not None,"reason":self.problem,"metadata":self.metadata,
                    "privacy":"Photos are decoded in RAM and never persisted by this application.",
                    "detector":"Выделенная область лица only; no pretrained detector or external AI service."}

    def predict(self,crop: np.ndarray,transform: dict) -> dict:
        with self.lock:
            self._load()
            if self.model is None:
                raise ModelUnavailable(self.problem)
            topology = make_topology(self.metadata["dataset"])
            tensor = torch.from_numpy(np.ascontiguousarray(crop.transpose(2,0,1))).float().unsqueeze(0)/127.5-1
            with torch.inference_mode():
                output = self.model(tensor)
                coordinates = output["coordinates"][0].numpy()
                flipped = self.model(tensor.flip(-1))["coordinates"][0].numpy()[list(topology.flip)].copy()
                flipped[:,0] = 1-flipped[:,0]
            if not np.isfinite(coordinates).all():
                raise ModelUnavailable("Model produced invalid coordinates.")
            normalizer = float(np.linalg.norm(coordinates[topology.normalizer[0]]-coordinates[topology.normalizer[1]]))
            disagreement = float(np.linalg.norm(coordinates-flipped,axis=1).mean()/max(normalizer,1e-6))
            result = {"points":image_points(coordinates,transform),"normalized":coordinates.tolist(),
                      "flip_disagreement_nme":disagreement,"metadata":self.metadata}
            if "heatmaps" in output:
                logits = output["heatmaps"][0]
                result["heatmaps"] = logits.flatten(1).softmax(-1).reshape(logits.shape).numpy()
            return result

import numpy as np
from backend.geometry.topology import make_topology

FAILURE_THRESHOLD = 0.10  # declared before evaluating, strict NME > threshold


def errors(predicted: np.ndarray, target: np.ndarray, dataset: str) -> tuple[np.ndarray,np.ndarray]:
    topology = make_topology(dataset)
    a,b = topology.normalizer
    normalizer = np.linalg.norm(target[:,a]-target[:,b],axis=-1)
    if np.any(normalizer <= 1e-8):
        raise ValueError("Degenerate ground-truth interocular distance.")
    localization = np.linalg.norm(predicted-target,axis=-1).mean(-1)
    return localization/normalizer,localization


def summarize(nme: np.ndarray, localization: np.ndarray) -> dict:
    return {"samples":len(nme),"nme":float(np.mean(nme)),"median_nme":float(np.median(nme)),
            "p90_nme":float(np.quantile(nme,0.9)),"mean_localization_error_px":float(np.mean(localization)),
            "failure_threshold":FAILURE_THRESHOLD,"failure_rate":float(np.mean(nme>FAILURE_THRESHOLD))}

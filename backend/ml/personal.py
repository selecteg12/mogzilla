"""Small, regularized personal scoring head on frozen SCUT visual features."""
import hashlib
import json
from pathlib import Path

import numpy as np

from backend.ml.inference import ModelUnavailable

FACE_MARGINS = (.12, .20, .28)
HEAD_MARGINS = (.32, .40, .48)


def feature_matrix(features: np.ndarray, mode: str) -> np.ndarray:
    """Input (..., 2, 2048): face and wider head crops, already flip averaged."""
    values = np.asarray(features, dtype=np.float64)
    values = values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
    if mode == 'face':
        return values[..., 0, :]
    if mode == 'head':
        return values[..., 1, :]
    if mode == 'dual':
        return values.reshape(*values.shape[:-2], -1) / np.sqrt(2)
    if mode == 'scalar':
        raise ValueError('Scalar calibration uses original scores, not embeddings')
    raise ValueError(f'Unknown feature mode: {mode}')


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> dict:
    """Dual ridge with an unpenalized intercept; all statistics fit on x only."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    weights = weights / np.mean(weights)
    mean = np.average(x, axis=0, weights=weights)
    intercept = float(np.average(y, weights=weights))
    centered = (x-mean) * np.sqrt(weights[:,None])
    target = (y-intercept) * np.sqrt(weights)
    dual = np.linalg.solve(centered @ centered.T + alpha*np.eye(len(y)), target)
    coef = centered.T @ dual
    return {'coef': coef, 'intercept': intercept-float(mean@coef)}


def apply_head(x: np.ndarray, head: dict, base: np.ndarray | None = None) -> np.ndarray:
    values = x @ head['coef'] + head['intercept']
    if head.get('residual'):
        if base is None:
            raise ValueError('Residual head requires original predictions')
        values = values + base
    return np.clip(values, 1, 10)


class PersonalHead:
    def __init__(self, model_dir: Path):
        metadata_path = model_dir / 'personal-preference.json'
        self.metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if self.metadata.get('schema') != 'facelab.personal-head.v1':
            raise ModelUnavailable('Неизвестный формат персональной модели.')
        path = model_dir / 'personal-preference.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata['weights_sha256']:
            raise ModelUnavailable('Контрольная сумма персональной модели не совпала.')
        backbone = model_dir / 'beauty_regressor.pt'
        if hashlib.sha256(backbone.read_bytes()).hexdigest() != self.metadata['backbone_sha256']:
            raise ModelUnavailable('Персональная модель требует другую версию базовых весов.')
        with np.load(path, allow_pickle=False) as weights:
            self.head = {'coef': weights['coef'].copy(), 'intercept': float(weights['intercept']), 'residual': bool(self.metadata['residual'])}
        expected = {'face':2048, 'head':2048, 'dual':4096, 'scalar':1}.get(self.metadata['feature_mode'])
        if self.head['coef'].shape != (expected,) or not np.isfinite(self.head['coef']).all() or not np.isfinite(self.head['intercept']):
            raise ModelUnavailable('Некорректные параметры персональной модели.')

    def predict(self, features: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
        mode = self.metadata['feature_mode']
        x = np.asarray(base_scores)[:,None] if mode == 'scalar' else feature_matrix(features, mode)
        return apply_head(x, self.head, np.asarray(base_scores))

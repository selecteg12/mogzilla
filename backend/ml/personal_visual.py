"""A hash-checked, locally fine-tuned last visual block; base SCUT stays intact."""
import copy
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from backend.ml.inference import ModelUnavailable


def normalized_views(pooled: torch.Tensor) -> torch.Tensor:
    """B x [face, head] x [original, flip] x 2048 -> B x 4096."""
    averaged = pooled.mean(dim=2)
    return F.normalize(averaged, dim=-1).flatten(1) / (2 ** .5)


class TrainableVisualHead(nn.Module):
    def __init__(self, block, coef, intercept):
        super().__init__()
        self.block = copy.deepcopy(block).eval()
        # Small personal batches must not alter pretrained BatchNorm statistics.
        for module in self.block.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(False)
        self.coef = nn.Parameter(torch.as_tensor(coef, dtype=torch.float32).clone())
        self.intercept = nn.Parameter(torch.tensor(float(intercept), dtype=torch.float32))

    def features(self, maps):
        batch = maps.shape[0]
        pooled = self.block(maps.reshape(-1,2048,7,7)).mean(dim=(-2,-1))
        return normalized_views(pooled.reshape(batch,2,2,2048))

    def forward(self, maps):
        return self.features(maps) @ self.coef + self.intercept


def load_visual_block(model_dir: Path, metadata: dict, base_block):
    description = metadata.get('visual_adapter')
    if description is None:
        return None
    if description.get('schema') != 'facelab.last-bottleneck.v1' or description.get('block') != 'layer4.2':
        raise ModelUnavailable('Неизвестный формат дообученного зрительного блока.')
    name = description.get('file')
    if not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name:
        raise ModelUnavailable('Некорректный путь дообученных весов.')
    path = model_dir / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != description.get('sha256'):
        raise ModelUnavailable('Контрольная сумма дообученного зрительного блока не совпала.')
    block = copy.deepcopy(base_block)
    state = torch.load(path, map_location='cpu', weights_only=True)
    block.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all() for value in block.state_dict().values()):
        raise ModelUnavailable('Дообученный зрительный блок содержит некорректные параметры.')
    return block.eval().requires_grad_(False)

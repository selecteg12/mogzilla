import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from backend.ml.evaluation import errors,summarize
from backend.ml.models import create_model,landmark_loss,soft_argmax
from training.prepare import prepare


@pytest.mark.parametrize("architecture",["coordinates","heatmaps"])
@pytest.mark.parametrize("count",[68,98])
def test_both_architectures_backpropagate(architecture,count):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = create_model(architecture,count)
    output = model(torch.randn(2,3,96,96))
    assert output["coordinates"].shape==(2,count,2)
    loss = landmark_loss(output,torch.rand(2,count,2))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not torch.equal(output["coordinates"][0],output["coordinates"][1])


def test_uniform_soft_argmax_is_center():
    assert torch.allclose(soft_argmax(torch.zeros(1,1,48,48)),torch.tensor([[[.5,.5]]]))


def test_nme_normalization_and_failure_threshold():
    target = np.zeros((2,68,2))
    target[:,45,0] = 100
    target[:,36,0] = 0
    prediction = target.copy(); prediction[0,:,1] += 5; prediction[1,:,1] += 15
    nme,px = errors(prediction,target,"300w")
    assert np.allclose(nme,[.05,.15])
    metrics = summarize(nme,px)
    assert metrics["failure_rate"]==.5
    assert metrics["mean_localization_error_px"]==10


def test_real_dataset_splits_and_fingerprint_when_available():
    directory = Path("training/datasets/prepared-afw")
    if not (directory/"dataset.json").is_file():
        pytest.skip("Local licensed AFW not imported")
    metadata = json.loads((directory/"dataset.json").read_text())
    content = (directory/"manifest.jsonl").read_bytes()
    assert hashlib.sha256(content).hexdigest()==metadata["manifest_sha256"]
    seen = {}
    for line in content.splitlines():
        row = json.loads(line)
        if row["image_sha256"] in seen:
            assert seen[row["image_sha256"]]==row["split"]
        seen[row["image_sha256"]] = row["split"]
    assert set(seen.values())=={"train","val","test"}


def test_wflw_image_independence_when_available():
    directory = Path("training/datasets/prepared-wflw")
    if not (directory/"dataset.json").is_file():
        pytest.skip("Local licensed WFLW not imported")
    metadata = json.loads((directory/"dataset.json").read_text())
    content = (directory/"manifest.jsonl").read_bytes()
    assert hashlib.sha256(content).hexdigest()==metadata["manifest_sha256"]
    splits = {s:set() for s in ["train","val","test"]}
    for line in content.splitlines():
        row = json.loads(line)
        splits[row["split"]].add(row["image_sha256"])
        assert len(row["points"])==98
    assert not splits["train"] & splits["val"]
    assert not splits["train"] & splits["test"]
    assert not splits["val"] & splits["test"]
    assert all(splits.values())

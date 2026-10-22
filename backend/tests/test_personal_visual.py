"""Numerical parity, real gradient updates, frozen BN and artifact validation."""
import hashlib
import copy

import numpy as np
import pytest
import torch
from torch import nn

from backend.ml.inference import ModelUnavailable
from backend.ml.holistic import HolisticAnalyzer
from backend.ml.personal import feature_matrix
from backend.ml.personal_visual import TrainableVisualHead, load_visual_block, normalized_views
from scripts.finetune_personal import sample_weights


def small_block():
    return nn.Sequential(nn.Conv2d(2048,2048,1,groups=2048,bias=False),nn.BatchNorm2d(2048),nn.ReLU())


def test_train_and_inference_average_flips_before_normalizing_identically():
    values = np.random.default_rng(7).normal(size=(3,2,2,2048)).astype(np.float32)
    actual = normalized_views(torch.from_numpy(values)).numpy()
    assert np.allclose(actual,feature_matrix(values.mean(axis=2),'dual'),atol=2e-8)


def test_gradients_change_visual_weights_without_changing_bn_or_original_block():
    torch.set_num_threads(4)
    torch.manual_seed(3)
    original = small_block().eval()
    original_state = {k:v.clone() for k,v in original.state_dict().items()}
    model = TrainableVisualHead(original,np.ones(4096),3.)
    before = model.block[0].weight.detach().clone()
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),lr=.01)
    loss = model(torch.randn(2,2,2,2048,7,7)).square().mean()
    loss.backward()
    assert torch.isfinite(model.block[0].weight.grad).all()
    optimizer.step()
    assert not torch.equal(before,model.block[0].weight)
    assert all(torch.equal(original.state_dict()[k],v) for k,v in original_state.items())
    assert torch.equal(model.block[1].running_mean,original[1].running_mean)
    assert model.block[1].weight.grad is None


def test_visual_checkpoint_validates_checksum_and_local_filename(tmp_path):
    block = small_block().eval()
    path = tmp_path/'visual.pt'
    torch.save(block.state_dict(),path)
    description = {'schema':'facelab.last-bottleneck.v1','block':'layer4.2','file':'visual.pt','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    loaded = load_visual_block(tmp_path,{'visual_adapter':description},block)
    assert not loaded.training
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ModelUnavailable,match='сумма'):
        load_visual_block(tmp_path,{'visual_adapter':description},block)
    with pytest.raises(ModelUnavailable,match='путь'):
        load_visual_block(tmp_path,{'visual_adapter':{**description,'file':'../visual.pt'}},block)


def test_sample_weights_ignore_all_validation_labels_and_sources():
    records = [{'cohort':'a' if i%2 else 'b','score':i%10+1} for i in range(20)]
    groups = np.array([f'g{i//2}' for i in range(20)])
    fitting = np.arange(12)
    first = sample_weights(records,groups,fitting)
    for i in range(12,20):
        records[i] = {'cohort':'new','score':1000}
    assert np.array_equal(first,sample_weights(records,groups,fitting))


def test_personal_visual_changes_do_not_change_original_scut_prediction():
    from torchvision.models import resnet50
    torch.manual_seed(19)
    torch.set_num_threads(4)
    analyzer = HolisticAnalyzer(personal=False)
    analyzer.model = resnet50(weights=None,num_classes=1).eval()
    crop = np.random.default_rng(4).integers(0,256,size=(224,224,3),dtype=np.uint8)
    original_features, original_scores = analyzer.embed_crops([crop])
    analyzer.visual_block = copy.deepcopy(analyzer.model.layer4[2]).eval()
    with torch.no_grad():
        analyzer.visual_block.conv1.weight.mul_(1.2)
    updated_features, original_scores_again = analyzer.embed_crops([crop])
    assert original_scores == original_scores_again
    assert not np.allclose(original_features,updated_features)

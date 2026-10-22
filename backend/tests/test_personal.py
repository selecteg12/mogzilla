"""Personal transfer learning: numerical fit, split isolation, artifact safety."""
import hashlib
import json

import numpy as np
import pytest

from backend.ml.inference import ModelUnavailable
from backend.ml.personal import PersonalHead, apply_head, feature_matrix, fit_ridge
from scripts.train_personal import folds_for, ranking_metrics, train_candidate


def test_ridge_matches_primal_solution_with_unpenalized_intercept():
    rng=np.random.default_rng(8)
    x=rng.normal(size=(24,5))
    y=x@np.arange(5)+3+rng.normal(size=24)*.1
    head=fit_ridge(x,y,.3)
    center=x-x.mean(axis=0)
    expected=np.linalg.solve(center.T@center+.3*np.eye(5),center.T@(y-y.mean()))
    assert np.allclose(head['coef'],expected)
    assert head['intercept']==pytest.approx(y.mean()-x.mean(axis=0)@expected)


def test_holdout_labels_do_not_change_fitted_head_and_groups_never_cross_folds():
    groups=np.array([f'p{i//2}' for i in range(60)])
    y=np.tile(np.arange(1,11),6).astype(float)
    folds=folds_for(groups,y,5,123)
    assert sorted(np.concatenate(folds))==list(range(60))
    for i,a in enumerate(folds):
        for b in folds[i+1:]:
            assert not set(groups[a]) & set(groups[b])
    dev=np.concatenate(folds[1:])
    x=np.random.default_rng(42).normal(size=(60,8))
    baseline=np.full(60,6.)
    first=train_candidate(x,y,baseline,dev,groups,.1,True)
    y[folds[0]]=9999
    second=train_candidate(x,y,baseline,dev,groups,.1,True)
    assert np.array_equal(first['coef'],second['coef'])
    assert first['intercept']==second['intercept']


def test_personal_features_normalize_views_and_require_residual_base():
    x=np.ones((3,2,2048))
    assert np.allclose(np.linalg.norm(feature_matrix(x,'dual'),axis=1),1)
    assert feature_matrix(x,'face').shape==(3,2048)
    with pytest.raises(ValueError,match='original'):
        apply_head(np.zeros((2,3)),{'coef':np.zeros(3),'intercept':4,'residual':True})


def test_personal_checkpoint_verifies_both_head_and_backbone(tmp_path):
    weights=tmp_path/'personal-preference.npz'
    np.savez(weights,coef=np.ones(2048),intercept=np.array(1.))
    backbone=tmp_path/'beauty_regressor.pt'
    backbone.write_bytes(b'backbone')
    metadata={'schema':'facelab.personal-head.v1','weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
              'backbone_sha256':hashlib.sha256(backbone.read_bytes()).hexdigest(),'residual':False,'feature_mode':'face'}
    (tmp_path/'personal-preference.json').write_text(json.dumps(metadata),encoding='utf-8')
    assert PersonalHead(tmp_path).head['coef'].shape==(2048,)
    weights.write_bytes(weights.read_bytes()+b'changed')
    with pytest.raises(ModelUnavailable,match='сумма'):
        PersonalHead(tmp_path)


def test_ranking_metrics_handle_tied_human_scores_and_reverse_order():
    scores=np.array([1,1,2,3,3])
    same=ranking_metrics(scores,scores)
    reverse=ranking_metrics(scores,10-scores)
    assert same=={'spearman':1.0,'pairwise_at_least_two_points':1.0,'pairs':4}
    assert reverse=={'spearman':-1.0,'pairwise_at_least_two_points':0.0,'pairs':4}

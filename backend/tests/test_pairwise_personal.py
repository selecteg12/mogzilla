import numpy as np
import torch

from backend.ml.personal import fit_ridge
from scripts.train_personal_pairwise import (
    fit_pairwise, joint_loss, preference_pairs, sample_weights,
)


def test_preferences_have_correct_direction_gap_and_no_same_group():
    y = np.array([9., 7., 7., 4.])
    groups = np.array(['same', 'b', 'same', 'd'])
    high, low = preference_pairs(y, groups)
    assert set(zip(high, low)) == {(0, 1), (0, 3), (1, 3), (2, 3)}
    assert np.all(y[high]-y[low] >= 2)
    assert np.all(groups[high] != groups[low])


def test_joint_loss_gradient_matches_finite_differences():
    rng = np.random.default_rng(23)
    x = torch.tensor(rng.normal(size=(5, 3)), dtype=torch.float64)
    y = torch.tensor([1., 4., 6., 8., 9.], dtype=torch.float64)
    w = torch.tensor([1., .5, 1., 1., 2.], dtype=torch.float64)
    hi, lo = torch.tensor([3, 4, 4]), torch.tensor([0, 1, 2])
    pw = torch.tensor([.2, .3, .5], dtype=torch.float64)
    parameters = torch.tensor([.2, -.3, .1, 5.], dtype=torch.float64, requires_grad=True)
    value = joint_loss(parameters, x, y, w, hi, lo, pw, .4, 2.)
    value.backward()
    analytic = parameters.grad.numpy().copy()
    numerical = []
    for i in range(4):
        plus, minus = parameters.detach().clone(), parameters.detach().clone()
        plus[i] += 1e-5
        minus[i] -= 1e-5
        numerical.append(float((joint_loss(plus, x, y, w, hi, lo, pw, .4, 2.)
                                - joint_loss(minus, x, y, w, hi, lo, pw, .4, 2.))/2e-5))
    assert np.allclose(analytic, numerical, atol=1e-8)


def test_zero_pairwise_weight_is_exact_ridge_with_same_group_weights():
    rng = np.random.default_rng(9)
    x, y = rng.normal(size=(12, 4)), rng.uniform(1, 10, 12)
    groups = np.array([str(i//2) for i in range(12)])
    cohorts = np.array(['a']*6 + ['b']*6)
    weights = sample_weights(y, groups, cohorts)
    expected = fit_ridge(x, y, .2, weights)
    actual, _ = fit_pairwise(x, y, groups, cohorts, .2, 0.)
    assert np.array_equal(expected['coef'], actual['coef'])
    assert expected['intercept'] == actual['intercept']


def test_pairwise_fitting_is_invariant_to_input_order():
    torch.set_num_threads(2)
    rng = np.random.default_rng(12)
    x = rng.normal(size=(18, 3))
    y = np.clip(5 + 2*x[:, 0] - x[:, 1], 1, 10)
    groups = np.array([str(i) for i in range(18)])
    cohorts = np.array(['a']*9 + ['b']*9)
    first, _ = fit_pairwise(x, y, groups, cohorts, .5, 2., True)
    order = rng.permutation(18)
    second, _ = fit_pairwise(x[order], y[order], groups[order], cohorts[order], .5, 2., True)
    assert np.allclose(first['coef'], second['coef'], atol=1e-5)
    assert abs(first['intercept']-second['intercept']) < 1e-5

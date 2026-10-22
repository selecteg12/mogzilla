"""Fit an auditable linear RankNet + score head on cached local features.

Only development-fold labels form training pairs. Existing holdout results have
already been inspected; their later comparison is explicitly retrospective.
The exported head uses the existing safe, linear personal-head format.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from backend.ml.personal import apply_head, feature_matrix, fit_ridge
from scripts.train_personal import folds_for, metrics, ranking_metrics


def sample_weights(scores, groups, cohorts, balanced=False):
    sizes = Counter(groups)
    weights = np.array([1 / sizes[g] for g in groups], dtype=np.float64)
    if balanced:
        bins = np.clip((np.asarray(scores).astype(int) - 1) // 2, 0, 4)
        for categories in (np.asarray(cohorts), bins):
            counts = Counter(categories)
            target = len(categories) / len(counts)
            weights *= np.sqrt([target / counts[value] for value in categories])
    return weights / weights.mean()


def preference_pairs(scores, groups):
    scores, groups = np.asarray(scores), np.asarray(groups)
    higher, lower = np.where((scores[:, None] - scores[None, :] >= 2)
                             & (groups[:, None] != groups[None, :]))
    return higher.astype(np.int64), lower.astype(np.int64)


def preference_weights(higher, lower, weights, cohorts, balanced=False):
    values = np.sqrt(weights[higher] * weights[lower])
    if balanced and len(values):
        # Equal total ranking loss per observed source-pair stratum, including
        # within-source comparisons. All masses are computed on training only.
        keys = [tuple(sorted((cohorts[i], cohorts[j]))) for i, j in zip(higher, lower)]
        masses = Counter()
        for key, value in zip(keys, values):
            masses[key] += float(value)
        target = values.sum() / len(masses)
        values *= np.array([target / masses[key] for key in keys])
    return values / values.sum() if len(values) else values


def joint_loss(parameters, x, y, weights, higher, lower, pair_weights, alpha, rank_weight):
    scores = x @ parameters[:-1] + parameters[-1]
    regression = .5 * torch.sum(weights * (scores-y).square()) / weights.sum()
    regularization = .5 * alpha / len(y) * parameters[:-1].square().sum()
    ranking = (pair_weights * F.softplus(-(scores[higher]-scores[lower]))).sum()
    return regression + regularization + rank_weight * ranking


def fit_pairwise(x, y, groups, cohorts, alpha, rank_weight, balanced=False, max_iter=120):
    """Arguments contain training rows only; no validation labels are accepted."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    groups, cohorts = np.asarray(groups), np.asarray(cohorts)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid training arrays')
    weights = sample_weights(y, groups, cohorts, balanced)
    head = fit_ridge(x, y, alpha, weights)
    head['residual'] = False
    higher, lower = preference_pairs(y, groups)
    if not rank_weight or not len(higher):
        return head, {'pairs': len(higher), 'iterations': 0, 'gradient_inf': 0.}
    pair_weights = preference_weights(higher, lower, weights, cohorts, balanced)
    xt, yt, wt = [torch.as_tensor(a, dtype=torch.float64) for a in (x, y, weights)]
    ht, lt = torch.as_tensor(higher), torch.as_tensor(lower)
    pt = torch.as_tensor(pair_weights, dtype=torch.float64)
    parameters = torch.nn.Parameter(torch.as_tensor(np.r_[head['coef'], head['intercept']], dtype=torch.float64))
    optimizer = torch.optim.LBFGS([parameters], max_iter=max_iter, max_eval=max_iter*2,
                                 tolerance_grad=1e-8, tolerance_change=1e-11,
                                 history_size=25, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = joint_loss(parameters, xt, yt, wt, ht, lt, pt, alpha, rank_weight)
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = closure()
    values = parameters.detach().numpy().copy()
    if not np.isfinite(values).all() or not torch.isfinite(loss):
        raise ValueError('Non-finite learned parameters')
    state = optimizer.state[parameters]
    return {'coef': values[:-1], 'intercept': float(values[-1]), 'residual': False}, {
        'pairs': len(higher), 'iterations': state['n_iter'], 'loss': float(loss.detach()),
        'gradient_inf': float(parameters.grad.abs().max()),
    }


def evaluate(scores, predictions, cohorts):
    report = {**metrics(scores, predictions), **ranking_metrics(scores, predictions)}
    report['by_cohort'] = {}
    for cohort in sorted(set(cohorts)):
        mask = cohorts == cohort
        report['by_cohort'][str(cohort)] = {
            **metrics(scores[mask], predictions[mask]),
            **ranking_metrics(scores[mask], predictions[mask]),
        }
    report['macro_cohort_spearman'] = round(float(np.mean([
        r['spearman'] for r in report['by_cohort'].values()])), 4)
    report['macro_cohort_pairwise'] = round(float(np.mean([
        r['pairwise_at_least_two_points'] for r in report['by_cohort'].values()
        if r['pairwise_at_least_two_points'] is not None])), 4)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261021)
    args = parser.parse_args()
    root, out = args.source, args.output
    if out.exists():
        raise ValueError('Output already exists; preserve earlier experiment artifacts.')
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    source = json.loads((root/'features.json').read_text(encoding='utf-8'))
    split = json.loads((root/'split.json').read_text(encoding='utf-8'))
    reviewed = json.loads((root/'groups.json').read_text(encoding='utf-8'))
    assert split['feature_manifest_sha256'] == hashlib.sha256((root/'features.json').read_bytes()).hexdigest()
    assert split['groups_sha256'] == hashlib.sha256((root/'groups.json').read_bytes()).hexdigest()
    records = source['records']
    with np.load(root/'features.npz', allow_pickle=False) as saved:
        features, raw, y = [saved[key].copy() for key in ('features', 'base_raw', 'scores')]
    central = np.stack((features[:, 1].mean(axis=1), features[:, 4].mean(axis=1)), axis=1)
    groups = np.array([reviewed['assignments'][r['id']] for r in records])
    cohorts = np.array([r['cohort'] for r in records])
    dev, test = [np.array(split[key], dtype=int) for key in ('development_indices', 'test_indices')]
    assert not set(groups[dev]) & set(groups[test])
    cv = [dev[fold] for fold in folds_for(groups[dev], y[dev], 4, args.seed+1)]
    configurations = [dict(mode=mode, alpha=alpha, rank_weight=rank_weight, balanced=balanced)
                      for mode in ('face', 'dual') for alpha in (.1, 1.)
                      for rank_weight, balanced in ((0., False), (.5, False), (2., False), (.5, True), (2., True))]
    protocol = {
        'schema': 'facelab.pairwise-protocol.v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'seed': args.seed, 'source': str(root.resolve()),
        'feature_manifest_sha256': split['feature_manifest_sha256'], 'groups_sha256': split['groups_sha256'],
        'development_indices': dev.tolist(), 'audit_indices': test.tolist(),
        'cv_folds': [fold.tolist() for fold in cv], 'configurations': configurations,
        'objective': '0.5 weighted MSE + alpha/(2*n) * ||w||^2 + lambda * mean weighted softplus(-(higher-lower)). Unclipped scores in loss; clip 1..10 at inference.',
        'pairs': 'Only training-fold rows; score gap >=2; different known groups. Generated pairs are dependent, not new independent samples.',
        'selection': 'Best pairwise macro-cohort CV Spearman among candidates whose CV MAE <= best ridge CV MAE + 0.10; tie-break macro pairwise then MAE. No audit labels select settings.',
        'evaluation_status': 'Exploratory. The 173 heldout images, including the 127 newer ones, have already been inspected in earlier experiments. Retrospective comparison is NOT a new blind test.',
        'promotion': 'Keep current models unchanged. Consider separate preview only; no automatic replacement.',
        'backbone_finetuned': False, 'new_human_labels_requested': False,
        'identity_limitations': reviewed['note'],
    }
    (out/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    matrices = {mode: feature_matrix(central, mode) for mode in ('face', 'dual')}
    candidates = []
    started = time.perf_counter()
    for number, config in enumerate(configurations, 1):
        x = matrices[config['mode']]
        predictions = np.full(len(y), np.nan)
        optimizations = []
        for validation in cv:
            train = np.array(sorted(set(dev)-set(validation)), dtype=int)
            assert not set(groups[train]) & set(groups[validation])
            head, info = fit_pairwise(x[train], y[train], groups[train], cohorts[train],
                                      config['alpha'], config['rank_weight'], config['balanced'])
            predictions[validation] = apply_head(x[validation], head)
            optimizations.append(info)
        report = evaluate(y[dev], predictions[dev], cohorts[dev])
        candidate = {**config, 'cv': report, 'optimizations': optimizations}
        candidates.append(candidate)
        np.save(out/f'oof-{number:02d}.npy', predictions[dev], allow_pickle=False)
        (out/'cv-candidates.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'candidate': number, 'total': len(configurations), **config,
                          'cv_mae': report['mae'], 'cv_macro_spearman': report['macro_cohort_spearman'],
                          'elapsed_seconds': round(time.perf_counter()-started, 1)}), flush=True)
    ridge = min((c for c in candidates if c['rank_weight'] == 0), key=lambda c: c['cv']['mae'])
    eligible = [c for c in candidates if c['rank_weight'] > 0 and c['cv']['mae'] <= ridge['cv']['mae']+.10]
    if not eligible:
        raise ValueError('No pairwise candidate passed the CV calibration constraint; CV artifacts preserved.')
    selected = max(eligible, key=lambda c: (c['cv']['macro_cohort_spearman'], c['cv']['macro_cohort_pairwise'], -c['cv']['mae']))
    # Freeze the chosen settings before looking at any retrospective audit values.
    (out/'selection.json').write_text(json.dumps({'selected': selected, 'ridge_baseline': ridge,
                                                'selection_used_audit_labels': False}, ensure_ascii=False, indent=2), encoding='utf-8')
    x = matrices[selected['mode']]
    head, optimization = fit_pairwise(x[dev], y[dev], groups[dev], cohorts[dev],
                                      selected['alpha'], selected['rank_weight'], selected['balanced'])
    predictions = apply_head(x, head)
    weights_path = out/'personal-preference.npz'
    np.savez_compressed(weights_path, coef=head['coef'], intercept=np.array(head['intercept']))
    report = {'schema': 'facelab.pairwise-evaluation.v1', 'selected': selected,
              'ridge_cv_baseline': ridge, 'final_optimization': optimization,
              'cv_rank_improved': selected['cv']['macro_cohort_spearman'] > ridge['cv']['macro_cohort_spearman'],
              'development_count': len(dev), 'test_count': len(test), 'total_rated_images': len(records)+len(source['failures']),
              'train': evaluate(y[dev], predictions[dev], cohorts[dev]),
              'evaluation_status': protocol['evaluation_status'], 'retrospective': {},
              'elapsed_seconds': round(time.perf_counter()-started, 1)}
    previous_predictions = {}
    base = (raw[:, 1].mean(axis=1)-1)*2.5
    for name, where in [('original_personal', root/'previous-active'), ('score_only', root)]:
        meta = json.loads((where/'personal-preference.json').read_text(encoding='utf-8'))
        assert meta['backbone_sha256'] == source['backbone_sha256']
        assert hashlib.sha256((where/'personal-preference.npz').read_bytes()).hexdigest() == meta['weights_sha256']
        with np.load(where/'personal-preference.npz', allow_pickle=False) as saved:
            old_head = {'coef': saved['coef'].copy(), 'intercept': float(saved['intercept']), 'residual': meta['residual']}
        old_x = base[:, None] if meta['feature_mode'] == 'scalar' else feature_matrix(central, meta['feature_mode'])
        previous_predictions[name] = apply_head(old_x, old_head, base)
    previous_predictions['pairwise'] = predictions
    previous_predictions['scut'] = np.clip(base, 0, 10)
    for name, indices in [('all_heldout', test), ('newer_127', np.array(split['fresh_test_indices'], dtype=int))]:
        report['retrospective'][name] = {model: evaluate(y[indices], values[indices], cohorts[indices])
                                        for model, values in previous_predictions.items()}
    (out/'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    new_indices = np.array(split['fresh_test_indices'], dtype=int)
    audit = report['retrospective']['all_heldout']
    old_meta = json.loads((root/'personal-preference.json').read_text(encoding='utf-8'))
    metadata = {**old_meta, 'model_id': 'facelab/personal-pairwise-2026-10-21',
                'revision': hashlib.sha256(weights_path.read_bytes()).hexdigest()[:12],
                'weights_sha256': hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                'feature_mode': selected['mode'], 'alpha': selected['alpha'], 'residual': False,
                'enabled': False, 'training_objective': {'kind': 'score_and_pairwise', **{k: selected[k] for k in ('rank_weight', 'balanced')}},
                'validation': {'original': metrics(y[test], np.clip(base[test], 0, 10)),
                               'personal': metrics(y[test], predictions[test]),
                               'scalar_calibration': old_meta['validation']['scalar_calibration'],
                               'mean_baseline': metrics(y[test], np.full(len(test), y[dev].mean())),
                               'ranking': {'original': ranking_metrics(y[test], base[test]), 'personal': ranking_metrics(y[test], predictions[test])},
                               'preview_only': True, 'evaluation_reused': True,
                               'fresh_comparison': {'previous': report['retrospective']['newer_127']['score_only'],
                                                    'updated': report['retrospective']['newer_127']['pairwise']},
                               'evaluation_note': protocol['evaluation_status']}}
    (out/'personal-preference.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps([
        {'id': r['id'], 'path': r['path'], 'score': float(y[i]), 'cohort': r['cohort'],
         'group': str(groups[i]), 'split': 'test' if i in test else 'development',
         'original': float(base[i]), 'personal': float(predictions[i]),
         'score_only': float(previous_predictions['score_only'][i])}
        for i, r in enumerate(records)], ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'selected': selected, 'ridge_cv_baseline': ridge, 'cv_rank_improved': report['cv_rank_improved'],
                      'newer_127': report['retrospective']['newer_127'], 'elapsed_seconds': report['elapsed_seconds']},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

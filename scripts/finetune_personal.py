"""Bounded last-bottleneck fine-tuning, immutable old audit and new photo holdout.

The selection phase sees development/validation labels only. Audit predictions
are generated once after epoch/model selection; no audit-driven reselection.
"""
import argparse
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet50

from backend.ml.holistic import MODEL_DIR
from backend.ml.personal import PersonalHead, apply_head, feature_matrix, fit_ridge
from backend.ml.personal_visual import TrainableVisualHead
from scripts.train_personal import folds_for, metrics, ranking_metrics


def sample_weights(records, groups, indices):
    """Statistics from the fitting subset only; limit any individual influence."""
    def band(score):
        return min(5, max(0, int(score)-5))
    sources = Counter(records[i]['cohort'] for i in indices)
    bands = Counter(band(records[i]['score']) for i in indices)
    sizes = Counter(groups[i] for i in indices)
    raw = np.array([1 / (math.sqrt(sources[records[i]['cohort']]*bands[band(records[i]['score'])]) * sizes[groups[i]]) for i in indices])
    return np.clip(raw / raw.mean(), .25, 4.)


def batch_maps(cache, records, indices):
    return torch.from_numpy(np.stack([np.load(cache/'maps'/f"{records[i]['id']}.npy", allow_pickle=False) for i in indices]))


def predictions(model, cache, records, indices, batch_size=8):
    output = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            output.extend(model(batch_maps(cache, records, indices[start:start+batch_size])).numpy().tolist())
    return np.clip(np.array(output), 1, 10)


def describe(y, predicted):
    result = {**metrics(y, predicted), **ranking_metrics(y, predicted)}
    upper = y >= 9
    result['upper'] = ({**metrics(y[upper], predicted[upper]),
                       'mean_target':round(float(y[upper].mean()),4),
                       'mean_prediction':round(float(predicted[upper].mean()),4),
                       'signed_error':round(float(np.mean(predicted[upper]-y[upper])),4)} if upper.any() else None)
    return result


def selection_value(y, predicted, reference):
    current, baseline = describe(y, predicted), describe(y, reference)
    valid = current['mae'] <= baseline['mae'] + .10 and current['spearman'] >= baseline['spearman'] - .03
    upper_error = current['upper']['mae'] if current['upper'] else current['mae']
    return (.7*current['mae'] + .3*upper_error) if valid else float('inf'), current


def train_epochs(model, cache, records, indices, groups, count, seed, callback=None):
    rng = np.random.default_rng(seed)
    weights = sample_weights(records, groups, indices)
    y = np.array([records[i]['score'] for i in indices], np.float32)
    optimizer = torch.optim.AdamW([
        {'params':[p for p in model.block.parameters() if p.requires_grad], 'lr':1e-5, 'weight_decay':1e-4},
        {'params':[model.coef, model.intercept], 'lr':.003, 'weight_decay':0.},
    ])
    start = time.perf_counter()
    for epoch in range(1, count+1):
        order = rng.permutation(len(indices))
        total = 0.
        for offset in range(0,len(indices),8):
            position = order[offset:offset+8]
            sample = indices[position]
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch_maps(cache, records, sample))
            losses = (predicted-torch.from_numpy(y[position])).square()
            loss = (losses*torch.tensor(weights[position],dtype=torch.float32)).mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite fine-tuning loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())*len(position)
        info = {'epoch':epoch,'train_weighted_mse':round(total/len(indices),5), 'seconds':round(time.perf_counter()-start,1)}
        if callback:
            callback(epoch, model, info)
        print(json.dumps(info,ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=6)
    parser.add_argument('--seed',type=int,default=20261023)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = args.experiment
    cache = root/'visual-cache'
    source = json.loads((cache/'cache.json').read_text(encoding='utf-8'))
    records = source['records']
    y = np.array([r['score'] for r in records], np.float64)
    grouping = json.loads((root/'groups.json').read_text(encoding='utf-8'))
    groups = np.array([grouping['assignments'].get(r['id'],r['id']) for r in records])
    previous_split = json.loads((args.previous/'split.json').read_text(encoding='utf-8'))
    previous_features = json.loads((args.previous/'features.json').read_text(encoding='utf-8'))
    heldout_ids = {previous_features['records'][i]['id'] for i in previous_split['test_indices']}
    heldout_groups = {groups[i] for i,r in enumerate(records) if r['id'] in heldout_ids}
    fresh_pool = np.array([i for i,r in enumerate(records) if r['cohort']=='upper' and groups[i] not in heldout_groups], dtype=int)
    fresh_test = fresh_pool[folds_for(groups[fresh_pool],y[fresh_pool],5,args.seed)[0]]
    heldout_groups |= set(groups[fresh_test])
    test = np.array([i for i in range(len(records)) if groups[i] in heldout_groups], dtype=int)
    dev = np.array([i for i in range(len(records)) if groups[i] not in heldout_groups], dtype=int)
    validation = dev[folds_for(groups[dev],y[dev],4,args.seed+1)[0]]
    fit = np.array(sorted(set(dev)-set(validation)),dtype=int)
    assert not set(groups[dev]) & set(groups[test])
    assert not set(groups[fit]) & set(groups[validation])
    with np.load(cache/'frozen-features.npz',allow_pickle=False) as saved:
        features = saved['features'].copy()
        base = (saved['base_raw'][:,0].mean(axis=1)-1)*2.5
    x = feature_matrix(features.mean(axis=2), 'dual')
    protocol = {'created_at':datetime.now(timezone.utc).isoformat(),'model_id':'facelab/personal-upper-2026-10-21',
        'source_manifest_sha256':source['manifest_sha256'],'seed':args.seed,'epochs_limit':args.epochs,
        'development_indices':dev.tolist(),'fit_indices':fit.tolist(),'validation_indices':validation.tolist(),
        'test_indices':test.tolist(),'fresh_photo_test_indices':fresh_test.tolist(),
        'fit_count':len(fit),'validation_count':len(validation),'development_count':len(dev),'test_count':len(test),'new_photo_test_count':len(fresh_test),
        'selection':'Epoch zero weighted ridge or 1..6 last-bottleneck epochs; minimize 0.7 global validation MAE + 0.3 upper(9-10) validation MAE, with global MAE <= ridge +0.10 and Spearman >= ridge -0.03. Fixed training settings; audits not used for reselection.',
        'trainable':'layer4.2 convolutions plus dual normalized linear head; BatchNorm parameters/statistics and earlier blocks frozen.',
        'optimizer':'AdamW; block lr=1e-5, head lr=.003, block weight_decay=1e-4; gradient norm<=1; batch=8 photos, both crops and flips averaged as in API.',
        'weights':'Inverse fitting-group size times inverse sqrt(fitting-source frequency * fitting-grade-band frequency); mean normalize and clip [.25,4]. Grade bands <=5,6,7,8,9,10.',
        'final_fit':'Reinitialize from original SCUT plus weighted ridge on all development, train exactly the selected number of epochs. No holdout gradients/labels.',
        'limitations':grouping['note'],'old_audit_reused':True,'subject_disjoint_evaluation':False}
    # This immutable plan is written before fitting or prediction on either audit.
    if (root/'finetune-protocol.json').exists():
        raise ValueError('Do not overwrite a completed or interrupted registered experiment')
    (root/'finetune-protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    ridge = fit_ridge(x[fit],y[fit],.1,sample_weights(records,groups,fit))
    reference = apply_head(x[validation],ridge)
    best_value, initial_metrics = selection_value(y[validation],reference,reference)
    history = [{'epoch':0,'selection_value':best_value,'validation':initial_metrics}]
    best_epoch = 0
    original = resnet50(weights=None,num_classes=1)
    original.load_state_dict(torch.load(MODEL_DIR/'beauty_regressor.pt',map_location='cpu',weights_only=True))
    model = TrainableVisualHead(original.layer4[2],ridge['coef'],ridge['intercept'])
    def callback(epoch, model, info):
        nonlocal best_value, best_epoch
        current = predictions(model,cache,records,validation)
        value, summary = selection_value(y[validation],current,reference)
        history.append({'epoch':epoch,'selection_value':value if np.isfinite(value) else None,'validation':summary})
        info['validation_mae'] = summary['mae']
        info['validation_upper_mae'] = summary['upper']['mae'] if summary['upper'] else None
        info['validation_spearman'] = summary['spearman']
        if value < best_value:
            best_value, best_epoch = value, epoch
        (root/'selection-progress.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')
    train_epochs(model,cache,records,fit,groups,args.epochs,args.seed+2,callback)
    selection = {'selected_epoch':best_epoch,'selection_value':best_value,'history':history,'holdouts_used_for_selection':False}
    (root/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    final_ridge = fit_ridge(x[dev],y[dev],.1,sample_weights(records,groups,dev))
    final = TrainableVisualHead(original.layer4[2],final_ridge['coef'],final_ridge['intercept'])
    if best_epoch:
        print(json.dumps({'phase':'refit_all_development','epochs':best_epoch,'photos':len(dev)}),flush=True)
        train_epochs(final,cache,records,dev,groups,best_epoch,args.seed+3)
    # All choices are now frozen. Generate audit predictions exactly once.
    current_head = PersonalHead(args.previous/'candidate-model')
    prior = current_head.predict(features.mean(axis=2),base)
    frozen = apply_head(x,final_ridge)
    updated = predictions(final,cache,records,np.arange(len(records))) if best_epoch else frozen
    def compare(indices):
        return {'previous':describe(y[indices],prior[indices]),'updated_frozen':describe(y[indices],frozen[indices]),'updated':describe(y[indices],updated[indices])} if len(indices) else None
    new_audit = np.array([i for i in test if records[i]['cohort']=='upper'],dtype=int)
    old_audit = np.array([i for i in test if records[i]['cohort']!='upper'],dtype=int)
    summary = {'model_id':protocol['model_id'],'selected_epoch':best_epoch,'usable':len(records),'development':len(dev),'test':len(test),'new_photo_test':len(new_audit),
        'new_photo_audit':compare(new_audit),'reused_old_audit':compare(old_audit),'all_audit':compare(test),
        'train':describe(y[dev],updated[dev]),'failures':source['failures'],
        'activation_recommended':bool(len(new_audit) and metrics(y[new_audit],updated[new_audit])['mae'] < metrics(y[new_audit],prior[new_audit])['mae'] and metrics(y[old_audit],updated[old_audit])['mae'] <= metrics(y[old_audit],prior[old_audit])['mae']+.10),
        'holdouts_used_for_training':False,'holdouts_used_for_selection':False,'subject_disjoint_evaluation':False,'old_audit_reused':True}
    candidate = root/'candidate-model'
    candidate.mkdir(exist_ok=True)
    for name in ('beauty_regressor.pt','face_landmarker.task'):
        shutil.copy2(MODEL_DIR/name,candidate/name)
    path = candidate/'personal-preference.npz'
    np.savez_compressed(path,coef=final.coef.detach().numpy().astype(np.float64),intercept=final.intercept.detach().numpy().astype(np.float64))
    metadata = {'schema':'facelab.personal-head.v1','model_id':protocol['model_id'],
        'weights_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'backbone_sha256':source['backbone_sha256'],
        'dataset_manifest_sha256':source['manifest_sha256'],'feature_mode':'dual','residual':False,'alpha':.1,
        'fit_images':len(dev),'test_images':len(test),'total_rated_images':len(records)+len(source['failures']),
        'identity_protocol':grouping['note'],'enabled':False,
        'validation':{'preview_only':True,'evaluation_reused':True,'original':metrics(y[test],np.clip(base[test],0,10)),
          'personal':metrics(y[test],updated[test]),'ranking':{'original':ranking_metrics(y[test],base[test]),'personal':ranking_metrics(y[test],updated[test])},
          'fresh_comparison':{'previous':describe(y[new_audit],prior[new_audit]),'updated':describe(y[new_audit],updated[new_audit])},
          'new_upper_audit':summary['new_photo_audit'],'subject_disjoint_evaluation':False},
        'training_method':'last_bottleneck_finetuning' if best_epoch else 'weighted_frozen_ridge',
        'selected_epochs':best_epoch,'heldout_ids':[records[i]['id'] for i in test]}
    if best_epoch:
        visual = candidate/'personal-visual.pt'
        torch.save(final.block.state_dict(),visual)
        metadata['visual_adapter'] = {'schema':'facelab.last-bottleneck.v1','block':'layer4.2','file':visual.name,'sha256':hashlib.sha256(visual.read_bytes()).hexdigest()}
    fingerprint = metadata['weights_sha256'] + metadata.get('visual_adapter',{}).get('sha256','')
    metadata['revision'] = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
    (candidate/'personal-preference.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    per_photo = [{**r,'split':'test' if i in test else 'development','group':str(groups[i]),'previous':round(float(prior[i]),4),'updated_frozen':round(float(frozen[i]),4),'updated':round(float(updated[i]),4)} for i,r in enumerate(records)]
    (root/'predictions.json').write_text(json.dumps(per_photo,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'evaluation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()

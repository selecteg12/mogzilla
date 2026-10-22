"""Grouped CV on development photos, one frozen holdout, no remote calls.

Fits a regularized final head on an immutable SCUT ResNet backbone. Never uses
holdout labels for fitting or hyperparameter selection. Keep groups.json under
the local experiment directory to record visually reviewed repeated subjects.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.ml.personal import apply_head, feature_matrix, fit_ridge

SEED = 20261020


def metrics(y, predictions):
    y, predictions = np.asarray(y), np.asarray(predictions)
    error = np.abs(y-predictions)
    return {'n':len(y), 'mae':round(float(error.mean()),4), 'rmse':round(float(np.sqrt(np.mean((y-predictions)**2))),4),
            'within_one':round(float(np.mean(error<=1)),4),
            'pearson':round(float(np.corrcoef(y,predictions)[0,1]),4) if np.std(predictions)>1e-10 else None}


def ranking_metrics(y, predictions):
    y, predictions = np.asarray(y), np.asarray(predictions)
    def ranks(values):
        result=np.empty(len(values))
        ordered=np.sort(values)
        for value in np.unique(values):
            result[values==value]=np.flatnonzero(ordered==value).mean()+1
        return result
    selected=np.triu(np.abs(y[:,None]-y[None,:])>=2,k=1)
    correct=np.sign(predictions[:,None]-predictions[None,:])==np.sign(y[:,None]-y[None,:])
    return {'spearman':round(float(np.corrcoef(ranks(y),ranks(predictions))[0,1]),4),
            'pairwise_at_least_two_points':round(float(correct[selected].mean()),4) if selected.any() else None,
            'pairs':int(selected.sum())}


def folds_for(groups, scores, count, seed):
    """Stratify whole groups into fixed folds; no group can cross a boundary."""
    members = defaultdict(list)
    for index,group in enumerate(groups):
        members[group].append(index)
    strata = defaultdict(list)
    for group,indices in members.items():
        strata[min(3,int(np.mean(scores[indices])-1)//3)].append(group)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(count)]
    for values in strata.values():
        rng.shuffle(values)
        for offset,group in enumerate(values):
            folds[offset % count].extend(members[group])
    return [np.array(sorted(values),dtype=int) for values in folds]


def train_candidate(x, y, base, indices, groups, alpha, residual):
    sizes = Counter(groups[indices])
    weights = np.array([1/sizes[group] for group in groups[indices]])
    target = y[indices] - base[indices] if residual else y[indices]
    head = fit_ridge(x[indices], target, alpha, weights)
    head['residual'] = residual
    return head


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--model-id',default='facelab/personal-preferences-2026-10-20')
    args=parser.parse_args()
    seed=args.seed
    root=args.experiment
    source=json.loads((root/'features.json').read_text(encoding='utf-8'))
    records=source['records']
    with np.load(root/'features.npz',allow_pickle=False) as data:
        features=data['features'].copy()
        raw=data['base_raw'].copy()
        y=data['scores'].copy()
    # n x variation x [face, wider head] x feature; flips averaged first.
    paired=np.stack((features[:,0:3].mean(axis=2),features[:,3:6].mean(axis=2)),axis=2)
    central=paired[:,1]
    base=(raw[:,1].mean(axis=1)-1)*2.5
    reviewed=json.loads((root/'groups.json').read_text(encoding='utf-8'))
    assignments=reviewed['assignments']
    groups=np.array([assignments.get(row['id'],row['id']) for row in records])
    split_path=root/'split.json'
    if split_path.exists():
        split=json.loads(split_path.read_text(encoding='utf-8'))
        if split['feature_manifest_sha256']!=hashlib.sha256((root/'features.json').read_bytes()).hexdigest():
            raise ValueError('Feature manifest changed; do not silently reuse the holdout.')
        if split['groups_sha256']!=hashlib.sha256((root/'groups.json').read_bytes()).hexdigest():
            raise ValueError('Reviewed groups changed; the fixed test protocol requires review.')
        test=np.array(split['test_indices'],dtype=int)
        dev=np.array(split['development_indices'],dtype=int)
    else:
        outer=folds_for(groups,y,5,seed)
        test=outer[0]
        dev=np.array(sorted(set(range(len(y)))-set(test)),dtype=int)
        split={'seed':seed,'development_indices':dev.tolist(),'test_indices':test.tolist(),
               'feature_manifest_sha256':hashlib.sha256((root/'features.json').read_bytes()).hexdigest(),
               'groups_sha256':hashlib.sha256((root/'groups.json').read_bytes()).hexdigest(),
               'policy':'Fixed group-stratified 80/20 split before any fitting; entire reviewed groups remain in one partition.'}
        split_path.write_text(json.dumps(split,indent=2),encoding='utf-8')
    assert not set(groups[dev]) & set(groups[test])
    cv=[dev[fold] for fold in folds_for(groups[dev],y[dev],4,seed+1)]
    matrices={mode:feature_matrix(central,mode) for mode in ('face','head','dual')}
    matrices['scalar']=base[:,None]
    candidates=[]
    for mode,x in matrices.items():
        for residual in ((False,) if mode=='scalar' else (False,True)):
            for alpha in (.001,.01,.1,1.,10.):
                predictions=np.empty(len(y))
                for validation in cv:
                    train=np.array(sorted(set(dev)-set(validation)),dtype=int)
                    assert not set(groups[train]) & set(groups[validation])
                    head=train_candidate(x,y,base,train,groups,alpha,residual)
                    predictions[validation]=apply_head(x[validation],head,base[validation])
                candidates.append({'mode':mode,'residual':residual,'alpha':alpha,'cv':metrics(y[dev],predictions[dev])})
    candidates.sort(key=lambda row:(row['cv']['mae'],row['cv']['rmse']))
    selected=candidates[0]
    x=matrices[selected['mode']]
    head=train_candidate(x,y,base,dev,groups,selected['alpha'],selected['residual'])
    predictions=apply_head(x,head,base)
    scalar=min((row for row in candidates if row['mode']=='scalar'),key=lambda row:row['cv']['mae'])
    scalar_head=train_candidate(matrices['scalar'],y,base,dev,groups,scalar['alpha'],False)
    scalar_predictions=apply_head(matrices['scalar'],scalar_head,base)
    result={'schema':'facelab.personal-evaluation.v1','created_at':datetime.now(timezone.utc).isoformat(),
            'seed':seed,'feature_count':len(y),'failed_features':source['failures'],'reviewed_groups':len(set(groups)),
            'development_count':len(dev),'test_count':len(test),'identity_protocol':reviewed['note'],
            'selected':selected,'cv_candidates':candidates,
            'test':{'original':metrics(y[test],np.clip(base[test],0,10)),
                    'mean_baseline':metrics(y[test],np.full(len(test),np.mean(y[dev]))),
                    'scalar_calibration':metrics(y[test],scalar_predictions[test]),
                    'personal':metrics(y[test],predictions[test])},
            'train':metrics(y[dev],predictions[dev]),
            'dataset_manifest_sha256':source['manifest_sha256'],'backbone_sha256':source['backbone_sha256'],
            'holdout_used_for_training':False,'refitted_on_all_photos':False,
            'limitations':['Small, selected sample rated by one person.', 'Repeated subjects were grouped by a local visual review; completeness of identity grouping is not guaranteed.',
                           'Possible overlap with original SCUT pretraining cannot be established.', 'Scores describe photos; robustness across lighting and new photos still requires more data.']}
    # A group bootstrap describes uncertainty of this small test, not universal accuracy.
    rng=np.random.default_rng(seed+2)
    test_groups=np.unique(groups[test])
    boot=[]
    for _ in range(2000):
        sample=np.concatenate([test[groups[test]==group] for group in rng.choice(test_groups,len(test_groups),replace=True)])
        boot.append(float(np.abs(y[sample]-predictions[sample]).mean()))
    result['test']['personal']['mae_group_bootstrap_95']=np.quantile(boot,[.025,.975]).round(4).tolist()
    result['test']['ranking']={'original':ranking_metrics(y[test],base[test]),'personal':ranking_metrics(y[test],predictions[test])}
    result['activation_recommended']=bool(result['test']['personal']['mae'] < result['test']['original']['mae'] and result['test']['personal']['mae'] < result['test']['mean_baseline']['mae'])
    weights_path=root/'personal-preference.npz'
    np.savez_compressed(weights_path,coef=head['coef'],intercept=np.array(head['intercept']))
    metadata={'schema':'facelab.personal-head.v1','model_id':args.model_id,
              'revision':hashlib.sha256(weights_path.read_bytes()).hexdigest()[:12],
              'feature_mode':selected['mode'],'residual':selected['residual'],'alpha':selected['alpha'],
              'weights_sha256':hashlib.sha256(weights_path.read_bytes()).hexdigest(),
              'backbone_sha256':source['backbone_sha256'],'dataset_manifest_sha256':source['manifest_sha256'],
              'fit_images':len(dev),'test_images':len(test),'total_rated_images':len(source['records'])+len(source['failures']),
              'validation':result['test'],'identity_protocol':reviewed['note'],
              'enabled':False,'heldout_ids':[records[i]['id'] for i in test]}
    (root/'personal-preference.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    per_photo=[{'id':row['id'],'path':row['path'],'score':float(y[i]),'group':str(groups[i]),'split':'test' if i in test else 'development','original':round(float(base[i]),3),'personal':round(float(predictions[i]),3)} for i,row in enumerate(records)]
    (root/'predictions.json').write_text(json.dumps(per_photo,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key not in {'cv_candidates','failed_features','limitations'}},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

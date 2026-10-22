"""Cache the frozen prefix before ResNet-50's last bottleneck, locally only."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.ml.holistic import HolisticAnalyzer, MODEL_DIR, aligned_crop
from scripts.extract_personal_features import training_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / 'maps'
    cache.mkdir(exist_ok=True)
    manifest_path = args.dataset / 'manifest.json'
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    rows = [r for r in json.loads(manifest_path.read_text(encoding='utf-8-sig'))['records'] if r['training_eligible']]
    analyzer = HolisticAnalyzer(personal=False)
    analyzer._load()
    model = analyzer.model
    import mediapipe as mp
    mean = np.array([.485, .456, .406], np.float32)
    std = np.array([.229, .224, .225], np.float32)
    records, failures, features, raw_scores = [], [], [], []
    start = time.perf_counter()
    for index, row in enumerate(rows):
        target = cache / f"{row['id']}.npy"
        try:
            if target.exists() and target.with_suffix('.json').exists():
                saved = json.loads(target.with_suffix('.json').read_text(encoding='utf-8'))
                if saved['sha256'] != row['sha256']:
                    raise ValueError('Cached photo fingerprint changed')
                records.append({**row, 'dimensions': saved['dimensions']})
                features.append(np.array(saved['features'], np.float32))
                raw_scores.append(saved['base_raw'])
                continue
            data = (args.dataset / row['path']).read_bytes()
            if hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError('Photo SHA-256 mismatch')
            rgb, _ = training_rgb(data)
            h, w = rgb.shape[:2]
            scale = min(1, 1600 / max(h, w))
            image = cv2.resize(rgb, (round(w*scale), round(h*scale))) if scale < 1 else rgb
            detection = analyzer.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)))
            if len(detection.face_landmarks) != 1:
                raise ValueError(f'Expected one face, found {len(detection.face_landmarks)}')
            points = np.array([[p.x*w, p.y*h] for p in detection.face_landmarks[0]])
            contour = points[[10,127,234,93,172,152,397,323,454,356]]
            if min(contour.max(0)-contour.min(0)) < 64 or np.linalg.norm(points[468]-points[473]) < 22:
                raise ValueError('Face too small for training')
            crops = [aligned_crop(rgb, points, margin) for margin in (.20, .40)]
            batch = np.stack([variant for crop in crops for variant in (crop, np.fliplr(crop).copy())]).astype(np.float32) / 255
            tensor = torch.from_numpy(((batch-mean)/std).transpose(0,3,1,2).copy())
            with torch.inference_mode():
                hidden = model.maxpool(model.relu(model.bn1(model.conv1(tensor))))
                for layer in (model.layer1, model.layer2, model.layer3, model.layer4[0], model.layer4[1]):
                    hidden = layer(hidden)
                pooled = model.avgpool(model.layer4[2](hidden)).flatten(1)
                values = model.fc(pooled).flatten().numpy().copy().reshape(2,2)
                pooled = pooled.numpy().copy().reshape(2,2,2048)
            if not np.isfinite(hidden.numpy()).all() or not np.isfinite(pooled).all():
                raise ValueError('Non-finite frozen features')
            np.save(target, hidden.numpy().copy().reshape(2,2,2048,7,7), allow_pickle=False)
            target.with_suffix('.json').write_text(json.dumps({'sha256':row['sha256'],'dimensions':[w,h], 'features':pooled.tolist(), 'base_raw':values.tolist()}), encoding='utf-8')
            records.append({**row,'dimensions':[w,h]})
            features.append(pooled)
            raw_scores.append(values)
        except (ValueError, OSError) as exc:
            failures.append({'id':row['id'],'path':row['path'],'score':row['score'],'error':str(exc)})
        if (index+1) % 40 == 0 or index+1 == len(rows):
            print(json.dumps({'processed':index+1,'total':len(rows),'usable':len(records),'failed':len(failures),'seconds':round(time.perf_counter()-start,1)}), flush=True)
    np.savez_compressed(args.output/'frozen-features.npz',features=np.stack(features),base_raw=np.stack(raw_scores),scores=np.array([r['score'] for r in records]))
    info = {'dataset':str(args.dataset.resolve()),'manifest_sha256':manifest_hash,
            'backbone_sha256':hashlib.sha256((MODEL_DIR/'beauty_regressor.pt').read_bytes()).hexdigest(),
            'records':records,'failures':failures,'margins':[.20,.40], 'flips':2,
            'cache_dtype':'float32','prefix_end':'layer4.1','seconds':round(time.perf_counter()-start,2)}
    (args.output/'cache.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'usable':len(records),'failures':failures,'seconds':info['seconds']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()

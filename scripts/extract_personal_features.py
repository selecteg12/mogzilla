"""Extract immutable SCUT ResNet features locally; never send photos anywhere."""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from backend.ml.holistic import HolisticAnalyzer, aligned_crop, MODEL_DIR
from backend.preprocessing.images import decode_image

MARGINS = (.12, .20, .28, .32, .40, .48)


def training_rgb(data):
    try:
        return decode_image(data), False
    except ValueError:
        # Labeler accepts AVIF and up to 40 MP. Normalize these locally, keeping
        # the original bytes/hash and ratings unchanged in the source archive.
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {'JPEG','PNG','WEBP','AVIF'} or image.width*image.height>40_000_000 or min(image.size)<32 or len(data)>20*1024**2:
                raise ValueError('Image exceeds labeler limits or has unsupported format')
            image = ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((2400,2400), Image.Resampling.LANCZOS)
            return np.asarray(image).copy(), True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    rows = [row for row in manifest['records'] if row['training_eligible']]
    analyzer = HolisticAnalyzer()
    analyzer._load()
    import mediapipe as mp

    features, base_values, metadata, failures = [], [], [], []
    if args.resume and (args.output/'features.json').is_file():
        previous = json.loads((args.output/'features.json').read_text(encoding='utf-8'))
        if previous['manifest_sha256'] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise ValueError('Dataset changed; cannot resume this feature cache.')
        metadata = previous['records']
        with np.load(args.output/'features.npz',allow_pickle=False) as saved:
            features = list(saved['features'])
            base_values = list(saved['base_raw'])
    done = {row['id'] for row in metadata}
    start = time.perf_counter()
    for index, row in enumerate(rows):
        if row['id'] in done:
            continue
        try:
            data = (args.dataset / row['path']).read_bytes()
            if hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError('SHA-256 mismatch')
            rgb, normalized = training_rgb(data)
            h, w = rgb.shape[:2]
            scale = min(1, 1600 / max(h, w))
            image = cv2.resize(rgb, (round(w*scale), round(h*scale))) if scale < 1 else rgb
            detection = analyzer.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)))
            if len(detection.face_landmarks) != 1:
                raise ValueError(f'Expected one face, found {len(detection.face_landmarks)}')
            points = np.array([[p.x*w, p.y*h] for p in detection.face_landmarks[0]])
            contour = points[[10,127,234,93,172,152,397,323,454,356]]
            low, high = contour.min(0), contour.max(0)
            if min(high-low) < 64 or np.linalg.norm(points[468]-points[473]) < 22:
                raise ValueError('Face too small for training')
            crops = [aligned_crop(rgb, points, margin) for margin in MARGINS]
            crop_features, raw = analyzer.embed_crops([variant for crop in crops for variant in (crop, np.fliplr(crop).copy())])
            features.append(crop_features.reshape(6, 2, 2048))
            base_values.append(np.array(raw).reshape(6, 2))
            thumb = Image.fromarray(crops[4])
            thumb_path = args.output / 'thumbs' / f'{row["id"]}.jpg'
            thumb_path.parent.mkdir(exist_ok=True)
            thumb.save(thumb_path, quality=88)
            pose = list(map(float, cv2.RQDecomp3x3(detection.facial_transformation_matrixes[0][:3,:3])[0]))
            metadata.append({**row, 'feature_index': len(metadata), 'pose': pose, 'dimensions': [w,h], 'training_normalized':normalized})
        except (ValueError, OSError) as error:
            failures.append({'id': row['id'], 'path': row['path'], 'score': row['score'], 'error': str(error)})
        if (index+1) % 20 == 0 or index+1 == len(rows):
            print(json.dumps({'processed': index+1, 'total': len(rows), 'usable': len(metadata), 'failed': len(failures), 'elapsed_seconds': round(time.perf_counter()-start,1)}), flush=True)

    np.savez_compressed(args.output / 'features.npz', features=np.stack(features), base_raw=np.stack(base_values), scores=np.array([row['score'] for row in metadata], dtype=np.float64))
    report = {'dataset': str(args.dataset.resolve()), 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'backbone_sha256': hashlib.sha256((MODEL_DIR/'beauty_regressor.pt').read_bytes()).hexdigest(),
              'margins': list(MARGINS), 'records': metadata, 'failures': failures, 'elapsed_seconds': round(time.perf_counter()-start,2)}
    (args.output/'features.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    # Images stay in ignored local experiment storage; no human scores on sheets.
    for offset in range(0,len(metadata),60):
        batch = metadata[offset:offset+60]
        sheet = Image.new('RGB',(1000, ((len(batch)+7)//8)*150),'#f1f1f1')
        draw = ImageDraw.Draw(sheet)
        for j,row in enumerate(batch):
            x,y = (j%8)*125,(j//8)*150
            with Image.open(args.output/'thumbs'/f'{row["id"]}.jpg') as image:
                sheet.paste(ImageOps.fit(image,(120,120)),(x,y))
            draw.text((x+3,y+123), f'{row["feature_index"]}: {row["id"][:12]}',fill='black')
        sheet.save(args.output/f'contact-{offset//60+1}.jpg',quality=90)
    print(json.dumps({'usable':len(metadata),'failures':failures,'output':str(args.output)},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()

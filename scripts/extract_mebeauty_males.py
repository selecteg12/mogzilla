"""Extract author-labeled male images from locally downloaded MEBeauty Parquets.

No network requests, identity matching, automatic preference labels or training.
Keep native bytes, official splits and source fingerprints for provenance.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

SOURCE = 'https://huggingface.co/datasets/dr-irina-lebedeva/MEBeauty-Facial-Beauty-Prediction'
PARTS = ['test-00000-of-00001.parquet', 'train-00000-of-00003.parquet',
         'train-00001-of-00003.parquet', 'train-00002-of-00003.parquet', 'validation-00000-of-00001.parquet']


def json_file(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def csv_file(path, fields, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checked_image(value):
    data=value.get('bytes') if isinstance(value,dict) else None
    if not isinstance(data,bytes) or not data:
        raise ValueError('Image bytes are not embedded in the downloaded table')
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in {'JPEG','PNG','WEBP','AVIF'}:
            raise ValueError(f'Unexpected image format: {image.format}')
        extension={'JPEG':'.jpg','PNG':'.png','WEBP':'.webp','AVIF':'.avif'}[image.format]
        width,height=image.size
        image.load()
    return data,extension,width,height


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--downloads',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pyarrow-dir',type=Path)
    args=parser.parse_args()
    if args.pyarrow_dir:
        sys.path.insert(0,str(args.pyarrow_dir.resolve()))
    import pyarrow
    import pyarrow.parquet as pq

    if args.output.exists():
        raise ValueError('Output directory already exists; do not overwrite user labels or images')
    paths=[args.downloads/name for name in PARTS]
    if not all(path.is_file() for path in paths):
        raise ValueError('All five fbp_extended Parquets are required')
    args.output.mkdir(parents=True)
    originals=args.output/'Фото'
    aligned=args.output/'Лица_256'
    originals.mkdir()
    aligned.mkdir()
    manifest, official, personal, sources = [], [], [], []
    seen_ids=set()
    genders=Counter()
    selected_splits=Counter()
    for path in paths:
        parquet=pq.ParquetFile(path)
        sources.append({'filename':path.name,'size':path.stat().st_size,
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'download_url':f'{SOURCE}/resolve/main/data/fbp_extended/{path.name}?download=true',
                        'rows':parquet.metadata.num_rows})
        for batch in parquet.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                genders[row['gender']]+=1
                if row['gender']!='male':
                    continue
                identifier=row['image_id']
                if not re.fullmatch(r'mebeauty_\d{6}',identifier) or identifier in seen_ids:
                    raise ValueError(f'Unsafe or duplicate image identifier: {identifier}')
                seen_ids.add(identifier)
                native,ext,width,height=checked_image(row['image_native'])
                crop,crop_ext,cw,ch=checked_image(row['image'])
                native_file=originals/f'{identifier}{ext}'
                crop_file=aligned/f'{identifier}{crop_ext}'
                native_file.write_bytes(native)
                crop_file.write_bytes(crop)
                digest=hashlib.sha256(native).hexdigest()
                relative=native_file.relative_to(args.output).as_posix()
                selected_splits[row['split']]+=1
                manifest.append({'id':identifier,'path':relative,'sha256':digest,'width':width,'height':height,
                                 'origin':'mebeauty_dataset','source_page':SOURCE,
                                 'usage_rights':'MEBeauty research-only; non-commercial academic facial-attractiveness research; no redistribution or biometric matching.',
                                 'user_score':None,'suggested_score':None,
                                 'source_split':row['split'],'source_cv_fold':row['cv_fold'],
                                 'aligned_path':crop_file.relative_to(args.output).as_posix(),
                                 'aligned_sha256':hashlib.sha256(crop).hexdigest(),
                                 'preference_status':'pending','training_eligible':False})
                official.append({'id':identifier,'filename':relative,'dataset_score':row['beauty_score'],
                                 'original_split':row['split'],'cv_fold':row['cv_fold'],
                                 'gender_label':'male','sha256':digest,'crop_sha256':hashlib.sha256(crop).hexdigest(),
                                 'width':width,'height':height,'crop_width':cw,'crop_height':ch})
                personal.append({'id':identifier,'filename':relative,'score':'','status':'pending','notes':''})
        print(json.dumps({'part':path.name,'processed_rows':sum(genders.values()),'male_photos':len(manifest)}),flush=True)

    manifest.sort(key=lambda row:row['id'])
    official.sort(key=lambda row:row['id'])
    personal.sort(key=lambda row:row['id'])
    json_file(args.output/'manifest.json',manifest)
    csv_file(args.output/'Ваши_оценки.csv',['id','filename','score','status','notes'],personal)
    csv_file(args.output/'Оценки_исходного_датасета.csv',list(official[0]),official)
    report={'schema':'facelab.mebeauty-extraction.v1','created_at':datetime.now(timezone.utc).isoformat(),
            'source':SOURCE,'config':'fbp_extended','filter':"gender == 'male' (provided dataset label)",
            'source_rows':sum(genders.values()),'source_gender_counts':dict(genders),
            'male_photos':len(manifest),'aligned_alternatives':len(manifest),
            'unique_original_sha256':len({row['sha256'] for row in manifest}),
            'original_split_counts':dict(selected_splits),'source_files':sources,
            'pyarrow_version':pyarrow.__version__,'personal_labels_created':0,
            'images_reencoded':False,'identity_matching_performed':False,'training_performed':False,
            'checks':['All selected original images and aligned alternatives decoded successfully.',
                      'Every output filename is generated from a validated dataset ID.',
                      'Official dataset scores are separate from empty personal labels.',
                      'Native and aligned source bytes are preserved.']}
    # Verify files on disk, not only the bytes read from Parquet.
    for row in manifest:
        assert hashlib.sha256((args.output/row['path']).read_bytes()).hexdigest()==row['sha256']
        assert hashlib.sha256((args.output/row['aligned_path']).read_bytes()).hexdigest()==row['aligned_sha256']
    json_file(args.output/'Проверка_извлечения.json',report)
    (args.output/'README.txt').write_text(
        'MEBeauty — мужские фотографии для личной разметки\n\n'
        f"Фото: {len(manifest)} оригинальных изображений. Лица_256: столько же альтернативных фрагментов лица.\n"
        'Для отбора и оценки используйте папку «Фото»: волосы и фон в ней сохранены как в исходном наборе.\n'
        'Ваши_оценки.csv — пустая таблица для личных баллов; Оценки_исходного_датасета.csv — отдельные чужие оценки.\n'
        'Оценки из датасета не присвоены вам. Обучение не запускалось.\n'
        'Все изображения отобраны по готовой отметке male, без определения пола или личности по фотографиям.\n'
        'Сохраняйте исходное разделение train/val/test при подготовке проверки; дообучение требует отдельного протокола.\n'
        'Условия MEBeauty: только некоммерческое академическое исследование привлекательности.\n'
        'Не публиковать и не загружать изображения/оценки в открытые хранилища; не использовать для биометрического сопоставления.\n'
        f'Источник: {SOURCE}\n'
        'Lebedeva, Guo, Ying (2022). MEBeauty: a multi-ethnic facial beauty dataset in-the-wild. DOI: 10.1007/s00521-021-06535-0.\n',
        encoding='utf-8-sig')
    print(json.dumps({'output':str(args.output.resolve()),'source_rows':sum(genders.values()),'male_photos':len(manifest),
                      'unique_original_sha256':report['unique_original_sha256'],'splits':dict(selected_splits)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

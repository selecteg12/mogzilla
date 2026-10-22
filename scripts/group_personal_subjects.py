"""Suggest repeated-subject pairs locally; a visual review fixes groups before training.

SFace is used only to prevent train/test overlap, never for preference scoring
or searching/naming identities. Its similarity threshold is not a guarantee.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from backend.ml.holistic import HolisticAnalyzer
from scripts.extract_personal_features import training_rgb

SFACE_SHA256 = '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args()
    root=args.experiment
    model=root/'dedup-sface.onnx'
    if hashlib.sha256(model.read_bytes()).hexdigest()!=SFACE_SHA256:
        raise ValueError('Unexpected SFace weights')
    recognizer=cv2.FaceRecognizerSF.create(str(model),'')
    cv2.setNumThreads(4)
    meta=json.loads((root/'features.json').read_text(encoding='utf-8'))
    records=meta['records']
    source=Path(meta['dataset'])
    analyzer=HolisticAnalyzer(personal=False)
    analyzer._load()
    import mediapipe as mp
    embeddings=[]
    for index,row in enumerate(records):
        rgb,_=training_rgb((source/row['path']).read_bytes())
        h,w=rgb.shape[:2]
        scale=min(1,1600/max(h,w))
        resized=cv2.resize(rgb,(round(w*scale),round(h*scale))) if scale<1 else rgb
        detection=analyzer.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(resized)))
        if len(detection.face_landmarks)!=1:
            raise ValueError(f'Non-repeatable detection for {row["id"]}')
        points=np.array([[p.x*w,p.y*h] for p in detection.face_landmarks[0]])
        contour=points[[10,127,234,93,172,152,397,323,454,356]]
        low,high=contour.min(0),contour.max(0)
        face=np.concatenate((low,high-low,points[[468,473,1,61,291]].flatten(),[1])).astype(np.float32)
        aligned=recognizer.alignCrop(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),face)
        feature=recognizer.feature(aligned).reshape(-1).copy()
        embeddings.append(feature/np.linalg.norm(feature))
        if (index+1)%50==0:print(f'Grouped features: {index+1}/{len(records)}',flush=True)
    embeddings=np.stack(embeddings)
    similarities=embeddings@embeddings.T
    pairs=[(i,j,float(similarities[i,j])) for i in range(len(records)) for j in range(i+1,len(records)) if similarities[i,j]>=.32]
    pairs.sort(key=lambda pair:-pair[2])
    np.savez_compressed(root/'dedup-features.npz',embeddings=embeddings)
    (root/'dedup-pairs.json').write_text(json.dumps({'threshold':.32,'pairs':pairs},indent=2),encoding='utf-8')
    for offset in range(0,len(pairs),40):
        batch=pairs[offset:offset+40]
        sheet=Image.new('RGB',(1200,((len(batch)+4)//5)*130),'#eeeeee')
        draw=ImageDraw.Draw(sheet)
        for cell,(i,j,similarity) in enumerate(batch):
            x,y=(cell%5)*240,(cell//5)*130
            for side,idx in enumerate((i,j)):
                with Image.open(root/'thumbs'/f'{records[idx]["id"]}.jpg') as image:
                    sheet.paste(ImageOps.fit(image,(110,108)),(x+side*115,y))
            draw.text((x+2,y+110),f'{i} / {j}    {similarity:.3f}',fill='black')
        sheet.save(root/f'subject-review-{offset//40+1}.jpg',quality=93)
    print(json.dumps({'pairs_to_review':len(pairs),'top_pairs':pairs[:8]}),flush=True)


if __name__=='__main__':
    main()

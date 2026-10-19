"""Own randomly initialized crop classifier; never loads an external face detector."""
from pathlib import Path
import os
import threading

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
REGION_SIZE = 96
THRESHOLD = 0.90


class RegionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = [3,16,24,48,64]
        for a,b in zip(channels,channels[1:]):
            layers.extend([nn.Conv2d(a,b,3,2,1,bias=False),nn.GroupNorm(8,b),nn.SiLU()])
        self.features = nn.Sequential(*layers,nn.AdaptiveAvgPool2d((3,3)))
        self.head = nn.Sequential(nn.Flatten(),nn.Linear(64*9,128),nn.SiLU(),nn.Linear(128,1))

    def forward(self,x):
        return self.head(self.features(x)).squeeze(-1)


def patch(rgb,box):
    x,y,w,h = box
    if min(w,h)<=0:
        raise ValueError("Пустая область лица.")
    matrix = np.array([[(REGION_SIZE-1)/w,0,-x*(REGION_SIZE-1)/w],
                       [0,(REGION_SIZE-1)/h,-y*(REGION_SIZE-1)/h]],np.float32)
    return cv2.warpAffine(rgb,matrix,(REGION_SIZE,REGION_SIZE),borderMode=cv2.BORDER_REFLECT_101)


def padded_box(box,factor=1.24):
    x,y,w,h = map(float,box)
    side = max(w,h)*factor
    return [x+w/2-side/2,y+h/2-side/2,side,side]


def iou(a,b):
    x=max(a[0],b[0]); y=max(a[1],b[1])
    right=min(a[0]+a[2],b[0]+b[2]); bottom=min(a[1]+a[3],b[1]+b[3])
    intersection=max(0,right-x)*max(0,bottom-y)
    return intersection/max(a[2]*a[3]+b[2]*b[3]-intersection,1e-9)


class RegionFinder:
    def __init__(self,path=None):
        self.path=Path(path) if path else Path(os.environ.get("FACEMETRICS_REGION_CHECKPOINT",str(ROOT/"training/checkpoints/regions-selected.pt")))
        self.lock=threading.Lock()
        self.model=None
        self.signature=None
        self.metadata=None

    def _load(self):
        if not self.path.is_file():
            self.model=None
            return
        signature=(self.path.stat().st_mtime_ns,self.path.stat().st_size)
        if signature==self.signature:
            return
        self.signature=signature
        self.model=None
        self.metadata=None
        ck=torch.load(self.path,map_location="cpu",weights_only=True)
        if ck.get("initialization")!="random_from_scratch" or ck.get("epochs",0)<1 or not ck.get("test"):
            raise ValueError("Классификатор областей должен быть обучен с нуля и проверен.")
        if ck.get("architecture")=="complete_face_box_v1":
            from backend.ml.face_detector import FaceBoxCNN
            if ck.get("selection")!="validation_only" or not ck.get("manifest_sha256"):
                raise ValueError("Нужна независимая проверка детектора.")
            model=FaceBoxCNN()
        else:
            model=RegionCNN()
        model.load_state_dict(ck["state_dict"])
        if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Веса детектора содержат нечисловые значения.")
        self.model=model.eval()
        self.metadata={k:v for k,v in ck.items() if k!="state_dict"}
        for key in ["train","validation_data","test_data"]:
            if isinstance(self.metadata.get(key),dict):
                self.metadata[key]={k:v for k,v in self.metadata[key].items() if k!="image_hashes"}

    def status(self):
        with self.lock:
            try:
                self._load()
            except (ValueError,RuntimeError,OSError,KeyError):
                self.model=None
            return {"ready":self.model is not None,"metadata":self.metadata,"threshold":THRESHOLD,
                    "description":"Собственная CNN ищет область лица. Новая архитектура обучается отвергать неполные лица и уточнять границы рамки. Выход не откалиброван как вероятность."}

    @torch.inference_mode()
    def score_boxes(self,rgb,boxes):
        scores=[]
        for start in range(0,len(boxes),96):
            batch=np.stack([patch(rgb,b) for b in boxes[start:start+96]])
            tensor=torch.from_numpy(batch.transpose(0,3,1,2).copy()).float()/127.5-1
            output=self.model(tensor)
            logits=output[0] if isinstance(output,tuple) else output
            scores.extend(logits.sigmoid().numpy().astype(float))
        return scores

    def score(self,rgb,box):
        with self.lock:
            self._load()
            if self.model is None:
                return None
            return float(self.score_boxes(rgb,[padded_box(box)])[0])

    def verify(self,rgb,box):
        with self.lock:
            self._load()
            if self.model is None or self.metadata.get("architecture")!="complete_face_box_v1":
                return None
            from backend.ml.face_detector import verify_box
            return verify_box(self.model,rgb,box,self.metadata.get("threshold",THRESHOLD))

    def find(self,rgb):
        with self.lock:
            self._load()
            if self.model is None:
                return {"status":"unavailable","candidates":[],"message":"Автопоиск ещё не обучен. Выделите лицо вручную."}
            h,w=rgb.shape[:2]
            if max(h,w)/min(h,w)>5:
                return {"status":"not_found","candidates":[],"message":"Для очень вытянутого кадра выделите лицо вручную."}
            if self.metadata.get("architecture")=="complete_face_box_v1":
                from backend.ml.face_detector import find_faces
                result=find_faces(self.model,rgb,self.metadata.get("threshold",THRESHOLD))
                result["training"]=self.metadata
                return result
            boxes=[]
            short=min(h,w)
            for factor in [.22,.31,.43,.59,.80,1.04]:
                side=short*factor
                nx=max(1,int(np.ceil((w-side)/max(side*.27,1)))+1)
                ny=max(1,int(np.ceil((h-side)/max(side*.27,1)))+1)
                for y in np.linspace(0,max(0,h-side),ny):
                    for x in np.linspace(0,max(0,w-side),nx):
                        boxes.append([float(x),float(y),float(side),float(side)])
            scores=self.score_boxes(rgb,boxes)
            order=np.argsort(scores)[::-1]
            chosen=[]
            for index in order:
                if scores[index]<THRESHOLD:
                    break
                candidate=boxes[index]
                if any(iou(candidate,item["window"])>.25 for item in chosen):
                    continue
                # The classifier sees the padded network crop; return the inner ROI.
                side=candidate[2]/1.24
                x=max(0,candidate[0]+candidate[2]/2-side/2)
                y=max(0,candidate[1]+candidate[3]/2-side/2)
                chosen.append({"box":[x,y,min(side,w-x),min(side,h-y)],"window":candidate,
                               "score":float(scores[index]),"score_kind":"uncalibrated_crop_classifier"})
                if len(chosen)==5:
                    break
            return {"status":"found" if chosen else "not_found","candidates":chosen,"windows_evaluated":len(boxes),
                    "threshold":THRESHOLD,"message":"Область предложена собственной CNN. Проверьте, что рамка охватывает всё лицо." if chosen else "Уверенная область лица не найдена. Выделите лицо вручную.",
                    "training":self.metadata}

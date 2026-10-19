import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from backend.geometry.topology import make_topology
from backend.preprocessing.images import crop_image, normalized_points


class LandmarkDataset(Dataset):
    def __init__(self, directory: Path, split: str, augment: bool = False):
        self.metadata = json.loads((directory/"dataset.json").read_text(encoding="utf-8"))
        self.records = [json.loads(line) for line in (directory/"manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.records = [r for r in self.records if r["split"]==split]
        self.root = Path(self.metadata["root"])
        self.topology = make_topology(self.metadata["dataset"])
        self.augment = augment
        self.cache = {}  # <= ~280 MB for 10k 96×96 RGB crops; no user uploads enter this cache.

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        record = self.records[index]
        if index not in self.cache:
            with Image.open(self.root/record["image"]) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            points = np.asarray(record["points"],np.float32)
            lo,hi = points.min(0),points.max(0)
            box = np.concatenate((lo,hi-lo)).astype(float)
            crop,transform = crop_image(rgb,box,allow_outside=True)
            target = normalized_points(points,transform).astype(np.float32)
            self.cache[index] = (crop,target,float(transform["side"]))
        base_crop,base_target,side = self.cache[index]
        crop,target = base_crop.copy(),base_target.copy()
        size = crop.shape[0]
        if self.augment:
            angle = float(np.random.uniform(-12,12))
            scale = float(np.random.uniform(0.94,1.06))
            matrix = cv2.getRotationMatrix2D(((size-1)/2,(size-1)/2),angle,scale)
            matrix[:,2] += np.random.uniform(-0.025,0.025,2)*(size-1)
            candidate = np.c_[target*(size-1),np.ones(len(target))]@matrix.T/(size-1)
            if candidate.min()>=0 and candidate.max()<=1:
                crop = cv2.warpAffine(crop,matrix,(size,size),borderMode=cv2.BORDER_REFLECT_101)
                target = candidate.astype(np.float32)
            if np.random.rand()<0.5:
                crop = np.ascontiguousarray(crop[:,::-1])
                target = target[list(self.topology.flip)].copy()
                target[:,0] = 1-target[:,0]
            crop = np.clip(crop.astype(np.float32)*np.random.uniform(0.8,1.2)+np.random.uniform(-12,12),0,255).astype(np.uint8)
            if np.random.rand()<0.15:
                crop = cv2.GaussianBlur(crop,(3,3),0.6)
        tensor = torch.from_numpy(np.ascontiguousarray(crop.transpose(2,0,1))).float()/127.5-1
        return tensor,torch.from_numpy(target),side

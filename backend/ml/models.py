"""Two compact, randomly initialized CNNs. No torchvision or downloaded parameters."""
import torch
from torch import nn
from torch.nn import functional as F


def block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin,cout,3,stride,1,bias=False),nn.GroupNorm(4,cout),nn.SiLU(),
                         nn.Conv2d(cout,cout,3,1,1,bias=False),nn.GroupNorm(4,cout),nn.SiLU())


def soft_argmax(logits: torch.Tensor) -> torch.Tensor:
    b,n,h,w = logits.shape
    probabilities = logits.flatten(2).softmax(-1).reshape(b,n,h,w)
    xs = torch.linspace(0,1,w,device=logits.device,dtype=logits.dtype)
    ys = torch.linspace(0,1,h,device=logits.device,dtype=logits.dtype)
    return torch.stack(((probabilities.sum(2)*xs).sum(-1),(probabilities.sum(3)*ys).sum(-1)),-1)


class CoordinateCNN(nn.Module):
    def __init__(self, landmarks: int):
        super().__init__()
        self.features = nn.Sequential(block(3,16,2),block(16,32,2),block(32,64,2),block(64,96,2),
                                      nn.AdaptiveAvgPool2d((4,4)))
        self.head = nn.Sequential(nn.Flatten(),nn.Linear(96*16,256),nn.SiLU(),nn.Dropout(0.1),
                                  nn.Linear(256,landmarks*2),nn.Sigmoid())
        self.landmarks = landmarks

    def forward(self, x: torch.Tensor) -> dict:
        return {"coordinates":self.head(self.features(x)).reshape(-1,self.landmarks,2)}


class HeatmapCNN(nn.Module):
    def __init__(self, landmarks: int):
        super().__init__()
        self.stem = block(3,16,2)
        self.down1 = block(16,32,2)
        self.down2 = block(32,64,2)
        self.up1 = block(96,32)
        self.up2 = block(48,32)
        self.head = nn.Conv2d(32,landmarks,1)

    def forward(self, x: torch.Tensor) -> dict:
        a = self.stem(x)
        b = self.down1(a)
        c = self.down2(b)
        d = self.up1(torch.cat((F.interpolate(c,size=b.shape[-2:],mode="bilinear",align_corners=True),b),1))
        e = self.up2(torch.cat((F.interpolate(d,size=a.shape[-2:],mode="bilinear",align_corners=True),a),1))
        logits = self.head(e)
        return {"coordinates":soft_argmax(logits),"heatmaps":logits}


def create_model(architecture: str, count: int) -> nn.Module:
    if architecture == "coordinates":
        return CoordinateCNN(count)
    if architecture == "heatmaps":
        return HeatmapCNN(count)
    raise ValueError("Architecture must be 'coordinates' or 'heatmaps'.")


def landmark_loss(output: dict, target: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(output["coordinates"],target,beta=0.02)
    if "heatmaps" in output:
        logits = output["heatmaps"]
        h,w = logits.shape[-2:]
        yy,xx = torch.meshgrid(torch.linspace(0,1,h,device=target.device),
                              torch.linspace(0,1,w,device=target.device),indexing="ij")
        dist = (xx-target[:,:,0,None,None])**2+(yy-target[:,:,1,None,None])**2
        gaussian = torch.exp(-dist/(2*(1.5/(w-1))**2)).flatten(2)
        gaussian = gaussian/gaussian.sum(-1,keepdim=True).clamp_min(1e-12)
        loss = loss + 0.1*F.kl_div(logits.flatten(2).log_softmax(-1),gaussian,reduction="batchmean")/target.shape[1]
    return loss

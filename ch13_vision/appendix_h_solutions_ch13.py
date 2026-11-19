"""Appendix H solutions for Chapter 13, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 13.10 coding exercises

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# %% [markdown]
# 1. Box IoU from scratch, verified against torchvision

# %%
# 1. Box IoU from scratch, verified against torchvision
from torchvision.ops import box_iou
def iou(a, b):                                   # boxes as [x1, y1, x2, y2]
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua                            # overlap / combined area
pairs = [([0,0,2,2],[1,0,3,2]), ([0,0,2,2],[0,0,2,2]), ([0,0,2,2],[5,5,7,7])]
mine = [round(iou(a,b),3) for a,b in pairs]
tv = [round(float(box_iou(torch.tensor([a],dtype=torch.float),
                          torch.tensor([b],dtype=torch.float))[0,0]),3) for a,b in pairs]
print(mine, tv, mine == tv)
# expected output: [0.333, 1.0, 0.0] [0.333, 1.0, 0.0] True

# %% [markdown]
# 2. Non-max suppression from scratch, verified against torchvision

# %%
# 2. Non-max suppression from scratch, verified against torchvision
from torchvision.ops import nms as tv_nms
def nms(boxes, scores, thr=0.5):
    order = list(np.argsort(scores)[::-1]); keep = []   # highest score first
    while order:
        i = order[0]; keep.append(int(i))
        order = [j for j in order[1:] if iou(boxes[i], boxes[j]) < thr]  # drop overlaps
    return keep
boxes = [[0,0,2,2],[0.1,0.1,2.1,2.1],[5,5,7,7],[5.1,5,7,7]]
scores = [0.9,0.85,0.8,0.7]
k = nms(boxes, scores); kt = sorted(tv_nms(torch.tensor(boxes), torch.tensor(scores), 0.5).tolist())
print(k, kt, k == kt)
# expected output: [0, 2] [0, 2] True

# %% [markdown]
# 3. Average precision and the mAP-style threshold sweep (simplified COCO protocol)

# %%
# 3. Average precision and the mAP-style threshold sweep (simplified COCO protocol)
def average_precision(pb, ps, gt, iou_thr=0.5):
    order = np.argsort(ps)[::-1]; matched = set()
    tp = np.zeros(len(order)); fp = np.zeros(len(order))
    for rank, idx in enumerate(order):
        best_iou, best_gt = 0, -1
        for g, gtb in enumerate(gt):
            if g in matched: continue                # each true box may be claimed only once
            v = iou(pb[idx], gtb)
            if v > best_iou: best_iou, best_gt = v, g
        if best_iou >= iou_thr:
            tp[rank] = 1; matched.add(best_gt)
        else:
            fp[rank] = 1
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c/len(gt); precision = tp_c/(tp_c+fp_c)
    r = np.concatenate([[0], recall]); p = np.concatenate([[1], precision])
    return float(np.trapezoid(p, r))             # area under the precision-recall curve
gt = [[0,0,10,10],[20,20,30,30]]                 # 2 true objects
preds = [[0,0,10,10],[20,21,30,31],[50,50,60,60]]   # 2 hits + 1 confident false alarm
ap = average_precision(preds, [0.95,0.80,0.88], gt)
aps = [average_precision(preds,[0.95,0.80,0.88],gt,t) for t in np.arange(0.5,1.0,0.05)]
print(round(ap,3), round(float(np.mean(aps)),3))
# expected output: 0.792 0.704
# the near-miss box drops from a hit once the IoU bar passes 0.818, lowering the ten-threshold mean

# %% [markdown]
# 4. Focal loss and the class-imbalance fix

# %%
# 4. Focal loss and the class-imbalance fix
logits = torch.tensor([4.0, -4.0]); targets = torch.tensor([1.0, 1.0])   # 2nd is a confident MISS
p = torch.sigmoid(logits); p_t = p*targets + (1-p)*(1-targets)
ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
factor = (1-p_t)**2                              # focal modulating factor, gamma=2
print([round(v,3) for v in p_t.tolist()], [round(v,3) for v in ce.tolist()],
      [round(v,4) for v in factor.tolist()])
# expected output: [0.982, 0.018] [0.018, 4.018] [0.0003, 0.9644]
lg = torch.full((1001,), -4.0); tg = torch.zeros(1001); tg[0] = 1.0   # 1 hard positive, 1000 easy negatives
p = torch.sigmoid(lg); pt = p*tg + (1-p)*(1-tg)
ce_b = F.binary_cross_entropy_with_logits(lg, tg, reduction="none"); foc_b = ((1-pt)**2)*ce_b
print(round(float(ce_b[0]/ce_b.sum()),3), round(float(foc_b[0]/foc_b.sum()),3))
# expected output: 0.181 0.998
# focal lifts the lone positive's share of the loss from 18% to 99.8%

# %% [markdown]
# 5. Dice, IoU, and detect-from-segment on synthetic discs (exact masks/boxes)

# %%
# 5. Dice, IoU, and detect-from-segment on synthetic discs (exact masks/boxes)
rng = np.random.default_rng(0); torch.manual_seed(0)
def dice(a,b): return 2*(a&b).sum()/(a.sum()+b.sum())
def iou_mask(a,b): return (a&b).sum()/(a|b).sum()
tA = np.zeros((8,8),bool); tA[2:6,2:6]=True; pA = np.zeros((8,8),bool); pA[3:7,2:6]=True
print(round(float(dice(tA,pA)),3), round(float(iou_mask(tA,pA)),3))
# expected output: 0.75 0.6
SZ=20
def make(n):                                     # one disc per image, with mask AND box
    X=np.zeros((n,1,SZ,SZ),"float32"); M=np.zeros((n,1,SZ,SZ),"float32"); bx=[]
    yy,xx=np.mgrid[0:SZ,0:SZ]
    for i in range(n):
        cy,cx,r=rng.integers(5,15),rng.integers(5,15),rng.integers(3,5)
        m=(xx-cx)**2+(yy-cy)**2<=r**2
        X[i,0]=m+rng.normal(0,0.3,(SZ,SZ)); M[i,0]=m
        bx.append([cx-r,cy-r,cx+r+1,cy+r+1])         # half-open, matching mask_to_box below
    return torch.tensor(X),torch.tensor(M),np.array(bx,float)
Xtr,Mtr,_=make(400); Xte,Mte,gtb=make(150)
seg=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),
                  nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,1,1))
opt=torch.optim.Adam(seg.parameters(),lr=0.01); lf=nn.BCEWithLogitsLoss()
for _ in range(150):
    opt.zero_grad(); lf(seg(Xtr),Mtr).backward(); opt.step()
with torch.no_grad():
    pr=(torch.sigmoid(seg(Xte))>0.5).numpy()[:,0]
tm=Mte.numpy()[:,0].astype(bool)
dsc=np.mean([2*(t&q).sum()/(t.sum()+q.sum()) for t,q in zip(tm,pr) if t.sum()>0])
def mask_to_box(m):                              # tight half-open box around a mask's pixels
    ys,xs=np.where(m)
    return [xs.min(),ys.min(),xs.max()+1,ys.max()+1] if len(xs) else None
ious=[iou(mask_to_box(q),gtb[i]) for i,q in enumerate(pr) if mask_to_box(q)]
print(round(float(dsc),3), round(float(np.mean(ious)),3),
      round(float(np.mean(np.array(ious)>=0.5)),3))
# expected output: 0.988 0.978 1.0

# %% [markdown]
# 6. Anchor grid generation

# %%
# 6. Anchor grid generation
def anchors_at(cx,cy,scales,ratios):
    out=[]
    for s in scales:
        for r in ratios:
            w,h=s*np.sqrt(r),s/np.sqrt(r)        # width-to-height ratio r
            out.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
    return out
scales,ratios=[32,64],[0.5,1.0,2.0]
centers=[(cx+16,cy+16) for cy in range(0,64,32) for cx in range(0,64,32)]   # 2x2 grid
allb=[b for (cx,cy) in centers for b in anchors_at(cx,cy,scales,ratios)]
print(len(centers), len(scales)*len(ratios), len(allb), [round(float(v),1) for v in allb[0]])
# expected output: 4 6 24 [4.7, -6.6, 27.3, 38.6]

# %% [markdown]
# 7. ViT patch embedding == a strided convolution, real chelsea photo

# %%
# 7. ViT patch embedding == a strided convolution, real chelsea photo
from skimage.data import chelsea
from PIL import Image
img = Image.fromarray(chelsea()).resize((224,224))
x = torch.tensor(np.array(img),dtype=torch.float).permute(2,0,1).unsqueeze(0)/255.0
P, D = 16, 32
conv = nn.Conv2d(3, D, kernel_size=P, stride=P)  # kernel=stride=16 -> patchify
tokens = conv(x)                                 # (1, D, 14, 14)
n_tok = tokens.shape[2]*tokens.shape[3]
patches = x.unfold(2,P,P).unfold(3,P,P).permute(0,2,3,1,4,5).reshape(-1, 3*P*P)
man = (patches @ conv.weight.reshape(D,-1).T + conv.bias).reshape(1,14,14,D).permute(0,3,1,2)
print(n_tok, 3*P*P, bool(torch.allclose(tokens, man, atol=1e-4)))
# expected output: 196 768 True

# %% [markdown]
# 8. Gram matrix separates style from content, real astronaut features

# %%
# 8. Gram matrix separates style from content, real astronaut features
from skimage.data import astronaut
from PIL import Image
torch.manual_seed(0); rng2 = np.random.default_rng(0)   # seeds the random filters and the shuffle
a = torch.tensor(np.array(Image.fromarray(astronaut()).resize((64,64))),
                 dtype=torch.float).permute(2,0,1).unsqueeze(0)/255.0
feat = nn.Conv2d(3,8,3,padding=1)(a).detach()[0].numpy()   # (8,64,64) feature map
def gram(Fm):
    f=Fm.reshape(Fm.shape[0],-1); return (f@f.T)/f.shape[1]  # channel correlations
C,Hh,Ww=feat.shape; perm=rng2.permutation(Hh*Ww)
shuf=feat.reshape(C,-1)[:,perm].reshape(C,Hh,Ww)            # same pixels, scrambled positions
print(round(float(np.abs(gram(feat)-gram(shuf)).mean()),4),
      round(float(np.abs(feat-shuf).mean()),4))
# expected output: 0.0 0.1551
# the Gram (style) is unchanged by the shuffle; the raw features (content) move

# %% [markdown]
# 9. CLIP zero-shot classification (illustrative: a few hundred MB download)

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
# 9. CLIP zero-shot classification (illustrative: a few hundred MB download)
from transformers import pipeline
clf = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32", device="cpu")
from skimage.data import coffee
im = Image.fromarray(coffee())
for r in clf(im, candidate_labels=["a cup of coffee","a cup of tea","a bowl of soup","a glass of beer"]):
    print(f'{r["label"]:<16} {r["score"]:.3f}')
# expected output: a cup of coffee  0.937
# expected output: a cup of tea     0.062
# expected output: a bowl of soup   0.001
# expected output: a glass of beer  0.000

# %% [markdown]
# 10. Miniproject: metric-learning verification on load_digits (triplet loss)

# %%
# 10. Miniproject: metric-learning verification on load_digits (triplet loss)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
torch.manual_seed(0); rng3=np.random.default_rng(0)
Xd,yd=load_digits(return_X_y=True); Xd=(Xd/16.0).astype("float32")
Xtr,Xte,ytr,yte=train_test_split(Xd,yd,test_size=0.3,random_state=0,stratify=yd)
Xtr_t=torch.tensor(Xtr)
embed=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,16))   # digit -> 16-d point
def sample_triplets(yv,k=256):
    a,p,ng=[],[],[]
    for _ in range(k):
        cls=rng3.integers(0,10)
        pi_,ni_=np.where(yv==cls)[0],np.where(yv!=cls)[0]
        ai,pi=rng3.choice(pi_,2,replace=True)
        a.append(ai); p.append(pi); ng.append(rng3.choice(ni_))
    return a,p,ng
opt=torch.optim.Adam(embed.parameters(),lr=0.01)
for _ in range(300):                             # pull same-digit pairs together, push others apart
    ai,pi,ni=sample_triplets(ytr)
    loss=F.triplet_margin_loss(embed(Xtr_t[ai]),embed(Xtr_t[pi]),embed(Xtr_t[ni]),margin=1.0)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    E=embed(torch.tensor(Xte)).numpy()
i,j=rng3.integers(0,len(Xte),4000),rng3.integers(0,len(Xte),4000)
same=(yte[i]==yte[j]).astype(int); dist=np.linalg.norm(E[i]-E[j],axis=1)
print(round(float(roc_auc_score(same,-dist)),3),
      round(float(dist[same==1].mean()),2), round(float(dist[same==0].mean()),2))
# expected output: 0.988 2.72 7.98
# small distance -> likely the same digit; verification without a per-digit classifier

# %% [markdown]
# 11. Anchor assignment on the chapter's 24 anchors and the box [4, 2, 36, 30]

# %%
# 11. Anchor assignment on the chapter's 24 anchors and the box [4, 2, 36, 30]
def anchors_at(cx, cy, scales, ratios):
    out = []
    for s in scales:
        for r in ratios:
            w, h = s * np.sqrt(r), s / np.sqrt(r)
            out.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    return out
def iou_ab(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua
scales, ratios = [32, 64], [0.5, 1.0, 2.0]
centers = [(cx+16, cy+16) for cy in range(0, 64, 32) for cx in range(0, 64, 32)]
boxes = [b for (cx, cy) in centers for b in anchors_at(cx, cy, scales, ratios)]
gt_box = [4, 2, 36, 30]
ious = np.array([iou_ab(b, gt_box) for b in boxes])
best = int(ious.argmax())
pos = (ious >= 0.7) | (np.arange(len(ious)) == best)
neg = ious < 0.3
ignored = ~(pos | neg)
print(int(pos.sum()), int(neg.sum()), int(ignored.sum()))
# expected output: 1 21 2

# %% [markdown]
# ## 13.10 coding drills

# %%
import numpy as np, torch, torch.nn as nn

# %% [markdown]
# Detection metrics

# %%
# Detection metrics
def encode(gt, anchor):
    xa,ya=(anchor[0]+anchor[2])/2,(anchor[1]+anchor[3])/2
    wa,ha=anchor[2]-anchor[0],anchor[3]-anchor[1]
    x,y=(gt[0]+gt[2])/2,(gt[1]+gt[3])/2; w,h=gt[2]-gt[0],gt[3]-gt[1]
    return [(x-xa)/wa,(y-ya)/ha,np.log(w/wa),np.log(h/ha)]
def decode(t, anchor):
    xa,ya=(anchor[0]+anchor[2])/2,(anchor[1]+anchor[3])/2
    wa,ha=anchor[2]-anchor[0],anchor[3]-anchor[1]
    x,y=xa+wa*t[0],ya+ha*t[1]; w,h=wa*np.exp(t[2]),ha*np.exp(t[3])
    return [x-w/2,y-h/2,x+w/2,y+h/2]
t=encode([2,6,30,28],[0,0,32,32])
print([round(float(v),4) for v in t])                        # expected output: [0.0, 0.0312, -0.1335, -0.3747]
print([round(float(v),1) for v in decode(t,[0,0,32,32])])    # expected output: [2.0, 6.0, 30.0, 28.0]
import torch.nn.functional as Fnn
sl1=Fnn.smooth_l1_loss(torch.tensor([0.40,-0.20,0.10,-0.20]), torch.tensor(t,dtype=torch.float), reduction="none")
print([round(float(v),4) for v in sl1])                      # expected output: [0.08, 0.0267, 0.0273, 0.0153]
def giou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    cx1,cy1=min(a[0],b[0]),min(a[1],b[1]); cx2,cy2=max(a[2],b[2]),max(a[3],b[3])
    c=(cx2-cx1)*(cy2-cy1)                                    # smallest enclosing box
    return inter/union - (c-union)/c
print(round(giou([0,0,2,2],[4,0,6,2]),3))                    # expected output: -0.333
print(round(0.70*(1-0.60),3))                                # expected output: 0.28
tp=np.array([1,0,1,1]); fp=np.array([0,1,0,0])
print(round(float(tp.sum()/(tp.sum()+fp.sum())),3), round(float(tp.sum()/4),3))  # expected output: 0.75 0.75
print(round((1-0.8)**2, 4))                                  # expected output: 0.04

# %% [markdown]
# Segmentation metrics

# %%
# Segmentation metrics
iou_v=0.6
print(round(2*iou_v/(1+iou_v),2))                            # expected output: 0.75
tp_ious=[0.8,0.6]; n_fp,n_fn=1,1
sq=sum(tp_ious)/len(tp_ious); rq=len(tp_ious)/(len(tp_ious)+n_fp/2+n_fn/2)
print(round(sq*rq,3), round(sq,3), round(rq,3))              # expected output: 0.467 0.7 0.667
xt=torch.randn(1,8,16,16)
print(tuple(nn.ConvTranspose2d(8,4,2,stride=2)(xt).shape))   # expected output: (1, 4, 32, 32)
print(tuple(nn.Conv2d(8,4,3,stride=2,padding=1)(xt).shape))  # expected output: (1, 4, 8, 8)

# %% [markdown]
# Anchors and boxes

# %%
# Anchors and boxes
s,r=64,2.0
print(round(s*np.sqrt(r),2), round(s/np.sqrt(r),2))          # expected output: 90.51 45.25
bx=[10,20,40,80]
print((bx[2]-bx[0])*(bx[3]-bx[1]))                           # expected output: 1800
print(((bx[0]+bx[2])/2, (bx[1]+bx[3])/2))                    # expected output: (25.0, 50.0)
m=np.zeros((6,6),bool); m[2:5,1:4]=True; ys,xs=np.where(m)
print([int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)])  # expected output: [1, 2, 4, 5]

# %% [markdown]
# Vision transformer

# %%
# Vision transformer
H,W,P=448,448,4                                              # Swin's fine 4x4 patches, on a 448x448 image
side=H//P; n=side*side
print(n)                                                     # expected output: 12544
win,nwin=7*7,(side//7)**2
print(n*n, nwin*win*win)                                     # expected output: 157351936 614656
print([(side//2**k)**2 for k in range(4)])                   # expected output: [12544, 3136, 784, 196]
npatch=(384//16)**2                                          # a 384x384 image cut into 16x16 patches
pe=np.random.default_rng(0).normal(size=(npatch,16*16*3)) @ np.random.default_rng(1).normal(0,0.02,(768,384))
print(pe.shape)                                              # expected output: (576, 384)

# %% [markdown]
# Contrastive and metric learning

# %%
# Contrastive and metric learning
u=np.array([1.0,0,1]); v=np.array([1.0,1,0])
print(round(float(u@v/(np.linalg.norm(u)*np.linalg.norm(v))),3))   # expected output: 0.5
def info_nce(z1,z2,temp=0.1):
    z1=z1/np.linalg.norm(z1,axis=1,keepdims=True); z2=z2/np.linalg.norm(z2,axis=1,keepdims=True)
    sim=z1@z2.T/temp; logp=sim-np.log(np.exp(sim).sum(axis=1,keepdims=True))
    return -np.mean(np.diag(logp))
rd=np.random.default_rng(1); bse=rd.normal(size=(8,16))
print(round(info_nce(bse,bse+rd.normal(0,0.1,(8,16))),3),
      round(info_nce(bse,rd.normal(size=(8,16))),3))         # expected output: 0.004 3.948
def triplet_loss(a,p,n,margin=1.0):
    return max(0.0, np.linalg.norm(a-p)-np.linalg.norm(a-n)+margin)
a0=np.array([0.0,0.0])
print(round(triplet_loss(a0,np.array([0.2,0]),np.array([2.0,0])),2),
      round(triplet_loss(a0,np.array([2.0,0]),np.array([0.2,0])),2))   # expected output: 0.0 2.8
z=np.array([[3.0,4.0],[0.0,5.0]])
print(np.round(z/np.linalg.norm(z,axis=1,keepdims=True),3).tolist())   # expected output: [[0.6, 0.8], [0.0, 1.0]]
s,th,m=8.0,np.deg2rad(40),0.4
print(round(s*np.cos(th+m),2), round(s*np.cos(th),2))        # expected output: 3.64 6.13

# %% [markdown]
# Style and Gram

# %%
# Style and Gram
def gram(Fm):
    f=Fm.reshape(Fm.shape[0],-1); return (f@f.T)/f.shape[1]
Fm=np.arange(2*4).reshape(2,4).astype(float)
print(gram(Fm).round(2).tolist())                            # expected output: [[3.5, 9.5], [9.5, 31.5]]

# %% [markdown]
# NeRF quadrature

# %%
# NeRF quadrature
sigma = np.array([1.0, 0.5, 3.0, 0.5])
delta = np.array([0.25, 0.25, 0.25, 0.25])
alpha = 1 - np.exp(-sigma * delta)
T = np.ones(4)
for i in range(1, 4):
    T[i] = T[i - 1] * np.exp(-sigma[i - 1] * delta[i - 1])
w = T * alpha
print([round(float(v), 4) for v in w], round(float(w.sum()), 4))
# expected output: [0.2212, 0.0915, 0.3626, 0.0381] 0.7135

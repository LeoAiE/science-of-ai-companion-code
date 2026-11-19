"""Chapter 13: Computer Vision in Practice

Walkthrough generated from the chapter slice. One cell per book code block.
"""

# %% [markdown]
# ## 13.2.1 Anchors and box encoding

# %%
import numpy as np
def anchors_at(cx, cy, scales, ratios):                 # boxes centered at (cx, cy)
    out = []
    for s in scales:                                    # object size in pixels
        for r in ratios:                                # width-to-height ratio
            w, h = s * np.sqrt(r), s / np.sqrt(r)
            out.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    return out
scales, ratios = [32, 64], [0.5, 1.0, 2.0]
centers = [(cx+16, cy+16) for cy in range(0, 64, 32) for cx in range(0, 64, 32)]  # 2x2 grid
boxes = [b for (cx, cy) in centers for b in anchors_at(cx, cy, scales, ratios)]
print("centers:", len(centers), " anchors/center:", len(scales)*len(ratios), " total:", len(boxes))
for b in boxes[:3]:
    print("  ", [round(float(v), 1) for v in b])

# %%
import numpy as np
def encode(gt, anchor):                                  # both boxes as [x1, y1, x2, y2]
    xa, ya = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
    wa, ha = anchor[2] - anchor[0], anchor[3] - anchor[1]
    x, y = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    w, h = gt[2] - gt[0], gt[3] - gt[1]
    return [(x - xa) / wa, (y - ya) / ha, np.log(w / wa), np.log(h / ha)]
def decode(t, anchor):                                   # invert: offsets + anchor -> box
    xa, ya = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
    wa, ha = anchor[2] - anchor[0], anchor[3] - anchor[1]
    x, y = xa + wa * t[0], ya + ha * t[1]
    w, h = wa * np.exp(t[2]), ha * np.exp(t[3])
    return [x - w/2, y - h/2, x + w/2, y + h/2]
anchor, gt_box = [0, 0, 32, 32], [4, 2, 36, 30]          # the square anchor and a true box near it
t = encode(gt_box, anchor)
print("targets t     :", [round(float(v), 4) for v in t])
print("decode(t) back:", [round(float(v), 1) for v in decode(t, anchor)])

# %% [markdown]
# ## 13.2.2 Intersection over union and anchor assignment

# %%
def iou(a, b):                                       # boxes as [x1, y1, x2, y2]
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1]); area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)         # overlap / combined area
print(f"{iou([0,0,2,2], [1,0,3,2]):.3f}")   # 0.333  half-overlapping
print(f"{iou([0,0,2,2], [0,0,2,2]):.3f}")   # 1.000  identical
print(f"{iou([0,0,2,2], [5,5,7,7]):.3f}")   # 0.000  disjoint

# %%
gt_ious = np.array([iou(b, gt_box) for b in boxes])         # all 24 anchors vs the one true box
best = int(gt_ious.argmax())
print(f"best anchor: #{best}, IoU = {gt_ious[best]:.3f}")   # the top-left 32x32 square, IoU 0.690
print(f"anchors with IoU >= 0.7: {int((gt_ious >= 0.7).sum())}, "
      f"IoU < 0.3: {int((gt_ious < 0.3).sum())}, in between: {int(((gt_ious >= 0.3) & (gt_ious < 0.7)).sum())}")

# %% [markdown]
# ## 13.2.3 Non-max suppression

# %%
import numpy as np
def nms(boxes, scores, thr=0.5):
    order = list(np.argsort(scores)[::-1]); keep = []   # consider highest-scoring boxes first
    while order:
        i = order[0]; keep.append(int(i))
        order = [j for j in order[1:] if iou(boxes[i], boxes[j]) < thr]   # drop overlaps with the kept box
    return keep
boxes = [[0, 0, 2, 2], [0.1, 0.1, 2.1, 2.1], [5, 5, 7, 7], [5.1, 5, 7, 7]]
scores = [0.9, 0.85, 0.8, 0.7]
print("NMS keeps indices:", nms(boxes, scores))   # [0, 2]: one box per real object, near-duplicates removed

# %% [markdown]
# ## 13.2.4 Mean average precision

# %%
import numpy as np
def average_precision(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    order = np.argsort(pred_scores)[::-1]; matched = set()
    tp = np.zeros(len(order)); fp = np.zeros(len(order))
    for rank, idx in enumerate(order):
        best_iou, best_gt = 0, -1
        for g, gt in enumerate(gt_boxes):
            if g in matched: continue                    # each true box may be claimed only once
            v = iou(pred_boxes[idx], gt)
            if v > best_iou: best_iou, best_gt = v, g
        if best_iou >= iou_thr:
            tp[rank] = 1; matched.add(best_gt)           # a correct, non-duplicate detection
        else:
            fp[rank] = 1
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / len(gt_boxes); precision = tp_c / (tp_c + fp_c)
    r = np.concatenate([[0], recall]); p = np.concatenate([[1], precision])
    return float(np.trapezoid(p, r))                     # area under the precision-recall curve
gt = [[0, 0, 10, 10], [20, 20, 30, 30]]                  # 2 true objects
preds = [[0, 0, 10, 10], [20, 21, 30, 31], [50, 50, 60, 60]]   # 2 hits + 1 false alarm
print(f"AP = {average_precision(preds, [0.95, 0.80, 0.88], gt):.3f}")   # 0.792: the confident false positive drags it below 1.0

# %%
import numpy as np
thresholds = np.arange(0.5, 1.0, 0.05)
aps = [average_precision(preds, [0.95, 0.80, 0.88], gt, t) for t in thresholds]
print("AP per threshold:", [round(a, 3) for a in aps])
print(f"mAP@[.5:.95] (simplified) = {np.mean(aps):.3f}")

# %% [markdown]
# ## 13.2.5 Focal loss and box regression

# %%
import torch
import torch.nn.functional as F
logits = torch.tensor([4.0, -4.0])        # strong positive prediction for the 1st, strong negative for the 2nd
targets = torch.tensor([1.0, 1.0])        # both are truly positive (so the 2nd is a confident MISS)
p = torch.sigmoid(logits)
p_t = p * targets + (1 - p) * (1 - targets)            # probability assigned to the TRUE class
ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
factor = (1 - p_t) ** 2                                # focal modulating factor, gamma=2
alpha = 0.25                                           # RetinaNet's class-balancing weight
alpha_t = alpha * targets + (1 - alpha) * (1 - targets)   # alpha for foreground, 1-alpha for background
focal = alpha_t * factor * ce                          # the full loss: alpha_t * (1-p_t)^gamma * CE
print("p_t (easy, hard)      :", [round(v, 3) for v in p_t.tolist()])     # [0.982, 0.018]
print("per-example CE        :", [round(v, 3) for v in ce.tolist()])      # [0.018, 4.018]
print("focal factor (1-p_t)^2:", [round(v, 4) for v in factor.tolist()])  # [0.0003, 0.9644]
print("full focal, alpha=0.25:", [round(v, 6) for v in focal.tolist()])   # [1e-06, 0.968727]
# the easy example's loss is scaled by ~0.0003 (nearly removed); the hard one keeps ~0.96: RetinaNet's fix for class imbalance

# %%
import torch, torch.nn.functional as F
from torchvision.ops import generalized_box_iou
t_true = torch.tensor([0.125, 0.0, 0.0, -0.1335])      # the worked targets from the encode step
t_pred = torch.tensor([0.250, 0.10, 0.0, -0.4000])     # a slightly-off prediction
sl1 = F.smooth_l1_loss(t_pred, t_true, reduction="none")
print("smooth L1 per coordinate:", [round(float(v), 4) for v in sl1])
print(f"smooth L1 at residual 2.0: {F.smooth_l1_loss(torch.tensor([2.0]), torch.tensor([0.0])):.2f}")
a = torch.tensor([[0., 0., 2., 2.]])
for b in ([[1., 0., 3., 2.]], [[3., 0., 5., 2.]], [[30., 0., 32., 2.]]):
    g = generalized_box_iou(a, torch.tensor(b))[0, 0]
    print(f"box {b[0]}: GIoU = {g:.3f}")

# %% [markdown]
# ## 13.2.6 Running DETR and YOLO

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
from transformers import pipeline
from skimage.data import cat
from PIL import Image
img = Image.fromarray(cat())                      # chelsea the cat, 451 wide by 300 tall; cat() == chelsea()
detector = pipeline("object-detection", model="facebook/detr-resnet-50", device="cpu")
for r in detector(img):
    b = r["box"]
    print(f'{r["label"]:8s} {r["score"]:.3f}  ({b["xmin"]},{b["ymin"]},{b["xmax"]},{b["ymax"]})')

# %%
import warnings; warnings.filterwarnings("ignore")
from ultralytics import YOLO
from skimage.data import cat
from PIL import Image
img = Image.fromarray(cat())                          # the same chelsea photo DETR just saw
model = YOLO("yolo11n.pt")                            # YOLO11 nano: a ~5 MB download on first use
res = model.predict(img, verbose=False)[0]
for box in res.boxes:
    x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0]]
    print(f"{res.names[int(box.cls)]:8s} {float(box.conf):.3f}  ({x1},{y1},{x2},{y2})")
print(f"IoU with DETR's cat box: {iou([x1, y1, x2, y2], [0, 0, 426, 299]):.3f}")

# %% [markdown]
# ## 13.2.7 Set prediction and the Hungarian match

# %%
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
cost = np.array([[0.4, 1.2], [1.5, 0.3], [0.9, 0.8], [2.0, 1.7]])
rows, cols = linear_sum_assignment(cost)
best, pick = 1e9, None
for qs in itertools.combinations(range(4), 2):
    for perm in itertools.permutations(qs):
        total = cost[perm[0], 0] + cost[perm[1], 1]
        if total < best:
            best, pick = total, perm
print("hungarian", list(zip(rows.tolist(), cols.tolist())), "cost", round(float(cost[rows, cols].sum()), 1))
print("brute", pick, "cost", round(best, 1))

# %%
import warnings; warnings.filterwarnings("ignore")
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from skimage.data import astronaut
from PIL import Image
img = Image.fromarray(astronaut())                       # a real 512-by-512 public photo
proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval()
with torch.no_grad():
    out = model(**proc(images=img, return_tensors="pt"))
probs = out.logits.softmax(-1)[0]                        # one row per query; the last column is no-object
best = probs.argmax(-1)
no_object = probs.shape[-1] - 1
print("queries", probs.shape[0], "columns", probs.shape[-1], "no-object wins", int((best == no_object).sum()))
W, H = img.size
cx, cy, w, h = (out.pred_boxes[0] * torch.tensor([W, H, W, H])).unbind(-1)   # normalized center and size, in pixels
corners = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).int()
for q in (best != no_object).nonzero().flatten().tolist():
    print(q, model.config.id2label[int(best[q])], round(float(probs[q, best[q]]), 3), tuple(corners[q].tolist()))

# %% [markdown]
# ## 13.2.8 Feature pyramids

# %%
import numpy as np
import torch
from PIL import Image
from skimage.data import chelsea
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
im = torch.tensor(np.array(Image.fromarray(chelsea()).resize((224, 224))), dtype=torch.float).permute(2, 0, 1)[None] / 255
net = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
x = net.relu(net.bn1(net.conv1(im)))
x = net.maxpool(x)
layers = {}
for name, mod in [("c2", net.layer1), ("c3", net.layer2), ("c4", net.layer3), ("c5", net.layer4)]:
    x = mod(x)
    layers[name] = x
fpn = FeaturePyramidNetwork([64, 128, 256, 512], 64).eval()
with torch.no_grad():
    pyr = fpn(layers)
print({k: tuple(v.shape) for k, v in layers.items()})
print({k: tuple(v.shape) for k, v in pyr.items()})
def level(s):
    return int(np.floor(4 + np.log2(s / 224)))
print([level(s) for s in (112, 224, 448)])

# %% [markdown]
# ## 13.2.9 Fine-tuning a detector on Penn-Fudan
# Example 13-8 fine-tunes the detector for ten epochs on 40 and on all 140 photographs outside the holdout, about 66 minutes
# (3,930 seconds in one run) on one CPU thread of the reference machine, so it lives in its own companion script and notebook in this
# folder, 07_pennfudan_full.py and 07_pennfudan_full.ipynb.

# %%
import io, os, zipfile, urllib.request
from pathlib import Path
import numpy as np
import torch, torchvision
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
torch.manual_seed(0)
torch.set_num_threads(1)
root = Path("pennfudan")
ped = root / "PennFudanPed"
if not ped.exists():
    root.mkdir(exist_ok=True)
    url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=120).read())).extractall(root)
ids = sorted(p.stem for p in (ped / "PNGImages").glob("*.png"))

def load_item(stem):
    im = Image.open(ped / "PNGImages" / f"{stem}.png").convert("RGB")
    mask = np.array(Image.open(ped / "PedMasks" / f"{stem}_mask.png"))
    boxes, labels = [], []
    for v in np.unique(mask):
        if v == 0:
            continue
        ys, xs = np.where(mask == v)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
        labels.append(1)
    image = torchvision.transforms.functional.to_tensor(im)
    target = {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64)}
    return image, target

train_ids, test_ids = ids[:-30][:40], ids[-30:]

def iou_box(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-9)

def average_precision(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0
    order = np.argsort(pred_scores)[::-1]
    matched, tp, fp = set(), np.zeros(len(order)), np.zeros(len(order))
    for rank, idx in enumerate(order):
        best_iou, best_gt = 0.0, -1
        for g, gt in enumerate(gt_boxes):
            if g in matched:
                continue
            v = iou_box(pred_boxes[idx], gt)
            if v > best_iou:
                best_iou, best_gt = v, g
        if best_iou >= iou_thr:
            tp[rank] = 1
            matched.add(best_gt)
        else:
            fp[rank] = 1
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall, precision = tp_c / len(gt_boxes), tp_c / (tp_c + fp_c)
    return float(np.trapezoid(np.concatenate([[1], precision]), np.concatenate([[0], recall])))

def person_ap(model, stems):
    model.eval()
    aps = []
    with torch.no_grad():
        for stem in stems:
            image, target = load_item(stem)
            out = model([image])[0]
            keep = (out["labels"] == 1) & (out["scores"] >= 0.5)
            aps.append(average_precision(out["boxes"][keep].numpy(), out["scores"][keep].numpy(), target["boxes"].numpy()))
    model.train()
    return float(np.mean(aps))

model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
in_f = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
print(f"in_features {in_f} fresh head AP {person_ap(model, test_ids):.3f}")
opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)
for epoch in (1, 2):
    model.train()
    totals = []
    for stem in train_ids:
        image, target = load_item(stem)
        loss_dict = model([image], [target])
        loss = sum(loss_dict.values())
        opt.zero_grad(); loss.backward(); opt.step()
        row = {k: float(v.detach()) for k, v in loss_dict.items()}
        row["total"] = float(loss.detach())
        totals.append(row)
    mean = {k: float(np.mean([r[k] for r in totals])) for k in totals[0]}
    print(f"epoch {epoch}", {k: round(v, 3) for k, v in mean.items()})
print(f"trained AP {person_ap(model, test_ids):.3f}")

# %%
coco = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")    # the COCO detector, box head untouched
print(f"untouched COCO person head AP {person_ap(coco, test_ids):.3f}")

# %% [markdown]
# ## 13.2.10 How a YOLO grid builds its targets

# %%
import numpy as np
box = np.array([17, 14, 362, 507], float)
cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
w, h = box[2] - box[0], box[3] - box[1]
S, img = 7, 512.0
col, row = int(cx / img * S), int(cy / img * S)
t = [cx / img * S - col, cy / img * S - row, w / img, h / img]
print("cell", [row, col], "targets", [round(float(v), 4) for v in t])

# %% [markdown]
# ## 13.2.11 Three box conventions

# %%
import torch
from torchvision.ops import box_convert
xyxy = torch.tensor([[17, 14, 362, 507]], dtype=torch.float)
xywh = box_convert(xyxy, "xyxy", "xywh")
cxcy = box_convert(xyxy, "xyxy", "cxcywh")
back = box_convert(xywh, "xywh", "xyxy")
print([round(float(v), 1) for v in xywh[0]])
print([round(float(v), 1) for v in cxcy[0]])
print([round(float(v), 1) for v in back[0]])

# %% [markdown]
# ## 13.2.12 Labels have to move with the pixels

# %%
W = 512
x1, y1, x2, y2 = 17, 14, 362, 507
print([W - x2, y1, W - x1, y2])

# %% [markdown]
# ## 13.2.13 DIoU and CIoU

# %%
import torch
from torchvision.ops import distance_box_iou, complete_box_iou
pairs = [([0, 0, 2, 2], [1, 0, 3, 2]), ([0, 0, 2, 2], [3, 0, 5, 2]), ([0, 0, 2, 2], [0, 0, 4, 2])]
for a, b in pairs:
    ta, tb = torch.tensor([a], dtype=torch.float), torch.tensor([b], dtype=torch.float)
    print(a, b, "DIoU", round(float(distance_box_iou(ta, tb)), 4), "CIoU", round(float(complete_box_iou(ta, tb)), 4))

# %% [markdown]
# ## 13.3.1 Dice, IoU, and a small fully convolutional network

# %%
import numpy as np
def dice(a, b): return 2 * (a & b).sum() / (a.sum() + b.sum())
def iou_mask(a, b): return (a & b).sum() / (a | b).sum()
true = np.zeros((8, 8), bool); true[2:6, 2:6] = True     # a 4x4 true region
pred = np.zeros((8, 8), bool); pred[3:7, 2:6] = True     # the prediction, shifted down one row
print(f"overlapping: Dice = {dice(true, pred):.3f}, IoU = {iou_mask(true, pred):.3f}")   # 0.750, 0.600
print(f"identical:   Dice = {dice(true, true):.3f}, IoU = {iou_mask(true, true):.3f}")   # 1.000, 1.000
v = iou_mask(true, pred)
print(f"identity check: 2*IoU/(1+IoU) = {2 * v / (1 + v):.3f}")   # 0.750, the Dice again
# Dice = 2|A and B|/(|A|+|B|), IoU = |A and B|/|A or B|; Dice >= IoU always

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
rng = np.random.default_rng(0); torch.manual_seed(0)
def dice(a, b): return 2 * (a & b).sum() / (a.sum() + b.sum())
def make_image(n, size=16):
    X = np.zeros((n, 1, size, size), "float32"); Y = np.zeros((n, 1, size, size), "float32")
    yy, xx = np.mgrid[0:size, 0:size]
    for i in range(n):
        cy, cx, r = rng.integers(4, 12), rng.integers(4, 12), rng.integers(2, 4)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        X[i, 0] = mask + rng.normal(0, 0.3, (size, size)); Y[i, 0] = mask   # noisy image, clean mask
    return torch.tensor(X), torch.tensor(Y)
Xtr, Ytr = make_image(400); Xte, Yte = make_image(200)
seg = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),     # fully convolutional: every layer keeps 16x16
                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(16, 1, 1))                           # 1x1 conv -> one logit per pixel
opt = torch.optim.Adam(seg.parameters(), lr=0.01); lf = nn.BCEWithLogitsLoss()
for _ in range(120):
    opt.zero_grad(); lf(seg(Xtr), Ytr).backward(); opt.step()
with torch.no_grad():
    pred = (torch.sigmoid(seg(Xte)) > 0.5).numpy()[:, 0]
truem = Yte.numpy()[:, 0].astype(bool)
print(f"mean Dice = {np.mean([dice(t, p) for t, p in zip(truem, pred) if t.sum() > 0]):.3f}")   # 0.981

# %% [markdown]
# ## 13.3.2 A pretrained semantic segmenter

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
from transformers import pipeline
from skimage.data import cat
from PIL import Image
import numpy as np
img = Image.fromarray(cat())
segmenter = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512", device="cpu")
result = segmenter(img)
masks = [(r["label"], (np.array(r["mask"]) > 127).mean()) for r in result]
print("segments found:", len(masks))
for label, frac in sorted(masks, key=lambda t: -t[1])[:4]:
    print(f'  {label:10s} covers {frac:5.1%} of the image')

# %%
labels = segmenter.model.config.id2label.values()
print("classes:", len(labels), " has cat:", "cat" in labels, " has animal:", "animal" in labels)
by_label = {r["label"]: np.array(r["mask"]) > 127 for r in result}   # one boolean mask per class
face = (slice(20, 290), slice(60, 400))                             # rows 20 to 289, columns 60 to 399
for label in ("towel", "animal"):
    print(f"{label:7s} covers {by_label[label][face].mean():5.1%} of the face crop")
ys, xs = np.nonzero(by_label["animal"])
print(f"animal centroid: x = {xs.mean():.0f}, y = {ys.mean():.0f}")

# %% [markdown]
# ## 13.3.3 U-Net and skip connections

# %%
import numpy as np, torch, torch.nn as nn
rng = np.random.default_rng(0); torch.manual_seed(0)
def make_hard(n, size=16, noise=0.5):                    # as Example 13-9, but the noise now rivals the signal
    X = np.zeros((n, 1, size, size), "float32"); Y = np.zeros((n, 1, size, size), "float32")
    yy, xx = np.mgrid[0:size, 0:size]
    for i in range(n):
        cy, cx, r = rng.integers(4, 12), rng.integers(4, 12), rng.integers(2, 4)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        X[i, 0] = mask + rng.normal(0, noise, (size, size)); Y[i, 0] = mask
    return torch.tensor(X), torch.tensor(Y)
Xh_tr, Yh_tr = make_hard(400); Xh_te, Yh_te = make_hard(200)
class TinyUNet(nn.Module):
    def __init__(self, skip=True):
        super().__init__()
        self.skip = skip
        self.enc = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU())   # fine detail at 16x16
        self.mid = nn.Sequential(nn.MaxPool2d(2),                            # downsample to 8x8 for context
                                 nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
        self.up = nn.ConvTranspose2d(16, 8, 2, stride=2)                     # learn the way back to 16x16
        self.dec = nn.Sequential(nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))
    def forward(self, x):
        e = self.enc(x)                                  # the features the skip will carry across
        u = self.up(self.mid(e))                         # down for context, back up for resolution
        skip = e if self.skip else torch.zeros_like(e)   # ablation switch: keep or drop the skip
        return self.dec(torch.cat([u, skip], 1))         # decoder sees context AND detail
def train_dice(model, steps=120):
    opt = torch.optim.Adam(model.parameters(), lr=0.01); lf = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad(); lf(model(Xh_tr), Yh_tr).backward(); opt.step()
    with torch.no_grad():
        pr = (torch.sigmoid(model(Xh_te)) > 0.5).numpy()[:, 0]
    tm = Yh_te.numpy()[:, 0].astype(bool)
    return np.mean([2 * (t & p).sum() / (t.sum() + p.sum()) for t, p in zip(tm, pr) if t.sum() > 0])
torch.manual_seed(0); with_skip = TinyUNet(skip=True)
torch.manual_seed(0); no_skip = TinyUNet(skip=False)     # identical weights, skip zeroed out
print(f"Dice with skip    = {train_dice(with_skip):.3f}")
print(f"Dice without skip = {train_dice(no_skip):.3f}")

# %% [markdown]
# ## 13.3.4 RoIAlign and panoptic quality

# %%
import torch
from torchvision.ops import roi_align, roi_pool
fm = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)   # feature value = 8*row + col
sub = torch.tensor([[0., 1.2, 1.2, 5.2, 5.2]])                   # a box at sub-pixel coordinates
snap = torch.tensor([[0., 1.0, 1.0, 5.0, 5.0]])                  # the same box snapped to integers
for name, box in [("sub-pixel", sub), ("snapped", snap)]:
    a = roi_align(fm, box, output_size=2, spatial_scale=1.0, sampling_ratio=2)
    p = roi_pool(fm, box, output_size=2, spatial_scale=1.0)
    print(f"{name:9s}  roi_align: {[round(float(v), 1) for v in a.flatten()]}"
          f"  roi_pool: {[round(float(v), 1) for v in p.flatten()]}")

# %%
import numpy as np
def panoptic_quality(tp_ious, n_fp, n_fn):
    tp = len(tp_ious)
    sq = np.mean(tp_ious) if tp else 0.0                 # how good the matched masks are
    rq = tp / (tp + n_fp / 2 + n_fn / 2)                 # how much of the scene was found
    return sq * rq, sq, rq
pq, sq, rq = panoptic_quality([0.8, 0.6], n_fp=1, n_fn=1)
print(f"PQ = {pq:.3f}  (SQ = {sq:.3f} x RQ = {rq:.3f})")   # 0.467 = 0.700 x 0.667

# %% [markdown]
# ## 13.3.5 Segment Anything

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
from transformers import pipeline
from skimage.data import cat
from PIL import Image
img = Image.fromarray(cat()).resize((256, 256))          # downscale so CPU inference is bearable
sam = pipeline("mask-generation", model="facebook/sam-vit-base", device="cpu")   # ~375 MB, slow on CPU
out = sam(img, points_per_crop=16, points_per_batch=64)  # probe a 16x16 grid; points_per_side is ignored
print(len(out["masks"]), "candidate masks")              # 6 candidate object masks

# %% [markdown]
# ## 13.3.6 Transposed convolution

# %%
import torch, torch.nn as nn
x = torch.randn(1, 8, 16, 16)                              # a 16x16 feature map, 8 channels
up = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)     # stride 2 doubles the spatial size
print(tuple(up(x).shape))                                  # (1, 4, 32, 32)
print(tuple(nn.Conv2d(8, 4, 3, stride=2, padding=1)(x).shape))   # (1, 4, 8, 8): a strided conv halves it

# %% [markdown]
# ## 13.3.7 Mean IoU, and the pixel-accuracy trap

# %%
import os
import numpy as np
from PIL import Image
root = "pennfudan/PennFudanPed/PedMasks"
name = sorted(os.listdir(root))[0]
person = np.array(Image.open(os.path.join(root, name))) > 0
pred = np.zeros_like(person)
print(name, "person fraction", round(float(person.mean()), 3))
print("majority accuracy", round(float((pred == person).mean()), 3))
print("person IoU", 0.0)

# %% [markdown]
# ## 13.3.8 Mask R-CNN on one photograph

# %%
import torch
from pathlib import Path
from PIL import Image
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
im = Image.open(Path("pennfudan/PennFudanPed/PNGImages/FudanPed00001.png")).convert("RGB")
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
ten = torchvision.transforms.functional.to_tensor(im)
with torch.no_grad():
    pred = model([ten])[0]
keep = (pred["labels"] == 1) & (pred["scores"] >= 0.5)
print("person instances", int(keep.sum()), [round(float(s), 3) for s in pred["scores"][keep]])

# %% [markdown]
# ## 13.3.9 Dice loss

# %%
import torch
prob = torch.tensor([0.2, 0.8, 0.1, 0.9, 0.05])
gt = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
bce = torch.nn.functional.binary_cross_entropy(prob, gt)
eps = 1e-6
dice = 1 - (2 * (prob * gt).sum() + eps) / (prob.sum() + gt.sum() + eps)
print(round(float(bce), 4), round(float(dice), 4))

# %%
import numpy as np, torch
from PIL import Image
person = np.array(Image.open("pennfudan/PennFudanPed/PedMasks/FudanPed00001_mask.png")) > 0
g = torch.tensor(person, dtype=torch.float32).flatten()
p = torch.full_like(g, 0.1)                                   # a person probability of 0.1 on every pixel
bce = torch.nn.functional.binary_cross_entropy(p, g, reduction="none")
eps = 1e-6
dice = 1 - (2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps)
print(round(float(bce.mean()), 3), round(float(bce[g == 1].sum() / bce.sum()), 3), round(float(dice), 3))

# %% [markdown]
# ## 13.3.10 Atrous spatial pyramid pooling

# %%
import torch
import torch.nn as nn
class TinyASPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Conv2d(8, 4, 3, padding=1, dilation=1)
        self.b2 = nn.Conv2d(8, 4, 3, padding=2, dilation=2)
        self.b3 = nn.Conv2d(8, 4, 3, padding=4, dilation=4)
        self.out = nn.Conv2d(12, 4, 1)
    def forward(self, x):
        return self.out(torch.cat([self.b1(x), self.b2(x), self.b3(x)], 1))
print(tuple(TinyASPP()(torch.zeros(1, 8, 16, 16)).shape))

# %%
import torch
from PIL import Image
from skimage.data import chelsea
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
model = deeplabv3_mobilenet_v3_large(weights=weights).eval()
x = weights.transforms()(Image.fromarray(chelsea()))[None]     # resize so the short side is 520, then normalize
with torch.no_grad():
    labels = model(x)["out"].argmax(1)[0]                         # one class id per output pixel
names = weights.meta["categories"]
print("label map", tuple(labels.shape), "classes", len(names))
for c in labels.unique().tolist():
    print(f"{names[c]:14s} {float((labels == c).float().mean()):.2f}")

# %% [markdown]
# ## 13.4 Vision transformers

# %%
C, H, W, P = 3, 224, 224, 16
n_patches = (H // P) * (W // P); token_dim = C * P * P
print(f"{H}x{W}x{C} image, {P}x{P} patches -> {n_patches} tokens of dim {token_dim}")   # 196 tokens, dim 768
print(f"plus 1 [CLS] token = {n_patches + 1} tokens fed to the transformer")            # 197

# %%
import numpy as np
rng = np.random.default_rng(0)
patches = rng.normal(size=(196, 768))                    # 196 flattened 16x16x3 patches
E = rng.normal(0, 0.02, (768, 384))                      # a learned patch-embedding matrix
patch_embeddings = patches @ E                           # project each patch to the model dimension
pos = rng.normal(0, 0.02, (196, 384))                    # learned position embeddings
seq = patch_embeddings + pos                             # the sequence the transformer ingests
print("patch embeddings", patches.shape, "@", E.shape, "->", patch_embeddings.shape)   # (196,768)@(768,384)->(196,384)
print("after adding position embeddings:", seq.shape)    # (196, 384)

# %%
n = 56 * 56                                   # a 224x224 image at Swin's 4x4 patches: 3,136 tokens
win = 7 * 7; nwin = (56 // 7) ** 2            # 7x7 local windows: 64 windows of 49 tokens
print(f"global attention pairs  : {n * n:>9,}")
print(f"windowed attention pairs: {nwin * win * win:>9,}  ({n * n // (nwin * win * win)}x fewer)")
print("tokens per Swin stage:", [56 * 56, 28 * 28, 14 * 14, 7 * 7])   # a CNN-style pyramid

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
from transformers import pipeline
from skimage.data import chelsea
from PIL import Image
clf = pipeline("image-classification", model="google/vit-base-patch16-224", device="cpu")
for r in clf(Image.fromarray(chelsea()), top_k=3):       # the chelsea photo, 451 wide by 300 tall, resized to 224x224 inside the pipeline
    print(f"{r['label']:<20} {r['score']:.3f}")

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from transformers import AutoImageProcessor, ViTForImageClassification
from skimage.data import chelsea
from PIL import Image
name = "google/vit-base-patch16-224"
proc = AutoImageProcessor.from_pretrained(name)
vit = ViTForImageClassification.from_pretrained(name, attn_implementation="eager")
px = proc(images=Image.fromarray(chelsea()), return_tensors="pt")
with torch.no_grad():
    atts = vit(**px, output_attentions=True).attentions    # 12 layers of (1, 12, 197, 197)
rollout = torch.eye(197)
for A in atts:                                             # walk the layers input to output
    A = A[0].mean(0)                                       # average the 12 heads
    A = 0.5 * A + 0.5 * torch.eye(197)                     # mix in the residual connection
    rollout = (A / A.sum(-1, keepdim=True)) @ rollout      # rows already sum to 1; renormalizing is a numerical guard
grid = rollout[0, 1:].reshape(14, 14).numpy()              # the CLS token's weight on each patch
heat = np.kron(grid, np.ones((16, 16)))                    # upsample each patch to its 16x16 pixels
top5 = [divmod(int(k), 14) for k in np.argsort(grid.ravel())[::-1][:5]]
print("top five patches (row, col):", top5)
r, c = np.unravel_index(grid.argmax(), grid.shape)
print("rollout grid:", grid.shape, " upsampled heatmap:", heat.shape)
print(f"top-attended patch: row {r}, column {c} of the 14x14 grid "
      f"(pixels y {r*16}-{r*16+16}, x {c*16}-{c*16+16} of the 224x224 input)")

# %% [markdown]
# ## 13.5 Self-supervised and contrastive learning

# %%
import numpy as np
rng = np.random.default_rng(0)
def info_nce(z1, z2, temp=0.1):                          # z1[i] and z2[i] are a positive pair
    z1 = z1 / np.linalg.norm(z1, axis=1, keepdims=True)
    z2 = z2 / np.linalg.norm(z2, axis=1, keepdims=True)
    sim = z1 @ z2.T / temp                               # cosine similarity, divided by temperature
    logp = sim - np.log(np.exp(sim).sum(axis=1, keepdims=True))   # row i should pick column i
    return -np.mean(np.diag(logp))
base = rng.normal(size=(8, 16))
pos = base + rng.normal(0, 0.1, (8, 16))
neg = rng.normal(size=(8, 16))
print(f"aligned pairs = {info_nce(base, pos):.3f}")      # 0.007  matched pairs are close
print(f"random pairs  = {info_nce(base, neg):.3f}")      # 4.731  unmatched pairs (lower = better)
for temp in (0.05, 0.5, 1.0):
    print(f"temp {temp:<4} aligned {info_nce(base, pos, temp):.3f}  random {info_nce(base, neg, temp):.3f}")
S = np.array([[0.9, 0.1], [0.2, 0.8]])                   # a 2-by-2 similarity matrix, worked by hand
for temp in (0.1, 1.0):
    logits = S / temp
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    print(f"hand temp {temp}: p {np.round(p, 4).tolist()} loss {-np.mean(np.log(np.diag(p))):.4f}")

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
from transformers import pipeline
from skimage.data import astronaut
from PIL import Image
clf = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32", device="cpu")
img = Image.fromarray(astronaut())                       # a real photo of an astronaut
for r in clf(img, candidate_labels=["a person", "a spaceship", "a cat", "a landscape"]):
    print(f"{r['label']:<14} {r['score']:.3f}")

# %%
from transformers import CLIPModel
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
scale = float(clip.logit_scale.exp())
print(f"logit scale {scale:.1f}, temperature {1/scale:.2f}")

# %% [markdown]
# Example 13-16 pretrains a small encoder with SimCLR on the 50,000 CIFAR-10 training images and probes it, about 58 minutes
# (3,458 seconds in one run) on six threads of the reference machine, so it lives in its own companion script and notebook in this
# folder, 08_simclr_cifar10.py and 08_simclr_cifar10.ipynb.

# %% [markdown]
# ## 13.6 Metric learning and face recognition

# %%
import numpy as np
def contrastive(y, d, m=1.0):
    return y * d ** 2 + (1 - y) * max(0.0, m - d) ** 2
print(f"match at 0.2:     {contrastive(1, 0.2):.2f}")
print(f"non-match at 0.3: {contrastive(0, 0.3):.2f}")
print(f"non-match at 1.5: {contrastive(0, 1.5):.2f}")

# %%
import numpy as np
def triplet_loss(a, p, n, margin=1.0):                   # loss = max(0, d(a,p) - d(a,n) + margin)
    return max(0.0, np.linalg.norm(a - p) - np.linalg.norm(a - n) + margin)
a = np.array([0.0, 0.0])
print(f"good triplet (pos near, neg far): loss = {triplet_loss(a, np.array([0.1, 0]), np.array([3.0, 0])):.2f}")   # 0.00
print(f"bad triplet  (pos far, neg near): loss = {triplet_loss(a, np.array([3.0, 0]), np.array([0.1, 0])):.2f}")   # 3.90 (penalized)

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
torch.manual_seed(0); rng = np.random.default_rng(0)
X, y = load_digits(return_X_y=True); X = (X / 16.0).astype("float32")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Xtr_t = torch.tensor(Xtr)
embed = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16))   # maps a digit to a 16-d point
def sample_triplets(yv, k=256):
    a, p, ng = [], [], []
    for _ in range(k):
        cls = rng.integers(0, 10)
        pos_idx, neg_idx = np.where(yv == cls)[0], np.where(yv != cls)[0]
        ai, pi = rng.choice(pos_idx, 2, replace=True)
        a.append(ai); p.append(pi); ng.append(rng.choice(neg_idx))
    return a, p, ng
opt = torch.optim.Adam(embed.parameters(), lr=0.01)
for _ in range(300):                                     # pull same-digit pairs together, push others apart
    ai, pi, ni = sample_triplets(ytr)
    loss = F.triplet_margin_loss(embed(Xtr_t[ai]), embed(Xtr_t[pi]), embed(Xtr_t[ni]), margin=1.0)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    E = embed(torch.tensor(Xte)).numpy()
i, j = rng.integers(0, len(Xte), 4000), rng.integers(0, len(Xte), 4000)
same = (yte[i] == yte[j]).astype(int); dist = np.linalg.norm(E[i] - E[j], axis=1)
print(f"verification AUC = {roc_auc_score(same, -dist):.3f}")   # 0.988  small distance -> likely the same digit
print(f"mean distance: same = {dist[same==1].mean():.2f}, different = {dist[same==0].mean():.2f}")   # same-digit pairs ~2.72, different ~7.98

# %%
ytr_t = torch.tensor(ytr)
with torch.no_grad():                                     # how many random triplets still teach anything?
    ai, pi, ni = sample_triplets(ytr, k=2000)
    za, zp, zn = embed(Xtr_t[ai]), embed(Xtr_t[pi]), embed(Xtr_t[ni])
    active = ((za - zp).norm(dim=1) - (za - zn).norm(dim=1) + 1.0 > 0).float().mean()
print(f"random triplets still active after training: {active:.1%}")
torch.manual_seed(1)
hard = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16))
opt = torch.optim.Adam(hard.parameters(), lr=0.01)
g = torch.Generator().manual_seed(1)
for step in range(300):
    idx = torch.randperm(len(Xtr_t), generator=g)[:128]   # a random batch of 128 digits
    Z = hard(Xtr_t[idx]); lab = ytr_t[idx]
    D = torch.cdist(Z, Z)                                 # every pairwise distance in the batch at once
    samecls = lab[:, None] == lab[None, :]
    pos = D.masked_fill(~samecls, -1).max(1).values           # hardest positive: farthest same class
    neg = D.masked_fill(samecls, float("inf")).min(1).values  # hardest negative: nearest other class
    batch_loss = F.relu(pos - neg + 1.0)                  # triplet loss on the mined triplets
    opt.zero_grad(); batch_loss.mean().backward(); opt.step()
with torch.no_grad():
    E2 = hard(torch.tensor(Xte)).numpy()
d2 = np.linalg.norm(E2[i] - E2[j], axis=1)
print(f"batch-hard verification AUC = {roc_auc_score(same, -d2):.3f}  "
      f"(random triplets gave {roc_auc_score(same, -dist):.3f})")
print(f"mined triplets active in the final batch: {(batch_loss > 0).float().mean():.1%}")

# %% [markdown]
# Example 13-19 trains the face embedding of 13.6 three times on the Olivetti faces on six threads, 3 minutes (180 seconds in one run)
# on the reference machine, so it lives outside this single-thread walkthrough, in its own companion script and notebook in this folder,
# 10_arcface_olivetti.py and 10_arcface_olivetti.ipynb. In the book it follows the worked ArcFace and CosFace numbers of the next cell.

# %%
import numpy as np
th_y, th_o = np.deg2rad(50), np.deg2rad(55)     # true class at 50 degrees, nearest rival at 55
s = 10.0                                        # the scale on the cosine logits
def ce_loss(l_true, l_other):                   # two-class softmax cross-entropy
    return -np.log(np.exp(l_true) / (np.exp(l_true) + np.exp(l_other)))
print(f"plain softmax  : loss = {ce_loss(s * np.cos(th_y), s * np.cos(th_o)):.3f}")
print(f"ArcFace m=0.5  : loss = {ce_loss(s * np.cos(th_y + 0.5), s * np.cos(th_o)):.3f}")
print(f"CosFace m=0.35 : loss = {ce_loss(s * (np.cos(th_y) - 0.35), s * np.cos(th_o)):.3f}")

# %%
# 13.6.1 retrieval, continuing from Example 13-17's E and yte
Z = E / np.linalg.norm(E, axis=1, keepdims=True)         # L2 normalize: cosine similarity becomes a dot product
S = Z @ Z.T                                              # every test digit scored against every other one
np.fill_diagonal(S, -np.inf)                             # a query may not retrieve itself
ranked = np.argsort(-S, axis=1)                          # gallery indices, most similar first
print("test digits:", len(Z), " query 0 is a", yte[0], " its 5 nearest are", yte[ranked[0, :5]].tolist())
for k in (1, 5, 10):
    p_at_k = (yte[ranked[:, :k]] == yte[:, None]).mean() # fraction of the k nearest that share the query's digit
    print(f"precision at {k:2d} = {p_at_k:.3f}")

# %%
# Example 13-20: photo-to-photo and text-to-photo retrieval with CLIP
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from transformers import CLIPModel, CLIPProcessor
from skimage import data
from PIL import Image
name = "openai/clip-vit-base-patch32"
model, proc = CLIPModel.from_pretrained(name), CLIPProcessor.from_pretrained(name)
names = ["astronaut", "chelsea", "coffee", "rocket", "camera", "coins"]
imgs = [Image.fromarray(getattr(data, n)()).convert("RGB") for n in names]   # camera and coins are grayscale
queries = ["a photo of a cat", "a rocket launch", "a man with a camera"]
with torch.no_grad():                                     # pooler_output holds the projected embeddings
    g = model.get_image_features(**proc(images=imgs, return_tensors="pt")).pooler_output
    t = model.get_text_features(**proc(text=queries, return_tensors="pt", padding=True)).pooler_output
g = g / g.norm(dim=-1, keepdim=True)                      # L2 normalize: cosine similarity is a dot product
t = t / t.norm(dim=-1, keepdim=True)
S = (g @ g.T).numpy(); np.fill_diagonal(S, -np.inf)       # photo against photo, never itself
for n, row in zip(names, S):
    print(f"{n:9s} nearest other photo: {names[row.argmax()]:9s} cosine {row.max():.3f}")
for q, row in zip(queries, (t @ g.T).numpy()):            # each text query against the same gallery
    print(f"{q!r:24s} -> {names[row.argmax()]:9s} cosine {row.max():.3f}")

# %% [markdown]
# ## 13.7 Other practical computer vision

# %%
import warnings; warnings.filterwarnings("ignore")
import torch
from skimage.data import page
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
name = "microsoft/trocr-small-printed"                     # a small line-level recognition model
proc = TrOCRProcessor.from_pretrained(name)
model = VisionEncoderDecoderModel.from_pretrained(name)
img = page()                                               # a real scanned page from skimage, 384 wide by 191 tall
def read_line(arr):                                        # recognition: one cropped text line in, text out
    px = proc(images=Image.fromarray(arr).convert("RGB"), return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = model.generate(px, max_length=48)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
print(read_line(img[0:30, :]))                             # the page's headline
print(read_line(img[45:70, :]))                            # one body line

# %%
import warnings; warnings.filterwarnings("ignore")
import torch
from skimage.data import astronaut
from PIL import Image
from transformers import AutoProcessor, VitPoseForPoseEstimation
name = "usyd-community/vitpose-base-simple"                # a ViT backbone with a keypoint head
proc = AutoProcessor.from_pretrained(name)
model = VitPoseForPoseEstimation.from_pretrained(name)
img = Image.fromarray(astronaut())
box = [[17, 14, 345, 493]]                                 # DETR's person box, as [x, y, width, height]
inputs = proc(img, boxes=[box], return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
pose = proc.post_process_pose_estimation(out, boxes=[box])[0][0]
print(len(pose["keypoints"]), "keypoints returned; those the model trusts (score > 0.5):")
for kp, lab, sc in zip(pose["keypoints"], pose["labels"], pose["scores"]):
    if sc > 0.5:
        print(f'{model.config.id2label[int(lab)]:11s} ({kp[0]:4.0f},{kp[1]:4.0f})  score {sc:.2f}')

# %%
import warnings; warnings.filterwarnings("ignore")
import torch
from skimage.data import astronaut, cat, coffee
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(name)   # ViT encoder + GPT-2 decoder
proc = ViTImageProcessor.from_pretrained(name)
tok = AutoTokenizer.from_pretrained(name)

def caption(arr):
    px = proc(images=Image.fromarray(arr), return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = model.generate(px, max_length=16, num_beams=4)   # beam search: keep the 4 best partial sentences (15.5 develops it)
    return tok.decode(ids[0], skip_special_tokens=True).strip()

for label, arr in [("astronaut", astronaut()), ("cat", cat()), ("coffee", coffee())]:
    print(f"{label:9s} -> {caption(arr)}")

# %%
# Example 13-24: tracking by detection with and without motion prediction, scored by MOTA
import numpy as np
from scipy.optimize import linear_sum_assignment
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
# two people cross: A walks right, B walks left, 3 pixels a frame (hand-placed boxes, not a video)
gt = [{"A": [3*t, 0, 3*t + 10, 20], "B": [30 - 3*t, 2, 40 - 3*t, 22]} for t in range(10)]
dets = [list(f.values()) for f in gt]                     # a perfect detector, except in two frames
dets[5] = [gt[5]["A"]]                                    # frame 5, where they meet: B is missed
dets[8] = dets[8][::-1] + [[60, 40, 70, 60]]              # frame 8: order swapped, plus one false box
def track(dets, use_velocity, min_iou=0.3):
    tracks, history, next_id = {}, [], 1                  # track id -> [last box, velocity]
    for frame in dets:
        ids = list(tracks)
        pred = [np.add(tracks[k][0], tracks[k][1] if use_velocity else 0) for k in ids]
        cost = np.array([[1 - iou(p, d) for d in frame] for p in pred]).reshape(len(ids), len(frame))
        shown, used = {}, set()
        for r, c in zip(*linear_sum_assignment(cost)):    # the Hungarian match on 1 - IoU
            if 1 - cost[r, c] >= min_iou:                 # reject a pair that barely overlaps
                k = ids[r]
                tracks[k] = [frame[c], np.subtract(frame[c], tracks[k][0])]
                shown[k] = frame[c]; used.add(c)
        for k, p in zip(ids, pred):                       # an unmatched track coasts on its prediction
            if k not in shown:
                tracks[k][0] = p
        for c, d in enumerate(frame):                     # an unmatched detection opens a new track
            if c not in used:
                tracks[next_id] = [d, np.zeros(4)]; shown[next_id] = d; next_id += 1
        history.append(shown)
    return history
def mota(gt, history, thr=0.5):
    fn = fp = idsw = n_gt = 0; last = {}
    for g, out in zip(gt, history):
        gk, ok = list(g), list(out); n_gt += len(gk)
        cost = np.array([[1 - iou(g[a], out[b]) for b in ok] for a in gk])
        pairs = [(gk[r], ok[c]) for r, c in zip(*linear_sum_assignment(cost)) if 1 - cost[r, c] >= thr]
        fn += len(gk) - len(pairs); fp += len(ok) - len(pairs)
        for person, tid in pairs:                         # a person whose track id changed is a switch
            idsw += person in last and last[person] != tid
            last[person] = tid
    return fn, fp, idsw, n_gt, 1 - (fn + fp + idsw) / n_gt
for v in (True, False):
    fn, fp, sw, n, m = mota(gt, track(dets, use_velocity=v))
    print(f"velocity {'on ' if v else 'off'}: FN {fn}  FP {fp}  IDSW {sw}  GT {n}  MOTA {m:.3f}")

# %% [markdown]
# ## 13.8 Generative vision preview

# %%
import numpy as np
rng = np.random.default_rng(0)
def gram(F):                                             # channel-by-channel correlations of (C, H, W)
    f = F.reshape(F.shape[0], -1)
    return (f @ f.T) / f.shape[1]
C, H, W = 8, 16, 16
base = rng.normal(size=(1, H, W))
A = rng.normal(size=(C, H, W)); A[:4] += 2 * base        # texture A: channels 0-3 correlated
B = rng.normal(size=(C, H, W)); B[4:] += 2 * base        # texture B: a different correlation structure
perm = rng.permutation(H * W)
A_shuf = A.reshape(C, -1)[:, perm].reshape(C, H, W)      # same pixels, scrambled positions
print(f"Gram (style) change      = {np.abs(gram(A) - gram(A_shuf)).mean():.4f}")   # 0.0000  style is position-independent
print(f"feature (content) change = {np.abs(A - A_shuf).mean():.4f}")               # 1.8266  content moved
print(f"Gram distance A vs B = {np.abs(gram(A) - gram(B)).mean():.3f}")            # 2.144  different textures
print(f"Gram distance A vs A = {np.abs(gram(A) - gram(A)).mean():.3f}")            # 0.000

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from skimage.data import cat, coffee
from torchvision.models import vgg19, VGG19_Weights

torch.manual_seed(0)
torch.set_num_threads(1)                                # pin the CPU result; the last digits move with the thread count
vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()   # the pretrained conv stack only
for p in vgg.parameters():
    p.requires_grad_(False)                              # the network is frozen; only the image will move
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) # VGG's ImageNet normalization
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def load(arr, size=128):                                 # photo -> normalized (1, 3, size, size) tensor
    x = torch.tensor(np.array(Image.fromarray(arr).resize((size, size))), dtype=torch.float)
    return ((x.permute(2, 0, 1) / 255.0 - mean) / std).unsqueeze(0)
content_img = load(cat())                                # the layout we keep
style_img = load(coffee())                               # the texture we borrow
STYLE_LAYERS = [0, 5, 10, 19, 28]                        # conv1_1 .. conv5_1: textures at five scales
CONTENT_LAYER = 21                                       # conv4_2: layout, objects, geometry
def features(x):                                         # collect the needed activations in one pass
    feats = {}
    for i, layer in enumerate(vgg[:29]):
        x = layer(x)
        if i in STYLE_LAYERS or i == CONTENT_LAYER:
            feats[i] = x.clone()                        # conv output; the next ReLU is inplace and would overwrite a bare reference
    return feats
def gram_t(F_):                                          # the Gram matrix, normalized by size
    b, c, h, w = F_.shape
    f = F_.reshape(c, h * w)
    return (f @ f.T) / (c * h * w)
with torch.no_grad():                                    # targets: content features and style Grams
    f_content = features(content_img)[CONTENT_LAYER]
    grams_style = {i: gram_t(f) for i, f in features(style_img).items() if i in STYLE_LAYERS}
x = content_img.clone().requires_grad_(True)             # start from the content image
opt = torch.optim.Adam([x], lr=0.05)
alpha, beta = 1.0, 1e6                                   # style needs the big weight: Grams are tiny numbers
for step in range(1, 201):
    feats = features(x)
    c_loss = F.mse_loss(feats[CONTENT_LAYER], f_content)
    s_loss = sum(F.mse_loss(gram_t(feats[i]), grams_style[i]) for i in STYLE_LAYERS)
    loss = alpha * c_loss + beta * s_loss
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (1, 50, 100, 200):
        print(f"step {step:3d}: content = {c_loss.item():7.3f}, style = {s_loss.item():.6f}, total = {loss.item():8.1f}")
out = (x.detach()[0] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
Image.fromarray((out * 255).astype("uint8")).resize((256, 256)).save("style_transfer_result.png")
print("saved style_transfer_result.png")

# %%
x2 = load(cat()).clone().requires_grad_(True)
opt2 = torch.optim.Adam([x2], lr=0.05)
LAYER = 19                                               # conv4_1: mid-level patterns (eyes, fur, swirls)
def act_strength(x):
    h = x
    for i, layer in enumerate(vgg[:LAYER + 1]):
        h = layer(h)
    return (h ** 2).mean()                               # how strongly the layer is firing
for step in range(1, 31):
    a = act_strength(x2)
    opt2.zero_grad(); (-a).backward(); opt2.step()       # ascend by descending the negative
    if step in (1, 10, 20, 30):
        print(f"step {step:2d}: mean squared activation = {a.item():.1f}")
out2 = (x2.detach()[0] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
Image.fromarray((out2 * 255).astype("uint8")).resize((256, 256)).save("deepdream_result.png")
print("saved deepdream_result.png")

# %%
import numpy as np
sigma = np.array([0.2, 6.0, 0.4, 0.4])          # density along one ray
delta = np.array([0.5, 0.5, 0.5, 0.5])          # spacing of the four samples
color = np.array([[0.9, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.9], [0.8, 0.8, 0.2]])
alpha = 1 - np.exp(-sigma * delta)              # opacity of each sample
T = np.ones(4)
for i in range(1, 4):
    T[i] = T[i - 1] * np.exp(-sigma[i - 1] * delta[i - 1])
w = T * alpha                                    # front-to-back compositing weights
C = (w[:, None] * color).sum(0)
print("weights:", [round(float(v), 4) for v in w], "sum", round(float(w.sum()), 4))
print("color:", [round(float(v), 4) for v in C])

# %% [markdown]
# Example 13-28 trains a small neural radiance field on the NeRF authors' Lego data, about 22 minutes (1,346 seconds in one run)
# on six threads of the reference machine, so it lives in its own companion script and notebook in this folder,
# 09_nerf_lego.py and 09_nerf_lego.ipynb.

# %% [markdown]
# ## 13.9 Hands-on lab: an end-to-end detect-and-segment pipeline

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
rng = np.random.default_rng(0); torch.manual_seed(0)
SZ = 20
def make(n):                                             # one disc per image, with mask AND ground-truth box
    X = np.zeros((n, 1, SZ, SZ), "float32"); M = np.zeros((n, 1, SZ, SZ), "float32"); boxes = []
    yy, xx = np.mgrid[0:SZ, 0:SZ]
    for i in range(n):
        cy, cx, r = rng.integers(5, 15), rng.integers(5, 15), rng.integers(3, 5)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        X[i, 0] = mask + rng.normal(0, 0.3, (SZ, SZ)); M[i, 0] = mask
        boxes.append([cx - r, cy - r, cx + r + 1, cy + r + 1])   # half-open box, the same convention mask_to_box uses
    return torch.tensor(X), torch.tensor(M), np.array(boxes, float)
Xtr, Mtr, _ = make(400); Xte, Mte, gt_boxes = make(150)
seg = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),   # Example 13-9's segmenter, unchanged
                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1))
opt = torch.optim.Adam(seg.parameters(), lr=0.01); lf = nn.BCEWithLogitsLoss()
for _ in range(150):
    opt.zero_grad(); lf(seg(Xtr), Mtr).backward(); opt.step()
with torch.no_grad():
    pred = (torch.sigmoid(seg(Xte)) > 0.5).numpy()[:, 0]
truem = Mte.numpy()[:, 0].astype(bool)
dice = np.mean([2*(t & p).sum() / (t.sum()+p.sum()) for t, p in zip(truem, pred) if t.sum() > 0])
print(f"segmentation: mean Dice = {dice:.3f}")            # 0.988
def mask_to_box(m):                                       # the new piece: a mask's tight bounding box
    ys, xs = np.where(m)
    return [xs.min(), ys.min(), xs.max()+1, ys.max()+1] if len(xs) else None
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
ious = [iou(mask_to_box(p), gt_boxes[i]) for i, p in enumerate(pred) if mask_to_box(p)]
print(f"detection (boxes from masks): mean IoU = {np.mean(ious):.3f}, "
      f"detect rate @IoU>=0.5 = {np.mean(np.array(ious) >= 0.5):.3f}")   # IoU 0.978, rate 1.000
ious_perfect = [iou(mask_to_box(t), gt_boxes[i]) for i, t in enumerate(truem)]
print(f"ceiling check (perfect masks): mean IoU = {np.mean(ious_perfect):.3f}")
mask_ious = [(t & p).sum() / (t | p).sum() for t, p in zip(truem, pred)]
print(f"mean mask IoU = {np.mean(mask_ious):.3f}")
print(f"imperfect box: {sum(v < 1 for v in ious)} of {len(ious)}; "
      f"imperfect mask: {sum(v < 1 for v in mask_ious)} of {len(mask_ious)}")
widths = gt_boxes[:, 2] - gt_boxes[:, 0]
print("test discs by radius:", {int(r): int((widths == 2 * r + 1).sum()) for r in (3, 4)})
closed = gt_boxes.copy(); closed[:, 2:] -= 1          # [cx-r, cy-r, cx+r, cy+r], one pixel short
ceil = [iou(mask_to_box(t), closed[i]) for i, t in enumerate(truem)]
print(f"mixed-convention ceiling: mean IoU = {np.mean(ceil):.3f}")

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from skimage.data import astronaut
from PIL import Image
from transformers import pipeline

img = Image.fromarray(astronaut())                       # a real 512x512 public photo
det = pipeline("object-detection", model="facebook/detr-resnet-50", device="cpu")
seg = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512", device="cpu")
dets = det(img)                                          # boxes above the pipeline's default 0.5 score; DETR uses no NMS
segs = seg(img)                                          # one semantic mask per class present
print(f"DETR found {len(dets)} objects, SegFormer found {len(segs)} segments")
for label, frac in sorted(((s["label"], (np.array(s["mask"]) > 0).mean()) for s in segs), key=lambda t: -t[1]):
    print(f"  {label:10s} {frac:.3f}")

H, W = np.array(img).shape[:2]
labelmap = np.full((H, W), "-", dtype=object)            # per-pixel segment label
for s in segs:
    labelmap[np.array(s["mask"]) > 0] = s["label"]

for d in sorted(dets, key=lambda d: -d["score"]):        # combine detection with segmentation
    b = d["box"]
    crop = labelmap[b["ymin"]:b["ymax"], b["xmin"]:b["xmax"]].ravel()
    vals, counts = np.unique(crop, return_counts=True)
    print(f'{d["label"]:12s} {d["score"]:.3f} '
          f'box=({b["xmin"]},{b["ymin"]},{b["xmax"]},{b["ymax"]}) '
          f'-> segment inside box: {vals[counts.argmax()]}')

# %%
# 13.9 experiment: the combine step on the chelsea photo, continuing from Example 13-30's det and seg
from skimage.data import cat
img2 = Image.fromarray(cat())                            # the chelsea photo of 13.2
dets2, segs2 = det(img2), seg(img2)
labelmap2 = np.full(np.array(img2).shape[:2], "-", dtype=object)
for s in segs2:
    labelmap2[np.array(s["mask"]) > 0] = s["label"]
for d in sorted(dets2, key=lambda d: -d["score"]):
    b = d["box"]
    vals, counts = np.unique(labelmap2[b["ymin"]:b["ymax"], b["xmin"]:b["xmax"]], return_counts=True)
    top = sorted(zip(vals, counts / counts.sum()), key=lambda t: -t[1])[:3]   # the three largest shares
    print(f'{d["label"]:7s} {d["score"]:.3f} box=({b["xmin"]},{b["ymin"]},{b["xmax"]},{b["ymax"]}) ->',
          ", ".join(f"{v} {share:.3f}" for v, share in top))

# %% [markdown]
# ## Coding exercises

# %%
import torch
from torchvision.ops import box_iou
def iou(a, b):
    """Intersection over union for two xyxy boxes. The body is yours."""
    ...
a = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.], [0., 0., 2., 2.]])
b = torch.tensor([[1., 0., 3., 2.], [0., 0., 2., 2.], [5., 5., 7., 7.]])
print("library IoU", [round(float(v), 3) for v in box_iou(a, b).diag()])

# %%
import torch
from torchvision.ops import nms
def nms_from_scratch(boxes, scores, thr=0.5):
    ...
boxes = torch.tensor([[0, 0, 2, 2], [0.1, 0.1, 2.1, 2.1], [5, 5, 7, 7], [5.1, 5, 7, 7]], dtype=torch.float)
scores = torch.tensor([0.9, 0.85, 0.8, 0.7])
print("library NMS", nms(boxes, scores, 0.5).tolist())

# %%
import numpy as np
def average_precision(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    ...
preds = [[0, 0, 10, 10], [20, 21, 30, 31], [50, 50, 60, 60]]
print("predictions", len(preds), "thresholds", len(np.arange(0.5, 1.0, 0.05)))

# %%
import torch
logits = torch.tensor([4.0, -4.0])
print("starter logits", [round(float(v), 1) for v in torch.sigmoid(logits)])

# %%
import numpy as np
def dice(a, b):
    ...
true = np.zeros((8, 8), bool)
true[2:6, 2:6] = True
print("true pixels", int(true.sum()))

# %%
scales, ratios = [32, 64], [0.5, 1.0, 2.0]
centers = [(cx + 16, cy + 16) for cy in range(0, 64, 32) for cx in range(0, 64, 32)]
def anchors_at(cx, cy, scales, ratios):
    ...
print("centers", len(centers), "anchors per center", len(scales) * len(ratios))

# %%
import torch
from torch import nn
from skimage.data import chelsea
from PIL import Image
img = Image.fromarray(chelsea()).resize((224, 224))
layer = nn.Conv2d(3, 8, kernel_size=16, stride=16)
print("photo", img.size, "conv weight", tuple(layer.weight.shape))

# %%
import numpy as np
from skimage.data import astronaut
photo = astronaut()
def gram(F):
    ...
print("astronaut", photo.shape)

# %%
labels = ["a cup of coffee", "a cup of tea", "a bowl of soup", "a glass of beer"]
print("candidates", len(labels), labels[0])

# %%
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
print("digits", X.shape, "classes", len(set(y)))

# %%
import numpy as np
def assign_anchors(anchors, gt_box, hi=0.7, lo=0.3):
    """Return counts of positive, negative, and ignored anchors."""
    ...
print("thresholds", 0.7, 0.3)

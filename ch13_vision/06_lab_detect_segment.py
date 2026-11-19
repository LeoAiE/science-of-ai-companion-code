# %% [markdown]
# # Lab: end-to-end detect-and-segment (synthetic disc + real DETR+SegFormer)
# Regenerated from Book.md; run top to bottom (HF models download once).

# %%
# Example 13-29: stage one, the Example 13-9 segmenter on a 20-pixel canvas, plus mask_to_box
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
# convention check: perfect masks against half-open gt boxes score IoU 1.0; against
# closed-interval boxes [cx-r, cy-r, cx+r, cy+r] they would top out at 0.762
ious_perfect = [iou(mask_to_box(t), gt_boxes[i]) for i, t in enumerate(truem)]
print(f"ceiling check (perfect masks): mean IoU = {np.mean(ious_perfect):.3f}")
mask_ious = [(t & p).sum() / (t | p).sum() for t, p in zip(truem, pred)]
print(f"mean mask IoU = {np.mean(mask_ious):.3f}")
print(f"imperfect box: {sum(v < 1 for v in ious)} of {len(ious)}; "
      f"imperfect mask: {sum(v < 1 for v in mask_ious)} of {len(mask_ious)}")
widths = gt_boxes[:, 2] - gt_boxes[:, 0]
print("test discs by radius:", {int(r): int((widths == 2 * r + 1).sum()) for r in (3, 4)})
closed = gt_boxes.copy(); closed[:, 2:] -= 1
ceil = [iou(mask_to_box(t), closed[i]) for i, t in enumerate(truem)]
print(f"mixed-convention ceiling: mean IoU = {np.mean(ceil):.3f}")

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
# Example 13-30: stage two, pretrained DETR + SegFormer on a real photo
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
# Experiment: the combine step on the chelsea photo, continuing from Example 13-30's det and seg
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

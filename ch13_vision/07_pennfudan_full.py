# %% [markdown]
# # Fine-tuning Faster R-CNN on all of Penn-Fudan (Example 13-8 of Section 13.2.9)
# A standalone companion script: it fine-tunes the COCO Faster R-CNN of 13.2.9 twice, ten epochs each, on one CPU
# thread, so it lives outside the chapter walkthrough. Run it from this folder; it reads the Penn-Fudan photographs
# from ./pennfudan and downloads them there if they are missing.

# %% [markdown]
# The helpers of the 13.2.9 listing, repeated so this script runs on its own: the data loader, the box IoU, the
# greedy average precision, and the person AP on the 30-image holdout.

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

test_ids = ids[-30:]

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

# %% [markdown]
# Example 13-8: the COCO person head as it ships, then the 13.2.9 recipe for ten epochs on the
# 40-image subset and on all 140 images outside the holdout, with the holdout AP after every epoch.

# %%
def fine_tune(train_stems, epochs=10):
    torch.manual_seed(0)                                  # the seed of the 13.2.9 listing, so its run is epochs 1 and 2 here
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)   # a fresh background-plus-person head
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)
    for epoch in range(1, epochs + 1):
        model.train()
        totals = []
        for stem in train_stems:                          # one image per step, in file order, as in 13.2.9
            image, target = load_item(stem)
            loss = sum(model([image], [target]).values())
            opt.zero_grad(); loss.backward(); opt.step()
            totals.append(float(loss.detach()))
        ap = person_ap(model, test_ids)                   # the 30 holdout photographs, never trained on
        print(f"{len(train_stems):3d} images  epoch {epoch:2d}  mean loss {np.mean(totals):.3f}  holdout AP {ap:.3f}", flush=True)

coco = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")   # the checkpoint as it ships: COCO label 1 is person
print(f"COCO person head, no fine-tuning: holdout AP {person_ap(coco, test_ids):.3f}", flush=True)
fine_tune(ids[:-30][:40])                                 # the 40-image subset of 13.2.9, now for ten epochs
fine_tune(ids[:-30])                                      # all 140 photographs outside the holdout

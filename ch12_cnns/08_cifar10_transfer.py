# %% [markdown]
# # Transfer learning on 1,000 CIFAR-10 photographs (Example 12-16 of Section 12.7.1)
# A standalone companion script: it trains seven ResNet-18 runs one after another on the CPU, about 47
# minutes in all (2,822 seconds in one run) on the reference machine, so it lives outside the chapter walkthrough.
# Run it from this folder; torchvision reads CIFAR-10 from ./data.

# %%
import torch
torch.set_num_threads(1)                      # one intra-op thread, so the printed accuracies reproduce
import torch.nn as nn
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights
torch.manual_seed(0)
weights = ResNet18_Weights.IMAGENET1K_V1
tfm = weights.transforms()                    # resize 256, crop 224, ImageNet mean and std
train_ds = datasets.CIFAR10("data", train=True, download=True)
test_ds = datasets.CIFAR10("data", train=False, download=True)
g = torch.Generator().manual_seed(0)

def subset(ds, per_class, gen):               # the class-balanced draw of 12.7.1
    by = {c: [] for c in range(10)}
    order = torch.randperm(len(ds), generator=gen).tolist()
    for i in order:
        c = ds[i][1]
        if len(by[c]) < per_class:
            by[c].append(i)
        if all(len(v) >= per_class for v in by.values()):
            break
    idx = [i for c in range(10) for i in by[c]]
    xs, ys = [], []
    for i in idx:
        img, lab = ds[i]
        xs.append(tfm(img))
        ys.append(lab)
    return torch.stack(xs), torch.tensor(ys)

Xtr, ytr = subset(train_ds, 100, g)           # 100 photographs per class: 1,000 training images
Xte, yte = subset(test_ds, 100, g)            # 100 per class from the test split: 1,000 test images
print(tuple(Xtr.shape), tuple(Xte.shape), flush=True)

def run(trains, lr, bn, epochs=3, batch=32):
    torch.manual_seed(0)
    model = resnet18(weights=None if trains == "scratch" else weights)
    model.fc = nn.Linear(512, 10)
    if trains in ("head", "layer4+head"):     # freeze everything, then unfreeze the parts that train
        for p in model.parameters():
            p.requires_grad = False
        parts = [model.fc] if trains == "head" else [model.layer4, model.fc]
        for part in parts:
            for p in part.parameters():
                p.requires_grad = True
    bn0 = model.bn1.running_mean.clone()      # the stem's batch-norm running means before training
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lf = nn.CrossEntropyLoss()
    for _epoch in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch):
            ii = perm[i:i + batch]
            model.train()                     # batch statistics, and the running statistics are updated
            if bn == "kept":                  # or: every BatchNorm2d in eval mode, ImageNet's statistics kept
                for mod in model.modules():
                    if isinstance(mod, nn.BatchNorm2d):
                        mod.eval()
            opt.zero_grad()
            lf(model(Xtr[ii]), ytr[ii]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(Xte[i:i + 100]).argmax(1) for i in range(0, len(Xte), 100)])
    acc = (pred == yte).float().mean().item()
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    moved = (model.bn1.running_mean - bn0).abs().max().item()
    return acc, ntr, moved

runs = [("scratch", 1e-2, "updated"), ("head", 1e-2, "kept"),
        ("all", 1e-3, "updated"), ("all", 1e-3, "kept"),
        ("all", 1e-4, "updated"), ("all", 1e-4, "kept"),
        ("layer4+head", 1e-4, "kept")]
print("trains       lr      BN stats  test acc  trainable  stem BN mean moved", flush=True)
for trains, lr, bn in runs:
    acc, ntr, moved = run(trains, lr, bn)
    print(f"{trains:12} {lr:<7g} {bn:8}  {acc:8.3f}  {ntr:9,}  {moved:.3f}", flush=True)

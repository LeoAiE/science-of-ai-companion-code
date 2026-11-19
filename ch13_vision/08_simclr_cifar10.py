# %% [markdown]
# # SimCLR on CIFAR-10, judged by a linear probe (Example 13-16 of Section 13.5)
# A standalone companion script: it pretrains a small convolutional encoder with the NT-Xent loss on the 50,000
# CIFAR-10 training images without their labels, trains the same encoder with labels for comparison, and scores
# all three encoders (untrained, SimCLR, supervised) with a logistic-regression probe. It takes tens of minutes on
# the CPU, so it lives outside the chapter walkthrough. Run it from this folder; it reads the CIFAR-10 copy that
# Chapter 12 downloaded into ../ch12_cnns/data and downloads it there if it is missing.

# %%
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets
from torchvision.ops import roi_align
torch.set_num_threads(6)                                 # six threads, one per core; the printed numbers come from this setting
root = "../ch12_cnns/data"                               # the CIFAR-10 copy of 12.7.1
train = datasets.CIFAR10(root, train=True, download=True)
test = datasets.CIFAR10(root, train=False, download=True)
Xtr = torch.tensor(train.data).permute(0, 3, 1, 2).float() / 255   # (50000, 3, 32, 32) in [0, 1]
Xte = torch.tensor(test.data).permute(0, 3, 1, 2).float() / 255
ytr, yte = np.array(train.targets), np.array(test.targets)
print("train", tuple(Xtr.shape), "test", tuple(Xte.shape))

def gray(x):                                             # gray level: a weighted sum of red, green, and blue
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

def augment(x, s=0.5):                                   # one random view of every image in the batch
    B = x.shape[0]
    area = torch.empty(B).uniform_(0.08, 1.0)            # crop 8 to 100 percent of the image area
    logr = torch.empty(B).uniform_(math.log(3 / 4), math.log(4 / 3))   # aspect ratio 3/4 to 4/3
    w = (area * logr.exp()).sqrt().clamp(max=1.0) * 32
    h = (area / logr.exp()).sqrt().clamp(max=1.0) * 32
    x0, y0 = torch.rand(B) * (32 - w), torch.rand(B) * (32 - h)
    boxes = torch.stack([torch.arange(B, dtype=torch.float), x0, y0, x0 + w, y0 + h], 1)
    v = roi_align(x, boxes, output_size=32, sampling_ratio=2, aligned=True)   # crop and resize back to 32x32
    flip = torch.rand(B) < 0.5
    v = torch.where(flip[:, None, None, None], v.flip(-1), v)
    b, c, sat = (torch.empty(B, 1, 1, 1).uniform_(1 - 0.8 * s, 1 + 0.8 * s) for _ in range(3))
    j = (v * b).clamp(0, 1)                              # brightness
    m = gray(j).mean((2, 3), keepdim=True)
    j = ((j - m) * c + m).clamp(0, 1)                    # contrast around the mean gray level
    g = gray(j)
    j = ((j - g) * sat + g).clamp(0, 1)                  # saturation around each pixel's own gray
    v = torch.where((torch.rand(B) < 0.8)[:, None, None, None], j, v)          # color jitter with probability 0.8
    drop = (torch.rand(B) < 0.2)[:, None, None, None]                          # grayscale with probability 0.2
    return torch.where(drop, gray(v).expand(-1, 3, -1, -1), v)

def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())

def make_encoder():                                      # f: a 32x32 photo -> a 256-number representation h
    return nn.Sequential(block(3, 32), nn.MaxPool2d(2), block(32, 64), nn.MaxPool2d(2),
                         block(64, 128), nn.MaxPool2d(2), block(128, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten())

def nt_xent(z, tau=0.5):                                 # rows k and k + N are the two views of image k
    z = F.normalize(z, dim=1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))                    # a view is never a candidate for itself
    n = z.shape[0] // 2
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(sim, target)

def probe(enc):                                          # linear evaluation: frozen features, logistic regression
    enc.eval()
    with torch.no_grad():
        ftr = torch.cat([enc(Xtr[i:i + 1000]) for i in range(0, len(Xtr), 1000)])
        fte = torch.cat([enc(Xte[i:i + 1000]) for i in range(0, len(Xte), 1000)])
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6              # standardize with the training statistics only
    ftr, fte = (ftr - mu) / sd, (fte - mu) / sd
    torch.manual_seed(0)
    lin = nn.Linear(256, 10)                             # the probe: one linear layer, softmax over the 10 classes
    opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=500, history_size=20, line_search_fn="strong_wolfe")
    yt = torch.tensor(ytr)
    def closure():                                       # full-batch cross-entropy plus a small L2 penalty
        opt.zero_grad()
        loss = F.cross_entropy(lin(ftr), yt) + 1e-4 * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return (lin(fte).argmax(1).numpy() == yte).mean()

# %% [markdown]
# Example 13-16: the untrained encoder's probe, then SimCLR pretraining without labels, then the same encoder
# trained with labels on the same views for the same number of epochs.

# %%
EPOCHS, N = 20, 256                                      # 195 steps of 256 images (512 views) per epoch
torch.manual_seed(0)
f0 = make_encoder()
print(f"untrained encoder   probe accuracy {probe(f0):.3f}", flush=True)

torch.manual_seed(0)
f = make_encoder()                                       # the same starting weights as f0
g = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128))   # projection head, thrown away afterward
opt = torch.optim.Adam(list(f.parameters()) + list(g.parameters()), lr=1e-3)
gen = torch.Generator().manual_seed(0)
for epoch in range(1, EPOCHS + 1):
    f.train(); g.train()
    perm = torch.randperm(len(Xtr), generator=gen)
    losses, hits = [], []
    for k in range(0, len(Xtr) - N + 1, N):
        xb = Xtr[perm[k:k + N]]                          # images only: the labels are never read here
        z = g(f(torch.cat([augment(xb), augment(xb)])))
        loss = nt_xent(z)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                            # does each view pick its partner out of the 511 others?
            zn = F.normalize(z, dim=1); s = zn @ zn.T; s.fill_diagonal_(-2)
            hits.append((s.argmax(1) == torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)])).float().mean().item())
        losses.append(loss.item())
    if epoch in (1, 2, 5, 10, 15, 20):
        print(f"SimCLR epoch {epoch:2d}  NT-Xent {np.mean(losses):.3f}  partner found {np.mean(hits):.3f}", flush=True)
print(f"SimCLR encoder      probe accuracy {probe(f):.3f}", flush=True)

torch.manual_seed(0)
fs = make_encoder()                                      # the same starting weights again, now trained with labels
head = nn.Linear(256, 10)
opt = torch.optim.Adam(list(fs.parameters()) + list(head.parameters()), lr=1e-3)
gen = torch.Generator().manual_seed(0)
yt = torch.tensor(ytr)
for epoch in range(1, EPOCHS + 1):
    fs.train()
    perm = torch.randperm(len(Xtr), generator=gen)
    for k in range(0, len(Xtr) - N + 1, N):
        idx = perm[k:k + N]
        loss = F.cross_entropy(head(fs(augment(Xtr[idx]))), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
print(f"supervised encoder  probe accuracy {probe(fs):.3f}", flush=True)

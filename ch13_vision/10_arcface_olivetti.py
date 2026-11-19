# %% [markdown]
# # Softmax, CosFace, and ArcFace on real faces (Example 13-19 of Section 13.6)
# A standalone companion script: it trains one small face embedding three times on 30 of the 40 people in the
# Olivetti faces (scikit-learn's fetch_olivetti_faces, AT&T Laboratories Cambridge) and scores face verification on
# the 10 people it never saw. It runs on six threads, so it lives outside the single-thread chapter walkthrough.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import roc_auc_score
torch.set_num_threads(6)                                 # six threads, one per core; the printed numbers come from this setting
import os, sys
_out, _err = sys.stdout, sys.stderr
sys.stdout = sys.stderr = open(os.devnull, "w")
try:
    faces = fetch_olivetti_faces()                       # 400 photos, 10 each of 40 people, 64x64 gray in [0, 1]
finally:
    sys.stdout.close(); sys.stdout, sys.stderr = _out, _err
X, y = torch.tensor(faces.images[:, None]), torch.tensor(faces.target)
seen = y < 30                                            # people 0 to 29 train; people 30 to 39 are never seen
Xs, ys, Xu, yu = X[seen], y[seen], X[~seen], y[~seen].numpy()
iu, ju = np.triu_indices(len(Xu), 1)                     # every pair of the 100 unseen photos
same = (yu[iu] == yu[ju]).astype(int)
print("train", tuple(Xs.shape), "unseen", tuple(Xu.shape), "pairs", len(same), "same person", int(same.sum()))

def verify(E):                                           # cosine similarity of every pair, scored by ROC AUC
    E = F.normalize(torch.as_tensor(E), dim=1).numpy()
    cos = (E[iu] * E[ju]).sum(1)
    return roc_auc_score(same, cos), cos[same == 1].mean(), cos[same == 0].mean()

print("raw pixels      AUC %.3f  mean cos same %.3f  different %.3f" % verify(Xu.reshape(len(Xu), -1)))

def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))

def make_net():                                          # a 64x64 face -> a 64-number embedding
    return nn.Sequential(block(1, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 64))

def jitter(x, g):                                        # shift up to 4 pixels each way, mirror half the faces
    pad = F.pad(x, (4, 4, 4, 4), mode="replicate")
    dx, dy = torch.randint(0, 9, (2,), generator=g).tolist()
    out = pad[:, :, dy:dy + 64, dx:dx + 64]
    return torch.where(torch.rand(len(x), generator=g)[:, None, None, None] < 0.5, out.flip(-1), out)

def train(kind, m=0.0, s=64.0, epochs=60):
    torch.manual_seed(0)                                 # the same starting weights for every loss
    net = make_net()
    Wc, b = nn.Parameter(torch.randn(30, 64) * 0.01), nn.Parameter(torch.zeros(30))   # one row per training person
    opt = torch.optim.Adam(list(net.parameters()) + [Wc, b], lr=1e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xs), generator=g)
        for k in range(0, len(Xs), 60):                  # five batches of 60 faces per epoch
            idx = perm[k:k + 60]
            e, lab = net(jitter(Xs[idx], g)), ys[idx]
            if kind == "softmax":
                logits = e @ Wc.T + b                    # the raw softmax classifier
            else:
                cos = F.normalize(e, dim=1) @ F.normalize(Wc, dim=1).T   # cos(theta_j) for every person j
                true = F.one_hot(lab, 30).bool()
                if kind == "arcface":                    # s cos(theta_y + m)
                    theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
                    cos = torch.where(true, torch.cos(theta + m), cos)
                else:                                    # CosFace: s (cos(theta_y) - m)
                    cos = torch.where(true, cos - m, cos)
                logits = s * cos
            loss = F.cross_entropy(logits, lab)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        auc, cs, cd = verify(net(Xu))                    # the margin is gone at test time: plain cosine only
    print(f"{kind:8s} m={m:<4}  AUC {auc:.3f}  mean cos same {cs:.3f}  different {cd:.3f}  last batch loss {loss.item():.3f}", flush=True)

train("softmax")
train("cosface", m=0.35)
train("arcface", m=0.5)

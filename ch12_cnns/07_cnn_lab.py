# %% [markdown]
# # Hands-on lab: end-to-end CNN on real digits
# Regenerated from Book.md; run top to bottom.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
torch.manual_seed(0)

X, y = load_digits(return_X_y=True)                       # 1797 real 8-by-8 handwritten digits
X = (X / 16.0).reshape(-1, 1, 8, 8).astype("float32")     # scale to 0..1, shape (N, 1, 8, 8)
# stratified 70/15/15 split into train / validation / test
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)
Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=0, stratify=ytmp)
print(f"train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}")

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 8-by-8 -> 4x4
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))   # 4x4 -> 2x2
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 2 * 2, 10)) # 128 -> 10 logits
    def forward(self, x): return self.head(self.features(x))

net = DigitCNN()
opt = torch.optim.Adam(net.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva)
n, batch = len(Xtr), 64

def accuracy(Xt, yt):
    net.eval()
    with torch.no_grad():
        return (net(Xt).argmax(1) == yt).float().mean().item()

for epoch in range(1, 13):                                # 12 passes over the training set
    net.train(); perm = torch.randperm(n); total = 0.0
    for i in range(0, n, batch):                          # shuffled minibatches of 64
        idx = perm[i:i + batch]
        opt.zero_grad()
        loss = loss_fn(net(Xtr_t[idx]), ytr_t[idx])
        loss.backward(); opt.step()
        total += loss.item() * len(idx)                   # sum, then average per epoch
    if epoch == 1 or epoch % 2 == 0:
        print(f"epoch {epoch:2d}  train loss {total / n:.4f}  val acc {accuracy(Xva_t, yva_t):.3f}")

print(f"parameters {sum(p.numel() for p in net.parameters()):,}")
print(f"final test accuracy {accuracy(torch.tensor(Xte), torch.tensor(yte)):.3f}")
net.eval()
with torch.no_grad():
    pred = net(torch.tensor(Xte)).argmax(1).numpy()
cm = confusion_matrix(yte, pred); per_class = cm.diagonal() / cm.sum(1)   # correct / total per digit
print("per-class test accuracy:")
print("  digit  " + " ".join(f"{d:4d}" for d in range(10)))
print("  acc    " + " ".join(f"{a:4.2f}" for a in per_class))
print(f"total test errors {int((pred != yte).sum())} of {len(yte)}")

# %% [markdown]
# ## The Table 12-2 comparison: MLP vs CNN vs CNN + batch norm

# %%
def mlp():                                             # the wrong tool: flatten, then dense
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
def cnn(bn=False):                                     # the right tool, optionally with batch norm
    b1 = [nn.BatchNorm2d(32)] if bn else []; b2 = [nn.BatchNorm2d(64)] if bn else []
    return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), *b1, nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), *b2, nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
def fit_and_score(model, epochs=30):                   # identical training for all three
    opt = torch.optim.Adam(model.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr); n, batch = len(Xtr), 64
    for _ in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            ii = perm[i:i + batch]
            opt.zero_grad(); lf(model(Xt[ii]), yt[ii]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return (model(torch.tensor(Xte)).argmax(1) == torch.tensor(yte)).float().mean().item()
for name, ctor in [("MLP", mlp), ("CNN", lambda: cnn(False)), ("CNN + BN", lambda: cnn(True))]:
    torch.manual_seed(0); m = ctor()
    print(f"{name:9} test acc = {fit_and_score(m):.3f}  params = {sum(p.numel() for p in m.parameters()):,}")
# MLP 0.978 (4,810 params)   CNN 0.978 (19,466)   CNN + BN 0.996 (19,658)

# %% [markdown]
# ## The contest on real images: MNIST (CNN vs MLP, fixed small budget)

# %%
from torchvision import datasets, transforms
torch.manual_seed(0)
mnist_tr = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
mnist_te = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
g = torch.Generator().manual_seed(0)                      # a fixed random subset keeps the run fast
tr_idx = torch.randperm(len(mnist_tr), generator=g)[:4000]
te_idx = torch.randperm(len(mnist_te), generator=g)[:1000]
Xtr_m = torch.stack([mnist_tr[i][0] for i in tr_idx]); ytr_m = torch.tensor([mnist_tr[i][1] for i in tr_idx])
Xte_m = torch.stack([mnist_te[i][0] for i in te_idx]); yte_m = torch.tensor([mnist_te[i][1] for i in te_idx])
print(f"train {tuple(Xtr_m.shape)}  test {tuple(Xte_m.shape)}")   # 28x28 images now, not 8-by-8
def mlp_m():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))
def cnn_m():
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
                         nn.Flatten(), nn.Linear(32 * 7 * 7, 10))
def fit_and_score_m(model, epochs=6):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
    n, batch = len(Xtr_m), 64
    for _ in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            ii = perm[i:i + batch]
            opt.zero_grad(); lf(model(Xtr_m[ii]), ytr_m[ii]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return (model(Xte_m).argmax(1) == yte_m).float().mean().item()
for name, ctor in [("MLP", mlp_m), ("CNN", cnn_m)]:
    torch.manual_seed(0); m = ctor()
    print(f"{name:4} test acc = {fit_and_score_m(m):.3f}  params = {sum(p.numel() for p in m.parameters()):,}")
# MLP 0.885 (50,890 params)   CNN 0.949 (20,490): the CNN's advantage appears at real resolution

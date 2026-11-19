"""Section 11.11 lab. Explanatory: the full code is also in the book.
Part A is Example 11-33. Part B is Example 11-34.
The scaler is fit on the training split only.
"""

# %%
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
torch.manual_seed(0)
Xc, yc = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.25, random_state=0, stratify=yc)
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
scaler = StandardScaler().fit(Xtr)                 # training rows only; then transform the other splits
Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
f = lambda a, dt: torch.tensor(a, dtype=dt)
dl = DataLoader(TensorDataset(f(Xtr, torch.float32), f(ytr, torch.long)), batch_size=64, shuffle=True)
Xtr_t, ytr_t = f(Xtr, torch.float32), f(ytr, torch.long)
Xva_t, yva_t = f(Xva, torch.float32), f(yva, torch.long)
Xte_t, yte_t = f(Xte, torch.float32), f(yte, torch.long)
net = nn.Sequential(nn.Linear(30, 32), nn.ReLU(), nn.Linear(32, 2))
opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
for ep in range(40):
    net.train()
    for xb, yb in dl:
        opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        if (ep + 1) in (1, 10, 20, 40):
            tr = lossf(net(Xtr_t), ytr_t).item(); va = lossf(net(Xva_t), yva_t).item()
            print(f"epoch {ep+1:2d}: train loss = {tr:.3f}, val loss = {va:.3f}")
net.eval()
with torch.no_grad():
    acc = (net(Xte_t).argmax(1) == yte_t).float().mean().item()
    auc = roc_auc_score(yte, net(Xte_t).softmax(1)[:, 1].numpy())
print(f"Part A tabular MLP: test accuracy = {acc:.3f}, ROC AUC = {auc:.3f}")

# %%
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.manual_seed(0); lossf = nn.CrossEntropyLoss()
f = lambda a, dt: torch.tensor(a, dtype=dt)
Xd, yd = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
scaler = StandardScaler().fit(Xtr)
Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
dltr = DataLoader(TensorDataset(f(Xtr, torch.float32), f(ytr, torch.long)), batch_size=64, shuffle=True)
Xva_t, yva_t = f(Xva, torch.float32), f(yva, torch.long)
Xte_t, yte_t = f(Xte, torch.float32), f(yte, torch.long)
def build(use_bn, use_drop):
    L = [nn.Linear(64, 64), *([nn.BatchNorm1d(64)] if use_bn else []), nn.ReLU(),
         *([nn.Dropout(0.3)] if use_drop else []),
         nn.Linear(64, 64), *([nn.BatchNorm1d(64)] if use_bn else []), nn.ReLU(),
         *([nn.Dropout(0.3)] if use_drop else []),
         nn.Linear(64, 10)]
    return nn.Sequential(*L)
def train_model(model, opt):
    for _ in range(40):
        model.train()
        for xb, yb in dltr:
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    return model.eval()
def val_of(model):
    with torch.no_grad():
        hit = int((model(Xva_t).argmax(1) == yva_t).sum())
    return hit / len(yva_t), hit
for lr in (0.01, 0.03, 0.1):
    torch.manual_seed(0); m = build(False, False)
    train_model(m, torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9))
    acc_lr, hit = val_of(m); print(f"SGD momentum lr={lr:<4} val acc = {acc_lr:.3f}")
torch.manual_seed(0); m = build(False, False)
train_model(m, torch.optim.SGD(m.parameters(), lr=1e-3, momentum=0.9))
acc_bad, _ = val_of(m); print(f"SGD momentum lr=0.001 val acc = {acc_bad:.3f}")
configs = [("SGD with momentum, rate 0.1", False, False, "sgd"),
           ("Adam", False, False, "adam"),
           ("Adam with batch norm", True, False, "adam"),
           ("Adam with dropout", False, True, "adam"),
           ("Adam with batch norm and dropout", True, True, "adam")]
results = []
for name, bn, drop, opt_name in configs:
    torch.manual_seed(0); m = build(bn, drop)
    opt = (torch.optim.Adam(m.parameters(), lr=1e-3) if opt_name == "adam"
           else torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9))
    train_model(m, opt); acc_v, hit = val_of(m)
    results.append((name, acc_v, m)); print(f"{name:36} val acc = {acc_v:.3f}  ({hit}/{len(yva_t)})")
best_i = max(range(len(results)), key=lambda i: (results[i][1], -i))
best_name, best_va, best_model = results[best_i]
with torch.no_grad():
    test_acc = (best_model(Xte_t).argmax(1) == yte_t).float().mean().item()
print(f"selected '{best_name}' (val {best_va:.3f}) -> test accuracy = {test_acc:.3f}")

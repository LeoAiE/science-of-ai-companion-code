"""Appendix H solutions for Chapter 12, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 12.11 coding exercises

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn

# %% [markdown]
# 1. Parameters for a 96x96 RGB image, 256-unit dense layer vs 32 filters

# %%
# 1. Parameters for a 96x96 RGB image, 256-unit dense layer vs 32 filters
H, W, C, hidden = 96, 96, 3, 256
mlp = H * W * C * hidden + hidden
conv = 32 * (3 * 3 * C + 1)
print(H * W * C, mlp, conv, mlp // conv)
# expected output: 27648 7078144 896 7899

# %% [markdown]
# 2. Chelsea, grayscale, box blur and Sobel

# %%
# 2. Chelsea, grayscale, box blur and Sobel
from skimage.data import chelsea
from skimage.color import rgb2gray
from scipy.signal import correlate2d

def conv2d(img, kernel):
    kh, kw = kernel.shape
    out = np.zeros((img.shape[0] - kh + 1, img.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * kernel)
    return out

gray = rgb2gray(chelsea())
box = np.ones((3, 3)) / 9.0
sobel = np.array([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])
blurred = conv2d(gray, box)
edges = conv2d(gray, sobel)
print(gray.shape)
print(f"{gray.mean():.4f} {blurred.mean():.4f} {np.abs(edges).max():.1f}")
print(bool(np.allclose(blurred, correlate2d(gray, box, mode="valid"))))
# expected output: (300, 451)
# expected output: 0.4603 0.4600 2.1
# expected output: True

# %% [markdown]
# 3. Output size, both stride-2 cases, and dilation

# %%
# 3. Output size, both stride-2 cases, and dilation
def out_size(w, k, p, s, d=1):
    return (w + 2 * p - d * (k - 1) - 1) // s + 1
print(out_size(32, 3, 0, 1), out_size(32, 3, 1, 1), out_size(32, 3, 0, 2),
      out_size(32, 3, 1, 2), out_size(32, 3, 0, 1, d=2))
print(int(nn.Conv2d(1, 1, 3, stride=2, padding=1)(torch.zeros(1, 1, 32, 32)).shape[-1]))
# expected output: 30 32 15 16 28
# expected output: 16

# %% [markdown]
# 4. Whole pooled maps, not the global maximum

# %%
# 4. Whole pooled maps, not the global maximum
def pool2d(x, k=2, s=2, mode="max"):
    Hh, Ww = x.shape
    out = np.zeros(((Hh - k) // s + 1, (Ww - k) // s + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            win = x[i * s:i * s + k, j * s:j * s + k]
            out[i, j] = win.max() if mode == "max" else win.mean()
    return out

from sklearn.datasets import load_digits
img = (load_digits().data[0] / 16.0).reshape(8, 8)
frame = nn.MaxPool2d(2)(torch.tensor(img, dtype=torch.float32).view(1, 1, 8, 8)).squeeze().numpy()
print(bool(np.allclose(pool2d(img), frame)), pool2d(img).shape)
big = np.zeros((8, 8)); big[2, 2] = 9.0
same = np.zeros((8, 8)); same[2, 3] = 9.0
cross = np.zeros((8, 8)); cross[2, 4] = 9.0
print(bool(np.array_equal(pool2d(big), pool2d(same))), bool(np.array_equal(pool2d(big), pool2d(cross))))
# expected output: True (4, 4)
# expected output: True False

# %% [markdown]
# 5. Strided receptive field, then 32-channel parameter comparison

# %%
# 5. Strided receptive field, then 32-channel parameter comparison
def rf_stack(layers):
    r, jump = 1, 1
    for k, s in layers:
        r += (k - 1) * jump
        jump *= s
    return r, jump
print(rf_stack([(3, 2), (3, 1), (2, 2), (3, 1)]))
def cp(k, cin, cout):
    return (k * k * cin + 1) * cout
print(2 * cp(3, 32, 32), cp(5, 32, 32))
# expected output: (17, 4)
# expected output: 18496 25632

# %% [markdown]
# 6. Synthetic moons, one thread

# %%
# 6. Synthetic moons, one thread
from sklearn.datasets import make_moons
torch.manual_seed(0)
Xm, ym = make_moons(1000, noise=0.2, random_state=0)
Xm = torch.tensor(Xm, dtype=torch.float32)
ym = torch.tensor(ym)

class Block(nn.Module):
    def __init__(self, residual):
        super().__init__()
        self.lin = nn.Linear(32, 32)
        self.residual = residual
    def forward(self, x):
        out = torch.relu(self.lin(x))
        return out + x if self.residual else out

class DeepNet(nn.Module):
    def __init__(self, depth, residual):
        super().__init__()
        self.inp = nn.Linear(2, 32)
        self.blocks = nn.Sequential(*[Block(residual) for _ in range(depth)])
        self.out = nn.Linear(32, 2)
    def forward(self, x):
        return self.out(self.blocks(torch.relu(self.inp(x))))

def train_moons(net, epochs=300):
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    lf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lf(net(Xm), ym)
        loss.backward()
        opt.step()
    return loss.item(), (net(Xm).argmax(1) == ym).float().mean().item()

for residual in (False, True):
    torch.manual_seed(0)
    loss, acc = train_moons(DeepNet(24, residual))
    print("residual" if residual else "plain", round(loss, 4), round(acc, 3))
# expected output: plain 0.6931 0.5
# expected output: residual 0.0662 0.974

# %% [markdown]
# 7. Separable convolution, 32 to 96, kernel 5

# %%
# 7. Separable convolution, 32 to 96, kernel 5
standard = nn.Conv2d(32, 96, 5)
depthwise = nn.Conv2d(32, 32, 5, groups=32)
pointwise = nn.Conv2d(32, 96, 1)
def npar(m):
    return sum(p.numel() for p in m.parameters())
std, sep = npar(standard), npar(depthwise) + npar(pointwise)
print(std, sep, round(std / sep, 1))
xin = torch.zeros(1, 32, 16, 16)
print(tuple(standard(xin).shape), tuple(pointwise(depthwise(xin)).shape))
# expected output: 76896 4000 19.2
# expected output: (1, 96, 12, 12) (1, 96, 12, 12)

# %% [markdown]
# 8. Frozen ResNet-18 head on the ImageNet weights, which 12.7.1 has already downloaded to the torch cache.

# %%
# 8. Frozen ResNet-18 head on the ImageNet weights, which 12.7.1 has already downloaded to the torch cache.
from torchvision.models import resnet18, ResNet18_Weights
backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
backbone.fc = nn.Linear(512, 10)
for p in backbone.parameters():
    p.requires_grad = False
for p in backbone.fc.parameters():
    p.requires_grad = True
with torch.no_grad():
    logits = backbone(torch.zeros(1, 3, 224, 224))
print(sum(p.numel() for p in backbone.parameters() if p.requires_grad), tuple(logits.shape))
# expected output: 5130 (1, 10)

# %% [markdown]
# 9. Twenty images per class, rotation only

# %%
# 9. Twenty images per class, rotation only
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split
rng = np.random.default_rng(0)
imgs, labels = load_digits(return_X_y=True)
imgs = (imgs / 16.0).reshape(-1, 8, 8).astype("float32")
Xtr, Xte, ytr, yte = train_test_split(imgs, labels, test_size=0.6, random_state=0, stratify=labels)
idx = np.concatenate([rng.choice(np.where(ytr == c)[0], 20, replace=False) for c in range(10)])
Xfew, yfew = Xtr[idx], ytr[idx]

def spin(im):
    return np.clip(rotate(im, rng.uniform(-15, 15), reshape=False, mode="constant"), 0, 1)

Xte_pert = np.stack([spin(im) for im in Xte])

def small_cnn():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))

def run_aug(use_aug, epochs=80):
    m = small_cnn()
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    lf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        batch = np.stack([spin(im) for im in Xfew]) if use_aug else Xfew
        xb = torch.tensor(batch[:, None], dtype=torch.float32)
        opt.zero_grad()
        lf(m(xb), torch.tensor(yfew)).backward()
        opt.step()
    with torch.no_grad():
        Xt = torch.tensor(Xte_pert[:, None], dtype=torch.float32)
        return (m(Xt).argmax(1) == torch.tensor(yte)).float().mean().item()

print("clean", round(run_aug(False), 3), "rotated", round(run_aug(True), 3))
# expected output: clean 0.747 rotated 0.851

# %% [markdown]
# 10. DigitCNN plus batch norm, 12 epochs, same split as Example 12-28

# %%
# 10. DigitCNN plus batch norm, 12 epochs, same split as Example 12-28
torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).reshape(-1, 1, 8, 8).astype("float32")
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)
Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=0, stratify=ytmp)

class DigitCNNBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 2 * 2, 10))
    def forward(self, x):
        return self.head(self.features(x))

net = DigitCNNBN()
opt = torch.optim.Adam(net.parameters(), lr=0.01)
lf = nn.CrossEntropyLoss()
Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
n, bs = len(Xtr), 64
for _epoch in range(12):
    net.train()
    perm = torch.randperm(n)
    for i in range(0, n, bs):
        ii = perm[i:i + bs]
        opt.zero_grad()
        lf(net(Xt[ii]), yt[ii]).backward()
        opt.step()
net.eval()
with torch.no_grad():
    pred = net(torch.tensor(Xte)).argmax(1).numpy()
print(sum(p.numel() for p in net.parameters()), round(float((pred == yte).mean()), 3), int((pred != yte).sum()))
# expected output: 6186 0.996 1

# %% [markdown]
# ## 12.11 coding drills

# %%
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn

def osz(w, k, p, s, d=1):
    return (w + 2 * p - d * (k - 1) - 1) // s + 1
print(osz(32, 3, 0, 1))
# expected output: 30
print(osz(32, 3, 1, 1))
# expected output: 32
print(osz(32, 3, 1, 2))
# expected output: 16
print(2 * (3 - 1) + 1)
# expected output: 5
print((3 * 3 * 3 + 1) * 64)
# expected output: 1792

def conv2d(img, k):
    kh, kw = k.shape
    out = np.zeros((img.shape[0] - kh + 1, img.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * k)
    return out

ident = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
patch = np.arange(9).reshape(3, 3).astype(float)
print(np.sum(patch * ident))
# expected output: 4.0
print(conv2d(np.ones((6, 6)) * 5.0, np.ones((3, 3)) / 9.0).mean())
# expected output: 5.0
edge = np.zeros((6, 6)); edge[:, 3:] = 1.0
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
print(np.abs(conv2d(edge, sobel_x)).max())
# expected output: 4.0

def pool2d(x, k=2, s=2, mode="max"):
    Hh, Ww = x.shape
    out = np.zeros(((Hh - k) // s + 1, (Ww - k) // s + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            win = x[i * s:i * s + k, j * s:j * s + k]
            out[i, j] = win.max() if mode == "max" else win.mean()
    return out

win = np.array([[1, 7], [4, 2]], dtype=float)
print(pool2d(win, mode="max").item())
# expected output: 7.0
print(pool2d(win, mode="avg").item())
# expected output: 3.5
a0 = np.zeros((4, 4)); a0[0, 0] = 9
a1 = np.zeros((4, 4)); a1[0, 1] = 9
a2 = np.zeros((4, 4)); a2[0, 2] = 9
print(bool(np.array_equal(pool2d(a0), pool2d(a1))), bool(np.array_equal(pool2d(a0), pool2d(a2))))
# expected output: True False

def npar(m):
    return sum(p.numel() for p in m.parameters())
print(npar(nn.Conv2d(32, 96, 5)))
# expected output: 76896
print(npar(nn.Conv2d(256, 64, 1)))
# expected output: 16448
print(npar(nn.Conv2d(64, 64, 3, groups=64)))
# expected output: 640
sep = npar(nn.Conv2d(32, 32, 5, groups=32)) + npar(nn.Conv2d(32, 96, 1))
print(round(npar(nn.Conv2d(32, 96, 5)) / sep, 1))
# expected output: 19.2

c1 = nn.Conv2d(1, 16, 3, padding=1)
print(tuple(c1(torch.zeros(1, 1, 8, 8)).shape))
# expected output: (1, 16, 8, 8)
print(tuple(nn.MaxPool2d(2)(c1(torch.zeros(1, 1, 8, 8))).shape))
# expected output: (1, 16, 4, 4)
back = nn.Conv2d(1, 16, 3)
for p in back.parameters():
    p.requires_grad = False
print(sum(p.numel() for p in back.parameters() if p.requires_grad))
# expected output: 0

r, jump = 1, 1
for k, s in ((3, 2), (3, 1), (2, 2)):
    r += (k - 1) * jump
    jump *= s
print(r, jump)
# expected output: 9 4
def cp(k, cin, cout):
    return (k * k * cin + 1) * cout
print(2 * cp(3, 32, 32) < cp(5, 32, 32))
# expected output: True

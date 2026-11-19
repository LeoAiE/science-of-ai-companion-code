"""Chapter 12: Convolutional Neural Networks

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from the chapter slice; keep the book as the source.
"""

# %% [markdown]
# ## 12.1 Why MLPs fail on images

# %%
H, W, C = 224, 224, 3
mlp_params = H * W * C * 1000 + 1000               # one dense layer of 1000 units
conv_params = 64 * (3 * 3 * C + 1)                 # 64 conv filters, 3x3, over 3 input channels
print(f"{H}x{W}x{C} = {H*W*C:,} inputs")
print(f"MLP first layer (1000 units): {mlp_params:,} params")
print(f"CNN layer (64 filters 3x3):   {conv_params:,} params  ({mlp_params//conv_params:,}x fewer)")
# 150,528 inputs -> MLP 150,529,000 params  vs  CNN 1,792 (84,000x fewer), and the conv count is fixed regardless of image size

# %% [markdown]
# ## 12.1.1 What we built, and a self-check

# %% [markdown]
# ## 12.2 A little classical computer vision

# %% [markdown]
# Example 12-1. A from-scratch 2-D convolution (cross-correlation), checked against SciPy.

# %%
import numpy as np
from scipy.signal import correlate2d
def conv2d(img, kernel):                              # slide, multiply, sum (this is cross-correlation)
    kh, kw = kernel.shape
    out = np.zeros((img.shape[0] - kh + 1, img.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)
    return out
img = np.zeros((32, 32)); img[8:24, 8:24] = 1.0       # a white square on a black background
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])   # a vertical-edge detector
mine = conv2d(img, sobel_x)
print("from-scratch conv matches SciPy:", np.allclose(mine, correlate2d(img, sobel_x, mode="valid")))
print(f"sobel_x max |response| = {np.abs(mine).max():.0f}")   # 4: zero inside the flat square, nonzero at its vertical edges

# %%
from skimage import data
cam = data.camera().astype(float)                    # a real 512x512 grayscale photograph
box = np.ones((3, 3)) / 9.0                          # a 3x3 averaging (box blur) kernel
blurred = conv2d(cam, box)
edges = conv2d(cam, sobel_x)                         # reuse the vertical-edge kernel of Example 12-1
print(f"original mean {cam.mean():.1f}, blurred mean {blurred.mean():.1f}")
print(f"sobel_x max |response| = {np.abs(edges).max():.0f}")

# %% [markdown]
# ## 12.2.1 What we built, and a self-check

# %% [markdown]
# ## 12.3 The convolution operation

# %%
def out_size(w, k, p, s): return (w - k + 2 * p) // s + 1   # output = floor((W - K + 2P)/S) + 1
print("stride 1, no padding:", out_size(32, 3, 0, 1))    # 30  (shrinks by K-1)
print("stride 1, padding 1 :", out_size(32, 3, 1, 1))    # 32  ('same' size)
print("stride 2, padding 1 :", out_size(32, 3, 1, 2))    # 16  (downsampled ~2x)

# %%
import numpy as np
def conv2d(img, kernel):                              # the valid cross-correlation from Example 12-1
    kh, kw = kernel.shape
    out = np.zeros((img.shape[0] - kh + 1, img.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)
    return out
def cross_img(pos):
    a = np.zeros((20, 20)); r, c = pos
    a[r, c-1:c+2] = 1; a[r-1:r+2, c] = 1               # draw a small plus sign at `pos`
    return a
plus_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])   # a plus-shaped detector
for pos in [(4, 4), (15, 14)]:
    resp = conv2d(cross_img(pos), plus_kernel)
    peak = tuple(int(v) for v in np.unravel_index(resp.argmax(), resp.shape))
    print(f"plus at {pos}: peak response = {resp.max():.0f} at output {peak}")
# the SAME filter fires (peak 5) wherever the pattern sits: (4,4)->(3,3), (15,14)->(14,13). This is translation equivariance.

# %%
import numpy as np
import torch
import torch.nn as nn
def conv2d(img, kernel):
    kh, kw = kernel.shape
    out = np.zeros((img.shape[0] - kh + 1, img.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)
    return out
img = np.zeros((32, 32)); img[8:24, 8:24] = 1.0
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
conv = nn.Conv2d(1, 1, kernel_size=3, bias=False)     # one 3x3 filter, no bias
with torch.no_grad():
    conv.weight[:] = torch.tensor(sobel_x, dtype=torch.float32).view(1, 1, 3, 3)
x = torch.tensor(img, dtype=torch.float32).view(1, 1, 32, 32)   # (batch, channel, H, W)
framework = conv(x).squeeze().detach().numpy()
by_hand = conv2d(img, sobel_x)
print("nn.Conv2d matches by-hand conv2d:", np.allclose(framework, by_hand, atol=1e-5))
print("max abs difference below 1e-12:", bool(np.abs(framework - by_hand).max() < 1e-12))

# %%
import numpy as np
import torch
from scipy.signal import correlate2d
torch.manual_seed(0)
Xs = torch.randn(6, 6, requires_grad=True); Ks = torch.randn(3, 3, requires_grad=True)
out = torch.nn.functional.conv2d(Xs.view(1, 1, 6, 6), Ks.view(1, 1, 3, 3))   # 4x4 output
G = torch.randn_like(out)                                # a random upstream gradient dL/dO
out.backward(G)                                          # autograd fills dL/dX and dL/dK
Gn, Xn, Kn = G.squeeze().numpy(), Xs.detach().numpy(), Ks.detach().numpy()
dK = correlate2d(Xn, Gn, mode="valid")                   # dL/dK: slide G over X
dX = correlate2d(np.pad(Gn, 2), np.flip(Kn), mode="valid")   # dL/dX: flipped kernel over padded G
print("dL/dK matches autograd:", np.allclose(dK, Ks.grad.numpy(), atol=1e-5))
print("dL/dX matches autograd:", np.allclose(dX, Xs.grad.numpy(), atol=1e-5))

# %%
import torch
import torch.nn as nn
def out_size_dilated(w, k, p, s, d=1):
    return (w + 2 * p - d * (k - 1) - 1) // s + 1
x = torch.zeros(1, 1, 32, 32)
print("nn.Conv2d dilation 2 spatial size:", tuple(nn.Conv2d(1, 1, 3, dilation=2)(x).shape))
print("formula:", out_size_dilated(32, 3, 0, 1, d=2))

# %% [markdown]
# ## 12.3.1 A convolution layer with many channels

# %%
import numpy as np
import torch
import torch.nn as nn
X = np.array([[[1, 2, 3], [0, 1, 0], [4, 0, 1]],
              [[0, 1, 0], [2, 0, 1], [0, 3, 0]]], dtype=np.float32)
W = np.array([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]]], dtype=np.float32)
bias = np.array([0.5], dtype=np.float32)
cell = float(bias[0] + np.sum(X[:, 0:2, 0:2] * W[0]))
print("top-left cell", cell)
layer = nn.Conv2d(2, 1, 2, bias=True)
with torch.no_grad():
    layer.weight.copy_(torch.tensor(W))
    layer.bias.copy_(torch.tensor(bias))
    got = float(layer(torch.tensor(X).view(1, 2, 3, 3))[0, 0, 0, 0])
print("nn.Conv2d top-left", got)

# %% [markdown]
# Example 12-2. A strided, padded multichannel convolution written as an explicit loop and as unfold plus einsum, checked against PyTorch's Conv2d.

# %%
import torch.nn.functional as F
from skimage.data import chelsea
from skimage.util import img_as_float
torch.manual_seed(0)
W = torch.randn(4, 3, 3, 3)
b = torch.randn(4)
Xin = torch.randn(3, 8, 8)
ref = nn.Conv2d(3, 4, 3, stride=2, padding=1)
with torch.no_grad():
    ref.weight.copy_(W); ref.bias.copy_(b)

def conv_loop(x, weight, bias, stride=1, padding=0):
    cout, cin, k, _ = weight.shape
    xp = np.pad(x.detach().numpy(), ((0, 0), (padding, padding), (padding, padding)))
    h = (xp.shape[1] - k) // stride + 1
    w = (xp.shape[2] - k) // stride + 1
    y = np.zeros((cout, h, w), dtype=np.float32)
    wt = weight.detach().numpy()
    bb = bias.detach().numpy()
    for oc in range(cout):
        for i in range(h):
            for j in range(w):
                patch = xp[:, i * stride:i * stride + k, j * stride:j * stride + k]
                y[oc, i, j] = np.sum(patch * wt[oc]) + bb[oc]
    return y

mine = conv_loop(Xin, W, b, stride=2, padding=1)
gotm = ref(Xin.view(1, 3, 8, 8)).detach().numpy()[0]
print("toy max abs below 1e-5:", bool(np.max(np.abs(mine - gotm)) < 1e-5), "shape", mine.shape)
photo = torch.tensor(img_as_float(chelsea()).astype(np.float32)).permute(2, 0, 1)
xp = F.pad(photo, (1, 1, 1, 1))
windows = xp.unfold(1, 3, 2).unfold(2, 3, 2)
ye = torch.einsum("chwuv,ocuv->ohw", windows, W) + b.view(-1, 1, 1)
yr = ref(photo.view(1, 3, photo.shape[1], photo.shape[2]))[0]
print("chelsea", tuple(ye.shape), "max abs below 1e-5:", bool((ye - yr).abs().max() < 1e-5))

# %% [markdown]
# ## 12.3.2 What we built, and a self-check

# %% [markdown]
# ## 12.4 Pooling

# %% [markdown]
# Example 12-3. Max and average pooling written from scratch and applied to a 4-by-4 feature map.

# %%
import numpy as np
def pool2d(x, k=2, s=2, mode="max"):
    H, W = x.shape; out = np.zeros(((H - k) // s + 1, (W - k) // s + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            win = x[i*s:i*s+k, j*s:j*s+k]
            out[i, j] = win.max() if mode == "max" else win.mean()
    return out
fmap = np.array([[1, 2, 5, 1], [3, 0, 1, 2], [0, 4, 2, 1], [1, 1, 0, 3]], dtype=float)
print("max pool:", pool2d(fmap, mode="max").tolist())   # [[3.0, 5.0], [4.0, 3.0]]  keeps the strongest in each window
print("avg pool:", pool2d(fmap, mode="avg").tolist())   # [[1.5, 2.25], [1.5, 1.5]]  smooths each window

# %%
import torch
import torch.nn as nn
t = torch.tensor(fmap).view(1, 1, 4, 4)              # (batch, channel, H, W)
print("nn.MaxPool2d(2):", nn.MaxPool2d(2)(t).squeeze().tolist())

# %%
big = np.zeros((8, 8)); big[2, 2] = 9                 # one strong activation
same = np.zeros((8, 8)); same[2, 3] = 9               # one pixel right, still inside the window that covers (2, 2)
cross = np.zeros((8, 8)); cross[2, 4] = 9             # one pixel further, across the window boundary
print("same window, pooled maps equal?", np.array_equal(pool2d(big), pool2d(same)))
print("next window, pooled maps equal?", np.array_equal(pool2d(big), pool2d(cross)))
# True, then False. Comparing only the global maximum would print True for both, and for a move to a far corner.

# %% [markdown]
# ## 12.4.1 Backpropagation through pooling

# %%
import torch
import torch.nn.functional as F
fmap = torch.tensor([[1, 2, 5, 1], [3, 0, 1, 2], [0, 4, 2, 1], [1, 1, 0, 3]], dtype=torch.float32)
x = fmap.view(1, 1, 4, 4).requires_grad_(True)
F.max_pool2d(x, 2).backward(torch.ones(1, 1, 2, 2))
print("max-pool input gradient:")
print(x.grad.view(4, 4).tolist())
x2 = fmap.view(1, 1, 4, 4).requires_grad_(True)
F.avg_pool2d(x2, 2).backward(torch.ones(1, 1, 2, 2))
print("avg-pool input gradient, corner", float(x2.grad[0, 0, 0, 0]))
print("all equal to 0.25", bool(torch.allclose(x2.grad, torch.full_like(x2.grad, 0.25))))

# %% [markdown]
# ## 12.4.2 What we built, and a self-check

# %% [markdown]
# ## 12.5 CNN anatomy

# %%
rf = 1
for layer in range(1, 5):
    rf += 2                                            # each 3x3 conv adds (K-1) = 2 to the receptive field
    print(f"after {layer} layer(s): {rf}x{rf} input pixels affect one output")
# 3x3 -> 5x5 -> 7x7 -> 9x9   (two stacked 3x3 convs match one 5x5 but with fewer params and more nonlinearity)

# %%
def rf_stack(layers):                                  # layers: (kernel, stride) pairs, input to output
    r, jump = 1, 1                                     # receptive field; jump = input pixels between outputs
    for k, s in layers:
        r += (k - 1) * jump                            # each new tap spans `jump` input pixels
        jump *= s
    return r
print("four 3x3 convs, stride 1:", rf_stack([(3, 1)] * 4))       # 9: the stride-1 special case
lab = [(3, 1), (2, 2), (3, 1), (2, 2)]                 # the 12.10 lab: conv3, pool2, conv3, pool2
print("the lab's conv/pool stack:", rf_stack(lab))               # 10: covers the whole 8-by-8 image

# %%
def conv_params(k, cin, cout): return (k * k * cin + 1) * cout   # (K*K*Cin + 1) * Cout
print("conv 3x3, 3->64  :", f"{conv_params(3, 3, 64):,}")     # 1,792
print("conv 3x3, 64->128:", f"{conv_params(3, 64, 128):,}")   # 73,856
print("1x1 conv, 256->64:", f"{conv_params(1, 256, 64):,}")   # 16,448  cheap channel mixing / dimensionality reduction
print("two 3x3, 64->64  :", f"{2 * conv_params(3, 64, 64):,}")   # 73,856   same 5x5 coverage
print("one 5x5, 64->64  :", f"{conv_params(5, 64, 64):,}")       # 102,464  39% more parameters, one less ReLU

# %%
import numpy as np
feat = np.random.default_rng(0).normal(size=(128, 7, 7))   # 128 feature maps of 7x7
gap = feat.mean(axis=(1, 2))                               # average each whole map down to one number
print("feature maps", feat.shape, "-> GAP vector", gap.shape)        # (128, 7, 7) -> (128,)
print("GAP+dense:", 128 * 10 + 10, "params  vs flatten+dense:", (7*7*128) * 10 + 10)   # 1,290 vs 62,730

# %% [markdown]
# ## 12.5.1 Batch normalization on a feature map

# %% [markdown]
# Example 12-4. Batch, group, and instance normalization computed by hand over their axes and checked against PyTorch, with adaptive average pooling checked against the spatial mean.

# %%
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
torch.manual_seed(1)
dig = torch.tensor(load_digits().data[:8] / 16.0, dtype=torch.float32).view(8, 1, 8, 8)
xb = nn.Conv2d(1, 4, 3, padding=1)(dig)
bn = nn.BatchNorm2d(4)
bn.train()
yb = bn(xb)
mu = xb.mean(dim=(0, 2, 3))
var = xb.var(dim=(0, 2, 3), unbiased=False)
hand = (xb - mu.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + bn.eps)
hand = bn.weight.view(1, -1, 1, 1) * hand + bn.bias.view(1, -1, 1, 1)
print("BatchNorm2d max abs below 1e-5:", bool((yb - hand).detach().abs().max() < 1e-5))
g = nn.GroupNorm(2, 4)
N, C, H, W = xb.shape
xg = xb.view(N, 2, C // 2, H, W)
mu_g = xg.mean(dim=(2, 3, 4), keepdim=True)
var_g = xg.var(dim=(2, 3, 4), unbiased=False, keepdim=True)
hand_g = ((xg - mu_g) / torch.sqrt(var_g + g.eps)).view(N, C, H, W)
hand_g = g.weight.view(1, -1, 1, 1) * hand_g + g.bias.view(1, -1, 1, 1)
print("GroupNorm max abs below 1e-5:", bool((g(xb) - hand_g).detach().abs().max() < 1e-5))
ins = nn.InstanceNorm2d(4, affine=True)
mu_i = xb.mean(dim=(2, 3), keepdim=True)
var_i = xb.var(dim=(2, 3), unbiased=False, keepdim=True)
hand_i = (xb - mu_i) / torch.sqrt(var_i + ins.eps)
hand_i = ins.weight.view(1, -1, 1, 1) * hand_i + ins.bias.view(1, -1, 1, 1)
print("InstanceNorm max abs below 1e-5:", bool((ins(xb) - hand_i).detach().abs().max() < 1e-5))
print("adaptive pool matches the mean", bool(torch.allclose(
    nn.AdaptiveAvgPool2d(1)(xb).flatten(1), xb.mean(dim=(2, 3)))))

# %% [markdown]
# ## 12.5.2 Spatial dropout

# %%
import torch
import torch.nn as nn
torch.manual_seed(0)
t = torch.ones(2, 4, 8, 8)
elem = nn.Dropout(0.5)(t)
chan = nn.Dropout2d(0.5)(t)
per = (chan == 0).float().mean(dim=(2, 3))
print("element dropout zero fraction", round(float((elem == 0).float().mean()), 3))
print("channel zero fractions", [round(float(v), 3) for v in per.mean(0)])
print("each map is all or nothing", bool(torch.all((per == 0) | (per == 1))))

# %% [markdown]
# ## 12.5.3 What we built, and a self-check

# %% [markdown]
# ## 12.6 Landmark architectures

# %% [markdown]
# Example 12-5. Plain, batch-normalized, and residual 24-block networks trained on scikit-learn's synthetic two-moons data, with a diagnosis of the plain network at initialization.

# %%
import torch, torch.nn as nn
torch.set_num_threads(1)                             # pin the intra-op threads so the printed losses reproduce
from sklearn.datasets import make_moons
torch.manual_seed(0)
X, y = make_moons(1000, noise=0.2, random_state=0)
X = torch.tensor(X, dtype=torch.float32); y = torch.tensor(y)
class Block(nn.Module):
    def __init__(self, residual, bn=False):
        super().__init__(); self.lin = nn.Linear(32, 32); self.residual = residual
        self.bn = nn.BatchNorm1d(32) if bn else nn.Identity()   # optional batch norm inside the block
    def forward(self, x):
        out = torch.relu(self.bn(self.lin(x)))
        return out + x if self.residual else out       # out + x is the skip connection
class DeepNet(nn.Module):
    def __init__(self, depth, residual, bn=False):
        super().__init__(); self.inp = nn.Linear(2, 32)
        self.blocks = nn.Sequential(*[Block(residual, bn) for _ in range(depth)]); self.out = nn.Linear(32, 2)
    def forward(self, x): return self.out(self.blocks(torch.relu(self.inp(x))))
def train(net, epochs=300):
    opt = torch.optim.Adam(net.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = lf(net(X), y); loss.backward(); opt.step()
    return loss.item(), (net(X).argmax(1) == y).float().mean().item()
for name, residual, bn in [("plain", False, False), ("plain + BN", False, True), ("residual", True, False)]:
    torch.manual_seed(0); loss, acc = train(DeepNet(24, residual, bn))
    print(f"{name:10}: loss={loss:.4f}  acc={acc:.3f}")
torch.manual_seed(0); net = DeepNet(24, False)         # diagnose the untrained plain stack
net.zero_grad(); nn.CrossEntropyLoss()(net(X), y).backward()
print("plain net at init: logit std below 1e-8:", bool(net(X).std(0).mean() < 1e-8),
      "first-layer grad norm below 1e-8:", bool(net.inp.weight.grad.norm() < 1e-8))

# %% [markdown]
# Example 12-6. Twenty plain or residual Conv2d blocks trained on all 1,797 load_digits images, next to the entropy of the label distribution.

# %%
import torch
import torch.nn as nn
torch.set_num_threads(1)
from sklearn.datasets import load_digits
Xd, yd = load_digits(return_X_y=True)
Xd = torch.tensor((Xd / 16.0).reshape(-1, 1, 8, 8), dtype=torch.float32); yd_t = torch.tensor(yd)
import numpy as np
_p = np.bincount(yd) / len(yd)
print(f"label entropy = {-(_p * np.log(_p)).sum():.6f}   ln 10 = {np.log(10):.6f}")
class ConvBlock(nn.Module):
    def __init__(self, residual):
        super().__init__(); self.conv = nn.Conv2d(16, 16, 3, padding=1); self.residual = residual
    def forward(self, x):
        out = torch.relu(self.conv(x))
        return out + x if self.residual else out       # the same skip, now around a convolution
def conv_net(depth, residual):
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                         *[ConvBlock(residual) for _ in range(depth)],
                         nn.Flatten(), nn.Linear(16 * 8 * 8, 10))
def train_conv(net, epochs=250):
    opt = torch.optim.Adam(net.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = lf(net(Xd), yd_t); loss.backward(); opt.step()
    return loss.item(), (net(Xd).argmax(1) == yd_t).float().mean().item()
for residual in [False, True]:
    torch.manual_seed(0); loss, acc = train_conv(conv_net(20, residual))
    print(f"conv {'residual' if residual else 'plain   '}: loss={loss:.4f}  acc={acc:.3f}")

# %%
K, Cin, Cout = 3, 64, 128
standard = K * K * Cin * Cout                          # one conv mixes space AND channels at once
separable = K * K * Cin + Cin * Cout                   # depthwise (space, per channel) + pointwise 1x1 (channels)
print(f"standard convolution : {standard:,} params")   # 73,728
print(f"depthwise-separable  : {separable:,} params  ({standard/separable:.1f}x fewer)")   # 8,768 (8.4x fewer)

# %%
import torch, torch.nn as nn
Cin, Cout, K = 64, 128, 3
standard = nn.Conv2d(Cin, Cout, K, padding=1)                       # mixes space AND channels at once
depthwise = nn.Conv2d(Cin, Cin, K, padding=1, groups=Cin)          # one KxK filter per channel, no channel mixing
pointwise = nn.Conv2d(Cin, Cout, 1)                                 # 1x1 conv mixes channels only
def n(m): return sum(p.numel() for p in m.parameters())
std = n(standard); sep = n(depthwise) + n(pointwise)
print(f"standard  Conv2d(64,128,3)          : {std:,} params")           # 73,856
print(f"depthwise Conv2d(64,64,3,groups=64) : {n(depthwise):,} params")   # 640
print(f"pointwise Conv2d(64,128,1)          : {n(pointwise):,} params")   # 8,320
print(f"separable total                     : {sep:,} params  ({std/sep:.1f}x fewer)")   # 8,960 (8.2x fewer)
fmap = torch.zeros(1, 64, 8, 8)                                     # a dummy 64-channel 8-by-8 feature map
print("standard :", tuple(standard(fmap).shape))                    # (1, 128, 8, 8)
print("separable:", tuple(pointwise(depthwise(fmap)).shape))        # (1, 128, 8, 8)

# %%
H = W = 8                                              # the dummy feature map's size
std_mac = K * K * Cin * Cout * H * W                   # K^2 * Cin * Cout * Hout * Wout
dw_mac  = K * K * Cin * H * W                          # depthwise: one KxK filter per channel
pw_mac  = Cin * Cout * H * W                           # pointwise 1x1: pure channel mixing
print(f"standard : {std_mac:,} MACs")
print(f"separable: {dw_mac + pw_mac:,} MACs  ({std_mac / (dw_mac + pw_mac):.1f}x fewer)")

# %% [markdown]
# Example 12-7. An Inception module with switchable 1-by-1 bottlenecks, and the parameters the bottlenecks save.

# %%
class Inception(nn.Module):
    def __init__(self, cin, bottleneck=True):
        super().__init__()
        r = 32 if bottleneck else cin                  # channels after the 1x1 reduction
        red3 = [nn.Conv2d(cin, r, 1), nn.ReLU()] if bottleneck else []
        red5 = [nn.Conv2d(cin, r, 1), nn.ReLU()] if bottleneck else []
        self.b1 = nn.Conv2d(cin, 64, 1)                                   # 1x1 branch
        self.b3 = nn.Sequential(*red3, nn.Conv2d(r, 128, 3, padding=1))   # (1x1 then) 3x3 branch
        self.b5 = nn.Sequential(*red5, nn.Conv2d(r, 32, 5, padding=2))    # (1x1 then) 5x5 branch
        self.bp = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(cin, 32, 1))   # pooling branch
    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)   # stack on channels
x = torch.zeros(1, 192, 8, 8)                          # a 192-channel feature map, as GoogLeNet's module 3a sees
with_b, without_b = Inception(192), Inception(192, bottleneck=False)
print("output shape:", tuple(with_b(x).shape))
print(f"params with bottleneck: {n(with_b):,}   without: {n(without_b):,}  ({n(without_b)/n(with_b):.1f}x)")

# %% [markdown]
# ## 12.6.1 The residual blocks ResNet actually stacks

# %% [markdown]
# Example 12-8. The basic and bottleneck residual blocks with projection shortcuts, and their parameter counts next to torchvision's ResNet-18 and ResNet-50.

# %%
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout))
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        skip = x if self.down is None else self.down(x)
        return torch.relu(out + skip)

class Bottleneck(nn.Module):
    def __init__(self, cin, mid, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, cout, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout))
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        skip = x if self.down is None else self.down(x)
        return torch.relu(out + skip)

def nparams(m):
    return sum(p.numel() for p in m.parameters())

bb = BasicBlock(64, 64)
down = BasicBlock(64, 128, stride=2)
bot = Bottleneck(256, 64, 256)
print("basic 64", nparams(bb), tuple(bb(torch.zeros(1, 64, 8, 8)).shape))
print("down 64 to 128", nparams(down), "projection", nparams(down.down))
print("bottleneck 256-64-256", nparams(bot))
print("basic at 256", nparams(BasicBlock(256, 256)))
print("resnet18", nparams(resnet18(weights=None)))
print("resnet50", nparams(resnet50(weights=None)))

# %% [markdown]
# Example 12-9. A residual network of three basic blocks, one of them downsampling, trained for eight full-batch epochs on the load_digits training split.

# %%
torch.set_num_threads(1)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0)

class TinyRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.b1 = BasicBlock(16, 16)
        self.b2 = BasicBlock(16, 32, stride=2)
        self.b3 = BasicBlock(32, 32)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))
    def forward(self, x):
        return self.head(self.b3(self.b2(self.b1(self.stem(x)))))

X, y = load_digits(return_X_y=True)
X = (X / 16.0).reshape(-1, 1, 8, 8).astype("float32")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
net = TinyRes()
opt = torch.optim.Adam(net.parameters(), lr=0.01)
lf = nn.CrossEntropyLoss()
Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
for _ in range(8):
    opt.zero_grad(); lf(net(Xt), yt).backward(); opt.step()
with torch.no_grad():
    acc = (net(torch.tensor(Xte)).argmax(1) == torch.tensor(yte)).float().mean().item()
print(f"tiny ResNet test acc {acc:.3f}  params {nparams(net):,}")

# %% [markdown]
# Example 12-10. The tiny ResNet scored in eval mode and with test-batch statistics, after eight and after thirty full-batch epochs.

# %%
fresh = nn.BatchNorm2d(16)                             # what every BatchNorm2d holds before its first step
print("fresh BatchNorm2d: running mean", float(fresh.running_mean[0]), " running var", float(fresh.running_var[0]),
      " momentum", fresh.momentum)
def tiny_res_acc(epochs):
    torch.manual_seed(0)                               # the same start as Example 12-9
    m = TinyRes(); o = torch.optim.Adam(m.parameters(), lr=0.01)
    for _ in range(epochs):
        o.zero_grad(); lf(m(Xt), yt).backward(); o.step()
    Xtest, ytest = torch.tensor(Xte), torch.tensor(yte)
    with torch.no_grad():
        m.eval()                                       # the running statistics, as a deployed model uses them
        running = (m(Xtest).argmax(1) == ytest).float().mean().item()
        m.train()                                      # the test batch's own mean and variance, as Example 12-9 did
        batch = (m(Xtest).argmax(1) == ytest).float().mean().item()
    return running, batch
for epochs in (8, 30):
    running, batch = tiny_res_acc(epochs)
    print(f"{epochs:2d} epochs: eval mode {running:.3f}   test-batch statistics {batch:.3f}"
          f"   1 - 0.9**{epochs} = {1 - 0.9 ** epochs:.3f}")

# %% [markdown]
# ## 12.6.2 Choosing a pretrained architecture

# %%
from torchvision.models import (
    ResNet18_Weights, EfficientNet_B0_Weights, ConvNeXt_Tiny_Weights,
    MobileNet_V3_Small_Weights, VGG16_Weights)
for w in (ResNet18_Weights.IMAGENET1K_V1, EfficientNet_B0_Weights.IMAGENET1K_V1,
          ConvNeXt_Tiny_Weights.IMAGENET1K_V1, MobileNet_V3_Small_Weights.IMAGENET1K_V1,
          VGG16_Weights.IMAGENET1K_V1):
    meta = w.meta
    acc = meta["_metrics"]["ImageNet-1K"]["acc@1"]
    print(f"{w.__class__.__name__[:22]:22} acc {acc:6}  params {meta['num_params']:10,}  GFLOPS {meta['_ops']}")

# %% [markdown]
# ## 12.6.3 Activation memory

# %%
acts = []
hooks = []
trunk = resnet18(weights=None)
for mod in trunk.modules():
    if isinstance(mod, nn.Conv2d):
        hooks.append(mod.register_forward_hook(lambda m, i, o, box=acts: box.append(o.numel())))
with torch.no_grad():
    trunk(torch.zeros(1, 3, 224, 224))
for h in hooks:
    h.remove()
print("conv outputs", len(acts), "elements", sum(acts), "MB", round(sum(acts) * 4 / 1e6, 2))
print("first conv elements", acts[0], "last conv elements", acts[-1])

# %% [markdown]
# ## 12.6.4 Squeeze and excitation

# %% [markdown]
# Example 12-11. A squeeze-and-excitation block, global average pooling followed by a two-layer bottleneck and a sigmoid gate, with its parameter count and output shape.

# %%
class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        hidden = max(c // r, 1)
        self.fc1 = nn.Linear(c, hidden)
        self.fc2 = nn.Linear(hidden, c)
    def forward(self, x):
        s = x.mean(dim=(2, 3))
        w = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        return x * w.view(x.shape[0], -1, 1, 1)

se = SE(32, 16)
print("SE params", sum(p.numel() for p in se.parameters()), "shape", tuple(se(torch.zeros(2, 32, 8, 8)).shape))

# %% [markdown]
# ## 12.6.5 EfficientNet's compound scaling

# %%
alpha, beta, gamma = 1.2, 1.1, 1.15
print("product", round(alpha * beta**2 * gamma**2, 3))
for phi in (1, 2, 3):
    print(phi, round(alpha**phi, 3), round(beta**phi, 3), round(gamma**phi, 3))
from torchvision.models import EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights
for w in (EfficientNet_B0_Weights.IMAGENET1K_V1, EfficientNet_B1_Weights.IMAGENET1K_V1,
          EfficientNet_B2_Weights.IMAGENET1K_V1, EfficientNet_B3_Weights.IMAGENET1K_V1):
    print(w.__class__.__name__, w.meta["num_params"], w.meta["_ops"],
          w.meta["_metrics"]["ImageNet-1K"]["acc@1"])

# %% [markdown]
# ## 12.6.6 DenseNet

# %% [markdown]
# Example 12-12. Four densely connected convolution layers with growth rate 12, the channel count after each one, and the parameter count of torchvision's DenseNet-121.

# %%
class DenseLayer(nn.Module):
    def __init__(self, cin, growth):
        super().__init__()
        self.conv = nn.Conv2d(cin, growth, 3, padding=1, bias=False)
    def forward(self, xs):
        return self.conv(torch.cat(xs, dim=1))

feats = [torch.zeros(1, 16, 8, 8)]
cin = 16
counts = [cin]
for _ in range(4):
    y = DenseLayer(cin, 12)(feats)
    feats.append(y)
    cin += 12
    counts.append(cin)
print("channels after each layer", counts)
from torchvision.models import densenet121
print("densenet121", sum(p.numel() for p in densenet121(weights=None).parameters()))

# %% [markdown]
# ## 12.6.7 A ConvNeXt block

# %% [markdown]
# Example 12-13. A ConvNeXt block without layer scale: a 7-by-7 depthwise convolution, layer normalization, a fourfold pointwise expansion with GELU, the projection back, and the residual add.

# %%
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.pw2 = nn.Linear(4 * dim, dim)
    def forward(self, x):
        skip = x
        x = self.dw(x).permute(0, 2, 3, 1)
        x = self.pw2(torch.nn.functional.gelu(self.pw1(self.norm(x))))
        return skip + x.permute(0, 3, 1, 2)

block = ConvNeXtBlock(32)
print("ConvNeXt block", tuple(block(torch.zeros(1, 32, 8, 8)).shape),
      sum(p.numel() for p in block.parameters()))

# %% [markdown]
# ## 12.6.8 What we built, and a self-check

# %% [markdown]
# ## 12.7 Transfer learning and fine-tuning

# %% [markdown]
# Example 12-14. Transfer learning simulated on load_digits: digits 5 to 9 learned from scratch, with a frozen backbone pretrained on digits 0 to 4, and by fine-tuning that backbone.

# %%
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
Xd, yd = load_digits(return_X_y=True)
Xd = (Xd / 16.0).reshape(-1, 1, 8, 8).astype("float32")
def split_classes(cs):
    m = np.isin(yd, cs); remap = {c: i for i, c in enumerate(cs)}
    return Xd[m], np.array([remap[v] for v in yd[m]])
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
    def forward(self, x): return self.net(x)
def train(model, X, y, params=None, lr=0.01, epochs=200):
    opt = torch.optim.Adam(params or model.parameters(), lr=lr); lf = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(X), torch.tensor(y)
    for _ in range(epochs):
        opt.zero_grad(); lf(model(Xt), yt).backward(); opt.step()
def acc(model, X, y):
    with torch.no_grad(): return (model(torch.tensor(X)).argmax(1) == torch.tensor(y)).float().mean().item()
def n_train(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
Xp, yp = split_classes([0, 1, 2, 3, 4])               # pretrain a backbone on digits 0-4
pre = nn.Sequential(Backbone(), nn.Linear(32, 5)); train(pre, Xp, yp)
Xt2, yt2 = split_classes([5, 6, 7, 8, 9])             # the NEW task: digits 5-9
Xtr, Xte, ytr, yte = train_test_split(Xt2, yt2, test_size=0.4, random_state=0, stratify=yt2)
print(f"target task (digits 5-9), {len(Xtr)} training images:")
torch.manual_seed(0); s = nn.Sequential(Backbone(), nn.Linear(32, 5)); train(s, Xtr, ytr)
print(f"  from scratch       acc={acc(s, Xte, yte):.3f}  trainable={n_train(s)}")
bb = Backbone(); bb.load_state_dict(pre[0].state_dict())          # reuse the pretrained features
for p in bb.parameters(): p.requires_grad = False                # freeze the backbone
fe = nn.Sequential(bb, nn.Linear(32, 5)); train(fe, Xtr, ytr, params=fe[1].parameters())
print(f"  feature extraction acc={acc(fe, Xte, yte):.3f}  trainable={n_train(fe)}")
bb2 = Backbone(); bb2.load_state_dict(pre[0].state_dict())
ft = nn.Sequential(bb2, nn.Linear(32, 5)); train(ft, Xtr, ytr, lr=0.003)   # fine-tune all of it at a small lr
print(f"  fine-tuning        acc={acc(ft, Xte, yte):.3f}  trainable={n_train(ft)}")
# from scratch 0.975 (4965 params)   feature extraction 0.914 (165 params, ~30x fewer)   fine-tuning 0.944 (4965 params)

# %%
import os, sys
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
_err = sys.stderr                                                       # Jupyter captures sys.stderr, not only fd 2
sys.stderr = open(os.devnull, "w")
try:
    from transformers import pipeline
    from skimage.data import chelsea
    from PIL import Image
    clf = pipeline("image-classification", model="microsoft/resnet-18", device="cpu")   # a real ImageNet-pretrained CNN
    img = Image.fromarray(chelsea())                                       # a real public photo: a cat
    for r in clf(img, top_k=3):
        print(f"  {r['label']:<20} {r['score']:.3f}")
    print(f"{sum(p.numel() for p in clf.model.parameters()):,} parameters")
finally:
    sys.stderr.close(); sys.stderr = _err
# Egyptian cat 0.397   tiger cat 0.395   tabby, tabby cat 0.204
# 11,689,512 parameters

# %% [markdown]
# ## 12.7.1 Fine-tuning on a small CIFAR-10 subset

# %% [markdown]
# Example 12-15. Five ResNet-18 runs on an 80-image CIFAR-10 subset: from scratch, a frozen pretrained backbone with a new head, and three full fine-tunes that vary the learning rate and the batch-norm mode.

# %%
import torch
torch.set_num_threads(1)
import torch.nn as nn
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights
torch.manual_seed(0)
weights = ResNet18_Weights.IMAGENET1K_V1
tfm = weights.transforms()
print(tfm)
train_ds = datasets.CIFAR10("data", train=True, download=True)
test_ds = datasets.CIFAR10("data", train=False, download=True)
print(len(train_ds), len(test_ds))
g = torch.Generator().manual_seed(0)

def subset(ds, per_class, gen):
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

Xtr, ytr = subset(train_ds, 8, g)
Xte, yte = subset(test_ds, 4, g)
print(tuple(Xtr.shape), tuple(Xte.shape))

def run_mode(mode, lr):
    torch.manual_seed(0)
    model = resnet18(weights=None if mode == "scratch" else weights)
    model.fc = nn.Linear(512, 10)
    if mode == "frozen":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    lf = nn.CrossEntropyLoss()
    n, batch = len(Xtr), 8
    for _epoch in range(2):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            ii = perm[i:i + batch]
            if mode == "frozen":
                model.eval()
                model.fc.train()
            else:
                model.train()
            if mode == "finetune_bn_eval":    # every weight trains, but BN keeps ImageNet's statistics
                for mod in model.modules():
                    if isinstance(mod, nn.BatchNorm2d):
                        mod.eval()
            opt.zero_grad()
            lf(model(Xtr[ii]), ytr[ii]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return round(acc, 3), ntr

for mode, lr in (("scratch", 0.01), ("frozen", 0.01), ("finetune", 0.001),
                 ("finetune_bn_eval", 0.001), ("finetune", 0.0001)):
    acc, ntr = run_mode(mode, lr)
    print(f"{mode:16} lr {lr:<6}  acc {acc:.3f}  trainable {ntr}")

# %% [markdown]
# Example 12-16 trains seven ResNet-18 runs, about 47 minutes (2,822 seconds in one run) on the CPU of the reference machine, so it
# lives in its own companion script and notebook in this folder, 08_cifar10_transfer.py and 08_cifar10_transfer.ipynb.

# %% [markdown]
# ## 12.7.2 Reading images from class folders

# %%
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from sklearn.datasets import load_digits
from torchvision.datasets import ImageFolder
root = Path("digit_folders")
X, y = load_digits(return_X_y=True)
X = (X / 16.0).reshape(-1, 8, 8)
rng = np.random.default_rng(0)
for c in range(10):
    (root / str(c)).mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(rng.choice(np.where(y == c)[0], 2, replace=False)):
        Image.fromarray((X[i] * 255).astype(np.uint8), mode="L").save(root / str(c) / f"{k}.png")
ds = ImageFolder(root)
img0, lab0 = ds[0]
print(ds.classes, img0.mode, img0.size, lab0)

# %% [markdown]
# ## 12.7.3 What we built, and a self-check

# %% [markdown]
# ## 12.8 Data augmentation for images

# %% [markdown]
# Example 12-17. A from-scratch augment function that rotates, shifts, and adds noise to a load_digits image, applied four times to the first digit.

# %%
import numpy as np
from scipy.ndimage import rotate, shift
from sklearn.datasets import load_digits
rng = np.random.default_rng(0)
Xd, yd = load_digits(return_X_y=True)
Xd = (Xd / 16.0).reshape(-1, 8, 8).astype("float32")
def augment(img):
    a = rotate(img, rng.uniform(-15, 15), reshape=False, mode="constant")   # small random rotation
    a = shift(a, rng.uniform(-1, 1, 2), mode="constant")                    # small random shift
    return np.clip(a + rng.normal(0, 0.05, a.shape), 0, 1)                  # mild noise, clipped to [0,1]
img = Xd[0]; augs = [augment(img) for _ in range(4)]
print(f"original mean = {img.mean():.3f}")
print("augmented means =", [round(float(a.mean()), 3) for a in augs])
# original 0.287; augmented [0.256, 0.248, 0.246, 0.258]: varied pixels, but still the same digit (label preserved)

# %% [markdown]
# Example 12-18. Mixup and CutMix of a 0 and a 7 computed by hand, with the soft label each blend implies.

# %%
i, j, lam = 0, 7, 0.7
mixed_img = lam * Xd[i] + (1 - lam) * Xd[j]            # blend the two images pixelwise
soft = np.zeros(10); soft[yd[i]] += lam; soft[yd[j]] += 1 - lam   # blend the labels the same way
print(f"mixup of a '{yd[i]}' and a '{yd[j]}', lambda={lam}:")
print(f"  means: 0.7*{Xd[i].mean():.4f} + 0.3*{Xd[j].mean():.4f} = {lam*Xd[i].mean() + (1-lam)*Xd[j].mean():.4f}"
      f"  vs mixed image mean {mixed_img.mean():.4f}")
print(f"  soft label = {np.round(soft, 3)}")
r0, c0, h, w = 2, 2, 4, 4                              # CutMix: paste a 4x4 patch of image j into image i
cut = Xd[i].copy(); cut[r0:r0+h, c0:c0+w] = Xd[j][r0:r0+h, c0:c0+w]
lam_cut = 1 - (h * w) / cut.size                       # corrected mixing weight: 1 - pasted-area fraction
soft_cut = np.zeros(10); soft_cut[yd[i]] += lam_cut; soft_cut[yd[j]] += 1 - lam_cut
print(f"CutMix patch rows {r0}:{r0+h}, cols {c0}:{c0+w}: lambda = 1 - 16/64 = {lam_cut:.2f}")
print(f"  soft label = {np.round(soft_cut, 3)}")

# %% [markdown]
# Example 12-19. Two digit CNNs trained on the same 300 images, one on clean copies and one on fresh augmentations each epoch, both tested on rotated, shifted, noisy test digits.

# %%
import torch, torch.nn as nn
torch.set_num_threads(1)
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.6, random_state=0, stratify=yd)
idx = np.concatenate([rng.choice(np.where(ytr == c)[0], 30, replace=False) for c in range(10)])
Xfew, yfew = Xtr[idx], ytr[idx]                       # 300 training images
Xte_pert = np.stack([augment(im) for im in Xte])      # the TEST set is rotated/shifted/noisy (real-world-like)
def cnn():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))
def run(use_aug, epochs=150):
    m = cnn(); opt = torch.optim.Adam(m.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        batch = np.stack([augment(im) for im in Xfew]) if use_aug else Xfew   # fresh augments each epoch
        xb = torch.tensor(batch[:, None], dtype=torch.float32); yb = torch.tensor(yfew)
        opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
    with torch.no_grad():
        Xt = torch.tensor(Xte_pert[:, None], dtype=torch.float32)
        return (m(Xt).argmax(1) == torch.tensor(yte)).float().mean().item()
print(f"clean training only : test acc = {run(False):.3f}")   # 0.468  fails on the perturbed test set
print(f"with augmentation   : test acc = {run(True):.3f}")    # 0.837  it learned the rotation and shift invariance

# %%
import torch
torch.set_num_threads(1)
from torchvision.transforms import v2
torch.manual_seed(0)
aug = v2.Compose([
    v2.RandomRotation(15),                      # random rotation in [-15, 15] degrees
    v2.RandomAffine(0, translate=(1/8, 1/8)),   # random shift of up to 1 pixel on an 8-pixel image
    v2.GaussianNoise(sigma=0.05, clip=True),    # mild noise, clipped back into [0, 1]
])
img_t = torch.tensor(Xd[0])[None]               # shape (1, 8, 8): channel, height, width
outs = [aug(img_t) for _ in range(4)]
print("output shape:", tuple(outs[0].shape))
print("value range : [%.3f, %.3f]" % (min(o.min() for o in outs), max(o.max() for o in outs)))
print("augmented means =", [round(o.mean().item(), 3) for o in outs])

# %% [markdown]
# Example 12-20. The perturbed-test CNN trained with augmentation alone, with v2.MixUp, and with v2.CutMix, scored for accuracy and calibration over three seeds.

# %%
from torchvision.transforms import v2
def ece(prob, y, bins=10):                             # expected calibration error, the M = 10 bins of 11.5.5
    conf, pred = prob.max(1)
    total = 0.0
    for m in range(bins):
        s = (conf > m / bins) & (conf <= (m + 1) / bins)   # the examples whose confidence falls in bin m + 1
        if s.any():
            total += float(s.float().mean() * ((pred[s] == y[s]).float().mean() - conf[s].mean()).abs())
    return total
def scores(model, X, y):                               # accuracy, ECE, and mean confidence on one test set
    with torch.no_grad():
        prob = model(torch.tensor(X[:, None], dtype=torch.float32)).softmax(1)
    y = torch.tensor(y)
    return (prob.argmax(1) == y).float().mean().item(), ece(prob, y), prob.max(1).values.mean().item()
def run_mix(mixer, seed, epochs=150):
    global rng
    rng = np.random.default_rng(seed)                  # the same augmented images for every method at this seed
    torch.manual_seed(seed)                            # and the same starting weights
    m = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                      nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                      nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))   # the CNN of Example 12-19
    opt = torch.optim.Adam(m.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        xb = torch.tensor(np.stack([augment(im) for im in Xfew])[:, None], dtype=torch.float32)
        yb = torch.tensor(yfew)
        perm = torch.randperm(len(yb))                 # MixUp and CutMix pair each image with its neighbor,
        xb, yb = xb[perm], yb[perm]                    # and Xfew is sorted by class, so shuffle first
        if mixer is None:
            target = yb                                # hard labels, as in Example 12-19
        else:
            xb, target = mixer(xb, yb)                 # mixed images and soft labels of shape (300, 10)
        opt.zero_grad(); lf(m(xb), target).backward(); opt.step()
    return scores(m, Xte, yte) + scores(m, Xte_pert, yte)
mixers = {"augment only": None,
          "augment + MixUp": v2.MixUp(alpha=1.0, num_classes=10),
          "augment + CutMix": v2.CutMix(alpha=1.0, num_classes=10)}
print("mean of seeds 0, 1, 2   clean: acc    ECE   conf   perturbed: acc    ECE   conf   (perturbed acc range)")
for name, mixer in mixers.items():
    r = np.array([run_mix(mixer, s) for s in (0, 1, 2)])   # one row per seed, six columns
    a = r.mean(0)
    print(f"{name:22} {a[0]:12.3f} {a[1]:6.3f} {a[2]:6.3f} {a[3]:15.3f} {a[4]:6.3f} {a[5]:6.3f}"
          f"   ({r[:, 3].min():.3f} to {r[:, 3].max():.3f})")

# %% [markdown]
# ## 12.8.1 Automated augmentation policies

# %%
import torch
from skimage.data import chelsea
from torchvision.transforms import v2
photo = torch.tensor(chelsea()).permute(2, 0, 1)   # uint8, channels first
aug = v2.RandAugment(num_ops=2, magnitude=6)
out = aug(photo)
print(tuple(photo.shape), tuple(out.shape), out.dtype == photo.dtype)
trivial = v2.TrivialAugmentWide()                  # one random operation at a random strength
out2 = trivial(photo)
print(tuple(out2.shape), out2.dtype == photo.dtype, trivial.num_magnitude_bins)

# %% [markdown]
# ## 12.8.2 What we built, and a self-check

# %% [markdown]
# ## 12.9 What CNNs learn

# %% [markdown]
# Example 12-21. The digit CNN that 12.9 inspects, trained for 150 full-batch epochs, and the value range of its first-layer filters.

# %%
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
Xd, yd = load_digits(return_X_y=True)
Xd = (Xd / 16.0).reshape(-1, 1, 8, 8).astype("float32")
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())   # last conv: 32 x 8 x 8
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))
    def forward(self, x): return self.head(self.features(x))
net = CNN(); opt = torch.optim.Adam(net.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
for _ in range(150):
    opt.zero_grad(); lf(net(Xt), yt).backward(); opt.step()
with torch.no_grad():
    print(f"trained CNN test accuracy = {(net(torch.tensor(Xte)).argmax(1) == torch.tensor(yte)).float().mean().item():.3f}")   # 0.973
w = net.features[0].weight.detach().numpy()[:, 0]      # the 16 first-layer 3x3 filters
print(f"first conv layer: {w.shape[0]} filters, value range [{w.min():.2f}, {w.max():.2f}]")   # 16 filters, [-1.41, 0.91]

# %% [markdown]
# Example 12-22. Forward hooks that capture the feature maps of both convolution layers for one test digit, and the 8-by-8 grid of the strongest first-layer map.

# %%
feats = {}
h1 = net.features[0].register_forward_hook(lambda m, i, o: feats.__setitem__("conv1", o))
h2 = net.features[2].register_forward_hook(lambda m, i, o: feats.__setitem__("conv2", o))
with torch.no_grad(): net(torch.tensor(Xte[:1]))       # one test digit through the trained net
h1.remove(); h2.remove()
c1, c2 = feats["conv1"][0], feats["conv2"][0]
print("digit label:", yte[0])
print("conv1 maps:", tuple(c1.shape), "  conv2 maps:", tuple(c2.shape))
print(f"fraction of positive (surviving) activations: conv1 {(c1 > 0).float().mean():.2f}, conv2 {(c2 > 0).float().mean():.2f}")
strongest = c1.abs().amax(dim=(1, 2))
print("strongest conv1 map:", int(strongest.argmax()), " peak |activation|:", round(float(strongest.max()), 1))
np.set_printoptions(precision=1, suppress=True, linewidth=120)
print("that map's 8x8 activation grid:"); print(c1[int(strongest.argmax())].numpy())

# %%
import torch
A1 = torch.tensor([[1., -3.], [0.5, 1.]])
A2 = torch.tensor([[0., 2.], [0., 0.]])
g1 = torch.ones(2, 2)
g2 = torch.tensor([[1., 1.], [0., 0.]])
alpha1, alpha2 = g1.mean(), g2.mean()
weighted = alpha1 * A1 + alpha2 * A2
print("alphas", float(alpha1), float(alpha2))
print("weighted", weighted.tolist())
print("ReLU", torch.relu(weighted).tolist())

# %% [markdown]
# Example 12-23. Grad-CAM on the trained digit CNN, and the share of the heatmap that falls on the ink.

# %%
acts, grads = {}, {}
net.features[-2].register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
net.features[-2].register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
def grad_cam(img):
    net.zero_grad(); out = net(torch.tensor(img[None])); cls = int(out.argmax())
    out[0, cls].backward()
    weights = grads["g"].mean(dim=(2, 3), keepdim=True)        # alpha_c: gradient-weighted map importance
    return torch.relu((weights * acts["a"]).sum(1))[0].detach().numpy(), cls
on_ink, ink_frac = [], []
for i in range(60):
    cam, cls = grad_cam(Xte[i])
    if cls != yte[i] or cam.sum() == 0: continue
    ink = Xte[i, 0] > 0.3                                       # the digit's strokes
    on_ink.append(cam[ink].sum() / cam.sum()); ink_frac.append(ink.mean())
print(f"Grad-CAM mass on the ink = {np.mean(on_ink):.2f}  (ink is only {np.mean(ink_frac):.2f} of the pixels)")
# 0.46 of the heatmap sits on the 0.38 of pixels that are strokes

# %%
import torch
from torchvision.models import resnet50
net50 = resnet50(weights=None)
box = {}
def _grab_shape(module, inputs, output):
    box["shape"] = tuple(output.shape)
hook = net50.layer4.register_forward_hook(_grab_shape)
with torch.no_grad():
    net50(torch.zeros(1, 3, 224, 224))
hook.remove()
print("ResNet-50 layer4:", box["shape"])

# %% [markdown]
# Example 12-24. Grad-CAM with the ImageNet ResNet-18 on a cat photograph and a coffee photograph placed side by side, one map per class.

# %%
import torch
import torch.nn.functional as F
from skimage.data import chelsea, coffee
from torchvision.models import resnet18, ResNet18_Weights
torch.set_num_threads(1)
weights = ResNet18_Weights.IMAGENET1K_V1               # the ImageNet weights of 12.7.1
cats = weights.meta["categories"]                     # the 1,000 ImageNet class names
model = resnet18(weights=weights).eval()

def square224(img):                                   # crop the central square, then resize it to 224 by 224
    t = torch.tensor(img).permute(2, 0, 1).float() / 255
    h, w = t.shape[1:]
    s = min(h, w)
    t = t[:, (h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return F.interpolate(t[None], size=224, mode="bilinear", antialias=True)[0]

pair = torch.cat([square224(chelsea()), square224(coffee())], dim=2)   # cat on the left, coffee cup on the right
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
x = ((pair - mean) / std)[None]                       # the ImageNet normalization; shape (1, 3, 224, 448)

feats = {}
model.layer4.register_forward_hook(lambda m, i, o: feats.__setitem__("A", o))
logits = model(x)                                     # one forward pass; every class below reuses its graph
print("layer4 maps", tuple(feats["A"].shape))
top = logits[0].detach().softmax(0).topk(5)           # the five most likely classes
for p, k in zip(top.values, top.indices):
    print(f"  {cats[int(k)]:<14} {float(p):.3f}")

def grad_cam(cls):
    A = feats["A"]
    g, = torch.autograd.grad(logits[0, cls], A, retain_graph=True)   # dy^c / dA^k_ij
    alpha = g.mean(dim=(2, 3), keepdim=True)                         # alpha_k^c: the average over the 7 x 14 grid
    return torch.relu((alpha * A).sum(1))[0].detach()                # L^c, shape (7, 14)

for name in ("tiger cat", "cup", "espresso"):        # the top cat class and the two top coffee classes
    cam = grad_cam(cats.index(name))
    left = float(cam[:, :7].sum() / cam.sum())        # columns 0 to 6 lie over the cat photograph
    r, c = divmod(int(cam.argmax()), cam.shape[1])
    print(f"{name}: share on the cat half {left:.2f}, peak at row {r}, column {c}")
    print((cam / cam.max() * 9).round().int().numpy())   # the map scaled to 0 to 9, one digit per cell

# %% [markdown]
# ## 12.9.1 Saliency, the gradient on the pixels

# %% [markdown]
# Example 12-25. A saliency map from one backward pass through a digit CNN trained for 40 full-batch epochs.

# %%
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
torch.set_num_threads(1)
Xd, yd = load_digits(return_X_y=True)
Xd = (Xd / 16.0).reshape(-1, 1, 8, 8).astype("float32")
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
net = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))
opt = torch.optim.Adam(net.parameters(), lr=0.01)
lf = nn.CrossEntropyLoss()
Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
for _ in range(40):
    opt.zero_grad(); lf(net(Xt), yt).backward(); opt.step()
x = torch.tensor(Xte[:1], requires_grad=True)
net(x)[0, int(yte[0])].backward()
sal = x.grad.abs()[0, 0]
ink = Xte[0, 0] > 0.3
print(f"saliency mass on ink {float(sal[ink].sum() / sal.sum()):.2f}  ink fraction {float(ink.mean()):.2f}")

# %% [markdown]
# Example 12-26. Activation maximization: thirty steps of gradient ascent from a blank image on one class score of the saliency network, with an L2 penalty on the pixels.

# %%
c = int(yte[0])                                    # the saliency digit's true class
x0 = torch.zeros(1, 1, 8, 8, requires_grad=True)   # start from a blank image
with torch.no_grad():
    s = net(x0)
print(f"blank image: class {c} score {float(s[0, c]):.2f}, predicted class {int(s.argmax())}")
for _ in range(30):
    obj = net(x0)[0, c] - 0.01 * (x0 ** 2).sum()   # class score minus an L2 penalty on the pixels
    grad, = torch.autograd.grad(obj, x0)
    with torch.no_grad():
        x0 += 1.0 * grad                           # one step up the gradient
with torch.no_grad():
    s = net(x0)
print(f"after 30 steps: class {c} score {float(s[0, c]):.2f}, probability {float(s.softmax(1)[0, c]):.3f}, predicted class {int(s.argmax())}")
print(f"pixel range [{float(x0.detach().min()):.2f}, {float(x0.detach().max()):.2f}]")

# %% [markdown]
# ## 12.9.2 What we built, and a self-check

# %% [markdown]
# ## 12.10 Hands-on lab: building a CNN end to end

# %% [markdown]
# Example 12-27. The lab's first experiment: MLP versus CNN versus batch-normalized CNN on the digits.

# %%
import torch, torch.nn as nn
torch.set_num_threads(1)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).reshape(-1, 1, 8, 8).astype("float32")
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)
Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=0, stratify=ytmp)
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

# %% [markdown]
# Example 12-28. DigitCNN: a compact two-block CNN trained end to end on the train/validation/test split.

# %%
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
# ## 12.10.1 The contest on real images: MNIST

# %% [markdown]
# Example 12-29. The MLP-versus-CNN contest on 28-by-28 MNIST images.

# %%
import torch, torch.nn as nn
torch.set_num_threads(1)
from torchvision import datasets, transforms
torch.manual_seed(0)
mnist_tr = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
mnist_te = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
g = torch.Generator().manual_seed(0)                      # a fixed random subset keeps the run fast
tr_idx = torch.randperm(len(mnist_tr), generator=g)[:4000]
te_idx = torch.randperm(len(mnist_te), generator=g)[:1000]
Xtr = torch.stack([mnist_tr[i][0] for i in tr_idx]); ytr = torch.tensor([mnist_tr[i][1] for i in tr_idx])
Xte = torch.stack([mnist_te[i][0] for i in te_idx]); yte = torch.tensor([mnist_te[i][1] for i in te_idx])
print(f"train {tuple(Xtr.shape)}  test {tuple(Xte.shape)}")   # 28x28 images now, not 8-by-8
def mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))
def cnn():
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
                         nn.Flatten(), nn.Linear(32 * 7 * 7, 10))
def fit_and_score(model, epochs=6):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
    n, batch = len(Xtr), 64
    for _ in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            ii = perm[i:i + batch]
            opt.zero_grad(); lf(model(Xtr[ii]), ytr[ii]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return (model(Xte).argmax(1) == yte).float().mean().item()
for name, ctor in [("MLP", mlp), ("CNN", cnn)]:
    torch.manual_seed(0); m = ctor()
    print(f"{name:4} test acc = {fit_and_score(m):.3f}  params = {sum(p.numel() for p in m.parameters()):,}")

# %% [markdown]
# Example 12-30. The contest of Example 12-29 on all 60,000 training and all 10,000 test images of MNIST.

# %%
Xtr = mnist_tr.data.unsqueeze(1).float() / 255           # all 60,000 training images, scaled as ToTensor scales them
ytr = mnist_tr.targets
Xte = mnist_te.data.unsqueeze(1).float() / 255           # all 10,000 test images
yte = mnist_te.targets
print(f"train {tuple(Xtr.shape)}  test {tuple(Xte.shape)}  same pixels as ToTensor: {torch.equal(Xtr[0], mnist_tr[0][0])}")
for name, ctor in [("MLP", mlp), ("CNN", cnn)]:
    torch.manual_seed(0); m = ctor()
    acc = fit_and_score(m)                                # the same six epochs, batches of 64, Adam at 1e-3
    print(f"{name:4} test acc = {acc:.4f}  errors = {round((1 - acc) * len(yte))} of {len(yte):,}")

# %% [markdown]
# ## 12.10.2 What we built, and a self-check

# %% [markdown]
# ## 12.11 Interview questions and coding exercises

# %% [markdown]
# ## Convolution and architecture

# %% [markdown]
# ## Pooling and components

# %% [markdown]
# ## Architectures and practice

# %% [markdown]
# ## Coding exercises

# %%
H, W, C, hidden, filters = 96, 96, 3, 256, 32
def parameter_counts(h, w, c, hidden, filters):
    ...
print(H * W * C, parameter_counts(H, W, C, hidden, filters))

# %%
from skimage.data import chelsea
def conv2d(img, kernel):
    ...
print(chelsea().shape, conv2d(None, None))

# %%
def out_size(w, k, p, s, d=1):
    ...
print(out_size(32, 3, 0, 1))

# %%
from sklearn.datasets import load_digits
def pool2d(x, k=2, s=2, mode="max"):
    ...
print(load_digits().data.shape, pool2d(None))

# %%
def rf_stack(layers):
    ...
print(rf_stack([(3, 2), (3, 1), (2, 2), (3, 1)]))

# %%
from sklearn.datasets import make_moons
X, y = make_moons(1000, noise=0.2, random_state=0)
print(X.shape, y.shape)

# %%
import torch
x = torch.zeros(1, 32, 16, 16)
print(tuple(x.shape))

# %%
from torchvision.models import resnet18
print(resnet18(weights=None).fc.in_features)

# %%
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
print(X.shape)

# %%
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
print(X.shape, len(set(y)))

# %% [markdown]
# ## Coding drills

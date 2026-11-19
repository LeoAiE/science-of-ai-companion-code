"""Chapter 24 walkthrough. Generated from the chapter slice."""

# %% [markdown]
# ## 24.1 The generative model family tree

# %%
import numpy as np
rng = np.random.default_rng(0)

N, P = 64, 3                                              # 64 'pixels' (8x8), store 3 patterns
patterns = rng.choice([-1, 1], size=(P, N))              # P random +/-1 patterns
W = sum(np.outer(p, p) for p in patterns) / N
np.fill_diagonal(W, 0)                                    # no self-connections

def recall(s, steps=20):
    s = s.copy()
    for _ in range(steps):
        for i in rng.permutation(N):                     # asynchronous updates
            s[i] = 1 if W[i] @ s >= 0 else -1
    return s

target = patterns[0].copy()
cue = target.copy()
flip = rng.choice(N, size=N // 5, replace=False)          # flip 12 of the 64 bits
cue[flip] *= -1
recovered = recall(cue)
def overlap(a, b): return np.mean(a == b)
print(f"Hopfield associative memory ({P} patterns stored in {N} units):")
print(f"  corrupted cue matches the target: {overlap(cue, target)*100:.0f}% of bits")
print(f"  after recall, recovered matches : {overlap(recovered, target)*100:.0f}% of bits")
print("overlaps p . target of the stored patterns:", [int(p @ target) for p in patterns])
energy = -0.5 * float(target @ W @ target)
print(f"energy of the stored pattern E = -1/2 s^T W s: {energy:.4f}")
print("the network relaxed the noisy cue back to the stored pattern: memory as energy minimization.")

# %% [markdown]
# ## 24.1.1 The normalized energy and the partition function

# %%
import numpy as np
E = np.array([0.0, 1.0, 2.0, 0.5])
Z = np.exp(-E).sum()
p = np.exp(-E) / Z
p_double = np.exp(-2 * E); p_double = p_double / p_double.sum()
p_shift = np.exp(-(E + 3)); p_shift = p_shift / p_shift.sum()
print(f"Z = {Z:.4f}")
print("p =", " ".join(f"{v:.4f}" for v in p), "sum", f"{p.sum():.1f}")
print("doubled energies:", " ".join(f"{v:.4f}" for v in p_double))
print("max change after adding 3 to every energy:", f"{np.max(np.abs(p_shift - p)):.1f}")

# %% [markdown]
# Example 24-2. A restricted Boltzmann machine trained by persistent contrastive divergence on binarized digits.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression

def free_energy(v, W, b, c):
    # F(v) = -b.v - sum_j log(1 + exp(c_j + W_j.v)); np.logaddexp(0, a) is log(1 + e^a) without overflow
    return -(v @ b) - np.logaddexp(0, v @ W.T + c).sum(1)

# the tiny case worked by hand: two visible units, one hidden unit
W1, b1, c1 = np.array([[1.0, -1.0]]), np.array([0.5, -0.2]), np.array([0.1])
F1 = free_energy(np.array([[1.0, 0.0], [0.0, 1.0]]), W1, b1, c1)
print(f"tiny RBM: F(1,0) = {F1[0]:.4f}, F(0,1) = {F1[1]:.4f}, probability ratio {np.exp(F1[1] - F1[0]):.2f}")

digits = load_digits()
X = (digits.images.reshape(-1, 64) > 7).astype(float)          # on/off pixels; Example 24-3 binarizes alike
Xtr, Xte, ytr, yte = train_test_split(X, digits.target, test_size=0.25, random_state=0)   # 1,347 / 450
rbm = BernoulliRBM(n_components=64, learning_rate=0.05, batch_size=10, n_iter=30,
                   random_state=0).fit(Xtr)                    # persistent contrastive divergence
W, b, c = rbm.components_, rbm.intercept_visible_, rbm.intercept_hidden_
print(f"trained an RBM with {W.shape[1]} visible and {W.shape[0]} hidden units on {len(Xtr)} digits")

noise = np.random.default_rng(0).integers(0, 2, (len(Xte), 64)).astype(float)   # random bit images
print(f"mean free energy: held-out digits {free_energy(Xte, W, b, c).mean():.2f}, "
      f"random bits {free_energy(noise, W, b, c).mean():.2f} (lower is more probable)")
# score_samples is scikit-learn's pseudo-likelihood: it flips one random bit per image, not log p(v)
print(f"pseudo-likelihood (score_samples): held-out {rbm.score_samples(Xte).mean():.2f}, "
      f"random bits {rbm.score_samples(noise).mean():.2f}")

# generate: run 10 block Gibbs chains (v -> h -> v) for 1000 steps from random bits
v = noise[:10].copy()
for _ in range(1000):
    v = rbm.gibbs(v).astype(float)
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)          # a digit classifier to read the chains
print("classes of the 10 chains after 1000 Gibbs steps:", clf.predict(v).tolist())
hamming = lambda A: np.abs(A[:, None] - Xtr[None]).sum(2).min(1).mean()   # pixels that differ
print(f"pixels that differ from the nearest training digit: chains {hamming(v):.1f}, "
      f"random bits {hamming(noise[10:20]):.1f}")
for r in range(8):                                             # two chains side by side
    print("   " + "".join("#" if v[0, r * 8 + k] else "." for k in range(8))
          + "   " + "".join("#" if v[1, r * 8 + k] else "." for k in range(8)))

# %% [markdown]
# ## 24.2 Autoregressive generation beyond text: PixelCNN and WaveNet

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
rng = np.random.default_rng(0)

# binarize the 8-by-8 digits: pixel is on/off
X = (load_digits().images.reshape(-1, 64) > 7).astype(int)
Xtr, Xte = train_test_split(X, test_size=0.25, random_state=0)   # fit on 75%, hold out 25%
D = 64

models = []
for i in range(D):
    yi = Xtr[:, i]
    if yi.min() == yi.max():                              # constant pixel -> store its probability
        models.append(("const", float(yi.mean())))
    elif i == 0:
        models.append(("const", float(yi.mean())))        # pixel 0 has no predecessors: use its marginal
    else:
        clf = LogisticRegression(max_iter=200).fit(Xtr[:, :i], yi)
        models.append(("clf", clf))
print(f"trained {D} per-pixel conditionals on {len(Xtr)} digits (raster order); p(image) = product of them all")

def sample_image():
    s = np.zeros(D, int)
    for i in range(D):
        kind, m = models[i]
        p = m if kind == "const" else m.predict_proba(s[:i].reshape(1, -1))[0, 1]
        s[i] = int(rng.random() < p)
    return s

def show(img):
    for r in range(8):
        print("   " + "".join("#" if img[r * 8 + c] else "." for c in range(8)))

print("\ntwo digits generated pixel by pixel from the model:")
for _ in range(2):
    show(sample_image()); print()

def log_likelihood(img):
    ll = 0.0
    for i in range(D):
        kind, m = models[i]
        p = m if kind == "const" else m.predict_proba(img[:i].reshape(1, -1))[0, 1]
        p = min(max(p, 1e-6), 1 - 1e-6)
        ll += np.log(p if img[i] else 1 - p)
    return ll
train_ll = np.mean([log_likelihood(Xtr[k]) for k in range(200)])
test_ll = np.mean([log_likelihood(Xte[k]) for k in range(200)])
noise_ll = np.mean([log_likelihood(rng.integers(0, 2, D)) for _ in range(200)])
print(f"mean log likelihood: training digits {train_ll:.1f}, held-out digits {test_ll:.1f}, random noise {noise_ll:.1f}")

# %% [markdown]
# Example 24-4. A PixelCNN of masked convolutions on binarized digits.

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.set_num_threads(2)                                        # the thread count of the book's builds

X = (load_digits().images > 7).astype(np.float32)              # binarized 8-by-8 digits, as in Example 24-3
Xtr, Xte = train_test_split(X, test_size=0.25, random_state=0)  # the same 1,347 / 450 split
Xtr, Xte = torch.tensor(Xtr)[:, None], torch.tensor(Xte)[:, None]   # (n, 1 channel, 8, 8)

class MaskedConv(nn.Conv2d):
    """A convolution whose kernel is zeroed on every later raster position (mask A also on the center)."""
    def __init__(self, cin, cout, k, mask_type):
        super().__init__(cin, cout, k, padding=k // 2)
        m = torch.ones_like(self.weight); c = k // 2
        m[:, :, c, c + (mask_type == "B"):] = 0                 # center row: hide the center (A) or what follows it (B)
        m[:, :, c + 1:, :] = 0                                  # every row below the center
        self.register_buffer("mask", m)
    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding)

print("mask A, 5x5 (sees", int(MaskedConv(1, 1, 5, "A").mask.sum()), "pixels):")
print(MaskedConv(1, 1, 5, "A").mask[0, 0].int().numpy())
print("mask B, 3x3 (sees", int(MaskedConv(1, 1, 3, "B").mask.sum()), "positions, the center included):")
print(MaskedConv(1, 1, 3, "B").mask[0, 0].int().numpy())

torch.manual_seed(0)
net = nn.Sequential(MaskedConv(1, 32, 5, "A"), nn.ReLU(),
                    MaskedConv(32, 32, 3, "B"), nn.ReLU(),
                    MaskedConv(32, 32, 3, "B"), nn.ReLU(),
                    MaskedConv(32, 32, 3, "B"), nn.ReLU(),
                    nn.Conv2d(32, 1, 1))                        # one logit per pixel: p(x_i = 1 | earlier pixels)
n_all = sum(p.numel() for p in net.parameters())
n_open = sum(int(m.mask.sum()) + m.bias.numel() for m in net if isinstance(m, MaskedConv)) + 33
const = (Xtr.reshape(-1, 64).min(0).values == Xtr.reshape(-1, 64).max(0).values).numpy()
n_lr = sum(1 if (i == 0 or const[i]) else i + 1 for i in range(64))   # Example 24-3's rule for its 64 models
print(f"PixelCNN parameters {n_all:,} ({n_open:,} not masked); Example 24-3's 64 regressions hold {n_lr:,} numbers")

def loglik(x):                                                  # exact log p(x): sum of 64 Bernoulli log-probabilities
    return -F.binary_cross_entropy_with_logits(net(x), x, reduction="none").sum((1, 2, 3))

opt = torch.optim.Adam(net.parameters(), lr=2e-3)
for step in range(1, 2001):
    xb = Xtr[torch.randint(0, len(Xtr), (128,))]                # a minibatch of 128 training digits
    loss = -loglik(xb).mean()                                   # negative log likelihood per image
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (500, 1000, 2000):
        with torch.no_grad():
            print(f"after {step:4d} steps: mean log likelihood, training {loglik(Xtr[:200]).mean():.2f}, "
                  f"held-out Xte[:200] {loglik(Xte[:200]).mean():.2f}")

g = torch.Generator().manual_seed(0)
with torch.no_grad():
    noise = (torch.rand(200, 1, 8, 8, generator=g) < 0.5).float()
    print(f"200 random bit images: {loglik(noise).mean():.2f}")
    x = Xte[:1].clone(); y0 = net(x)                            # causality: flip pixel 36 (row 4, column 4)
    x[0, 0, 4, 4] = 1 - x[0, 0, 4, 4]
    moved = ((net(x) - y0).abs()[0, 0].flatten() > 0).nonzero().flatten()
    print(f"flipping pixel 36 changes the outputs at raster positions {moved.min().item()} to {moved.max().item()} only")
    s = torch.zeros(2, 1, 8, 8)                                 # sample: one network pass per pixel, 64 passes
    for i in range(8):
        for j in range(8):
            p = torch.sigmoid(net(s))[:, 0, i, j]
            s[:, 0, i, j] = (torch.rand(2, generator=g) < p).float()
for r in range(8):
    print("   " + "".join("#" if s[0, 0, r, k] else "." for k in range(8))
          + "   " + "".join("#" if s[1, 0, r, k] else "." for k in range(8)))

# %% [markdown]
# ## 24.2.1 A dilated stack's receptive field

# %%
dilations = [2 ** k for k in range(9)]
print("dilations", dilations)
print("receptive field", 1 + sum(dilations))
print("same depth, every dilation 1:", 1 + len(dilations))

# %% [markdown]
# Example 24-5. A gated, dilated causal stack against an undilated one on LibriSpeech audio.

# %%
import io
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Audio
torch.set_num_threads(2)                                        # the thread count of the book's builds

# 73 LibriSpeech recordings at 16 kHz; keep the raw bytes and decode them with soundfile
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(decode=False))
waves = [sf.read(io.BytesIO(a["bytes"]))[0].astype(np.float32) for a in ds["audio"]]
print(f"{len(waves)} recordings, {sum(len(w) for w in waves) / 16000:.1f} seconds")

def mulaw(x, mu=255):
    x = np.clip(x / max(1e-8, np.abs(x).max()), -1, 1)          # scale each recording into [-1, 1]
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)    # mu-law companding
    return ((y + 1) / 2 * mu + 0.5).astype(np.int64)            # quantize to the codes 0..255
print("mu-law codes of 0.01 and 0.5:", mulaw(np.array([0.01, 0.5, 1.0]))[:2].tolist())

codes = [torch.tensor(mulaw(w)) for w in waves]
train, held = codes[:60], codes[60:]                            # recordings 0-59 train, 60-72 held out
counts = torch.bincount(torch.cat(train), minlength=256).double(); p = counts / counts.sum()
print(f"entropy of the training codes {-(p[p > 0] * p[p > 0].log2()).sum().item():.3f} bits; a uniform guess 8 bits")

class Stack(nn.Module):
    """Gated residual causal convolutions of width 2 over mu-law codes, predicting the next code."""
    def __init__(self, dilations, C=32):
        super().__init__()
        self.dil = dilations
        self.emb = nn.Embedding(256, C)                         # each code becomes a 32-number vector
        self.filt = nn.ModuleList([nn.Conv1d(C, C, 2, dilation=d) for d in dilations])
        self.gate = nn.ModuleList([nn.Conv1d(C, C, 2, dilation=d) for d in dilations])
        self.res = nn.ModuleList([nn.Conv1d(C, C, 1) for d in dilations])
        self.out = nn.Sequential(nn.ReLU(), nn.Conv1d(C, C, 1), nn.ReLU(), nn.Conv1d(C, 256, 1))
        self.rf = 1 + sum(dilations)                            # receptive field, as Section 24.2.1 counts it
    def forward(self, x):                                       # x: (batch, time) codes
        h = self.emb(x).transpose(1, 2)
        for f, g, r, d in zip(self.filt, self.gate, self.res, self.dil):
            hp = F.pad(h, (d, 0))                               # pad on the left only: no peeking ahead
            h = h + r(torch.tanh(f(hp)) * torch.sigmoid(g(hp)))  # gated unit, residual connection
        return self.out(h)                                      # logits over the 256 next codes

def batch(gen, seqs, n=8, L=1024, rf=512):                      # 8 random crops: 512 context + 1024 targets
    xs = []
    for _ in range(n):
        s = seqs[int(torch.randint(0, len(seqs), (1,), generator=gen))]
        st = int(torch.randint(0, len(s) - L - rf - 1, (1,), generator=gen))
        xs.append(s[st:st + L + rf + 1])
    return torch.stack(xs)

def bits_per_sample(model, seqs, rf=512):                       # held-out cross-entropy in bits
    tot, n = 0.0, 0
    with torch.no_grad():
        for s in seqs:
            logits = model(s[None, :-1])
            tot += F.cross_entropy(logits[:, :, rf:], s[None, 1 + rf:], reduction="sum").item()
            n += len(s) - 1 - rf
    return tot / n / np.log(2)

for name, dil in (("dilated 1..256", [2 ** k for k in range(9)]), ("undilated", [1] * 9)):
    torch.manual_seed(0); gen = torch.Generator().manual_seed(0)
    m = Stack(dil); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for step in range(1500):
        xb = batch(gen, train)
        loss = F.cross_entropy(m(xb[:, :-1])[:, :, 512:], xb[:, 513:])   # predict each next code
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"{name:15s}: receptive field {m.rf:3d}, parameters {sum(q.numel() for q in m.parameters()):,}, "
          f"held-out {bits_per_sample(m, held):.3f} bits per sample")

# %% [markdown]
# ## 24.3.1 The evidence lower bound

# %%
import numpy as np
mu = np.array([0.2, -0.4]); logvar = np.array([-0.3, 0.5])
terms = mu**2 + np.exp(logvar) - logvar - 1                   # one term per latent dimension
print("per-dimension terms:", np.round(terms, 4))
print(f"KL = 0.5 * sum = {0.5 * terms.sum():.4f}")
print(f"code's form -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) = {-0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar)):.4f}")

# %%
import warnings; warnings.filterwarnings("ignore")
import tempfile, os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
torch.set_num_threads(2)                     # two threads, as in the book's builds: the printed losses depend on the count
torch.manual_seed(0); np.random.seed(0)

# a 5000-image subset of MNIST (28x28 handwritten digits), flattened to 784 pixels in [0,1]
root = os.path.join(tempfile.gettempdir(), "mnist_data")
mnist = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
X = mnist.data[:5000].reshape(-1, 784).float() / 255.0
LATENT = 16

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU())        # shared encoder body
        self.mu = nn.Linear(256, LATENT)                                # latent mean
        self.logvar = nn.Linear(256, LATENT)                            # latent log-variance
        self.dec = nn.Sequential(nn.Linear(LATENT, 256), nn.ReLU(),     # decoder back to 784 pixels
                                 nn.Linear(256, 784), nn.Sigmoid())
    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)         # reparameterization trick
        return self.dec(z), mu, logvar

vae = VAE(); opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
N, B = len(X), 128
for epoch in range(40):                                                 # 40 epochs of minibatches of 128
    perm = torch.randperm(N)
    for i in range(0, N, B):
        xb = X[perm[i:i+B]]
        recon, mu, logvar = vae(xb)
        recon_loss = nn.functional.binary_cross_entropy(recon, xb, reduction="sum") / len(xb)
        kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()) / len(xb)   # KL(q(z|x) || N(0,1))
        loss = recon_loss + kl                                          # negative ELBO: reconstruction + KL
        opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():                                                    # report full-set averages
    recon, mu, logvar = vae(X)
    recon_loss = nn.functional.binary_cross_entropy(recon, X, reduction="sum") / len(X)
    kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()) / len(X)
print(f"VAE trained on 5000 MNIST images: reconstruction loss {recon_loss.item():.1f}, KL {kl.item():.1f} (both per image)")

vae.eval()
with torch.no_grad():
    samples = vae.dec(torch.randn(25, LATENT))                          # decode 25 prior draws z ~ N(0, I)
    _, mu, logvar = vae(X)
print(f"latent stats after training: mean of mu ~ {mu.mean().item():+.2f}, std of mu ~ {mu.std().item():.2f}")
print("prior samples decode to recognizable digits (Figure 24-3); the KL made N(0, I) a valid region to sample.")

# %% [markdown]
# Example 24-7. A VAE against plain, denoising and sparse autoencoders on Example 24-6's MNIST subset.

# %%
import warnings; warnings.filterwarnings("ignore")
import tempfile, os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(2)                                        # the thread count of the book's builds

root = os.path.join(tempfile.gettempdir(), "mnist_data")       # the MNIST cache of Example 24-6
mnist = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
X = mnist.data[:5000].reshape(-1, 784).float() / 255.0         # Example 24-6's 5,000 training images
Xho = mnist.data[5000:6000].reshape(-1, 784).float() / 255.0   # 1,000 images no model trains on
clf = LogisticRegression(max_iter=300).fit(X.numpy(), mnist.targets[:5000].numpy())  # reads decoded digits
LATENT = 16

class Net(nn.Module):                                           # Example 24-6's layers, 784 -> 256 -> 16 -> 256 -> 784
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU())
        self.mu = nn.Linear(256, LATENT); self.logvar = nn.Linear(256, LATENT)
        self.dec = nn.Sequential(nn.Linear(LATENT, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

def train(kind, l1=0.0):
    torch.manual_seed(0); np.random.seed(0)                     # every model starts from the same weights
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(40):                                     # 40 epochs of minibatches of 128, as Example 24-6
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            xb = X[perm[i:i + 128]]
            inp = xb * (torch.rand_like(xb) > 0.5).float() if kind == "denoising" else xb   # zero about half
            h = net.enc(inp); mu, logvar = net.mu(h), net.logvar(h)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if kind == "VAE" else mu  # plain: z = mu
            loss = nn.functional.binary_cross_entropy(net.dec(z), xb, reduction="sum") / len(xb)  # clean target
            if kind == "VAE":
                loss = loss - 0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / len(xb)    # the KL term
            if l1:
                loss = loss + l1 * h.abs().sum() / len(xb)      # sparsity: L1 on the 256 hidden units
            opt.zero_grad(); loss.backward(); opt.step()
    return net

nets = {kind: train(kind, l1) for kind, l1 in
        (("VAE", 0), ("plain", 0), ("denoising", 0), ("sparse 1e-2", 1e-2), ("sparse 3e-2", 3e-2))}

print("codes of the 5,000 images, and 500 prior draws z ~ N(0, I) decoded:")
with torch.no_grad():
    for kind in ("VAE", "plain"):
        net = nets[kind]; mu = net.mu(net.enc(X))
        torch.manual_seed(1); z = torch.randn(500, LATENT); s = net.dec(z)   # the same draws for both
        cls = clf.predict(s.numpy()); top = clf.predict_proba(s.numpy()).max(1).mean()
        covered = sum(np.mean(cls == k) > 0.02 for k in range(10))
        near = torch.cdist(s, X).min(1).values.mean().item()   # distance to the nearest training image
        print(f"  {kind:5s}: code mean {mu.mean():+.4f} std {mu.std():.4f}, mean code length "
              f"{mu.norm(dim=1).mean():.2f} (prior draws {z.norm(dim=1).mean():.2f}); "
              f"classes {covered}/10, top class probability {top:.3f}, nearest real {near:.2f}")

    torch.manual_seed(2); keep = (torch.rand_like(Xho) > 0.5).float()   # zero about half of each held-out image
    for kind in ("plain", "denoising"):
        net = nets[kind]; rec = net.dec(net.mu(net.enc(Xho * keep)))
        print(f"held-out images with half the pixels zeroed, {kind:9s}: reconstruction MSE "
              f"{((rec - Xho) ** 2).mean():.4f}")

    for kind in ("plain", "sparse 1e-2", "sparse 3e-2"):
        h = nets[kind].enc(X)
        print(f"{kind:11s}: share of the 256 hidden units above 0.01, per image {(h > 0.01).float().mean():.3f}")

# %% [markdown]
# ## 24.3.2 Posterior collapse, and VQ-VAE

# %%
import numpy as np
from sklearn.datasets import load_digits
rng = np.random.default_rng(0)
rows = load_digits().images.reshape(-1, 64)[:32] / 16.0
code = rng.random((8, 64))
dist = ((rows[:, None] - code[None]) ** 2).sum(2)
idx = dist.argmin(1)
mse0 = dist[np.arange(len(rows)), idx].mean()
for k in range(8):
    chosen = rows[idx == k]
    if len(chosen):
        code[k] = chosen.mean(0)
dist2 = ((rows[:, None] - code[None]) ** 2).sum(2)
mse1 = dist2[np.arange(len(rows)), dist2.argmin(1)].mean()
print(f"mean squared distance {mse0:.4f} -> {mse1:.4f}; codes used {len(set(idx.tolist()))} of 8")

# %% [markdown]
# Example 24-9. A VQ-VAE with the straight-through estimator, from two codebook starts.

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.set_num_threads(2)                                        # the thread count of the book's builds

# the quantization worked by hand: one encoder output against three codes
book = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
d2 = torch.cdist(torch.tensor([[0.3, 0.9]]), book) ** 2
print("squared distances", [round(v, 2) for v in d2[0].tolist()], "-> code", int(d2.argmin()))

X = load_digits().images.reshape(-1, 64) / 16.0                # pixels in [0, 1]
Xtr, Xte = train_test_split(X, test_size=0.25, random_state=0)  # Example 24-3's split
Xtr, Xte = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
P, Dm, K, BETA = 4, 4, 16, 0.25          # 4 positions of 4 numbers each, 16 codes, the commitment weight

def train_vqvae(init):
    torch.manual_seed(0)
    enc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, P * Dm))
    dec = nn.Sequential(nn.Linear(P * Dm, 128), nn.ReLU(), nn.Linear(128, 64), nn.Sigmoid())
    with torch.no_grad():
        z0 = enc(Xtr).reshape(-1, Dm)                           # the untrained encoder's outputs
    start = z0[torch.randperm(len(z0))[:K]].clone() if init == "data" else torch.randn(K, Dm)
    codebook = nn.Parameter(start)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + [codebook], lr=2e-3)
    def quantize(x):
        ze = enc(x).reshape(-1, Dm)                             # one row per position
        idx = torch.cdist(ze, codebook).argmin(1)               # nearest code; a tie goes to the lower index
        e = F.embedding(idx, codebook)                          # the lookup codebook[idx], reproducible backward
        return ze, e, ze + (e - ze).detach(), idx               # straight through: forward e, gradient to ze
    for step in range(3000):                                    # full batch steps
        ze, e, zq, idx = quantize(Xtr)
        rec = dec(zq.reshape(len(Xtr), -1))
        loss = (F.mse_loss(rec, Xtr) + F.mse_loss(e, ze.detach())   # reconstruction + codebook term
                + BETA * F.mse_loss(ze, e.detach()))                # + commitment term
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ze, e, zq, idx = quantize(Xte)
        mse = F.mse_loss(dec(zq.reshape(len(Xte), -1)), Xte).item()
        p = torch.bincount(idx, minlength=K).float() / len(idx)   # how often each code is chosen
        perplexity = torch.exp(-(p[p > 0] * p[p > 0].log()).sum()).item()
    print(f"codebook start {init:6s}: held-out reconstruction MSE {mse:.4f}; codes used "
          f"{int((p > 0).sum())} of {K}; perplexity {perplexity:.2f}; {P} tokens per image")

train_vqvae("random")                                           # 16 standard normal vectors
train_vqvae("data")                                             # 16 encoder outputs of training digits

# %% [markdown]
# ## 24.4 Normalizing flows

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
torch.manual_seed(0); np.random.seed(0)

data = make_moons(1500, noise=0.08)[0]
data = torch.tensor((data - data.mean(0)) / data.std(0), dtype=torch.float32)

class Flow(nn.Module):
    def __init__(self, n_layers=6):
        super().__init__()
        self.nets = nn.ModuleList([nn.Sequential(nn.Linear(1, 32), nn.ReLU(),
                                                 nn.Linear(32, 2)) for _ in range(n_layers)])
        self.swap = [i % 2 for i in range(n_layers)]         # layer i conditions on coordinate swap[i]
    def x_to_z(self, x):
        z = x; logdet = torch.zeros(len(x))
        for net, sw in zip(self.nets, self.swap):
            a, b = z[:, sw], z[:, 1 - sw]                    # a = conditioner, b = transformed
            out = net(a[:, None]); log_s = torch.tanh(out[:, 0]); t = out[:, 1]
            b = (b - t) * torch.exp(-log_s)                  # inverse affine on b
            logdet = logdet - log_s
            z = torch.stack([a, b], 1) if sw == 0 else torch.stack([b, a], 1)  # each coordinate back in its own column
        return z, logdet
    def z_to_x(self, z):                                     # sampling direction (invert)
        x = z
        for net, sw in reversed(list(zip(self.nets, self.swap))):
            a, b = x[:, sw], x[:, 1 - sw]
            out = net(a[:, None]); log_s = torch.tanh(out[:, 0]); t = out[:, 1]
            b = b * torch.exp(log_s) + t                     # undo the affine map on b
            x = torch.stack([a, b], 1) if sw == 0 else torch.stack([b, a], 1)
        return x

flow = Flow(); opt = torch.optim.Adam(flow.parameters(), lr=5e-3)

def log_prob(x):
    z, logdet = flow.x_to_z(x)
    log_pz = -0.5 * (z ** 2).sum(1) - np.log(2 * np.pi)      # log N(z; 0, I) in 2D
    return log_pz + logdet

for epoch in range(600):
    loss = -log_prob(data).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f"flow trained: mean log likelihood on data = {log_prob(data).mean().item():.2f}")

flow.eval()
with torch.no_grad():
    off = torch.randn(1500, 2) * 1.5                          # random points not on the moons
    print(f"mean log likelihood: real 'moons' data {log_prob(data).mean():.2f}  vs  random points {log_prob(off).mean():.2f}")
    z_data, _ = flow.x_to_z(data)                             # invertibility check: x -> z -> x
    round_trip = (flow.z_to_x(z_data) - data).abs().max()
    print(f"round-trip max |z_to_x(x_to_z(x)) - x| = {round_trip:.1e}")
    print(f"both coordinates are transformed: max |z[:, 0] - x[:, 0]| = {(z_data[:, 0] - data[:, 0]).abs().max():.3f}")
    z_base = torch.randn(1500, 2)
    samples = flow.z_to_x(z_base).numpy()
    real = data.numpy()
    print(f"generated samples: mean {samples.mean(0).round(2)}, std {samples.std(0).round(2)}")
    print(f"real data:         mean {real.mean(0).round(2)}, std {real.std(0).round(2)}")
    # how moon-like are the samples? mean distance from each point to its nearest real point
    def nearest(pts): return np.sqrt(((pts[:, None] - real[None]) ** 2).sum(2)).min(1).mean()
    print(f"mean distance to the nearest real point: flow samples {nearest(samples):.3f}, base draws z {nearest(z_base.numpy()):.3f}")

# %% [markdown]
# ## 24.5 Generative adversarial networks

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
torch.manual_seed(0); rng = np.random.default_rng(0)

# 8 Gaussian 'modes' arranged in a ring
angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
centers = np.c_[np.cos(angles), np.sin(angles)] * 2
def real_batch(n):
    c = centers[rng.integers(0, 8, n)]
    return torch.tensor(c + rng.normal(0, 0.1, (n, 2)), dtype=torch.float32)

Z = 16
G = nn.Sequential(nn.Linear(Z, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
D = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9))
bce = nn.BCEWithLogitsLoss()

for step in range(6000):
    real = real_batch(128)
    z = torch.randn(128, Z); fake = G(z)
    # train D: real -> 1, fake -> 0
    lossD = bce(D(real), torch.ones(128, 1)) + bce(D(fake.detach()), torch.zeros(128, 1))
    optD.zero_grad(); lossD.backward(); optD.step()
    # train G: make D output 1 on fakes (nonsaturating loss)
    z = torch.randn(128, Z); fake = G(z)
    lossG = bce(D(fake), torch.ones(128, 1))
    optG.zero_grad(); lossG.backward(); optG.step()
    if step % 2000 == 0:
        with torch.no_grad():
            d_real = torch.sigmoid(D(real)).mean().item(); d_fake = torch.sigmoid(D(fake)).mean().item()
        print(f"  step {step:4d}: D(real)={d_real:.2f} D(fake)={d_fake:.2f} (0.5/0.5 = perfect game)")

with torch.no_grad():
    gen = G(torch.randn(4000, Z)).numpy()
dists = np.linalg.norm(gen[:, None] - centers[None], axis=2)   # distance to each mode center
nearest = dists.argmin(1)
covered = sum(1 for m in range(8) if np.mean(nearest == m) > 0.02)   # mode has >2% of samples
print(f"\ngenerator covers {covered} of 8 modes")
counts = np.bincount(nearest, minlength=8)
print(f"  samples per mode: {counts.tolist()}")
if covered < 8:
    print("  MODE COLLAPSE: the generator abandoned some modes to focus on fooling D on a few -")
    print("  a hallmark GAN failure. Fixes: minibatch discrimination, WGAN-GP, unrolled GANs, spectral norm.")
else:
    print("  all modes covered (a well-balanced run); collapse is what happens when this balance breaks.")

print("\nAll GAN checks ran.")

# %% [markdown]
# ## 24.5.1 Conditional GANs: controllable generation

# %% [markdown]
# Example 24-12. A conditional GAN on digits, scored by how often it draws the requested digit.

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(2)                                        # the thread count of the book's builds

digits = load_digits()
X01 = torch.tensor(digits.images.reshape(-1, 64) / 16.0, dtype=torch.float32)   # pixels in [0, 1]
Y = torch.tensor(digits.target)
clf = LogisticRegression(max_iter=500).fit(X01.numpy(), digits.target)   # the lab's digit classifier
Z = 16
bce = nn.BCEWithLogitsLoss()

def train_cgan(steps, width=128):
    torch.manual_seed(0)
    Gc = nn.Sequential(nn.Linear(Z + 10, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                       nn.Linear(width, 64), nn.Sigmoid())      # G(z, y): noise and a one-hot label in
    Dc = nn.Sequential(nn.Linear(64 + 10, width), nn.LeakyReLU(0.2), nn.Linear(width, width),
                       nn.LeakyReLU(0.2), nn.Linear(width, 1))  # D(x, y): an image and a label in
    og = torch.optim.Adam(Gc.parameters(), lr=2e-4, betas=(0.5, 0.9))
    od = torch.optim.Adam(Dc.parameters(), lr=2e-4, betas=(0.5, 0.9))
    for _ in range(steps):
        i = torch.randint(0, len(X01), (128,))
        real, yr = X01[i], F.one_hot(Y[i], 10).float()          # real digits with their true labels
        yf = F.one_hot(torch.randint(0, 10, (128,)), 10).float()   # labels requested from G
        fake = Gc(torch.cat([torch.randn(128, Z), yf], 1))
        ld = (bce(Dc(torch.cat([real, yr], 1)), torch.ones(128, 1))
              + bce(Dc(torch.cat([fake.detach(), yf], 1)), torch.zeros(128, 1)))
        od.zero_grad(); ld.backward(); od.step()
        yf = F.one_hot(torch.randint(0, 10, (128,)), 10).float()
        lg = bce(Dc(torch.cat([Gc(torch.cat([torch.randn(128, Z), yf], 1)), yf], 1)), torch.ones(128, 1))
        og.zero_grad(); lg.backward(); og.step()                # nonsaturating generator loss
    want = torch.arange(10).repeat_interleave(50)               # ask for 50 samples of every digit
    with torch.no_grad():
        s = Gc(torch.cat([torch.randn(500, Z), F.one_hot(want, 10).float()], 1)).numpy()
    return s, want.numpy()

s, want = train_cgan(10_000)
pred = clf.predict(s)
print(f"share of the 500 samples the classifier reads as the requested digit {np.mean(pred == want):.3f}")
print("per requested digit 0-9:", [round(float(np.mean(pred[want == k] == k)), 2) for k in range(10)])
levels = " .:-=+*#"                                             # 8 intensity levels, dark -> bright
for half in (range(0, 5), range(5, 10)):                        # the first sample of each requested digit
    for r in range(8):
        print("  " + "  ".join("".join(levels[min(7, int(s[50 * k, 8 * r + c] * 8))] for c in range(8))
                               for k in half))
    print()

# %% [markdown]
# ## 24.5.2 The gradient penalty on real digits

# %%
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
torch.manual_seed(0)
X = torch.tensor(load_digits().images.reshape(-1, 64) / 16.0, dtype=torch.float32)
real, x_tilde = X[:128].detach(), X[128:256].detach()   # x_tilde stands in for generated samples
u = torch.rand(128, 1)
xhat = (u * real + (1 - u) * x_tilde).detach().requires_grad_(True)
critic = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
grad = torch.autograd.grad(critic(xhat).sum(), xhat)[0]
gnorm = grad.reshape(128, -1).norm(dim=1)
penalty = ((gnorm - 1) ** 2).mean()
print(f"penalty {penalty.item():.4f}  mean gradient norm {gnorm.mean().item():.4f}")

# %% [markdown]
# Example 24-14. A Wasserstein GAN with gradient penalty against Example 24-11's GAN on the ring of Gaussians.

# %%
import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(2)                                        # the thread count of the book's builds

angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
centers = np.c_[np.cos(angles), np.sin(angles)] * 2            # Example 24-11's ring of eight modes
Z = 16

def train_ring(kind, steps=6000, lam=10.0, n_critic=5):
    torch.manual_seed(0); rng = np.random.default_rng(0)        # Example 24-11's seeds
    def real_batch(n):
        c = centers[rng.integers(0, 8, n)]
        return torch.tensor(c + rng.normal(0, 0.1, (n, 2)), dtype=torch.float32)
    G = nn.Sequential(nn.Linear(Z, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    D = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    lr = 2e-4 if kind == "gan" else 1e-4
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    bce = nn.BCEWithLogitsLoss()
    for step in range(steps):
        if kind == "gan":                                       # Example 24-11's loop, unchanged
            real = real_batch(128); fake = G(torch.randn(128, Z))
            lossD = bce(D(real), torch.ones(128, 1)) + bce(D(fake.detach()), torch.zeros(128, 1))
            optD.zero_grad(); lossD.backward(); optD.step()
            lossG = bce(D(G(torch.randn(128, Z))), torch.ones(128, 1))
        else:                                                   # WGAN-GP: a critic with no sigmoid
            for _ in range(n_critic):
                real = real_batch(128); fake = G(torch.randn(128, Z)).detach()
                u = torch.rand(128, 1)
                xh = (u * real + (1 - u) * fake).requires_grad_(True)   # points between real and fake
                g = torch.autograd.grad(D(xh).sum(), xh, create_graph=True)[0]
                gp = ((g.norm(dim=1) - 1) ** 2).mean()          # gradient penalty
                d_fake, d_real = D(fake).mean(), D(real).mean()
                lossD = d_fake - d_real + lam * gp              # E[D(fake)] - E[D(real)] + lambda * penalty
                optD.zero_grad(); lossD.backward(); optD.step()
            lossG = -D(G(torch.randn(128, Z))).mean()           # G raises the critic's score of fakes
        optG.zero_grad(); lossG.backward(); optG.step()
    with torch.no_grad():
        gen = G(torch.randn(4000, Z)).numpy()
    dist = np.linalg.norm(gen[:, None] - centers[None], axis=2)  # distance to each mode center
    counts = np.bincount(dist.argmin(1), minlength=8)
    covered = int((counts / len(gen) > 0.02).sum())             # Example 24-11's rule: more than 2 percent
    print(f"{kind:7s}: covers {covered} of 8 modes; samples per mode {counts.tolist()}; "
          f"within 0.3 of a center {np.mean(dist.min(1) < 0.3):.3f}")
    if kind != "gan":
        print(f"         last critic step: E[D(real)] - E[D(fake)] = {(d_real - d_fake).item():.3f}, "
              f"penalty {gp.item():.4f}")

train_ring("gan")                                               # Example 24-11 again, for the comparison
train_ring("WGAN-GP")

# %% [markdown]
# ## 24.6 Diffusion models

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
torch.manual_seed(0)

X = torch.tensor(load_digits().images.reshape(-1, 64) / 16.0 * 2 - 1, dtype=torch.float32)  # [-1,1]

T = 50
t_lin = torch.linspace(0, 1, T + 1)
abar = torch.cos((t_lin + 0.008) / 1.008 * np.pi / 2) ** 2
abar = abar / abar[0]                                     # the cosine curve with value 1 at t = 0
print(f"unclipped last step: beta = {1 - abar[-1] / abar[-2]:.6f}, 1/sqrt(alpha) = {torch.sqrt(abar[-2] / abar[-1]):,.0f}")
beta = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)        # every beta in (0, 1) (Nichol and Dhariwal, 2021)
alpha = 1 - beta
abar = torch.cumprod(alpha, 0)                            # abar for steps 1..T at indices 0..T-1
print(f"clipped last step:   beta = {beta[-1]:.6f}, 1/sqrt(alpha) = {1 / torch.sqrt(alpha[-1]):.1f}")
print(f"forward process signal fraction abar_t: step 1={abar[0]:.2f}, mid={abar[T//2]:.2f}, T={abar[-1]:.3f}")
print("by step T almost all signal is gone -> the data has become noise we know how to sample.")

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64 + 1, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, 64))
    def forward(self, x, t):
        return self.net(torch.cat([x, t[:, None].float() / T], dim=1))

model = Denoiser(); opt = torch.optim.Adam(model.parameters(), lr=2e-3)
for epoch in range(4000):
    idx = torch.randint(0, len(X), (256,)); x0 = X[idx]
    t = torch.randint(0, T, (256,))
    eps = torch.randn_like(x0)
    xt = torch.sqrt(abar[t])[:, None] * x0 + torch.sqrt(1 - abar[t])[:, None] * eps   # noise in one shot
    loss = nn.functional.mse_loss(model(xt, t), eps)        # learn to predict the noise
    opt.zero_grad(); loss.backward(); opt.step()
print(f"\ndenoiser trained: final noise-prediction MSE = {loss.item():.3f}")

@torch.no_grad()
def sample(n=2):
    x = torch.randn(n, 64)                                  # pure noise at step T
    for ti in reversed(range(T)):
        t = torch.full((n,), ti)
        eps = model(x, t)
        x0_hat = ((x - torch.sqrt(1 - abar[ti]) * eps) / torch.sqrt(abar[ti])).clamp(-1, 1)   # clean guess, in the data range
        abar_prev = abar[ti - 1] if ti > 0 else torch.tensor(1.0)
        mean = (torch.sqrt(abar_prev) * beta[ti] / (1 - abar[ti])) * x0_hat + (torch.sqrt(alpha[ti]) * (1 - abar_prev) / (1 - abar[ti])) * x
        var = (1 - abar_prev) / (1 - abar[ti]) * beta[ti]    # posterior variance, tilde beta_t
        x = mean + (torch.sqrt(var) * torch.randn_like(x) if ti > 0 else 0)
    return x

def show(vec):
    img = vec.reshape(8, 8)
    levels = " .:-=+*#"                                   # 8 intensity levels, dark -> bright
    for r in range(8):
        print("   " + "".join(levels[min(7, max(0, int((img[r, c] + 1) / 2 * 8)))] for c in range(8)))
print("\ntwo digits generated by denoising from pure noise:")
for s in sample(2).numpy():
    show(s); print()

print("All diffusion checks ran.")

# %% [markdown]
# ## 24.6.1 Faster sampling: DDIM, flow matching, and consistency models

# %%
# model, cosine abar schedule, training, and show() are the same as Example 24-15
@torch.no_grad()
def ddim(x, steps=10):
    seq = torch.linspace(T - 1, 0, steps).round().long().tolist()   # short deterministic schedule
    for i, ti in enumerate(seq):
        eps = model(x, torch.full((x.shape[0],), ti))
        x0_pred = ((x - torch.sqrt(1 - abar[ti]) * eps) / torch.sqrt(abar[ti])).clamp(-1, 1)
        if i == len(seq) - 1:
            x = x0_pred                                     # last step: land on the clean estimate
        else:
            a_next = abar[seq[i + 1]]
            x = torch.sqrt(a_next) * x0_pred + torch.sqrt(1 - a_next) * eps   # NO added noise
    return x

torch.manual_seed(7); z = torch.randn(1, 64)
print("\nDDIM, 10 deterministic steps (Example 24-15 used 50 stochastic steps):")
show(ddim(z.clone(), steps=10)[0].numpy())
a = ddim(z.clone(), steps=10); b = ddim(z.clone(), steps=10)
print(f"\nsame z run twice, max abs difference = {(a - b).abs().max().item():.2e} (deterministic)")
print(f"network evaluations: DDIM 10 vs {T} for the stochastic sampler ({T // 10}x fewer).")

# %%
import numpy as np
eps_uncond = np.array([0.2, -0.1])                  # the prediction without the condition
eps_cond = np.array([0.5, 0.3])                     # the prediction with the condition
for w in (0, 1, 3, 7.5):
    e = eps_uncond + w * (eps_cond - eps_uncond)    # the guided noise prediction
    print(f"w = {w}: ({e[0]:.2f}, {e[1]:.2f})")

# %% [markdown]
# Example 24-17. Classifier-free guidance on digits at four guidance weights.

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(2)                                        # the thread count of the book's builds

digits = load_digits()
Xd = torch.tensor(digits.images.reshape(-1, 64) / 16.0 * 2 - 1, dtype=torch.float32)   # [-1, 1]
Yd = torch.tensor(digits.target)
X01d = ((Xd + 1) / 2).numpy()
clf_d = LogisticRegression(max_iter=500).fit(X01d, digits.target)   # the lab's digit classifier

def quality(s01):                                               # the lab's two numbers for samples in [0, 1]
    s = np.clip(s01, 0, 1)
    near = np.sqrt(((s[:, None] - X01d[None]) ** 2).sum(2)).min(1).mean()   # nearest real digit
    cls = clf_d.predict(s)
    return round(float(near), 2), sum(int(np.mean(cls == k) > 0.02) for k in range(10))

def show_row(rows01):                                           # several 8-by-8 digits side by side
    levels = " .:-=+*#"
    for r in range(8):
        print("   " + "   ".join("".join(levels[min(7, max(0, int(v[8 * r + c] * 8)))] for c in range(8))
                                for v in rows01))

T = 50                                                          # Example 24-15's schedule, line for line
t_lin = torch.linspace(0, 1, T + 1)
abar = torch.cos((t_lin + 0.008) / 1.008 * np.pi / 2) ** 2
abar = abar / abar[0]
beta = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
alpha = 1 - beta
abar = torch.cumprod(alpha, 0)

torch.manual_seed(0)
cnet = nn.Sequential(nn.Linear(64 + 1 + 10, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                     nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 64))   # input: x_t, t / T, one-hot label
opt = torch.optim.Adam(cnet.parameters(), lr=2e-3)
for step in range(6000):
    i = torch.randint(0, len(Xd), (256,)); x0 = Xd[i]
    y = F.one_hot(Yd[i], 10).float() * (torch.rand(256, 1) > 0.1).float()   # zeros: label dropped (~10%)
    t = torch.randint(0, T, (256,)); eps = torch.randn_like(x0)
    xt = torch.sqrt(abar[t])[:, None] * x0 + torch.sqrt(1 - abar[t])[:, None] * eps
    loss = F.mse_loss(cnet(torch.cat([xt, t[:, None] / T, y], 1)), eps)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"conditional denoiser trained: final noise-prediction MSE {loss.item():.3f}")

@torch.no_grad()
def sample_guided(want, w, gen):                                # Example 24-15's sampler with a guided eps
    n = len(want); x = torch.randn(n, 64, generator=gen)
    yc = F.one_hot(want, 10).float(); yu = torch.zeros_like(yc)  # the label, and "no label"
    for ti in reversed(range(T)):
        tt = torch.full((n, 1), ti / T)
        e_u = cnet(torch.cat([x, tt, yu], 1)); e_c = cnet(torch.cat([x, tt, yc], 1))
        e = e_u + w * (e_c - e_u)                               # classifier-free guidance
        x0_hat = ((x - torch.sqrt(1 - abar[ti]) * e) / torch.sqrt(abar[ti])).clamp(-1, 1)
        abar_prev = abar[ti - 1] if ti > 0 else torch.tensor(1.0)
        mean = (torch.sqrt(abar_prev) * beta[ti] / (1 - abar[ti])) * x0_hat \
            + (torch.sqrt(alpha[ti]) * (1 - abar_prev) / (1 - abar[ti])) * x
        var = (1 - abar_prev) / (1 - abar[ti]) * beta[ti]
        x = mean + (torch.sqrt(var) * torch.randn(n, 64, generator=gen) if ti > 0 else 0)
    return ((x + 1) / 2).numpy()

want = torch.arange(10).repeat_interleave(50)                   # 50 samples of every digit
first3 = []
for w in (0.0, 1.0, 3.0, 7.5):
    s = sample_guided(want, w, torch.Generator().manual_seed(0))   # the same noise at every w
    hit = np.mean(clf_d.predict(np.clip(s, 0, 1)) == want.numpy())
    spread = np.mean([np.sqrt(((a[:, None] - a[None]) ** 2).sum(2)).sum() / (50 * 49)
                      for a in (s[want.numpy() == k] for k in range(10))])   # mean distance, distinct pairs
    print(f"w = {w:3.1f}: requested digit {hit:.3f}; nearest real, classes {quality(s)}; "
          f"distance between samples of one digit {spread:.2f}")
    first3.append(s[150])                                       # the first sample requested as a 3
print("the first sample requested as a 3, at w = 0, 1, 3, 7.5:")
show_row(first3)

# %% [markdown]
# Example 24-18. Consistency distillation of the Example 24-15 denoiser into a one-step sampler.

# %%
import copy
import torch
import torch.nn.functional as F
torch.set_num_threads(2)                                        # the thread count of the book's builds
# continues the session: model, Denoiser, X, T and abar from Example 24-15, ddim from Example 24-16,
# quality and show_row from Example 24-17

grid = torch.linspace(T - 1, 0, 10).round().long().flip(0)     # Example 24-16's 10 levels, low to high
print("grid of noise levels", grid.tolist())

@torch.no_grad()
def teacher_step(x, n, n_next):                                 # one DDIM step of the trained denoiser
    a = abar[n][:, None]; e = model(x, n)
    x0 = ((x - torch.sqrt(1 - a) * e) / torch.sqrt(a)).clamp(-1, 1)
    a2 = abar[n_next][:, None]
    return torch.sqrt(a2) * x0 + torch.sqrt(1 - a2) * e

ztest = torch.randn(500, 64, generator=torch.Generator().manual_seed(1))   # pure noise at index 49
n49 = torch.full((500,), T - 1)
score = lambda x: quality(((x.clamp(-1, 1) + 1) / 2).numpy())

torch.manual_seed(1)
S = Denoiser()                                                  # the student: same layers, predicts x0 itself
opt = torch.optim.Adam(S.parameters(), lr=1e-3)
for step in range(3000):                                        # warm start: plain denoising at the grid levels
    x0 = X[torch.randint(0, len(X), (256,))]; n = grid[torch.randint(0, 10, (256,))]
    a = abar[n][:, None]
    loss = F.mse_loss(S(torch.sqrt(a) * x0 + torch.sqrt(1 - a) * torch.randn_like(x0), n), x0)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    print("warm start, one step from noise: nearest real, classes", score(S(ztest, n49)))

S_target = copy.deepcopy(S)                                     # the target network: an average of the student
for p in S_target.parameters():
    p.requires_grad_(False)
for step in range(6000):                                        # consistency distillation
    x0 = X[torch.randint(0, len(X), (256,))]
    k = torch.randint(0, 9, (256,)); n_hi, n_lo = grid[k + 1], grid[k]     # adjacent levels
    a = abar[n_hi][:, None]
    x_hi = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * torch.randn_like(x0)
    with torch.no_grad():
        x_lo = teacher_step(x_hi, n_hi, n_lo)                   # the teacher moves one grid step
        target = torch.where((k == 0)[:, None], x_lo, S_target(x_lo, n_lo))   # boundary: f(x, 0) = x
    loss = F.mse_loss(S(x_hi, n_hi), target)                    # same trajectory, same answer
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        for pt, ps in zip(S_target.parameters(), S.parameters()):
            pt.mul_(0.95).add_(ps, alpha=0.05)                  # exponential moving average, mu = 0.95

with torch.no_grad():
    one = S(ztest, n49).clamp(-1, 1)                            # one network pass
    a = abar[22]
    renoised = torch.sqrt(a) * one + torch.sqrt(1 - a) * torch.randn(500, 64, generator=torch.Generator().manual_seed(2))
    two = S(renoised, torch.full((500,), 22))                   # noise again to index 22, one more pass
    print("consistency model, 1 step: nearest real, classes", score(one))
    print("consistency model, 2 steps: nearest real, classes", score(two))
    print("teacher DDIM, 10 steps   : nearest real, classes", score(ddim(ztest.clone(), steps=10)))
    print("the first three one-step samples:")
    show_row(((one[:3] + 1) / 2).numpy())

# %% [markdown]
# ## 24.6.2 The straight-line velocity

# %%
import numpy as np
from sklearn.datasets import load_digits
rng = np.random.default_rng(0)
x0 = (load_digits().images[0].reshape(-1) > 7).astype(float)
eps = rng.normal(size=x0.shape)
t = 0.5
xt = (1 - t) * x0 + t * eps
v = eps - x0
print(f"x_t[0] = {xt[0]:.4f}   ||v|| = {np.linalg.norm(v):.4f}")

# %% [markdown]
# Example 24-19. Flow matching on the straight path, sampled with Euler steps.

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(2)                                        # the thread count of the book's builds

digits = load_digits()
Xf = torch.tensor(digits.images.reshape(-1, 64) / 16.0 * 2 - 1, dtype=torch.float32)   # [-1, 1]
Xf01 = ((Xf + 1) / 2).numpy()
clf_f = LogisticRegression(max_iter=500).fit(Xf01, digits.target)   # the lab's digit classifier
def quality_f(s01):                                             # the lab's two numbers for samples in [0, 1]
    s = np.clip(s01, 0, 1)
    near = np.sqrt(((s[:, None] - Xf01[None]) ** 2).sum(2)).min(1).mean()
    cls = clf_f.predict(s)
    return round(float(near), 2), sum(int(np.mean(cls == k) > 0.02) for k in range(10))

torch.manual_seed(0)
vnet = nn.Sequential(nn.Linear(64 + 1, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                     nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 64))   # Example 24-15's layers, input x_t and t
opt = torch.optim.Adam(vnet.parameters(), lr=2e-3)
for step in range(6000):
    x0 = Xf[torch.randint(0, len(Xf), (256,))]
    eps = torch.randn_like(x0); t = torch.rand(256, 1)          # t uniform on (0, 1)
    xt = (1 - t) * x0 + t * eps                                 # a point on the straight segment
    loss = F.mse_loss(vnet(torch.cat([xt, t], 1)), eps - x0)    # regress the velocity v = eps - x0
    opt.zero_grad(); loss.backward(); opt.step()
print(f"velocity network trained: final MSE {loss.item():.3f}")

@torch.no_grad()
def euler(steps, n=500):                                        # integrate from t = 1 (noise) to t = 0 (data)
    x = torch.randn(n, 64, generator=torch.Generator().manual_seed(1)); dt = 1.0 / steps
    for k in range(steps):
        t = 1.0 - k * dt
        x = x - dt * vnet(torch.cat([x, torch.full((n, 1), t)], 1))   # x <- x - dt v
    return ((x.clamp(-1, 1) + 1) / 2).numpy()

for steps in (1, 2, 5, 10, 50):
    print(f"Euler, {steps:2d} steps: nearest real, classes {quality_f(euler(steps))}")

# %% [markdown]
# ## 24.7 Text-to-image systems

# %%
import torch
from transformers import CLIPModel, CLIPProcessor
torch.set_num_threads(2)                                  # the thread count of the book's builds
torch.manual_seed(0)

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")    # loads to the CPU
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tok = proc(text=["a photo of a flower"], return_tensors="pt")
with torch.no_grad():
    text_tokens = clip.text_model(**tok).last_hidden_state[0]      # one 512-value vector per token

feat = torch.randn(64, 32)          # stand-in denoiser features: an 8 x 8 grid, 32 channels each
d = 32                              # width of the attention
WQ = torch.randn(32, d) / 32 ** 0.5           # untrained projections: they show the wiring only
WK = torch.randn(512, d) / 512 ** 0.5
WV = torch.randn(512, d) / 512 ** 0.5
Q = feat @ WQ                       # queries from the image positions
K, V = text_tokens @ WK, text_tokens @ WV     # keys and values from the text tokens
weights = torch.softmax(Q @ K.T / d ** 0.5, dim=-1)               # 64 positions x tokens
out = weights @ V
print("tokens:", proc.tokenizer.convert_ids_to_tokens(tok["input_ids"][0]))
print(f"text tokens {tuple(text_tokens.shape)}, image positions {tuple(feat.shape)}")
print(f"weights {tuple(weights.shape)}, every row sums to 1: {torch.allclose(weights.sum(1), torch.ones(64))}")
print(f"output {tuple(out.shape)}: one text summary for each of the 64 positions")

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
from PIL import Image
from sklearn.datasets import load_sample_images
from transformers import CLIPModel, CLIPProcessor
torch.manual_seed(0)

# REAL CLIP: matching text-image pairs are close in a shared embedding space.
# openai/clip-vit-base-patch32; Radford et al. (2021) trained CLIP on 400 million image-text pairs.
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
imgs = load_sample_images().images                       # 'china' (a temple) and 'flower'
pil = [Image.fromarray(a) for a in imgs]
names = ["china (a temple)", "flower"]
prompts = ["a photo of a flower", "a photo of a temple building", "a photo of a cat"]
inp = proc(text=prompts, images=pil, return_tensors="pt", padding=True)
with torch.no_grad():
    out = model(**inp)
ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
cos = (ie @ te.T).numpy()                                # cosine similarity, images x prompts
probs = torch.softmax(out.logits_per_image, dim=-1).numpy()   # CLIP's temperature-scaled softmax
print("REAL CLIP cosine similarity (rows=images, cols=prompts); the true caption should win:")
print(f"  prompts: {prompts}")
for i, n in enumerate(names):
    row = "  ".join(f"{cos[i, j]:+.3f}" for j in range(len(prompts)))
    print(f"  {n:22s}: {row}   -> best: '{prompts[cos[i].argmax()]}'")
print("\nsoftmax over prompts (from logits_per_image; CLIP's large logit scale makes it decisive):")
for i, n in enumerate(names):
    row = "  ".join(f"{probs[i, j]:.3f}" for j in range(len(prompts)))
    print(f"  {n:22s}: {row}   -> P={probs[i].max():.3f} for '{prompts[probs[i].argmax()]}'")

# A sketch of prompt steering: RAISE text-image similarity toward a real CLIP target.
target = te[0]                                           # real 'a photo of a flower' text embedding
v = torch.randn(target.shape[0]); v = v / v.norm()       # start from a random unit vector
print("\nsteering a random vector toward the REAL 'flower' text embedding (5 updates):")
print(f"  before any update: cosine to prompt = {(v @ target).item():+.2f}")
for step in range(1, 6):
    v = v + 0.15 * (target - v); v = v / v.norm()        # interpolate toward the target, renormalize
    print(f"  after update {step}: cosine to prompt = {(v @ target).item():+.2f}")
print("a sketch of the objective only: latent diffusion conditions its denoiser through cross-attention.")

# Why LATENT diffusion: run the expensive process on far fewer numbers.
H = W = 512
pixel_elems = H * W * 3                                   # RGB pixels
lat_h = lat_w = 64; lat_c = 4                             # VAE latent (8x downsampled, 4 channels)
latent_elems = lat_h * lat_w * lat_c
print(f"\ndiffusing in PIXEL space : {H}x{W}x3   = {pixel_elems:,} values per step")
print(f"diffusing in LATENT space: {lat_h}x{lat_w}x{lat_c} = {latent_elems:,} values per step")
print(f"  -> {pixel_elems/latent_elems:.0f}x fewer values; a VAE encodes to latent, diffuses there,")
print("     then decodes to pixels.")

# Classifier-free guidance and negative prompts: dial how strongly the prompt is obeyed.
print("\nclassifier-free guidance: eps = eps_uncond + w * (eps_cond - eps_uncond)")
print("  w=1: the conditional prediction.  w>1 (the CompVis v1 script defaults to 7.5): extrapolate,")
print("  which can raise prompt adherence and cut diversity.")
print("negative prompt: replace eps_uncond with eps_neg (conditioned on the unwanted concept),")
print("  eps = eps_neg + w * (eps_cond - eps_neg), pushing the image AWAY from 'blurry, extra fingers'.")
print("\nfull pipeline: text -> text-encoder token embeddings -> cross-attention in a LATENT diffusion denoiser")
print("with classifier-free guidance -> VAE decoder -> image.")
print("\nAll text-to-image checks ran.")

# %% [markdown]
# Example 24-21. Latent diffusion in a VAE's 16-number latent against diffusion on 784 pixels.

# %%
import warnings; warnings.filterwarnings("ignore")
import tempfile, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(2)                                        # the thread count of the book's builds

root = os.path.join(tempfile.gettempdir(), "mnist_data")       # Example 24-6's data and VAE, retrained
mnist = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
X = mnist.data[:5000].reshape(-1, 784).float() / 255.0
clf = LogisticRegression(max_iter=300).fit(X.numpy(), mnist.targets[:5000].numpy())
def score(s):                                                   # nearest real, classes, mean top probability
    s = s.clamp(0, 1)
    P = clf.predict_proba(s.numpy()); cls = clf.predict(s.numpy())
    return (round(torch.cdist(s, X).min(1).values.mean().item(), 2),
            sum(int(np.mean(cls == k) > 0.02) for k in range(10)), round(float(P.max(1).mean()), 3))
LATENT = 16
torch.manual_seed(0); np.random.seed(0)

class VAE(nn.Module):                                           # Example 24-6, line for line
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU())
        self.mu = nn.Linear(256, LATENT); self.logvar = nn.Linear(256, LATENT)
        self.dec = nn.Sequential(nn.Linear(LATENT, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())
    def forward(self, x):
        h = self.enc(x); mu, logvar = self.mu(h), self.logvar(h)
        return self.dec(mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)), mu, logvar

vae = VAE(); opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
for epoch in range(40):
    perm = torch.randperm(len(X))
    for i in range(0, len(X), 128):
        xb = X[perm[i:i + 128]]; recon, mu, logvar = vae(xb)
        recon_loss = F.binary_cross_entropy(recon, xb, reduction="sum") / len(xb)
        kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / len(xb)
        loss = recon_loss + kl                                  # the negative ELBO of Example 24-6
        opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():                                           # stage 1 done: encode every image once
    Zl = vae.mu(vae.enc(X)); m, sd = Zl.mean(0), Zl.std(0)
    Zs = (Zl - m) / sd                                          # 5,000 latents, each dimension standardized
    print("uniform noise images    :", score(torch.rand(500, 784, generator=torch.Generator().manual_seed(0))))
    print("VAE prior draws decoded :", score(vae.dec(torch.randn(500, LATENT, generator=torch.Generator().manual_seed(1)))))

T = 50                                                          # Example 24-15's schedule and sampler
t_lin = torch.linspace(0, 1, T + 1)
abar = torch.cos((t_lin + 0.008) / 1.008 * np.pi / 2) ** 2
abar = abar / abar[0]
beta = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
alpha = 1 - beta
abar = torch.cumprod(alpha, 0)

def train_denoiser(data, limit, steps=4000):                    # stage 2: diffusion on rows of `data`
    dim = data.shape[1]
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(dim + 1, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                        nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, dim))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for step in range(steps):
        x0 = data[torch.randint(0, len(data), (256,))]; t = torch.randint(0, T, (256,)); eps = torch.randn_like(x0)
        xt = torch.sqrt(abar[t])[:, None] * x0 + torch.sqrt(1 - abar[t])[:, None] * eps
        loss = F.mse_loss(net(torch.cat([xt, t[:, None] / T], 1)), eps)
        opt.zero_grad(); loss.backward(); opt.step()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        x = torch.randn(500, dim, generator=g)
        for ti in reversed(range(T)):
            e = net(torch.cat([x, torch.full((500, 1), ti / T)], 1))
            x0_hat = ((x - torch.sqrt(1 - abar[ti]) * e) / torch.sqrt(abar[ti])).clamp(-limit, limit)
            abar_prev = abar[ti - 1] if ti > 0 else torch.tensor(1.0)
            mean = (torch.sqrt(abar_prev) * beta[ti] / (1 - abar[ti])) * x0_hat \
                + (torch.sqrt(alpha[ti]) * (1 - abar_prev) / (1 - abar[ti])) * x
            var = (1 - abar_prev) / (1 - abar[ti]) * beta[ti]
            x = mean + (torch.sqrt(var) * torch.randn(500, dim, generator=g) if ti > 0 else 0)
    return loss.item(), x

l_lat, z = train_denoiser(Zs, limit=float(Zs.abs().max()))      # clamp at the largest training value
with torch.no_grad():
    lat_samples = vae.dec(z * sd + m)                           # stage 3: decode the latents to pixels
print(f"latent diffusion, {LATENT} values per step: final MSE {l_lat:.3f}; decoded samples {score(lat_samples)}")
l_pix, xp = train_denoiser(X * 2 - 1, limit=1.0)                # the same denoiser width on 784 pixels
print(f"pixel diffusion, {X.shape[1]} values per step: final MSE {l_pix:.3f}; samples {score((xp + 1) / 2)}")
print(f"values per denoising step: {X.shape[1]} / {LATENT} = {X.shape[1] // LATENT} times fewer in the latent")

# %% [markdown]
# ## 24.8 Multimodal generation

# %%
H = W = 512
image_vals = H * W * 3
fps, seconds = 24, 5
video_vals = image_vals * fps * seconds
print(f"one 512x512 image      : {image_vals:,} values")
print(f"a {seconds}s {fps}fps video    : {video_vals:,} values  ({fps*seconds}x an image)")
print("plus TEMPORAL CONSISTENCY: objects, lighting and motion must stay coherent across frames,")
print("which generating each frame on its own does not guarantee.")

lat_h = lat_w = 64; lat_c = 4; temporal_downsample = 4
video_latent = lat_h * lat_w * lat_c * (fps * seconds // temporal_downsample)
print(f"\nvideo LATENT (8x spatial, {temporal_downsample}x temporal downsample): {video_latent:,} values")
print(f"  -> {video_vals/video_latent:.0f}x smaller; this book's arithmetic, not a measured video model")

print("\nAll multimodal checks ran.")

# %% [markdown]
# ## 24.9 Hands-on lab: VAE, GAN, and diffusion on digits

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(2)                                  # the thread count of the book's builds

digits = load_digits()
X01 = torch.tensor(digits.images.reshape(-1, 64) / 16.0, dtype=torch.float32)   # [0,1]
Xreal = X01.numpy()
clf = LogisticRegression(max_iter=500).fit(Xreal, digits.target)                # for diversity scoring

def score(samples01):
    s = np.clip(samples01, 0, 1)
    d = np.sqrt(((s[:, None] - Xreal[None]) ** 2).sum(2)).min(1).mean()          # mean nearest-real L2
    classes = clf.predict(s)
    covered = sum(1 for c in range(10) if np.mean(classes == c) > 0.02)
    return d, covered

def evaluate(samples01, name):
    d, covered = score(samples01)
    print(f"  {name:10s}: quality (nearest-real dist) {d:.2f}   diversity (classes covered) {covered}/10")

class VAE(nn.Module):
    def __init__(self, L=16):
        super().__init__(); self.e = nn.Sequential(nn.Linear(64, 48), nn.ReLU())
        self.mu = nn.Linear(48, L); self.lv = nn.Linear(48, L)
        self.d = nn.Sequential(nn.Linear(L, 48), nn.ReLU(), nn.Linear(48, 64), nn.Sigmoid()); self.L = L
    def forward(self, x):
        h = self.e(x); mu, lv = self.mu(h), self.lv(h); z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.d(z), mu, lv

def train_three(seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    vae = VAE(); ov = torch.optim.Adam(vae.parameters(), lr=2e-3)
    for _ in range(400):
        r, mu, lv = vae(X01)
        loss = nn.functional.binary_cross_entropy(r, X01, reduction="sum") / len(X01) \
            - 0.5 * torch.sum(1 + lv - mu ** 2 - lv.exp()) / len(X01)
        ov.zero_grad(); loss.backward(); ov.step()
    with torch.no_grad():
        vae_samples = vae.d(torch.randn(500, vae.L)).numpy()

    Z = 16
    G = nn.Sequential(nn.Linear(Z, 64), nn.ReLU(), nn.Linear(64, 64), nn.Sigmoid())
    Dnet = nn.Sequential(nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
    og = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))
    od = torch.optim.Adam(Dnet.parameters(), lr=2e-4, betas=(0.5, 0.9))
    bce = nn.BCEWithLogitsLoss()
    for _ in range(3000):
        real = X01[torch.randint(0, len(X01), (128,))]
        fake = G(torch.randn(128, Z))
        ld = bce(Dnet(real), torch.ones(128, 1)) + bce(Dnet(fake.detach()), torch.zeros(128, 1))
        od.zero_grad(); ld.backward(); od.step()
        fake = G(torch.randn(128, Z))
        lg = bce(Dnet(fake), torch.ones(128, 1))
        og.zero_grad(); lg.backward(); og.step()
    with torch.no_grad():
        gan_samples = G(torch.randn(500, Z)).numpy()

    Xd = X01 * 2 - 1; T = 50
    tl = torch.linspace(0, 1, T + 1)
    ab = torch.cos((tl + 0.008) / 1.008 * np.pi / 2) ** 2; ab = ab / ab[0]   # the cosine curve, 1 at t = 0
    be = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)          # every beta in (0, 1) (Nichol and Dhariwal, 2021)
    al = 1 - be; ab = torch.cumprod(al, 0)                # abar for steps 1..T at indices 0..T-1
    den = nn.Sequential(nn.Linear(65, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                        nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 64))
    odf = torch.optim.Adam(den.parameters(), lr=2e-3)
    for _ in range(5000):
        i = torch.randint(0, len(Xd), (256,)); x0 = Xd[i]; t = torch.randint(0, T, (256,)); eps = torch.randn_like(x0)
        xt = torch.sqrt(ab[t])[:, None] * x0 + torch.sqrt(1 - ab[t])[:, None] * eps
        loss = nn.functional.mse_loss(den(torch.cat([xt, t[:, None] / T], 1)), eps)
        odf.zero_grad(); loss.backward(); odf.step()
    with torch.no_grad():
        x = torch.randn(500, 64)
        for ti in reversed(range(T)):
            e = den(torch.cat([x, torch.full((500, 1), ti / T)], 1))
            x0_hat = ((x - torch.sqrt(1 - ab[ti]) * e) / torch.sqrt(ab[ti])).clamp(-1, 1)   # predicted clean digit
            ab_prev = ab[ti - 1] if ti > 0 else torch.tensor(1.0)
            mean = (torch.sqrt(ab_prev) * be[ti] / (1 - ab[ti])) * x0_hat \
                + (torch.sqrt(al[ti]) * (1 - ab_prev) / (1 - ab[ti])) * x                # posterior mean
            var = (1 - ab_prev) / (1 - ab[ti]) * be[ti]                                  # posterior variance
            x = mean + (torch.sqrt(var) * torch.randn_like(x) if ti > 0 else 0)
        diff_samples = ((x + 1) / 2).numpy()
    return vae_samples, gan_samples, diff_samples, rng

vae_samples, gan_samples, diff_samples, rng = train_three(0)
noise = rng.random((500, 64))                             # baseline anchor for the quality scale
print("\ngenerative models on 8-by-8 digits (lower distance = more realistic; higher coverage = diverse):")
evaluate(noise, "noise")
evaluate(vae_samples, "VAE")
evaluate(gan_samples, "GAN")
evaluate(diff_samples, "diffusion")

print("\nAll generative-lab checks ran.")

# %% [markdown]
# Example 24-24. The lab repeated at four more seeds.

# %%
torch.set_num_threads(2)                                  # the thread count of the book's builds
print("the same three models at four more seeds (nearest-real distance, classes covered):")
for seed in range(1, 5):
    vae_s, gan_s, diff_s, rng_s = train_three(seed)
    rows = [("noise", rng_s.random((500, 64))), ("VAE", vae_s), ("GAN", gan_s), ("diffusion", diff_s)]
    print(f"  seed {seed}: " + "   ".join(f"{name} {d:.2f} ({c}/10)" for name, s in rows for d, c in [score(s)]))

# %% [markdown]
# ## 24.9.1 Standard evaluation metrics: FID, Inception Score, and precision/recall

# %%
import numpy as np
from scipy.linalg import sqrtm
from sklearn.datasets import load_digits
rng = np.random.default_rng(0)

X = load_digits().images.reshape(-1, 64) / 16.0          # real 8-by-8 digits in [0,1]

def frechet(A, B):                                       # the FID formula on feature vectors
    ma, mb = A.mean(0), B.mean(0)
    Ca, Cb = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    covmean = sqrtm(Ca @ Cb).real                        # matrix square root of the product
    return ((ma - mb) ** 2).sum() + np.trace(Ca + Cb - 2 * covmean)

half = len(X) // 2
print("Frechet distance on 64-dim pixel features (stand-in for Inception features); lower = closer:")
print(f"  real vs held-out real : {frechet(X[:half], X[half:]):.3f}   (same distribution -> near 0)")
print(f"  real vs pure noise     : {frechet(X[:half], rng.random((half, 64))):.3f}   (far apart -> large)")

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0)
digits = load_digits()
X = digits.images.reshape(-1, 64) / 16.0
clf = LogisticRegression(max_iter=500).fit(X, digits.target)

def inception_score(P):
    py = P.mean(0)
    kl = (P * (np.log(np.clip(P, 1e-12, 1)) - np.log(np.clip(py, 1e-12, 1)))).sum(1).mean()
    return float(np.exp(kl))

real_p = clf.predict_proba(X[:300])
noise_p = clf.predict_proba(rng.random((300, 64)))
print(f"Inception Score  real {inception_score(real_p):.4f}   noise {inception_score(noise_p):.4f}")

# %%
def kth_radius(A, k=3):                    # each point's distance to its kth nearest neighbor in its own set
    d = np.sqrt(((A[:, None] - A[None]) ** 2).sum(2))
    return np.sort(d, 1)[:, k]             # column 0 is the point itself, at distance 0
def share_inside(B, A, rA):                # share of B inside at least one ball around A (boundary counts)
    d = np.sqrt(((B[:, None] - A[None]) ** 2).sum(2))
    return (d <= rA[None]).any(1).mean()

real = Xreal[:500]; r_real = kth_radius(real)
print("improved precision and recall, k = 3, in the 64-pixel space, against 500 real digits:")
for name, s in (("noise", noise), ("VAE", vae_samples), ("GAN", gan_samples), ("diffusion", diff_samples)):
    s = np.clip(s, 0, 1)
    print(f"  {name:10s}: precision {share_inside(s, real, r_real):.3f}   recall {share_inside(real, s, kth_radius(s)):.3f}")

# %% [markdown]
# ## Coding exercises

# %%
from sklearn.datasets import load_digits
print(load_digits().images.shape)
def recall():
    ...
print(recall())

# %%
from sklearn.datasets import load_digits
X = (load_digits().images.reshape(-1, 64) > 7).astype(int)
print(X.shape)
def log_likelihood(img):
    ...
print(log_likelihood(X[0]))

# %%
from sklearn.datasets import load_digits
print((load_digits().images.reshape(-1, 64) / 16.0).shape)
def train_vae():
    ...
print(train_vae())

# %%
import torch
mu = torch.tensor([0.2, -0.4])
logvar = torch.tensor([-0.3, 0.5])
print(tuple(mu.shape))
def kl_closed(mu, logvar):
    ...
print(kl_closed(mu, logvar))

# %%
from sklearn.datasets import load_breast_cancer
X = load_breast_cancer().data[:, [0, 3]]                  # mean radius, mean area
print(X.shape)
def train_flow():
    ...
print(train_flow())

# %%
from sklearn.datasets import load_digits
print((load_digits().images.reshape(-1, 64) / 16.0).shape)
def train_gan():
    ...
print(train_gan())

# %%
from sklearn.datasets import load_digits
print(load_digits().images.shape)
def signal_fraction():
    ...
print(signal_fraction())

# %%
from sklearn.datasets import load_digits
print(load_digits().images.reshape(-1, 64).shape)
def ddim_pair():
    ...
print(ddim_pair())

# %%
from sklearn.datasets import load_digits
X = load_digits().images.reshape(-1, 64) / 16.0
print(X.shape)
def frechet_to_real(samples):
    ...
print(frechet_to_real(None))

# %%
print(512 * 512 * 3, 64 * 64 * 4)
def ratio(H, f, C):
    ...
print(ratio(512, 8, 4))

# %%
import numpy as np
from sklearn.datasets import load_digits
rows = load_digits().images.reshape(-1, 64)[:32] / 16.0
print(rows.shape)
def quantize_rounds(rows, code, rounds=10):
    ...
print(quantize_rounds(rows, np.random.default_rng(0).random((8, 64))))

# %%
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
print(load_digits().data.shape)
def inception_score_of(samples01):
    ...
print(inception_score_of(None))

# %%
import torch
from sklearn.datasets import load_digits
real = torch.tensor(load_digits().images[:128].reshape(128, 64) / 16.0, dtype=torch.float32)
print(tuple(real.shape))
def train_critic(steps=200, lam=10.0):
    ...
print(train_critic())


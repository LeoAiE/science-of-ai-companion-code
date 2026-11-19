"""Appendix H solutions for Chapter 24, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 24.10 coding exercises

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.linalg import sqrtm
torch.set_num_threads(2)                                        # the thread count of the book's builds

# %% [markdown]
# 1. Hopfield associative memory on real digits

# %%
# 1. Hopfield associative memory on real digits
rng = np.random.default_rng(0)
imgs = load_digits().images                                     # 8-by-8 digits, 0..16
patterns = np.array([np.where(imgs[i].reshape(-1) > 6, 1, -1) for i in (0, 3, 6)])
N = 64
W = sum(np.outer(p, p) for p in patterns) / N; np.fill_diagonal(W, 0)
def recall(s, steps=15):
    s = s.copy()
    for _ in range(steps):
        for i in rng.permutation(N):
            s[i] = 1 if W[i] @ s >= 0 else -1
    return s
target = patterns[0].copy(); cue = target.copy()
cue[rng.choice(N, N // 5, replace=False)] *= -1                 # flip 12 of the 64 pixels, about a fifth
rec = recall(cue)
print("1.", f"cue vs target {round((cue==target).mean()*100)}% ; recovered {round((rec==target).mean()*100)}%")
# expected output: 1. cue vs target 81% ; recovered 100%

# %% [markdown]
# 2. Autoregressive per-pixel likelihood as an anomaly detector

# %%
# 2. Autoregressive per-pixel likelihood as an anomaly detector
X = (load_digits().images.reshape(-1, 64) > 7).astype(int); D = 64
Xtr, Xte = train_test_split(X, test_size=0.25, random_state=0)  # fit on 1,347 digits, hold out 450
models = []
for i in range(D):
    yi = Xtr[:, i]
    if i == 0 or yi.min() == yi.max():
        models.append(("const", float(yi.mean())))
    else:
        models.append(("clf", LogisticRegression(max_iter=200).fit(Xtr[:, :i], yi)))
def loglik(img):
    ll = 0.0
    for i in range(D):
        kind, m = models[i]
        p = m if kind == "const" else m.predict_proba(img[:i].reshape(1, -1))[0, 1]
        p = min(max(p, 1e-6), 1 - 1e-6)
        ll += np.log(p if img[i] else 1 - p)
    return ll
real_ll = np.array([loglik(Xte[k]) for k in range(300)])            # 300 held-out digits
noise_ll = np.array([loglik(n) for n in rng.integers(0, 2, (300, D))])
auc = roc_auc_score(np.r_[np.ones(300), np.zeros(300)], np.r_[real_ll, noise_ll])
print("2.", f"mean LL held-out real {real_ll.mean():.1f} vs noise {noise_ll.mean():.1f}; anomaly AUC {auc:.3f}")
# expected output: 2. mean LL held-out real -18.7 vs noise -154.7; anomaly AUC 1.000

# %% [markdown]
# 3. VAE from scratch on load_digits (end-to-end)

# %%
# 3. VAE from scratch on load_digits (end-to-end)
digits3 = load_digits()
Xv = torch.tensor(digits3.images.reshape(-1, 64) / 16.0, dtype=torch.float32); L = 8
clf3 = LogisticRegression(max_iter=500).fit(Xv.numpy(), digits3.target)   # the lab's digit classifier
def lab_metrics(s01):                                           # the two metrics of the Section 24.9 lab
    s = np.clip(s01, 0, 1)
    d = np.sqrt(((s[:, None] - Xv.numpy()[None]) ** 2).sum(2)).min(1).mean()   # mean nearest-real distance
    c = clf3.predict(s)
    return d, sum(1 for k in range(10) if np.mean(c == k) > 0.02)            # classes covered
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.mu = nn.Linear(32, L); self.lv = nn.Linear(32, L)
        self.d = nn.Sequential(nn.Linear(L, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())
    def forward(self, x, kl_on=True):
        h = self.e(x); mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if kl_on else mu   # reparameterization trick
        return self.d(z), mu, lv
for kl_on in (True, False):                                     # the second run drops the KL term and samples z = mu, a plain autoencoder
    torch.manual_seed(0)
    vae = VAE(); opt = torch.optim.Adam(vae.parameters(), lr=3e-3)
    for _ in range(500):
        r, mu, lv = vae(Xv, kl_on)
        recon = nn.functional.binary_cross_entropy(r, Xv, reduction="sum") / len(Xv)
        kl = -0.5 * torch.sum(1 + lv - mu**2 - lv.exp()) / len(Xv)  # KL(q(z|x) || N(0,I)), printed in both runs
        loss = recon + kl if kl_on else recon                   # without KL it is computed but not optimized
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _, mu, lv = vae(Xv, kl_on); samples = vae.d(torch.randn(500, L)).numpy()   # decode 500 prior draws
    d, cov = lab_metrics(samples)
    print("3.", "with KL:" if kl_on else "without KL:", f"recon {recon.item():.1f}, KL {kl.item():.1f}; "
          f"latent mean {mu.mean():+.2f} std {mu.std():.2f}; prior samples nearest-real {d:.2f}, classes {cov}/10")
# expected output: 3. with KL: recon 22.0, KL 2.7; latent mean +0.01 std 0.64; prior samples nearest-real 1.23, classes 10/10
# expected output: 3. without KL: recon 17.8, KL 91.3; latent mean -2.00 std 4.28; prior samples nearest-real 2.27, classes 6/10

# %% [markdown]
# 4. Gaussian KL: closed form vs Monte Carlo

# %%
# 4. Gaussian KL: closed form vs Monte Carlo
torch.manual_seed(1)
mu4 = torch.tensor([0.2, -0.4]); lv4 = torch.tensor([-0.3, 0.5]); sig = torch.exp(0.5 * lv4)   # the starter's values
kl_closed = 0.5 * torch.sum(mu4**2 + sig**2 - lv4 - 1).item()
z = mu4 + sig * torch.randn(400000, 2)
logq = (-0.5 * ((z - mu4) / sig)**2 - torch.log(sig) - 0.5 * np.log(2 * np.pi)).sum(1)
logp = (-0.5 * z**2 - 0.5 * np.log(2 * np.pi)).sum(1)
print("4.", f"KL closed form {kl_closed:.4f} vs Monte Carlo {(logq - logp).mean().item():.4f}")
# expected output: 4. KL closed form 0.1948 vs Monte Carlo 0.1962

# %% [markdown]
# 5. Normalizing-flow coupling layer: invertibility + trained likelihood gap

# %%
# 5. Normalizing-flow coupling layer: invertibility + trained likelihood gap
torch.manual_seed(0)
raw = load_breast_cancer().data[:, [0, 3]]                      # mean radius, mean area
data = torch.tensor((raw - raw.mean(0)) / raw.std(0), dtype=torch.float32)
class Flow(nn.Module):
    def __init__(self, n=6):
        super().__init__()
        self.nets = nn.ModuleList([nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 2)) for _ in range(n)])
        self.swap = [i % 2 for i in range(n)]
    def x_to_z(self, x):                                        # data -> base; 'a' stays fixed
        z = x; logdet = torch.zeros(len(x))
        for net, sw in zip(self.nets, self.swap):
            a = z[:, sw]; o = net(a[:, None]); ls = torch.tanh(o[:, 0]); t = o[:, 1]
            z = z.clone(); z[:, 1 - sw] = (z[:, 1 - sw] - t) * torch.exp(-ls); logdet = logdet - ls
        return z, logdet
    def z_to_x(self, z):                                        # exact inverse
        x = z
        for net, sw in reversed(list(zip(self.nets, self.swap))):
            a = x[:, sw]; o = net(a[:, None]); ls = torch.tanh(o[:, 0]); t = o[:, 1]
            x = x.clone(); x[:, 1 - sw] = x[:, 1 - sw] * torch.exp(ls) + t
        return x
flow = Flow()
with torch.no_grad():
    zz, _ = flow.x_to_z(data); roundtrip = (data - flow.z_to_x(zz)).abs().max().item()
def log_prob(x):
    z, logdet = flow.x_to_z(x)
    return -0.5 * (z**2).sum(1) - np.log(2 * np.pi) + logdet
opt = torch.optim.Adam(flow.parameters(), lr=5e-3)
for _ in range(400):
    loss = -log_prob(data).mean(); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    off = data[:, [0]].repeat(1, 2).clone(); off[:, 1] = data[torch.randperm(len(data)), 1]
    print("5.", f"round-trip max err {roundtrip:.1e}; LL real {log_prob(data).mean():.2f} vs off-dist {log_prob(off).mean():.2f}")
# expected output: 5. round-trip max err 2.9e-06; LL real 1.25 vs off-dist -54.87

# %% [markdown]
# 6. GAN on real digits: class coverage

# %%
# 6. GAN on real digits: class coverage
torch.manual_seed(0)
X01 = torch.tensor(load_digits().images.reshape(-1, 64) / 16.0, dtype=torch.float32)
clf = LogisticRegression(max_iter=500).fit(X01.numpy(), load_digits().target)
Z = 16
G = nn.Sequential(nn.Linear(Z, 64), nn.ReLU(), nn.Linear(64, 64), nn.Sigmoid())
Dn = nn.Sequential(nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
og = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))
od = torch.optim.Adam(Dn.parameters(), lr=2e-4, betas=(0.5, 0.9)); bce = nn.BCEWithLogitsLoss()
for _ in range(2500):
    real = X01[torch.randint(0, len(X01), (128,))]; fake = G(torch.randn(128, Z))
    ld = bce(Dn(real), torch.ones(128, 1)) + bce(Dn(fake.detach()), torch.zeros(128, 1))
    od.zero_grad(); ld.backward(); od.step()
    fake = G(torch.randn(128, Z)); lg = bce(Dn(fake), torch.ones(128, 1))   # nonsaturating G loss
    og.zero_grad(); lg.backward(); og.step()
with torch.no_grad():
    gen = G(torch.randn(500, Z)).numpy()
    dreal = torch.sigmoid(Dn(real)).mean().item(); dfake = torch.sigmoid(Dn(fake)).mean().item()
cov = sum(1 for c in range(10) if np.mean(clf.predict(np.clip(gen, 0, 1)) == c) > 0.02)
print("6.", f"D(real) {dreal:.2f} D(fake) {dfake:.2f}; GAN covers {cov}/10 classes")
# expected output: 6. D(real) 0.52 D(fake) 0.47; GAN covers 10/10 classes

# %% [markdown]
# 7. Diffusion forward process: cosine schedule + variance preservation

# %%
# 7. Diffusion forward process: cosine schedule + variance preservation
torch.manual_seed(0)
Xd = torch.tensor(load_digits().images.reshape(-1, 64) / 16.0 * 2 - 1, dtype=torch.float32)
x0 = Xd / Xd.std(); T = 50
tl = torch.linspace(0, 1, T + 1); abar = torch.cos((tl + 0.008) / 1.008 * np.pi / 2)**2
abar = (abar / abar[0])[1:]
mono = bool((abar[1:] <= abar[:-1]).all())
xt = torch.sqrt(abar[T // 2]) * x0 + torch.sqrt(1 - abar[T // 2]) * torch.randn_like(x0)
print("7.", f"abar 1->{abar[0]:.2f} mid {abar[T//2]:.2f} T {abar[-1]:.3f}; monotone {mono}; "
      f"Var(x0) {x0.var():.2f} Var(x_mid) {xt.var():.2f}")
# expected output: 7. abar 1->1.00 mid 0.46 T 0.000; monotone True; Var(x0) 1.00 Var(x_mid) 1.01

# %% [markdown]
# 8. Train a denoiser and add DDIM deterministic sampling

# %%
# 8. Train a denoiser and add DDIM deterministic sampling
torch.manual_seed(0)
Xdf = torch.tensor(load_digits().images.reshape(-1, 64) / 16.0 * 2 - 1, dtype=torch.float32)
T = 50; tl = torch.linspace(0, 1, T + 1); ab = torch.cos((tl + 0.008) / 1.008 * np.pi / 2)**2; ab = (ab / ab[0])[1:]
den = nn.Sequential(nn.Linear(65, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 64))
opt = torch.optim.Adam(den.parameters(), lr=2e-3)
for _ in range(3000):
    i = torch.randint(0, len(Xdf), (256,)); a = Xdf[i]; t = torch.randint(0, T, (256,)); e = torch.randn_like(a)
    xt = torch.sqrt(ab[t])[:, None] * a + torch.sqrt(1 - ab[t])[:, None] * e
    loss = nn.functional.mse_loss(den(torch.cat([xt, t[:, None] / T], 1)), e)   # predict the noise
    opt.zero_grad(); loss.backward(); opt.step()
@torch.no_grad()
def ddim(x, steps=10):
    seq = torch.linspace(T - 1, 0, steps).round().long().tolist()
    for i, ti in enumerate(seq):
        e = den(torch.cat([x, torch.full((len(x), 1), ti / T)], 1))
        x0p = ((x - torch.sqrt(1 - ab[ti]) * e) / torch.sqrt(ab[ti])).clamp(-1, 1)
        x = x0p if i == len(seq) - 1 else torch.sqrt(ab[seq[i + 1]]) * x0p + torch.sqrt(1 - ab[seq[i + 1]]) * e
    return x
torch.manual_seed(7); zz = torch.randn(4, 64)
s1 = ddim(zz.clone(), 10); s2 = ddim(zz.clone(), 10)
print("8.", f"denoiser MSE {loss.item():.3f}; DDIM same-seed max diff {(s1-s2).abs().max().item():.1e} (10 vs {T} steps)")
# expected output: 8. denoiser MSE 0.232; DDIM same-seed max diff 0.0e+00 (10 vs 50 steps)

# %% [markdown]
# 9. Frechet distance of the exercise 6 GAN and the exercise 8 DDIM sampler to the real digits

# %%
# 9. Frechet distance of the exercise 6 GAN and the exercise 8 DDIM sampler to the real digits
Xf = load_digits().images.reshape(-1, 64) / 16.0
def frechet(A, B):                                              # the FID formula on 64 pixel features
    Ca, Cb = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    cm = sqrtm(Ca @ Cb).real
    return ((A.mean(0) - B.mean(0))**2).sum() + np.trace(Ca + Cb - 2 * cm)
h = len(Xf) // 2                                                # the first half is the reference, as in Section 24.9.1
torch.manual_seed(9)
with torch.no_grad():
    gan9 = G(torch.randn(500, Z)).numpy()                       # 500 samples of the exercise 6 generator, in [0, 1]
    ddim9 = ((ddim(torch.randn(500, 64), 10) + 1) / 2).numpy()  # 500 ten-step DDIM samples of exercise 8, mapped to [0, 1]
print("9.", f"Frechet to real digits: GAN {frechet(Xf[:h], gan9):.3f}; DDIM {frechet(Xf[:h], ddim9):.3f}")
# expected output: 9. Frechet to real digits: GAN 1.828; DDIM 0.835

# %% [markdown]
# 10. Latent-diffusion accounting + classifier-free guidance

# %%
# 10. Latent-diffusion accounting + classifier-free guidance
def ratio(H, f=8, C=4):                                         # pixel values over latent values per step
    return (H * H * 3) / ((H // f) * (H // f) * C)              # (H*H*3) / ((H/f)*(H/f)*C) = 3 f^2 / C at any H
# f = 8 with C = 4 is the CompVis v1 latent; f = 8 with C = 16 is the autoencoder Esser et al. (2024) choose
print("10.", f"f=8 C=4: 512 px {ratio(512):.0f}x, 768 px {ratio(768):.0f}x; f=8 C=16: {ratio(512, C=16):.0f}x; 3*8**2/4 = {3 * 8**2 / 4:.0f}")
e_un = np.array([0.10, -0.20, 0.30]); e_co = np.array([0.40, 0.10, 0.05])
cfg = {w: (e_un + w * (e_co - e_un)).round(3).tolist() for w in (1.0, 3.0, 7.5)}
print("10.", f"CFG w=1 {cfg[1.0]} w=3 {cfg[3.0]} w=7.5 {cfg[7.5]}")
# expected output: 10. f=8 C=4: 512 px 48x, 768 px 48x; f=8 C=16: 12x; 3*8**2/4 = 48
# expected output: 10. CFG w=1 [0.4, 0.1, 0.05] w=3 [1.0, 0.7, -0.45] w=7.5 [2.35, 2.05, -1.575]

# %% [markdown]
# 11. Ten rounds of nearest code and mean update on the first 32 digits

# %%
# 11. Ten rounds of nearest code and mean update on the first 32 digits
rows = load_digits().images.reshape(-1, 64)[:32] / 16.0
def ten_rounds(code):
    dist = ((rows[:, None] - code[None]) ** 2).sum(2)
    log = [f"start {dist.min(1).mean():.4f}"]
    for _ in range(10):
        idx = dist.argmin(1)                                    # nearest code; argmin breaks a tie toward the lower index
        for k in range(len(code)):
            chosen = rows[idx == k]
            if len(chosen):
                code[k] = chosen.mean(0)                        # move each used code to the mean of its rows
        dist = ((rows[:, None] - code[None]) ** 2).sum(2)
        log.append(f"{dist.min(1).mean():.4f}/{len(set(idx.tolist()))}")   # distance after the round / codes used in it
    return " ".join(log)
print("11. random codebook:", ten_rounds(np.random.default_rng(0).random((8, 64))))
print("11. 8 rows as codebook:", ten_rounds(rows[np.random.default_rng(0).choice(32, 8, replace=False)].copy()))
# expected output: 11. random codebook: start 14.0928 3.8536/3 3.7353/3 3.6929/3 3.6609/3 3.6609/3 3.6609/3 3.6609/3 3.6609/3 3.6609/3 3.6609/3
# expected output: 11. 8 rows as codebook: start 4.0784 2.2185/8 2.1138/8 2.1138/8 2.1138/8 2.1138/8 2.1138/8 2.1138/8 2.1138/8 2.1138/8 2.1138/8

# %% [markdown]
# 12. Inception Score of the exercise 9 GAN and DDIM samples

# %%
# 12. Inception Score of the exercise 9 GAN and DDIM samples
digits12 = load_digits()
Xis = digits12.images.reshape(-1, 64) / 16.0
clf_is = LogisticRegression(max_iter=500).fit(Xis, digits12.target)   # the digit classifier of Section 24.9.1
def iscore(P):
    py = P.mean(0)
    kl = (P * (np.log(np.clip(P, 1e-12, 1)) - np.log(np.clip(py, 1e-12, 1)))).sum(1).mean()
    return float(np.exp(kl))
P_gan, P_ddim = clf_is.predict_proba(np.clip(gan9, 0, 1)), clf_is.predict_proba(np.clip(ddim9, 0, 1))  # exercise 9's samples
print("12.", f"Inception Score GAN {iscore(P_gan):.4f} DDIM {iscore(P_ddim):.4f}")
# expected output: 12. Inception Score GAN 3.8076 DDIM 3.0759

# %% [markdown]
# 13. Training the critic on the WGAN-GP critic loss

# %%
# 13. Training the critic on the WGAN-GP critic loss
torch.manual_seed(0)
Xp = torch.tensor(load_digits().images.reshape(-1, 64) / 16.0, dtype=torch.float32)
real = Xp[:128]                                                 # the first 128 digits
fake = torch.tensor(np.random.default_rng(0).random((128, 64)), dtype=torch.float32)   # stand-in fakes
u0 = torch.rand(128, 1)                                         # fixed mixing weights for the before and after report
critic = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))   # the critic of Section 24.5.2
def penalty_and_norm(u):
    xhat = (u * real + (1 - u) * fake).requires_grad_(True)
    grad = torch.autograd.grad(critic(xhat).sum(), xhat, create_graph=True)[0]   # keep the graph to train on it
    gnorm = grad.norm(dim=1)
    return ((gnorm - 1) ** 2).mean(), gnorm.mean()
p0, n0 = penalty_and_norm(u0)
opt = torch.optim.Adam(critic.parameters())                     # Adam with its default settings
for _ in range(200):
    gp, _ = penalty_and_norm(torch.rand(128, 1))
    loss = critic(fake).mean() - critic(real).mean() + 10.0 * gp   # WGAN-GP critic loss, lambda = 10
    opt.zero_grad(); loss.backward(); opt.step()
p1, n1 = penalty_and_norm(u0)
print("13.", f"penalty {p0.item():.4f} -> {p1.item():.4f}; mean gradient norm {n0.item():.4f} -> {n1.item():.4f}")
# expected output: 13. penalty 0.5831 -> 0.0173; mean gradient norm 0.2367 -> 1.1313

# %% [markdown]
# ## 24.10 coding drills

# %%
import numpy as np, torch
import torch.nn.functional as Fn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from scipy.linalg import sqrtm
rng = np.random.default_rng(0)

# %% [markdown]
# Family tree & Hopfield

# %%
# Family tree & Hopfield
p = np.where(load_digits().images[0].reshape(-1) > 6, 1, -1)
W = np.outer(p, p) / 64; np.fill_diagonal(W, 0)
print(W.shape)                                              # expected output: (64, 64)
print(round(float(-0.5 * p @ W @ p), 3))                    # expected output: -31.5
s = p.copy(); s[0] = 1 if W[0] @ s >= 0 else -1
print(s[0] == p[0])                                         # expected output: True
a, b = np.array([1, -1, 1, 1]), np.array([1, 1, 1, -1])
print((a == b).mean())                                      # expected output: 0.5

# %% [markdown]
# Autoregressive & likelihood

# %%
# Autoregressive & likelihood
Xb = (load_digits().images.reshape(-1, 64) > 7).astype(int)
print(round(Xb[:, 20].mean(), 3))                           # expected output: 0.461
ps = np.array([0.9, 0.2, 0.8]); bits = np.array([1, 0, 1])
print(round(float(np.sum(np.log(np.where(bits, ps, 1 - ps)))), 3))  # expected output: -0.552
# chain rule of probability: p(x1..xn) = p(x1) p(x2|x1) ... p(xn|x1..xn-1), exact for any fixed ordering,
# so log p(x) is the sum of the logs of the conditionals

# %% [markdown]
# VAE & the ELBO

# %%
# VAE & the ELBO
torch.manual_seed(0)
mu = torch.tensor([0.0, 1.0]); lv = torch.tensor([0.0, -0.5])
z = mu + torch.exp(0.5 * lv) * torch.randn(2)
print([round(v, 3) for v in z.tolist()])                    # expected output: [1.541, 0.771]
print(round((-0.5 * torch.sum(1 + lv - mu**2 - lv.exp())).item(), 4))  # expected output: 0.5533
# posterior collapse symptom: KL falls to ~0 while reconstructions stay mediocre

# %% [markdown]
# Normalizing flows

# %%
# Normalizing flows
ls, t, bval = torch.tensor(0.3), torch.tensor(0.5), torch.tensor(-1.0)
bfwd = (bval - t) * torch.exp(-ls); binv = bfwd * torch.exp(ls) + t
print(round(float(binv), 3), round(float(bval), 3))         # expected output: -1.0 -1.0
print(round(float(-ls), 3))                                 # expected output: -0.3

# %% [markdown]
# GANs & diffusion

# %%
# GANs & diffusion
print(round(Fn.binary_cross_entropy_with_logits(torch.tensor([[2.0], [-1.0]]), torch.ones(2, 1)).item(), 3))
# expected output: 0.72
tl = torch.linspace(0, 1, 51); ab = torch.cos((tl + 0.008) / 1.008 * np.pi / 2)**2; ab = (ab / ab[0])[1:]
print(round(ab[0].item(), 3), round(ab[-1].item(), 3))      # expected output: 0.998 0.0
x0, e, tt = torch.tensor([1.0]), torch.tensor([0.5]), 25
print(round((torch.sqrt(ab[tt]) * x0 + torch.sqrt(1 - ab[tt]) * e).item(), 3))  # expected output: 1.047

# %% [markdown]
# Evaluation

# %%
# Evaluation
A, B = rng.normal(0, 1, (200, 3)), rng.normal(0, 1, (200, 3))
cm = sqrtm(np.cov(A, rowvar=False) @ np.cov(B, rowvar=False)).real
fr = ((A.mean(0) - B.mean(0))**2).sum() + np.trace(np.cov(A, rowvar=False) + np.cov(B, rowvar=False) - 2 * cm)
print(round(float(fr), 3))                                  # expected output: 0.099
gen = rng.random((50, 64)); real = load_digits().images.reshape(-1, 64) / 16.0
print(round(float(np.sqrt(((gen[:, None] - real[None])**2).sum(2)).min(1).mean()), 3))  # expected output: 3.534
clf = LogisticRegression(max_iter=300).fit(real, load_digits().target)
print(sum(1 for c in range(10) if np.mean(clf.predict(gen) == c) > 0.02), "/10")  # expected output: 8 /10

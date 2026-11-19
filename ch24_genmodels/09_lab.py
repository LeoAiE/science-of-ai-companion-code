# %% [markdown]
# # Section 24.9 Lab: VAE, GAN, and diffusion on digits
#
# The lab of Section 24.9 on scikit-learn's load_digits: a VAE, a GAN and a diffusion model trained at
# seed 0 and at seeds 1 to 4, then the Frechet distance and the precision and recall snippets of 24.9.1.

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

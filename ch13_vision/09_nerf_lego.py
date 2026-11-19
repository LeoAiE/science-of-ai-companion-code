# %% [markdown]
# # A small NeRF trained on the Lego scene (Example 13-28 of Section 13.8)
# A standalone companion script: it trains a neural radiance field on the 100-by-100 Lego renderings that the NeRF
# authors publish as tiny_nerf_data.npz, scores six held-out views by PSNR, and saves the first held-out view next
# to its rendering. One run took 1,346 seconds, about 22 minutes, on six CPU threads of the reference machine, so it lives outside the chapter walkthrough. Run it from
# this folder; it reads ./nerf/tiny_nerf_data.npz and downloads it there if it is missing.

# %%
import os, urllib.request
import numpy as np, torch, torch.nn as nn
from PIL import Image
torch.set_num_threads(6)                                 # six threads, one per core; the printed numbers come from this setting
torch.manual_seed(0)
path = "nerf/tiny_nerf_data.npz"
if not os.path.exists(path):
    os.makedirs("nerf", exist_ok=True)
    urllib.request.urlretrieve("https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz", path)
data = np.load(path)
images, poses = torch.tensor(data["images"]), torch.tensor(data["poses"])
focal = float(data["focal"])
H, W = images.shape[1:3]
print("images", tuple(images.shape), "poses", tuple(poses.shape), f"focal {focal:.2f}")

def camera_rays(pose):                                   # one ray per pixel of a pinhole camera
    row, col = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    cam = torch.stack([(col - W / 2) / focal, -(row - H / 2) / focal, -torch.ones_like(col)], -1)   # camera looks down -z
    d = cam @ pose[:3, :3].T                             # rotate the directions into the world frame
    return pose[:3, 3].expand_as(d), d                   # every ray starts at the camera center

train_ids, test_ids = range(100), range(100, 106)
rays = [camera_rays(poses[k]) for k in train_ids]
O = torch.cat([o.reshape(-1, 3) for o, _ in rays]); D = torch.cat([d.reshape(-1, 3) for _, d in rays])
C = images[:100].reshape(-1, 3)                          # 1,000,000 training rays and their true colors
print("training rays", O.shape[0])

L = 6                                                    # frequencies 2^0 pi ... 2^5 pi
def encode(p):                                           # gamma(p) of Mildenhall et al., plus p itself
    a = p[..., None] * (2.0 ** torch.arange(L)) * torch.pi
    return torch.cat([p, a.sin().flatten(-2), a.cos().flatten(-2)], -1)

def make_field():                                        # (x, y, z) -> density and color
    return nn.Sequential(nn.Linear(3 + 6 * L, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 4))

def render(field, o, d, n=64, near=2.0, far=6.0, jitter=False):
    u = torch.rand(o.shape[0], n) if jitter else torch.full((o.shape[0], n), 0.5)   # where inside each bin
    t = near + (torch.arange(n) + u) * (far - near) / n  # one depth in each of n equal bins between near and far
    out = field(encode(o[:, None] + d[:, None] * t[..., None]))
    sigma, rgb = torch.relu(out[..., 3]), torch.sigmoid(out[..., :3])
    delta = torch.cat([t[:, 1:] - t[:, :-1], torch.full_like(t[:, :1], 1e10)], 1) * d.norm(dim=-1, keepdim=True)
    alpha = 1 - torch.exp(-sigma * delta)                # the opacity of each sample, as in 13.8
    sd = sigma * delta
    T = torch.exp(-torch.cumsum(torch.cat([torch.zeros_like(sd[:, :1]), sd[:, :-1]], 1), 1))   # T_i = exp(-sum_{j<i} sigma_j delta_j)
    return ((T * alpha)[..., None] * rgb).sum(1)         # the quadrature: sum of T_i (1 - e^(-sigma_i delta_i)) c_i

def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean()))

def render_view(field, k):
    o, d = camera_rays(poses[k])
    with torch.no_grad():
        return torch.cat([render(field, o.reshape(-1, 3)[i:i + 2000], d.reshape(-1, 3)[i:i + 2000])
                          for i in range(0, H * W, 2000)]).reshape(H, W, 3)

# %% [markdown]
# Two baselines that need no training, then the field itself: 6,000 Adam steps of 1,024 random rays.

# %%
mean_img = images[:100].mean(0)
print(f"mean training image   test PSNR {np.mean([psnr(mean_img, images[k]) for k in test_ids]):.2f} dB")
centers = poses[:100, :3, 3]
nearest = [int(((centers - poses[k, :3, 3]) ** 2).sum(1).argmin()) for k in test_ids]
print(f"nearest training view test PSNR {np.mean([psnr(images[j], images[k]) for j, k in zip(nearest, test_ids)]):.2f} dB")

field = make_field()
opt = torch.optim.Adam(field.parameters(), lr=5e-4)
for step in range(1, 6001):
    idx = torch.randint(0, O.shape[0], (1024,))
    loss = ((render(field, O[idx], D[idx], jitter=True) - C[idx]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (100, 500, 1000, 2000, 4000, 6000):
        test_psnr = np.mean([psnr(render_view(field, k), images[k]) for k in test_ids])
        print(f"step {step:4d}  train MSE {loss.item():.4f}  test PSNR {test_psnr:.2f} dB", flush=True)

k = test_ids[0]
pair = torch.cat([images[k], render_view(field, k).clamp(0, 1)], 1)   # held-out view, then the rendering
Image.fromarray((pair.numpy() * 255).astype("uint8")).resize((400, 200), Image.NEAREST).save("nerf_holdout_render.png")
print("saved nerf_holdout_render.png")

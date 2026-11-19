# %% [markdown]
# # Section 8.7 - Lab: customer segmentation (Wholesale Customers) and digit visualization
# Regenerated from Book.md (single source of truth). Every block
# is a book code example; run top to bottom to reproduce the printed numbers.

# %%
import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
ds = fetch_openml("wholesale-customers", as_frame=True, parser="auto")
# OpenML names the columns V1..V7: V1 = Region (1 Lisbon, 2 Oporto, 3 other) and
# V2..V7 = the six annual spends (Fresh, Milk, Grocery, Frozen, Detergents_Paper,
# Delicassen); the frame repeats the region as a categorical Region column, and the
# target is Channel. We keep only the six spends, dropping Region and holding out Channel.
spend = ds.data[["V2", "V3", "V4", "V5", "V6", "V7"]].copy()
spend.columns = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
channel = ds.target.astype(int).to_numpy()    # 1 = Horeca (hotel/restaurant/cafe), 2 = Retail
raw = spend.to_numpy(float)
print(raw.min(0).astype(int).tolist(), raw.max(0).astype(int).tolist(), int((raw < 100).sum()), int((raw < 100).any(1).sum()))
# [3, 55, 3, 25, 3, 3] [112151, 73498, 92780, 60869, 40827, 47943] 95 79
X = np.log1p(raw)                              # spends run from 3 to 112151
Xs = StandardScaler().fit_transform(X)
for k in range(2, 8):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
    print(k, f"{silhouette_score(Xs, lab):.3f}")
# 2 0.290   3 0.259   4 0.188   5 0.195   6 0.193   7 0.196
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Xs)
print(round(adjusted_rand_score(channel, km.labels_), 3))   # 0.514  vs the held-out channel label
prof = spend.copy(); prof["segment"] = km.labels_
print(prof["segment"].value_counts().sort_index().tolist())
# [252, 188]
print(prof.groupby("segment").median().round(0).astype(int))
# segment 0 (n=252): Fresh 10348  Milk 1871  Grocery  2406  Frozen 2225  Deterg  328  Delic  750
# segment 1 (n=188): Fresh  5406  Milk 7690  Grocery 11527  Frozen 1062  Deterg 4495  Delic 1388
lab2 = km.labels_
lab3 = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Xs)
print(sorted(int((lab3 == j).sum()) for j in range(3)))
fresh_id = int(np.argmax([np.median(raw[lab2 == j, 0]) for j in range(2)]))
for j in range(3):
    m = lab3 == j
    med = np.median(raw[m], axis=0).round(0).astype(int)
    print(j, int(m.sum()), int(((lab2 == fresh_id) & m).sum()), med.tolist())
# [81, 147, 212]
# 0 81 9 [1454, 6264, 10487, 388, 4196, 436]
# 1 147 32 [12126, 7184, 9965, 2005, 3378, 2005]
# 2 212 211 [9635, 1598, 2151, 2151, 274, 686]

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, trustworthiness
import umap
rng = np.random.default_rng(0)
Xd, yd = load_digits(return_X_y=True)
idx = rng.choice(len(Xd), 1000, replace=False); Xd, yd = Xd[idx], yd[idx]
emb = {"PCA": PCA(2).fit_transform(Xd),
       "Kernel PCA": KernelPCA(n_components=2, kernel="rbf", gamma=0.001).fit_transform(Xd),
       "t-SNE": TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(Xd),
       "UMAP": umap.UMAP(n_components=2, random_state=0).fit_transform(Xd)}
for name, e in emb.items():
    print(name, round(trustworthiness(Xd, e, n_neighbors=10), 3))
# PCA 0.825   Kernel PCA 0.82   t-SNE 0.991   UMAP 0.987   (t-SNE and UMAP best preserve neighbors)

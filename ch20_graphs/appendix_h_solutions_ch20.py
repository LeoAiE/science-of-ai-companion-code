"""Appendix H solutions for Chapter 20, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 20.6 coding exercises

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, networkx as nx

# shared real networks (both ship with NetworkX, fully offline)
G = nx.karate_club_graph(); N = G.number_of_nodes()
club = np.array([0 if G.nodes[v]["club"] == "Mr. Hi" else 1 for v in range(N)])
A = nx.to_numpy_array(G, weight=None)                     # unweighted 34x34 adjacency

# %% [markdown]
# 1. Adjacency, degree, and the homophily vote, unweighted and weighted

# %%
# 1. Adjacency, degree, and the homophily vote, unweighted and weighted
deg = A.sum(1).astype(int)                                # degree = row sum
print("nodes, edges:", N, G.number_of_edges())
print("min/max/mean degree:", deg.min(), deg.max(), round(deg.mean(), 2))
W = nx.to_numpy_array(G)                                  # weighted: NetworkX stores the number of contexts two members interacted in
def neighbor_vote(M, i):
    wts = M[i]                                            # 1 for every neighbor, or that interaction count
    return int(round(float((wts * club).sum() / wts.sum())))   # weighted share of faction 1, rounded
pred_u = np.array([neighbor_vote(A, i) for i in range(N)])
pred_w = np.array([neighbor_vote(W, i) for i in range(N)])
print("neighbor-vote accuracy, unweighted:", round((pred_u == club).mean(), 2), " missed:", np.where(pred_u != club)[0].tolist())
print("neighbor-vote accuracy, weighted  :", round((pred_w == club).mean(), 2), " missed:", np.where(pred_w != club)[0].tolist())
print("members whose vote changes:", np.where(pred_u != pred_w)[0].tolist())
# expected output: nodes, edges: 34 78
# expected output: min/max/mean degree: 1 17 4.59
# expected output: neighbor-vote accuracy, unweighted: 0.91  missed: [8, 9, 30]
# expected output: neighbor-vote accuracy, weighted  : 0.97  missed: [8]
# expected output: members whose vote changes: [9, 30]
# Members 9 and 30 are exact ties in the unweighted vote (1 of 2 and 2 of 4 neighbors on each side), and
# int(round(0.5)) = 0 files both under Mr. Hi although both belong to the Officer's faction; the interaction
# counts tip both toward the Officer's side (2 of 3 and 6 of 11), so the weighted vote misses only member 8.

# %% [markdown]
# 2. PageRank by power iteration on Les Misérables

# %%
# 2. PageRank by power iteration on Les Misérables
LM = nx.les_miserables_graph()
nodes = sorted(LM.nodes()); n = len(nodes)
Alm = nx.to_numpy_array(LM, nodelist=nodes, weight=None)
M = Alm / np.maximum(Alm.sum(0), 1)                       # column-stochastic transition
d = 0.85; r = np.ones(n) / n
for it in range(200):
    r_new = (1 - d) / n + d * (M @ r)
    if np.abs(r_new - r).sum() < 1e-12: r = r_new; break
    r = r_new
nxr = nx.pagerank(LM, alpha=0.85, weight=None)
maxdiff = max(abs(nxr[nodes[k]] - r[k]) for k in range(n))
top = sorted(zip(nodes, r), key=lambda x: -x[1])[:3]
print("converged iters:", it + 1)
print("top-3:", [(nm, round(float(s), 3)) for nm, s in top])
print("max diff vs networkx:", f"{maxdiff:.1e}")
print("networkx stopping threshold len(G)*tol:", f"{n * 1e-6:.1e}")   # its rule: stop once the summed change < len(G) * tol
nxr12 = nx.pagerank(LM, alpha=0.85, weight=None, tol=1e-12, max_iter=1000)   # the library run to the loop's tolerance
print("max diff with networkx tol=1e-12:", f"{max(abs(nxr12[nodes[k]] - r[k]) for k in range(n)):.1e}")
# expected output: converged iters: 93
# expected output: top-3: [('Valjean', 0.075), ('Myriel', 0.043), ('Gavroche', 0.036)]
# expected output: max diff vs networkx: 2.4e-05
# expected output: networkx stopping threshold len(G)*tol: 7.7e-05
# expected output: max diff with networkx tol=1e-12: 3.3e-11
# The 2.4e-05 gap is not a failed match. nx.pagerank stops once the summed absolute change over all nodes falls
# below len(G) * tol, 77 x 1e-06 = 7.7e-05 here (NetworkX developers, 2026, pagerank page; NetworkX 3.7), so
# the loop above, run to 1e-12, is the more converged of the two, and passing tol=1e-12 to the library closes the
# gap to 3.3e-11.

# %% [markdown]
# 3. Graph Laplacian and the Fiedler split

# %%
# 3. Graph Laplacian and the Fiedler split
degk = A.sum(1)
Lap = np.diag(degk) - A                                   # L = D - A, on the karate club
Dinv = np.diag(1.0 / np.sqrt(degk))
Lsym = np.eye(N) - Dinv @ A @ Dinv                        # L_sym = I - D^-1/2 A D^-1/2
esym = np.linalg.eigvalsh(Lsym)
w = np.linalg.eigvalsh(Lap)
print("L=D-A rows sum to 0:", np.allclose(Lap.sum(1), 0))
# the solver's smallest eigenvalue is within rounding error of 0 (about 1e-16, and sometimes negative), so + 0.0 turns a -0.0 into 0.0
print("L_sym eigenvalue range:", round(float(esym[0]), 3) + 0.0, round(float(esym[-1]), 3) + 0.0)
print("smallest Lap eigenvalue:", round(float(w[0]), 6) + 0.0)
degm = Alm.sum(1)                                         # the Fiedler split, on the Les Miserables graph
Lm = np.diag(degm) - Alm
Dm = np.diag(1.0 / np.sqrt(degm)); em = np.linalg.eigvalsh(np.eye(n) - Dm @ Alm @ Dm)
print("Les Miserables: L rows sum to 0:", np.allclose(Lm.sum(1), 0), " L_sym range:", round(float(em[0]), 3) + 0.0, round(float(em[-1]), 3) + 0.0)
wm, Vm = np.linalg.eigh(Lm)
side = (Vm[:, 1] > 0).astype(int)                         # sign of the Fiedler vector (its overall sign is arbitrary)
vj = side[nodes.index("Valjean")]                         # so name the two sides by who is on them
print("Fiedler split: side with Valjean", int((side == vj).sum()), "characters, other side", int((side != vj).sum()))
print("Javert, Cosette on Valjean's side:", [bool(side[nodes.index(c)] == vj) for c in ("Javert", "Cosette")])
print("other side:", [nodes[k] for k in range(n) if side[k] != vj])
print("edges crossing the split:", int(sum(Alm[i, j] for i in range(n) for j in range(i + 1, n) if side[i] != side[j])))
# expected output: L=D-A rows sum to 0: True
# expected output: L_sym eigenvalue range: 0.0 1.715
# expected output: smallest Lap eigenvalue: 0.0
# expected output: Les Miserables: L rows sum to 0: True  L_sym range: 0.0 1.865
# expected output: Fiedler split: side with Valjean 67 characters, other side 10
# expected output: Javert, Cosette on Valjean's side: [True, True]
# expected output: other side: ['Champtercier', 'Count', 'CountessDeLo', 'Cravatte', 'Geborand', 'MlleBaptistine', 'MmeMagloire', 'Myriel', 'Napoleon', 'OldMan']
# expected output: edges crossing the split: 3
# This graph has no faction labels, so the split is read from who lands where: the Fiedler vector cuts off a group
# of ten around the bishop Myriel, joined to the rest of the story by 3 of the 254 edges, while Valjean, Javert,
# and Cosette sit on the large side with the other 64 characters.

# %% [markdown]
# 4. DeepWalk node embeddings compared with adjacency rows, two walk settings

# %%
# 4. DeepWalk node embeddings compared with adjacency rows, two walk settings
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
neighbors = [np.where(A[i])[0] for i in range(N)]
def walk_ari(seed, length, window):
    rng = np.random.default_rng(seed)
    def walk(start):
        w = [start]
        for _ in range(length - 1):
            nb = neighbors[w[-1]]
            if len(nb) == 0: break
            w.append(int(rng.choice(nb)))                # unbiased step = DeepWalk
        return w
    walks = [walk(s) for s in range(N) for _ in range(10)]   # ten walks from every member
    C = np.zeros((N, N))
    for wk in walks:
        for i, node in enumerate(wk):
            for j in range(max(0, i - window), min(len(wk), i + window + 1)):
                if i != j: C[node, wk[j]] += 1            # windowed cooccurrence
    P = C / (C.sum(1, keepdims=True) + 1e-9)
    U, S, _ = np.linalg.svd(np.log(P + 1e-5), full_matrices=False)
    emb = U[:, :16] * S[:16]                              # 16-dim embeddings
    return adjusted_rand_score(club, KMeans(2, n_init=10, random_state=0).fit_predict(emb))
for length, window in [(20, 4), (40, 5)]:                 # Section 20.2's setting, then longer walks and a wider window
    aris = [walk_ari(seed, length, window) for seed in range(10)]
    print(f"walk length {length}, window {window}: ARI seed 0 {aris[0]:.2f}, mean of 10 seeds {np.mean(aris):.2f}, "
          f"seeds at 0.77: {sum(round(a, 2) == 0.77 for a in aris)} of 10")
A_unw = nx.to_numpy_array(G, weight=None)
A_w = nx.to_numpy_array(G)                                # interaction counts, a different matrix
print("ARI unweighted rows:", round(adjusted_rand_score(club, KMeans(2, n_init=10, random_state=0).fit_predict(A_unw)), 2))
print("ARI weighted rows:", round(adjusted_rand_score(club, KMeans(2, n_init=10, random_state=0).fit_predict(A_w)), 2))
# expected output: walk length 20, window 4: ARI seed 0 0.77, mean of 10 seeds 0.63, seeds at 0.77: 3 of 10
# expected output: walk length 40, window 5: ARI seed 0 0.77, mean of 10 seeds 0.75, seeds at 0.77: 9 of 10
# expected output: ARI unweighted rows: 0.88
# expected output: ARI weighted rows: 0.11
# Longer walks and a wider window give every pair more cooccurrence counts, so the factorization depends less on
# one seed's walks: seed 0 stays at 0.77 and the ten seed mean rises from 0.63 to 0.75, with nine seeds instead
# of three at 0.77. The two adjacency rows use no walks, so their 0.88 and 0.11 do not move.

# %% [markdown]
# 5. TransE link prediction on FB15k-237

# %%
# 5. TransE link prediction on FB15k-237 (Example 20-6's loop; the loader downloads the files once from GitHub)
import os, tempfile, torch
from torch_geometric.datasets import FB15k_237
torch.manual_seed(0); torch.set_num_threads(2)
root = os.path.join(tempfile.gettempdir(), "fb15k237")
splits = {s: FB15k_237(root, split=s)[0] for s in ["train", "val", "test"]}
facts = {s: torch.stack([d.edge_index[0], d.edge_type, d.edge_index[1]], 1) for s, d in splits.items()}
E = splits["train"].num_nodes; R = int(max(d.edge_type.max() for d in splits.values())) + 1
print("entities, relations, facts per split:", E, R, {s: len(f) for s, f in facts.items()})
g = torch.Generator().manual_seed(0)
dim, margin, bs, epochs = 50, 1.0, 1024, 20
ent = torch.nn.Parameter(torch.randn(E, dim, generator=g) * 0.1)
rel = torch.nn.Parameter(torch.randn(R, dim, generator=g) * 0.1)
opt = torch.optim.Adam([ent, rel], lr=0.01); tr = facts["train"]
for epoch in range(epochs):
    order = torch.randperm(len(tr), generator=g)
    for i in range(0, len(tr), bs):
        b = tr[order[i:i + bs]]; h, rr, t = b[:, 0], b[:, 1], b[:, 2]
        tn = torch.randint(0, E, (len(b),), generator=g)              # corrupt the tail
        e = torch.nn.functional.normalize(ent, dim=1)                 # unit entity vectors
        eh, er = e.index_select(0, h), rel.index_select(0, rr)        # index_select: a reproducible backward
        pos = (eh + er - e.index_select(0, t)).norm(dim=1); neg = (eh + er - e.index_select(0, tn)).norm(dim=1)
        loss = torch.clamp(margin + pos - neg, min=0).mean()          # margin ranking loss
        opt.zero_grad(); loss.backward(); opt.step()
known = {}                                                            # every true tail of (head, relation)
for s in facts:
    for h, rr, t in facts[s].tolist(): known.setdefault((h, rr), set()).add(t)
e = torch.nn.functional.normalize(ent, dim=1).detach(); rl = rel.detach()
raw, filt = [], []
with torch.no_grad():
    te = facts["test"]
    for i in range(0, len(te), 256):
        b = te[i:i + 256]; D = torch.cdist(e[b[:, 0]] + rl[b[:, 1]], e)   # nearer entity = better tail
        for j, (h, rr, t) in enumerate(b.tolist()):
            d_true = D[j, t]; raw.append(int((D[j] < d_true).sum()) + 1)
            D[j, [x for x in known[(h, rr)] if x != t]] = float("inf")   # the filtered protocol
            filt.append(int((D[j] < d_true).sum()) + 1)
for name, rk in [("raw", np.array(raw)), ("filtered", np.array(filt))]:
    print(f"{name}: Hits@1 {np.mean(rk == 1):.3f}, Hits@10 {np.mean(rk <= 10):.3f}, MRR {np.mean(1 / rk):.3f}, "
          f"mean rank {rk.mean():.1f}")
print(f"random baseline: Hits@1 {1 / E:.5f}, Hits@10 {10 / E:.5f}, mean rank {(E + 1) / 2:.1f}")
# expected output: entities, relations, facts per split: 14541 237 {'train': 272115, 'val': 17535, 'test': 20466}
# expected output: raw: Hits@1 0.143, Hits@10 0.371, MRR 0.219, mean rank 241.0
# expected output: filtered: Hits@1 0.186, Hits@10 0.453, MRR 0.275, mean rank 220.6
# expected output: random baseline: Hits@1 0.00007, Hits@10 0.00069, mean rank 7271.0

# %% [markdown]
# 6. From-scratch GCN vs MLP on the karate club (4 seed labels)

# %%
# 6. From-scratch GCN vs MLP on the karate club (4 seed labels)
import torch.nn as nn
np.random.seed(0); torch.manual_seed(0); torch.set_num_threads(2)
A_tilde = A + np.eye(N); dd = A_tilde.sum(1)
A_hat = torch.tensor(A_tilde / np.sqrt(np.outer(dd, dd)), dtype=torch.float32)  # D~^-1/2 (A+I) D~^-1/2
X = torch.eye(N); yt = torch.tensor(club)
seeds = [0, 33, 16, 24]      # the two leaders plus one more member of each faction: 16 (Mr. Hi) and 24 (Officer)
train_m = torch.zeros(N, dtype=torch.bool); train_m[seeds] = True; test_m = ~train_m
class GCN(nn.Module):
    def __init__(self): super().__init__(); self.l1 = nn.Linear(N, 16); self.l2 = nn.Linear(16, 2)
    def forward(self, X, Ah): return Ah @ self.l2(torch.relu(Ah @ self.l1(X)))   # aggregate twice
class MLP(nn.Module):
    def __init__(self): super().__init__(); self.l1 = nn.Linear(N, 16); self.l2 = nn.Linear(16, 2)
    def forward(self, X, Ah): return self.l2(torch.relu(self.l1(X)))             # never uses Ah
def fit(model):
    opt = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
    for _ in range(200):
        out = model(X, A_hat); loss = nn.functional.cross_entropy(out[train_m], yt[train_m])
        opt.zero_grad(); loss.backward(); opt.step()
    pred = model(X, A_hat).argmax(1)
    missed = np.where(((pred != yt) & test_m).numpy())[0].tolist()
    return round((pred[test_m] == yt[test_m]).float().mean().item(), 2), missed
print("MLP test acc, missed members:", *fit(MLP()))
print("GCN test acc, missed members:", *fit(GCN()))
for v in (2, 8):                                          # the two members the GCN gets wrong
    print(f"member {v} ({G.nodes[v]['club']}): friends in the other faction",
          [int(u) for u in np.where(A[v])[0] if club[u] != club[v]], "of", int(A[v].sum()))
# expected output: MLP test acc, missed members: 0.5 [9, 14, 15, 18, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32]
# expected output: GCN test acc, missed members: 0.93 [2, 8]
# expected output: member 2 (Mr. Hi): friends in the other faction [9, 27, 28, 32] of 10
# expected output: member 8 (Mr. Hi): friends in the other faction [30, 32, 33] of 5
# With identity features every member is its own input column, and the loss never touches a column that belongs
# to an unlabeled member, so nothing learned from the four labeled rows transfers to the other 30: the MLP gives
# them all the same output and files every one under Mr. Hi, right for the 15 Mr. Hi members and wrong for the
# 15 Officer members, 0.5. The GCN reaches 0.93 from four labels against 0.97 from two in Example 20-14: it still
# misses member 8, three of whose five friends are on the Officer's side, and now also member 2, who has four
# Officer friends among ten; both sit on the boundary between the factions, where the aggregate is close and a
# different set of labels can tip it.

# %% [markdown]
# 7. Link prediction with a leakage check on Les Misérables (30 percent hidden, seed 1)

# %%
# 7. Link prediction with a leakage check on Les Misérables (30 percent hidden, seed 1)
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(1)
edges = np.argwhere(np.triu(Alm, 1) > 0)
rng.shuffle(edges); n_hold = int(round(0.3 * len(edges)))  # hide 30 percent of the edges
held = edges[:n_hold]
A_train = Alm.copy()
for i, j in held: A_train[i, j] = A_train[j, i] = 0        # delete held-out edges: no leakage
alive = A_train.sum(1) > 0                                 # characters that keep at least one training edge
held = held[alive[held[:, 0]] & alive[held[:, 1]]]         # a character with no edge left has a zero embedding: drop those pairs
def factors(mat):
    Uu, Ss, Vt = np.linalg.svd(mat, full_matrices=False)
    return Uu[:, :16], Ss[:16], Vt[:16]
U2, S2, Vt2 = factors(A_train)
emb2 = U2 * np.sqrt(S2)                                    # Example 20-16's embedding: its dot products are entries of U S U^T
rec2 = (U2 * S2) @ Vt2                                     # the rank-16 reconstruction U S V^T, which keeps the signs
neg = []
while len(neg) < len(held):                                # equal-size negative sampling
    i, j = int(rng.integers(n)), int(rng.integers(n))
    if i != j and Alm[i, j] == 0: neg.append((i, j))
y = [1]*len(held) + [0]*len(held)
pairs = [tuple(p) for p in held] + neg
UL, SL, VtL = factors(Alm)                                 # LEAKY: factorizes the matrix with the held-out edges still in it
embL = UL * np.sqrt(SL); recL = (UL * SL) @ VtL
print("nodes, edges, hidden:", n, len(edges), n_hold, " dropped (an endpoint keeps no edge):", n_hold - len(held))
print("U sqrt(S) dot product honest ROC AUC:", round(roc_auc_score(y, [emb2[i] @ emb2[j] for i, j in pairs]), 2),
      " leaky ROC AUC:", round(roc_auc_score(y, [embL[i] @ embL[j] for i, j in pairs]), 2))
print("rank-16 U S V^T entry honest ROC AUC:", round(roc_auc_score(y, [rec2[i, j] for i, j in pairs]), 2),
      " leaky ROC AUC:", round(roc_auc_score(y, [recL[i, j] for i, j in pairs]), 2))
print("negative singular directions among the top 16 of A_train:", int((np.einsum("ij,ij->j", U2, Vt2.T) < 0).sum()))
# expected output: nodes, edges, hidden: 77 254 76  dropped (an endpoint keeps no edge): 3
# expected output: U sqrt(S) dot product honest ROC AUC: 0.92  leaky ROC AUC: 0.86
# expected output: rank-16 U S V^T entry honest ROC AUC: 0.73  leaky ROC AUC: 1.0
# expected output: negative singular directions among the top 16 of A_train: 8
# The adjacency is symmetric, so its SVD returns the absolute eigenvalues as S with V_k = sign(lambda_k) U_k: a dot
# product of two U sqrt(S) rows is an entry of U S U^T, which flips the 8 negative directions and is not a
# reconstruction of A, so leaving the hidden edges in does not lift that score (0.86 against 0.92). An entry of
# U S V^T is the rank-16 reconstruction, and there the leak shows: 0.73 with the hidden edges deleted, 1.0 with
# them left in, because the reconstruction of the full matrix reproduces the very entries being predicted.

# %% [markdown]
# 8. Oversmoothing: repeated aggregation shrinks the spread of the node vectors

# %%
# 8. Oversmoothing: repeated aggregation shrinks the spread of the node vectors
Ah = A_tilde / np.sqrt(np.outer(dd, dd))
def spread(Mm): return float(np.mean(np.var(Mm, axis=0)))
H = np.eye(N); s0 = spread(H); disp = []
for L in range(21):
    if L in (0, 2, 5, 20): disp.append((L, round(spread(H) / s0, 3)))
    H = Ah @ H                                            # one more round of averaging
print("relative feature spread at layers 0,2,5,20:", disp)
for L in range(21, 101):                                  # keep going: the spread stops falling
    if L in (50, 100): print(f"relative spread at round {L}: {spread(H) / s0:.4f}")
    H = Ah @ H
u = np.sqrt(dd) / np.linalg.norm(np.sqrt(dd))            # the direction A_hat^k converges to
print("floor, the spread of u u^T with u = sqrt(self-looped degree):", round(spread(np.outer(u, u)) / s0, 4))
# expected output: relative feature spread at layers 0,2,5,20: [(0, 1.0), (2, 0.055), (5, 0.016), (20, 0.003)]
# expected output: relative spread at round 50: 0.0025
# expected output: relative spread at round 100: 0.0025
# expected output: floor, the spread of u u^T with u = sqrt(self-looped degree): 0.0025
# The spread falls by a factor of about 400 and then levels off near 0.0025 of its start rather than reaching zero: A_hat^k
# converges to u u^T with u proportional to the square root of the self-looped degree, as Section 20.3.2 states,
# so the node vectors line up with that one direction, each scaled by its degree, instead of becoming copies of
# one vector.

# %% [markdown]
# 9. Miniproject: GCN vs MLP on CiteSeer with PyTorch Geometric

# %%
# 9. Miniproject: GCN vs MLP on CiteSeer with PyTorch Geometric
import os, tempfile
import torch.nn.functional as Fnn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
torch.manual_seed(0); torch.set_num_threads(2)
cs = Planetoid(root=os.path.join(tempfile.gettempdir(), "citeseer"), name="CiteSeer")[0]  # downloads once, caches
class PyGGCN(torch.nn.Module):
    def __init__(self, nf, nc, h=16):
        super().__init__(); self.c1 = GCNConv(nf, h); self.c2 = GCNConv(h, nc)
    def forward(self, x, ei):
        x = Fnn.dropout(Fnn.relu(self.c1(x, ei)), p=0.5, training=self.training)
        return self.c2(x, ei)
class PyGMLP(torch.nn.Module):
    def __init__(self, nf, nc, h=16):
        super().__init__(); self.l1 = torch.nn.Linear(nf, h); self.l2 = torch.nn.Linear(h, nc)
    def forward(self, x, ei):
        x = Fnn.dropout(Fnn.relu(self.l1(x)), p=0.5, training=self.training)
        return self.l2(x)                                 # ignores the edges ei
def run(model):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        model.train(); opt.zero_grad()
        out = model(cs.x, cs.edge_index)
        Fnn.cross_entropy(out[cs.train_mask], cs.y[cs.train_mask]).backward(); opt.step()
    model.eval()
    p = model(cs.x, cs.edge_index).argmax(1)
    return (p[cs.test_mask] == cs.y[cs.test_mask]).float().mean().item()
nf, nc = cs.num_node_features, int(cs.y.max()) + 1
print("CiteSeer:", cs.num_nodes, "papers,", cs.num_edges // 2, "citation links,", nf, "words,", nc, "classes,",
      int(cs.train_mask.sum()), "train labels,", int(cs.test_mask.sum()), "test papers")
print("MLP test acc:", round(run(PyGMLP(nf, nc)), 3))
print("GCN test acc:", round(run(PyGGCN(nf, nc)), 3))
from torch_geometric.utils import degree
alone = degree(cs.edge_index[0], cs.num_nodes) == 0       # papers with no citation link at all
print("papers with no citation links:", int(alone.sum()), " of them test papers:", int((alone & cs.test_mask).sum()))
# expected output: CiteSeer: 3327 papers, 4552 citation links, 3703 words, 6 classes, 120 train labels, 1000 test papers
# expected output: MLP test acc: 0.58
# expected output: GCN test acc: 0.688
# expected output: papers with no citation links: 48  of them test papers: 12
# CiteSeer is one of the three citation networks of the Planetoid split (Yang, Cohen, and Salakhutdinov, 2016),
# which PyTorch Geometric downloads once from the same GitHub repository as Cora: 3327 papers, each a 3703-word
# bag-of-words vector, 6 classes, and 120 labeled papers. The same two models and the same 200 epochs give 0.58
# for the words alone and 0.688 with the citation links, a lift of about eleven points against the twenty-five
# on Cora. Forty-eight papers, 12 of them test papers, have no citation link at all. GCNConv adds a self loop to
# every paper, as Section 20.3.1's A + I does (PyG Team, 2026a, the GCNConv page), so for such a paper each layer
# sees only the paper's own words: the layer still applies the weights it learned on the whole graph, but it cannot
# bring in any neighbor's signal, so for those 48 papers message passing adds nothing to the words.

# %% [markdown]
# ## 20.6 coding drills

# %%
import numpy as np, networkx as nx

# %% [markdown]
# Graph representation

# %%
# Graph representation
el = [(0,1),(0,2),(1,2),(2,3)]
Ad = np.zeros((4,4), int)
for i,j in el: Ad[i,j]=Ad[j,i]=1                          # symmetric adjacency from an edge list
print(Ad.tolist())                                        # expected output: [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]]
print(Ad.sum(1).tolist())                                 # degrees; expected output: [2, 2, 3, 1]
print(int(Ad.sum() // 2))                                 # edge count; expected output: 4
print(np.array_equal(Ad, Ad.T))                           # symmetric?; expected output: True
print([(i,j) for i in range(4) for j in range(i+1,4) if Ad[i,j]])  # back to edge list; expected output: [(0, 1), (0, 2), (1, 2), (2, 3)]
print(round(Ad.sum() / (4*3), 3))                         # density = edges/possible; expected output: 0.667

# %% [markdown]
# PageRank and the Laplacian

# %%
# PageRank and the Laplacian
Gk = nx.karate_club_graph(); Ak = nx.to_numpy_array(Gk, weight=None)
Mk = Ak / Ak.sum(0)
r1 = 0.15/34 + 0.85 * (Mk @ (np.ones(34)/34))             # one power-iteration step
print(round(r1.sum(), 3))                                 # still a distribution; expected output: 1.0
print(max(nx.pagerank(Gk, weight=None).items(), key=lambda x: x[1])[0])  # top-PageRank node, unweighted as in Section 20.2.1; expected output: 33
Lk = np.diag(Ak.sum(1)) - Ak                              # L = D - A
print(np.allclose(Lk.sum(1), 0))                          # rows sum to zero; expected output: True
print(int(np.sum(np.linalg.eigvalsh(Lk) < 1e-9)))         # #connected components; expected output: 1

# %% [markdown]
# Embeddings and walks

# %%
# Embeddings and walks
rng = np.random.default_rng(1)
nb = [np.where(Ak[i])[0] for i in range(34)]
wlk = [0]
for _ in range(5): wlk.append(int(rng.choice(nb[wlk[-1]])))  # length-6 random walk
print(wlk)                                                # expected output: [0, 8, 30, 33, 32, 2]
Uu, Ss, _ = np.linalg.svd(Ak, full_matrices=False)
emb = Uu[:, :8] * np.sqrt(Ss[:8])                         # 8-dim SVD embeddings
print(emb.shape)                                          # expected output: (34, 8)
def cos(a,b): return float(a@b / (np.linalg.norm(a)*np.linalg.norm(b)))
print(round(cos(emb[0], emb[33]), 2))                     # cosine of two node vectors; expected output: 0.11

# %% [markdown]
# Message passing and the GCN

# %%
# Message passing and the GCN
At = Ak + np.eye(34); dt = At.sum(1)
Aht = At / np.sqrt(np.outer(dt, dt))                      # A_hat = D~^-1/2 (A+I) D~^-1/2
print(round(Aht[0,1], 3), round(1/np.sqrt(dt[0]*dt[1]), 3))  # off-diagonal = 1/sqrt(dv du); expected output: 0.077 0.077
feat = np.arange(34).reshape(-1,1).astype(float)
print(round(float((Aht @ feat)[0,0]), 2))                # one aggregation of node 0; expected output: 20.26

# %% [markdown]
# Link-prediction metrics

# %%
# Link-prediction metrics
from sklearn.metrics import roc_auc_score
print(round(float(emb[0] @ emb[1]), 3))                   # dot-product pair score; expected output: 1.04
sc_pos = np.array([0.9, 0.4, 0.7]); sc_neg = np.array([0.2, 0.5, 0.1])
print(round(roc_auc_score([1,1,1,0,0,0], list(sc_pos)+list(sc_neg)), 2))  # expected output: 0.89
rk = 1 + int(np.sum(sc_neg >= sc_pos[0]))                 # rank of a true edge among negatives
print(rk, round(1/rk, 2), int(rk <= 1))                   # rank, reciprocal rank, Hits@1; expected output: 1 1.0 1

# %% [markdown]
# TransE arithmetic

# %%
# TransE arithmetic
h = np.array([1.,0.,0.]); r = np.array([0.,1.,0.]); t = np.array([1.,1.,0.])
print(round(float(np.linalg.norm(h + r - t)), 3))         # translation residual; expected output: 0.0
cands = {"A":np.array([1.,1.,0.]), "B":np.array([0.,0.,1.]), "C":np.array([2.,2.,2.])}
print(min(cands, key=lambda k: np.linalg.norm(cands[k] - (h + r))))  # nearest tail to h+r; expected output: A

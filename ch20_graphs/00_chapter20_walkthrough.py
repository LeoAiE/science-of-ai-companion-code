"""Chapter 20: Graph Machine Learning

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from the chapter text; keep the book as the source.
"""

# %% [markdown]
# ## 20.1 When data is a graph

# %% [markdown]
# ## 20.1.1 Connections are the signal

# %% [markdown]
# Example 20-1. A graph, its adjacency matrix, and homophily.

# %%
import numpy as np
import networkx as nx

# A constructed 8-node toy, small enough to print a full legible adjacency matrix.
names = ["Ana", "Ben", "Cy", "Dee", "Eli", "Fay", "Gus", "Hal"]
edges = [(0, 1), (0, 2), (1, 2), (2, 3),          # cluster A: Ana,Ben,Cy,Dee
         (4, 5), (4, 6), (5, 6), (6, 7),          # cluster B: Eli,Fay,Gus,Hal
         (3, 4)]                                   # one bridge between the clusters
n = len(names)
A = np.zeros((n, n), int)
for i, j in edges:
    A[i, j] = A[j, i] = 1                          # undirected: symmetric
print("constructed toy adjacency matrix (rows/cols = people):")
print(A)
print(f"degree of each node (number of friends) = {A.sum(1).tolist()}")
print(f"edge list ({len(edges)} edges): {edges}")
print("a dense matrix is n^2; an edge list is O(#edges). Real graphs are sparse, so edge lists win.")

# A REAL network: Zachary's karate club, 34 members that split into two clubs.
G = nx.karate_club_graph()
N = G.number_of_nodes()
club = np.array([0 if G.nodes[v]["club"] == "Mr. Hi" else 1 for v in range(N)])
Ak = nx.to_numpy_array(G, weight=None)            # weight=None -> unweighted, correct degrees
print(f"\nreal graph: Zachary karate club, {N} nodes, {G.number_of_edges()} edges, two clubs")

def neighbor_vote(i):
    nb = np.where(Ak[i])[0]                        # who node i is connected to
    return int(round(club[nb].mean()))            # ties at 0.5 round to 0 (banker's rounding)
preds = np.array([neighbor_vote(i) for i in range(N)])
acc = (preds == club).mean()
print(f"predict each member's club by MAJORITY VOTE of neighbors (no personal features):")
print(f"  accuracy = {acc:.2f}  (who-you-know predicts your faction)")
print("this is homophily: linked nodes share labels, the signal GNNs exploit.")

print("\ngraph types:")
print("  undirected (friendship) vs directed (web links, who-follows-whom)")
print("  weighted (edge strengths) vs unweighted")
print("  homogeneous (one node type) vs heterogeneous / knowledge graph (typed nodes & edges)")
print("  static vs dynamic (edges appearing over time)")
print("\nAll when-data-is-a-graph checks ran.")

# %% [markdown]
# ## 20.1.2 Sparse storage, and a batch of graphs

# %% [markdown]
# Node 0's 16 neighbors are the first row of that index array.

# %%
import networkx as nx
G = nx.karate_club_graph()
A = nx.to_scipy_sparse_array(G, weight=None, format="csr")
print(A.shape, A.nnz)
print("node 0 neighbors", A.indices[A.indptr[0]:A.indptr[1]].tolist())

# %% [markdown]
# Stacked, they are one 43-node graph, and the pointer array is `[0, 17, 30, 43]`.

# %%
import warnings
warnings.filterwarnings("ignore", message=".*IProgress.*")
warnings.filterwarnings("ignore", message=".*torch.jit.script.*")   # torch 2.15 deprecates torch.jit.script, which torch-geometric 2.8.1 still calls on import
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
ds = TUDataset(root="mutag", name="MUTAG")
batch = next(iter(DataLoader(ds[:3], batch_size=3)))
print(batch.num_nodes, batch.batch.bincount().tolist(), batch.ptr.tolist())

# %% [markdown]
# ## 20.2 Graph basics and classical methods

# %% [markdown]
# ## 20.2.1 PageRank: importance from the link structure

# %% [markdown]
# One PageRank update by hand on the five-page web, from the uniform start.

# %%
import numpy as np
names = ["A", "B", "C", "D", "E"]
links = [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2), (4, 2), (2, 4)]   # (from, to)
N = len(names)
M = np.zeros((N, N))
for i, j in links:
    M[j, i] = 1                                   # column i lists the pages that page i links to
M = M / M.sum(0)                                  # divide each column by that page's out-degree
d = 0.85
r0 = np.ones(N) / N                               # the uniform start, 0.2 per page
r1 = (1 - d) / N + d * (M @ r0)                   # one update
print("M (column i = page i's out-links, each column sums to 1):")
print(M)
print("M @ r0 =", (M @ r0).round(3).tolist())
print("r1     =", r1.round(3).tolist(), " sum =", round(float(r1.sum()), 3))

# %% [markdown]
# Example 20-2. PageRank by power iteration.

# %%
import numpy as np
import networkx as nx

# A constructed toy web of five pages, small enough to trace power iteration by hand.
names = ["A", "B", "C", "D", "E"]
links = [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2), (4, 2), (2, 4)]   # C is linked to a lot
N = len(names)
links_mat = np.zeros((N, N))
for i, j in links:
    links_mat[j, i] = 1                                       # column i = out-links of page i
out_deg = links_mat.sum(0)
M = np.where(out_deg > 0, links_mat / np.maximum(out_deg, 1), 1.0 / N)     # column-stochastic transition
d = 0.85
r = np.ones(N) / N                                     # start uniform
for it in range(100):
    r_new = (1 - d) / N + d * (M @ r)
    if np.abs(r_new - r).sum() < 1e-10:
        r = r_new; break
    r = r_new
print(f"constructed toy web: converged in {it+1} iterations; PageRank sums to {r.sum():.3f}")
for name, score in sorted(zip(names, r), key=lambda x: -x[1]):
    print(f"  page {name}: PageRank {score:.3f}")
G = nx.DiGraph(); G.add_edges_from([(names[i], names[j]) for i, j in links])
nx_pr = nx.pagerank(G, alpha=0.85)
ours = {names[k]: r[k] for k in range(N)}
maxdiff = max(abs(nx_pr[nm] - ours[nm]) for nm in names)
print(f"  max difference vs networkx.pagerank = {maxdiff:.2e}  (power iteration matches the library)")

# The same algorithm on a REAL network: Zachary's karate club.
K = nx.karate_club_graph()
pr = nx.pagerank(K, alpha=0.85, weight=None)          # unweighted, the chapter's default for the karate club (Section 20.1)
print("\nreal graph PageRank on karate club, top 4 members:")
for node, score in sorted(pr.items(), key=lambda x: -x[1])[:4]:
    print(f"  node {node}: {score:.3f}  ({K.nodes[node]['club']})")
print("nodes 33 and 0 rank highest: they are the two faction leaders.")

# Personalized (topic-sensitive) PageRank: teleport to a seed set, not uniformly.
ppr = nx.pagerank(K, alpha=0.85, weight=None, personalization={0: 1.0})   # one line: bias toward node 0
print("\npersonalized PageRank (teleport only to node 0), top 4:")
for node, score in sorted(ppr.items(), key=lambda x: -x[1])[:4]:
    print(f"  node {node}: {score:.3f}")

# The graph Laplacian, the operator behind spectral methods and GCN smoothing.
Ak = nx.to_numpy_array(K, weight=None)
deg = Ak.sum(1)
Lap = np.diag(deg) - Ak                                # L = D - A
Dinv = np.diag(1.0 / np.sqrt(deg))
Lsym = np.eye(K.number_of_nodes()) - Dinv @ Ak @ Dinv  # L_sym = I - D^-1/2 A D^-1/2
evals = np.linalg.eigvalsh(Lsym)
# The solver returns a smallest eigenvalue within rounding error of zero, which can be a tiny
# negative number; rounding to three decimals and adding 0.0 turns a -0.0 into 0.0 before printing.
lo = round(float(evals[0]), 3) + 0.0
hi = round(float(evals[-1]), 3) + 0.0
print(f"\nLaplacian L = D - A: every row sums to 0 -> {np.allclose(Lap.sum(1), 0)}")
print(f"symmetric normalized L_sym eigenvalues span [{lo:.3f}, {hi:.3f}] (always within [0, 2])")
print("GCN propagation in 20.3 is I - L_sym with self-loops: a Laplacian-derived smoothing operator.")
print("\nAll PageRank checks ran.")

# %% [markdown]
# ## 20.2.2 The graph Laplacian

# %% [markdown]
# The Laplacian quadratic form on the eight-node graph of Example 20-1, for two concrete vectors.

# %%
import numpy as np
edges = [(0, 1), (0, 2), (1, 2), (2, 3), (4, 5), (4, 6), (5, 6), (6, 7), (3, 4)]
n = 8
A = np.zeros((n, n))
for i, j in edges:
    A[i, j] = A[j, i] = 1
L = np.diag(A.sum(1)) - A                                  # L = D - A
x = np.array([1, 1, 1, 1, -1, -1, -1, -1])                 # +1 on cluster A, -1 on cluster B
y = np.array([1, -1, 1, -1, 1, -1, 1, -1])                 # signs alternate along the node numbers
for name, v in [("x", x), ("y", y)]:
    quad = float(v @ L @ v)                                # the matrix product
    edge_sum = sum((v[i] - v[j]) ** 2 for i, j in edges)   # the sum over edges
    cut = sum(v[i] != v[j] for i, j in edges)              # edges whose ends differ
    print(f"{name}: x^T L x = {quad:.0f}, edge sum = {edge_sum}, edges cut = {cut}, "
          f"Rayleigh quotient = {quad / float(v @ v):.3f}")
print("lambda_2 of this graph =", round(float(np.linalg.eigvalsh(L)[1]), 3))

# %% [markdown]
# Example 20-3 cuts the karate club by the sign of its Fiedler vector.
#
# Example 20-3. The karate club's algebraic connectivity, and its cut by the sign of the Fiedler vector graded against the true factions.

# %%
import numpy as np
import networkx as nx

K = nx.karate_club_graph(); N = K.number_of_nodes()
A = nx.to_numpy_array(K, weight=None)
club = np.array([0 if K.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(N)])
L = np.diag(A.sum(1)) - A                                # L = D - A
w, V = np.linalg.eigh(L)                                 # eigenvalues ascending, eigenvectors in columns
fiedler = V[:, 1]                                        # eigenvector of the second-smallest eigenvalue
split = (fiedler > 0).astype(int)                        # cut the graph by the sign of its entries
agree = max((split == club).mean(), (split != club).mean())
print(f"algebraic connectivity lambda_2 = {w[1]:.3f}  (0 would mean a disconnected graph)")
print(f"Fiedler-vector sign split vs the true factions: agreement = {agree:.2f}")

# %% [markdown]
# ## 20.2.3 Node embeddings: DeepWalk and node2vec

# %% [markdown]
# Example 20-4. DeepWalk node embeddings from random walks, and node2vec's biased walk.

# %%
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
rng = np.random.default_rng(0)

# A REAL network: Zachary's karate club, whose two clubs are the ground-truth communities.
G = nx.karate_club_graph()
N = G.number_of_nodes()
labels = np.array([0 if G.nodes[v]["club"] == "Mr. Hi" else 1 for v in range(N)])
A = nx.to_numpy_array(G, weight=None)                 # unweighted structure for the walks
neighbors = [np.where(A[i])[0] for i in range(N)]
print(f"karate club: {N} nodes, {G.number_of_edges()} edges, two known clubs")

def random_walk(start, length=20):
    walk = [start]
    for _ in range(length - 1):
        nb = neighbors[walk[-1]]
        if len(nb) == 0: break
        walk.append(int(rng.choice(nb)))             # unbiased step = DeepWalk
    return walk
walks = [random_walk(s) for s in range(N) for _ in range(10)]   # 10 walks per node
print(f"generated {len(walks)} random walks of length 20 (each walk is a 'sentence' of node IDs)")

window = 4
C = np.zeros((N, N))
for w in walks:
    for idx, node in enumerate(w):
        for jdx in range(max(0, idx - window), min(len(w), idx + window + 1)):
            if idx != jdx: C[node, w[jdx]] += 1       # cooccurrence within the window
P = C / (C.sum(1, keepdims=True) + 1e-9)              # row-normalize
emb_mat = np.log(P + 1e-5)                            # log P(j|i), not pointwise mutual information
U, S, _ = np.linalg.svd(emb_mat, full_matrices=False)  # factorize -> embeddings
emb = U[:, :16] * S[:16]                              # 16-dim node embeddings

km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(emb)
ari = adjusted_rand_score(labels, km.labels_)
A_unw = nx.to_numpy_array(G, weight=None)            # unweighted rows, the matrix the walks use
ari_unw = adjusted_rand_score(labels, KMeans(2, n_init=10, random_state=0).fit_predict(A_unw))
A_w = nx.to_numpy_array(G)                            # weighted rows mix interaction counts
ari_w = adjusted_rand_score(labels, KMeans(2, n_init=10, random_state=0).fit_predict(A_w))
seed_aris = []
for seed in range(10):
    rng = np.random.default_rng(seed)
    walks_s = [random_walk(s) for s in range(N) for _ in range(10)]
    Cs = np.zeros((N, N))
    for w in walks_s:
        for idx, node in enumerate(w):
            for jdx in range(max(0, idx - window), min(len(w), idx + window + 1)):
                if idx != jdx: Cs[node, w[jdx]] += 1
    Ps = Cs / (Cs.sum(1, keepdims=True) + 1e-9)
    Us, Ss, _ = np.linalg.svd(np.log(Ps + 1e-5), full_matrices=False)
    embs = Us[:, :16] * Ss[:16]
    seed_aris.append(adjusted_rand_score(
        labels, KMeans(2, n_init=10, random_state=0).fit_predict(embs)))
print(f"\ncluster the 16-dim node embeddings into 2 groups, compare to the true clubs:")
print(f"  DeepWalk-style embeddings, seed 0 = {ari:.2f}  (1.0 = perfect recovery)")
print(f"  mean over 10 walk seeds           = {np.mean(seed_aris):.2f}")
print(f"  unweighted adjacency rows         = {ari_unw:.2f}")
print(f"  weighted adjacency rows           = {ari_w:.2f}")
print("on this club the unweighted one-hop rows already separate the factions.")

# node2vec: bias the SECOND-ORDER walk. The step out of the current node depends on
# where the walk came from: weight 1/p to step straight back to prev, weight 1 to a
# neighbor of prev (staying local, BFS-like), weight 1/q to range farther (DFS-like).
def biased_walk(start, p=1.0, q=1.0, length=20):
    walk = [start]
    while len(walk) < length:
        cur = walk[-1]; nb = neighbors[cur]
        if len(walk) == 1:                                # first step has no history: uniform
            walk.append(int(rng.choice(nb))); continue
        prev = walk[-2]
        wts = np.array([1.0 / p if x == prev              # returning to where we came from
                        else (1.0 if A[prev, x] else 1.0 / q)  # near prev vs ranging outward
                        for x in nb])
        walk.append(int(rng.choice(nb, p=wts / wts.sum())))
    return walk

def embed(walk_list):                                     # same count -> log -> SVD pipeline as above
    Cn = np.zeros((N, N))
    for w in walk_list:
        for idx, node in enumerate(w):
            for jdx in range(max(0, idx - window), min(len(w), idx + window + 1)):
                if idx != jdx: Cn[node, w[jdx]] += 1
    Pn = Cn / (Cn.sum(1, keepdims=True) + 1e-9)
    Un, Sn, _ = np.linalg.svd(np.log(Pn + 1e-5), full_matrices=False)
    return Un[:, :16] * Sn[:16]

print("\nnode2vec's biased walks on the same graph (p = 1, 30 walks per node, averaged over 5 restarts):")
for q_val, style in [(1.0, "p = q = 1: unbiased, the DeepWalk baseline"),
                     (0.25, "q<1: range outward, DFS-like"), (4.0, "q>1: stay local, BFS-like")]:
    aris, stay = [], []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        wl = [biased_walk(s, q=q_val) for s in range(N) for _ in range(30)]
        aris.append(adjusted_rand_score(labels, KMeans(2, n_init=10, random_state=0).fit_predict(embed(wl))))
        stay.append(np.mean([labels[v] == labels[w[0]] for w in wl for v in w[1:]]))
    print(f"  q = {q_val:<4}: later steps inside the start faction {np.mean(stay):.2f}, "
          f"ARI vs true clubs {np.mean(aris):.2f}   ({style})")
print("q = 1.0 is DeepWalk's unbiased walk (p = q = 1) at the same 30 walks per node as the biased rows.")
print("\nAll node-embedding checks ran.")

# %% [markdown]
# Example 20-5. Skip-gram with negative sampling trained from scratch on Example 20-4's walks, scored beside the log cooccurrence surrogate.

# %%
import torch
torch.manual_seed(0); torch.set_num_threads(2)

def skipgram_pairs(walk_list):                          # every (center, context) pair within Example 20-4's window
    return torch.tensor([(w[i], w[j]) for w in walk_list for i in range(len(w))
                         for j in range(max(0, i - window), min(len(w), i + window + 1)) if i != j])

def train_skipgram(walk_list, seed, dim=16, K=5, epochs=5, B=256, lr=0.01):
    """Skip-gram with negative sampling, the loss J of Section 15.2.2, from scratch in PyTorch:
    center vectors v, context vectors u, K negatives per pair from the walks' unigram counts raised to 3/4."""
    gen = torch.Generator().manual_seed(seed)
    pairs = skipgram_pairs(walk_list)
    counts = torch.bincount(torch.tensor([v for w in walk_list for v in w]), minlength=N).float()
    noise = counts ** 0.75; noise = noise / noise.sum()  # word2vec's smoothed unigram distribution
    V = torch.nn.Parameter((torch.rand(N, dim, generator=gen) - 0.5) / dim)   # center vectors: small random start
    U = torch.nn.Parameter(torch.zeros(N, dim))                                  # context vectors: zero start
    opt = torch.optim.Adam([V, U], lr=lr)
    losses = []
    for epoch in range(epochs):
        order = torch.randperm(len(pairs), generator=gen); total = 0.0
        for b in range(0, len(pairs), B):
            c, o = pairs[order[b:b + B]].T
            n = torch.multinomial(noise, len(c) * K, replacement=True, generator=gen).view(len(c), K)
            log_pos = torch.nn.functional.logsigmoid((V[c] * U[o]).sum(1))              # log sigma(u_o . v_c)
            log_neg = torch.nn.functional.logsigmoid(-(V[c].unsqueeze(1) * U[n]).sum(2)) # log sigma(-u_n . v_c)
            loss = -(log_pos + log_neg.sum(1)).mean()                                   # J, averaged over the batch
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item() * len(c)
        losses.append(total / len(pairs))
    return V.detach().numpy(), losses, len(pairs)

def ari_of(E):                                           # the same KMeans call and score as Example 20-4
    return adjusted_rand_score(labels, KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(E))

sg, sv = [], []
for seed in range(10):                                   # the same ten walk seeds as Example 20-4
    rng = np.random.default_rng(seed)                    # random_walk reads this generator
    wl = [random_walk(s) for s in range(N) for _ in range(10)]
    emb_sg, losses, n_pairs = train_skipgram(wl, seed)
    sg.append(ari_of(emb_sg)); sv.append(ari_of(embed(wl)))      # embed: Example 20-4's count, log, SVD surrogate
    if seed == 0:
        print(f"seed 0 regenerates Example 20-4's {len(wl)} walks: {wl == walks}; {n_pairs} (center, context) pairs, "
              f"K = 5 negatives each")
        print(f"  skip-gram mean loss per pair: epoch 1 {losses[0]:.3f}, epoch 5 {losses[-1]:.3f}")
print("cluster the 16-dim embeddings into 2 groups, compare to the true clubs (adjusted Rand index):")
print(f"  skip-gram with negative sampling, seed 0 = {sg[0]:.2f}   log cooccurrence SVD surrogate, seed 0 = {sv[0]:.2f}")
print(f"  mean over the same 10 walk seeds       = {np.mean(sg):.2f}   surrogate mean                     = {np.mean(sv):.2f}")
print(f"  per seed, skip-gram: {[round(a, 2) for a in sg]}")
print(f"  per seed, surrogate: {[round(a, 2) for a in sv]}")

# %% [markdown]
# ## 20.2.4 Knowledge graphs and their embeddings

# %% [markdown]
# TransE's margin ranking loss evaluated on the worked triple.

# %%
import numpy as np
h = np.array([1.0, 0.5]); r = np.array([2.0, 1.0]); t = np.array([0.5, 1.0])
gamma = 1.0                                              # the margin, 1.0 in Example 20-6
pos = np.linalg.norm(h + r - t)                          # distance of the true fact
for t_neg in [np.array([0.0, 0.0]), np.array([-1.0, 0.0])]:
    neg = np.linalg.norm(h + r - t_neg)                  # distance of the corrupted fact
    loss = max(0.0, gamma + pos - neg)
    print(f"t' = {t_neg.tolist()}: ||h + r - t|| = {pos:.4f}, ||h + r - t'|| = {neg:.4f}, "
          f"loss = max(0, {gamma:.0f} + {pos:.4f} - {neg:.4f}) = {loss:.4f}")

# %% [markdown]
# Example 20-6. TransE knowledge-graph embeddings and link prediction.

# %%
import warnings; warnings.filterwarnings("ignore")
import os, tempfile
import numpy as np
import torch
from torch_geometric.datasets import WordNet18RR
rng = np.random.default_rng(0); torch.manual_seed(0); torch.set_num_threads(2)

# Three named facts, constructed by hand, to show the relation-as-translation idea: h + r ~= t.
d0 = 4
names = ["Paris", "France", "Berlin", "Germany", "Tokyo", "Japan"]
ent0 = {nm: rng.normal(0, 1, d0) for nm in names}
capital_of = rng.normal(0, 1, d0)                     # one relation = one translation vector
for city, country in [("Paris", "France"), ("Berlin", "Germany"), ("Tokyo", "Japan")]:
    ent0[country] = ent0[city] + capital_of           # construct tails so h + r == t exactly
for city, country in [("Paris", "France"), ("Berlin", "Germany"), ("Tokyo", "Japan")]:
    err = np.linalg.norm(ent0[city] + capital_of - ent0[country])
    print(f"({city}, capital_of, {country}): ||h + r - t|| = {err:.3f}")
query = ent0["Tokyo"] + capital_of
best = min(ent0, key=lambda e: np.linalg.norm(ent0[e] - query))
print(f"nearest entity to (Tokyo + capital_of) = {best}  (link prediction by vector arithmetic)")

# A REAL knowledge graph: WordNet18RR, WordNet synsets joined by 11 relations, downloaded once from GitHub.
root = os.path.join(tempfile.gettempdir(), "wn18rr")
data = WordNet18RR(root=root)[0]
N = data.num_nodes
rel_names = sorted({line.split("\t")[1] for line in open(os.path.join(root, "raw", "train.txt"))})
rel_id = {name: WordNet18RR.edge2id[name] for name in rel_names}   # the loader's number for each name
R = len(rel_id)
facts = torch.stack([data.edge_index[0], data.edge_type, data.edge_index[1]], 1)   # rows (head, relation, tail)
train, valid, test = facts[data.train_mask], facts[data.val_mask], facts[data.test_mask]
print(f"\nWordNet18RR: {N} entities, {R} relations; {len(train)} training, {len(valid)} validation, "
      f"{len(test)} test facts")
train_set = set(map(tuple, train.tolist()))
print("relation, training facts, test facts, share of training facts whose reverse (t, r, h) is also one:")
for name in rel_names:
    rows = train[train[:, 1] == rel_id[name]]
    reverse = np.mean([(t, r, h) in train_set for h, r, t in rows.tolist()])
    print(f"  {name:29s} {len(rows):6d} {int((test[:, 1] == rel_id[name]).sum()):5d}   {reverse:.2f}")

g = torch.Generator().manual_seed(0)                   # one seeded generator for the init, the shuffles, the corruptions
d, margin, batch_size, epochs = 50, 1.0, 1024, 30
ent = torch.nn.Parameter(torch.randn(N, d, generator=g) * 0.1)
rel = torch.nn.Parameter(torch.randn(R, d, generator=g) * 0.1)
opt = torch.optim.Adam([ent, rel], lr=0.01)

def rows(M, idx):                                          # the rows of M at idx, gathered with index_select rather
    return M.index_select(0, idx)                          # than M[idx]: its backward is reproducible run to run

print(f"\nTransE: {d}-dim embeddings, margin {margin}, {epochs} epochs of minibatches of {batch_size} facts")
for epoch in range(1, epochs + 1):
    order = torch.randperm(len(train), generator=g); total = 0.0
    for i in range(0, len(train), batch_size):
        b = train[order[i:i + batch_size]]
        h, r, t = b[:, 0], b[:, 1], b[:, 2]
        t_neg = torch.randint(0, N, (len(b),), generator=g)   # corrupt the tail: a (likely) false fact
        e = torch.nn.functional.normalize(ent, dim=1)         # keep entity norms bounded
        pos = (rows(e, h) + rows(rel, r) - rows(e, t)).norm(dim=1)       # distance for the true fact
        neg = (rows(e, h) + rows(rel, r) - rows(e, t_neg)).norm(dim=1)   # distance for the corrupted fact
        loss = torch.clamp(margin + pos - neg, min=0).mean()  # want pos + margin < neg
        opt.zero_grad(); loss.backward(); opt.step(); total += loss.item() * len(b)
    if epoch in (1, 10, 20, 30):
        print(f"  epoch {epoch:2d}: mean margin loss {total / len(train):.4f}")

known = {}                                                 # every true tail of each (head, relation), all splits
for h, r, t in facts.tolist():
    known.setdefault((h, r), set()).add(t)

def rank_tails(distance_to_all, rows):
    """Rank each fact's true tail among all N entities: rank = 1 + the number of candidates strictly nearer.
    Raw keeps every candidate; filtered removes the other known true tails of the same head and relation.
    Also records whether the head itself is nearer than the true tail (the head is never filtered out)."""
    raw, filtered, head_first = [], [], []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            b = rows[i:i + 256]
            D = distance_to_all(b[:, 0], b[:, 1])             # (facts, N): the distance of every candidate tail
            for j, (h, r, t) in enumerate(b.tolist()):
                d_true = D[j, t]
                raw.append(int((D[j] < d_true).sum()) + 1)
                head_first.append(bool(D[j, h] < d_true))
                D[j, [x for x in known[(h, r)] if x != t]] = float("inf")
                filtered.append(int((D[j] < d_true).sum()) + 1)
    return np.array(raw), np.array(filtered), np.array(head_first)

def report(label, ranks):
    print(f"  {label:9s} MRR {np.mean(1 / ranks):.3f}   Hits@1 {np.mean(ranks == 1):.3f}   "
          f"Hits@10 {np.mean(ranks <= 10):.3f}   mean rank {ranks.mean():.1f}")

def transe_to_all(h, r):                                   # exact distances: cdist's matmul shortcut rounds the smallest ones
    e = torch.nn.functional.normalize(ent, dim=1)
    return torch.cdist(e[h] + rel[r], e, compute_mode="donot_use_mm_for_euclid_dist")   # exact ||h + r - t||

raw, filtered, head_first = rank_tails(transe_to_all, test)
print(f"\nlink prediction on the {len(test)} test facts (rank the true tail among all {N} entities):")
report("raw", raw); report("filtered", filtered)
k = np.arange(1, N + 1)                                    # a random ranking: every rank equally likely
print(f"  random    MRR {np.mean(1 / k):.5f} Hits@1 {1 / N:.5f} Hits@10 {10 / N:.5f} mean rank {(N + 1) / 2:.1f}")
print("filtered, by relation (the head itself is a candidate, so it can outrank the true tail):")
for name in ["_derivationally_related_form", "_hypernym"]:
    m = (test[:, 1] == rel_id[name]).numpy()
    print(f"  {name}: Hits@1 {np.mean(filtered[m] == 1):.3f}  Hits@10 {np.mean(filtered[m] <= 10):.3f}  "
          f"head outranks tail {np.mean(head_first[m]):.2f}  learned ||r|| {rel[rel_id[name]].norm().item():.2f}  "
          f"({m.sum()} test facts)")

# DistMult scores a triple by the trilinear product sum_k h_k r_k t_k. Because h and t
# only ever MULTIPLY, swapping them cannot change the score: symmetric by construction.
hd = np.array([1.0, 0.5]); rd = np.array([2.0, 1.0]); td = np.array([0.5, 1.0])
print(f"\nDistMult on a worked triple, h = {hd}, r = {rd}, t = {td}:")
print(f"  score(h, r, t) = 1.0*2.0*0.5 + 0.5*1.0*1.0 = {np.sum(hd * rd * td):.1f}")
print(f"  score(t, r, h) = {np.sum(td * rd * hd):.1f}   (identical: DistMult cannot tell direction)")
print(f"  TransE can: ||h + r - t|| = {np.linalg.norm(hd + rd - td):.2f} "
      f"but ||t + r - h|| = {np.linalg.norm(td + rd - hd):.2f}")
print("\nAll knowledge-graph checks ran.")

# %% [markdown]
# Example 20-7. DistMult, ComplEx, and RotatE trained with Example 20-6's loop on WordNet18RR, ranked beside TransE.

# %%
def make(model):
    """Fresh parameters for one scoring function, with the two distances the loop and the ranking need."""
    ent = torch.nn.Parameter(torch.randn(N, d, generator=g) * 0.1)             # 50 numbers per entity, as before
    rel = torch.nn.Parameter(torch.randn(R, d // 2 if model == "RotatE" else d, generator=g) * 0.1)
    def query(h, r):                                       # q(h, r): the vector the true tail should match
        e = torch.nn.functional.normalize(ent, dim=1)      # unit entity vectors, as in Example 20-6
        if model == "DistMult":
            return rows(e, h) * rows(rel, r), e            # coordinate by coordinate product
        a, b = rows(e, h).chunk(2, dim=1)                  # 25 complex coordinates: real and imaginary halves
        if model == "ComplEx":
            c, dd = rows(rel, r).chunk(2, dim=1)
            return torch.cat([a * c - b * dd, a * dd + b * c], 1), e           # complex product h * r
        theta = rows(rel, r)                               # RotatE: r_k = e^(i theta_k), a rotation per coordinate
        cos, sin = torch.cos(theta), torch.sin(theta)
        return torch.cat([a * cos - b * sin, a * sin + b * cos], 1), e
    dot = model in ("DistMult", "ComplEx")                 # these two score by a dot product, higher is better,
    def distance(q, e_t):                                  # so minus the score plays the part of the distance
        return -(q * e_t).sum(1) if dot else (q - e_t).norm(dim=1)
    def distance_to_all(h, r):
        q, e = query(h, r)
        return -(q @ e.T) if dot else torch.cdist(q, e, compute_mode="donot_use_mm_for_euclid_dist")
    return [ent, rel], query, distance, distance_to_all

def fit(query, distance, params):                          # Example 20-6's loop with the query and distance swapped in
    opt = torch.optim.Adam(params, lr=0.01)
    for epoch in range(epochs):
        order = torch.randperm(len(train), generator=g)
        for i in range(0, len(train), batch_size):
            b = train[order[i:i + batch_size]]
            h, r, t = b[:, 0], b[:, 1], b[:, 2]
            t_neg = torch.randint(0, N, (len(b),), generator=g)
            q, e = query(h, r)
            loss = torch.clamp(margin + distance(q, rows(e, t)) - distance(q, rows(e, t_neg)), min=0).mean()
            opt.zero_grad(); loss.backward(); opt.step()

results = {"TransE": (filtered, head_first)}               # Example 20-6's filtered ranks
for model in ["DistMult", "ComplEx", "RotatE"]:
    params, query, distance, distance_to_all = make(model)
    fit(query, distance, params)
    results[model] = rank_tails(distance_to_all, test)[1:]
sym = (test[:, 1] == rel_id["_derivationally_related_form"]).numpy()
hyp = (test[:, 1] == rel_id["_hypernym"]).numpy()
print("filtered ranking of the test tails, the same loop and 50 numbers per entity for every model:")
print("  model       MRR  Hits@1  Hits@10 | _derivationally_related_form: Hits@1  Hits@10  head first | _hypernym: Hits@10")
for model, (rk, hf) in results.items():
    print(f"  {model:9s} {np.mean(1 / rk):5.3f}  {np.mean(rk == 1):5.3f}  {np.mean(rk <= 10):6.3f}  | "
          f"{'':30s}{np.mean(rk[sym] == 1):5.3f}  {np.mean(rk[sym] <= 10):6.3f}  {np.mean(hf[sym]):5.2f}      | "
          f"{'':11s}{np.mean(rk[hyp] <= 10):6.3f}")

# %% [markdown]
# ## 20.2.5 Reading a graph before you model it

# %% [markdown]
# Example 20-8 computes these statistics and three of the centralities on the unweighted karate club, where the numbers are small enough to read in one run.
#
# Example 20-8. Density, clustering, path lengths, degree assortativity, edge homophily, and three centralities of the unweighted karate club.

# %%
import numpy as np
import networkx as nx
G = nx.karate_club_graph()
H = nx.Graph()
H.add_nodes_from(G.nodes(data=True))
H.add_edges_from(G.edges())                 # drop the interaction weights
n, m = H.number_of_nodes(), H.number_of_edges()
club = np.array([0 if H.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(n)])
print(f"nodes {n}, edges {m}, density {nx.density(H):.3f}")
print(f"average clustering {nx.average_clustering(H):.3f}, transitivity {nx.transitivity(H):.3f}")
print(f"diameter {nx.diameter(H)}, average shortest path {nx.average_shortest_path_length(H):.3f}")
print(f"degree assortativity {nx.degree_assortativity_coefficient(H):.3f}")
same = sum(club[u] == club[v] for u, v in H.edges())
print(f"edge homophily {same}/{m} = {same/m:.3f}")
bc = nx.betweenness_centrality(H)
ec = nx.eigenvector_centrality(H)
cc = nx.closeness_centrality(H)
def top2(score):
    order = sorted(score, key=score.get, reverse=True)[:2]
    return [(i, round(score[i], 3)) for i in order]
print("betweenness", top2(bc))
print("eigenvector", top2(ec))
print("closeness", top2(cc))

# %% [markdown]
# ## 20.2.6 Communities by modularity

# %% [markdown]
# The modularity optimum on this graph is not the historical split. Example 20-9 computes both scores.
#
# Example 20-9. Louvain communities of the unweighted karate club, and the modularity of the true two-faction split.

# %%
import networkx as nx
G = nx.karate_club_graph()
H = nx.Graph()
H.add_nodes_from(G.nodes(data=True))
H.add_edges_from(G.edges())
n = H.number_of_nodes()
club = [0 if H.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(n)]
parts = nx.community.louvain_communities(H, seed=0)
f0 = {i for i in range(n) if club[i] == 0}
f1 = set(range(n)) - f0
print(f"Louvain communities {len(parts)}, modularity {nx.community.modularity(H, parts):.3f}")
print(f"two-faction modularity {nx.community.modularity(H, [f0, f1]):.3f}")

# %% [markdown]
# ## 20.3 Graph Neural Networks: message passing, GCN, GraphSAGE, GAT

# %% [markdown]
# ## 20.3.1 The graph convolutional network

# %% [markdown]
# Example 20-10. A GCN versus a feature-only MLP.

# %%
import numpy as np
import torch
import torch.nn as nn
rng = np.random.default_rng(0); torch.manual_seed(0)

sizes = [50, 50, 50]; N = sum(sizes); C = 3
labels = np.concatenate([[c] * s for c, s in enumerate(sizes)])
p_in, p_out = 0.10, 0.02
A = np.zeros((N, N))
for i in range(N):
    for j in range(i + 1, N):
        if rng.random() < (p_in if labels[i] == labels[j] else p_out):
            A[i, j] = A[j, i] = 1
onehot = np.eye(C)[labels]
X = 0.35 * onehot + rng.normal(0, 1.0, (N, C))           # tiny label signal + big noise
print(f"{N} nodes, 3 communities; features are mostly noise (signal-to-noise is low on purpose)")

A_tilde = A + np.eye(N)                                   # add self-loops so a node keeps its own signal
deg = A_tilde.sum(1)
A_hat = A_tilde / np.sqrt(np.outer(deg, deg))            # symmetric normalization
A_hat = torch.tensor(A_hat, dtype=torch.float32)
Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(labels)
train_mask = torch.zeros(N, dtype=torch.bool); train_mask[rng.choice(N, 60, replace=False)] = True
test_mask = ~train_mask

class MLP(nn.Module):
    def __init__(self): super().__init__(); self.l1 = nn.Linear(C, 16); self.l2 = nn.Linear(16, C)
    def forward(self, X, A): return self.l2(torch.relu(self.l1(X)))           # never looks at A

class GCN(nn.Module):
    def __init__(self): super().__init__(); self.l1 = nn.Linear(C, 16); self.l2 = nn.Linear(16, C)
    def forward(self, X, A):
        H = torch.relu(A @ self.l1(X))                                  # aggregate neighbors, layer 1
        return A @ self.l2(H)                                           # aggregate neighbors, layer 2

def train_eval(model):
    opt = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
    for _ in range(200):
        out = model(Xt, A_hat); loss = nn.functional.cross_entropy(out[train_mask], yt[train_mask])
        opt.zero_grad(); loss.backward(); opt.step()
    pred = model(Xt, A_hat).argmax(1)
    return (pred[test_mask] == yt[test_mask]).float().mean().item()

print(f"\nnode classification test accuracy (60 labeled nodes, predict the other {N-60}):")
print(f"  MLP on features only (graph ignored) : {train_eval(MLP()):.2f}")
print(f"  GCN (message passing over the graph) : {train_eval(GCN()):.2f}")
print("same features, same labels: the GCN wins by mixing each node's noisy features with its")
print("neighbors'; homophily means neighbors share a label, so aggregation DENOISES the signal.")

print("\nGNN variants:")
print("  GCN       : aggregate = degree-weighted SUM of self and neighbors (fixed weights)")
print("  GraphSAGE : SAMPLE a fixed number of neighbors, then mean/max/LSTM aggregate; INDUCTIVE")
print("              (works on unseen nodes) and scales to huge graphs")
print("  GAT       : ATTENTION over neighbors; learn how much each neighbor matters, not just degree")
print("deeper stacks see farther, but too many layers cause OVERSMOOTHING (vectors line up in one direction).")

print("\nAll GNN checks ran.")

# %% [markdown]
# ## 20.3.2 GraphSAGE, GAT, and the family

# %% [markdown]
# GraphSAGE's update on a three-node path, by hand and with SAGEConv.
#
# Example 20-11. GraphSAGE's mean update by hand on a three-node path, checked against `SAGEConv`.

# %%
import numpy as np
h = {"a": 1.0, "b": 2.0, "c": 4.0}                       # one number per node on the path a - b - c
nbrs = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
W = np.array([1.0, 0.5])                                 # weight 1 on the node's own value, 0.5 on the neighbor mean
for v in "abc":
    agg = np.mean([h[u] for u in nbrs[v]])               # the mean aggregator over every neighbor
    print(f"{v}: relu(1 * {h[v]:.0f} + 0.5 * {agg}) = {max(0.0, float(W @ np.array([h[v], agg]))):.2f}")
import torch
from torch_geometric.nn import SAGEConv
x = torch.tensor([[1.0], [2.0], [4.0]])
ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])          # a-b and b-c, each stored in both directions
sage = SAGEConv(1, 1, bias=False)
with torch.no_grad():
    sage.lin_r.weight.fill_(1.0); sage.lin_l.weight.fill_(0.5)   # root weight 1, neighbor-mean weight 0.5
print("SAGEConv:", torch.relu(sage(x, ei)).flatten().tolist())

# %% [markdown]
# Most GNNs are only two or three layers deep. Example 20-12 puts numbers on the oversmoothing: it multiplies a random vector on the karate club by the normalized self-looped adjacency forty times and compares the result with the square root of the self-looped degree.
#
# Example 20-12. Forty multiplications by the normalized self-looped adjacency of the karate club, compared with the square root of the self-looped degree.

# %%
import numpy as np, networkx as nx
K = nx.karate_club_graph(); N = K.number_of_nodes()
A = nx.to_numpy_array(K, weight=None)
At = A + np.eye(N); deg = At.sum(1)
Ah = At / np.sqrt(np.outer(deg, deg))             # symmetric normalization, not a row-stochastic average
rng = np.random.default_rng(0)
h = rng.normal(size=N)
for _ in range(40):
    h = Ah @ h
h = h / np.linalg.norm(h)
target = np.sqrt(deg); target = target / np.linalg.norm(target)
if float(h @ target) < 0:
    h = -h
print(f"correlation with sqrt(self-looped degree) = {float(h @ target):.2f}")
print(f"A_hat row sums span [{Ah.sum(1).min():.2f}, {Ah.sum(1).max():.2f}]")

# %% [markdown]
# GIN's update on the same path, and the two-against-five contrast.

# %%
from torch_geometric.nn import GINConv
print("GIN, epsilon 0, MLP left out:", GINConv(torch.nn.Identity())(x, ei).flatten().tolist())   # the same path
two, five = [1.0, 1.0], [1.0] * 5                        # two identical neighbors against five of them
print("mean", sum(two) / 2, sum(five) / 5, " max", max(two), max(five), " sum", sum(two), sum(five))

# %% [markdown]
# ## 20.3.3 Typed edges: relational GNNs

# %% [markdown]
# Counting its parameters checks the arithmetic.

# %%
from torch_geometric.nn import GCNConv, RGCNConv

def n_params(layer): return sum(p.numel() for p in layer.parameters())
print("GCNConv, one shared 16 x 16 matrix        :", n_params(GCNConv(16, 16, bias=False)))
print("RGCNConv, ten relations plus the self-loop:", n_params(RGCNConv(16, 16, num_relations=10, bias=False)))
print("RGCNConv, the same from three shared bases:", n_params(RGCNConv(16, 16, num_relations=10, num_bases=3, bias=False)))

# %% [markdown]
# Example 20-13. Entity classification on AIFB with R-GCN, with and without the edge types.

# %%
import warnings
warnings.filterwarnings("ignore", message=".*IProgress.*")
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.nn import RGCNConv

ds = Entities(root="aifb", name="AIFB")   # the four raw files and the processed copy ship with the companion code
data = ds[0]
N, R, C = data.num_nodes, ds.num_relations, ds.num_classes
print(f"AIFB: {N} entities, {data.num_edges} typed edge entries, {R} edge types, {C} classes, "
      f"{data.train_idx.numel()} train and {data.test_idx.numel()} test entities")

class RGCN(torch.nn.Module):
    def __init__(self, n_rel, bases):
        super().__init__()
        self.entity = torch.nn.Embedding(N, 16)                # no features: a learned vector per entity
        self.c1 = RGCNConv(16, 16, n_rel, num_bases=bases)     # one W_r per edge type, mixed from shared bases
        self.c2 = RGCNConv(16, C, n_rel, num_bases=bases)
    def forward(self, edge_type):
        h = F.relu(self.c1(self.entity.weight, data.edge_index, edge_type))
        return self.c2(h, data.edge_index, edge_type)

def accuracies(n_rel, bases, edge_type, seed):
    torch.manual_seed(seed)
    m = RGCN(n_rel, bases)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(50):                                     # two layers, 16 units, Adam at 0.01, 50 epochs
        m.train(); opt.zero_grad()
        F.cross_entropy(m(edge_type)[data.train_idx], data.train_y).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(edge_type).argmax(1)
    acc = lambda idx, y: round((pred[idx] == y).float().mean().item(), 3)
    return acc(data.train_idx, data.train_y), acc(data.test_idx, data.test_y)

no_types = torch.zeros_like(data.edge_type)                 # the same edges with every type erased
print("(train, test) accuracy for seeds 0, 1, 2")
print("R-GCN, 90 edge types, 30 bases:", [accuracies(R, 30, data.edge_type, s) for s in range(3)])
print("same network, one edge type   :", [accuracies(1, None, no_types, s) for s in range(3)])
print(f"majority class: {int(torch.bincount(data.test_y).max())} of {data.test_idx.numel()} test entities")

# %% [markdown]
# ## 20.3.4 A GCN on a real network: Zachary's karate club

# %% [markdown]
# Example 20-14. A from-scratch GCN labeling the karate club from two seeds.

# %%
import numpy as np, networkx as nx, torch, torch.nn as nn
np.random.seed(0); torch.manual_seed(0)

G = nx.karate_club_graph(); N = G.number_of_nodes()
A = nx.to_numpy_array(G, weight=None)                     # UNWEIGHTED adjacency (34x34)
y = np.array([0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(N)])

A_tilde = A + np.eye(N); deg = A_tilde.sum(1)             # self-loops, then degrees
A_hat = A_tilde / np.sqrt(np.outer(deg, deg))            # A_hat = D~^-1/2 (A+I) D~^-1/2
A_hat = torch.tensor(A_hat, dtype=torch.float32)
X = torch.eye(N); yt = torch.tensor(y)                   # featureless: identity per node

train = torch.zeros(N, dtype=torch.bool); train[[0, 33]] = True   # ONLY the two leaders
test = ~train

class GCN(nn.Module):
    def __init__(self): super().__init__(); self.l1 = nn.Linear(N, 16); self.l2 = nn.Linear(16, 2)
    def forward(self, X, A):
        H = torch.relu(A @ self.l1(X))                       # aggregate neighbors, layer 1
        return A @ self.l2(H)                                # aggregate neighbors, layer 2

m = GCN(); opt = torch.optim.Adam(m.parameters(), lr=0.02, weight_decay=5e-4)
for _ in range(200):
    out = m(X, A_hat); loss = nn.functional.cross_entropy(out[train], yt[train])
    opt.zero_grad(); loss.backward(); opt.step()
pred = m(X, A_hat).argmax(1)
acc = (pred[test] == yt[test]).float().mean().item()
print(f"karate club: {N} nodes, 2 factions; trained on ONLY 2 labels (the leaders 0 and 33)")
print(f"GCN test accuracy on the other {int(test.sum())} members: {acc:.2f}")

class GCNDeg(nn.Module):
    def __init__(self):
        super().__init__(); self.l1 = nn.Linear(1, 16); self.l2 = nn.Linear(16, 2)
    def forward(self, X, A):
        return A @ self.l2(torch.relu(A @ self.l1(X)))

torch.manual_seed(0)
md = GCNDeg(); opt = torch.optim.Adam(md.parameters(), lr=0.02, weight_decay=5e-4)
Xd = torch.tensor(A.sum(1), dtype=torch.float32).view(-1, 1)   # degree is the only feature
for _ in range(200):
    out = md(Xd, A_hat); loss = nn.functional.cross_entropy(out[train], yt[train])
    opt.zero_grad(); loss.backward(); opt.step()
acc_d = (md(Xd, A_hat).argmax(1)[test] == yt[test]).float().mean().item()
print(f"same training with degree as the only feature: test accuracy {acc_d:.2f}")

# %% [markdown]
# ## 20.3.5 Beyond message passing: graph transformers

# %% [markdown]
# Laplacian eigenvector and random-walk positional encodings on the karate club.
#
# Example 20-15. Laplacian eigenvector and random-walk positional encodings of the karate club, the second checked against `AddRandomWalkPE`.

# %%
import numpy as np, networkx as nx
G = nx.karate_club_graph(); N = G.number_of_nodes()
A = nx.to_numpy_array(G, weight=None); d = A.sum(1)
L_sym = np.eye(N) - A / np.sqrt(np.outer(d, d))           # the symmetric normalized Laplacian of Section 20.2.2
lam, V = np.linalg.eigh(L_sym)                            # ascending eigenvalues; column 0 is the trivial eigenvector
lap_pe = V[:, 1:3]                                        # the first two nontrivial eigenvectors, one row per node
lap_pe = lap_pe * np.sign(lap_pe[np.abs(lap_pe).argmax(0), [0, 1]])   # a sign convention: largest entry positive
P = A / d[:, None]                                        # the random-walk matrix D^-1 A
Pk = np.eye(N); rw_pe = []
for _ in range(4):
    Pk = Pk @ P; rw_pe.append(np.diag(Pk))                # (P^k)_vv: the chance that a k-step walk from v is back at v
rw_pe = np.array(rw_pe).T
print("smallest L_sym eigenvalues:", (np.round(lam[:3], 3) + 0.0).tolist())   # + 0.0 turns a rounded -0.0 into 0.0
for v in [0, 11, 33]:
    print(f"node {v} (degree {int(d[v])}): Laplacian PE {np.round(lap_pe[v], 3).tolist()}, "
          f"random-walk PE {np.round(rw_pe[v], 4).tolist()}")
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.utils import from_networkx
data = AddRandomWalkPE(walk_length=4, attr_name="rw")(from_networkx(G))      # the library's random-walk encoding
print("AddRandomWalkPE row 11:", np.round(data.rw[11].numpy().astype(float), 4).tolist())

# %% [markdown]
# ## 20.4 Tasks: node classification, link prediction, graph classification

# %% [markdown]
# Example 20-16. Link prediction on the Les Misérables network.

# %%
import numpy as np, networkx as nx
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)

G = nx.les_miserables_graph()                             # 77 characters, 254 coappearance edges
nodes = sorted(G.nodes()); N = len(nodes)
A = nx.to_numpy_array(G, nodelist=nodes, weight=None)     # UNWEIGHTED adjacency
edges = np.argwhere(np.triu(A, 1) > 0)

rng.shuffle(edges); n_hold = len(edges) // 5             # hide 20% of the edges
held = edges[:n_hold]
A_train = A.copy()
for i, j in held: A_train[i, j] = A_train[j, i] = 0       # NO LEAKAGE: hidden edges gone from training
cut_off = set(np.where(A_train.sum(1) == 0)[0])           # characters left with no training edge at all
held = np.array([(i, j) for i, j in held if i not in cut_off and j not in cut_off])
print(f"Les Miserables: {N} nodes, {len(edges)} edges; hid {n_hold} of them")
print(f"{len(cut_off)} characters keep no training edge, so {n_hold - len(held)} held-out edges that touch "
      f"them are dropped; {len(held)} are scored")

U, S, Vt = np.linalg.svd(A_train, full_matrices=False)    # 16-dim node embeddings
emb = U[:, :16] * np.sqrt(S[:16])
def score(i, j): return emb[i] @ emb[j]

non_edges = []                                            # NEGATIVE SAMPLING: equal random nonedges
while len(non_edges) < len(held):
    i, j = int(rng.integers(N)), int(rng.integers(N))
    if i != j and A[i, j] == 0: non_edges.append((i, j))
y = [1] * len(held) + [0] * len(held)
s = [score(i, j) for i, j in held] + [score(i, j) for i, j in non_edges]
print(f"LINK PREDICTION ROC AUC = {roc_auc_score(y, s):.2f}  (0.5 = random)")
Uf, Sf, Vtf = np.linalg.svd(A, full_matrices=False)      # same score, but the held-out edges were NOT deleted
emb_leaky = Uf[:, :16] * np.sqrt(Sf[:16])
s_leaky = [emb_leaky[i] @ emb_leaky[j] for i, j in held] + [emb_leaky[i] @ emb_leaky[j] for i, j in non_edges]
print(f"LEAKY ROC AUC, held-out edges left in the matrix = {roc_auc_score(y, s_leaky):.2f}")

def ranking_metrics(k=10, n_neg=50):                      # Hits@k and MRR: rank each true edge
    ranks = []
    for (i, j) in held:
        pos = score(i, j); negs = []
        while len(negs) < n_neg:
            a = int(rng.integers(N))
            if a != i and A[i, a] == 0: negs.append(score(i, a))
        ranks.append(1 + sum(ns >= pos for ns in negs))   # rank of the true edge (1 = best)
    ranks = np.array(ranks)
    return (ranks <= k).mean(), (1.0 / ranks).mean()
h, mrr = ranking_metrics()
print(f"Hits@10 = {h:.2f}   MRR = {mrr:.2f}  (rank each true edge among 50 nonedges)")

# %% [markdown]
# The sign-correct rank-16 reconstruction, reusing Example 20-16's variables.
#
# Example 20-17. The sign-correct rank-16 reconstruction of Example 20-16's training adjacency, scored on the same pairs honestly and with the leak.

# %%
lam = np.linalg.eigvalsh(A_train)                         # A_train is symmetric, so its eigenvalues are real
top16 = lam[np.argsort(-np.abs(lam))[:16]]                # the 16 largest in absolute value
flipped = int((np.sum(U[:, :16] * Vt[:16].T, axis=0) < 0).sum())   # columns of V that are minus the columns of U
print(f"{int((top16 < 0).sum())} of the 16 largest |eigenvalues| of A_train are negative; "
      f"{flipped} columns of V are minus the matching columns of U")
recon = (U[:, :16] * S[:16]) @ Vt[:16]                    # the rank-16 reconstruction U S V^T of A_train itself
recon_leaky = (Uf[:, :16] * Sf[:16]) @ Vtf[:16]           # the same with the held-out edges left in
pairs = [tuple(p) for p in held] + non_edges
print(f"rank-16 reconstruction U S V^T: honest ROC AUC = {roc_auc_score(y, [recon[i, j] for i, j in pairs]):.2f}, "
      f"leaky = {roc_auc_score(y, [recon_leaky[i, j] for i, j in pairs]):.2f}")

# %% [markdown]
# Example 20-18. Graph classification on the MUTAG molecules with a GCN and mean-pool readout.

# %%
import warnings
warnings.filterwarnings("ignore", message=".*IProgress.*")
import torch, torch.nn as nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
torch.manual_seed(0)

ds = TUDataset(root="mutag", name="MUTAG").shuffle()      # 188 real molecules, 2 classes
train, test = ds[:150], ds[150:]
train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=64)

class GraphGCN(nn.Module):
    def __init__(self, hid=32):
        super().__init__()
        self.c1 = GCNConv(ds.num_node_features, hid); self.c2 = GCNConv(hid, hid)
        self.lin = nn.Linear(hid, ds.num_classes)
    def forward(self, x, edge_index, batch):
        x = torch.relu(self.c1(x, edge_index))
        x = torch.relu(self.c2(x, edge_index))
        x = global_mean_pool(x, batch)                    # READOUT: node vectors -> one graph vector
        return self.lin(x)

m = GraphGCN(); opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
def run(loader, train_mode):
    m.train(train_mode); correct = 0
    for d in loader:
        out = m(d.x, d.edge_index, d.batch)
        if train_mode:
            loss = nn.functional.cross_entropy(out, d.y)
            opt.zero_grad(); loss.backward(); opt.step()
        correct += (out.argmax(1) == d.y).sum().item()
    return correct / len(loader.dataset)
for _ in range(100): run(train_loader, True)
print(f"MUTAG: {len(ds)} molecular graphs, {ds.num_node_features} atom features, {ds.num_classes} classes")
print(f"train acc {run(train_loader, False):.2f}   TEST acc {run(test_loader, False):.2f}")
test_labels = torch.tensor([int(d.y) for d in test])
majority_n = int(max((test_labels == 0).sum(), (test_labels == 1).sum()))
print(f"majority class on this test split: {majority_n} of {len(test)}")

# %% [markdown]
# ## 20.4.2 Counting scores for a missing edge

# %% [markdown]
# Example 20-19. The five counting scores on the Les Misérables split.

# %%
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score
H = nx.les_miserables_graph()
nodes = sorted(H.nodes())
A = nx.to_numpy_array(H, nodelist=nodes, weight=None)
rng = np.random.default_rng(0)
edges = np.argwhere(np.triu(A, 1) > 0)
rng.shuffle(edges)
n_hold = len(edges) // 5
held = [tuple(p) for p in edges[:n_hold]]
T = nx.Graph()
T.add_nodes_from(range(len(nodes)))
T.add_edges_from(map(tuple, edges[n_hold:]))
non_edges = []
seen = set(held)
while len(non_edges) < n_hold:
    i, j = int(rng.integers(len(nodes))), int(rng.integers(len(nodes)))
    if i == j:
        continue
    a, b = (i, j) if i < j else (j, i)
    if A[a, b] == 0 and (a, b) not in seen:
        seen.add((a, b))
        non_edges.append((a, b))
y = [1] * n_hold + [0] * n_hold
pairs = held + non_edges
def auc_of(scores):
    s = [scores.get(tuple(sorted(p)), 0.0) for p in pairs]
    return roc_auc_score(y, s)
cn = {tuple(sorted(p)): len(list(nx.common_neighbors(T, *p))) for p in pairs}
print("common neighbors", round(float(auc_of(cn)), 3))
for fn, name in [
    (nx.jaccard_coefficient, "jaccard"),
    (nx.adamic_adar_index, "adamic-adar"),
    (nx.resource_allocation_index, "resource allocation"),
    (nx.preferential_attachment, "preferential attachment"),
]:
    scored = {tuple(sorted((u, v))): val for u, v, val in fn(T, pairs)}
    print(name, round(float(auc_of(scored)), 3))

# %% [markdown]
# One held-out pair by hand, on the same training graph T.

# %%
import math
u, v = nodes.index("Anzelma"), nodes.index("Eponine")   # a held-out coappearance, scored on the training graph T
shared = sorted(nx.common_neighbors(T, u, v))
print("shared:", [(nodes[w], T.degree(w)) for w in shared], " degrees:", T.degree(u), T.degree(v))
print("CN", len(shared), " Jaccard", round(len(shared) / len(set(T[u]) | set(T[v])), 4),
      " AA", round(sum(1 / math.log(T.degree(w)) for w in shared), 4),
      " RA", round(sum(1 / T.degree(w) for w in shared), 4), " PA", T.degree(u) * T.degree(v))

# %% [markdown]
# ## 20.4.3 Sums that count: GIN on MUTAG

# %% [markdown]
# Example 20-20. GCN or GIN layers with a mean or a sum readout on MUTAG, five seeds each.

# %%
import warnings
warnings.filterwarnings("ignore", message=".*IProgress.*")
import numpy as np
import torch, torch.nn as nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool

torch.manual_seed(0)
ds = TUDataset(root="mutag", name="MUTAG").shuffle()      # Example 20-18's shuffle: the same 150 and 38
train, test = ds[:150], ds[150:]
by_share = {}                                             # mean readout of the raw atom types -> sum readouts
for d in ds:
    by_share.setdefault(tuple(d.x.mean(0).tolist()), set()).add(tuple(int(c) for c in d.x.sum(0)))
print(f"{len(ds)} molecules: {sum(len(v) for v in by_share.values())} distinct atom-count vectors, "
      f"{len(by_share)} distinct atom-share vectors")
print("one share vector, three count vectors:", sorted(max(by_share.values(), key=len)))

def gin(i, o):                                            # sum the neighbors, add (1 + eps) times self, then an MLP
    return GINConv(nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o)))

class GraphNet(nn.Module):
    def __init__(self, conv, pool, hid=32):
        super().__init__()
        self.c1, self.c2 = conv(ds.num_node_features, hid), conv(hid, hid)
        self.pool, self.lin = pool, nn.Linear(hid, ds.num_classes)
    def forward(self, x, edge_index, batch):
        x = torch.relu(self.c1(x, edge_index))
        x = torch.relu(self.c2(x, edge_index))
        return self.lin(self.pool(x, batch))                    # READOUT: a mean or a sum over the atoms

def accuracies(conv, pool, seed):
    torch.manual_seed(seed)
    m = GraphNet(conv, pool); opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    loader = DataLoader(train, batch_size=32, shuffle=True)
    for _ in range(100):                                  # Example 20-18's optimizer, batch size, and epochs
        for d in loader:
            loss = nn.functional.cross_entropy(m(d.x, d.edge_index, d.batch), d.y)
            opt.zero_grad(); loss.backward(); opt.step()
    def acc(graphs):
        d = next(iter(DataLoader(graphs, batch_size=len(graphs))))   # every molecule in one batch
        return (m(d.x, d.edge_index, d.batch).argmax(1) == d.y).float().mean().item()
    return acc(train), acc(test)

for name, conv, pool in [("GCN, mean readout", GCNConv, global_mean_pool),
                         ("GIN, mean readout", gin, global_mean_pool),
                         ("GCN, sum readout ", GCNConv, global_add_pool),
                         ("GIN, sum readout ", gin, global_add_pool)]:
    tr, te = zip(*[accuracies(conv, pool, s) for s in range(5)])
    print(f"{name}: test by seed {np.round(te, 3).tolist()}, mean {np.mean(te):.3f}; mean train {np.mean(tr):.3f}")

# %% [markdown]
# ## 20.5 Hands-on lab: node classification on a citation network with PyTorch Geometric

# %% [markdown]
# Example 20-21. The lab: GCN node classification on Cora.

# %%
import warnings; warnings.filterwarnings("ignore")
import os, tempfile
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
torch.manual_seed(0)

data = Planetoid(root=os.path.join(tempfile.gettempdir(), "cora"), name="Cora")[0]
print(f"Cora: {data.num_nodes} papers, {data.num_edges // 2} citation links "
      f"({data.num_edges} directed entries in edge_index), "
      f"{data.num_node_features} features, 7 classes")
print(f"labeled for training: {int(data.train_mask.sum())} papers; test: {int(data.test_mask.sum())}")

class GCN(torch.nn.Module):
    def __init__(self, nf, nc, h=16):
        super().__init__(); self.c1 = GCNConv(nf, h); self.c2 = GCNConv(h, nc)
    def forward(self, x, ei):
        x = F.dropout(F.relu(self.c1(x, ei)), p=0.5, training=self.training)
        return self.c2(x, ei)

class MLP(torch.nn.Module):
    def __init__(self, nf, nc, h=16):
        super().__init__(); self.l1 = torch.nn.Linear(nf, h); self.l2 = torch.nn.Linear(h, nc)
    def forward(self, x, ei):
        x = F.dropout(F.relu(self.l1(x)), p=0.5, training=self.training)
        return self.l2(x)

def run(model):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(200):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step()
    model.eval()
    pred = model(data.x, data.edge_index).argmax(1)
    return (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

nf, nc = data.num_node_features, 7
mlp_acc = run(MLP(nf, nc))
gcn_acc = run(GCN(nf, nc))
print(f"\ntest accuracy classifying papers into 7 topics from 140 labels:")
print(f"  MLP  (features only, ignores citations) : {mlp_acc:.3f}")
print(f"  GCN  (features + citation graph)        : {gcn_acc:.3f}")
print(f"  the citation links add {(gcn_acc - mlp_acc)*100:.0f} accuracy points: a paper's neighbors")
print(f"  in the citation graph reveal its topic, which the bag-of-words alone misses.")

print("\nthe GCN propagates each paper's features along citations, so papers citing each other")
print("(usually on the same topic) get similar representations: homophily again. Swap GCNConv")
print("for SAGEConv or GATConv to try GraphSAGE or attention; add layers to see farther (watch")
print("for oversmoothing); switch the head for link prediction or graph classification.")

print("\nAll Cora-GNN lab checks ran.")

# %% [markdown]
# Example 20-22. Swapping in a mean aggregator and attention.

# %%
from torch_geometric.nn import SAGEConv, GATConv

class SAGE(GCN):                                        # same net; only the convolution changes
    def __init__(self, nf, nc, h=16):
        super().__init__(nf, nc, h); self.c1 = SAGEConv(nf, h); self.c2 = SAGEConv(h, nc)

class GAT(GCN):                                         # 8 attention heads of 8 dims each
    def __init__(self, nf, nc, h=8, heads=8):
        super().__init__(nf, nc); self.c1 = GATConv(nf, h, heads=heads); self.c2 = GATConv(h * heads, nc)

torch.manual_seed(0)
print(f"SAGEConv mean aggregator, full neighborhood : {run(SAGE(nf, nc)):.3f}")
torch.manual_seed(0)
print(f"GAT       (GATConv swapped in)  test accuracy : {run(GAT(nf, nc)):.3f}")

# %% [markdown]
# ## 20.5.1 Label propagation: the links without the words

# %% [markdown]
# Example 20-23. Label propagation from scratch: a path, the karate club, and the closed form.

# %%
import numpy as np
import networkx as nx
import scipy.sparse as sp

def propagation_matrix(A):
    A = sp.csr_array(A)
    return sp.diags(1.0 / A.sum(1)) @ A                  # P = D^-1 A: row i averages node i's neighbors

def label_propagation(P, labeled, Y_L, clamp=True, tol=1e-6, max_steps=5000):
    Y = np.zeros((P.shape[0], Y_L.shape[1])); Y[labeled] = Y_L
    for step in range(1, max_steps + 1):
        Y_next = P @ Y                                   # propagate: each node takes its neighbors' mean
        if clamp:
            Y_next[labeled] = Y_L                        # clamp: the labeled rows return to their labels
        change = np.abs(Y_next - Y).max()
        Y = Y_next
        if change < tol:
            break
    return Y, step

path = nx.to_numpy_array(nx.path_graph(5))               # five nodes in a line, labels at the two ends
Y, _ = label_propagation(propagation_matrix(path), [0, 4], np.eye(2))
print("path, share of label A at nodes 0 to 4:", np.round(Y[:, 0], 3).tolist())

G = nx.karate_club_graph(); N = G.number_of_nodes()
A = nx.to_numpy_array(G, weight=None)
club = np.array([0 if G.nodes[v]["club"] == "Mr. Hi" else 1 for v in range(N)])
P = propagation_matrix(A)
rest = np.arange(1, N - 1)                               # every member except the leaders 0 and 33
Y, steps = label_propagation(P, [0, 33], np.eye(2))
missed = rest[Y[rest].argmax(1) != club[rest]]
print(f"karate club from the two leaders: {steps} steps, accuracy on the other 32 = "
      f"{(Y[rest].argmax(1) == club[rest]).mean():.2f}, missed {missed.tolist()}")
Pd = P.toarray()
Y_U = np.linalg.solve(np.eye(len(rest)) - Pd[np.ix_(rest, rest)], Pd[np.ix_(rest, [0, 33])])
print(f"closed form (I - P_UU)^-1 P_UL Y_L vs the loop: max difference {np.abs(Y_U - Y[rest]).max():.1e}")

Y1, _ = label_propagation(P, [0, 33], np.eye(2), max_steps=1)          # one round: a single neighbor vote
no_signal = int((Y1[rest].sum(1) == 0).sum())
ties = int(((Y1[rest, 0] == Y1[rest, 1]) & (Y1[rest].sum(1) > 0)).sum())
print(f"one round only: {no_signal} members get no label, {ties} are exact ties")
Yf, _ = label_propagation(P, [0, 33], np.eye(2), clamp=False)          # no clamp: the labels wash out
print("no clamp: every row ends near", np.round(Yf.mean(0), 4).tolist(),
      f"(largest gap {np.abs(Yf - Yf.mean(0)).max():.0e}), so {int((Yf.argmax(1) == 1).sum())} of {N} "
      f"members are called faction 1")

# %% [markdown]
# Example 20-24 reuses `data` from Example 20-21.
#
# Example 20-24. Label propagation on Cora from the 140 training labels, beside PyTorch Geometric's `LabelPropagation`.

# %%
from torch_geometric.nn import LabelPropagation

ei = data.edge_index.numpy()
A_cora = sp.csr_array((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(data.num_nodes, data.num_nodes))
y = data.y.numpy(); labeled = np.where(data.train_mask.numpy())[0]; test = data.test_mask.numpy()
Y, steps = label_propagation(propagation_matrix(A_cora), labeled, np.eye(7)[y[labeled]])
pred = Y.argmax(1)
print(f"label propagation on Cora: {steps} steps, test accuracy {(pred[test] == y[test]).mean():.3f}")
cut_off = Y[test].sum(1) == 0                            # no path to any of the 140 labeled papers
print(f"test papers no label can reach: {int(cut_off.sum())}; accuracy on the other "
      f"{int((~cut_off).sum())}: {(pred[test][~cut_off] == y[test][~cut_off]).mean():.3f}")
out = LabelPropagation(num_layers=50, alpha=0.9)(data.y, data.edge_index, mask=data.train_mask)
print(f"PyG LabelPropagation(num_layers=50, alpha=0.9): test accuracy "
      f"{(out.argmax(1)[data.test_mask] == data.y[data.test_mask]).float().mean().item():.3f}")

# %% [markdown]
# ## 20.5.2 Neighbor-sampled minibatches

# %% [markdown]
# Example 20-25. Neighbor sampling from scratch, and GraphSAGE trained on the sampled minibatches.

# %%
from torch_geometric.data import Data
from torch_geometric.utils import degree, k_hop_subgraph

deg = degree(data.edge_index[1], data.num_nodes)         # links per paper
hub = int(deg.argmax())                                  # the best-linked paper
whole = k_hop_subgraph(hub, 2, data.edge_index, num_nodes=data.num_nodes)[0].numel()
src, dst = data.edge_index                               # a message travels from src to dst
into = [[] for _ in range(data.num_nodes)]               # the edges that point into each paper
for e, v in enumerate(dst.tolist()):
    into[v].append(e)

def sample_piece(targets, fanouts, gen):
    """What NeighborLoader does: for each target draw up to fanouts[0] of the edges pointing into it, then up to
    fanouts[1] into each newly reached paper, without replacement; keep only the sampled edges; number the
    papers with the targets first."""
    nodes = [int(v) for v in targets]; local = {v: i for i, v in enumerate(nodes)}
    frontier, kept, per_hop = list(nodes), [], [len(nodes)]
    for fan in fanouts:
        new = []
        for v in frontier:
            edges = into[v]
            if len(edges) > fan:                         # more neighbors than the fanout: draw a subset
                edges = [edges[i] for i in torch.randperm(len(edges), generator=gen)[:fan].tolist()]
            for e in edges:
                kept.append(e); u = int(src[e])
                if u not in local:                       # a paper reached for the first time
                    local[u] = len(nodes); nodes.append(u); new.append(u)
        frontier = new; per_hop.append(len(new))         # the next hop expands only the new papers
    kept = torch.tensor(kept)
    edge_index = torch.tensor([[local[u] for u in src[kept].tolist()], [local[v] for v in dst[kept].tolist()]])
    n_id = torch.tensor(nodes)
    return Data(x=data.x[n_id], y=data.y[n_id], edge_index=edge_index, n_id=n_id,
                batch_size=len(nodes[:len(targets)]), num_sampled_nodes=per_hop)

gen = torch.Generator().manual_seed(0)                   # one seeded generator for every draw
probe = sample_piece(torch.tensor([hub]), [10, 10], gen) # one sampled piece for the hub
print(f"paper {hub}: {int(deg[hub])} links, {whole} papers within two hops, {probe.num_nodes} after "
      f"sampling, {probe.num_sampled_nodes} per hop (the cap is 1 + 10 + 100 = 111)")
targets = data.train_mask.nonzero().view(-1)             # the 140 labeled papers
batch = sample_piece(targets[torch.randperm(len(targets), generator=gen)[:32]], [10, 10], gen)
print(f"one minibatch: {batch.batch_size} targets in a sampled subgraph of {batch.num_nodes} papers "
      f"and {batch.num_edges} edges")
torch.manual_seed(0)
model = SAGE(nf, nc)                                     # Example 20-22's GraphSAGE network
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
for epoch in range(200):
    model.train()
    order = targets[torch.randperm(len(targets), generator=gen)]      # reshuffle the targets every epoch
    for i in range(0, len(order), 32):                   # 5 minibatches cover the 140 targets
        batch = sample_piece(order[i:i + 32], [10, 10], gen)          # a fresh sample for every minibatch
        opt.zero_grad()
        out = model(batch.x, batch.edge_index)[:batch.batch_size]     # the targets come first
        F.cross_entropy(out, batch.y[:batch.batch_size]).backward(); opt.step()
model.eval()
pred = model(data.x, data.edge_index).argmax(1)          # predict on the full graph
print(f"GraphSAGE trained on sampled minibatches: test accuracy "
      f"{(pred[data.test_mask] == data.y[data.test_mask]).float().mean().item():.3f}")

# %% [markdown]
# ## 20.6 Interview questions and coding exercises

# %% [markdown]
# ## Coding exercises

# %% [markdown]
# Coding exercise 1: Adjacency, degree, and the homophily vote, unweighted and weighted.

# %%
import networkx as nx
G = nx.karate_club_graph()
A = nx.to_numpy_array(G, weight=None)                     # unweighted: 1 where two members are friends
W = nx.to_numpy_array(G)                                  # weighted: the edge attribute weight
print(A.shape, int(A.sum() // 2), int(W.max()))

def neighbor_vote(M, i):
    ...

print(neighbor_vote(A, 0), neighbor_vote(W, 0))

# %% [markdown]
# Coding exercise 2: PageRank from scratch.

# %%
import networkx as nx
G = nx.les_miserables_graph()
nodes = sorted(G.nodes())
A = nx.to_numpy_array(G, nodelist=nodes, weight=None)    # unweighted: every coappearance counts once
print(G.number_of_nodes(), G.number_of_edges(), A.shape)

def pagerank_from_scratch(A, nodes):
    ...

print(pagerank_from_scratch(A, nodes))

# %% [markdown]
# Coding exercise 3: The graph Laplacian and the Fiedler split on Les Misérables.

# %%
import networkx as nx
G = nx.les_miserables_graph()
nodes = sorted(G.nodes())
print(G.number_of_nodes(), G.number_of_edges(), nx.is_connected(G))

def fiedler_sides(G, nodes):
    ...

print(fiedler_sides(G, nodes))

# %% [markdown]
# Coding exercise 4: DeepWalk node embeddings with longer walks and a wider window.

# %%
import networkx as nx
G = nx.karate_club_graph()
walk_length, window = 40, 5                              # Example 20-4 used 20 and 4
print(G.number_of_nodes(), G.number_of_edges(), walk_length, window)

def embedding_ari(G, walk_length, window):
    ...

print(embedding_ari(G, walk_length, window))

# %% [markdown]
# Coding exercise 5: TransE link prediction on FB15k-237.

# %%
import os, tempfile, torch
from torch_geometric.datasets import FB15k_237
root = os.path.join(tempfile.gettempdir(), "fb15k237")
splits = {s: FB15k_237(root, split=s)[0] for s in ["train", "val", "test"]}
print({s: d.num_edges for s, d in splits.items()}, splits["train"].num_nodes,
      int(max(d.edge_type.max() for d in splits.values())) + 1)

def train_transe(train_facts, num_entities, num_relations):
    ...

print(train_transe(splits["train"], splits["train"].num_nodes, 237))

# %% [markdown]
# Coding exercise 6: A from-scratch GCN versus an MLP with four seed labels.

# %%
import networkx as nx
G = nx.karate_club_graph()
seeds = [0, 33, 16, 24]                                  # the two leaders plus one more member of each faction
print(G.number_of_nodes(), [G.nodes[v]["club"] for v in seeds])

def gcn_and_mlp(G, seeds):
    ...

print(gcn_and_mlp(G, seeds))

# %% [markdown]
# Coding exercise 7: Link prediction with a leakage check that bites.

# %%
import numpy as np, networkx as nx
G = nx.les_miserables_graph()
nodes = sorted(G.nodes())
A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
rng = np.random.default_rng(1)                           # seed 1, not Example 20-16's seed 0
print(G.number_of_nodes(), G.number_of_edges(), int(round(0.3 * G.number_of_edges())))

def honest_and_leaky(A, rng):
    ...

print(honest_and_leaky(A, rng))

# %% [markdown]
# Coding exercise 8: Oversmoothing by hand.

# %%
import networkx as nx
G = nx.karate_club_graph()
print(G.number_of_nodes())

def relative_spread(G):
    ...

print(relative_spread(G))

# %% [markdown]
# Coding exercise 9: Miniproject: GCN versus MLP on CiteSeer with PyTorch Geometric.

# %%
import os
import tempfile
from torch_geometric.datasets import Planetoid
data = Planetoid(root=os.path.join(tempfile.gettempdir(), "citeseer"), name="CiteSeer")[0]
print(data.num_nodes, data.num_edges // 2, int(data.train_mask.sum()), int(data.test_mask.sum()))

def citeseer_accuracies(data):
    ...

print(citeseer_accuracies(data))

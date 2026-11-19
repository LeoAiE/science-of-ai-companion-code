# %% [markdown]
# # Section 20.5 Lab: node classification on Cora
#
# Regenerated from Book.md: Examples 20-21 and 20-22 and the code of Sections 20.5.1 and 20.5.2.
# Data: the Cora citation network through PyTorch Geometric's Planetoid loader, plus NetworkX's
# karate club and a path of five nodes in the label propagation check.

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
# ## The promised swap: GraphSAGE and GAT on the same lab

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

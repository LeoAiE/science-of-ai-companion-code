"""Chapter 14: Sequence Models: RNNs and Friends

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from Book.md; keep the book as the source.
"""

# %% [markdown]
# ## 14.2 The vanilla RNN

# %% [markdown]
# Example 14-1 implements the forward pass from scratch and confirms that it handles any length.

# %% [markdown]
# Example 14-1. A vanilla RNN forward pass written from scratch in NumPy, run with one set of weights on sequences of length 3 and 7.

# %%
import numpy as np
rng = np.random.default_rng(0)
def rnn_forward(xs, Wxh, Whh, b):
    h = np.zeros(Whh.shape[0]); states = []
    for x in xs:                                      # process one element at a time
        h = np.tanh(Wxh @ x + Whh @ h + b)            # the recurrence: new state from input + old state
        states.append(h)
    return np.array(states)
H, D = 4, 3
Wxh = rng.normal(0, 0.5, (H, D)); Whh = rng.normal(0, 0.5, (H, H)); b = np.zeros(H)
for length in [3, 7]:
    states = rnn_forward(rng.normal(size=(length, D)), Wxh, Whh, b)
    print(f"sequence length {length}: produced {states.shape[0]} hidden states of dim {states.shape[1]}")
# the SAME weights handle any length; the hidden state carries information forward

# %% [markdown]
# A two-step scalar check of the backpropagation-through-time sum.

# %%
import numpy as np
W, U, b = 0.5, 0.3, 0.0
xs = [1.0, -1.0]
h = 0.4
hs = [h]
for x in xs:
    h = np.tanh(U * x + W * h + b)
    hs.append(h)
h1, h2 = hs[1], hs[2]
dL = h2                                          # d(0.5 h2^2) / d h2
def direct(ht, hprev):
    return (1 - ht ** 2) * hprev                 # direct partial of h_t w.r.t. W, hprev held fixed
def jac(ht):
    return (1 - ht ** 2) * W                     # dh_t / dh_{t-1}
term1 = dL * jac(h2) * direct(h1, hs[0])        # k = 1, one Jacobian
term2 = dL * direct(h2, h1)                     # k = 2, empty product
def loss(Wv):
    hh = 0.4
    for x in xs:
        hh = np.tanh(U * x + Wv * hh + b)
    return 0.5 * hh ** 2
eps = 1e-6
fd = (loss(W + eps) - loss(W - eps)) / (2 * eps)
print(f"h1 = {h1:.6f}, h2 = {h2:.6f}")
print(f"k=1 {term1:.6e}, k=2 {term2:.6e}, sum {term1 + term2:.6e}")
print(f"finite difference {fd:.6e}, abs error {abs(fd - (term1 + term2)):.3e}")

# %% [markdown]
# Split the sequence into fixed-length chunks, run the forward pass chunk by chunk while carrying the hidden state forward, but detach that hidden state at each chunk boundary so backpropagation only ever flows within a single chunk.

# %% [markdown]
# Example 14-2. Truncated backpropagation through time on a 200-step random series, in chunks of 25 with the hidden state detached at each chunk boundary.

# %%
import torch, torch.nn as nn
torch.manual_seed(0)
rnn = nn.RNN(1, 8, batch_first=True); head = nn.Linear(8, 1)   # a tiny RNN over a long series
opt = torch.optim.Adam(list(rnn.parameters()) + list(head.parameters()), lr=0.01)
series = torch.randn(1, 200, 1); target = torch.randn(1, 200, 1)   # one very long sequence: 200 steps
chunk = 25                                              # backprop window: 25 steps, not the full 200
h = torch.zeros(1, 1, 8)                                # hidden state, carried across chunks
for start in range(0, series.size(1), chunk):
    h = h.detach()                                      # cut the graph so backprop stops at the chunk edge
    xc = series[:, start:start + chunk]
    out, h = rnn(xc, h)                                 # h flows forward, but its history is now detached
    loss = ((head(out) - target[:, start:start + chunk]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f"TBPTT: trained across {series.size(1)} steps in chunks of {chunk}; last-chunk loss = {loss.item():.4f}")

# %% [markdown]
# ## Recurrent layer shapes

# %%
import torch, torch.nn as nn
torch.manual_seed(0)
N, L, Hin, H = 2, 5, 3, 4
x_bf = torch.randn(N, L, Hin)
x_tb = x_bf.transpose(0, 1).contiguous()
lstm_bf = nn.LSTM(Hin, H, num_layers=2, bidirectional=True, batch_first=True)
lstm_tb = nn.LSTM(Hin, H, num_layers=2, bidirectional=True, batch_first=False)
lstm_tb.load_state_dict(lstm_bf.state_dict())
with torch.no_grad():
    out_bf, (hn_bf, cn_bf) = lstm_bf(x_bf)
    out_tb, (hn_tb, cn_tb) = lstm_tb(x_tb)
print("batch_first out", tuple(out_bf.shape), "h_n", tuple(hn_bf.shape), "c_n", tuple(cn_bf.shape))
print("time_first out", tuple(out_tb.shape), "h_n", tuple(hn_tb.shape))
print("h_n ignores batch_first:", torch.allclose(hn_bf, hn_tb))
print("top forward at last step:", torch.allclose(out_bf[:, -1, :H], hn_bf[-2]))
print("top backward at first step:", torch.allclose(out_bf[:, 0, H:], hn_bf[-1]))
print("top backward at last step equals h_n[-1]:", torch.allclose(out_bf[:, -1, H:], hn_bf[-1]))
torch.manual_seed(1)
seq = torch.randn(1, 3, Hin)
fused = nn.LSTM(Hin, H, batch_first=True)
cell = nn.LSTMCell(Hin, H)
with torch.no_grad():
    cell.weight_ih.copy_(fused.weight_ih_l0)
    cell.weight_hh.copy_(fused.weight_hh_l0)
    cell.bias_ih.copy_(fused.bias_ih_l0)
    cell.bias_hh.copy_(fused.bias_hh_l0)
h = torch.zeros(1, H); c = torch.zeros(1, H); steps = []
for t in range(seq.size(1)):
    h, c = cell(seq[:, t], (h, c))
    steps.append(h)
with torch.no_grad():
    fused_out, _ = fused(seq)
print("LSTMCell matches nn.LSTM:", torch.allclose(torch.stack(steps, 1), fused_out, atol=1e-6))

# %% [markdown]
# ## 14.3 The vanishing gradient problem in RNNs

# %% [markdown]
# Measuring that product's norm as the sequence lengthens shows it collapsing.

# %% [markdown]
# Example 14-3. The norm of a product of per-step RNN Jacobians for recurrent matrices of spectral norm 0.5 and 0.9, beside their spectral radii, and the vanishing powers of a matrix of spectral norm 1.3.

# %%
import numpy as np
def grad_through_time(spectral, T, H=4, seed=0):      # ||d h_T / d h_0|| and the spectral radius
    rng = np.random.default_rng(seed)                 # self-seeded, so every call reproduces exactly
    M = rng.normal(size=(H, H)); M = spectral * M / np.linalg.norm(M, 2)   # fix the spectral norm
    h = rng.normal(size=H) * 0.1; J = np.eye(H)
    for _ in range(T):
        h = np.tanh(M @ h); J = (np.diag(1 - h ** 2) @ M) @ J   # one Jacobian factor per step
    radius = float(np.max(np.abs(np.linalg.eigvals(M))))        # largest absolute eigenvalue
    return np.linalg.norm(J, 2), radius
print(f"{'T':>4}   spectral 0.5      spectral 0.9")
for T in [5, 10, 25, 50]:
    n05, _ = grad_through_time(0.5, T)
    n09, _ = grad_through_time(0.9, T)
    print(f"{T:>4}   {n05:>12.2e}   {n09:>12.2e}")
n05, r05 = grad_through_time(0.5, 50)
n09, r09 = grad_through_time(0.9, 50)
print(f"spectral radius: norm 0.5 -> {r05:.3f}, norm 0.9 -> {r09:.3f}")
print(f"per-step factor (norm at T=50) ** (1/50): {n05 ** (1/50):.3f} and {n09 ** (1/50):.3f}")
print(f"radius ** 50: {r05 ** 50:.2e} and {r09 ** 50:.2e}; upper bound 0.5 ** 50 = {0.5 ** 50:.2e}")
rng = np.random.default_rng(3)
A = rng.normal(size=(4, 4)); A = 1.3 * A / np.linalg.norm(A, 2)   # spectral norm 1.3
rad = float(np.max(np.abs(np.linalg.eigvals(A))))
P = np.eye(4); powers = {}
for t in range(1, 101):
    P = A @ P
    if t in (10, 50, 100):
        powers[t] = np.linalg.norm(P, 2)
print(f"seed 3, spectral norm 1.3, radius {rad:.3f}: "
      f"||A^10||={powers[10]:.2e}, ||A^50||={powers[50]:.2e}, ||A^100||={powers[100]:.2e}")

# %% [markdown]
# after the backward pass, rescale every gradient so their combined norm never exceeds a chosen ceiling.

# %%
import torch, torch.nn as nn
torch.manual_seed(0)
net = nn.RNN(1, 8, batch_first=True)
out, _ = net(torch.randn(1, 60, 1)); loss = (out ** 2).sum(); loss.backward()   # a deliberately large loss
before = torch.sqrt(sum(p.grad.pow(2).sum() for p in net.parameters()))         # global gradient norm
nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)                        # rescale so the norm <= 1.0
after = torch.sqrt(sum(p.grad.pow(2).sum() for p in net.parameters()))
print(f"gradient clipping: global grad norm {before.item():.3f} -> {after.item():.3f}")

# %% [markdown]
# to isolate the effect of dependency distance we plant the answer at an exactly known gap , which no naturally occurring text lets us control.

# %% [markdown]
# Example 14-4. A 16-unit vanilla RNN trained to recall a bit planted at the first step, at gaps of 10, 50, and 150.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(2)      # the gap-150 accuracy changes in its last digits with the thread count
def memory_data(n, T, seed):
    r = np.random.default_rng(seed)
    X = r.normal(0, 0.3, (n, T, 1)).astype("float32"); bit = r.integers(0, 2, n)
    X[:, 0, 0] = bit                                  # the answer is planted at step 0
    return torch.tensor(X), torch.tensor(bit)
class RNNClf(nn.Module):
    def __init__(self):
        super().__init__(); self.rnn = nn.RNN(1, 16, batch_first=True); self.fc = nn.Linear(16, 2)
    def forward(self, x):
        out, _ = self.rnn(x); return self.fc(out[:, -1])   # classify from the LAST hidden state
for T in [10, 50, 150]:
    torch.manual_seed(0)
    Xtr, ytr = memory_data(800, T, seed=1); Xte, yte = memory_data(400, T, seed=2)
    m = RNNClf(); opt = torch.optim.Adam(m.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
    for _ in range(200):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        print(f"  gap T={T:>3}: test accuracy = {(m(Xte).argmax(1) == yte).float().mean().item():.3f}")
# gap 10: 1.000   gap 50: 1.000   gap 150: 0.498 (near chance): the gradient vanished before it could learn the long dependency

# %% [markdown]
# ## 14.4 Long short-term memory (LSTM)

# %% [markdown]
# grad_through_time is the helper of Example 14-3, reused unchanged.

# %%
n100, _ = grad_through_time(0.9, 100)
print(f"vanilla RNN ||d h_T/d h_0|| at T=100 = {n100:.2e}  (vanished, uncontrollable)")
print("LSTM cell-state gradient over 100 steps = (forget gate)^100:")
for f in [0.90, 0.99, 1.00]:
    print(f"  forget gate ~ {f:.2f}:  {f ** 100:.3e}")
# the RNN gradient is gone; the forget-gate product stays near 1 when the gate stays near 1: the LSTM CHOOSES to remember

# %% [markdown]
# These two behaviors, the exponential collapse and the flat forget-gate product, are exactly the solid and dashed curves that Figure 14-2 plotted back in 14.3; the LSTM is the dashed line made real.

# %% [markdown]
# Example 14-5. One LSTM step written from scratch in NumPy, run for five steps on toy inputs, with the cell update checked by hand on unit 0.

# %%
import numpy as np
def lstm_step(x, h, c, P):
    z = P["Wx"] @ x + P["Wh"] @ h + P["b"]            # one matmul, split into 4 blocks (3 gates + candidate)
    f, i, o, g = np.split(z, 4)
    f, i, o = 1/(1+np.exp(-f)), 1/(1+np.exp(-i)), 1/(1+np.exp(-o))   # sigmoid gates in (0,1)
    g = np.tanh(g)                                    # the candidate update
    c = f * c + i * g                                 # cell state: forget old, add new
    h = o * np.tanh(c)                                # hidden = a gated view of the cell
    return h, c, (f, i, o, g)
rng = np.random.default_rng(0)                        # reseed so this block reproduces standalone
Hd = 4
P = {"Wx": rng.normal(0, 0.3, (4*Hd, 3)), "Wh": rng.normal(0, 0.3, (4*Hd, Hd)), "b": np.zeros(4*Hd)}
h, c = np.zeros(Hd), np.zeros(Hd)
xs = rng.normal(size=(5, 3))
for x in xs:
    c_prev = c                                        # keep the old cell so we can check the update by hand
    h, c, (f, i, o, g) = lstm_step(x, h, c, P)
print(f"last-step forget gate f = {np.round(f, 3)}")
print(f"last-step input  gate i = {np.round(i, 3)}")
print(f"last-step output gate o = {np.round(o, 3)}")
print(f"last-step candidate  g = {np.round(g, 3)}")
print(f"unit 0: c = f*c_prev + i*g = {f[0]:.3f}*{c_prev[0]:.3f} + {i[0]:.3f}*{g[0]:.3f} = {f[0]*c_prev[0] + i[0]*g[0]:.3f}")
print(f"final cell state c = {np.round(c, 3)}, norm = {np.linalg.norm(c):.3f}")

# %% [markdown]
# nn.LSTM stacks its four gate blocks in the order input, forget, candidate, output, while our P stacks them forget, input, output, candidate, so the rows are reordered before copying, and PyTorch keeps two bias vectors where we keep one, so

# %%
import torch, torch.nn as nn
lstm = nn.LSTM(input_size=3, hidden_size=Hd, batch_first=True)   # PyTorch's built-in LSTM
# nn.LSTM stacks its gate blocks in the order i, f, g, o, where P stacks f, i, o, g:
# reorder the rows, load the weights, and the module becomes OUR cell exactly
reorder = np.r_[Hd:2*Hd, 0:Hd, 3*Hd:4*Hd, 2*Hd:3*Hd]             # our [f,i,o,g] rows -> torch's [i,f,g,o]
with torch.no_grad():
    lstm.weight_ih_l0.copy_(torch.tensor(P["Wx"][reorder], dtype=torch.float32))
    lstm.weight_hh_l0.copy_(torch.tensor(P["Wh"][reorder], dtype=torch.float32))
    lstm.bias_ih_l0.copy_(torch.tensor(P["b"][reorder], dtype=torch.float32))
    lstm.bias_hh_l0.zero_()                                      # PyTorch keeps two bias vectors; ours is one
seq = torch.tensor(xs, dtype=torch.float32).unsqueeze(0)         # the SAME toy input: (batch=1, T=5, input=3)
with torch.no_grad():
    out, (hn, cn) = lstm(seq)                                    # out: every hidden state; hn, cn: final pair
print(f"nn.LSTM: out {tuple(out.shape)}, final h {tuple(hn.shape)}, final c {tuple(cn.shape)}")
print(f"from-scratch final h = {np.round(h, 4)}")
print(f"nn.LSTM      final h = {np.round(hn[0, 0].numpy(), 4)}")
diff = max(np.abs(h - hn[0, 0].numpy()).max(), np.abs(c - cn[0, 0].numpy()).max())
print("max |difference| over final h and c below 1e-6:", bool(diff < 1e-6))

# %% [markdown]
# ## 14.5 The GRU

# %% [markdown]
# The reset gate multiplies the previous state before it feeds the candidate, and the update gate then blends that candidate with the old state, component by component.

# %% [markdown]
# Example 14-6. One GRU step written from scratch in NumPy, run for five steps on toy inputs, with the interpolation checked by hand on unit 0.

# %%
import numpy as np
rng = np.random.default_rng(0)
def gru_step(x, h, P):
    z = 1/(1+np.exp(-(P["Wz"] @ x + P["Uz"] @ h + P["bz"])))       # update gate in (0,1)
    r = 1/(1+np.exp(-(P["Wr"] @ x + P["Ur"] @ h + P["br"])))       # reset gate in (0,1)
    h_tilde = np.tanh(P["Wh"] @ x + P["Uh"] @ (r * h) + P["bh"])   # candidate uses reset-gated past
    h_new = (1 - z) * h + z * h_tilde                             # interpolate old state vs candidate
    return h_new, z, r, h_tilde, h
Hd, D = 4, 3
def W(a, b): return rng.normal(0, 0.3, (a, b))
P = {"Wz": W(Hd, D), "Uz": W(Hd, Hd), "bz": np.zeros(Hd),
     "Wr": W(Hd, D), "Ur": W(Hd, Hd), "br": np.zeros(Hd),
     "Wh": W(Hd, D), "Uh": W(Hd, Hd), "bh": np.zeros(Hd)}
h = np.zeros(Hd)
for x in rng.normal(size=(5, D)):
    h, z, r, h_tilde, h_prev = gru_step(x, h, P)
print(f"from-scratch GRU ran 5 steps; final hidden norm = {np.linalg.norm(h):.3f}")   # 0.201
print(f"last update gate z = {np.round(z, 3)}")   # [0.453 0.529 0.536 0.477]
print(f"last reset  gate r = {np.round(r, 3)}")   # [0.576 0.504 0.47  0.332]
hand = (1 - z[0]) * h_prev[0] + z[0] * h_tilde[0]
print(f"unit 0: (1-z)*h_prev + z*h_tilde = {(1-z[0]):.3f}*{h_prev[0]:.3f} + {z[0]:.3f}*{h_tilde[0]:.3f} = {hand:.3f}")
print(f"unit 0 of the new state = {h[0]:.3f}")

# %% [markdown]
# This is the same gated interpolation the LSTM performs, reached with two gates instead of three and no separate cell state, which is exactly why the GRU is lighter.

# %% [markdown]
# Example 14-7 builds the three-term Jacobian of the update rule in the function `gru_jacobian`, on the weights `P` of Example 14-6. It checks the formula against central differences on one step, then multiplies 100 of these Jacobians along a run of 100 random inputs, once with the update-gate bias at 0, as built, and once at $-5$, which holds $z_t$ near 0.

# %% [markdown]
# Example 14-7. The three-term GRU Jacobian checked against central differences, then multiplied over 100 steps with the update-gate bias at 0 and at $-5$.

# %%
def gru_jacobian(x, h, P):                            # the three-term dh_t/dh_{t-1} for one step
    h_new, z, r, h_tilde, _ = gru_step(x, h, P)
    dz = (z * (1 - z))[:, None] * P["Uz"]             # dz_t/dh_{t-1}, from the sigmoid
    dr = (r * (1 - r))[:, None] * P["Ur"]             # dr_t/dh_{t-1}, from the sigmoid
    dcand = (1 - h_tilde ** 2)[:, None] * (P["Uh"] @ (np.diag(r) + h[:, None] * dr))   # candidate path
    J = np.diag(1 - z) + (h_tilde - h)[:, None] * dz + z[:, None] * dcand
    return h_new, J, z
rng_j = np.random.default_rng(1)                      # fresh inputs, so this block reproduces on its own
x1, h1 = rng_j.normal(size=D), rng_j.normal(0, 0.5, Hd)
_, J1, _ = gru_jacobian(x1, h1, P)
eps = 1e-6
fd = np.column_stack([(gru_step(x1, h1 + eps * e, P)[0] - gru_step(x1, h1 - eps * e, P)[0]) / (2 * eps)
                      for e in np.eye(Hd)])           # central differences, one column per unit
print("one step: formula matches central differences:", np.allclose(J1, fd, atol=1e-8))
xs = rng_j.normal(size=(100, D))
for bz in [0.0, -5.0]:
    Q = dict(P, bz=np.full(Hd, bz))                   # same weights, update-gate bias moved
    hh, Jprod, keep, zsum = np.zeros(Hd), np.eye(Hd), np.ones(Hd), 0.0
    for x in xs:
        hh, J, zz = gru_jacobian(x, hh, Q)
        Jprod = J @ Jprod                             # full product over 100 steps
        keep = keep * (1 - zz); zsum += zz.mean()     # direct-path product of (1 - z), per unit
    print(f"b_z = {bz:4.1f}: mean z = {zsum / len(xs):.3f}, "
          f"||dh_100/dh_0|| = {np.linalg.norm(Jprod, 2):.3e}, largest prod(1 - z) = {keep.max():.3e}")

# %% [markdown]
# That is the GRU's version of the LSTM's 0.99^100 = 0.366 in 14.4: an update gate held near 0 keeps the gradient, and a gate left near one half loses it.

# %%
import torch.nn as nn
def n_params(cell): return sum(p.numel() for p in cell(1, 32, batch_first=True).parameters())
print(f"RNN  : {n_params(nn.RNN):>5}  (1 weight block)")
print(f"GRU  : {n_params(nn.GRU):>5}  (3 weight blocks: reset gate, update gate, candidate)")
print(f"LSTM : {n_params(nn.LSTM):>5}  (4 weight blocks: forget, input, output gates + candidate)")
# RNN 1120, GRU 3360, LSTM 4480: each extra block is another weight matrix, so more parameters but more control

# %% [markdown]
# ## 14.5.1 Gated cells on the memory task

# %% [markdown]
# Example 14-8. LSTM and GRU cells on the memory task of 14.3, with default and opened keep gates.

# %%
import warnings; warnings.filterwarnings("ignore")
import math, torch, torch.nn as nn
threads = torch.get_num_threads(); torch.set_num_threads(1)   # one intra-op thread, so the printed numbers reproduce
H = 16
class GatedClf(nn.Module):                            # 14.3's classifier, with the recurrent cell as a choice
    def __init__(self, cell, open_for=None):
        super().__init__(); self.rnn = cell(1, H, batch_first=True); self.fc = nn.Linear(H, 2)
        if open_for is not None:                      # start the keep gate at open_for / (open_for + 1)
            b = math.log(open_for)
            with torch.no_grad():                     # rows H:2H are the LSTM forget gate and the GRU update gate
                self.rnn.bias_ih_l0[H:2*H].fill_(b); self.rnn.bias_hh_l0[H:2*H].zero_()
                if cell is nn.LSTM:                   # rows 0:H, the LSTM input gate, start at 1 / (open_for + 1)
                    self.rnn.bias_ih_l0[:H].fill_(-b); self.rnn.bias_hh_l0[:H].zero_()
    def forward(self, x):
        out, _ = self.rnn(x); return self.fc(out[:, -1])
def start_signal(cell, opened, T=150):              # before training: mean |dL/dx| at step 0 over that at the last step
    torch.manual_seed(0); m = GatedClf(cell, T if opened else None)
    X, y = memory_data(800, T, seed=1); X.requires_grad_(True)
    nn.CrossEntropyLoss()(m(X), y).backward()
    return (X.grad[:, 0].abs().mean() / X.grad[:, -1].abs().mean()).item()
def gap_accuracy(cell, opened, T):                  # 14.3's data, seeds, and 200 full-batch Adam steps
    torch.manual_seed(0)
    Xtr, ytr = memory_data(800, T, seed=1); Xte, yte = memory_data(400, T, seed=2)
    m = GatedClf(cell, T if opened else None)
    opt = torch.optim.Adam(m.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
    for _ in range(200):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        return (m(Xte).argmax(1) == yte).float().mean().item()
configs = [(nn.RNN, False), (nn.LSTM, False), (nn.GRU, False), (nn.LSTM, True), (nn.GRU, True)]
def name(cell, opened): return f"{cell.__name__}, {'gate opened' if opened else 'default init'}"
print("gradient at step 0 / gradient at the last step, gap 150, before training:")
for cell, opened in configs:
    print(f"  {name(cell, opened):<18} {start_signal(cell, opened):.1e}")
print("test accuracy after 200 steps at gaps 10, 50, 150:")
for cell, opened in configs[1:]:
    print(f"  {name(cell, opened):<18} " + "  ".join(f"{gap_accuracy(cell, opened, T):.3f}" for T in (10, 50, 150)))
torch.set_num_threads(threads)                        # restore the thread count for the blocks that follow

# %% [markdown]
# ## 14.6 Bidirectional and stacked RNNs

# %% [markdown]
# A per-token task that needs the right neighbor shows the difference.

# %% [markdown]
# Example 14-9. Unidirectional and bidirectional LSTM taggers trained to mark the local peaks of random sequences, where each label depends on the next input.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
def peak_data(n, T, seed):
    r = np.random.default_rng(seed)
    x = r.uniform(0, 1, (n, T)).astype("float32"); y = np.zeros((n, T), "int64")
    y[:, 1:-1] = ((x[:, 1:-1] > x[:, :-2]) & (x[:, 1:-1] > x[:, 2:])).astype("int64")   # local peaks
    return torch.tensor(x).unsqueeze(-1), torch.tensor(y)
class Tagger(nn.Module):
    def __init__(self, bidir):
        super().__init__()
        self.rnn = nn.LSTM(1, 32, batch_first=True, bidirectional=bidir)
        self.fc = nn.Linear(64 if bidir else 32, 2)
    def forward(self, x):
        out, _ = self.rnn(x); return self.fc(out)     # a label at EVERY position
def train_eval(bidir):
    torch.manual_seed(0)
    Xtr, ytr = peak_data(600, 30, 1); Xte, yte = peak_data(300, 30, 2)
    m = Tagger(bidir); opt = torch.optim.Adam(m.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
    for _ in range(80):
        opt.zero_grad(); lf(m(Xtr).reshape(-1, 2), ytr.reshape(-1)).backward(); opt.step()
    with torch.no_grad():
        return (m(Xte).argmax(-1) == yte).float().mean().item()
print(f"unidirectional LSTM : accuracy = {train_eval(False):.3f}")   # 0.861  cannot see x[i+1] when labeling i
print(f"bidirectional LSTM  : accuracy = {train_eval(True):.3f}")    # 0.933  a backward pass sees the whole sequence

# %% [markdown]
# Counting them directly takes two lines, since num_layers is just an argument.

# %%
count = lambda m: sum(p.numel() for p in m.parameters())
print(f"1 layer: {count(nn.LSTM(1, 32, batch_first=True)):,}   "
      f"3 layers: {count(nn.LSTM(1, 32, num_layers=3, batch_first=True)):,}")

# %% [markdown]
# ## 14.6.1 Real text: embeddings, padding, and masking

# %% [markdown]
# Training uses a 4,000-review subset and testing a 1,000-review subset, which keeps the listing small enough to train on a CPU.

# %%
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
torch.set_num_threads(2)      # the packed accuracy changes in its last digit with the thread count
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from collections import Counter
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
from datasets import load_dataset
torch.manual_seed(0)

ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")   # real movie-review sentiment
train = ds["train"].shuffle(seed=0).select(range(4000))
test = ds["test"].shuffle(seed=0).select(range(1000))

PAD, UNK = 0, 1
counts = Counter(w for ex in train for w in ex["text"].split())
vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(4000))}   # top-4000 words + PAD + UNK
Vsize = len(vocab) + 2

def encode(t): return [vocab.get(w, UNK) for w in t.split()][:40]
def tensors(split):
    rows = [(torch.tensor(encode(ex["text"])), ex["label"]) for ex in split if ex["text"].split()]
    seqs = [r[0] for r in rows]; ys = torch.tensor([r[1] for r in rows])
    lens = torch.tensor([len(s) for s in seqs])
    return seqs, lens, ys

tr_s, tr_l, tr_y = tensors(train); te_s, te_l, te_y = tensors(test)

demo = pad_sequence(tr_s[:3], batch_first=True, padding_value=PAD)
print("raw lengths of 3 reviews:", tr_l[:3].tolist(), "-> one padded batch shape:", tuple(demo.shape))

def iterate(seqs, lens, ys, bs=64, shuffle=True):
    order = torch.randperm(len(seqs)) if shuffle else torch.arange(len(seqs))
    for k in range(0, len(seqs), bs):
        b = order[k:k + bs]
        padded = pad_sequence([seqs[i] for i in b], batch_first=True, padding_value=PAD)
        yield padded, lens[b], ys[b]

class BiLSTM(nn.Module):
    def __init__(self, V, emb=64, hid=64, mask=True):
        super().__init__()
        self.mask = mask
        self.emb = nn.Embedding(V, emb, padding_idx=PAD)     # the PAD row is held at zero
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hid, 2)
    def forward(self, padded, lengths):
        e = self.emb(padded)
        if self.mask:
            packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h, _) = self.lstm(packed)        # packing tells the LSTM to STOP at each real length
        else:
            _, (h, _) = self.lstm(e)             # naive: the forward pass runs through trailing PAD tokens before h[-2] is read
        return self.fc(torch.cat([h[-2], h[-1]], dim=1))   # last forward state + last backward state

def run(mask):
    torch.manual_seed(0)
    m = BiLSTM(Vsize, mask=mask)
    opt = torch.optim.Adam(m.parameters(), lr=0.005); lf = nn.CrossEntropyLoss()
    for _ in range(6):
        for padded, lengths, yy in iterate(tr_s, tr_l, tr_y):
            opt.zero_grad(); lf(m(padded, lengths), yy).backward(); opt.step()
    correct = total = 0
    with torch.no_grad():
        for padded, lengths, yy in iterate(te_s, te_l, te_y, shuffle=False):
            correct += (m(padded, lengths).argmax(1) == yy).sum().item(); total += len(yy)
    return correct / total

print(f"vocab size = {Vsize}")
print(f"bidirectional LSTM, naive (pad tokens fed in)      : accuracy = {run(False):.3f}")
print(f"bidirectional LSTM, packed/masked (pads skipped)   : accuracy = {run(True):.3f}")

# %% [markdown]
# ## Dropout on a recurrent network

# %%
import warnings, torch, torch.nn as nn
torch.manual_seed(0)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    layer = nn.LSTM(3, 4, num_layers=1, dropout=0.3, batch_first=True)
print("num_layers", layer.num_layers, "dropout arg", layer.dropout, "warnings", len(caught))
p = 0.3
torch.manual_seed(0)
mask = (torch.rand(4) > p).float() / (1 - p)          # one mask for the sequence
h = torch.tensor([0.2, -0.4, 0.6, -0.8])
print("mask", [round(float(v), 3) for v in mask])
print("masked h, step A", [round(float(v), 3) for v in mask * h])
print("masked h, step B", [round(float(v), 3) for v in mask * h])
fresh = (torch.rand(4) > p).float() / (1 - p)
print("a second draw", [round(float(v), 3) for v in fresh])

# %% [markdown]
# ## 14.7 1D convolutions and TCNs for sequences

# %% [markdown]
# It uses causal convolutions, which look only at the past so they can predict the next step, and dilated convolutions, which skip a growing number of positions between filter taps, so the receptive field expands exponentially with depth.

# %%
K = 3
rf_plain = 1 + sum((K - 1) for _ in range(8))                       # plain stack: receptive field grows linearly
dilations = [1, 2, 4, 8, 16, 32, 64, 128]
rf_dilated = 1 + sum((K - 1) * d for d in dilations)                # dilated stack: grows exponentially
print(f"8 plain conv layers   : receptive field = {rf_plain} steps")     # 17  (linear)
print(f"8 dilated conv layers : receptive field = {rf_dilated} steps")   # 511 (exponential): a TCN sees far back, fully in parallel

# %% [markdown]
# ## 14.7.1 nn.Conv1d: causal and dilated convolutions in code

# %% [markdown]
# To grow the receptive field cheaply, the dilation spaces the filter taps apart, so a length-3 filter with dilation 8 reads positions , , and .

# %% [markdown]
# Example 14-11. A causal dilated Conv1d layer, a check of its causality, the measured receptive field of an eight-layer dilated stack, and plain and dilated five-layer TCNs trained to recall a bit planted at the first of 40 steps.

# %%
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation                 # left pad only, so no output sees the future
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)
    def forward(self, x):                             # x: (batch, channels, time)
        return self.conv(F.pad(x, (self.pad, 0)))     # pad = (left, right); right=0 keeps it causal

# --- causality check: perturbing the LAST input must not change any earlier output ---
layer = CausalConv1d(1, 1, k=3, dilation=1)
x = torch.randn(1, 1, 6)
y1 = layer(x)
x2 = x.clone(); x2[0, 0, -1] += 5.0                   # change only the final time step
y2 = layer(x2)
print("causal check: outputs at t<5 unchanged when only t=5 input changes:",
      torch.allclose(y1[0, 0, :5], y2[0, 0, :5]))

# --- empirical receptive field of an 8-layer dilated stack: how many inputs feed the LAST output ---
dils = [1, 2, 4, 8, 16, 32, 64, 128]
stack = nn.Sequential(*[CausalConv1d(1, 1, 3, d) for d in dils])
T = 700; base = torch.zeros(1, 1, T, requires_grad=True)
stack(base)[0, 0, -1].backward()                      # gradient is nonzero exactly on the receptive field
rf = int((base.grad[0, 0].abs() > 1e-12).sum())
print(f"dilated stack empirical receptive field at the last step = {rf} steps")

# --- controlled long-range memory task: predict a bit planted at the FAR-PAST position 0 ---
# Synthetic on purpose: a real text corpus cannot guarantee the label depends on a token an EXACT
# distance T back, which is precisely what isolates the receptive-field payoff.
def make_copy(n, T=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (n, 1, T), generator=g).float() * 2 - 1   # +/-1 noise
    b = torch.randint(0, 2, (n,), generator=g)
    x[:, 0, 0] = b.float() * 2 - 1                    # the ONLY informative value sits at position 0
    return x, b

class TCN(nn.Module):
    def __init__(self, dils, ch=16):
        super().__init__()
        layers = []; cin = 1
        for d in dils:
            layers += [CausalConv1d(cin, ch, 3, d), nn.ReLU()]; cin = ch
        self.net = nn.Sequential(*layers); self.head = nn.Linear(ch, 2)
    def forward(self, x):
        return self.head(self.net(x)[:, :, -1])       # read the final time step

def train_tcn(dils, Tlen=40):
    torch.manual_seed(0)
    m = TCN(dils); opt = torch.optim.Adam(m.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
    Xtr, ytr = make_copy(2000, Tlen, seed=1)
    for _ in range(60):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    Xte, yte = make_copy(600, Tlen, seed=2)
    with torch.no_grad():
        return (m(Xte).argmax(1) == yte).float().mean().item()

acc_plain = train_tcn([1, 1, 1, 1, 1])                # 5 plain layers: RF = 1 + 2*5 = 11 < 40
acc_dil   = train_tcn([1, 2, 4, 8, 16])               # 5 dilated layers: RF = 1 + 2*31 = 63 >= 40
print(f"plain   5-layer TCN (RF 11): copy-at-distance-40 accuracy = {acc_plain:.3f}")
print(f"dilated 5-layer TCN (RF 63): copy-at-distance-40 accuracy = {acc_dil:.3f}")

# %% [markdown]
# ## 14.8 Sequence-to-sequence models

# %% [markdown]
# Demonstrating this with a retrieval task, where a specific item must be recovered, shows the loss.

# %%
import numpy as np
rng = np.random.default_rng(0)
n, d = 8, 16
K = rng.normal(size=(n, d))                           # encoder "keys" (one per input position)
V = rng.normal(size=(n, d))                           # encoder "values"
target = 5
def cos(a, b): return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
bottleneck = V.mean(axis=0)                           # a seq2seq encoder squashes everything into ONE vector
print(f"single fixed vector (bottleneck): cosine to V[{target}] = {cos(bottleneck, V[target]):.3f}")
# 0.573: averaging all positions into one summary loses the specific item, the seq2seq bottleneck

# %% [markdown]
# ## 14.8.1 Building a seq2seq model with teacher forcing

# %% [markdown]
# a full translation corpus needs far more data and compute than a few CPU seconds allow, whereas reversal is a genuine sequence-to-sequence transduction whose correct output is known exactly, so it isolates the encoder-decoder-plus-teacher-f

# %%
import torch, torch.nn as nn
torch.manual_seed(0)
V = 10; BOS = 10; VOCAB = 11; L = 8      # tokens 0..9 are content; id 10 is a start-of-sequence marker

def make_batch(n, seed):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, V, (n, L), generator=g)
    tgt = torch.flip(src, dims=[1])       # the target is the input reversed
    return src, tgt

class Encoder(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, 32); self.lstm = nn.LSTM(32, h, batch_first=True)
    def forward(self, x):
        _, (h, c) = self.lstm(self.emb(x)); return (h, c)          # the context: final hidden and cell

class Decoder(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, 32); self.lstm = nn.LSTM(32, h, batch_first=True); self.fc = nn.Linear(h, VOCAB)
    def forward(self, tokens, state):
        out, state = self.lstm(self.emb(tokens), state); return self.fc(out), state

enc, dec = Encoder(), Decoder()
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=0.01)
lf = nn.CrossEntropyLoss()
for step in range(400):
    src, tgt = make_batch(128, seed=step)
    state = enc(src)
    dec_in = torch.cat([torch.full((128, 1), BOS), tgt[:, :-1]], dim=1)   # TEACHER FORCING: feed ground-truth prev token
    logits, _ = dec(dec_in, state)
    opt.zero_grad(); lf(logits.reshape(-1, VOCAB), tgt.reshape(-1)).backward(); opt.step()

# evaluate with autoregressive (free-running) decoding: feed the model its OWN prediction
src, tgt = make_batch(500, seed=9999)
with torch.no_grad():
    state = enc(src)
    tok = torch.full((500, 1), BOS); preds = []
    for _ in range(L):
        logits, state = dec(tok, state)
        tok = logits[:, -1].argmax(-1, keepdim=True); preds.append(tok)
    preds = torch.cat(preds, dim=1)
tokacc = (preds == tgt).float().mean().item()
seqacc = (preds == tgt).all(dim=1).float().mean().item()
print(f"reverse task, free-running decode: token accuracy = {tokacc:.3f}")          # 0.988
print(f"reverse task, free-running decode: whole-sequence accuracy = {seqacc:.3f}")  # 0.962
print("example src :", src[0].tolist())    # [2, 5, 6, 1, 5, 7, 0, 3]
print("example pred:", preds[0].tolist())  # [3, 0, 7, 5, 1, 6, 5, 2]
print("example tgt :", tgt[0].tolist())    # [3, 0, 7, 5, 1, 6, 5, 2]

# %% [markdown]
# ## 14.9 Attention

# %% [markdown]
# Implementing it from scratch and querying for a specific item shows it retrieve cleanly.

# %% [markdown]
# Example 14-13. Scaled dot-product attention written from scratch in NumPy, queried for the item at position 5 of 8.

# %%
import numpy as np
rng = np.random.default_rng(0)
def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()
def attention(q, K, V):
    w = softmax(K @ q / np.sqrt(K.shape[1]))          # alignment score per position -> weights summing to 1
    return w @ V, w                                   # context = weighted sum of values
n, d = 8, 16
K = rng.normal(size=(n, d)); V = rng.normal(size=(n, d)); target = 5
q = K[target] + rng.normal(0, 0.1, d)                 # a query that matches position 5
context, w = attention(q, K, V)
def cos(a, b): return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"weights sum to {w.sum():.3f}; weight on position {target} = {w[target]:.3f} (highest of {n})")
print(f"cosine(context, true value V[{target}]) = {cos(context, V[target]):.3f}")
# weight 0.644 on the matching position; cosine 0.965 to the true value (vs the bottleneck's 0.573): attention RETRIEVES it

# %% [markdown]
# The cost is quadratic, every output position attends to every input position.

# %%
print("attention scores (every query attends every key) = n^2:")
for length in [8, 64, 512]:
    print(f"  length {length:>3}: {length * length:>7,} pairwise scores")
# 8 -> 64,  64 -> 4,096,  512 -> 262,144: the quadratic cost is the price of the bottleneck-free view (the transformer challenge of Chapter 16)

# %% [markdown]
# The decoder is trained with teacher forcing and its gradient norm is clipped, the same two habits the lab in 14.10 uses.

# %%
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); rng = np.random.default_rng(0)
# Controlled task: reverse a sequence. output[i] copies input[L-1-i], so the correct
# alignment is the antidiagonal, a known target we can read straight off the weights.
PAD, BOS, EOS, L, NSYM = 0, 1, 2, 7, 10
def batch(n):
    core = rng.integers(3, 3 + NSYM, size=(n, L))         # content symbols 3..12
    src = np.concatenate([core, np.full((n, 1), EOS)], 1)
    rev = core[:, ::-1]
    tgt_in  = np.concatenate([np.full((n, 1), BOS), rev], 1)   # teacher-forcing input
    tgt_out = np.concatenate([rev, np.full((n, 1), EOS)], 1)   # target to predict
    return torch.tensor(src), torch.tensor(tgt_in), torch.tensor(tgt_out)
VOCAB = 3 + NSYM
class AttnSeq2Seq(nn.Module):
    def __init__(self, emb=32, hid=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, emb, padding_idx=PAD)
        self.enc = nn.GRU(emb, hid, batch_first=True)
        self.dec = nn.GRU(emb, hid, batch_first=True)
        self.W1 = nn.Linear(hid, hid, bias=False)         # additive (concat) score, Luong placement:
        self.W2 = nn.Linear(hid, hid, bias=False)         # v^T tanh(W1 h_dec + W2 h_enc)
        self.v  = nn.Linear(hid, 1, bias=False)
        self.out = nn.Linear(hid * 2, VOCAB)
    def forward(self, src, tgt_in):
        enc_h, h = self.enc(self.emb(src))                # every encoder state, kept for lookup
        dec_h, _ = self.dec(self.emb(tgt_in), h)
        scores = self.v(torch.tanh(self.W1(dec_h)[:, :, None] + self.W2(enc_h)[:, None])).squeeze(-1)
        attn = F.softmax(scores, dim=-1)                  # (B, U, S): weight over source per step
        ctx = attn @ enc_h                                # context = weighted sum of encoder states
        return self.out(torch.cat([dec_h, ctx], -1)), attn
model = AttnSeq2Seq(); opt = torch.optim.Adam(model.parameters(), lr=0.005); lf = nn.CrossEntropyLoss()
for step in range(600):
    src, ti, to = batch(64); logits, _ = model(src, ti)
    opt.zero_grad(); lf(logits.reshape(-1, VOCAB), to.reshape(-1)).backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
src, ti, to = batch(500)
with torch.no_grad():
    logits, attn = model(src, ti); pred = logits.argmax(-1)
print(f"reverse task: token accuracy {(pred == to).float().mean().item():.3f}, "
      f"exact-sequence accuracy {(pred == to).all(1).float().mean().item():.3f}")
print("source position attended per decoder step:", attn[0].argmax(1).tolist())

# %% [markdown]
# ## 14.10 Hands-on lab: text generation and sequence classification

# %% [markdown]
# To measure honestly, Example 14-15 also scores the model on text it never trained on, a 2,000-character slice of the dataset's own held-out test split, and it checks how much of the generated sample is copied verbatim from the training text.

# %%
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
torch.set_num_threads(2)      # the held-out loss changes in its last digit with the thread count
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
from datasets import load_dataset
torch.manual_seed(0)
raw = load_dataset("Trelis/tiny-shakespeare")         # real Shakespeare, character level
text = "".join(raw["train"]["Text"])[:8000]           # a small slice, about 8000 characters
chars = sorted(set(text)); V = len(chars)             # vocabulary of distinct characters
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text]); SEQ = 40
X = torch.stack([data[i:i+SEQ] for i in range(len(data)-SEQ)])       # each row: 40 characters
Y = torch.stack([data[i+1:i+SEQ+1] for i in range(len(data)-SEQ)])   # the next character at every step
val_text = "".join(c for c in "".join(raw["test"]["Text"])[:2400] if c in stoi)[:2000]   # the dataset's own
vdata = torch.tensor([stoi[c] for c in val_text])                    # held-out test split: text never trained on
Xv = torch.stack([vdata[i:i+SEQ] for i in range(len(vdata)-SEQ)])
Yv = torch.stack([vdata[i+1:i+SEQ+1] for i in range(len(vdata)-SEQ)])
class CharLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 32); self.lstm = nn.LSTM(32, 128, batch_first=True); self.fc = nn.Linear(128, V)
    def forward(self, x, h=None):
        out, h = self.lstm(self.emb(x), h); return self.fc(out), h
model = CharLSTM(); opt = torch.optim.Adam(model.parameters(), lr=0.004); lf = nn.CrossEntropyLoss()
for epoch in range(20):
    perm = torch.randperm(len(X))
    for k in range(0, len(X), 128):
        idx = perm[k:k+128]; logits, _ = model(X[idx])          # teacher forcing: the true chars are the inputs
        opt.zero_grad(); lf(logits.reshape(-1, V), Y[idx].reshape(-1)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)       # gradient clipping guards the BPTT blow-up
        opt.step()
model.eval()
with torch.no_grad():
    tr_logits, _ = model(X); val_logits, _ = model(Xv)
    tr_acc  = (tr_logits.argmax(-1) == Y).float().mean().item()
    val_acc = (val_logits.argmax(-1) == Yv).float().mean().item()
    val_loss = lf(val_logits.reshape(-1, V), Yv.reshape(-1)).item()
print(f"train next-char accuracy = {tr_acc:.3f}, held-out accuracy = {val_acc:.3f}, held-out loss = {val_loss:.3f}")
def generate(seed, n=200, temp=0.6):                  # sample one character at a time, feeding it back in
    idx = torch.tensor([[stoi[c] for c in seed]]); h = None; out = seed
    with torch.no_grad():
        logits, h = model(idx, h)
        for _ in range(n):
            nxt = int(torch.multinomial(torch.softmax(logits[0, -1] / temp, -1), 1))
            out += itos[nxt]; logits, h = model(torch.tensor([[nxt]]), h)
    return out
sample = generate("First Citizen:\n")
print(sample)
cont = sample[len("First Citizen:\n"):]               # how much of the sample is copied training text?
k = 0
while k < len(cont) and cont[:k+1] in text: k += 1
print(f"longest verbatim prefix of the continuation found in the training slice: {k} of {len(cont)} characters")

# %% [markdown]
# Example 14-16. Example 14-15's character model retrained with one dropout mask per sequence on its input, recurrent, and output connections.

# %%
import torch, torch.nn as nn
threads = torch.get_num_threads(); torch.set_num_threads(1)   # one intra-op thread, so the printed numbers reproduce
class VarCharLSTM(nn.Module):                         # Example 14-15's model, stepped one character at a time
    def __init__(self, p):
        super().__init__(); self.p = p
        self.emb = nn.Embedding(V, 32); self.cell = nn.LSTMCell(32, 128); self.fc = nn.Linear(128, V)
    def mask(self, n, size, like):                    # one mask per sequence, kept units scaled by 1/(1-p)
        if not self.training or self.p == 0:
            return 1.0
        return torch.bernoulli(like.new_full((n, size), 1 - self.p)) / (1 - self.p)
    def forward(self, x, state=None):
        e = self.emb(x); n = x.size(0)
        h, c = state if state is not None else (e.new_zeros(n, 128), e.new_zeros(n, 128))
        m_in, m_rec, m_out = self.mask(n, 32, e), self.mask(n, 128, e), self.mask(n, 128, e)   # drawn once
        outs = []
        for t in range(x.size(1)):                    # the same three masks at every step
            h, c = self.cell(e[:, t] * m_in, (h * m_rec, c))
            outs.append(h * m_out)
        return self.fc(torch.stack(outs, 1)), (h, c)
def train_and_score(p):
    torch.manual_seed(0)
    model = VarCharLSTM(p); opt = torch.optim.Adam(model.parameters(), lr=0.004); lf = nn.CrossEntropyLoss()
    for epoch in range(20):                           # Example 14-15's schedule, unchanged
        model.train(); perm = torch.randperm(len(X))
        for k in range(0, len(X), 128):
            idx = perm[k:k+128]; logits, _ = model(X[idx])
            opt.zero_grad(); lf(logits.reshape(-1, V), Y[idx].reshape(-1)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    model.eval()                                      # no masks at evaluation: every unit is used
    with torch.no_grad():
        tr_logits, _ = model(X); val_logits, _ = model(Xv)
        tr_acc = (tr_logits.argmax(-1) == Y).float().mean().item()
        val_acc = (val_logits.argmax(-1) == Yv).float().mean().item()
        val_loss = lf(val_logits.reshape(-1, V), Yv.reshape(-1)).item()
        logits, state = model(torch.tensor([[stoi[ch] for ch in "First Citizen:\n"]])); cont = ""
        for _ in range(200):                          # 200 characters at temperature 0.6, as Example 14-15 samples
            nxt = int(torch.multinomial(torch.softmax(logits[0, -1] / 0.6, -1), 1))
            cont += itos[nxt]; logits, state = model(torch.tensor([[nxt]]), state)
    k = 0
    while k < len(cont) and cont[:k+1] in text: k += 1
    print(f"p = {p}: train accuracy = {tr_acc:.3f}, held-out accuracy = {val_acc:.3f}, "
          f"held-out loss = {val_loss:.3f}, verbatim prefix = {k} of 200")
    return cont
samples = {p: train_and_score(p) for p in (0.0, 0.3)}
print(samples[0.3])
torch.set_num_threads(threads)                        # restore the thread count for the blocks that follow

# %% [markdown]
# Reviews are right-padded to a fixed length of 40 so a batch forms one rectangular tensor, the embedding is built with padding_idx=0 so the pad slot stays a zero vector that gets no gradient, and the classifier reads the LSTM state at each review's last real word rather than at a trailing pad.

# %%
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from collections import Counter
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
from datasets import load_dataset
torch.manual_seed(0)
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")     # real movie-review sentiment
def toks(s): return s.split()                                      # the same whitespace tokenizer as 14.6.1
counts = Counter(w for s in ds["train"]["text"] for w in toks(s))
vocab = [w for w, _ in counts.most_common(4000)]                   # keep the 4000 most frequent words
stoi = {w: i + 2 for i, w in enumerate(vocab)}                     # index 0 = padding, index 1 = unknown
T = 40                                                             # the same truncation length as 14.6.1
def encode(s):
    ids = [stoi.get(w, 1) for w in toks(s)][:T]
    return ids + [0] * (T - len(ids))                             # right-pad short reviews with 0
def tensors(split):
    return (torch.tensor([encode(s) for s in ds[split]["text"]]),
            torch.tensor(ds[split]["label"]))
Xtr, ytr = tensors("train"); Xte, yte = tensors("test")
class SentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(stoi) + 2, 64, padding_idx=0)  # the pad row stays zero and is ignored
        self.lstm = nn.LSTM(64, 96, batch_first=True); self.fc = nn.Linear(96, 2)
    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1)                    # real length of each review, ignoring pad
        out, _ = self.lstm(self.emb(x))
        last = out[torch.arange(len(x)), lengths - 1]             # take the state at the last real word
        return self.fc(last)
clf = SentLSTM(); opt = torch.optim.Adam(clf.parameters(), lr=0.003); lf = nn.CrossEntropyLoss()
for epoch in range(8):
    perm = torch.randperm(len(Xtr))
    for k in range(0, len(Xtr), 64):
        idx = perm[k:k+64]
        opt.zero_grad(); lf(clf(Xtr[idx]), ytr[idx]).backward(); opt.step()
with torch.no_grad():
    print(f"vocab {len(stoi)+2}, test reviews {len(Xte)}")
    print(f"test accuracy = {(clf(Xte).argmax(1) == yte).float().mean().item():.3f}")

# %% [markdown]
# Example 14-18. A bag-of-words logistic regression on Example 14-17's split, vocabulary, and first 40 tokens.

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
def first_words(s): return toks(s)[:T]                         # the same first 40 tokens that encode() keeps
bag = CountVectorizer(analyzer=first_words, vocabulary=vocab)   # one column per word of the 4000-word vocabulary
C_tr = bag.transform(ds["train"]["text"]); C_te = bag.transform(ds["test"]["text"])   # a word outside it adds nothing
bow = LogisticRegression(max_iter=2000).fit(C_tr, ds["train"]["label"])   # one weight per word count, order ignored
print(f"count matrix {C_tr.shape}, test accuracy = {bow.score(C_te, ds['test']['label']):.3f}")

# %% [markdown]
# ## 14.10 your turn

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
from datasets import load_dataset
raw = load_dataset("Trelis/tiny-shakespeare")
text = "".join(raw["train"]["Text"])
a, b = text[:8000], text[:80000]
print(len(a), len(set(a)), len(b), len(set(b)))

# %% [markdown]
# ## Temperature

# %%
import torch
torch.manual_seed(0)
z = torch.tensor([2.0, 1.0])
for T in (0.6, 1.0, 1.5):
    p = torch.softmax(z / T, dim=0)
    print(f"T={T}: P(first) = {p[0].item():.3f}")
g = torch.Generator().manual_seed(0)
draws = torch.multinomial(torch.softmax(z / 0.6, 0), 10000, replacement=True, generator=g)
print(f"frequency of the first token in 10000 draws at T=0.6: {(draws == 0).float().mean().item():.3f}")

# %%
import numpy as np
def rnn_forward(xs, Wxh, Whh, b):
    ...
print(rnn_forward([np.zeros(2)], np.zeros((3, 2)), np.zeros((3, 3)), np.zeros(3)))

# %%
def grad_norm(spectral, T):
    ...
print(grad_norm(0.5, 10))

# %%
import torch
def lstm_step(x, h, c, weights):
    ...
print(lstm_step(torch.zeros(3), torch.zeros(4), torch.zeros(4), None))

# %%
import torch.nn as nn
def n_params(cell):
    ...
print(n_params(nn.RNN))

# %%
import torch
def memory_data(n, T, seed):
    ...
print(memory_data(4, 10, 0))

# %%
import torch.nn as nn
class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, k, dilation):
        super().__init__()
        ...
    def forward(self, x):
        ...
print(CausalConv1d(1, 1, 3, 1))

# %%
import numpy as np
def attention(q, K, V):
    ...
print(attention(np.zeros(2), np.zeros((3, 2)), np.zeros((3, 2))))

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
from datasets import load_dataset
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
print(len(ds["train"]), len(ds["test"]))

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
from datasets import load_dataset
raw = load_dataset("Trelis/tiny-shakespeare")
text = "".join(raw["train"]["Text"])[:80000]
print(len(text), len(set(text)))

# %%
import torch
def make_batch(n, seed):
    ...
print(make_batch(2, 0))

# %%
import torch, torch.nn as nn
torch.manual_seed(0)
series = torch.randn(1, 200, 1)
print(tuple(series.shape))

# %%
import numpy as np
def additive_weights(s, enc, W1, W2, v):
    ...
print(additive_weights(np.zeros(3), np.zeros((5, 3)), np.eye(3), np.eye(3), np.ones(3)))

# %%
import torch
from torch.nn.utils.rnn import pack_padded_sequence
print(pack_padded_sequence(torch.zeros(1, 4, 2), torch.tensor([3]), batch_first=True, enforce_sorted=False).batch_sizes.tolist())

# %%
import torch, torch.nn as nn
torch.manual_seed(0)
layer = nn.LSTM(1, 16)
H = 16
print(round(float(torch.sigmoid(layer.bias_ih_l0[H:2 * H]).mean()), 3))

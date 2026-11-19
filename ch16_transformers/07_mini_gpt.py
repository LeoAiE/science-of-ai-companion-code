# %% [markdown]
# # Section 16.9 - A mini-GPT from scratch, trained on tiny-shakespeare
#
# Regenerated from Book.md (single source of truth). Each code cell below is a
# verbatim example from the chapter prose, apart from the last, which runs the
# bfloat16 experiment of Section 16.10.2; running the script reproduces the
# numbers the book reports.

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

torch.manual_seed(0)
torch.set_num_threads(2)   # pin two CPU threads: the trained weights change slightly with the thread count

# Real public data: Trelis/tiny-shakespeare, ~30k characters is plenty for a tiny model.
ds = load_dataset("Trelis/tiny-shakespeare")["train"]
text = "".join(ds[i]["Text"] for i in range(12))    # concatenate the first 12 rows

chars = sorted(set(text))
V = len(chars)                                       # the vocabulary is the distinct characters
stoi = {c: i for i, c in enumerate(chars)}           # char -> integer id
itos = {i: c for i, c in enumerate(chars)}           # integer id -> char
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
print(f"corpus: {len(text)} chars, vocab size V = {V}")

block_size = 32        # context length: how many previous chars the model sees at once
n_embd = 64            # embedding / model dimension
n_head = 4             # attention heads (each of size n_embd / n_head = 16)
n_layer = 3            # stacked transformer blocks
batch_size = 32
steps = 2000
lr = 3e-3

def get_batch():                                     # a batch of (context, next-char) pairs
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])   # targets = inputs shifted by one
    return x, y

# %%
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd)        # queries, keys, values at once
        self.proj = nn.Linear(n_embd, n_embd)           # mix heads back together
        self.n_head = n_head
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)   # (B, heads, T, head_size)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (hs ** 0.5)        # scaled dot-product scores
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))   # mask the future
        att = F.softmax(att, dim=-1)
        out = att @ v                                         # weighted sum of values
        out = out.transpose(1, 2).contiguous().view(B, T, C) # concatenate heads
        return self.proj(out)

# %%
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))      # residual around attention
        x = x + self.ffn(self.ln2(x))       # residual around feed-forward
        return x

# %%
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, n_embd)              # token -> vector
        self.pos_emb = nn.Embedding(block_size, n_embd)     # position -> vector
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, V)                    # vector -> next-token logits

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T)
        x = self.tok_emb(idx) + self.pos_emb(pos)           # token meaning + position
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new):
        for _ in range(n_new):
            logits, _ = self(idx[:, -block_size:])          # crop to the context window
            probs = F.softmax(logits[:, -1, :], dim=-1)     # next-token distribution
            nxt = torch.multinomial(probs, 1)               # sample one token
            idx = torch.cat([idx, nxt], dim=1)
        return idx

# %%
model = MiniGPT()
n_params = sum(p.numel() for p in model.parameters())
print(f"mini-GPT: {n_layer} layers, {n_head} heads, d={n_embd}, {n_params:,} parameters")
print(f"loss of a random model should be about ln(V) = {torch.log(torch.tensor(float(V))):.3f}")

# Rows 12-23 of the same dataset are never in the training string (rows 0-11).
held_text = "".join(ds[i]["Text"] for i in range(12, 24))
held = torch.tensor([stoi[c] for c in held_text if c in stoi], dtype=torch.long)
print(f"held-out: {len(held_text)} chars, {len(held)} in the vocabulary, {len(held_text) - len(held)} dropped")
g = torch.Generator().manual_seed(1)
val_batches = []
for _ in range(8):
    ix = torch.randint(0, len(held) - block_size - 1, (batch_size,), generator=g)
    xb = torch.stack([held[i:i + block_size] for i in ix])
    yb = torch.stack([held[i + 1:i + block_size + 1] for i in ix])
    val_batches.append((xb, yb))

def val_loss():
    total = 0.0
    with torch.no_grad():
        for xb, yb in val_batches:
            _, vloss = model(xb, yb)
            total += vloss.item()
    return total / len(val_batches)

opt = torch.optim.AdamW(model.parameters(), lr=lr)
_, loss0 = model(*get_batch())
print(f"\nstep    0: loss {loss0.item():.3f}  (perplexity {torch.exp(loss0).item():.1f})  val {val_loss():.3f}")
for step in range(1, steps + 1):
    x, y = get_batch()
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(f"step {step:4d}: loss {loss.item():.3f}  (perplexity {torch.exp(loss).item():.2f})  val {val_loss():.3f}")

# %%
torch.manual_seed(1)
start = torch.tensor([[stoi['T']]], dtype=torch.long)   # seed with a capital T
out = model.generate(start, n_new=300)[0].tolist()
print("".join(itos[i] for i in out))

# %% [markdown]
# ## Experiment of Section 16.10.2: the Accelerate loop under bfloat16 mixed precision
#
# Section 16.10.2 suggests passing `mixed_precision="bf16"` to the `Accelerator` and rerunning
# Example 16-20 in a fresh session, because Accelerate refuses to change the settings of an
# `Accelerator` already created in the same process. The walkthrough creates a plain one, so the experiment
# runs here, in a session that has created none: the same listing with that one argument added, then the
# largest gap between the two lists of losses and the last loss of each.

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

windows = data.unfold(0, block_size + 1, block_size)        # 33-character windows, a new one every 32
train_set = TensorDataset(windows[:, :-1], windows[:, 1:])  # (context, next-character targets) pairs

def one_epoch(accelerator=None):
    torch.manual_seed(0)                                    # the same starting weights on every call
    model = MiniGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(0))   # the same batch order on every call
    if accelerator is not None:
        model, opt, loader = accelerator.prepare(model, opt, loader)  # wrapped for the setup it found
    losses = []
    for x, y in loader:                                     # with Accelerate, on accelerator.device
        _, loss = model(x, y)
        opt.zero_grad()
        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)                      # in place of loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

plain = one_epoch()                                         # a plain PyTorch loop
accelerator = Accelerator(mixed_precision="bf16", cpu=True)           # the experiment: bfloat16 autocast
accel = one_epoch(accelerator)                              # the same loop through Accelerate
print(f"device {accelerator.device}, processes {accelerator.num_processes}, "
      f"mixed precision {accelerator.mixed_precision}")
print(f"{len(train_set)} windows, {len(plain)} steps; loss {plain[0]:.3f} at the first step, "
      f"{plain[-1]:.3f} at the last")
print("identical losses with and without Accelerate:", plain == accel)
print(f"largest gap between the two lists of losses: {max(abs(p - q) for p, q in zip(plain, accel)):.4f}")
print(f"last loss: {accel[-1]:.4f} under bfloat16, {plain[-1]:.4f} in 32-bit floats")

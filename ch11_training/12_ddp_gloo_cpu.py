"""Chapter 11, section 11.9.3: Distributed Data Parallel and fully_shard, really run on the CPU.

Two processes on one machine talk over the gloo backend on localhost. Each holds the
DigitNet of 11.8.2, trains on its own share of every batch of the scikit-learn digits
data, and DDP averages the gradients so both replicas take the same step. A replay of
the same 64-row batches in one process checks that this equals one large batch, and a
second run shards the same model with fully_shard (FSDP2).

Run it with:  python 12_ddp_gloo_cpu.py
It is a plain script, not a notebook cell, because torch.multiprocessing.spawn starts
fresh processes that import their worker function from this file.
"""
import socket
import torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

WORLD, PER_RANK, EPOCHS = 2, 32, 20   # 2 processes x 32 rows: the batch of 64 that 11.8.2 uses

class DigitNet(nn.Module):            # the 64-32-10 network of 11.8.2
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    def forward(self, x): return self.net(x)

def data():                           # the split and scaling of 11.8.2
    Xd, yd = load_digits(return_X_y=True)
    Xd = StandardScaler().fit_transform(Xd)
    Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0)
    t = lambda a, dt: torch.tensor(a, dtype=dt)
    return TensorDataset(t(Xtr, torch.float32), t(ytr, torch.long)), t(Xte, torch.float32), t(yte, torch.long)

def say(*a): print(*a, flush=True)    # flush, so lines from different processes keep their order

def flat(model):                      # every parameter in one vector, to compare models
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

def correct(model, X, y):             # how many test images the model gets right
    model.eval()
    with torch.no_grad():
        return int((model(X).argmax(1) == y).sum())

def gather(t):                        # one tensor from every rank, in rank order
    out = [torch.zeros_like(t) for _ in range(WORLD)]
    dist.all_gather(out, t)
    return out

def train(model, train_ds):           # the four-step loop of 11.8.2, on this rank's rows
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)   # built after the wrap
    lossf = nn.CrossEntropyLoss()
    sampler = DistributedSampler(train_ds)                 # this rank's share, reshuffled per epoch
    loader = DataLoader(train_ds, batch_size=PER_RANK, sampler=sampler)
    steps = 0
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)                           # a new shuffle, the same on every rank
        for xb, yb in loader:
            opt.zero_grad()
            lossf(model(xb), yb).backward()                # the gradients are averaged across ranks here
            opt.step(); steps += 1
    sampler.set_epoch(0)
    return dict(rows=len(sampler), batches=len(loader), steps=steps, epoch0=list(sampler))

def ddp_worker(rank, init, queue):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=WORLD)
    train_ds, Xte, yte = data()
    torch.manual_seed(rank)           # a different seed on each rank, on purpose
    model = DigitNet()
    before = gather(flat(model))
    model = DDP(model)                # CPU tensors, so no device_ids
    after = gather(flat(model))
    info = train(model, train_ds)
    final = gather(flat(model))
    info["right"] = correct(model.module, Xte, yte)
    infos = [None] * WORLD
    dist.all_gather_object(infos, info)
    if rank == 0:
        for r, i in enumerate(infos):
            say(f"rank {r}: {i['rows']} rows per epoch, {i['batches']} batches per epoch, {i['steps']} steps")
        say("rows ranks 0 and 1 both see in epoch 0:", len(set(infos[0]["epoch0"]) & set(infos[1]["epoch0"])))
        say(f"max |rank 0 - rank 1| before the wrap {(before[0] - before[1]).abs().max():.4f}, "
            f"after it {(after[0] - after[1]).abs().max():.3g}, "
            f"rank 1 against rank 0's start {(after[1] - before[0]).abs().max():.3g}")
        say(f"max |rank 0 - rank 1| after training, over {final[0].numel()} parameters: "
            f"{(final[0] - final[1]).abs().max():.3g}")
        for r, i in enumerate(infos):
            say(f"rank {r} test accuracy {i['right'] / len(yte):.3f} ({i['right']} of {len(yte)})")
        queue.put(final[0].numpy())
    dist.destroy_process_group()

def one_process():                    # the same 64-row batches, in the same order, in one process
    torch.set_num_threads(1)
    train_ds, Xte, yte = data()
    Xtr, ytr = train_ds.tensors
    torch.manual_seed(0); model = DigitNet()               # rank 0's starting weights
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
    shares = [DistributedSampler(train_ds, num_replicas=WORLD, rank=r) for r in range(WORLD)]
    for epoch in range(EPOCHS):
        for s in shares:
            s.set_epoch(epoch)
        idx = [list(s) for s in shares]
        for start in range(0, len(idx[0]), PER_RANK):
            batch = sum((i[start:start + PER_RANK] for i in idx), [])   # every rank's rows at this step
            opt.zero_grad(); lossf(model(Xtr[batch]), ytr[batch]).backward(); opt.step()
    return model, correct(model, Xte, yte), len(yte)

def fsdp_worker(rank, init, ddp_final):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=WORLD)
    train_ds, Xte, yte = data()
    try:
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.device_mesh import init_device_mesh
        # Explicit CPU mesh: on a Mac torch may pick MPS as the default device type for
        # fully_shard's mesh, and torch.mps has no is_initialized (AttributeError).
        mesh = init_device_mesh("cpu", mesh_shape=(WORLD,), mesh_dim_names=("fsdp",))
        torch.manual_seed(0); model = DigitNet()           # the same seed on both ranks
        for layer in model.net:
            if isinstance(layer, nn.Linear):
                fully_shard(layer, mesh=mesh)              # shard the blocks first
        fully_shard(model, mesh=mesh)                      # then the root
        shards = {n: tuple(p.to_local().shape) for n, p in model.named_parameters()}
        local = sum(p.to_local().numel() for p in model.parameters())
        info = train(model, train_ds)
        full = torch.cat([p.full_tensor().detach().reshape(-1) for p in model.parameters()])
        info.update(shards=shards, local=local, right=correct(model, Xte, yte),
                    diff=(full - torch.from_numpy(ddp_final)).abs().max().item())
    except Exception as e:
        info = dict(error=f"{type(e).__name__}: {e}")
    infos = [None] * WORLD
    dist.all_gather_object(infos, info)
    if rank == 0:
        for r, i in enumerate(infos):
            if "error" in i:
                say(f"rank {r}: fully_shard failed: {i['error']}"); continue
            say(f"rank {r}: local shards {i['shards']}, {i['local']} of {ddp_final.size} values")
        if "error" not in infos[0]:
            i = infos[0]
            say(f"FSDP2, {i['steps']} steps: test accuracy {i['right'] / len(yte):.3f} "
                f"({i['right']} of {len(yte)}), max |FSDP2 - DDP| = {i['diff']:.3g}")
    dist.destroy_process_group()

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]

if __name__ == "__main__":
    say(f"torch {torch.__version__}, gloo available: {dist.is_gloo_available()}, "
        f"{WORLD} processes x {PER_RANK} rows = batch {WORLD * PER_RANK}, {EPOCHS} epochs")
    queue = mp.get_context("spawn").Queue()
    ctx = mp.spawn(ddp_worker, args=(f"tcp://127.0.0.1:{free_port()}", queue), nprocs=WORLD, join=False)
    ddp_final = queue.get(timeout=600)
    while not ctx.join():
        pass
    base, right, n = one_process()
    gap = float((flat(base) - torch.from_numpy(ddp_final)).abs().max())
    say(f"one process, same batches: test accuracy {right / n:.3f} ({right} of {n}), "
        f"max |DDP - one process| below 1e-6: {gap < 1e-6}")
    mp.spawn(fsdp_worker, args=(f"tcp://127.0.0.1:{free_port()}", ddp_final), nprocs=WORLD, join=True)

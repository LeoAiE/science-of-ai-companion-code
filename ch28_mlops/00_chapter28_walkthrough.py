"""Chapter 28: MLOps & Deploying AI Systems

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from Book.md; keep the book as the source.
"""

# %% [markdown]
# ## 28.1 Before you model: scoping an ML project

# %% [markdown]
# Scope first, model second.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
rng = np.random.default_rng(0)

# Task: flag transactions over $1000. The rule is KNOWN and exact - ML is the wrong tool.
X = rng.uniform(0, 2000, size=(2000, 1))
y = (X[:, 0] > 1000).astype(int)                            # the true label IS a simple threshold
Xtr, Xte, ytr, yte = X[:1500], X[1500:], y[:1500], y[1500:]

rule_pred = (Xte[:, 0] > 1000).astype(int)                  # a one-line rule, no training
rule_acc = (rule_pred == yte).mean()
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr)
tree_acc = tree.score(Xte, yte)
print("deterministic task ('amount > 1000'):")
print(f"  one-line rule accuracy : {rule_acc:.3f}   (zero training, instant, fully transparent)")
print(f"  trained tree accuracy  : {tree_acc:.3f}   (needs data, training, and can drift)")
print("  -> when the rule is known and exact, ML only adds cost and opacity. Don't use ML.")

# Business goal: a follow-up team can review only the top 40 highest-risk of 171 held-out
# cases; catch as many malignant tumors as possible. Real data: breast cancer Wisconsin
# (569 tumors x 30 features), with the malignant class as the positive, high-value target.
data = load_breast_cancer()
X_all, y_all = data.data, (data.target == 0).astype(int)    # sklearn: 0=malignant -> positive=1
Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.3, random_state=0, stratify=y_all)
budget = 40

base_rate = yte.mean()                                      # baseline: a random pick catches the base rate
random_caught = base_rate * budget
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr)
risk = model.predict_proba(Xte)[:, 1]
top = np.argsort(-risk)[:budget]                            # review the 40 highest-risk cases
model_caught = int(yte[top].sum())
n_malignant = int(yte.sum())                                # every malignant case in the held-out split
print(f"\ntargeting the top {budget} of {len(yte)} by predicted risk (malignant base rate {base_rate:.0%}):")
print(f"  random selection catches  ~{random_caught:.0f} malignant cases   (the baseline to beat)")
print(f"  model-ranked selection catches {model_caught} malignant cases")
print(f"  held-out malignant cases: {n_malignant}; precision {model_caught}/{budget} = {model_caught / budget:.3f}, "
      f"recall {model_caught}/{n_malignant} = {model_caught / n_malignant:.3f}")
print(f"  lift = {model_caught / random_caught:.1f}x over the base rate: THIS is the model's business value.")

print("\nscope first: skip ML when a rule will do, and never ship a model without a baseline and a")
print("success metric stated in the numbers the business actually cares about.")

print("\nAll scoping checks ran.")

# %% [markdown]
# ## 28.2 The ML system lifecycle: why notebooks are not production

# %% [markdown]
# Point-in-time correctness is easy to see in miniature with pandas. The store keeps a feature's full history, one row per recomputation, and `merge_asof` joins each training example to the most recent feature value at or before the example's own timestamp, never a later one. The snippet contrasts that with the naive shortcut of pasting each customer's newest value onto every historical example.

# %%
# A feature store in miniature: point-in-time-correct retrieval with merge_asof.
import pandas as pd

feature_history = pd.DataFrame({                            # the store: avg_spend recomputed over time
    "customer": ["A", "A", "A", "B", "B"],
    "day":      [1, 32, 60, 15, 51],                        # the day each feature value was computed
    "avg_spend": [120.0, 95.0, 260.0, 40.0, 55.0]})
examples = pd.DataFrame({                                   # training examples with their own timestamps
    "customer": ["A", "B", "A"],
    "day":      [20, 32, 41],
    "label":    [0, 1, 0]}).sort_values("day").reset_index(drop=True)

correct = pd.merge_asof(examples, feature_history.sort_values("day"),
                        on="day", by="customer")            # latest value AT OR BEFORE each example's day
newest = feature_history.sort_values("day").groupby("customer")["avg_spend"].last()

print("point-in-time-correct retrieval (merge_asof) vs a naive latest-value join:")
for _, row in correct.iterrows():
    naive = newest[row["customer"]]                         # WRONG: today's value pasted onto an old row
    leak = "  <- future value leaked" if naive != row["avg_spend"] else ""
    print(f"  customer {row['customer']}, example at day {row['day']:>2}: "
          f"as-of value {row['avg_spend']:6.1f} | naive latest {naive:6.1f}{leak}")

# %% [markdown]
# This snippet also lives in the chapter's companion script for 28.2.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pickle, os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
rng = np.random.default_rng(0)
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

def ingest():                                               # stage 1: get data (real, bundled with sklearn)
    data = load_breast_cancer()                             # 569 tumors x 30 features, malignant/benign
    Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3,
                                          random_state=0, stratify=data.target)
    return Xtr, ytr, Xte, yte
def fit_preprocess(Xtr):                                    # stage 2: FIT the transform on TRAIN only
    return StandardScaler().fit(Xtr)
def train(Xtr_s, ytr):                                      # stage 3: fit the model on scaled train
    return LogisticRegression(max_iter=5000).fit(Xtr_s, ytr)

Xtr, ytr, Xte, yte = ingest()
scaler = fit_preprocess(Xtr)                                # this fitted scaler is an ARTIFACT to save
model = train(scaler.transform(Xtr), ytr)
acc = model.score(scaler.transform(Xte), yte)              # stage 4: evaluate
with open(os.path.join(HERE, "artifacts.pkl"), "wb") as f: # stage 5: PERSIST both artifacts together
    pickle.dump({"scaler": scaler, "model": model}, f)
print("lifecycle stages ran: ingest -> preprocess(fit) -> train -> evaluate -> persist")
print(f"  held-out accuracy = {acc:.3f}")
print(f"  artifacts saved: {os.path.basename(os.path.join(HERE, 'artifacts.pkl'))} (scaler + model together)")

# At serving, requests usually arrive ONE row at a time. We reload the saved artifacts and
# score each held-out row as if it were a live request.
with open(os.path.join(HERE, "artifacts.pkl"), "rb") as f:
    art = pickle.load(f)

right_preds, wrong_preds = [], []
for row in Xte:
    r = row.reshape(1, -1)
    right_preds.append(art["model"].predict(art["scaler"].transform(r))[0])   # RIGHT: saved scaler
    own = StandardScaler().fit(r).transform(r)               # WRONG: fit on this 1 row, so its mean IS the row...
    wrong_preds.append(art["model"].predict(own)[0])         # ...x - mean = 0, and var 0 gets scale 1 -> all 0
right = (np.array(right_preds) == yte).mean()
wrong = (np.array(wrong_preds) == yte).mean()
print("\nscoring each held-out row as a live single-request:")
print(f"  RIGHT: reuse saved training scaler         : accuracy {right:.3f}")
print(f"  WRONG: fit a fresh scaler on each request  : accuracy {wrong:.3f}   (collapses to majority-class share {max(yte.mean(),1-yte.mean()):.3f})")
print("\nfitting a scaler on one request makes that row its mean, so x - mean = 0 for every feature,")
print("and its zero variance gets a scale of 1 (0/1 = 0, not NaN): the model sees all zeros and")
print("predicts one class for all. A production system persists and REUSES the")
print("exact training-time transform: a notebook fits it inline and loses it. That is the gap.")

print("\nAll lifecycle checks ran.")

# %% [markdown]
# ## 28.3 Experiment tracking and reproducibility

# %% [markdown]
# Without that lineage a good result is a lucky accident you cannot reproduce; with it, a run is fully specified by its data version, its code commit, and its seed.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def trained_score(seed):
    data = load_breast_cancer()                             # real 569 x 30 diagnostic data
    Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.33,
                                          random_state=seed, stratify=data.target)
    m = DecisionTreeClassifier(max_depth=4, random_state=seed).fit(Xtr, ytr)
    return m.score(Xte, yte)

print("same fixed seed twice ->", f"{trained_score(42):.4f}", "and", f"{trained_score(42):.4f}", "(identical -> reproducible)")
print("two different seeds   ->", f"{trained_score(1):.4f}", "and", f"{trained_score(2):.4f}", "(differ -> why you PIN the seed)")

data = load_breast_cancer()
Xtr, Xva, ytr, yva = train_test_split(data.data, data.target, test_size=0.33,
                                      random_state=0, stratify=data.target)

runs = []                                                   # the "tracking store" (MLflow uses a DB + files)
def log_run(params, metrics):                               # one row per experiment, like mlflow.log_*
    runs.append({**params, **metrics})

for depth in [2, 3, 4, 5, 6, 8, None]:                      # a hyperparameter sweep (None = no depth cap)
    m = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, ytr)
    log_run({"max_depth": depth},
            {"train_acc": round(m.score(Xtr, ytr), 3), "val_acc": round(m.score(Xva, yva), 3),
             "fitted_depth": m.get_depth(), "leaves": m.get_n_leaves()})   # the tree it actually grew

print("\ntracked runs (each row is one logged experiment):")
print(f"  {'max_depth':>10} | {'train_acc':>9} | {'val_acc':>7} | {'fitted_depth':>12} | {'leaves':>6}")
for r in runs:
    print(f"  {str(r['max_depth']):>10} | {r['train_acc']:>9} | {r['val_acc']:>7} | {r['fitted_depth']:>12} | {r['leaves']:>6}")
best_val = max(r["val_acc"] for r in runs)                  # query the store for the best score
tied = [r["max_depth"] for r in runs if r["val_acc"] == best_val]
simplest = min(tied, key=lambda d: float("inf") if d is None else d)   # ties go to the smallest cap; None is deepest
tie_str = ", ".join(str(d) for d in tied)
print(f"  best by val_acc: {best_val} at max_depth {tie_str} -> keep the simplest: {simplest}")
full = runs[-1]                                             # max_depth=None: the tree grown with no cap
same = ", ".join(str(r["max_depth"]) for r in runs
                 if (r["fitted_depth"], r["leaves"]) == (full["fitted_depth"], full["leaves"]))
print(f"  max_depth {same} all fit depth {full['fitted_depth']} with {full['leaves']} leaves, like the uncapped tree")

print("\npin the seed so a result repeats; log params+metrics so you can compare and reproduce the")
print("winner. MLflow / W&B add a UI, artifact and model storage, and history shared across a team.")

print("\nAll experiment-tracking checks ran.")

# %% [markdown]
# Example 28-4 logs the sweep of Example 28-3 with MLflow 3.18.0, installed here as the package mlflow-skinny, reads the runs back with `mlflow.search_runs`, and lets the store rank them.
#
# Example 28-4. The depth sweep of Example 28-3 tracked with MLflow and ranked by a query.

# %%
import warnings; warnings.filterwarnings("ignore")
import os, pathlib, tempfile
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"             # MLflow sends anonymous usage data by default: opt out
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"               # no import hint naming MLflow's install folder, a line
                                                            # MLflow logs when an AI coding assistant runs the code
os.environ["MLFLOW_LOGGING_LEVEL"] = "WARNING"              # MLflow's INFO lines start with the date and time
import mlflow                                               # mlflow reads all three settings when it is imported
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

store = tempfile.mkdtemp(prefix="ch28_mlflow_")             # a fresh folder outside the companion folder
mlflow.set_tracking_uri(f"sqlite:///{pathlib.Path(store, 'mlflow.db').as_posix()}")  # runs go in one SQLite file
experiment_id = mlflow.create_experiment(                   # artifacts go in the same folder, not in ./mlruns
    "ch28-depth-sweep", artifact_location=pathlib.Path(store, "artifacts").as_uri())
mlflow.set_experiment(experiment_id=experiment_id)

data = load_breast_cancer()                                 # the split and the sweep of Example 28-3
Xtr, Xva, ytr, yva = train_test_split(data.data, data.target, test_size=0.33,
                                      random_state=0, stratify=data.target)
sweep = [2, 3, 4, 5, 6, 8, None]
for depth in sweep:
    with mlflow.start_run(run_name=f"max_depth={depth}"):   # one tracked run per setting
        m = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, ytr)
        mlflow.log_param("max_depth", depth)                # stored as a string: "2", ..., "None"
        mlflow.log_metrics({"train_acc": round(m.score(Xtr, ytr), 3),
                            "val_acc": round(m.score(Xva, yva), 3),
                            "fitted_depth": m.get_depth(),  # how deep the tree actually grew
                            "leaves": m.get_n_leaves()})

runs = mlflow.search_runs(experiment_ids=[experiment_id])   # read the store back, as a pandas DataFrame
by_depth = {r["params.max_depth"]: r for _, r in runs.iterrows()}
print(f"runs read back from the MLflow store: {len(runs)}")
print(f"  {'max_depth':>9} | {'train_acc':>9} | {'val_acc':>7} | {'fitted depth':>12} | {'leaves':>6}")
for depth in sweep:                                         # print them in sweep order
    r = by_depth[str(depth)]
    print(f"  {str(depth):>9} | {r['metrics.train_acc']:>9} | {r['metrics.val_acc']:>7} | "
          f"{int(r['metrics.fitted_depth']):>12} | {int(r['metrics.leaves']):>6}")
best = mlflow.search_runs(experiment_ids=[experiment_id], max_results=1,       # let the store rank the runs:
                          order_by=["metrics.val_acc DESC", "metrics.leaves ASC"]).iloc[0]  # ties to fewer leaves
print(f"best run: max_depth={best['params.max_depth']}, val_acc {best['metrics.val_acc']}, "
      f"{int(best['metrics.leaves'])} leaves")

# %% [markdown]
# ## 28.4 Model serving

# %% [markdown]
# Figure 28-2 sketches the request path a served model follows and contrasts online real-time serving with scheduled batch scoring.

# %%
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from fastapi import FastAPI
from pydantic import BaseModel

data = load_breast_cancer()                                 # real 569 x 30 diagnostic data
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3,
                                      random_state=0, stratify=data.target)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr)

app = FastAPI()
class PredictRequest(BaseModel):                            # Pydantic validates the request shape for us
    features: list[float]

@app.post("/predict")
def predict(req: PredictRequest):
    x = np.array(req.features, dtype=float).reshape(1, -1)
    prob = float(model.predict_proba(x)[0, 1])
    return {"probability": round(prob, 3), "label": int(prob > 0.5)}

# The demo and the timing benchmark run only when this file is executed directly.
# A server that imports the module (uvicorn serving:app, as the Dockerfile does)
# gets the trained model and the app above without rerunning the benchmark.
if __name__ == "__main__":
    from fastapi.testclient import TestClient               # imported only for the demo; needs httpx2 or httpx
    client = TestClient(app)                                # in-process client; the same app a live server runs
    row = Xte[3]                                            # one real held-out tumor (30 features)
    good = client.post("/predict", json={"features": row.tolist()})
    bad = client.post("/predict", json={"features": "not a list"})
    print("REST endpoint /predict:")
    print(f"  valid request  -> HTTP {good.status_code}, body {good.json()}   (true label {int(yte[3])})")
    print(f"  invalid request -> HTTP {bad.status_code} (Pydantic rejected bad input automatically)")

    N = 5000
    batch = np.repeat(Xte[:1], N, axis=0)                   # 5000 copies of one real row, to time serving
    t0 = time.perf_counter()                                # serve one request at a time
    for r in batch:
        model.predict_proba(r.reshape(1, -1))
    one_at_a_time = time.perf_counter() - t0
    t0 = time.perf_counter()                                # serve all N as a single batch
    model.predict_proba(batch)
    batched = time.perf_counter() - t0

    print(f"\nserving {N} requests:")
    print(f"  one-at-a-time : {one_at_a_time*1000:6.1f} ms total -> {N/one_at_a_time:8.0f} req/s")
    print(f"  batched       : {batched*1000:6.1f} ms total -> {N/batched:8.0f} req/s")
    print(f"  batching is ~{one_at_a_time/batched:.0f}x higher throughput (vectorized), the cost being that a")
    print("  request must wait for the batch to fill: the latency/throughput tradeoff every server tunes.")
    print("\n(For a demo UI in minutes, Gradio or Streamlit wrap this same predict() in a web page;")
    print(" they are not run here, but the served function is identical.)")

    print("\nAll serving checks ran.")

# %% [markdown]
# Example 28-6 starts uvicorn on the companion file 04_serving.py as a second process and sends it Example 28-5's two requests over HTTP.
#
# Example 28-6. Example 28-5's app behind a live uvicorn server, called over real HTTP.

# %%
import warnings; warnings.filterwarnings("ignore")
import os, runpy, socket, subprocess, sys, time
import httpx
from fastapi.testclient import TestClient

serving = runpy.run_path("04_serving.py", run_name="serving")  # Example 28-5's file, loaded in this process
row = serving["Xte"][3].tolist()                               # the held-out tumor Example 28-5 sends
in_process = TestClient(serving["app"])                        # the answers to compare against
tc_good = in_process.post("/predict", json={"features": row})
tc_bad = in_process.post("/predict", json={"features": "not a list"})

with socket.socket() as s:                                     # ask the operating system for a free port
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
server = subprocess.Popen(                                     # a separate process: uvicorn serving the same file
    [sys.executable, "-m", "uvicorn", "04_serving:app", "--host", "127.0.0.1",
     "--port", str(port), "--log-level", "warning"],
    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),         # leave no __pycache__ folder beside the script
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
base = f"http://127.0.0.1:{port}"
try:
    deadline = time.monotonic() + 60                           # wait up to a minute: the import fits the model
    while True:
        if server.poll() is not None:                          # the server stopped: show why
            raise RuntimeError(server.stderr.read().decode())
        try:
            if httpx.get(base + "/openapi.json", timeout=1).status_code == 200:
                break                                          # the app answers, so the server is up
        except httpx.HTTPError:
            pass                                               # nothing is listening on the port yet
        if time.monotonic() > deadline:
            raise RuntimeError("the server did not answer within a minute")
        time.sleep(0.25)
    live_good = httpx.post(base + "/predict", json={"features": row}, timeout=10)   # real HTTP over a socket
    live_bad = httpx.post(base + "/predict", json={"features": "not a list"}, timeout=10)
finally:
    server.terminate()                                         # stop the server, even if a request failed
    server.wait(timeout=30)
    server.stderr.close()

print("live server (uvicorn on 127.0.0.1), real HTTP requests:")
print(f"  valid request   -> HTTP {live_good.status_code}, body {live_good.json()}   "
      f"(true label {int(serving['yte'][3])})")
print(f"  invalid request -> HTTP {live_bad.status_code}")
print("  same status and body as the in-process TestClient:",
      (live_good.status_code, live_good.json()) == (tc_good.status_code, tc_good.json()),
      (live_bad.status_code, live_bad.json()) == (tc_bad.status_code, tc_bad.json()))

# %% [markdown]
# ## 28.4.1 An asynchronous handler

# %% [markdown]
# Example 28-7. An asynchronous FastAPI handler that scores on a worker thread, and two routes that count overlapping waits.

# %%
import warnings; warnings.filterwarnings("ignore")
import asyncio
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

data = load_breast_cancer()                                 # same 569 x 30 split as Example 28-5
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3,
                                      random_state=0, stratify=data.target)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr)

app = FastAPI()

class PredictRequest(BaseModel):
    features: list[float]

async def feature_lookup(n_features: int) -> int:
    await asyncio.sleep(0)                                  # yields; a real call would wait on the network
    return n_features

def score_row(features: list[float]) -> float:
    x = np.array(features, dtype=float).reshape(1, -1)     # CPU bound: this function never awaits
    return float(model.predict_proba(x)[0, 1])

@app.post("/predict_async")
async def predict_async(req: PredictRequest):
    n_seen = await feature_lookup(len(req.features))       # the await is the part async helps with
    prob = await asyncio.to_thread(score_row, req.features)  # score on a worker thread, not the loop
    return {"probability": round(prob, 3), "label": int(prob > 0.5),
            "features_seen": n_seen}

@app.get("/waits/{mode}")
async def waits(mode: str):
    state = {"in_flight": 0, "peak": 0}

    async def one_wait():
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        await asyncio.sleep(0.02)                           # long enough that four waits can overlap
        state["in_flight"] -= 1

    if mode == "gather":
        await asyncio.gather(*(one_wait() for _ in range(4)))
    else:
        for _ in range(4):
            await one_wait()
    return {"mode": mode, "peak_in_flight": state["peak"]}

client = TestClient(app)                                   # in-process client; same handler path as a live server
row = Xte[3]                                                # the same held-out tumor as Example 28-5
good = client.post("/predict_async", json={"features": row.tolist()})
overlapped = client.get("/waits/gather")
serial = client.get("/waits/serial")
print("async endpoint /predict_async:")
print(f"  valid request -> HTTP {good.status_code}, body {good.json()}   (true label {int(yte[3])})")
print(f"  four waits, gather -> {overlapped.json()}")
print(f"  four waits, serial -> {serial.json()}")

# %% [markdown]
# ## 28.5 Model export and portability

# %% [markdown]
# And because a runtime loads a file by name, exported artifacts should be versioned and named by model and version, for example model_v3.onnx, so a serving runtime always loads a known, tested build rather than whatever file happened to be in the folder.

# %%
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, torch, torch.nn as nn
import onnxruntime as ort
torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

class Net(nn.Module):                                       # a tiny MLP standing in for any trained model
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
    def forward(self, x): return self.net(x)

model = Net().eval()
example = torch.randn(1, 4)
with torch.no_grad():
    reference = model(example).numpy()                      # the "ground truth" output to match
print("original PyTorch model output:", np.round(reference[0], 4))

ts_path = os.path.join(HERE, "model_ts.pt")
scripted = torch.jit.trace(model, example)                 # capture the graph by tracing one input
scripted.save(ts_path)                                     # a self-contained file: graph + weights
reloaded = torch.jit.load(ts_path)                         # load WITHOUT needing the Net class
with torch.no_grad():
    ts_out = reloaded(example).numpy()
print(f"\nTorchScript reloaded output: {np.round(ts_out[0], 4)}")
print(f"  max difference vs original: {np.abs(ts_out - reference).max():.2e}  (identical -> portable)")

onnx_path = os.path.join(HERE, "model.onnx")
torch.onnx.export(model, example, onnx_path, input_names=["x"], output_names=["logits"],
                  dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
                  dynamo=False)                             # legacy exporter (no onnxscript needed)
sess = ort.InferenceSession(onnx_path)                      # onnxruntime: no PyTorch needed at all
onnx_out = sess.run(None, {"x": example.numpy()})[0]
print(f"\nONNX (onnxruntime) output: {np.round(onnx_out[0], 4)}")
print(f"  max difference vs original: {np.abs(onnx_out - reference).max():.2e}  (matches across frameworks)")

print("\nexport decouples the model from its training code: TorchScript runs in libtorch (C++/mobile),")
print("ONNX runs in onnxruntime on servers, phones, and edge chips: same weights, same answers,")
print("deployable far beyond the Python process that trained it.")

print("\nAll export/portability checks ran.")

# %% [markdown]
# ## 28.6 Containerization basics: Docker for ML

# %% [markdown]
# `dockerfile` block from 28.6 Containerization basics: Docker for ML (shown for reference, not executed):
#
#     FROM python:3.13.18-slim
#     WORKDIR /app
#     COPY requirements.txt .
#     RUN pip install --no-cache-dir -r requirements.txt
#     COPY . .
#     RUN python 06_containerization.py
#     RUN cp 04_serving.py serving.py
#     EXPOSE 8000
#     CMD ["uvicorn", "serving:app", "--host", "0.0.0.0", "--port", "8000"]

# %% [markdown]
# That reproducibility, the environment traveling with the code, is what the orchestration tier described at the end of this section builds on.

# %%
import sys, platform, importlib, runpy

print("environment a container pins (so it is identical everywhere):")
print(f"  python   : {sys.version.split()[0]}")
print(f"  platform : {platform.system()} {platform.machine()}")
pkgs = ["numpy", "scikit-learn", "fastapi", "uvicorn", "pydantic"]
pins = []
for name in pkgs:
    mod = importlib.import_module("sklearn" if name == "scikit-learn" else name)
    ver = getattr(mod, "__version__", "?")
    pins.append(f"{name}=={ver}")
    print(f"  {name:13s}: {ver}")
print("  pinned requirements (baked into the image):")
print("    " + "  ".join(pins))

def smoke_test():                                           # build gate: real serving code, one known tumor
    ns = runpy.run_path("04_serving.py", run_name="serving")  # as a server imports it; skips the demo
    out = ns["predict"](ns["PredictRequest"](features=ns["Xte"][3].tolist()))   # held-out tumor Xte[3]
    assert out["label"] == int(ns["yte"][3]), out          # a wrong label fails the build
    return out, int(ns["yte"][3])

print("\ncontainer smoke test (run at build time; an import error or a wrong label stops the build):")
out, truth = smoke_test()
print(f"  load 04_serving.py -> predict(held-out Xte[3]) -> {out}")
print(f"  true label {truth} -> test passed")
print("\na container image = your code + its pinned environment, built once and run the same on a")
print("laptop, a CI runner, and a production server.")

print("\nAll containerization checks ran.")

# %% [markdown]
# ## 28.7 Cloud ML platforms and the build-versus-buy decision

# %% [markdown]
# Portable choices like the ONNX export of 28.5 and container-based serving are partial defenses, and the mere prospect of lock-in is itself a reason some teams keep a self-hosted path open even when a managed service looks cheaper today.

# %%
# Assumed round numbers for illustration, not vendor prices (USD). Plug in your own quotes.
API_PER_1K   = 1.00        # managed API: $1.00 per 1,000 calls
HOST_FIXED   = 1200.0      # self-host: instances + amortized ops/maintenance per month
HOST_PER_1K  = 0.10        # self-host: marginal compute per 1,000 calls

def api_cost(calls):  return API_PER_1K  * calls / 1000
def host_cost(calls): return HOST_FIXED + HOST_PER_1K * calls / 1000

print("monthly cost: BUY (managed API) vs BUILD (self-host)")
print(f"  {'calls/month':>12} | {'buy (API)':>10} | {'build (host)':>12} | cheaper")
for calls in [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]:
    b, h = api_cost(calls), host_cost(calls)
    print(f"  {calls:>12,} | {b:>10,.0f} | {h:>12,.0f} | {'BUY' if b < h else 'BUILD'}")

# crossover: API_PER_1K*u == HOST_FIXED + HOST_PER_1K*u, with u = calls/1000
u = HOST_FIXED / (API_PER_1K - HOST_PER_1K)
crossover = u * 1000
print(f"\n  crossover volume = {crossover:,.0f} calls/month")
print(f"  below it, BUY is cheaper (no fixed cost); above it, BUILD wins (marginal cost is far lower).")

print("\nbeyond the breakeven, weigh:")
for factor in ["control & customization (your data, your fine-tuning) -> favors BUILD",
               "time to market & no ML team needed -> favors BUY",
               "data privacy / residency requirements -> often forces BUILD",
               "latency & offline use -> favors BUILD; spiky/low volume -> favors BUY"]:
    print(f"  - {factor}")
print("\nstart by BUYing a managed API to ship fast; BUILD once volume, control, or privacy justify the")
print("fixed cost. The platforms (SageMaker/Azure ML/Vertex) are what you build ON when you do.")

print("\nAll build-vs-buy checks ran.")

# %% [markdown]
# ## 28.8 Monitoring models in production

# %% [markdown]
# Together these are the discipline of production machine learning: watch the inputs for data drift, watch the metric for concept drift, and prove any change is a real improvement through a shadowed or canaried A/B test before it sees full traffic.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer
rng = np.random.default_rng(0)                              # drives the SIMULATED concept drift and A/B draws

# Data drift on a REAL feature, breast cancer "mean radius": a random split of the 569 tumors gives a training
# reference window and a live window from the same distribution; a SIMULATED covariate shift moves the live window.
feat = load_breast_cancer(return_X_y=True)[0][:, 0]         # a real continuous feature
idx = np.random.default_rng(1).permutation(len(feat))       # own generator: leave the main stream intact
bc_ref = feat[idx[:300]]                                    # training reference window
bc_same = feat[idx[300:]]                                   # live window, same distribution
bc_shift = bc_same * 1.10 + 2.0                             # a SIMULATED covariate shift on the real feature

def psi_bins(reference, current, bins=10):
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))   # bin by reference quantiles
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference) + 1e-6
    c = np.histogram(current, edges)[0] / len(current) + 1e-6
    return r, c, (c - r) * np.log(c / r)                    # each bin's two fractions and its PSI term
def psi(reference, current, bins=10):
    return float(np.sum(psi_bins(reference, current, bins)[2]))
def band(value):                                            # rule-of-thumb bands (Yurdakul and Naranjo, 2020)
    if value < 0.10:
        return "little change"
    if value <= 0.25:
        return "moderate change"
    return "significant change, action required"

print("data drift (PSI vs training reference, on real 'mean radius'):")
for name, live in [("same-dist live window", bc_same), ("shifted live window", bc_shift)]:
    value = psi(bc_ref, live)
    print(f"  {name:21s} : PSI = {value:.3f}  -> {band(value)}")
ref_frac, live_frac, bin_terms = psi_bins(bc_ref, bc_shift)
for b in (8, 1):                                            # the two bins the text works by hand
    print(f"    shifted window, bin {b}: reference {ref_frac[b - 1]:.4f}, live {live_frac[b - 1]:.4f}, "
          f"term {bin_terms[b - 1]:.4f}")

# KS test: the continuous-feature counterpart to PSI, on the same real windows.
# It reports a distance D and a p-value; a small p means the live window differs from the reference.
ks_same = stats.ks_2samp(bc_ref, bc_same)
ks_shift = stats.ks_2samp(bc_ref, bc_shift)
print("KS test (continuous-feature counterpart to PSI, on real 'mean radius'):")
print(f"  reference vs same-dist live : D={ks_same.statistic:.3f}, p={ks_same.pvalue:.3f}  -> no drift")
print(f"  reference vs shifted live   : D={ks_shift.statistic:.3f}, p={ks_shift.pvalue:.3f}  -> drift")

# SIMULATED concept drift, one mechanic: the inputs are draws from one fixed normal distribution, the model
# predicts y = (x > 0), and the true decision boundary is moved on purpose from 0 to 1.0 at week 3
# (the X->y relationship changed). Inputs are unchanged; only the live metric reveals the problem.
print("\nconcept drift (live accuracy per time window; alarm under 0.70):")
for week, boundary in [(1, 0.0), (2, 0.0), (3, 1.0), (4, 1.0)]:
    x = rng.normal(0, 1, 4000)
    noise = rng.random(len(x)) < 0.08                       # flip 8% of the labels: noisy, never perfect
    y_true = ((x > boundary).astype(int) ^ noise).astype(int)
    y_pred = (x > 0).astype(int)                            # the deployed model never changed
    acc = (y_pred == y_true).mean()
    print(f"  week {week}: accuracy {acc:.2f}" + ("   <- ALARM: concept drift" if acc < 0.70 else ""))

# SIMULATED A/B test: each user's conversion is a random draw at an ASSUMED rate, not an observation.
nA = nB = 8000
convA = rng.random(nA) < 0.118                              # model A: assumed 11.8% conversion
convB = rng.random(nB) < 0.132                              # model B: assumed 13.2% conversion
xA, xB = int(convA.sum()), int(convB.sum())                 # conversions counted in each arm
pA, pB = xA / nA, xB / nB
pool = (xA + xB) / (nA + nB)                                # two-proportion z-test
se = np.sqrt(pool * (1 - pool) * (1 / nA + 1 / nB))
z = (pB - pA) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))
ship = z > 0 and p_value < 0.05                             # B must be better (z > 0), not merely different
print(f"\nA/B test (conversion): A {xA}/{nA} = {pA:.4f}  B {xB}/{nB} = {pB:.4f}  lift = {pB - pA:+.3f}")
print(f"  z = {z:.2f}, p-value = {p_value:.3f} -> "
      f"{'B is significantly better; roll it out' if ship else 'B is not significantly better; keep A'}")
print("\ndata drift watches the inputs, concept drift watches the metric, and an A/B test (with a")
print("shadow or canary rollout first) decides safely whether a new model actually beats the old one.")

print("\nAll monitoring checks ran.")

# %% [markdown]
# ## 28.9 CI/CD for ML; model versioning and registries

# %% [markdown]
# For any of this to be reproducible, the pipeline has to version not only the model but the data and code that produced it, so every registered version carries a recorded lineage back to its inputs; the experiment-tracking and reproducibility practices of 28.3 are what supply that lineage.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
X, y = load_breast_cancer(return_X_y=True)                  # 569 real samples, 30 features, binary
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)

GATE = 0.90                                                 # minimum acceptable validation accuracy
def ci_pipeline(model, label):
    checks = {}
    checks["code: output shape"] = model.predict(Xva[:5]).shape == (5,)        # a normal unit test
    checks["data: no NaNs in features"] = not np.isnan(Xtr).any()              # a data validation test
    acc = model.score(Xva, yva)
    checks[f"perf: val_acc >= {GATE}"] = acc >= GATE                           # the ML-specific gate
    passed = all(checks.values())
    print(f"  CI for {label} (val_acc={acc:.3f}):")
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"    => {'PROMOTE (deploy)' if passed else 'BLOCKED (do not deploy)'}")
    return passed, acc

def fresh():                                                # StandardScaler + LogisticRegression pipeline
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
good = fresh().fit(Xtr, ytr)                                # a healthy retrain on clean labels
shuffled = np.random.default_rng(0).permutation(ytr)       # a data bug: labels got scrambled
weak = fresh().fit(Xtr, shuffled)                          # learns noise -> must FAIL the gate
print("continuous integration runs on every candidate model:")
ci_pipeline(good, "candidate v2 (full retrain)")
ci_pipeline(weak, "candidate v3 (trained on corrupted labels)")

class Registry:
    def __init__(self): self.versions, self.production = {}, None
    def register(self, name, acc):                          # log a new version (like mlflow model registry)
        self.versions[name] = {"acc": acc, "stage": "Staging"}
        print(f"  registered {name} (acc={acc:.3f}, stage=Staging)")
    def promote(self, name):                                # move a version to Production
        if self.production: self.versions[self.production]["stage"] = "Archived"
        self.versions[name]["stage"] = "Production"; self.production = name
        print(f"  promoted {name} -> Production")

v1_model = fresh().fit(Xtr[:, :10], ytr)                    # an earlier model: the ten "mean" features only
reg = Registry()
print("\nmodel registry:")
reg.register("v1", v1_model.score(Xva[:, :10], yva)); reg.promote("v1")   # v1 is live
reg.register("v2", good.score(Xva, yva)); reg.promote("v2")               # v2 beats it -> promote
print(f"  current production: {reg.production} (acc={reg.versions[reg.production]['acc']:.3f})")
reg.promote("v1")                                           # v2 misbehaves in prod -> instant rollback
print(f"  after rollback, production: {reg.production} (acc={reg.versions[reg.production]['acc']:.3f})")
print("\nCI gates ship only models that beat a quality bar; the registry versions every model with its")
print("metrics and stage, so promoting an improvement, or rolling back a bad deploy, is one safe step.")

print("\nAll CI/CD + registry checks ran.")

# %% [markdown]
# Example 28-14 registers the two real candidates that clear Example 28-13's gate in MLflow's Model Registry, moves the alias champion from version 1 to version 2 and back, and loads the champion by its URI.
#
# Example 28-14. Two real versions in MLflow's Model Registry, promoted and rolled back with an alias.

# %%
import warnings; warnings.filterwarnings("ignore")
import os, pathlib, tempfile
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"             # the three settings of Example 28-4, set
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"               # again before the import so that this listing
os.environ["MLFLOW_LOGGING_LEVEL"] = "WARNING"              # also runs on its own
import mlflow
from mlflow import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

store = tempfile.mkdtemp(prefix="ch28_registry_")           # a store of its own, outside the companion folder
mlflow.set_tracking_uri(f"sqlite:///{pathlib.Path(store, 'mlflow.db').as_posix()}")
experiment_id = mlflow.create_experiment(                   # each logged model's files go in this folder
    "ch28-registry", artifact_location=pathlib.Path(store, "artifacts").as_uri())
mlflow.set_experiment(experiment_id=experiment_id)

X, y = load_breast_cancer(return_X_y=True)                  # the split of Example 28-13
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)
name = "breast-cancer-classifier"
candidates = [("ten mean features", list(range(10))),       # the two candidates that pass Example 28-13's gate
              ("all thirty features", list(range(30)))]
for label, cols in candidates:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr[:, cols], ytr)
    with mlflow.start_run(run_name=label):
        mlflow.log_metric("val_acc", round(model.score(Xva[:, cols], yva), 3))
        mlflow.sklearn.log_model(model, name="model", registered_model_name=name,  # log it and register a version
                                 serialization_format="pickle")  # the default format, skops, is not installed

client = MlflowClient()                                     # the registry side of the same store
versions = sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version))
print(f"registered model {name!r}:")
for v in versions:
    run = client.get_run(v.run_id)                          # each version points back to the run that logged it
    print(f"  version {v.version}: {run.info.run_name}, val_acc {run.data.metrics['val_acc']:.3f}")
client.set_registered_model_alias(name, "champion", "1")    # version 1 goes live
print("  champion -> version", client.get_model_version_by_alias(name, "champion").version)
client.set_registered_model_alias(name, "champion", "2")    # promote: move the alias, retrain nothing
print("  promoted: champion -> version", client.get_model_version_by_alias(name, "champion").version)
client.set_registered_model_alias(name, "champion", "1")    # roll back: point the alias at version 1 again
print("  rolled back: champion -> version", client.get_model_version_by_alias(name, "champion").version)
live = mlflow.sklearn.load_model(f"models:/{name}@champion")  # what a server loads: the version behind the alias
print(f"  model loaded by alias scores {live.score(Xva[:, :10], yva):.3f} on the ten mean features")
print("  stage field of version 1:", repr(versions[0].current_stage))

# %% [markdown]
# ## 28.9.1 A data contract on the served row

# %% [markdown]
# Example 28-15. A data contract that checks the length and finiteness of a served row.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()                                 # breast cancer Wisconsin, 569 x 30
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3,
                                      random_state=0, stratify=data.target)
n_features = Xtr.shape[1]                                   # the contract's column count

def contract(row, n_features):
    x = np.asarray(row, dtype=float).reshape(-1)
    if x.shape != (n_features,):                            # a short or long request never reaches the model
        return "reject", "wrong length"
    if not np.all(np.isfinite(x)):                          # NaN and infinity are not scores
        return "reject", "non-finite value"
    return "accept", "ok"

good = Xte[0]                                               # one real held-out tumor
short = good[: n_features - 1]
poison = good.copy()
poison[0] = np.nan
print("data contract on a served breast-cancer row:")
print(f"  training columns: {n_features}")
for name, row in [("held-out row", good), ("29 features", short), ("NaN in feature 0", poison)]:
    decision, why = contract(row, n_features)
    print(f"  {name}: {decision} ({why})")

# %% [markdown]
# ## 28.10 Scaling inference

# %% [markdown]
# Example 28-16 measures response caching, the KV cache speedup, and the memory quantization saves.

# %%
import warnings; warnings.filterwarnings("ignore")
import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2", dtype=torch.float32).eval()

cache = {}
def generate(prompt, n=30):
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)
def cached_generate(prompt):
    if prompt in cache: return cache[prompt], True          # cache HIT -> instant
    result = generate(prompt); cache[prompt] = result
    return result, False                                    # cache MISS -> computed

prompt = "The future of machine learning is"
t0 = time.perf_counter(); _, hit1 = cached_generate(prompt); miss_t = time.perf_counter() - t0
t0 = time.perf_counter(); _, hit2 = cached_generate(prompt); hit_t = time.perf_counter() - t0
print("response caching (same prompt twice):")
print(f"  1st call (miss, computed): {miss_t*1000:7.1f} ms")
print(f"  2nd call (hit, cached)   : {hit_t*1000:7.3f} ms  -> ~{miss_t/max(hit_t,1e-9):.0f}x faster, no model work")

enc = tok("In a world where machines can think,", return_tensors="pt")
def time_gen(use_cache, n=40):
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=n, do_sample=False, use_cache=use_cache,
                       pad_token_id=tok.eos_token_id)
    return time.perf_counter() - t0

with_kv = time_gen(True)
without_kv = time_gen(False)
print(f"\nKV-cache, generating 40 tokens:")
print(f"  use_cache=True  (reuse past K/V): {with_kv*1000:7.1f} ms")
print(f"  use_cache=False (recompute all) : {without_kv*1000:7.1f} ms  -> {without_kv/with_kv:.1f}x slower without it")

params = 7_000_000_000                                      # a 7B-parameter model
print("\nmemory to hold a 7B model's weights:")
for name, bits in [("fp32", 32), ("fp16", 16), ("int8", 8), ("int4", 4)]:
    gb = params * bits / 8 / 1e9
    print(f"  {name:4s} ({bits:2d}-bit): {gb:5.1f} GB")
print("  int4 is 1/8 of fp32: the same weights in an eighth of the memory.")
print("\ncaching skips repeat work, the KV-cache removes redundant prefix reencoding (attention itself")
print("stays quadratic), and quantization shrinks the model: vLLM bundles these with")
print("continuous batching to serve LLMs at scale.")

print("\nAll scaling-inference checks ran.")

# %% [markdown]
# ## 28.10.1 Latency, throughput, and the batching tradeoff

# %% [markdown]
# A batch takes longer to compute than one input but far less time than its inputs would take one at a time, so batching raises throughput sharply; the cost is that a request must wait for the batch to fill and then wait for the whole batch to compute.

# %%
import time, numpy as np, torch, torch.nn as nn
torch.manual_seed(0)

# A small served model. We only time forward passes at different batch sizes, so the inputs are
# random and no dataset is needed: this measures serving mechanics, not a trained model's accuracy.
net = nn.Sequential(nn.Linear(256, 2048), nn.ReLU(), nn.Linear(2048, 2048),
                    nn.ReLU(), nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 1)).eval()

def time_batch(bs, reps=50):
    x = torch.randn(bs, 256)
    with torch.no_grad():
        net(x)                                              # warm up
        t0 = time.perf_counter()
        for _ in range(reps):
            net(x)
        dt = (time.perf_counter() - t0) / reps              # seconds per batch
    return dt * 1000, dt * 1000 / bs, bs / dt               # batch ms, per-request ms, throughput req/s

timings = {bs: time_batch(bs) for bs in [1, 8, 32, 64, 128]}   # measure each size ONCE, reuse below
print("batching tradeoff (one served model, timing forward passes):")
print(f"  {'batch':>5} {'batch ms':>9} {'per-req ms':>11} {'throughput req/s':>18}")
for bs, (batch_ms, per_req_ms, thru) in timings.items():
    print(f"  {bs:>5} {batch_ms:>9.2f} {per_req_ms:>11.4f} {thru:>18,.0f}")

BUDGET_MS, MAX_WAIT_MS = 50.0, 20.0                         # a latency budget and a cap on the batch-fill wait
print("\nlatency budget with a max-wait window (batch fills OR window expires):")
for bs in [32, 64, 128]:
    batch_ms = timings[bs][0]                               # the SAME measurement as the table above
    worst = MAX_WAIT_MS + batch_ms                          # a request pays queue wait + batch compute
    print(f"  batch<= {bs:>3}: worst latency = {MAX_WAIT_MS:.0f} + {batch_ms:.2f} = {worst:.2f} ms"
          f"  -> {'within' if worst <= BUDGET_MS else 'OVER'} {BUDGET_MS:.0f} ms budget")

COST_PER_REPLICA_HR = 1.20                                  # assumed example price of one replica for an hour
peak_rps, one_replica_rps = 900, 250                       # assumed peak demand and the load one replica serves
replicas = int(np.ceil(peak_rps / one_replica_rps))
print("\nhorizontal scaling: replicas behind a load balancer, cost per prediction:")
print(f"  peak {peak_rps} req/s at {one_replica_rps} req/s per replica -> {replicas} replicas at peak")
for rps in [900, 250, 0]:                                  # peak, quiet, idle
    r = int(np.ceil(rps / one_replica_rps))                # autoscale to load; 0 at idle = scale-to-zero
    hourly = r * COST_PER_REPLICA_HR
    cpp = (hourly / (rps * 3600)) if rps else 0.0
    tag = "scale-to-zero ($0/hr)" if r == 0 else f"{r} replica(s)"
    print(f"  load {rps:>3} req/s -> {tag:<21} ${hourly:>5.2f}/hr  cost/prediction ${cpp:.8f}")

# %% [markdown]
# Example 28-18 simulates the mechanics: eight requests arrive over time needing 45 tokens of generation in total, and the server has four slots that each decode one token per step.
#
# Example 28-18. Static and continuous batching, simulated on eight requests and four slots.

# %%
# Continuous batching, simulated. Each request generates one token per step; the server has
# B slots. STATIC batching loads up to B requests and runs the whole batch until its longest
# member finishes, leaving slots idle. CONTINUOUS batching admits and retires requests every step.
B = 4                                                       # server slots (concurrent sequences)
ARRIVALS = [0, 0, 0, 0, 2, 4, 6, 8]                         # step each request arrives
LENGTHS  = [12, 3, 3, 3, 6, 6, 6, 6]                        # tokens each request must generate

def simulate(mode):
    n = len(ARRIVALS)
    remaining, batch, queue = {}, set(), []                 # slot contents / static batch ids / waiting
    finish, seen = [None] * n, [False] * n
    t, busy, occupancy = 0, 0, []
    while any(f is None for f in finish):
        for i in range(n):                                  # arrivals join the queue at their step
            if not seen[i] and ARRIVALS[i] <= t:
                seen[i] = True; queue.append(i)
        if mode == "continuous" or not batch:               # static admits only between batches
            while queue and len(remaining) < B:
                i = queue.pop(0); remaining[i] = LENGTHS[i]; batch.add(i)
        busy += len(remaining); occupancy.append(len(remaining))
        for i in list(remaining):                           # one decode step per occupied slot
            remaining[i] -= 1
            if remaining[i] == 0:
                del remaining[i]; finish[i] = t + 1         # finished requests leave immediately
        if not remaining:
            batch.clear()                                   # static: next batch may load only now
        t += 1
    lat = [finish[i] - ARRIVALS[i] for i in range(n)]
    return t, sum(lat) / n, busy / (t * B), occupancy

print("continuous batching vs static batching (8 requests, 4 slots, 45 tokens of real work):")
for mode in ["static", "continuous"]:
    steps, mean_lat, util, occ = simulate(mode)
    print(f"  {mode:10s}: {steps} steps, mean latency {mean_lat:.1f} steps, "
          f"slot utilization {util:.0%}   busy slots per step: {''.join(map(str, occ))}")

# %% [markdown]
# ## 28.10.3 A stateless prediction function

# %% [markdown]
# Example 28-19 trains the same breast cancer pipeline as Example 28-5 and calls both handlers with FastAPI's TestClient. Row 3 is the held-out tumor that example already served.
#
# Example 28-19. A stateless handler and a leaky one, called with held-out tumors.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3,
                                      random_state=0, stratify=data.target)
n_features = Xtr.shape[1]
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr)
app = FastAPI()

class PredictRequest(BaseModel):
    features: list[float]

last_row = {"features": None}                               # process memory the leaky handler should not use

def score(features):
    x = np.asarray(features, dtype=float).reshape(1, -1)
    prob = float(model.predict_proba(x)[0, 1])
    return {"probability": round(prob, 3), "label": int(prob > 0.5)}

@app.post("/predict_stateless")
def predict_stateless(req: PredictRequest):
    if len(req.features) != n_features:                     # no previous row to fall back on
        return {"error": "wrong length"}
    return score(req.features)

@app.post("/predict_leaky")
def predict_leaky(req: PredictRequest):
    feats = req.features
    if len(feats) != n_features:
        feats = last_row["features"]                        # copies the previous caller
    else:
        last_row["features"] = list(feats)
    return score(feats)

client = TestClient(app)
row_a = Xte[0].tolist()
row_b = Xte[3].tolist()                                     # the held-out tumor from Example 28-5
a = client.post("/predict_stateless", json={"features": row_a})
b = client.post("/predict_stateless", json={"features": row_b})
short_ok = client.post("/predict_stateless", json={"features": row_a[:5]})
client.post("/predict_leaky", json={"features": row_a})
leaked = client.post("/predict_leaky", json={"features": row_a[:5]})
print("stateless handler, two different held-out tumors:")
print(f"  row 0 -> {a.json()}")
print(f"  row 3 -> {b.json()}")
print(f"  five features on stateless -> {short_ok.json()}")
print(f"  five features on leaky, after row 0 -> {leaked.json()}")
print(f"  leak repeats row 0: {leaked.json() == a.json()}")

# %% [markdown]
# ## 28.11 LLMOps

# %% [markdown]
# LLMOps carries the engineering rigor of the rest of this chapter over to the parts that are unique to language models: version the prompt like code, regression test every change on an eval set, and treat tokens and latency as budgets you measure and defend.

# %%
import warnings; warnings.filterwarnings("ignore")
import time
import torch
from datasets import load_dataset
from transformers import pipeline
llm = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct",
               device="cpu", dtype=torch.float32)             # the CPU and 32-bit floats, as in Example 25-1
def ask(system, text, n=10):
    return llm([{"role": "system", "content": system}, {"role": "user", "content": text}],
               max_new_tokens=n, do_sample=False)[0]["generated_text"][-1]["content"].strip().lower()

# Real data: the validation split of the Rotten Tomatoes reviews (Pang and Lee, 2005); label 1 = positive.
val = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
pos = [r["text"] for r in val if r["label"] == 1][:20]     # the first 20 positive reviews of the split
neg = [r["text"] for r in val if r["label"] == 0][:20]     # and its first 20 negative ones
EVAL = [(t, "positive") for t in pos] + [(t, "negative") for t in neg]
print(f"eval set: {len(EVAL)} of the {len(val):,} validation reviews, {len(pos)} positive and {len(neg)} negative")
PROMPTS = {
    "v1 (mis-specified)": "Rate this review from 1 to 5 stars. Reply with only the number.",
    "v2 (correct)":       "Classify the sentiment. Reply with exactly one word: positive or negative.",
}
def evaluate(system):                                       # the eval pipeline: accuracy on the labeled set
    correct, replies = 0, set()
    for text, label in EVAL:
        out = ask(system, text)
        replies.add(out)                                    # keep each distinct reply: what the parser had to read
        pred = "positive" if "positive" in out or "good" in out else "negative"
        correct += (pred == label)
    return correct / len(EVAL), sorted(replies)

print("prompt eval pipeline (accuracy on a fixed labeled set):")
scores = {}
for name, sys_prompt in PROMPTS.items():
    scores[name], replies = evaluate(sys_prompt)
    print(f"  {name:18s}: accuracy {scores[name]:.2f}, distinct replies {replies}")
best = max(scores, key=scores.get)
print(f"  -> ship {best} (versioned; the losing prompt is kept in history, not deleted)")

PRICE_PER_1M_TOK = 0.50                                     # assumed example API price: $0.50 per 1M tokens
monthly_calls, BUDGET = 2_000_000, 75.0                     # assumed monthly traffic and budget
tok = llm.tokenizer                                         # count tokens with the model's own tokenizer
chat = [{"role": "system", "content": PROMPTS[best]}, {"role": "user", "content": EVAL[0][0]}]
prompt_tok = len(tok.apply_chat_template(chat, add_generation_prompt=True)["input_ids"])
t0 = time.perf_counter(); reply = ask(PROMPTS[best], EVAL[0][0]); latency = time.perf_counter() - t0
reply_tok = len(tok(reply, add_special_tokens=False)["input_ids"])
tokens_per_call = prompt_tok + reply_tok                    # one real call: the formatted prompt plus the reply
cost_per_call = tokens_per_call / 1_000_000 * PRICE_PER_1M_TOK
monthly_cost = cost_per_call * monthly_calls
print(f"\ncost & latency budget:")
print(f"  measured latency/call : {latency*1000:.0f} ms")
print(f"  tokens/call           : {prompt_tok} prompt + {reply_tok} reply = {tokens_per_call}")
print(f"  cost/call             : ${cost_per_call:.7f} at the assumed ${PRICE_PER_1M_TOK:.2f} per 1M tokens")
print(f"  projected monthly cost at {monthly_calls:,} calls: ${monthly_cost:,.2f}  (budget ${BUDGET:.0f})")
print(f"  -> {'WITHIN budget' if monthly_cost <= BUDGET else 'OVER budget: shorten prompts / cache / smaller model'}")
print("\nin LLMOps the prompt is versioned code, every change is regression-tested on an eval set, and")
print("token cost and latency are budgeted line items: the same engineering rigor as any other system.")

print("\nAll LLMOps checks ran.")

# %% [markdown]
# ## 28.11.1 The wider LLMOps surface: guardrails, retrieval, evaluation, and caching

# %% [markdown]
# The piece that is genuinely new to LLM serving is the semantic cache, and Example 28-21 builds it, matching requests by cosine similarity of TF-IDF vectors, which stand in here for the dense sentence embeddings a real system would use.

# %%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Semantic cache: match by embedding similarity, not exact string. TF-IDF stands in for a dense
# sentence embedding so the demo is fast and reproducible; a paraphrase should still hit.
corpus = ["how do I reset my password", "reset password steps forgot login",
          "what is the refund policy", "refund money back returns policy",
          "track my shipping order delivery status",
          "cancel my subscription account plan billing"]
vec = TfidfVectorizer(stop_words="english").fit(corpus)    # drop stopwords so content words drive similarity

class SemanticCache:
    def __init__(self, threshold=0.6):
        self.threshold, self.keys, self.vals, self.mats = threshold, [], [], None
    def get(self, query):
        if not self.keys: return None, 0.0
        sims = cosine_similarity(vec.transform([query]), self.mats)[0]
        i = int(np.argmax(sims))
        return (self.vals[i], sims[i]) if sims[i] >= self.threshold else (None, sims[i])
    def put(self, query, answer):
        self.keys.append(query); self.vals.append(answer)
        self.mats = vec.transform(self.keys)

cache = SemanticCache(threshold=0.6)
cache.put("how do I reset my password", "Go to Settings > Security > Reset password.")
cache.put("what is the refund policy", "Refunds within 30 days with a receipt.")

print("\nsemantic cache (cache by embedding similarity, threshold 0.60):")
for q in ["how do I reset my password",                    # exact -> hit
          "forgot my login, need to reset password",       # paraphrase -> semantic hit
          "what's your refund and returns policy",         # paraphrase -> semantic hit
          "where is my shipping order"]:                   # unrelated to cached answers -> miss
    ans, sim = cache.get(q)
    exact = q in cache.keys
    print(f"  sim={sim:.2f} exact-cache={'hit' if exact else 'miss':4} "
          f"semantic-cache={'HIT ' if ans else 'miss'} <- {q}")

# %% [markdown]
# ## 28.12 Hands-on lab: deploy a model as a monitored API

# %% [markdown]
# Example 28-22 is the lab, a FastAPI service that serves predictions and reports its own health, driven with normal then shifted traffic.

# %%
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
rng = np.random.default_rng(0)

# Real data: the Breast Cancer Wisconsin set (569 samples, 30 features, benign vs malignant).
data = load_breast_cancer()
X, y = data.data, data.target                              # y: 1 = benign, 0 = malignant
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
scaler = StandardScaler().fit(Xtr)
model = LogisticRegression(max_iter=5000).fit(scaler.transform(Xtr), ytr)
F0 = 0                                                     # feature 0 = "mean radius", a strong predictor
REF = Xtr[:, F0].copy()                                    # training reference for feature 0 (drift baseline)
print(f"trained on {len(Xtr)} real samples, {X.shape[1]} features; "
      f"test accuracy {model.score(scaler.transform(Xte), yte):.3f}")

def psi(reference, current, bins=10):                       # Population Stability Index (from 28.8)
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference) + 1e-6
    c = np.histogram(current, edges)[0] / len(current) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))

app = FastAPI()
MON = {"latencies": [], "preds": [], "feat0": []}           # the monitoring store

class Req(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(req: Req):
    t0 = time.perf_counter()
    x = scaler.transform(np.array(req.features, dtype=float).reshape(1, -1))
    label = int(model.predict(x)[0])
    MON["latencies"].append(time.perf_counter() - t0)       # record latency...
    MON["preds"].append(label)                              # ...prediction...
    MON["feat0"].append(req.features[F0])                   # ...and the input feature for drift
    return {"label": label}

W = 400                                                     # rolling window: report over the most recent requests
@app.get("/metrics")                                        # what an ops dashboard would scrape
def metrics():
    drift = psi(REF, np.array(MON["feat0"][-W:])) if MON["feat0"] else 0.0
    return {"requests": len(MON["preds"]),
            "avg_latency_ms": round(1000 * np.mean(MON["latencies"][-W:]), 3),
            "positive_rate": round(float(np.mean(MON["preds"][-W:])), 3),
            "feature0_psi": round(drift, 3),
            "drift_alarm": drift > 0.25}

client = TestClient(app)
pool = Xte.copy()                                          # draw serving traffic from held-out real rows

def send_batch(n=400, shift=0.0):                          # one wave of requests, optionally shifted
    idx = rng.integers(0, len(pool), size=n)               # sample real rows with replacement
    for i in idx:
        row = pool[i].astype(float).copy()
        row[F0] += shift                                   # simulated drift: push feature 0 (mean radius)
        client.post("/predict", json={"features": row.tolist()})

SHIFT = 8 * REF.std()                                      # simulated drift: an 8-sigma push on mean radius
send_batch(shift=0.0)                                      # healthy traffic, same distribution as training
print("after NORMAL traffic:")
print("  /metrics ->", client.get("/metrics").json())
send_batch(shift=SHIFT)                                    # simulated shift: the world moves under the model
print("after SHIFTED traffic (simulated drift on feature 0):")
print("  /metrics ->", client.get("/metrics").json())
print("\none service: it trains on real data, serves predictions over HTTP, and reports its own health -")
print("latency, prediction mix, and data drift: so the drift alarm fires the moment inputs move.")

print("\nAll deploy-and-monitor lab checks ran.")
# %% [markdown]
# Example 28-23 serves the lab's numbers in the Prometheus text format from an app and a registry of its own, under the request class `PromRequest` and names that begin with `prom_`, so the lab's service, which the lab's latency alarm task reads, stays as Example 28-22 left it.
#
# Example 28-23. The lab's metrics served in the Prometheus text format from a private registry.

# %%
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# The lab's data, model, and traffic again, under names of its own, so the lab's service is left as it is.
prom_rng = np.random.default_rng(0)                         # a generator of its own: the lab's two waves again
prom_X, prom_y = load_breast_cancer(return_X_y=True)
prom_Xtr, prom_Xte, prom_ytr, prom_yte = train_test_split(prom_X, prom_y, test_size=0.2,
                                                          random_state=0, stratify=prom_y)
prom_scaler = StandardScaler().fit(prom_Xtr)
prom_model = LogisticRegression(max_iter=5000).fit(prom_scaler.transform(prom_Xtr), prom_ytr)
prom_ref = prom_Xtr[:, 0].copy()                            # training reference for feature 0, mean radius

def prom_psi(reference, current, bins=10):                  # the PSI of 28.8 and of the lab
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference) + 1e-6
    c = np.histogram(current, edges)[0] / len(current) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))

prom_registry = CollectorRegistry()                         # a private registry, so a rerun starts from zero
prom_requests = Counter("predict_requests", "Predictions served, by predicted label", ["label"],
                        registry=prom_registry)             # exposed as predict_requests_total
prom_latency = Histogram("predict_latency_seconds", "Handler latency of /predict", registry=prom_registry,
                         buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.05))
prom_drift = Gauge("feature0_psi", "PSI of the last 400 feature 0 values against training",
                   registry=prom_registry)
prom_feat0 = []                                             # the inputs the drift gauge is computed from
prom_app = FastAPI()

class PromRequest(BaseModel):
    features: list[float]

@prom_app.post("/predict")
def prom_predict(req: PromRequest):
    t0 = time.perf_counter()
    x = prom_scaler.transform(np.array(req.features, dtype=float).reshape(1, -1))
    label = int(prom_model.predict(x)[0])
    prom_latency.observe(time.perf_counter() - t0)          # one latency observation per request
    prom_requests.labels(label=str(label)).inc()            # one more request under its predicted label
    prom_feat0.append(req.features[0])
    return {"label": label}

@prom_app.get("/metrics/prometheus")                        # the page a Prometheus server would scrape
def prom_metrics():
    prom_drift.set(prom_psi(prom_ref, np.array(prom_feat0[-400:])) if prom_feat0 else 0.0)
    return Response(generate_latest(prom_registry), media_type=CONTENT_TYPE_LATEST)

prom_client = TestClient(prom_app)
prom_pool = prom_Xte.copy()

def prom_wave(n=400, shift=0.0):                            # the lab's send_batch, sent to this app
    for i in prom_rng.integers(0, len(prom_pool), size=n):
        row = prom_pool[i].astype(float).copy()
        row[0] += shift
        prom_client.post("/predict", json={"features": row.tolist()})

def prom_show(title):                                       # print only the lines that repeat from run to run
    print(title)
    for line in prom_client.get("/metrics/prometheus").text.splitlines():
        if line.startswith("#") or "_bucket" in line or "_sum" in line or "_created" in line:
            continue                                        # comments, and the timing lines: buckets, sum, start time
        print("  " + line)

prom_wave(shift=0.0)
prom_show("after NORMAL traffic, /metrics/prometheus:")
prom_wave(shift=8 * prom_ref.std())                         # the lab's simulated push of 8 standard deviations
prom_show("after SHIFTED traffic, /metrics/prometheus:")

# %%
def latency_alarm(metrics, threshold_ms):
    ...
print(latency_alarm(client.get("/metrics").json(), 5.0))

# %% [markdown]
# ## 28.13 Interview questions and coding exercises

# %%
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
print(data.data.shape)

def lift_over_baseline(budget=60):
    ...
print(lift_over_baseline())

# %%
from sklearn.datasets import load_wine
data = load_wine()
print(data.data.shape)

def score_saved_scaler(row):
    ...
print(score_saved_scaler(data.data[0]))

# %%
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

def simplest_tied_depth(runs):
    ...
print(simplest_tied_depth([]))

# %%
from sklearn.datasets import load_breast_cancer
texture = load_breast_cancer().data[:, 1]
print(texture.shape)

def psi(reference, current, bins=10):
    ...
print(psi(texture[:50], texture[:50]))

# %%
windows = []
print(len(windows))

def weeks_under(accuracies, threshold=0.85):
    ...
print(weeks_under(windows))

# %%
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)
print(X.shape)

def passes_gate(accuracy, gate=0.90):
    ...
print(passes_gate(0.5))

# %%
import torch
example = torch.zeros(1, 6)
print(tuple(example.shape))

def max_export_gap(reference, reloaded):
    ...
print(max_export_gap(example, example))

# %%
from sklearn.datasets import load_wine
data = load_wine()
print(data.data.shape[1])

def predict(features):
    ...
print(predict(data.data[0].tolist()))

# %%
print(13_000_000_000)

def memory_gb(n_params, bits):
    ...
print(memory_gb(13_000_000_000, 32))

# %%
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
print(data.data.shape)

def drift_alarm(psi, limit=0.25):
    ...
print(drift_alarm(0.0))

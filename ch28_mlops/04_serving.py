# %% [markdown]
# # Section 28.4 Model serving
#
# The code cell is Example 28-5 of Section 28.4: a FastAPI /predict endpoint for a StandardScaler and
# LogisticRegression pipeline trained on the breast cancer data bundled with scikit-learn, exercised with
# TestClient and timed one request at a time and as one batch. A server that imports the module, as the
# container of Section 28.6 does, gets the model and the app without running the demo.

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

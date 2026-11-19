# %%
"""Example 30-14 plus the lab's offline test harness. Train and freeze the lab
model (Example 30-14), then import the serving app from app.py (Example 30-15)
and exercise it in-process with TestClient: a warm-up call first, then a timed
call. Offline; no cloud account involved.
"""
import os
import time
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# ---- 1. Train and package the model once (Example 30-14) ---------------------
data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=5000)).fit(X_tr, y_tr)
joblib.dump(model, os.path.join(HERE, "model.joblib"))
print(f"trained pipeline, holdout accuracy = {model.score(X_te, y_te):.4f}")
print(f"expects {X.shape[1]} features; saved to model.joblib")

# ---- 2. Import the serving app (Example 30-15: app.py, which only serves) ----
from app import app

# ---- 3. Exercise it in-process, no server needed ----------------------------
from fastapi.testclient import TestClient

client = TestClient(app)
sample = X_te[0].tolist()          # a real holdout tumor from Example 30-14's split

# Warm-up: the FIRST TestClient request pays one-time startup costs (it imports
# the async plumbing behind TestClient) that say nothing about the model. Pay
# them outside the timed block.
client.post("/predict", json={"values": sample})

t0 = time.perf_counter()
resp = client.post("/predict", json={"values": sample})
latency_ms = (time.perf_counter() - t0) * 1000
print(f"health check: {client.get('/health').json()}")
print(f"POST /predict -> {resp.json()}")
print(f"true label for this sample: {int(y_te[0])}")
print(f"warm in-process latency: {latency_ms:.1f} ms")

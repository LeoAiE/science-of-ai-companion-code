"""Appendix H solutions for Chapter 30, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 30.11 coding exercises

# %%
import os, tempfile, fnmatch, time
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
import boto3

TMP = tempfile.gettempdir()
X, y = load_breast_cancer(return_X_y=True)   # 569 tumors, 30 features

# %% [markdown]
# 1. Package a model once with joblib (round-trip test)

# %%
# 1. Package a model once with joblib (round-trip test)
def package_once(path):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000)).fit(X, y)
    joblib.dump(model, path)
    return model, joblib.load(path)
path = os.path.join(TMP, "bc_model.joblib")
model, reloaded = package_once(path)
print(round(model.score(X, y), 4))                            # expected output: 0.9877
print(os.path.getsize(path))                                  # expected output: 2305
print(np.array_equal(model.predict(X), reloaded.predict(X)))  # expected output: True

# %% [markdown]
# 2. Wrap the artifact in a FastAPI /predict app, exercised with TestClient

# %%
# 2. Wrap the artifact in a FastAPI /predict app, exercised with TestClient
class Tumor(BaseModel):
    features: list[float]
def build_app():
    served = joblib.load(path)
    app = FastAPI()
    @app.get("/health")
    def health():
        return {"status": "ok"}
    @app.post("/predict")
    def predict(t: Tumor):
        p = served.predict_proba([t.features])[0]     # [P(malignant), P(benign)]
        return {"malignant_probability": round(float(p[0]), 4),
                "label": "malignant" if p[0] >= 0.5 else "benign"}
    return app
client = TestClient(build_app())
print(client.get("/health").json())               # expected output: {'status': 'ok'}
r = client.post("/predict", json={"features": X[0].tolist()})
print(r.status_code, r.json())
# expected output: 200 {'malignant_probability': 1.0, 'label': 'malignant'}

# %% [markdown]
# 3. Storage-cost calculator across S3 classes

# %%
# 3. Storage-cost calculator across S3 classes
PRICE_PER_GB_MONTH = {"Standard": 0.023, "Standard-IA": 0.0125, "Glacier Deep Archive": 0.00099}
PRICE_PER_GB_RETRIEVAL = {"Standard": 0.0, "Standard-IA": 0.01, "Glacier Deep Archive": 0.02}
PRICE_PER_GB_EGRESS = 0.09                         # the rates of Example 30-8
def monthly_cost(footprint_gb, gb_read_back, gb_egress):
    return {cls: round(footprint_gb * PRICE_PER_GB_MONTH[cls]
                       + gb_read_back * PRICE_PER_GB_RETRIEVAL[cls]
                       + gb_egress * PRICE_PER_GB_EGRESS, 2)
            for cls in PRICE_PER_GB_MONTH}
totals = monthly_cost(100, 400, 10)                # 100 GB stored, 400 GB read back, 10 GB out
print(totals)
# expected output: {'Standard': 3.2, 'Standard-IA': 6.15, 'Glacier Deep Archive': 9.0}
print(min(totals, key=totals.get))                # expected output: Standard
# reading 400 GB back costs nothing in Standard, $4.00 in Standard-IA and $8.00 in Deep Archive

# %% [markdown]
# 4. Monthly bill under three pricing models

# %%
# 4. Monthly bill under three pricing models
ON_DEMAND_HOURLY = 0.0835   # Linux t3.large on demand, US East (N. Virginia), T3 page, as of November 2026
RESERVED_HOURLY = 0.05      # the same page's 1-year Partial Upfront effective rate
HOURS = 730.0
SPOT_DISCOUNT = 0.70        # an assumption, as in Example 30-10, not a quoted rate
# storage and egress at US East list prices, AWS Price List, November 2026; same for all
FIXED = 50.0 * 0.023 + 100.0 * 0.09
def monthly_bills():
    return {
        "on-demand": round(ON_DEMAND_HOURLY * HOURS + FIXED, 2),
        "reserved": round(RESERVED_HOURLY * HOURS + FIXED, 2),
        "spot": round(ON_DEMAND_HOURLY * (1 - SPOT_DISCOUNT) * HOURS + FIXED, 2),
    }
bills = monthly_bills()
print(bills)
# expected output: {'on-demand': 71.11, 'reserved': 46.65, 'spot': 28.44}
print(round(100 * (1 - bills["reserved"] / bills["on-demand"]), 1))   # expected output: 34.4

# %% [markdown]
# 5. Break-even: always-on endpoint vs per-request serverless

# %%
# 5. Break-even: always-on endpoint vs per-request serverless
def break_even(fixed, per_million):
    return fixed / (per_million / 1_000_000)
always_on = 0.0418 * 730 + 50 * 0.023              # Example 30-10's compute + storage, unrounded
PER_MILLION = 0.80   # $0.80 per million requests: an assumed stand-in price, not a quoted rate
print(round(always_on, 2), f"{break_even(always_on, PER_MILLION):,.0f}")   # expected output: 31.66 39,580,000
for n in (1_000_000, 200_000_000):
    print(n, "serverless" if n / 1_000_000 * PER_MILLION < always_on else "always-on")
# expected output: 1000000 serverless
# expected output: 200000000 always-on

# %% [markdown]
# 6. Staged canary rollout simulation

# %%
# 6. Staged canary rollout simulation
def canary():
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    def fit(C):
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=C, max_iter=5000)).fit(X_tr, y_tr)
    incumbent = fit(1.0)       # Example 30-12's candidate, now in production
    candidate = fit(10.0)      # less regularized
    print(round(incumbent.score(X_te, y_te), 4), round(candidate.score(X_te, y_te), 4))
    MARGIN = 0.0
    order = np.random.default_rng(42).permutation(len(y_te))   # Example 30-12's shuffle seed
    for frac in [0.05, 0.25, 0.50, 1.00]:
        k = round(frac * len(order))
        cand_rows, inc_rows = order[:k], order[k:]   # the candidate serves the first k shuffled rows
        acc_c = candidate.score(X_te[cand_rows], y_te[cand_rows])
        # the incumbent on the rows it still serves; at 100 percent, its full holdout accuracy
        acc_i = (incumbent.score(X_te[inc_rows], y_te[inc_rows]) if len(inc_rows)
                 else incumbent.score(X_te, y_te))
        ok = acc_c >= acc_i + MARGIN
        print(f"{frac:.0%} to the candidate ({k} rows): {acc_c:.4f} vs incumbent {acc_i:.4f}"
              f" -> {'advance' if ok else 'ROLLBACK'}")
        if not ok:
            return "ROLLBACK"
    return "PROMOTE"
print(canary())
# expected output: 0.9883 0.9825
# expected output: 5% to the candidate (9 rows): 1.0000 vs incumbent 0.9877 -> advance
# expected output: 25% to the candidate (43 rows): 1.0000 vs incumbent 0.9844 -> advance
# expected output: 50% to the candidate (86 rows): 0.9767 vs incumbent 0.9882 -> ROLLBACK
# expected output: ROLLBACK
# the first two slices held none of the candidate's three holdout errors, so the gate catches it at 50 percent

# %% [markdown]
# 7. boto3 offline introspection (no credentials, no network)

# %%
# 7. boto3 offline introspection (no credentials, no network)
def offline_report():
    print(boto3.__version__)
    print(len(boto3.Session().get_available_regions("ec2")))
    s3 = boto3.client("s3", region_name="eu-central-1")
    print(s3.meta.region_name, s3.meta.endpoint_url)
offline_report()
# expected output: 1.43.123
# expected output: 34
# expected output: eu-central-1 https://s3.eu-central-1.amazonaws.com

# %% [markdown]
# 8. IAM least-privilege policy evaluator

# %%
# 8. IAM least-privilege policy evaluator
policy = {
    "Statement": [
        {"Effect": "Allow", "Action": "s3:PutObject",
         "Resource": "arn:aws:s3:::my-models-bucket/*"},
        {"Effect": "Deny", "Action": "s3:PutObject",
         "Resource": "arn:aws:s3:::my-models-bucket/prod/*"},
    ]
}
def allowed(policy, action, resource):
    ok = False
    for st in policy["Statement"]:
        acts = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
        ress = st["Resource"] if isinstance(st["Resource"], list) else [st["Resource"]]
        # fnmatch matches shell-style wildcards, so "my-models-bucket/*" covers every key in the bucket
        if any(fnmatch.fnmatch(action, a) for a in acts) and \
           any(fnmatch.fnmatch(resource, r) for r in ress):
            if st["Effect"] == "Deny":
                return False            # an explicit Deny always wins
            ok = True
    return ok
print(allowed(policy, "s3:PutObject", "arn:aws:s3:::my-models-bucket/m.joblib"))       # expected output: True
print(allowed(policy, "s3:GetObject", "arn:aws:s3:::my-models-bucket/m.joblib"))       # expected output: False
print(allowed(policy, "s3:PutObject", "arn:aws:s3:::other/x"))                         # expected output: False
print(allowed(policy, "s3:PutObject", "arn:aws:s3:::my-models-bucket/prod/m.joblib"))  # expected output: False

# %% [markdown]
# 9. Cross-cloud Rosetta translator (rows of Table 30-3)

# %%
# 9. Cross-cloud Rosetta translator (rows of Table 30-3)
TABLE = {
    "managed container service": {"AWS": "ECS Express Mode", "GCP": "Cloud Run",     "Azure": "Container Apps"},
    "object storage":            {"AWS": "S3",               "GCP": "Cloud Storage", "Azure": "Blob Storage"},
    "managed ML endpoint":       {"AWS": "SageMaker AI",     "GCP": "Vertex AI",     "Azure": "Azure ML"},
}
def translate(service_name, to_cloud):
    for row in TABLE.values():
        if service_name.lower() in {v.lower() for v in row.values()}:
            return row[to_cloud]
    return None
print(translate("Cloud Run", "Azure"))              # expected output: Container Apps
print(translate("S3", "GCP"))                       # expected output: Cloud Storage

# %% [markdown]
# 10. Miniproject: train, package, serve, measure, gate

# %%
# 10. Miniproject: train, package, serve, measure, gate
class Feat(BaseModel):
    values: list[float]
def mini():
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    mp = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X_tr, y_tr)
    mp_path = os.path.join(TMP, "mp.joblib")
    joblib.dump(mp, mp_path)
    print(round(mp.score(X_te, y_te), 4), os.path.getsize(mp_path))
    srv = joblib.load(mp_path)
    mp_app = FastAPI()
    @mp_app.post("/predict")
    def p(f: Feat):
        pr = float(srv.predict_proba([f.values])[0][1])  # class 1 = benign
        return {"probability_benign": round(pr, 4), "label": int(pr >= 0.5)}
    mc = TestClient(mp_app)
    body = {"values": X_te[0].tolist()}
    mc.post("/predict", json=body)       # the first request pays TestClient's one-time startup costs
    t0 = time.perf_counter()
    resp = mc.post("/predict", json=body)
    ms = (time.perf_counter() - t0) * 1000
    print(resp.json(), int(y_te[0]))
    print(f"in-process latency: {ms:.1f} ms")
    weak = make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.01, max_iter=5000)).fit(X_tr, y_tr)
    MARGIN = 0.0
    print("PROMOTE" if mp.score(X_te, y_te) >= weak.score(X_te, y_te) + MARGIN else "ROLLBACK")
mini()
# expected output: 0.9883 2305
# expected output: {'probability_benign': 0.0321, 'label': 0} 0
# in-process latency: 1.9 ms
# (one run on the reference machine, the author's Mac with an Apple M1 Max, 10 cores, under
# Python 3.13.14; the time varies by machine and from run to run.)
# expected output: PROMOTE

# %% [markdown]
# ## 30.11 coding drills

# %%
import os, tempfile
import numpy as np, joblib, boto3
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

TMP = tempfile.gettempdir()
X, y = load_breast_cancer(return_X_y=True)

# %% [markdown]
# Packaging with joblib

# %%
# Packaging with joblib
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000)).fit(X, y)
plain = os.path.join(TMP, "d.joblib"); joblib.dump(m, plain)
small = os.path.join(TMP, "d_c3.joblib"); joblib.dump(m, small, compress=3)
print(os.path.getsize(plain), os.path.getsize(small))              # expected output: 2305 1811
print(np.array_equal(m.predict(X), joblib.load(small).predict(X)))  # expected output: True

# boto3 offline
print(boto3.__version__)                             # expected output: 1.43.123
# the S3 region list is data bundled inside this boto3 build, so the count needs no account
print(len(boto3.Session().get_available_regions("s3")))   # expected output: 34
c = boto3.client("s3", region_name="ap-south-1")
print(c.meta.region_name, c.meta.endpoint_url)       # expected output: ap-south-1 https://s3.ap-south-1.amazonaws.com

# %% [markdown]
# Cost arithmetic

# %%
# Cost arithmetic
# t3.large on demand and at the 1-year rate, EC2 T3 Instances page, as of November 2026
print(round(0.0835 * 730, 2), round(0.05 * 730, 2))  # expected output: 60.96 36.5
print(round(250 * 0.09, 2))                          # 250 GB out at the US East list price  # expected output: 22.5
fixed = 0.0418 * 730 + 50 * 0.023                    # Example 30-10's fixed cost, unrounded (31.664)
# $0.80 per million requests: an assumed stand-in price, not a quoted rate
print(f"{fixed / (0.80 / 1_000_000):,.0f}")          # expected output: 39,580,000

# %% [markdown]
# Storage tiers

# %%
# Storage tiers
PRICE = {"Standard": 0.023, "Standard-IA": 0.0125, "Glacier Deep Archive": 0.00099}
print(min(PRICE, key=lambda c: 800 * PRICE[c]))     # expected output: Glacier Deep Archive
print(120 * 10 / 30)                                 # 120 GB held for 10 of 30 days  # expected output: 40.0
print("database")   # prediction records looked up by user id           # expected output: database
print("object")     # a nightly 300 GB snapshot that one batch job reads  # expected output: object

# %% [markdown]
# FastAPI serving

# %%
# FastAPI serving
srv = joblib.load(plain)
app = FastAPI()
class F(BaseModel):
    features: list[float]
@app.get("/health")
def h(): return {"status": "ok"}
@app.post("/predict")
def pr(f: F):
    p = srv.predict_proba([f.features])[0]
    return {"malignant_probability": round(float(p[0]), 4)}
cl = TestClient(app)
print(cl.get("/health").json())                      # expected output: {'status': 'ok'}
# a body without the "features" field fails FastAPI's request validation
print(cl.post("/predict", json={"values": X[1].tolist()}).status_code)   # expected output: 422
resp = cl.post("/predict", json={"features": X[1].tolist()})
print(resp.json()["malignant_probability"])          # expected output: 1.0

# %% [markdown]
# Rollout

# %%
# Rollout
def gate(cand, inc, margin=0.0): return cand >= inc + margin
print(gate(0.9883, 0.9415))                          # expected output: True
print(gate(0.9532, 0.9591))                          # expected output: False
traffic = {"blue": 75, "green": 25}
print(sum(traffic.values()) == 100)                  # expected output: True
REGISTRY = {"ECR": {"GCP": "Artifact Registry", "Azure": "Container Registry (ACR)"}}   # Table 30-3
print(REGISTRY["ECR"]["GCP"])                        # expected output: Artifact Registry

# %%
"""serve_local.py: runnable offline. A FastAPI app that loads model.joblib and
serves POST /predict. It is the serving layer on EC2 and on a managed container
service, while Lambda wraps the same model.joblib in a handler, lambda_function.py,
and SageMaker AI in an inference script with model_fn (30.4). We exercise it
in process with TestClient (no server, no network). Create model.joblib first with
Example 30-4 of 30.4.2 (00_chapter30_walkthrough.py) or with lab_build_serve.py."""
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_breast_cancer

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
model = joblib.load(os.path.join(HERE, "model.joblib"))

app = FastAPI(title="breast-cancer-classifier")

class Tumor(BaseModel):
    # 30 numeric features, the exact order load_breast_cancer produces.
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(t: Tumor):
    proba = model.predict_proba([t.features])[0]
    return {
        "malignant_probability": round(float(proba[0]), 4),
        "label": "malignant" if proba[0] >= 0.5 else "benign",
    }

if __name__ == "__main__":
    from fastapi.testclient import TestClient
    client = TestClient(app)

    print("GET /health ->", client.get("/health").json())

    data = load_breast_cancer()
    sample = data.data[0].tolist()   # first tumor, known malignant
    r = client.post("/predict", json={"features": sample})
    print("POST /predict ->", r.status_code, r.json())

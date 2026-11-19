"""Example 30-15. app.py: the serving app the container carries.

Loads the frozen model.joblib beside this file and exposes /health and /predict.
Importing this module never trains and never tests; it only serves, which is
exactly the property the container needs when uvicorn imports it as app:app.
The training step and the offline test harness live in lab_build_serve.py.
"""
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
served = joblib.load(os.path.join(HERE, "model.joblib"))

app = FastAPI(title="breast-cancer-classifier")


class Features(BaseModel):
    values: list[float]   # 30 numbers, one per feature


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(f: Features):
    # class 1 is "benign" in load_breast_cancer (target_names[1]).
    proba = round(float(served.predict_proba([f.values])[0][1]), 4)
    return {"probability_benign": proba, "label": int(proba >= 0.5)}

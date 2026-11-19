# %% [markdown]
# # 7.8.4 When deep learning does win: the TabPFN race
#
# Companion for Book.md section 7.8.4; the code cell is the book's Example 7-12, and its printed output is what the book reports.
# It runs in its own process, apart from the chapter walkthrough, because in the book's reference environment
# (macOS, PyTorch 2.15, XGBoost 3.4.1) an XGBoost fit in a Python process that had already imported PyTorch crashed.
# Needs the tabpfn package (tested with tabpfn 9.2.0); the TabPFN v2 classifier weights (Prior-Labs/TabPFN-v2-clf,
# a 29 MB checkpoint) download from the Hugging Face Hub on first use.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

def tabpfn_v2(**kw):                      # the Nature 2025 model: weights from Prior-Labs/TabPFN-v2-clf
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2, random_state=0, **kw)

Xb, yb = load_breast_cancer(return_X_y=True)                     # 569 tumors, 30 features
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)   # the same five folds for both models
for name, model in [("TabPFN v2", tabpfn_v2()), ("HistGB", HistGradientBoostingClassifier(random_state=0))]:
    r = cross_validate(model, Xb, yb, cv=cv, scoring=["roc_auc", "accuracy"])   # TabPFN's fit only stores rows
    print(name, round(r["test_roc_auc"].mean(), 4), round(r["test_accuracy"].mean(), 4),
          np.round(r["test_accuracy"], 3).tolist())

adult = fetch_openml("adult", version=2, as_frame=True)          # the split of 7.4.2: 7,500 training rows
df = adult.frame.sample(10000, random_state=0)
y = (df["class"] == ">50K").astype(int).to_numpy()
Xdf = df.drop(columns="class"); cat = Xdf.select_dtypes("category").columns
cat_idx = [Xdf.columns.get_loc(c) for c in cat]                  # tell TabPFN which columns are categories
Xdf[cat] = OrdinalEncoder(encoded_missing_value=-1).fit_transform(Xdf[cat])
X = Xdf.to_numpy(dtype=float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
pfn = tabpfn_v2(categorical_features_indices=cat_idx).fit(X_tr, y_tr)
hgb = HistGradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
for name, m in [("TabPFN v2", pfn), ("HistGB", hgb)]:
    print(name, round(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]), 4))
print(pfn.devices_[0].type, sum(p.numel() for p in pfn.models_[0].parameters()))

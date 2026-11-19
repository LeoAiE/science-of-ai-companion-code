# %% [markdown]
# # 7.9 Hands-on lab: a mini-Kaggle competition
# 
# Companion for Book.md section 7.9, regenerated from the book; each block is self-contained
# with the real Adult census data, and its printed output is what the book reports.
# The forest and booster fits cap their threads (n_jobs=2, and OMP_NUM_THREADS=2 for
# HistGradientBoosting) to avoid the oversubscription that section 7.9.3 describes.

# %% [markdown]
# ## 7.9.1 The data and the problem

# %%
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

adult = fetch_openml("adult", version=2, as_frame=True)   # UCI Adult census income
df = adult.frame
target = adult.target_names[0]                              # the column name is "class"
print(f"full data: {len(df)} rows, positive rate (>50K) = {(df[target] == '>50K').mean():.4f}")

# The full set is 48,842 rows; we take a fixed 10,000-row subsample so the whole
# lab (five models, cross-validation, stacking, SHAP) runs in under two minutes.
df = df.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df[target] == ">50K").astype(int).values
X = df.drop(columns=[target])

num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]

print(f"subsample: {len(X)} rows, {len(num)} numeric + {len(cat)} categorical features")
print(f"positive rate (>50K) in subsample = {y.mean():.4f}")
print(f"missing cells in subsample        = {int(X.isna().sum().sum())}")
print(f"columns with missing values       = {X.columns[X.isna().any()].tolist()}")

# %% [markdown]
# ## 7.9.2 A leakage-free preprocessor

# %%
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df["class"] == ">50K").astype(int).values
X = df.drop(columns=["class"])
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]

# Hold out 25% as the untouched leaderboard test set, stratified on the label.
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
])

Xtr = pre.fit_transform(X_tr)   # fit learns medians, means, and categories from TRAIN
Xte = pre.transform(X_te)       # test is transformed with the TRAIN statistics only

print(f"train matrix {Xtr.shape}, test matrix {Xte.shape}")
print(f"numeric train columns have mean ~0: {Xtr[:, :len(num)].mean():+.2e}")
print(f"same numeric columns on test are not recentered: {Xte[:, :len(num)].mean():+.3f}")

# For contrast only, never in practice: a leaky copy fit on ALL 10,000 rows, test rows included.
leaky = clone(pre).fit(X)   # clone copies the settings, not the fitted statistics
print(f"leaky fit on all rows, same test columns: {leaky.transform(X_te)[:, :len(num)].mean():+.3f}")

# %% [markdown]
# ## 7.9.3 A cross-validated leaderboard

# %%
import os
os.environ["OMP_NUM_THREADS"] = "2"   # cap the OpenMP threads the HistGB fit spawns
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df["class"] == ">50K").astype(int).values
X = df.drop(columns=["class"])
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
])

# Every candidate is a full pipeline: the preprocessor is refit inside each CV
# fold, so no validation fold ever leaks into the fit statistics.
models = {
    "LogisticRegression":   LogisticRegression(max_iter=2000),
    "RandomForest":         RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=0),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                           random_state=0),
    "XGBoost":              XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                          subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                                          eval_metric="logloss", random_state=0),
    "LightGBM":             LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                           subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                                           verbose=-1, random_state=0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print("5-fold CV ROC AUC on the training set (mean +/- std):")
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    s = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)
    print(f"  {name:22} {s.mean():.4f} +/- {s.std():.4f}")

# %% [markdown]
# ## 7.9.4 Letting the boosters size themselves

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df["class"] == ">50K").astype(int).values
X = df.drop(columns=["class"])
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
# carve a validation slice out of TRAIN so early stopping never sees the test set
X_fit, X_val, y_fit, y_val = train_test_split(
    X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=0)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
]).fit(X_fit)
Xf, Xv = pre.transform(X_fit), pre.transform(X_val)

# XGBoost: cap at 2000 trees but stop when validation AUC has not improved for 50 rounds
xgb = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=5,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                    eval_metric="auc", early_stopping_rounds=50, random_state=0)
xgb.fit(Xf, y_fit, eval_set=[(Xv, y_val)], verbose=False)
print(f"XGBoost stopped at {xgb.best_iteration + 1} trees (cap was 2000), "
      f"val AUC = {roc_auc_score(y_val, xgb.predict_proba(Xv)[:, 1]):.4f}")

# LightGBM: same idea, early stopping via a callback
lgbm = LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=31,
                      subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                      verbose=-1, random_state=0)
lgbm.fit(Xf, y_fit, eval_set=[(Xv, y_val)], eval_metric="auc",
         callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
print(f"LightGBM stopped at {lgbm.best_iteration_} trees (cap was 2000), "
      f"val AUC = {roc_auc_score(y_val, lgbm.predict_proba(Xv)[:, 1]):.4f}")

# %% [markdown]
# ## 7.9.5 Stacking the top models

# %%
import warnings; warnings.filterwarnings("ignore")
import os; os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df["class"] == ">50K").astype(int).values
X = df.drop(columns=["class"])
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
])

xgb  = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
                     colsample_bytree=0.8, n_jobs=2, eval_metric="logloss", random_state=0)
lgbm = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                      colsample_bytree=0.8, n_jobs=2, verbose=-1, random_state=0)
hgb  = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=0)

# n_jobs=1 on the stack: the base boosters already use n_jobs=2, and nesting
# parallelism inside parallelism oversubscribes the cores and makes fits crawl.
stack = StackingClassifier(
    estimators=[("xgb", xgb), ("lgbm", lgbm), ("hgb", hgb)],
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5, n_jobs=1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
pipe = Pipeline([("pre", pre), ("clf", stack)])
s = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)
print(f"Stacked ensemble  5-fold CV ROC AUC = {s.mean():.4f} +/- {s.std():.4f}")
print("(compare to the best single model, XGBoost at 0.9170 from the leaderboard)")

# %% [markdown]
# ## 7.9.6 The final held-out report

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df["class"] == ">50K").astype(int).values
X = df.drop(columns=["class"])
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
])

# Cross-validation chose XGBoost; refit it on ALL the training data, score once on test.
final = Pipeline([("pre", pre),
                  ("clf", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                        subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                                        eval_metric="logloss", random_state=0))])
final.fit(X_tr, y_tr)
proba = final.predict_proba(X_te)[:, 1]
pred = (proba >= 0.5).astype(int)
print(f"FINAL held-out test:  ROC AUC = {roc_auc_score(y_te, proba):.4f}")
print(f"                      accuracy = {accuracy_score(y_te, pred):.4f}")
print(f"majority-class accuracy (predict everyone <=50K) = {(y_te == 0).mean():.4f}")

# %% [markdown]
# ## 7.9.7 Interpreting the winner

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df["class"] == ">50K").astype(int).values
X = df.drop(columns=["class"])
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
])
final = Pipeline([("pre", pre),
                  ("clf", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                        subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                                        eval_metric="logloss", random_state=0))]).fit(X_tr, y_tr)

# Permutation importance on the RAW columns: shuffle one column of the test set,
# measure how much held-out ROC AUC drops. Model-agnostic and leakage-safe.
perm = permutation_importance(final, X_te, y_te, scoring="roc_auc",
                              n_repeats=10, random_state=0, n_jobs=2)
order = perm.importances_mean.argsort()[::-1]
print("Top permutation importances (drop in test ROC AUC when shuffled):")
for i in order[:6]:
    print(f"  {X.columns[i]:16} {perm.importances_mean[i]:.4f} +/- {perm.importances_std[i]:.4f}")

# SHAP: exact per-feature contributions from the tree structure, on transformed columns.
try:
    import shap
    Xte_t = final.named_steps["pre"].transform(X_te)
    names = final.named_steps["pre"].get_feature_names_out()
    expl = shap.TreeExplainer(final.named_steps["clf"])
    sv = expl.shap_values(Xte_t)
    mean_abs = np.abs(sv).mean(axis=0)
    top = mean_abs.argsort()[::-1][:6]
    print("\nTop SHAP features (mean |SHAP value| over the test set):")
    for i in top:
        print(f"  {names[i]:28} {mean_abs[i]:.4f}")
except ImportError:
    print("\n(shap not installed; permutation importance above is the interpretation)")

# %% [markdown]
# ## 7.9.8 Is it calibrated?

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df["class"] == ">50K").astype(int).values
X = df.drop(columns=["class"])
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
])
final = Pipeline([("pre", pre),
                  ("clf", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                        subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                                        eval_metric="logloss", random_state=0))]).fit(X_tr, y_tr)
proba = final.predict_proba(X_te)[:, 1]

# Brier score: mean squared error of the probabilities (0 is perfect).
print(f"Brier score on the held-out set = {brier_score_loss(y_te, proba):.4f}")
frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=5, strategy="quantile")
print("\nReliability (quantile bins): predicted vs. actual positive rate")
for p, a in zip(mean_pred, frac_pos):
    print(f"  predicted ~{p:.2f}  ->  actual {a:.2f}")

# %% [markdown]
# # 6.7 Hands-on lab: a classifier showdown
# 
# Companion for Book.md section 6.7, regenerated from the book; each block is self-contained
# with real (or justified synthetic) data and its printed output is what the book reports.

# %% [markdown]
# ## 6.7.2 The leaderboard

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)        # 569 tumors, 30 features, malignant=0 / benign=1
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=0)

# Scale models whose geometry or penalty depends on feature units.
# Plain LDA and the tree do not.
models = {
    "kNN (k=7)":     make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
    "GaussianNB":    make_pipeline(StandardScaler(), GaussianNB()),
    "LDA":           LinearDiscriminantAnalysis(),
    "QDA":           make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.1)),
    "Decision tree": DecisionTreeClassifier(max_depth=3, random_state=0),
    "Logistic reg.": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "SVM (linear)":  make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=0)),
    "SVM (RBF)":     make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=0)),
}

cv = StratifiedKFold(5, shuffle=True, random_state=0)   # the same folds for every model
print(f"{'model':16} CV acc (SD)    CV AUC (SD)")
auc_by_model = {}
for name, m in models.items():
    acc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="accuracy")   # one score per fold
    auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc")    # uses decision_function for SVC
    auc_by_model[name] = auc.mean()
    print(f"{name:16} {acc.mean():.3f} ({acc.std():.3f})  {auc.mean():.3f} ({auc.std():.3f})")
print("best by CV AUC:", max(auc_by_model, key=auc_by_model.get))

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
leaders = {
    "Logistic reg.": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "SVM (linear)":  make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=0)),
    "SVM (RBF)":     make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=0)),
}
for seed in range(1, 6):                             # deal the same rows into five new folds
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    auc = {name: cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
           for name, m in leaders.items()}
    print(seed, max(auc, key=auc.get), round(max(auc.values()) - min(auc.values()), 4))
# 1 Logistic reg. 0.0012 | 2 SVM (RBF) 0.0023 | 3 SVM (RBF) 0.003 | 4 SVM (RBF) 0.0001 | 5 Logistic reg. 0.0005

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import brier_score_loss
X, y = load_breast_cancer(return_X_y=True)
X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
proba = cross_val_predict(make_pipeline(StandardScaler(), GaussianNB()),
                          X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
low, high = proba < 0.1, proba >= 0.9
print(round(brier_score_loss(y_tr, proba), 3))                              # 0.065
print(int(low.sum()), round(proba[low].mean(), 3), round(y_tr[low].mean(), 3))    # 153  0.001  0.078
print(int(high.sum()), round(proba[high].mean(), 3), round(y_tr[high].mean(), 3)) # 264  0.999  0.947

# %% [markdown]
# ## 6.7.3 Judging the winner on held-out data

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=0)).fit(X_tr, y_tr)
print(confusion_matrix(y_te, svm.predict(X_te)))      # [[50  3]  [ 3 87]]  (class 0 = malignant)
print(round(svm.score(X_te, y_te), 2),                # 0.96 accuracy
      round(roc_auc_score(y_te, svm.decision_function(X_te)), 3))   # 0.993 ROC AUC

from sklearn.calibration import CalibratedClassifierCV
cal = make_pipeline(StandardScaler(),
                    CalibratedClassifierCV(SVC(kernel="rbf", random_state=0),
                                           ensemble=False)).fit(X_tr, y_tr)
print(round(roc_auc_score(y_te, cal.predict_proba(X_te)[:, 1]), 3))  # 0.993 from probabilities

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
X, y = load_breast_cancer(return_X_y=True)
X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)   # test rows unused
cv = StratifiedKFold(5, shuffle=True, random_state=0)               # the folds of Example 6-13
f = cross_val_predict(make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=0)),
                      X_tr, y_tr, cv=cv, method="decision_function")   # out-of-fold scores, benign above 0
malignant = y_tr == 0
print("   t  caught  missed  false alarms  cost")
for t in [0.0, 0.25, 0.5, 1.0, 1.5]:
    flag = f < t                                                      # flag as malignant below the threshold
    caught, missed = int((flag & malignant).sum()), int((~flag & malignant).sum())
    alarms = int((flag & ~malignant).sum())
    print(f"{t:4.2f}  {caught:6d}  {missed:6d}  {alarms:12d}  {5 * missed + alarms:4d}")   # 5 per miss, 1 per alarm

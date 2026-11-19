"""Appendix H solutions for Chapter 3, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 3.11 coding exercises

# %%
import warnings
warnings.filterwarnings("ignore")
import os, tempfile, sqlite3
import numpy as np
import pandas as pd
import seaborn as sns

tmp = tempfile.mkdtemp()

# %% [markdown]
# 1. Parquet keeps types, CSV loses them

# %%
# 1. Parquet keeps types, CSV loses them
df = pd.DataFrame({"zip": ["00123", "00045"], "spend": [120.5, 80.0], "orders": [3, 2]})
csv_path, pq_path = os.path.join(tmp, "t.csv"), os.path.join(tmp, "t.parquet")
df.to_csv(csv_path, index=False); df.to_parquet(pq_path)
from_csv, from_pq = pd.read_csv(csv_path), pd.read_parquet(pq_path)
print("CSV zip:", from_csv["zip"].iloc[0], "dtype:", from_csv["zip"].dtype)   # expected output: CSV zip: 123 dtype: int64
print("Parquet zip:", from_pq["zip"].iloc[0], "spend dtype:", from_pq["spend"].dtype)  # expected output: Parquet zip: 00123 spend dtype: float64
# Rule: accept CSV when handed it, but save your own intermediates as Parquet (smaller, faster, type-safe).

# %% [markdown]
# 2. SQL on a real table (Titanic in SQLite)

# %%
# 2. SQL on a real table (Titanic in SQLite)
titanic = sns.load_dataset("titanic")[["survived", "pclass", "sex", "age", "fare"]]
con = sqlite3.connect(":memory:")
titanic.to_sql("passengers", con, index=False, if_exists="replace")
q = lambda s: pd.read_sql_query(s, con)
having = q("""SELECT pclass, COUNT(*) AS n, ROUND(AVG(fare),2) AS avg_fare
              FROM passengers GROUP BY pclass HAVING AVG(fare) > 15 ORDER BY avg_fare DESC""")
print(having.to_dict("records"))
# expected output: [{'pclass': 1, 'n': 216, 'avg_fare': 84.15}, {'pclass': 2, 'n': 184, 'avg_fare': 20.66}]
win = q("""SELECT pclass, ROUND(AVG(fare),2) AS avg_fare,
                  RANK() OVER (ORDER BY AVG(fare) DESC) AS rk
           FROM passengers GROUP BY pclass ORDER BY rk""")
print(win.to_dict("records"))
# expected output: [{'pclass': 1, 'avg_fare': 84.15, 'rk': 1}, {'pclass': 2, 'avg_fare': 20.66, 'rk': 2}, {'pclass': 3, 'avg_fare': 13.68, 'rk': 3}]
print("pandas:", titanic.groupby("pclass")["fare"].mean().round(2).to_dict())
# expected output: pandas: {1: 84.15, 2: 20.66, 3: 13.68}

# %% [markdown]
# 3. Three ways to fill a gap, and the leak underneath

# %%
# 3. Three ways to fill a gap, and the leak underneath
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
tt = sns.load_dataset("titanic")
age = tt["age"]
print("missing:", int(age.isna().sum()), "| mean:", round(age.mean(), 2), "| median:", round(age.median(), 2))
# expected output: missing: 177 | mean: 29.7 | median: 28.0
# Skew pulls the mean up; the median is safer.
filled = KNNImputer(n_neighbors=5).fit_transform(tt[["age", "fare", "pclass"]])[:, 0]
print("KNN fill mean of missing:", round(float(filled[age.isna().values].mean()), 2))  # expected output: KNN fill mean of missing: 30.64
# For 121 of the 177 gaps the fifth-nearest distance is tied among more donors than there are slots, so
# which tied donors fill the five slots, and with them this mean, can differ across platforms and NumPy versions.
fare = tt["fare"].dropna()
tr, te = train_test_split(fare, test_size=0.3, random_state=0)
print("full-data median:", round(float(fare.median()), 2), "| train-only median:", round(float(tr.median()), 2))
# expected output: full-data median: 14.45 | train-only median: 15.0
# Use the train-only value.

# %% [markdown]
# 4. Outliers and the masking trap

# %%
# 4. Outliers and the masking trap
fare = sns.load_dataset("titanic")["fare"].dropna()
z = (fare - fare.mean()) / fare.std()
print("max |z|:", round(float(z.abs().max()), 2), "| z>3 flags:", int((z.abs() > 3).sum()))  # expected output: max |z|: 9.66 | z>3 flags: 20
q1, q3 = fare.quantile(0.25), fare.quantile(0.75)
low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
print(f"IQR bounds: [{low:.2f}, {high:.2f}] | IQR flags:", int(((fare < low) | (fare > high)).sum()))
# expected output: IQR bounds: [-26.72, 65.63] | IQR flags: 116
# The extreme fares inflate std and mask themselves.
capped = fare.clip(low, high)
print("max before:", round(float(fare.max()), 2), "| after winsorize:", round(float(capped.max()), 2))  # expected output: max before: 512.33 | after winsorize: 65.63

# %% [markdown]
# 5. Straightening a skew, and picking the right correlation

# %%
# 5. Straightening a skew, and picking the right correlation
from sklearn.preprocessing import PowerTransformer
from sklearn.datasets import fetch_california_housing
cal = fetch_california_housing(as_frame=True).frame
pop = cal["Population"]                                     # people per block group; the smallest value is 3
print("raw skew:", round(float(pop.skew()), 2), "| log skew:", round(float(np.log(pop).skew()), 2))
# expected output: raw skew: 4.94 | log skew: -1.07
for method in ["box-cox", "yeo-johnson"]:
    pt = PowerTransformer(method=method).fit(pop.to_frame())
    print(method, "lambda:", round(float(pt.lambdas_[0]), 3),
          "| skew:", round(float(pd.Series(pt.transform(pop.to_frame()).ravel()).skew()), 2))
# expected output: box-cox lambda: 0.236 | skew: 0.11
# expected output: yeo-johnson lambda: 0.234 | skew: 0.11
x, y = cal["AveRooms"], cal["MedHouseVal"]
print("pearson:", round(x.corr(y, method="pearson"), 3),
      "| spearman:", round(x.corr(y, method="spearman"), 3),
      "| kendall:", round(x.corr(y, method="kendall"), 3))
# expected output: pearson: 0.152 | spearman: 0.263 | kendall: 0.179
print("AveRooms median:", round(float(x.median()), 2), "| max:", round(float(x.max()), 1),
      "| above 20:", int((x > 20).sum()))
# expected output: AveRooms median: 5.23 | max: 141.9 | above 20: 69
keep = x <= x.quantile(0.99)                                # leave out the top 1 percent of AveRooms
print("pearson without the top 1 percent:", round(x[keep].corr(y[keep]), 3))
# expected output: pearson without the top 1 percent: 0.329
# Why the log overshoots: the log is Box-Cox at lambda = 0, a fixed and very strong compression, and on
# this column it turns a right skew of 4.94 into a left skew of -1.07. PowerTransformer estimates lambda
# by maximum likelihood, settles on the milder 0.236, and lands at a skew of 0.11. Because the column is
# strictly positive, Box-Cox and Yeo-Johnson are both allowed, and they pick nearly the same lambda and
# give the same skew. Make Yeo-Johnson the default anyway: Box-Cox needs strictly positive inputs and
# fails on a column with zeros or negatives, which Yeo-Johnson accepts. The rank-based correlations come
# out stronger because a few block groups have extreme room averages (69 above 20 rooms per household,
# the largest 141.9, against a median of 5.23). Pearson weighs every point by its distance from the mean,
# so those extremes, paired with ordinary house values, pull it toward zero; ranks give each extreme only
# its position, and leaving out the top 1 percent alone lifts the Pearson from 0.152 to 0.329.

# %% [markdown]
# 6. Redundant features by correlation matrix and VIF

# %%
# 6. Redundant features by correlation matrix and VIF
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
Xh = fetch_california_housing(as_frame=True).data           # the eight features, without the target
corr = Xh.corr()
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()   # each pair once
top = upper.reindex(upper.abs().sort_values(ascending=False).index).head(2)
print("top pairs:", {f"{a}-{b}": round(float(r), 2) for (a, b), r in top.items()})
# expected output: top pairs: {'Latitude-Longitude': -0.92, 'AveRooms-AveBedrms': 0.85}
def vif_table(X):
    Xc = sm.add_constant(X)                                 # VIF needs an intercept column
    return {c: round(float(variance_inflation_factor(Xc.values, j)), 1) for j, c in enumerate(X.columns, start=1)}
print("all eight:", vif_table(Xh))
# expected output: all eight: {'MedInc': 2.5, 'HouseAge': 1.2, 'AveRooms': 8.3, 'AveBedrms': 7.0, 'Population': 1.1, 'AveOccup': 1.0, 'Latitude': 9.3, 'Longitude': 9.0}
print("without Longitude:", vif_table(Xh.drop(columns=["Longitude"])))
# expected output: without Longitude: {'MedInc': 2.2, 'HouseAge': 1.1, 'AveRooms': 7.8, 'AveBedrms': 6.8, 'Population': 1.1, 'AveOccup': 1.0, 'Latitude': 1.1}
r = Xh["AveRooms"].corr(Xh["AveBedrms"])
print("pair-only VIF of AveRooms:", round(float(1 / (1 - r ** 2)), 2))
# expected output: pair-only VIF of AveRooms: 3.55
r2_rooms = lambda cols: round(float(sm.OLS(Xh["AveRooms"], sm.add_constant(Xh[cols])).fit().rsquared), 4)
print("R^2 of AveRooms on AveBedrms:", r2_rooms(["AveBedrms"]), "| on AveBedrms and MedInc:", r2_rooms(["AveBedrms", "MedInc"]))
# expected output: R^2 of AveRooms on AveBedrms: 0.7185 | on AveBedrms and MedInc: 0.863
# What the VIFs say: a VIF of 9.3 means the other seven features explain 1 - 1/9.3 = 89 percent of
# Latitude's variance, and nearly all of that comes from its partner, Longitude (r = -0.92): with
# Longitude gone, no remaining feature tracks Latitude and its VIF falls to 1.1. AveRooms and AveBedrms
# are redundant with each other, which dropping Longitude does not touch, so their VIFs barely move (7.8
# and 6.8); dropping or combining one of that pair is the remedy there. The pairwise correlation of 0.85
# alone would give AveRooms a VIF of 1 / (1 - r^2) = 3.55, yet its VIF is 8.3: the bedrooms column alone
# explains 0.7185 of AveRooms's variance, adding median income raises that to 0.863, and all seven other
# features reach 1 - 1/8.3 = 0.88. That is what VIF catches that a pairwise correlation misses, a feature
# that is close to a combination of several others when no single pair looks as alarming.

# %% [markdown]
# 7. Two filters and an embedded selector

# %%
# 7. Two filters and an embedded selector
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
bc = load_breast_cancer(as_frame=True)                      # 30 nonnegative measurements per tumor
raw_scores = pd.Series(SelectKBest(chi2, k=3).fit(bc.data, bc.target).scores_, index=bc.data.columns)
print("chi2 raw top3:", raw_scores.sort_values(ascending=False).round(1).head(3).to_dict())
# expected output: chi2 raw top3: {'worst area': 112598.4, 'mean area': 53991.7, 'area error': 8758.5}
X01 = pd.DataFrame(MinMaxScaler().fit_transform(bc.data), columns=bc.data.columns)
mm_scores = pd.Series(SelectKBest(chi2, k=3).fit(X01, bc.target).scores_, index=bc.data.columns)
print("chi2 min-max top3:", mm_scores.sort_values(ascending=False).round(1).head(3).to_dict())
# expected output: chi2 min-max top3: {'mean concave points': 52.4, 'worst concave points': 46.3, 'mean concavity': 46.2}
Xn = Xh.copy()
Xn["noise"] = np.random.default_rng(0).normal(size=len(Xn))   # a column with no relation to house value
yn = fetch_california_housing(as_frame=True).target
print("F ranks:", pd.Series(f_regression(Xn, yn)[0], index=Xn.columns).rank(ascending=False).astype(int).to_dict())
# expected output: F ranks: {'MedInc': 1, 'HouseAge': 4, 'AveRooms': 2, 'AveBedrms': 5, 'Population': 7, 'AveOccup': 8, 'Latitude': 3, 'Longitude': 6, 'noise': 9}
Zn = StandardScaler().fit_transform(Xn)
for alpha in [0.01, 0.05, 0.1]:
    coef = Lasso(alpha=alpha, max_iter=10000).fit(Zn, yn).coef_
    print(f"alpha {alpha}: zero:", [c for c, v in zip(Xn.columns, coef) if v == 0],
          "| kept:", {c: round(float(v), 3) for c, v in zip(Xn.columns, coef) if v != 0})
# expected output: alpha 0.01: zero: ['Population', 'noise'] | kept: {'MedInc': 0.777, 'HouseAge': 0.125, 'AveRooms': -0.129, 'AveBedrms': 0.169, 'AveOccup': -0.029, 'Latitude': -0.795, 'Longitude': -0.759}
# expected output: alpha 0.05: zero: ['AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'noise'] | kept: {'MedInc': 0.737, 'HouseAge': 0.139, 'Latitude': -0.26, 'Longitude': -0.217}
# expected output: alpha 0.1: zero: ['AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Longitude', 'noise'] | kept: {'MedInc': 0.706, 'HouseAge': 0.106, 'Latitude': -0.011}
r2 = lambda cols: round(LinearRegression().fit(Xh[cols], yn).score(Xh[cols], yn), 4)
print("R^2 MedInc:", r2(["MedInc"]), "| + AveRooms:", r2(["MedInc", "AveRooms"]),
      "| + Latitude:", r2(["MedInc", "Latitude"]), "| + Longitude:", r2(["MedInc", "Longitude"]),
      "| + both:", r2(["MedInc", "Latitude", "Longitude"]))
# expected output: R^2 MedInc: 0.4734 | + AveRooms: 0.4794 | + Latitude: 0.4815 | + Longitude: 0.4747 | + both: 0.5843
# Why the two chi-square lists share nothing: the raw scores follow magnitude, since multiplying a feature
# by c multiplies its score by c, and the three area columns hold the largest numbers in the table (worst
# area reaches 4254). After min-max scaling every feature runs from 0 to 1, the score reflects how each
# feature's total splits between the classes, and the concavity measures rise to the top. Trust the scaled
# list, though chi-square stays on firmest ground with counts and indicators (3.5.5).
# Why AveRooms falls early: the F-test scores each feature alone, and alone AveRooms carries a
# linear signal (rank 2), but most of it repeats median income: MedInc alone gives R^2 = 0.4734 and adding
# AveRooms lifts it only to 0.4794. The Lasso judges each feature by what it adds beside the others, so at
# alpha 0.05 it zeroes AveRooms and keeps Latitude and Longitude, which add little one at a time but
# together lift R^2 to 0.5843 by locating the block group. The noise column is exactly zero at every alpha.

# %% [markdown]
# 8. Simpson's paradox and the rate-versus-count trap

# %%
# 8. Simpson's paradox and the rate-versus-count trap
pen = sns.load_dataset("penguins").dropna(subset=["flipper_length_mm", "bill_depth_mm"])
print("birds:", len(pen), "| pooled corr:", round(pen["flipper_length_mm"].corr(pen["bill_depth_mm"]), 3))
# expected output: birds: 342 | pooled corr: -0.584
by = pen.groupby("species").apply(
    lambda g: round(g["flipper_length_mm"].corr(g["bill_depth_mm"]), 3), include_groups=False)
print("by species:", by.to_dict())
# expected output: by species: {'Adelie': 0.308, 'Chinstrap': 0.58, 'Gentoo': 0.707}
print(pen.groupby("species")[["flipper_length_mm", "bill_depth_mm"]].mean().round(1).to_dict("index"))
# expected output: {'Adelie': {'flipper_length_mm': 190.0, 'bill_depth_mm': 18.3}, 'Chinstrap': {'flipper_length_mm': 195.8, 'bill_depth_mm': 18.4}, 'Gentoo': {'flipper_length_mm': 217.2, 'bill_depth_mm': 15.0}}
# Species is a common cause of both measurements, a confounder: Gentoo birds have the longest flippers
# (217.2 mm on average) and the shallowest bills (15.0 mm), so pooling them with the other two species
# draws a falling line through three rising clouds. Act on the within-species sign.
tt2 = sns.load_dataset("titanic")
bp = tt2.groupby("embark_town")["survived"].agg(survivors="sum", passengers="count", rate="mean")
print("survivors:", bp["survivors"].to_dict(), "| passengers:", bp["passengers"].to_dict(),
      "| rate:", bp["rate"].round(3).to_dict())
# expected output: survivors: {'Cherbourg': 93, 'Queenstown': 30, 'Southampton': 217} | passengers: {'Cherbourg': 168, 'Queenstown': 77, 'Southampton': 644} | rate: {'Cherbourg': 0.554, 'Queenstown': 0.39, 'Southampton': 0.337}
# Southampton logs the most survivors only because 644 of the 889 passengers with a recorded port
# boarded there (the groupby leaves out the two with none). Dividing by the passengers who boarded at
# each port is the fix, and it shows that the Southampton passengers had the lowest survival rate.

# %% [markdown]
# 9. Making accuracy tell the truth on imbalanced data

# %%
# 9. Making accuracy tell the truth on imbalanced data
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter
df9 = sns.load_dataset("titanic").dropna(subset=["embark_town"])   # the 889 passengers with a recorded port
feats = ["pclass", "age", "fare", "sibsp", "parch"]
Xi, yi = df9[feats], (df9["embark_town"] == "Queenstown").astype(int)
Xtr, Xte, ytr, yte = train_test_split(Xi, yi, test_size=0.25, random_state=0, stratify=yi)
print("positive rate:", round(float(yi.mean()), 3), "| test rows:", len(yte), "| test positives:", int(yte.sum()),
      "| test majority share:", round(float(1 - yte.mean()), 3))
# expected output: positive rate: 0.087 | test rows: 223 | test positives: 19 | test majority share: 0.915
base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
pb = base.predict(Xte)
print("baseline: acc=%.3f rec=%.3f | positive predictions: %d" % (accuracy_score(yte, pb), recall_score(yte, pb), pb.sum()))
# expected output: baseline: acc=0.915 rec=0.000 | positive predictions: 0
wm = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                   LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xtr, ytr)
pw = wm.predict(Xte)
print("weighted: acc=%.3f prec=%.3f rec=%.3f" % (
    accuracy_score(yte, pw), precision_score(yte, pw, zero_division=0), recall_score(yte, pw)))
# expected output: weighted: acc=0.538 prec=0.144 rec=0.895
smote = ImbPipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("sm", SMOTE(random_state=0)), ("m", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
_, ysm = SMOTE(random_state=0).fit_resample(SimpleImputer(strategy="median").fit_transform(Xtr), ytr)
ps = smote.predict(Xte)
print("SMOTE balance:", dict(sorted(Counter(ysm).items())), "| acc=%.3f prec=%.3f rec=%.3f" % (
    accuracy_score(yte, ps), precision_score(yte, ps, zero_division=0), recall_score(yte, ps)))
# expected output: SMOTE balance: {0: 608, 1: 608} | acc=0.552 prec=0.148 rec=0.895
# The plain model never predicts Queenstown, so its accuracy is exactly the share of the test set that
# boarded elsewhere, 204 of 223 = 0.915, and its recall is zero. Both fixes find 17 of the 19 Queenstown
# passengers (recall 0.895) but pay with accuracy, which falls to 0.538 and 0.552, and with precision: only
# about one Queenstown prediction in seven is right (0.144 and 0.148). Resampling touches the training
# split only; the test set keeps its real balance, 19 Queenstown passengers among 223.

# %% [markdown]
# 10. The chapter's core workflow in one leakage-free pipeline (miniproject)

# %%
# 10. The chapter's core workflow in one leakage-free pipeline (miniproject)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score
raw = sns.load_dataset("penguins")
labeled = raw.dropna(subset=["sex"])                        # a bird with no recorded sex has no label
num = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
cat = ["species", "island"]
print("labeled birds:", len(labeled), "| missing values among them:", int(labeled[num + cat].isna().sum().sum()),
      "| males:", int((labeled["sex"] == "Male").sum()))
# expected output: labeled birds: 333 | missing values among them: 0 | males: 168
Xf, yf = labeled[num + cat], (labeled["sex"] == "Male").astype(int)
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
Xtr, Xte, ytr, yte = train_test_split(Xf, yf, test_size=0.25, stratify=yf, random_state=0)
model.fit(Xtr, ytr)
pred, proba = model.predict(Xte), model.predict_proba(Xte)[:, 1]
print("test: acc=%.3f f1=%.3f auc=%.3f" % (accuracy_score(yte, pred), f1_score(yte, pred), roc_auc_score(yte, proba)))
# expected output: test: acc=0.905 f1=0.905 auc=0.957
print("species per unshuffled fold:", [labeled["species"].iloc[te].nunique() for _, te in StratifiedKFold(5).split(Xf, yf)])
# expected output: species per unshuffled fold: [1, 1, 2, 2, 1]
folds = StratifiedKFold(5, shuffle=True, random_state=0)    # the table is sorted by species, so shuffle
cv = cross_val_score(model, Xf, yf, cv=folds, scoring="f1")
print("cv f1: %.3f +/- %.3f" % (cv.mean(), cv.std()))
# expected output: cv f1: 0.919 +/- 0.043
# Setting aside the 11 unlabeled birds leaks nothing: the rule looks only at whether a row has a label,
# learns no statistic from the other rows, and would remove the same rows whatever the split. The labeled
# birds happen to have no missing values, but the imputers stay so the fitted pipeline can still score a
# future bird with a gap. Every learned statistic (the medians, the scaler's mean and standard deviation,
# the encoder's categories, the model's coefficients) is fit inside the pipeline on the training rows, or
# under cross-validation on the training folds alone, so nothing leaks. Unshuffled folds would follow the
# table's species order, and three of the five would hold a single species, so each score would measure
# the model on one kind of bird rather than on the whole colony.

# %% [markdown]
# 11. Target encoding without the leak, checked three ways

# %%
# 11. Target encoding without the leak, checked three ways
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder
decks = sns.load_dataset("titanic").dropna(subset=["deck"]).reset_index(drop=True)
d11, y11 = decks["deck"].astype(str), decks["survived"]     # 203 passengers on seven decks
folds11 = KFold(5, shuffle=True, random_state=0)
naive11 = d11.map(y11.groupby(d11).mean())                  # each deck mean includes the row's own label
oof11 = pd.Series(index=decks.index, dtype=float)
for tr11, va11 in folds11.split(decks):                      # encode each fold from the other four only
    oof11.iloc[va11] = d11.iloc[va11].map(y11.iloc[tr11].groupby(d11.iloc[tr11]).mean()).to_numpy()
print("naive corr:", round(naive11.corr(y11), 3), "| out-of-fold corr:", round(oof11.corr(y11), 3),
      "| unencoded rows:", int(oof11.isna().sum()))
# expected output: naive corr: 0.203 | out-of-fold corr: 0.08 | unencoded rows: 0
# The two correlations are the ones Example 3-10 printed, which confirms the columns before the checks below.
n_a, s_a, m = int((d11 == "A").sum()), int(y11[d11 == "A"].sum()), 10
lam_a = n_a / (n_a + m)                                     # lambda_c = n_c / (n_c + m)
print(f"deck A: n={n_a} survivors={s_a} lambda={lam_a:.2f} "
      f"smoothed={lam_a * s_a / n_a + (1 - lam_a) * y11.mean():.4f}")
# expected output: deck A: n=15 survivors=7 lambda=0.60 smoothed=0.5480
X11 = decks[["deck"]].astype(str)
enc0 = TargetEncoder(smooth=0.0, cv=folds11)                # no smoothing, the same five folds
print("fit_transform == out-of-fold:", bool(np.allclose(enc0.fit_transform(X11, y11).ravel(), oof11)),
      "| fit then transform == naive:", bool(np.allclose(enc0.fit(X11, y11).transform(X11).ravel(), naive11)))
# expected output: fit_transform == out-of-fold: True | fit then transform == naive: True
enc10 = TargetEncoder(smooth=10.0).fit(X11, y11)            # encodings_ are learned on all 203 rows
print(f"TargetEncoder deck A: {enc10.encodings_[0][list(enc10.categories_[0]).index('A')]:.4f}")
# expected output: TargetEncoder deck A: 0.5480
# By hand, 0.6 x 7/15 + 0.4 x 136/203 = 0.2800 + 0.2680 = 0.5480. fit_transform cross-fits: it encodes
# each row with means learned without that row's fold. fit followed by transform encodes the training
# rows with means that contain their own labels, which is the naive leak; the scikit-learn TargetEncoder
# documentation warns that the two calls differ. Use fit_transform on the training rows and transform
# only on rows the encoder has not seen.

# %% [markdown]
# 12. Two classifiers as two annotators: Cohen's kappa by hand

# %%
# 12. Two classifiers as two annotators: Cohen's kappa by hand
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
Xb, yb = load_breast_cancer(return_X_y=True)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.25, stratify=yb, random_state=0)
ann_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xb_tr, yb_tr).predict(Xb_te)
ann_rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2).fit(Xb_tr, yb_tr).predict(Xb_te)
table12 = pd.crosstab(pd.Series(ann_lr, name="logreg"), pd.Series(ann_rf, name="forest")).to_numpy()
print(table12.tolist())        # rows: the logistic regression says 0, 1; columns: the forest says 0, 1
# expected output: [[49, 4], [4, 86]]
n12 = int(table12.sum())
p_o = np.trace(table12) / n12                                            # the diagonal: identical labels
p_e = float((table12.sum(axis=1) * table12.sum(axis=0)).sum()) / n12 ** 2   # chance, from the two margins
print(f"n={n12} p_o={p_o:.4f} p_e={p_e:.4f} kappa={(p_o - p_e) / (1 - p_e):.4f} "
      f"sklearn={cohen_kappa_score(ann_lr, ann_rf):.4f}")
# expected output: n=143 p_o=0.9441 p_e=0.5335 kappa=0.8801 sklearn=0.8801
print("accuracy: logreg %.3f | forest %.3f | agreed but both wrong: %d" % (
    (ann_lr == yb_te).mean(), (ann_rf == yb_te).mean(), ((ann_lr == ann_rf) & (ann_lr != yb_te)).sum()))
# expected output: accuracy: logreg 0.958 | forest 0.944 | agreed but both wrong: 3
# Each model labels 53 tumors malignant (class 0) and 90 benign, so a random pairing of those labels
# would agree p_e = (53 x 53 + 90 x 90) / 143^2 = 0.5335 of the time. Kappa keeps only the agreement
# beyond that, (0.9441 - 0.5335) / (1 - 0.5335) = 0.8801, which is why it sits below the raw 0.944.
# Agreement is not correctness: the two models agree on 135 tumors and are both wrong on 3 of them,
# which only the true labels reveal.

# %% [markdown]
# 13. Testing the missingness mechanism on the real age gaps

# %%
# 13. Testing the missingness mechanism on the real age gaps
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
t13 = sns.load_dataset("titanic")
t13["age_missing"] = t13["age"].isna().astype(int)          # 1 for the 177 passengers without an age
for col in ["pclass", "embarked"]:
    rate = t13.groupby(col)["age_missing"].mean().round(3).to_dict()
    p = chi2_contingency(pd.crosstab(t13[col], t13["age_missing"]), correction=False).pvalue
    print(col, rate, "| chi-square p = %.1e" % p)
# expected output: pclass {1: 0.139, 2: 0.06, 3: 0.277} | chi-square p = 9.9e-11
# expected output: embarked {'C': 0.226, 'Q': 0.636, 'S': 0.14} | chi-square p = 5.0e-24
num13, cat13 = ["pclass", "sibsp", "parch", "fare"], ["sex", "embarked"]
pre13 = ColumnTransformer([("num", StandardScaler(), num13),
                           ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                 OneHotEncoder(handle_unknown="ignore")), cat13)])
miss_model = make_pipeline(pre13, LogisticRegression(max_iter=1000))
cv13 = StratifiedKFold(5, shuffle=True, random_state=0)
auc13 = cross_val_score(miss_model, t13[num13 + cat13], t13["age_missing"], cv=cv13, scoring="roc_auc")
shuffled = np.random.default_rng(0).permutation(t13["age_missing"].to_numpy())   # random by construction
auc0 = cross_val_score(miss_model, t13[num13 + cat13], shuffled, cv=cv13, scoring="roc_auc")
print("AUC for age_missing: %.3f +/- %.3f | for a shuffled copy: %.3f" % (auc13.mean(), auc13.std(), auc0.mean()))
# expected output: AUC for age_missing: 0.714 +/- 0.047 | for a shuffled copy: 0.524
# Missing completely at random would give every passenger the same chance of a missing age, whatever
# else is recorded. Instead the rate runs from 0.06 in second class to 0.277 in third and from 0.14 at
# Southampton to 0.636 at Queenstown, the chi-square p values of 9.9e-11 and 5.0e-24 reject independence
# from class and from port, and the observed columns predict which ages are missing with an AUC of 0.714,
# where the shuffled copy, random by construction, scores 0.524. So that case is rejected. The pattern is
# what missing at random allows, gaps that depend on recorded columns, but it cannot rule out missing not
# at random: testing whether a gap depends on the unrecorded age itself would need the ages that were
# never recorded, and for every such model a missing-at-random model fits the observed data equally well
# (Molenberghs and colleagues, 2008, cited in 3.3.2). Scenarios: (a) completely at random, because the
# link drops are unrelated to everything; (b) at random, because the blanks depend on the recorded age
# and not on the weight; (c) not at random, because the score is missing on account of its own value.

# %% [markdown]
# 14. Nested cross-validation and the optimism of a tuned score

# %%
# 14. Nested cross-validation and the optimism of a tuned score
from sklearn.model_selection import GridSearchCV
Xn, yn = load_breast_cancer(return_X_y=True)
pipe14 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
grid14 = {"logisticregression__C": np.logspace(-3, 3, 13)}
inner14 = StratifiedKFold(5, shuffle=True, random_state=0)
outer14 = StratifiedKFold(5, shuffle=True, random_state=1)
search14 = GridSearchCV(pipe14, grid14, cv=inner14).fit(Xn, yn)
print("best C: %.3f | best_score_: %.4f" % (search14.best_params_["logisticregression__C"], search14.best_score_))
# expected output: best C: 0.316 | best_score_: 0.9807
near_top = {round(float(c), 3): round(float(s), 4) for c, s in
            zip(grid14["logisticregression__C"], search14.cv_results_["mean_test_score"]) if 0.09 < c < 3.2}
print("inner CV accuracy by C:", near_top)
# expected output: inner CV accuracy by C: {0.1: 0.9754, 0.316: 0.9807, 1.0: 0.9789, 3.162: 0.9701}
chosen, outer_scores = [], []
for tr14, te14 in outer14.split(Xn, yn):                    # the outer loop, written out by hand
    fold_search = GridSearchCV(pipe14, grid14, cv=inner14).fit(Xn[tr14], yn[tr14])   # tunes on 4/5
    chosen.append(round(float(fold_search.best_params_["logisticregression__C"]), 3))
    outer_scores.append(fold_search.score(Xn[te14], yn[te14]))                        # scored on 1/5
print("C per outer fold:", chosen, "| nested: %.4f +/- %.4f" % (np.mean(outer_scores), np.std(outer_scores)))
# expected output: C per outer fold: [0.316, 3.162, 0.1, 1.0, 1.0] | nested: 0.9737 +/- 0.0124
nested = cross_val_score(GridSearchCV(pipe14, grid14, cv=inner14), Xn, yn, cv=outer14)
print("cross_val_score: %.4f | optimism: %.4f" % (nested.mean(), search14.best_score_ - nested.mean()))
# expected output: cross_val_score: 0.9737 | optimism: 0.0070
# best_score_ is the highest of 13 cross-validated means, measured on the same five folds that chose the
# winner, so it carries the luck that made it win. Each outer fold reruns the whole search on four fifths
# of the rows and is scored on the fifth the search never saw, so the nested 0.9737 estimates the whole
# procedure, tuning included, on new data; the 0.0070 between the two is the optimism. The chosen C moves
# from fold to fold because neighboring values score almost alike: on all 569 rows the four winners,
# C = 0.1, 0.316, 1.0, and 3.162, lie within 0.0106 of one another, about six tumors, so a different
# four fifths of the rows can crown a different winner. Nested cross-validation therefore scores the
# procedure "tune C, then refit", not any single value of C.

# %% [markdown]
# ## 3.11 coding drills

# %%
import warnings
warnings.filterwarnings("ignore")
import os, tempfile, sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
tmp = tempfile.mkdtemp()

# %% [markdown]
# Files and stores

# %%
# Files and stores
d = pd.DataFrame({"a": [1], "b": [2.5]}); d.to_parquet(os.path.join(tmp, "d.parquet"))
print(pd.read_parquet(os.path.join(tmp, "d.parquet")).dtypes.astype(str).to_dict())  # expected output: {'a': 'int64', 'b': 'float64'}
docs = [{"_id": 1, "name": "Ana", "orders": [{"amt": 120}, {"amt": 80}]},
        {"_id": 2, "name": "Ben", "orders": [{"amt": 200}]}]
flat = pd.json_normalize(docs, record_path="orders", meta=["_id", "name"])
print(flat.shape, flat.columns.tolist())   # expected output: (3, 3) ['amt', '_id', 'name']
big = os.path.join(tmp, "big.csv")
pd.DataFrame({"v": np.random.default_rng(0).normal(100, 15, 50_000)}).to_csv(big, index=False)
total, cnt = 0.0, 0
for ch in pd.read_csv(big, chunksize=10_000):
    total += ch["v"].sum(); cnt += len(ch)
print(round(total / cnt, 4), round(pd.read_csv(big)["v"].mean(), 4))   # expected output: 100.0127 100.0127

# %% [markdown]
# SQL on the Titanic

# %%
# SQL on the Titanic
con = sqlite3.connect(":memory:")
sns.load_dataset("titanic")[["survived", "pclass", "sex", "age", "fare"]].to_sql("passengers", con, index=False)
q = lambda s: pd.read_sql_query(s, con)
print(q("SELECT sex, COUNT(*) AS n FROM passengers GROUP BY sex ORDER BY n DESC").to_dict("records"))
# expected output: [{'sex': 'male', 'n': 577}, {'sex': 'female', 'n': 314}]
print(q("SELECT ROUND(AVG(age),2) AS a FROM passengers WHERE pclass=1").to_dict("records"))  # expected output: [{'a': 38.23}]

# %% [markdown]
# Cleaning

# %%
# Cleaning
s = pd.Series(["NYC", "nyc ", " LA", "LA", "NYC"])
print(sorted(s.str.strip().str.upper().unique()))   # expected output: ['LA', 'NYC']
rd = pd.DataFrame({"x": [1, 1, 2], "y": [1, 1, 2]})
print(int(rd.duplicated().sum()), len(rd.drop_duplicates()))   # expected output: 1 2
dts = pd.to_datetime(pd.Series(["05/03/2021", "2021-03-05", "n/a"]),
                     format="mixed", dayfirst=True, errors="coerce")
print([str(x.date()) if pd.notna(x) else "NaT" for x in dts], int(dts.isna().sum()))
# expected output: ['2021-03-05', '2021-03-05', 'NaT'] 1
logs = pd.Series(['1.2.3.4 - - [t] "GET /home" 200 "Mozilla"', '5.6.7.8 - - [t] "GET /x" 404 "Googlebot"'])
ext = logs.str.extract(r'(?P<ip>\S+) - - \[[^\]]+\] "(?P<method>\S+) (?P<path>\S+)" (?P<status>\d+) "(?P<agent>[^"]*)"')
print(ext["path"].tolist(), int(ext["agent"].str.contains("bot", case=False).sum()))  # expected output: ['/home', '/x'] 1

# %% [markdown]
# Exploratory analysis

# %%
# Exploratory analysis
carat = sns.load_dataset("diamonds")["carat"]
print(round(float(carat.skew()), 2), round(float(np.log(carat).skew()), 2))   # expected output: 1.12 0.1
peng = sns.load_dataset("penguins").dropna()
print(round(peng["flipper_length_mm"].corr(peng["body_mass_g"]), 2))   # expected output: 0.87
print((peng["species"].value_counts(normalize=True) * 100).round(1).to_dict())
# expected output: {'Adelie': 43.8, 'Gentoo': 35.7, 'Chinstrap': 20.4}
fare = np.sort(sns.load_dataset("titanic")["fare"].dropna().to_numpy())
ecdf = np.arange(1, fare.size + 1) / fare.size
print(round(float(fare[np.searchsorted(ecdf, 0.50)]), 2))   # expected output: 14.45

# %% [markdown]
# Feature engineering

# %%
# Feature engineering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
emb = sns.load_dataset("titanic")["embarked"]
print(list(pd.get_dummies(emb, prefix="p", drop_first=True).columns))   # expected output: ['p_Q', 'p_S']
f = sns.load_dataset("titanic")[["fare"]].dropna()
print("%.2f %.2f" % (StandardScaler().fit_transform(f).mean(), StandardScaler().fit_transform(f).std()))  # expected output: 0.00 1.00
mm = MinMaxScaler().fit_transform(f); print("%.2f %.2f" % (mm.min(), mm.max()))   # expected output: 0.00 1.00
bins = pd.cut(sns.load_dataset("titanic")["age"], bins=[0, 12, 60, 100], labels=["child", "adult", "senior"])
print(bins.value_counts().to_dict())   # expected output: {'adult': 623, 'child': 69, 'senior': 22}
poly = PolynomialFeatures(degree=2, include_bias=False).fit(np.ones((1, 2)))
print(list(poly.get_feature_names_out(["age", "fare"])))   # expected output: ['age', 'fare', 'age^2', 'age fare', 'fare^2']
print(pd.to_datetime(pd.Series(["2010-05-01"])).dt.day_name().iloc[0])   # expected output: Saturday

# %% [markdown]
# Splits, imbalance, and scale

# %%
# Splits, imbalance, and scale
from sklearn.model_selection import train_test_split, TimeSeriesSplit
tt = sns.load_dataset("titanic")
_, _, _, ys = train_test_split(tt[["fare"]].fillna(0), tt["survived"], test_size=0.25, random_state=4, stratify=tt["survived"])
print(round(float(tt["survived"].mean()), 3), round(float(ys.mean()), 3))   # expected output: 0.384 0.386
tr0, te0 = next(iter(TimeSeriesSplit(n_splits=3).split(range(9))))
print(tr0.tolist(), te0.tolist())   # expected output: [0, 1, 2] [3, 4]
from collections import defaultdict
g = defaultdict(list)
for pc, fa in zip(tt["pclass"], tt["fare"]):
    g[pc].append((fa, 1))
red = {k: round(sum(f for f, c in v) / sum(c for f, c in v), 2) for k, v in g.items()}
print(dict(sorted(red.items())))   # expected output: {1: 84.15, 2: 20.66, 3: 13.68}

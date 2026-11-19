"""Chapter 4: Machine Learning Fundamentals

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from Book.md; keep the book as the source.
"""

# %% [markdown]
# ## 4.1.2 The three paradigms of learning

# %% [markdown]
# The two paradigms that fit a dataset show the contrast directly on the iris flowers, 150 examples with four measurements and three species.

# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score
X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# supervised: with the species labels, learn to predict them
clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print(round(accuracy_score(y_te, clf.predict(X_te)), 3))     # 1.0

# unsupervised: with no labels at all, find groups by similarity
clusters = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
print(round(adjusted_rand_score(y, clusters), 3))            # 0.73

# %% [markdown]
# ## 4.1.3 Two reasons to fit a model: prediction and inference

# %% [markdown]
# The first half of the block asks the inference question with statsmodels, because it reports each coefficient with its standard error and p-value. The second half asks the prediction question, scoring a linear model and a random forest only on patients they did not train on.

# %%
import statsmodels.api as sm
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
d = load_diabetes(as_frame=True)
X, y = d.data, d.target                        # 442 patients, 10 scaled measurements, progression one year later

# inference: fit one linear model to every patient and read each coefficient with its uncertainty
ols = sm.OLS(y, sm.add_constant(X)).fit()
print(ols.summary2().tables[1][["Coef.", "Std.Err.", "P>|t|"]].round(3))
print("in-sample R2", round(ols.rsquared, 3))

# prediction: score each model only on patients it never saw
cv = KFold(5, shuffle=True, random_state=0)
for name, model in [("linear regression", LinearRegression()),
                    ("random forest", RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=2))]:
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    print(f"{name:18s} cross-validated R2 {scores.mean():.3f} +/- {scores.std():.3f}")
rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=2).fit(X, y)
top = sorted(zip(rf.feature_importances_.round(3), X.columns), reverse=True)[:4]
print("forest importances:", [(f, float(v)) for v, f in top])
#          Coef.  Std.Err.  P>|t|
# const  152.133     2.576  0.000
# age    -10.010    59.749  0.867
# sex   -239.816    61.222  0.000
# bmi    519.846    66.533  0.000
# bp     324.385    65.422  0.000
# s1    -792.176   416.680  0.058
# s2     476.739   339.030  0.160
# s3     101.043   212.531  0.635
# s4     177.063   161.476  0.273
# s5     751.274   171.900  0.000
# s6      67.627    65.984  0.306
# in-sample R2 0.518
# linear regression  cross-validated R2 0.489 +/- 0.090
# random forest      cross-validated R2 0.427 +/- 0.100
# forest importances: [('s5', 0.302), ('bmi', 0.282), ('bp', 0.094), ('s6', 0.07)]

# %% [markdown]
# ## 4.2.1 Learning with few or no labels

# %% [markdown]
# The effect is real and easy to show on handwritten digits.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score

X, y = load_digits(return_X_y=True)                 # 1797 handwritten digits, 10 classes
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# hide 97% of the training labels by marking them -1 ("unlabeled")
rng = np.random.default_rng(1)
unlabeled = rng.random(len(y_tr)) < 0.97
y_semi = y_tr.copy()
y_semi[unlabeled] = -1
print(f"labeled examples: {(y_semi != -1).sum()} of {len(y_tr)}")   # 38 of 1257

# supervised: train on the few labeled rows only
sup = LogisticRegression(max_iter=2000).fit(X_tr[~unlabeled], y_tr[~unlabeled])
print(round(accuracy_score(y_te, sup.predict(X_te)), 3))            # 0.794

# semi-supervised: the same few labels plus the many unlabeled points
semi = SelfTrainingClassifier(LogisticRegression(max_iter=2000)).fit(X_tr, y_semi)
print(round(accuracy_score(y_te, semi.predict(X_te)), 3))           # 0.844

# %% [markdown]
# The graph-based family runs in one call on the same 38 labels.

# %%
from sklearn.semi_supervised import LabelSpreading
# X_tr, X_te, y_tr, y_te, y_semi, and unlabeled carry over from the self-training block
spread = LabelSpreading(kernel="knn", n_neighbors=7).fit(X_tr, y_semi)   # a graph over all 1257 training digits
hidden = spread.transduction_[unlabeled]          # the labels the graph gave the 1219 hidden training digits
print(round(accuracy_score(y_tr[unlabeled], hidden), 3))           # 0.847
print(round(accuracy_score(y_te, spread.predict(X_te)), 3))        # 0.87

# %% [markdown]
# No human labeled anything; the target column is part of the data itself.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

digits = load_digits()
X_img = digits.data / 16.0                # 1797 real digit images, pixel values scaled to 0-1
masked = 28                               # hide one pixel: row 3, column 4 of the 8-by-8 grid
X_in = np.delete(X_img, masked, axis=1)   # the input is the other 63 pixels
y_out = X_img[:, masked]                  # the "label" is the hidden pixel, made from raw data
X_tr, X_te, y_tr, y_te = train_test_split(X_in, y_out, test_size=0.3, random_state=0)
model = Ridge(alpha=1.0).fit(X_tr, y_tr)
print(round(r2_score(y_te, model.predict(X_te)), 3))                # 0.736
print(round(r2_score(y_te, np.full(len(y_te), y_tr.mean())), 3))    # -0.002  predict-the-mean

# %% [markdown]
# ## 4.2.2 Reusing knowledge across tasks

# %% [markdown]
# The honest comparison is against the same architecture trained from scratch on those same few labels.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

digits = load_digits()
src = digits.target < 5                       # source task: digits 0-4, with plenty of data
body = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000,
                     random_state=0).fit(digits.data[src], digits.target[src])

def features(Xq):                             # the frozen body: the source task's hidden layer
    return np.maximum(0, Xq @ body.coefs_[0] + body.intercepts_[0])

X_new, y_new = digits.data[~src], digits.target[~src]   # related target task: digits 5-9
for n_labels in (15, 30, 50):
    X_tr, X_te, y_tr, y_te = train_test_split(X_new, y_new, train_size=n_labels,
                                              random_state=0, stratify=y_new)
    scratch = MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000,
                            random_state=0).fit(X_tr, y_tr)             # same net, from zero
    head = LogisticRegression(max_iter=5000).fit(features(X_tr), y_tr)  # small head, frozen body
    print(f"{n_labels:2d} labels: from scratch {scratch.score(X_te, y_te):.3f}"
          f"   frozen body + new head {head.score(features(X_te), y_te):.3f}")

# %% [markdown]
# Below, one hidden layer of 32 units is trained to answer two questions about every digit image at once, whether the digit is five or more and whether it is even, and the comparison is against two networks of the same shape trained on one question each.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
X_tr, X_te = X_tr / 16.0, X_te / 16.0                    # pixel values 0-16 scaled to 0-1
# two related binary tasks read off the same images: "is the digit 5 or more?" and "is it even?"
Y_tr = np.c_[y_tr >= 5, y_tr % 2 == 0].astype(int)
Y_te = np.c_[y_te >= 5, y_te % 2 == 0].astype(int)
n_params = lambda net: sum(w.size for w in net.coefs_) + sum(b.size for b in net.intercepts_)

shared = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=0).fit(X_tr, Y_tr)
P = shared.predict(X_te)                                   # one hidden layer serves both outputs
print("shared net  :", round(accuracy_score(Y_te[:, 0], P[:, 0]), 3),
      round(accuracy_score(Y_te[:, 1], P[:, 1]), 3), "parameters", n_params(shared))
for t, name in enumerate(["five or more", "even"]):
    single = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=0).fit(X_tr, Y_tr[:, t])
    print(f"own net for '{name}': {accuracy_score(Y_te[:, t], single.predict(X_te)):.3f}",
          "parameters", n_params(single))
# shared net  : 0.976 0.976 parameters 2146
# own net for 'five or more': 0.976 parameters 2113
# own net for 'even': 0.976 parameters 2113

# %% [markdown]
# ## 4.2.3 Learning over time

# %% [markdown]
# Online learning is easy to watch. scikit-learn estimators with a `partial_fit` method accept the data one batch at a time and keep their weights between calls; the block feeds a logistic model the 1,257 training digits in ten batches of about 126 and scores the test set after each.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
X_tr, X_te = X_tr / 16.0, X_te / 16.0
online = SGDClassifier(loss="log_loss", random_state=0)   # a logistic model trained one batch at a time
accs = []
for idx in np.array_split(np.arange(len(y_tr)), 10):      # the 1257 training digits arrive in 10 batches
    online.partial_fit(X_tr[idx], y_tr[idx], classes=np.arange(10))   # every class declared on the first call
    accs.append(round(accuracy_score(y_te, online.predict(X_te)), 3))
print(accs)
# [0.667, 0.833, 0.819, 0.854, 0.794, 0.907, 0.794, 0.889, 0.87, 0.904]

# %% [markdown]
# Catastrophic forgetting is just as easy to produce. The same `partial_fit`, on a small neural network trained for thirty passes over the digits 0 through 4 and then thirty passes over only the digits 5 through 9, erases the first task.

# %%
from sklearn.neural_network import MLPClassifier
# X_tr, X_te, y_tr, y_te carry over from the SGD online learning block
low_tr, high_tr, low_te, high_te = y_tr < 5, y_tr >= 5, y_te < 5, y_te >= 5
net = MLPClassifier(hidden_layer_sizes=(64,), random_state=0)
for epoch in range(30):                                   # task A: learn the digits 0-4
    net.partial_fit(X_tr[low_tr], y_tr[low_tr], classes=np.arange(10))
print("after task A: accuracy on 0-4", round(accuracy_score(y_te[low_te], net.predict(X_te[low_te])), 3))
for epoch in range(30):                                   # task B: now learn only the digits 5-9
    net.partial_fit(X_tr[high_tr], y_tr[high_tr])
print("after task B: accuracy on 0-4", round(accuracy_score(y_te[low_te], net.predict(X_te[low_te])), 3),
      " on 5-9", round(accuracy_score(y_te[high_te], net.predict(X_te[high_te])), 3))
print("labels it now gives the 0-4 test digits:", np.unique(net.predict(X_te[low_te])))
# after task A: accuracy on 0-4 0.967
# after task B: accuracy on 0-4 0.0  on 5-9 0.948
# labels it now gives the 0-4 test digits: [5 6 7 8 9]

# %% [markdown]
# ## 4.2.4 Learning to learn

# %% [markdown]
# Example 4-1 measures a raw-pixel baseline and then meta-trains a prototypical network (Snell, Swersky, and Zemel, 2017) on Omniglot (Lake, Salakhutdinov, and Tenenbaum, 2015), testing it on characters from alphabets it never saw. The first run downloads about 16 MB into a `data` folder.

# %%
import numpy as np
import torch
from PIL import Image
from torchvision import datasets

# Omniglot (Lake et al., 2015): 1623 handwritten characters from 50 alphabets, 20 drawings of each
omni_bg = datasets.Omniglot("data", background=True, download=True)    # 30 alphabets, for meta-training
omni_ev = datasets.Omniglot("data", background=False, download=True)   # 20 other alphabets, never trained on

def to_arrays(ds):                    # 105 by 105 drawing -> 28 by 28, ink = 1, flattened to 784 numbers
    X = np.stack([1 - np.asarray(img.resize((28, 28), Image.Resampling.BILINEAR), dtype=np.float32) / 255
                  for img, _ in ds])
    return X.reshape(len(X), -1), np.array([char for _, char in ds])

X_bg, y_bg = to_arrays(omni_bg)
X_ev, y_ev = to_arrays(omni_ev)
print(X_bg.shape, len(np.unique(y_bg)), "characters |", X_ev.shape, len(np.unique(y_ev)), "characters")

def sample_episode(X, y, rng, n_way, k_shot, n_query):
    """One small few-shot task: n_way random characters, each with k_shot labeled drawings
    (the support set) and n_query further drawings to classify (the query set)."""
    chars = rng.choice(np.unique(y), n_way, replace=False)
    picks = [rng.permutation(np.flatnonzero(y == c))[:k_shot + n_query] for c in chars]
    support = torch.tensor(np.stack([X[p[:k_shot]] for p in picks]))   # (n_way, k_shot, 784)
    query = torch.tensor(np.stack([X[p[k_shot:]] for p in picks]))     # (n_way, n_query, 784)
    return support, query

def proto_scores(embed, support, query):
    """Score every query against every class: minus its squared distance to the class prototype."""
    n_way, k_shot, d = support.shape
    protos = embed(support.reshape(-1, d)).reshape(n_way, k_shot, -1).mean(1)   # c_k, one per class
    q = embed(query.reshape(-1, d))                                              # f(x) for each query
    return -((q[:, None, :] - protos[None, :, :]) ** 2).sum(-1)                  # (queries, n_way)

def few_shot_accuracy(embed, k_shot, n_way=5, n_query=15, n_episodes=600):
    rng = np.random.default_rng(0)        # the same 600 test tasks for every embedding compared
    accs = []
    for _ in range(n_episodes):
        support, query = sample_episode(X_ev, y_ev, rng, n_way, k_shot, n_query)
        with torch.no_grad():
            pred = proto_scores(embed, support, query).argmax(1)   # the nearest prototype wins
        accs.append((pred == torch.arange(n_way).repeat_interleave(n_query)).float().mean().item())
    return np.mean(accs), 1.96 * np.std(accs) / np.sqrt(n_episodes)   # mean and 95% half width

def report_few_shot(name, embed, n_way=5):
    for k in (1, 5):
        acc, half = few_shot_accuracy(embed, k, n_way)
        print(f"{name:15s} {n_way}-way {k}-shot accuracy {acc:.3f} +/- {half:.3f}")

report_few_shot("raw pixels", lambda x: x)   # no learning at all: each prototype is an average image

# meta-training: the embedding f is a small network, trained on episodes of the 964 background characters
torch.manual_seed(0)
proto_net = torch.nn.Sequential(torch.nn.Linear(784, 256), torch.nn.ReLU(),
                                torch.nn.Linear(256, 256), torch.nn.ReLU(),
                                torch.nn.Linear(256, 64))      # 784 pixels in, a 64-number embedding out
ep_rng = np.random.default_rng(0)
ep_truth = torch.arange(20).repeat_interleave(5)               # 20-way episodes, 5 queries per character
ep_losses = []
for step in range(1, 10001):                                   # 10,000 episodes, each a fresh task
    support, query = sample_episode(X_bg, y_bg, ep_rng, n_way=20, k_shot=5, n_query=5)
    ep_scores = proto_scores(proto_net, support, query)
    ep_loss = (torch.logsumexp(ep_scores, 1) - ep_scores[torch.arange(100), ep_truth]).mean()  # 2.5.2
    ep_loss.backward()                                         # gradients flow through the prototypes too
    with torch.no_grad():
        for param in proto_net.parameters():                   # one plain gradient descent step (2.6)
            param -= 0.1 * param.grad
            param.grad = None
    ep_losses.append(ep_loss.item())
    if step in (1000, 5000, 10000):
        print(f"episodes {step - 999:5d} to {step:5d}: mean loss {np.mean(ep_losses[-1000:]):.3f}")
report_few_shot("meta-trained", proto_net)
# (19280, 784) 964 characters | (13180, 784) 659 characters
# raw pixels      5-way 1-shot accuracy 0.453 +/- 0.008
# raw pixels      5-way 5-shot accuracy 0.711 +/- 0.007
# episodes     1 to  1000: mean loss 0.993
# episodes  4001 to  5000: mean loss 0.108
# episodes  9001 to 10000: mean loss 0.055
# meta-trained    5-way 1-shot accuracy 0.868 +/- 0.007
# meta-trained    5-way 5-shot accuracy 0.957 +/- 0.003

# %% [markdown]
# Episodes are not the only way to learn a good embedding, and the honest comparison is the transfer recipe of 4.2.2: train an ordinary classifier on all 964 background characters with the same architecture, steps, learning rate, and drawings per step, then keep its body as the embedding.

# %%
# X_bg, y_bg, report_few_shot, np, and torch carry over from Example 4-1
torch.manual_seed(0)
cls_body = torch.nn.Sequential(torch.nn.Linear(784, 256), torch.nn.ReLU(),
                               torch.nn.Linear(256, 256), torch.nn.ReLU(),
                               torch.nn.Linear(256, 64))      # the same architecture as proto_net
cls_head = torch.nn.Linear(64, 964)                           # one output per background character
X_bg_t, y_bg_t = torch.tensor(X_bg), torch.tensor(y_bg)
cls_rng = np.random.default_rng(0)
for step in range(10000):                 # as many steps as episodes, 200 drawings each, like an episode
    batch = torch.tensor(cls_rng.choice(len(y_bg_t), 200, replace=False))
    cls_logits = cls_head(cls_body(X_bg_t[batch]))
    cls_loss = (torch.logsumexp(cls_logits, 1) - cls_logits[torch.arange(200), y_bg_t[batch]]).mean()
    cls_loss.backward()
    with torch.no_grad():
        for param in list(cls_body.parameters()) + list(cls_head.parameters()):
            param -= 0.1 * param.grad
            param.grad = None
report_few_shot("classifier body", cls_body)   # drop the 964-way head, keep the body as the embedding
# classifier body 5-way 1-shot accuracy 0.849 +/- 0.007
# classifier body 5-way 5-shot accuracy 0.949 +/- 0.003

# %% [markdown]
# ## 4.2.6 Four ways to categorize any model

# %% [markdown]
# The two distinctions often line up, with parametric models tending to be model-based and nonparametric ones instance-based, and the contrast is concrete in code.

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

X, y = load_breast_cancer(return_X_y=True)          # 569 tumors, 30 features, benign vs malignant
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
X_tr = StandardScaler().fit_transform(X_tr)         # put the 30 features on one scale

log = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)    # learns fixed weights
knn = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)  # just stores the data
print("features:", X_tr.shape[1])                                              # 30
print("logistic learned parameters:", log.coef_.size + log.intercept_.size)   # 31
print("kNN stored training rows:", knn.n_samples_fit_)                         # 398

# %% [markdown]
# ## 4.3.2 Loss functions

# %% [markdown]
# Four surrogates account for almost all of practice: the hinge loss, the logistic loss, the exponential loss, and the squared loss. The block evaluates all five at six margins.

# %%
import numpy as np
m = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])               # margins y * f(x), with y in {-1, +1}
losses = {"0/1":         (m <= 0).astype(float),             # one for a wrong label, zero for a right one
          "hinge":       np.maximum(0.0, 1.0 - m),           # support vector machines
          "logistic":    np.log2(1.0 + np.exp(-m)),          # logistic regression, in bits
          "exponential": np.exp(-m),                         # AdaBoost
          "squared":     (1.0 - m) ** 2}                     # least squares pressed into service as a classifier
print("margin      " + "".join(f"{v:8.1f}" for v in m))
for name, vals in losses.items():
    print(f"{name:12s}" + "".join(f"{v:8.3f}" for v in vals))
# margin          -2.0    -1.0     0.0     0.5     1.0     2.0
# 0/1            1.000   1.000   1.000   0.000   0.000   0.000
# hinge          3.000   2.000   1.000   0.500   0.000   0.000
# logistic       3.069   1.895   1.000   0.684   0.452   0.183
# exponential    7.389   2.718   1.000   0.607   0.368   0.135
# squared        9.000   4.000   1.000   0.250   0.000   1.000

# %% [markdown]
# The next block fits the two most common surrogates to the 569 tumors of the Wisconsin breast cancer data, logistic regression minimizing the logistic loss and a linear support vector machine minimizing the hinge loss, and then evaluates each fitted model on both surrogates and on the zero-one loss.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
s = 2 * y_tr - 1                                             # relabel {0, 1} as {-1, +1} so the margin is y * f(x)
for name, model in [("logistic regression (log loss)", LogisticRegression(max_iter=5000)),
                    ("linear SVM (hinge loss)", LinearSVC(loss="hinge", max_iter=200000))]:   # the default is squared hinge
    pipe = make_pipeline(StandardScaler(), model).fit(X_tr, y_tr)
    margin = s * pipe.decision_function(X_tr)                # the score f(x), signed by the true label
    print(f"{name:32s} mean hinge {np.maximum(0, 1 - margin).mean():.3f}   "
          f"mean logistic {np.log2(1 + np.exp(-margin)).mean():.3f}   "
          f"train 0/1 error {(margin <= 0).mean():.3f}   test 0/1 error {1 - pipe.score(X_te, y_te):.3f}")
# logistic regression (log loss)   mean hinge 0.033   mean logistic 0.071   train 0/1 error 0.008   test 0/1 error 0.041
# linear SVM (hinge loss)          mean hinge 0.025   mean logistic 0.095   train 0/1 error 0.008   test 0/1 error 0.047

# %% [markdown]
# ## 4.3.3 Empirical risk and its minimization

# %% [markdown]
# The sample is deliberately kept small so that a high-degree polynomial has enough freedom to overfit; the same forty-five cars and this exact pipeline return in 4.5 and 4.6, so those sections build on the numbers printed here.

# %%
import numpy as np, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
mpg = sns.load_dataset("mpg").dropna(subset=["horsepower", "mpg"])
sub = mpg.sample(n=45, random_state=1)      # a small real sample, so a high-degree fit can overfit
X = sub[["horsepower"]].to_numpy(float); y = sub["mpg"].to_numpy(float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.35, random_state=0)

for d in [1, 3, 5, 9, 15]:
    model = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
    print(d, round(mean_squared_error(y_tr, model.predict(X_tr)), 3),   # empirical (training) risk
             round(mean_squared_error(y_te, model.predict(X_te)), 3))   # true-risk estimate (test)
# 1  19.593    31.202
# 3  9.674     17.026
# 5  9.599     18.767
# 9  8.018     4030.191
# 15 6.052     370469.946

# %% [markdown]
# One numerical caveat belongs on the record before the chapter leans on these numbers. The raw powers of a single standardized column become nearly collinear once the degree is high, which is the ill-conditioning that the condition number of 2.1.10 measures, and `LinearRegression` solves such a system with a least-squares routine that treats directions below a numerical cutoff as absent, reporting in its `rank_` attribute how many feature directions it actually used.

# %%
for d in [9, 15]:
    model = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
    Z = model[:-1].transform(X_tr)[:, 1:]        # the standardized power features, minus the all-zero bias column
    print(d, model[-1].rank_, f"{np.linalg.cond(Z):.2g}")   # degree, directions used, condition number
# 9 7 1.2e+08
# 15 8 1.1e+15

# %% [markdown]
# ## 4.3.4 Why learning generalizes at all: capacity, VC dimension, and PAC learning

# %% [markdown]
# The code counts how often the worst of K coins strays beyond the single-coin tolerance, and then beyond the tolerance that includes the ln K term.

# %%
import numpy as np
H, eps, delta = 2 ** 20, 0.05, 0.05
print(round((np.log(H) + np.log(1 / delta)) / eps, 1))               # realizable: 337.2, so 338 examples
print(round((np.log(H) + np.log(2 / delta)) / (2 * eps ** 2), 1))    # agnostic: 3510.4, so 3,511 examples

rng = np.random.default_rng(0)
p, n, trials = 0.3, 200, 100_000                      # each hypothesis is a coin with true error p, seen on n examples
eps1 = np.sqrt(np.log(2 / delta) / (2 * n))           # the single-hypothesis tolerance, 0.096
for K in [1, 10, 100, 1000]:                          # K hypotheses whose training errors are independent coins
    err_hat = rng.binomial(n, p, size=(trials, K)) / n            # training errors, trials x K
    worst = np.abs(err_hat - p).max(axis=1)                       # the largest miss among the K
    epsK = np.sqrt((np.log(K) + np.log(2 / delta)) / (2 * n))     # the tolerance with the ln K term
    print(f"K={K:5d}  miss beyond {eps1:.3f}: {(worst > eps1).mean():.3f}   "
          f"beyond {epsK:.3f}: {(worst > epsK).mean():.4f}")
# 337.2
# 3510.4
# K=    1  miss beyond 0.096: 0.002   beyond 0.096: 0.0025
# K=   10  miss beyond 0.096: 0.025   beyond 0.122: 0.0016
# K=  100  miss beyond 0.096: 0.225   beyond 0.144: 0.0012
# K= 1000  miss beyond 0.096: 0.922   beyond 0.163: 0.0007

# %% [markdown]
# ## 4.3.5 The curse of dimensionality

# %% [markdown]
# we scatter a thousand random points in the unit cube across a range of dimensions and watch the spread of their pairwise distances shrink relative to the average distance.

# %%
import numpy as np
from scipy.spatial.distance import pdist
rng = np.random.default_rng(0)
for d in [1, 2, 10, 100, 1000]:
    X = rng.random((1000, d))                        # 1000 random points in the d-dim unit cube
    dist = pdist(X)                                   # every pairwise Euclidean distance
    spread = (dist.max() - dist.min()) / dist.mean()  # near-to-far spread, relative to the typical distance
    print(d, "mean dist", round(dist.mean(), 3), "relative spread", round(spread, 3))
# 1    mean dist 0.329   relative spread 3.04
# 2    mean dist 0.526   relative spread 2.587
# 10   mean dist 1.268   relative spread 1.659
# 100  mean dist 4.071   relative spread 0.533
# 1000 mean dist 12.914  relative spread 0.174

# %% [markdown]
# ## 4.3.6 The assumption underneath: i.i.d. data and distribution shift

# %% [markdown]
# Example 4-2 runs the ladder twice on the California housing data, using six features and deliberately leaving out latitude and longitude, then performs adversarial validation on both splits.

# %%
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score
housing = fetch_california_housing(as_frame=True).frame
feats = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]   # no coordinates
south = housing["Latitude"] < 35               # the model is built in the south and sold into the north
print(int(south.sum()), int((~south).sum()))   # 11292 9348

def ladder(fit_rows, new_rows, tag):
    # 80 percent of the fitting rows train the model and 20 percent are held out as the train-dev set,
    # drawn from the same distribution as the training rows; the new rows are the deployment test set
    X_fit, y_fit = housing.loc[fit_rows, feats], housing.loc[fit_rows, "MedHouseVal"]
    X_tr, X_dev, y_tr, y_dev = train_test_split(X_fit, y_fit, test_size=0.2, random_state=0)
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=0, n_jobs=2).fit(X_tr, y_tr)
    X_te, y_te = housing.loc[new_rows, feats], housing.loc[new_rows, "MedHouseVal"]
    print(f"{tag:11s} train R2 {r2_score(y_tr, rf.predict(X_tr)):.3f}   train-dev R2 {r2_score(y_dev, rf.predict(X_dev)):.3f}"
          f"   test R2 {r2_score(y_te, rf.predict(X_te)):.3f}")

rng = np.random.default_rng(0)
is_test = rng.random(len(housing)) < 0.4       # a random split with about the same test share as the geographic one
ladder(~is_test, is_test, "random")
ladder(south, ~south, "geographic")

def adversarial_auc(is_new):
    # adversarial validation: label every row by the side of the split it is on and try to predict that label
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=0, n_jobs=2)
    p = cross_val_predict(clf, housing[feats], is_new, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    return roc_auc_score(is_new, p)            # 0.5 means the two sides cannot be told apart
print("adversarial AUC, random split    ", round(adversarial_auc(is_test), 3))
print("adversarial AUC, geographic split", round(adversarial_auc(~south), 3))
region = south.map({True: "south", False: "north"})
print(housing.groupby(region)[["MedInc", "AveRooms", "Population", "MedHouseVal"]].mean().round(2))
# 11292 9348
# random      train R2 0.746   train-dev R2 0.676   test R2 0.696
# geographic  train R2 0.747   train-dev R2 0.674   test R2 0.631
# adversarial AUC, random split     0.49
# adversarial AUC, geographic split 0.749
#           MedInc  AveRooms  Population  MedHouseVal
# Latitude
# north       3.73      5.67     1295.43         1.89
# south       3.99      5.23     1533.14         2.21

# %% [markdown]
# ## 4.4.2 The tradeoff in action

# %% [markdown]
# We fix the truth here as a sine curve corrupted by known Gaussian noise, exactly so the ground truth is available to compare the trained models against.

# %%
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
true_f = lambda x: np.sin(1.5 * x)              # the ground truth we get to know
x_test = np.linspace(-3, 3, 60).reshape(-1, 1)
y_true = true_f(x_test).ravel()                 # noiseless truth at the test points
rng_noise = np.random.default_rng(1)            # a separate generator for fresh noisy targets

for d in [1, 3, 5, 7, 9]:
    preds = []
    for _ in range(200):                          # 200 datasets from the same source
        X = rng.uniform(-3, 3, 50).reshape(-1, 1)
        y = true_f(X).ravel() + rng.normal(0, 0.3, 50)
        m = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression()).fit(X, y)
        preds.append(m.predict(x_test))
    preds = np.array(preds)
    bias2 = np.mean((preds.mean(axis=0) - y_true) ** 2)    # avg model vs truth
    variance = np.mean(preds.var(axis=0))                  # disagreement among models
    noisy = y_true + rng_noise.normal(0, 0.3, size=preds.shape)   # fresh noisy targets, sigma = 0.3
    print(d, round(bias2, 3), round(variance, 3), round(bias2 + variance, 3),
          round(bias2 + variance + 0.3 ** 2, 3),             # predicted expected squared error
          round(np.mean((preds - noisy) ** 2), 3))           # measured against the noisy targets
# 1 0.486 0.025 0.512 0.602 0.601
# 3 0.08 0.02 0.1 0.19 0.189
# 5 0.002 0.017 0.02 0.11 0.108
# 7 0.0 0.039 0.039 0.129 0.131
# 9 0.002 0.212 0.213 0.303 0.3

# %% [markdown]
# ## 4.4.3 The Bayes error rate: classification's irreducible floor

# %% [markdown]
# On real data is unknown, which is the whole reason the Bayes error has to be estimated rather than computed.

# %%
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# class 0 ~ N(-1, 1), class 1 ~ N(+1, 1), equal priors; we KNOW P(y|x), so the floor is computable
rng = np.random.default_rng(4)
n = 50000
X = np.concatenate([rng.normal(-1, 1, n // 2), rng.normal(1, 1, n // 2)]).reshape(-1, 1)
y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
Xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=4)
clf = LogisticRegression().fit(Xtr, ytr)

bayes_error = 0.5 * norm.cdf(0, loc=1, scale=1) + 0.5 * (1 - norm.cdf(0, loc=-1, scale=1))
print(round(bayes_error, 4))     # 0.1587   = the standard-normal tail beyond one sigma
print(round(1 - clf.score(xte, yte), 4))    # 0.1595   it hugs the floor

# %% [markdown]
# ## 4.5.1 The two failure modes

# %% [markdown]
# The first and third causes can be measured on the handwritten digits that 4.2.1 used: flip the labels of one training digit in five, append columns of pure Gaussian noise, and end with the screen of 3.8.6.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
dig = load_digits()
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(dig.data, dig.target, test_size=0.3, random_state=0, stratify=dig.target)

# cause 1, mislabeled examples: flip 20 percent of the training labels to some other digit
rng = np.random.default_rng(0)
flip = rng.random(len(yd_tr)) < 0.20
y_noisy = yd_tr.copy()
y_noisy[flip] = (yd_tr[flip] + rng.integers(1, 10, flip.sum())) % 10   # a different digit, never the true one
print("flipped", int(flip.sum()), "of", len(yd_tr))
for k in [1, 15]:
    clean = KNeighborsClassifier(k).fit(Xd_tr, yd_tr).score(Xd_te, yd_te)
    noisy = KNeighborsClassifier(k).fit(Xd_tr, y_noisy).score(Xd_te, yd_te)
    print(f"k={k:2d}  clean labels {clean:.3f}   20% flipped {noisy:.3f}")

# cause 3, rare features and spurious correlations: columns of pure noise appended to the 64 pixels
for extra in [0, 64, 640]:
    noise_tr = rng.standard_normal((len(Xd_tr), extra)); noise_te = rng.standard_normal((len(Xd_te), extra))
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    acc = lr.fit(np.hstack([Xd_tr, noise_tr]), yd_tr).score(np.hstack([Xd_te, noise_te]), yd_te)
    print(f"{extra:4d} noise columns: logistic regression test accuracy {acc:.3f}")

# the screen of 3.8.6: out-of-fold probability of each given label, lowest first
oof = cross_val_predict(KNeighborsClassifier(15), Xd_tr, y_noisy, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                        method="predict_proba")
trust = oof[np.arange(len(y_noisy)), y_noisy]              # how much the other folds believe each given label
worst = np.argsort(trust)                                 # least believable labels first
print("truly flipped among the 100 least trusted:", int(flip[worst[:100]].sum()), " among the 250 least trusted:", int(flip[worst[:250]].sum()))
# flipped 252 of 1257
# k= 1  clean labels 0.983   20% flipped 0.778
# k=15  clean labels 0.974   20% flipped 0.974
#    0 noise columns: logistic regression test accuracy 0.972
#   64 noise columns: logistic regression test accuracy 0.943
#  640 noise columns: logistic regression test accuracy 0.869
# truly flipped among the 100 least trusted: 98  among the 250 least trusted: 243

# %% [markdown]
# ## 4.5.2 Diagnosis with learning curves

# %% [markdown]
# a too-simple degree-one model, a balanced degree-three model, and a too-complex degree-fifteen model.

# %%
import numpy as np, seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
mpg = sns.load_dataset("mpg").dropna(subset=["horsepower", "mpg"])
sub = mpg.sample(n=45, random_state=1)
X = sub[["horsepower"]].to_numpy(float); y = sub["mpg"].to_numpy(float)

for degree, name in [(1, "underfit"), (3, "good"), (15, "overfit")]:
    model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression())
    sizes, tr, va = learning_curve(model, X, y, train_sizes=np.linspace(0.3, 1.0, 5),
                                   cv=5, scoring="neg_mean_squared_error")
    print(name)
    for n, t, v in zip(sizes, -tr.mean(axis=1), -va.mean(axis=1)):
        print(f"  n={n:2d}  train {t:7.3f}  val {v:11.3g}")
# underfit: train 15.728, 18.666, 22.119, 21.865, 22.333
#           val   41.9, 46.8, 42.3, 42.3, 33.6
# good:     train 9.939, 13.149, 12.922, 12.369, 11.309
#           val   554, 338, 64.1, 27.9, 14.4
# overfit:  train 5.480, 9.833, 9.942, 9.722, 9.512
#           val   6.01e+12, 2.05e+11, 5.22e+10, 1.16e+10, 2.93e+06

# %% [markdown]
# Here it sweeps the rbf kernel coefficient gamma of a support vector classifier on the breast cancer dataset, 569 real tumors labeled benign or malignant; the classifier itself waits for 6.5, so treat it here as a black box with one complexity dia

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
gammas = np.logspace(-4, 1, 6)
pipe = make_pipeline(StandardScaler(), SVC())
tr, va = validation_curve(pipe, X, y, param_name="svc__gamma",
                          param_range=gammas, cv=5, scoring="accuracy")
for g, t, v in zip(gammas, tr.mean(axis=1), va.mean(axis=1)):
    print(f"gamma {g:8.4f}: train {t:.3f}  val {v:.3f}")
# gamma   0.0001: train 0.794  val 0.796
# gamma   0.0010: train 0.951  val 0.947
# gamma   0.0100: train 0.978  val 0.968
# gamma   0.1000: train 0.990  val 0.960
# gamma   1.0000: train 1.000  val 0.631
# gamma  10.0000: train 1.000  val 0.627

# %% [markdown]
# ## 4.5.4 The modern twist: double descent

# %% [markdown]
# The Legendre polynomials start from P0(t) = 1 and P1(t) = t and follow the recurrence (n+1) P(n+1) = (2n+1) t P(n) - n P(n-1); they are orthogonal on [-1, 1], so a design matrix built from them stays well conditioned on evenly spread points where raw powers do not.

# %%
import numpy as np
print(np.polynomial.legendre.legvander(np.array([0.5]), 3))   # P0 to P3 at t = 0.5
t = np.linspace(-1, 1, 1001)                                  # 1,001 points spread evenly over [-1, 1]
for D in (5, 10, 15, 20):                                     # degree, raw-power condition number, Legendre condition number
    print(D, f"{np.linalg.cond(np.polynomial.polynomial.polyvander(t, D)):.3g}",
          f"{np.linalg.cond(np.polynomial.legendre.legvander(t, D)):.3g}")
# [[ 1.      0.5    -0.125  -0.4375]]
# 5 43.1 3.3
# 10 3.05e+03 4.54
# 15 2.28e+05 5.51
# 20 1.75e+07 6.33

# %% [markdown]
# Example 4-4. Double descent on the forty-five cars: a Legendre feature sweep through the interpolation threshold with minimum-norm fits.

# %%
import numpy as np, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mpg = sns.load_dataset("mpg").dropna(subset=["horsepower", "mpg"])
sub = mpg.sample(n=45, random_state=1)                    # the chapter's same 45 cars
X = sub[["horsepower"]].to_numpy(float); y = sub["mpg"].to_numpy(float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.35, random_state=0)

lo, hi = 46.0, 230.0                          # horsepower's range in the full mpg data
def legendre_features(Xcol, D):               # degree-D Legendre basis on [-1, 1]
    t = 2 * (Xcol.ravel() - lo) / (hi - lo) - 1
    return np.polynomial.legendre.legvander(t, D)

hp = X_tr.ravel()
group_mean = np.array([y_tr[hp == v].mean() for v in hp])     # the best any function of horsepower can do at each car
print(len(hp), len(np.unique(hp)), round(np.mean((y_tr - group_mean) ** 2), 2))   # 29 24 1.09
print([int(np.linalg.matrix_rank(legendre_features(X_tr, D))) for D in (23, 24, 25)])   # [23, 23, 24]

y_mean = y_tr.mean()
for D in [1, 3, 9, 15, 25, 29, 40, 60, 100, 300, 1000]:
    A_tr, A_te = legendre_features(X_tr, D), legendre_features(X_te, D)
    coef = np.linalg.pinv(A_tr) @ (y_tr - y_mean)   # the minimum-norm least-squares fit
    tr = mean_squared_error(y_tr, A_tr @ coef + y_mean)
    te = mean_squared_error(y_te, A_te @ coef + y_mean)
    print(f"degree {D:4d}  train MSE {tr:6.2f}  test MSE {te:12.4g}  coefficient norm {np.linalg.norm(coef):10.4g}")
# degree    1  train MSE  19.59  test MSE         31.2  coefficient norm      16.44
# degree    3  train MSE   9.67  test MSE        17.03  coefficient norm      15.34
# degree    9  train MSE   5.29  test MSE         6819  coefficient norm      427.5
# degree   15  train MSE   4.24  test MSE    1.048e+12  coefficient norm  9.241e+06
# degree   25  train MSE   1.09  test MSE    3.342e+26  coefficient norm   6.01e+13
# degree   29  train MSE   1.09  test MSE    8.622e+15  coefficient norm  3.447e+09
# degree   40  train MSE   1.09  test MSE    2.283e+08  coefficient norm  4.072e+05
# degree   60  train MSE   1.09  test MSE        741.1  coefficient norm      285.6
# degree  100  train MSE   1.09  test MSE        15.87  coefficient norm      28.89
# degree  300  train MSE   1.09  test MSE        25.52  coefficient norm      16.84
# degree 1000  train MSE   1.09  test MSE        27.13  coefficient norm      15.38

# %% [markdown]
# ## 4.6.1 The idea: penalize complexity

# %% [markdown]
# The effect on the wildly overfit degree-fifteen fit of 4.3.3 is decisive, and we rebuild that small forty-five-car sample of horsepower and mpg here so the block runs on its own.

# %%
import numpy as np, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
mpg = sns.load_dataset("mpg").dropna(subset=["horsepower", "mpg"])
sub = mpg.sample(n=45, random_state=1)      # a small real sample, so a high-degree fit can overfit
X = sub[["horsepower"]].to_numpy(float); y = sub["mpg"].to_numpy(float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.35, random_state=0)

from sklearn.linear_model import Ridge
unreg = make_pipeline(PolynomialFeatures(15), StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
reg   = make_pipeline(PolynomialFeatures(15), StandardScaler(), Ridge(alpha=1.0)).fit(X_tr, y_tr)
print(round(mean_squared_error(y_te, unreg.predict(X_te)), 3))   # 370469.946  overfit
print(round(mean_squared_error(y_te, reg.predict(X_te)), 3))     # 28.375      regularized

# %% [markdown]
# ## 4.6.2 Ridge, Lasso, and what they do to the weights

# %% [markdown]
# we standardize the ten real features of the diabetes dataset, 442 patients whose one-year disease progression we are predicting, and watch what each penalty does to the ten coefficients.

# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
d = load_diabetes()                              # 442 patients, 10 real features
Xs = StandardScaler().fit_transform(d.data); y = d.target
for name, m in [("OLS", LinearRegression()), ("Ridge (L2)", Ridge(alpha=10)),
                ("Lasso (L1)", Lasso(alpha=3)),
                ("ElasticNet", ElasticNet(alpha=3, l1_ratio=0.7))]:
    m.fit(Xs, y)
    print(f"{name:11s} nonzero: {int(np.sum(np.abs(m.coef_) > 1e-6)):2d}  "
          f"norm: {np.linalg.norm(m.coef_):5.1f}", np.round(m.coef_, 1))
# OLS         nonzero: 10  norm:  65.5 [ -0.5 -11.4  24.7  15.4 -37.7  22.7   4.8   8.4  35.7   3.2]
# Ridge (L2)  nonzero: 10  norm:  42.6 [ -0.3 -10.9  24.6  15.1 -11.3   1.8  -6.6   5.6  25.3   3.5]
# Lasso (L1)  nonzero:  7  norm:  36.7 [ 0.  -5.8 24.4 12.3 -0.6 -0.  -9.5  0.  21.9  0.8]
# ElasticNet  nonzero:  8  norm:  23.5 [ 0.4 -2.7 14.6  9.2  0.  -0.  -6.6  4.6 12.6  4.7]

# %% [markdown]
# Sweep Lasso's `alpha` to watch the selection tighten.

# %%
# Xs and y carry over from the Ridge, Lasso, and Elastic Net comparison
for a in [0.3, 1, 3, 10, 30]:
    m = Lasso(alpha=a).fit(Xs, y)
    print(f"alpha {a:4}  nonzero: {int(np.sum(np.abs(m.coef_) > 1e-6)):2d}  norm: {np.linalg.norm(m.coef_):5.1f}")
# alpha  0.3  nonzero:  8  norm:  42.0
# alpha    1  nonzero:  7  norm:  40.5
# alpha    3  nonzero:  7  norm:  36.7
# alpha   10  nonzero:  4  norm:  30.8
# alpha   30  nonzero:  2  norm:  14.2

# %% [markdown]
# ## 4.6.4 Elastic Net and choosing a penalty

# %% [markdown]
# The penalty formulas are only symbols until they are evaluated on fitted weights, so evaluate them on the coefficients of 4.6.2.

# %%
# Xs and y carry over from the 4.6.2 block; refit the three penalized models under their own names
ri, la, en = Ridge(alpha=10).fit(Xs, y), Lasso(alpha=3).fit(Xs, y), ElasticNet(alpha=3, l1_ratio=0.7).fit(Xs, y)
l1 = np.abs(en.coef_).sum(); l2sq = (en.coef_ ** 2).sum()         # Elastic Net's two sums
print(round(l1, 2), round(l2sq, 2), round(0.7 * l1 + 0.15 * l2sq, 2), round(3 * (0.7 * l1 + 0.15 * l2sq), 1))
print(round(np.abs(la.coef_).sum(), 2), round((ri.coef_ ** 2).sum(), 2))   # Lasso's L1 sum, Ridge's squared L2 sum
print(round(3 * np.abs(la.coef_).sum(), 1), round(10 * (ri.coef_ ** 2).sum(), 1))   # their penalty terms
# 55.42 551.67 121.55 364.6
# 75.27 1812.09
# 225.8 18120.9

# %% [markdown]
# ## 4.6.5 Regularization as constraining capacity

# %% [markdown]
# tracks five numbers per epoch: the weight norm, the validation log loss, the test log loss, the test accuracy, and the mean probability placed on the wrong class over the misclassified test cases.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

Xbc, ybc = load_breast_cancer(return_X_y=True)            # 569 real tumors
X_grow, X_hold, y_grow, y_hold = train_test_split(Xbc, ybc, test_size=0.3,
                                                  random_state=0, stratify=ybc)
X_fit, X_val, y_fit, y_val = train_test_split(X_grow, y_grow, train_size=40,
                                              random_state=0, stratify=y_grow)
sc = StandardScaler().fit(X_fit)
Z_fit, Z_val, Z_hold = sc.transform(X_fit), sc.transform(X_val), sc.transform(X_hold)

sgd = SGDClassifier(loss="log_loss", penalty=None,        # NO explicit regularization
                    learning_rate="constant", eta0=0.01,
                    max_iter=1, tol=None, warm_start=True, random_state=0)
hist = []
for epoch in range(1, 2001):                              # one epoch per fit() call
    sgd.fit(Z_fit, y_fit)
    p_hold = sgd.predict_proba(Z_hold)[:, 1]
    wrong = (p_hold >= 0.5).astype(int) != y_hold         # the misclassified test cases
    hist.append((epoch, np.linalg.norm(sgd.coef_),
                 log_loss(y_val, sgd.predict_proba(Z_val)[:, 1], labels=[0, 1]),
                 log_loss(y_hold, p_hold, labels=[0, 1]),
                 sgd.score(Z_hold, y_hold),                # hard test accuracy
                 np.abs(p_hold - y_hold)[wrong].mean()))   # mean probability put on the wrong class
best = min(hist, key=lambda h: h[2])                      # the epoch early stopping picks
for e, w, v, t, a, c in [hist[0], hist[24], best, hist[199], hist[-1]]:
    tag = "  <- early stopping stops here" if e == best[0] else ""
    print(f"epoch {e:4d}  ||w|| {w:5.2f}  val loss {v:.3f}  test loss {t:.3f}  "
          f"test acc {a:.3f}  wrong-class conf {c:.3f}{tag}")
# epoch    1  ||w||  0.35  val loss 0.319  test loss 0.353  test acc 0.936  wrong-class conf 0.561
# epoch   25  ||w||  1.66  val loss 0.089  test loss 0.144  test acc 0.936  wrong-class conf 0.705
# epoch   55  ||w||  2.20  val loss 0.083  test loss 0.146  test acc 0.930  wrong-class conf 0.739  <- early stopping stops here
# epoch  200  ||w||  3.44  val loss 0.095  test loss 0.178  test acc 0.942  wrong-class conf 0.860
# epoch 2000  ||w||  6.56  val loss 0.152  test loss 0.286  test acc 0.942  wrong-class conf 0.945

# %% [markdown]
# The learning rate sets the clock speed: rerun Example 4-5's loop at three step sizes and compare the stopping epoch and weight norm.

# %%
# Z_fit, Z_val, Z_hold and the labels carry over from Example 4-5; only the step size eta0 changes
for eta in [0.001, 0.01, 0.1]:
    sgd_e = SGDClassifier(loss="log_loss", penalty=None, learning_rate="constant", eta0=eta,
                          max_iter=1, tol=None, warm_start=True, random_state=0)
    runs = []
    for epoch in range(1, 2001):                          # the same one-epoch-per-fit loop as Example 4-5
        sgd_e.fit(Z_fit, y_fit)
        runs.append((epoch, np.linalg.norm(sgd_e.coef_),
                     log_loss(y_val, sgd_e.predict_proba(Z_val)[:, 1], labels=[0, 1]),
                     log_loss(y_hold, sgd_e.predict_proba(Z_hold)[:, 1], labels=[0, 1])))
    e, w, v, t = min(runs, key=lambda r: r[2])            # the epoch early stopping picks
    print(f"eta0 {eta:5}  stop at epoch {e:4d}  ||w|| {w:.2f}  test loss {t:.3f}   "
          f"epoch 2000: ||w|| {runs[-1][1]:.2f}  test loss {runs[-1][3]:.3f}")
# eta0 0.001  stop at epoch  556  ||w|| 2.20  test loss 0.147   epoch 2000: ||w|| 3.43  test loss 0.178
# eta0  0.01  stop at epoch   55  ||w|| 2.20  test loss 0.146   epoch 2000: ||w|| 6.56  test loss 0.286
# eta0   0.1  stop at epoch    5  ||w|| 2.25  test loss 0.140   epoch 2000: ||w|| 10.11  test loss 0.437

# %% [markdown]
# Jittering each training car's horsepower by a few units is label-preserving, since a three-horsepower difference does not meaningfully change fuel economy, and it manufactures twenty extra copies of the training set.

# %%
from sklearn.metrics import mean_squared_error
# X_tr, X_te, y_tr, y_te and the pipeline imports carry over from 4.6.1's block
rng = np.random.default_rng(0)
X_aug = np.vstack([X_tr + rng.normal(0, 3.0, X_tr.shape) for _ in range(20)] + [X_tr])
y_aug = np.concatenate([y_tr] * 21)                     # 21 copies of the 29 cars: 609 rows
wild = make_pipeline(PolynomialFeatures(15), StandardScaler(), LinearRegression())
print(round(mean_squared_error(y_te, wild.fit(X_tr, y_tr).predict(X_te)), 1))    # 370469.9
print(round(mean_squared_error(y_te, wild.fit(X_aug, y_aug).predict(X_te)), 1))  # 21175.5

# %% [markdown]
# ## 4.7.1 Regression metrics

# %% [markdown]
# So  compares the model against the mean predictor: 1 is perfect, 0 is no better than always guessing the mean, and negative is worse than that.

# %%
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
X, y = fetch_california_housing(return_X_y=True)   # target is median house value in $100,000s
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
pred = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=2).fit(X_tr, y_tr).predict(X_te)
mse = mean_squared_error(y_te, pred)
print(round(mse, 3), round(np.sqrt(mse), 3),
      round(mean_absolute_error(y_te, pred), 3), round(r2_score(y_te, pred), 3))
# 0.274 0.523 0.339 0.795

# %% [markdown]
# A held-out test R^2 already pays for useless features through worse predictions, so it needs no such correction. The demonstration fits ordinary least squares on the 14,448 training rows, then adds one column of pure noise and refits.

# %%
from sklearn.linear_model import LinearRegression
# X_tr, y_tr carry over from the 4.7.1 regression block
def adj_r2(r2, n, p): return 1 - (1 - r2) * (n - 1) / (n - p - 1)
ols = LinearRegression().fit(X_tr, y_tr)
r2_8 = r2_score(y_tr, ols.predict(X_tr))                     # in-sample R^2 with the 8 real features
X_junk = np.hstack([X_tr, np.random.default_rng(0).normal(size=(len(X_tr), 1))])   # add one pure-noise feature
r2_9 = r2_score(y_tr, LinearRegression().fit(X_junk, y_tr).predict(X_junk))
n = len(y_tr)
print(n, round(r2_8, 6), round(r2_9, 6))                     # 14448 0.611294 0.611298
print(round(adj_r2(r2_8, n, 8), 6), round(adj_r2(r2_9, n, 9), 6))   # 0.611079 0.611055

# %% [markdown]
# ## 4.7.2 The confusion matrix and classification metrics

# %% [markdown]
# the later blocks carry clf, imp, the split, pred, and proba over from it, citing it by number, the same convention 4.8 uses with Example 4-7, and all the classification numbers in this section come from this one model.

# %%
import numpy as np, pandas as pd, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score)
df = sns.load_dataset("titanic")
feats = ["pclass", "age", "fare", "sibsp", "parch"]
died, surv = df[df["survived"] == 0], df[df["survived"] == 1]
imb = pd.concat([died, surv.sample(n=95, random_state=0)]).reset_index(drop=True)  # ~15% positive
X, y = imb[feats], imb["survived"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
imp = SimpleImputer(strategy="median").fit(X_tr)
clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2).fit(imp.transform(X_tr), y_tr)
pred = clf.predict(imp.transform(X_te)); proba = clf.predict_proba(imp.transform(X_te))[:, 1]

tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
print(tp, fp, fn, tn)                           # 7 8 22 157
print(round(accuracy_score(y_te, pred), 3))     # 0.845
print(round(precision_score(y_te, pred), 3))    # 0.467
print(round(recall_score(y_te, pred), 3))       # 0.241
print(round(f1_score(y_te, pred), 3))           # 0.318

# %% [markdown]
# Now hold the forest's sensitivity and specificity fixed and change only how rare survival is.

# %%
# tp, fp, fn, tn carry over from Example 4-6: 7 8 22 157
P, N = tp + fn, tn + fp                                   # actual positives and negatives: 29 and 165
rows = [("sensitivity (recall, TPR)",        tp / P),
        ("specificity (TNR)",                tn / N),
        ("false negative rate (miss rate)",  fn / P),
        ("false positive rate (fall-out)",   fp / N),
        ("precision (PPV)",                  tp / (tp + fp)),
        ("false discovery rate",             fp / (tp + fp)),
        ("negative predictive value (NPV)",  tn / (tn + fn)),
        ("prevalence",                       P / (P + N))]
for name, v in rows:
    print(f"{name:34s} {v:.3f}")

sens, spec = tp / P, tn / N                               # fixed properties of the model; now vary the prevalence
def ppv(prev): return sens * prev / (sens * prev + (1 - spec) * (1 - prev))
def npv(prev): return spec * (1 - prev) / (spec * (1 - prev) + (1 - sens) * prev)
for prev in [29 / 194, 0.05, 0.01]:
    print(f"prevalence {prev:.3f}: PPV {ppv(prev):.3f}  NPV {npv(prev):.3f}")
s, c = 0.99, 0.95                                         # a far better test screening a rare condition
for prev in [0.10, 0.01, 0.001]:
    print(f"sens 0.99 spec 0.95 prevalence {prev:.3f}: PPV {s * prev / (s * prev + (1 - c) * (1 - prev)):.3f}")
# sensitivity (recall, TPR)          0.241
# specificity (TNR)                  0.952
# false negative rate (miss rate)    0.759
# false positive rate (fall-out)     0.048
# precision (PPV)                    0.467
# false discovery rate               0.533
# negative predictive value (NPV)    0.877
# prevalence                         0.149
# prevalence 0.149: PPV 0.467  NPV 0.877
# prevalence 0.050: PPV 0.208  NPV 0.960
# prevalence 0.010: PPV 0.048  NPV 0.992
# sens 0.99 spec 0.95 prevalence 0.100: PPV 0.687
# sens 0.99 spec 0.95 prevalence 0.010: PPV 0.167
# sens 0.99 spec 0.95 prevalence 0.001: PPV 0.019

# %% [markdown]
# ## 4.7.3 Metrics for more than two classes

# %% [markdown]
# Per-class precision, recall, and F1 come from treating each class in turn as the positive one against all the rest, and classification_report prints them in a table along with support, the number of true examples of each class.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
    precision_score, f1_score, top_k_accuracy_score)
dig = load_digits()                       # 1797 handwritten 8-by-8 images, 10 classes (0-9)
# distinct names, so the digits model does not overwrite the Titanic setup the later blocks reuse
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(dig.data, dig.target, test_size=0.3,
                                              random_state=0, stratify=dig.target)
dclf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2).fit(Xd_tr, yd_tr)
dpred = dclf.predict(Xd_te); dproba = dclf.predict_proba(Xd_te)
print(confusion_matrix(yd_te, dpred).shape)           # (10, 10)
print(confusion_matrix(yd_te, dpred)[8])              # the row for true 8s: [ 0  3  0  0  0  0  0  0 49  0]
print(classification_report(yd_te, dpred, digits=3))  # per-class P/R/F1 from 0.942 to 1.000; accuracy 0.978 over 540
for avg in ("macro", "micro", "weighted"):
    print(avg, round(precision_score(yd_te, dpred, average=avg), 3),
          round(f1_score(yd_te, dpred, average=avg), 3))
# macro 0.978 0.978 ; micro 0.978 0.978 ; weighted 0.978 0.978
print("top-1", round(top_k_accuracy_score(yd_te, dproba, k=1), 3),
      "top-3", round(top_k_accuracy_score(yd_te, dproba, k=3), 3))   # top-1 0.978 top-3 0.998

# %% [markdown]
# A small made-up confusion matrix with one rare, poorly predicted class makes macro, weighted, and micro averaging disagree.

# %%
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
cm3 = np.array([[78,  2, 0],                    # true A, the common class: 80 cases
                [ 3, 12, 0],                    # true B: 15 cases
                [ 3,  1, 1]])                   # true C, the rare class: 5 cases
yt3 = np.repeat(np.arange(3), cm3.sum(axis=1))                          # one true label per counted case
yp3 = np.concatenate([np.repeat(np.arange(3), row) for row in cm3])     # and the label the model gave it
print(np.round(f1_score(yt3, yp3, average=None), 3))                    # per-class F1 for A, B, C
for avg in ("macro", "weighted", "micro"):
    print(avg, round(f1_score(yt3, yp3, average=avg), 3))
print("accuracy", accuracy_score(yt3, yp3))
# [0.951 0.8   0.333]
# macro 0.695
# weighted 0.898
# micro 0.91
# accuracy 0.91

# %% [markdown]
# ## 4.7.4 The threshold and its curves

# %% [markdown]
# So $\text{AP} = 1/2 + 1/3 = 0.833$. Negate every score and the ranking reverses, giving an AUC of 0.25.

# %%
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
y4 = np.array([1, 0, 1, 0]); s4 = np.array([0.9, 0.6, 0.4, 0.2])   # the four-case hand example
print(roc_auc_score(y4, s4), round(average_precision_score(y4, s4), 3))   # 0.75 0.833
print(roc_auc_score(y4, -s4))                                           # 0.25: the same ranking, reversed

# %% [markdown]
# The forest's own two areas follow.

# %%
from sklearn.metrics import roc_auc_score, average_precision_score
# proba and y_te carry over from Example 4-6, the section's setup block
print(round(roc_auc_score(y_te, proba), 3))             # 0.744
print(round(average_precision_score(y_te, proba), 3))   # 0.431

# %% [markdown]
# The test split has 29 survivors and 165 others, so 4,785 pairs, and the block counts them all.

# %%
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, log_loss
# proba and y_te carry over from Example 4-6
pos, neg = proba[np.asarray(y_te) == 1], proba[np.asarray(y_te) == 0]
print(len(pos), len(neg), len(pos) * len(neg))                      # 29 165 4785
diff = pos[:, None] - neg[None, :]                                  # every positive against every negative
wins, ties = int((diff > 0).sum()), int((diff == 0).sum())
print(wins, ties, round((wins + 0.5 * ties) / diff.size, 3))        # 3474 173 0.744
print(len(np.unique(proba)))                                        # 86 distinct scores among 194 passengers
U = mannwhitneyu(pos, neg).statistic                                # the Mann-Whitney U of the positive group
print(U, round(U / (len(pos) * len(neg)), 3))                       # 3560.5 0.744
print(round(roc_auc_score(y_te, proba), 3))                         # 0.744
# any order-preserving change of the scores leaves the area alone, however much it changes the probabilities
print(round(roc_auc_score(y_te, proba ** 3), 3), round(roc_auc_score(y_te, 0.5 * proba), 3))   # 0.744 0.744
print(round(log_loss(y_te, proba), 3), round(log_loss(y_te, 0.5 * proba), 3))                   # 0.882 0.907

# %% [markdown]
# maximize F1 for the best balance, fix a floor on recall and take the most precise threshold that clears it, or set a cost ratio and minimize total cost.

# %%
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
# X_tr, y_tr, y_te, proba, SimpleImputer, and RandomForestClassifier carry over from Example 4-6

pipe = make_pipeline(SimpleImputer(strategy="median"),
                     RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2))
oof = cross_val_predict(pipe, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]   # out-of-fold probabilities
prec, rec, thr = precision_recall_curve(y_tr, oof)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
t_f1 = thr[np.argmax(f1s)]                                   # rule 1: maximize F1
ok = np.where(rec[:-1] >= 0.60)[0]
t_rec = thr[ok[np.argmax(prec[:-1][ok])]]                    # rule 2: most precise threshold with recall >= 0.60
def cost(t, p, y):                                           # rule 3: a miss costs five false alarms
    y = np.asarray(y); return int(((p >= t) & (y == 0)).sum() + 5 * ((p < t) & (y == 1)).sum())
t_cost = thr[np.argmin([cost(t, oof, y_tr) for t in thr])]

for name, t in [("max-F1", t_f1), ("recall>=0.60", t_rec), ("min-cost", t_cost), ("default", 0.5)]:
    yp = (proba >= t).astype(int)                            # apply the chosen threshold once, on the test set
    print(f"{name:13s} thr {t:.2f}  test prec {precision_score(y_te, yp):.3f}  rec {recall_score(y_te, yp):.3f}  cost {cost(t, proba, y_te)}")
# max-F1        thr 0.04  test prec 0.222  rec 0.759  cost 112
# recall>=0.60  thr 0.11  test prec 0.271  rec 0.655  cost 101
# min-cost      thr 0.04  test prec 0.222  rec 0.759  cost 112
# default       thr 0.50  test prec 0.467  rec 0.241  cost 118

# %% [markdown]
# The class balance does not appear in the cost rule: 0.5 is the right cutoff whenever the two mistakes cost the same and the probabilities are honest, however rare the positives. With a miss costing five false alarms the cutoff is 1/(1+5) = 1/6, about 0.167.

# %%
t_rule = 1 / (1 + 5)                                         # c_FP / (c_FP + c_FN) with c_FN = 5 c_FP
yp = (proba >= t_rule).astype(int)
print(round(t_rule, 3), round(precision_score(y_te, yp), 3), round(recall_score(y_te, yp), 3), cost(t_rule, proba, y_te))
# 0.167 0.317 0.655 91
# what NOT to report: the lowest cost any threshold reaches on the test set itself, and that threshold
print(min((cost(t, proba, y_te), round(float(t), 3)) for t in np.unique(proba)))
# (82, 0.33)

# %% [markdown]
# With a scorer that charges one unit per false alarm and five per miss it reproduces the cost rule of the block above without the handwritten scan.

# %%
import numpy as np
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, precision_score, recall_score
# X_tr, y_tr, X_te, y_te, SimpleImputer, and RandomForestClassifier carry over from Example 4-6
def total_cost(y_true, y_pred):                           # a false alarm costs 1, a miss costs 5
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return ((y_pred == 1) & (y_true == 0)).sum() + 5 * ((y_pred == 0) & (y_true == 1)).sum()

pipe = make_pipeline(SimpleImputer(strategy="median"),
                     RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2))
tuned = TunedThresholdClassifierCV(pipe, scoring=make_scorer(total_cost, greater_is_better=False),
                                   cv=5, random_state=0).fit(X_tr, y_tr)
yp = tuned.predict(X_te)                                  # predicts with the tuned threshold, not 0.5
print(round(tuned.best_threshold_, 3), round(-tuned.best_score_, 1))                             # 0.036 51.0
print(round(precision_score(y_te, yp), 3), round(recall_score(y_te, yp), 3), total_cost(y_te, yp))   # 0.218 0.759 114
fine = TunedThresholdClassifierCV(pipe, scoring=make_scorer(total_cost, greater_is_better=False),
                                  cv=5, thresholds=1000, random_state=0).fit(X_tr, y_tr)   # a 1000-point grid
print(round(fine.best_threshold_, 3), total_cost(y_te, fine.predict(X_te)))                      # 0.041 112

# %% [markdown]
# ## 4.7.5 Why accuracy lies on imbalanced data

# %% [markdown]
# The trained model is therefore very slightly worse than that constant, useless guess, even though its accuracy looks respectable.

# %%
from sklearn.metrics import accuracy_score
# pred and y_te carry over from Example 4-6, the section's setup block
print(round(max(y_te.mean(), 1 - y_te.mean()), 3))   # 0.851  always-predict-majority accuracy
print(round(accuracy_score(y_te, pred), 3))          # 0.845  the trained model

# %% [markdown]
# ## 4.7.6 Chance-corrected scores, probability scores, and calibration

# %% [markdown]
# Both metrics fall toward zero for a classifier that is merely guessing the common class, and both hand values are exactly what scikit-learn confirms below.

# %%
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, accuracy_score
# pred and y_te carry over from Example 4-6, the section's setup block
print("accuracy:", round(accuracy_score(y_te, pred), 3))          # 0.845
print("MCC:", round(matthews_corrcoef(y_te, pred), 3))            # 0.258
print("Cohen's kappa:", round(cohen_kappa_score(y_te, pred), 3))  # 0.241

# %% [markdown]
# One given $p = 0$ would cost $-\ln 0 = \infty$, so scikit-learn first clips the probability up to machine epsilon, about $2.2 \times 10^{-16}$, and charges $-\ln(2.2 \times 10^{-16}) = 36.04$ nats, while the Brier term is simply 1.

# %%
from sklearn.metrics import log_loss, brier_score_loss
for p in [0.9, 0.1, 0.0]:                                  # three survivors (y = 1) scored at three probabilities
    print(p, round(log_loss([1], [p], labels=[0, 1]), 4), round(brier_score_loss([1], [p]), 4))
# 0.9 0.1054 0.01
# 0.1 2.3026 0.81
# 0.0 36.0437 1.0

# %% [markdown]
# The forest's own numbers follow.

# %%
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss
# clf, imp, X_te, y_te, y_tr, and proba carry over from Example 4-6, the section's setup block
print("log loss:", round(log_loss(y_te, clf.predict_proba(imp.transform(X_te))), 3))  # 0.882
print("Brier score:", round(brier_score_loss(y_te, proba), 3))                        # 0.111
print("survivors given probability 0:", int(((proba == 0) & (y_te.values == 1)).sum()))   # 3
const = np.full(len(y_te), y_tr.mean())                     # always predict the training prevalence, 0.147
print("constant guess:", round(log_loss(y_te, const), 3), round(brier_score_loss(y_te, const), 3))   # 0.422 0.127

# %% [markdown]
# You check this with a reliability curve, which bins the predictions by their predicted probability and plots, for each bin, the predicted probability against the observed fraction of positives; a perfectly calibrated model lies on the diago

# %%
import numpy as np
from sklearn.calibration import calibration_curve
# proba and y_te carry over from Example 4-6, the section's setup block
frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=5, strategy="quantile")
print("predicted prob per bin:", np.round(mean_pred, 2))   # [0.   0.01 0.05 0.18 0.49]
print("actual frac positive :", np.round(frac_pos, 2))     # [0.06 0.04 0.15 0.08 0.41]

# %% [markdown]
# The block calibrates the forest itself both ways, scoring the raw and the calibrated probabilities on the same test set with the same two metrics, and then does the same for a `LinearSVC`, whose raw decision scores are not probabilities at all.

# %%
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
# imp, X_tr, X_te, y_tr, y_te, and proba carry over from Example 4-6, the section's setup block
Xtr_i, Xte_i = imp.transform(X_tr), imp.transform(X_te)
def cal_row(label, p):
    print(f"{label:19s} log loss {log_loss(y_te, p):.3f}  Brier {brier_score_loss(y_te, p):.4f}"
          f"  AUC {roc_auc_score(y_te, p):.3f}  acc {accuracy_score(y_te, (p >= 0.5).astype(int)):.3f}"
          f"  min p {p.min():.3f}")
cal_row("forest, raw", proba)
for method in ["sigmoid", "isotonic"]:                    # calibrate the SAME forest, two ways
    cal = CalibratedClassifierCV(RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2),
                                 method=method, cv=3).fit(Xtr_i, y_tr)
    cal_row(f"forest, {method}", cal.predict_proba(Xte_i)[:, 1])
for method in ["sigmoid", "isotonic"]:                    # give a LinearSVC probabilities it never had
    svm = CalibratedClassifierCV(LinearSVC(random_state=0), method=method, cv=3).fit(Xtr_i, y_tr)
    cal_row(f"LinearSVC, {method}", svm.predict_proba(Xte_i)[:, 1])
# forest, raw         log loss 0.882  Brier 0.1109  AUC 0.744  acc 0.845  min p 0.000
# forest, sigmoid     log loss 0.374  Brier 0.1125  AUC 0.733  acc 0.851  min p 0.097
# forest, isotonic    log loss 0.379  Brier 0.1132  AUC 0.738  acc 0.856  min p 0.017
# LinearSVC, sigmoid  log loss 0.363  Brier 0.1088  AUC 0.745  acc 0.856  min p 0.030
# LinearSVC, isotonic log loss 0.354  Brier 0.1061  AUC 0.762  acc 0.871  min p 0.000

# %% [markdown]
# ## 4.8.2 Grid search

# %% [markdown]
# The block is Example 4-7, and the rest of the section's searches reuse its pipe, X, and y.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

X, y = load_breast_cancer(return_X_y=True)     # 569 tumors, 30 features, malignant vs benign
pipe = make_pipeline(StandardScaler(), SVC())  # standardize, then an RBF-kernel SVM

grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [1e-3, 1e-2, 1e-1, 1]}
gs = GridSearchCV(pipe, grid, cv=5).fit(X, y)
print(gs.best_params_, round(gs.best_score_, 4))   # {'svc__C': 10, 'svc__gamma': 0.01} 0.9789

# %% [markdown]
# ## 4.8.3 Random search

# %% [markdown]
# The payoff is concrete here.

# %%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
# pipe, X, and y carry over from Example 4-7, the grid-search block

dist = {"svc__C": loguniform(1e-1, 1e3), "svc__gamma": loguniform(1e-4, 1e0)}
rs = RandomizedSearchCV(pipe, dist, n_iter=8, cv=5, random_state=0).fit(X, y)
print(round(rs.best_score_, 4))    # 0.9789  with only 40 fits
print({k: float(round(v, 3)) for k, v in rs.best_params_.items()})
# {'svc__C': 4.95, 'svc__gamma': 0.038}

# %% [markdown]
# The block draws 10,000 values each way and counts them per decade.

# %%
import numpy as np
from scipy.stats import loguniform
rng = np.random.default_rng(0)
N = 10_000
lin = rng.uniform(0.1, 1000, N)                          # uniform in the raw value
log_np = 10 ** rng.uniform(-1, 3, N)                     # uniform in log10, written by hand
log_sp = loguniform(0.1, 1000).rvs(N, random_state=0)    # the same law, from SciPy
edges = [0.1, 1, 10, 100, 1000]                          # the four decades
for name, v in [("uniform", lin), ("10**uniform", log_np), ("loguniform", log_sp)]:
    counts, _ = np.histogram(v, bins=edges)              # draws per decade
    print(f"{name:12s}", counts, "median", round(float(np.median(v)), 2))
# uniform      [  11   76  944 8969] median 500.86
# 10**uniform  [2369 2574 2522 2535] median 10.59
# loguniform   [2550 2514 2453 2483] median 9.42

# %% [markdown]
# The block below reruns the random search of this section with the same eight trials and the same seed, changing only the two distributions from `loguniform` to `uniform`.

# %%
from scipy.stats import uniform
# pipe, X, and y carry over from Example 4-7; SciPy's uniform(loc, scale) covers [loc, loc + scale]
dist_lin = {"svc__C": uniform(1e-1, 1e3 - 1e-1), "svc__gamma": uniform(1e-4, 1e0 - 1e-4)}
rs_lin = RandomizedSearchCV(pipe, dist_lin, n_iter=8, cv=5, random_state=0).fit(X, y)
print(round(rs_lin.best_score_, 4), {k: float(round(v, 3)) for k, v in rs_lin.best_params_.items()})
# 0.9525 {'svc__C': 71.129, 'svc__gamma': 0.087}
gammas = np.array(rs_lin.cv_results_["param_svc__gamma"], float)
print(int(np.sum(gammas > 0.1)), round(float(rs_lin.cv_results_["mean_test_score"].mean()), 4))   # 7 0.7641
g_log = np.array(rs.cv_results_["param_svc__gamma"], float)   # rs is the log-uniform search of this section
print(int(np.sum(g_log > 0.1)), round(float(rs.cv_results_["mean_test_score"].mean()), 4))        # 2 0.8996
print(int(np.bincount(y).max()), len(y), round(float(np.bincount(y).max() / len(y)), 3))         # 357 569 0.627

# %% [markdown]
# ## 4.8.4 Bayesian optimization

# %% [markdown]
# You give Optuna an objective function that returns the score for a trial's suggested hyperparameters.

# %%
import optuna
from sklearn.model_selection import cross_val_score
optuna.logging.set_verbosity(optuna.logging.WARNING)   # quiet the per-trial logging
# pipe, X, and y carry over from Example 4-7, the grid-search block

def objective(trial):
    C = trial.suggest_float("C", 1e-1, 1e3, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 1e0, log=True)
    model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma))
    return cross_val_score(model, X, y, cv=5).mean()

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=25)   # 10 random startup trials, then 15 guided ones
print(round(study.best_value, 4))                                     # 0.9807
print({k: float(round(v, 3)) for k, v in study.best_params.items()})  # {'C': 7.285, 'gamma': 0.029}
print("best found at trial", study.best_trial.number)                 # best found at trial 22
print(round(max(t.value for t in study.trials[:10]), 4))              # 0.9789  best of the ten random startup trials
print(optuna.__version__)                                             # 5.0.0  the installed version

# %% [markdown]
# ## 4.8.5 Successive halving and Hyperband

# %% [markdown]
# Scikit-learn exposes it as HalvingRandomSearchCV, behind an experimental import that must be enabled first.

# %%
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  enables the import below
from sklearn.model_selection import HalvingRandomSearchCV
# pipe, X, y (Example 4-7) and dist (the random-search block) carry over

hs = HalvingRandomSearchCV(pipe, dist, factor=3, cv=5, random_state=0).fit(X, y)
print(round(hs.best_score_, 4))                 # 0.9796
print(hs.n_candidates_, list(hs.n_resources_))  # [28, 10, 4, 2] [20, 60, 180, 540]

# %% [markdown]
# Hyperband runs several brackets of successive halving, from the most aggressive to plain random search; the block computes the bracket schedule of Li et al. (2018, Algorithm 1) for the halving run's numbers.

# %%
import math

def hyperband_brackets(R, eta=3, unit=20):
    """Hyperband's bracket schedule (Li et al., 2018, Algorithm 1); R is the largest resource in units of the smallest."""
    s_max = 0
    while eta ** (s_max + 1) <= R:                        # s_max = floor(log_eta R), counted in integers
        s_max += 1
    B = (s_max + 1) * R                                   # the budget every bracket is given
    total = 0
    for s in range(s_max, -1, -1):                        # from the most aggressive bracket down to plain random search
        n = math.ceil(B / R * eta ** s / (s + 1))         # candidates the bracket starts with
        r = R // eta ** s                                 # their first resource in units (exact here: R is a power of eta)
        rungs = [(n // eta ** i, r * eta ** i) for i in range(s + 1)]   # (candidates, resource) at each rung
        spent = sum(k * res for k, res in rungs)          # units of resource the bracket uses
        total += spent
        print(f"s={s}", [(k, res * unit) for k, res in rungs], "budget", spent, round(spent / R, 2))
    print("all brackets", total, round(total / R, 2))

hyperband_brackets(27)          # the halving run's numbers: 20 rows smallest, 540 rows largest, factor 3
# s=3 [(27, 20), (9, 60), (3, 180), (1, 540)] budget 108 4.0
# s=2 [(12, 60), (4, 180), (1, 540)] budget 99 3.67
# s=1 [(6, 180), (2, 540)] budget 108 4.0
# s=0 [(4, 540)] budget 108 4.0
# all brackets 423 15.67

# %% [markdown]
# Each Hyperband bracket is one successive halving run, so `HalvingRandomSearchCV` with the bracket's `n_candidates` and `min_resources` runs the four brackets as live successive-halving searches; a plain random search over as many candidates on all the data is the yardstick.

# %%
# pipe, X, y (Example 4-7), dist, RandomizedSearchCV, and HalvingRandomSearchCV carry over from 4.8.2 to 4.8.5
brackets = []
for s, n, rows in [(3, 27, 20), (2, 12, 60), (1, 6, 180), (0, 4, 540)]:   # Hyperband's four brackets
    hb = HalvingRandomSearchCV(pipe, dist, n_candidates=n, min_resources=rows, factor=3,
                               cv=5, random_state=s).fit(X, y)          # one successive halving run
    rows_used = sum(k * r for k, r in zip(hb.n_candidates_, hb.n_resources_))   # rows over every rung
    print(f"s={s}: candidates {hb.n_candidates_} rows {hb.n_resources_} "
          f"row cost {rows_used:5d}  best {hb.best_score_:.4f}")
    brackets.append(hb)
winner = max(brackets, key=lambda b: b.best_score_)     # the best final-rung score over all brackets
print(round(winner.best_score_, 4), {k: float(round(v, 3)) for k, v in winner.best_params_.items()})
n_all = sum(b.n_candidates_[0] for b in brackets)       # candidates sampled across the brackets
print(5 * sum(sum(b.n_candidates_) for b in brackets), "fits;", n_all, "candidates")
full = RandomizedSearchCV(pipe, dist, n_iter=n_all, cv=5, random_state=0).fit(X, y)   # all on all rows
print(round(full.best_score_, 4), "row cost", n_all * len(y))
# s=3: candidates [27, 9, 3, 1] rows [20, 60, 180, 540] row cost  2160  best 0.9759
# s=2: candidates [12, 4, 2] rows [60, 180, 540] row cost  2520  best 0.9722
# s=1: candidates [6, 2] rows [180, 540] row cost  2160  best 0.9778
# s=0: candidates [4] rows [540] row cost  2160  best 0.9778
# 0.9778 {'svc__C': 3.864, 'svc__gamma': 0.014}
# 350 fits; 49 candidates
# 0.9807 row cost 27881

# %% [markdown]
# ## 4.8.6 Nested cross-validation for honest model selection

# %% [markdown]
# The gap shows up on the breast-cancer data with a larger grid, where the selection bias is easier to see.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
pipe = make_pipeline(StandardScaler(), SVC())
grid = {"svc__C": np.logspace(-2, 3, 8), "svc__gamma": np.logspace(-4, 1, 8)}

inner = StratifiedKFold(5, shuffle=True, random_state=0)
outer = StratifiedKFold(5, shuffle=True, random_state=0)

tuned = GridSearchCV(pipe, grid, cv=inner).fit(X, y)
print(round(tuned.best_score_, 4))                              # 0.9807  optimistic
scores = cross_val_score(GridSearchCV(pipe, grid, cv=inner), X, y, cv=outer)
print(round(scores.mean(), 4))                                  # 0.9754  honest

# %% [markdown]
# ## 4.8.7 The one-standard-error rule for near-tied settings

# %% [markdown]
# `LassoCV` tries a path of 100 values of alpha, spaced on the log scale of 4.8.3, and stores the ten-fold errors of every one in `mse_path_`.

# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LassoCV, Lasso
Xd, yd = load_diabetes(return_X_y=True)
lcv = LassoCV(cv=10, random_state=0).fit(Xd, yd)      # mse_path_: one row per alpha, one column per fold
mean = lcv.mse_path_.mean(axis=1)                      # the curve of cross-validated MSE against alpha
se = lcv.mse_path_.std(axis=1, ddof=1) / np.sqrt(lcv.mse_path_.shape[1])   # standard error of each mean
i_min = int(np.argmin(mean))
threshold = mean[i_min] + se[i_min]                    # the horizontal line one standard error above the minimum
alpha_1se = float(lcv.alphas_[mean <= threshold].max())   # the most regularized setting under the line
print(len(lcv.alphas_), round(float(lcv.alphas_[0] / lcv.alphas_[1]), 3))     # 100 1.072
print(round(float(lcv.alpha_), 4), round(float(mean[i_min]), 1), round(float(se[i_min]), 1), round(float(threshold), 1))
# 0.0571 2987.2 216.0 3203.2
print(round(alpha_1se, 3), round(float(mean[lcv.alphas_ == alpha_1se][0]), 1), int(np.sum(mean <= threshold)))
# 0.403 3197.5 76
print(np.round(lcv.mse_path_[i_min]).astype(int))                     # the ten fold errors at the minimum
# [2688 2816 3341 2894 3515 2819 3538 2270 4183 1808]
lcv5 = LassoCV(cv=5, random_state=0).fit(Xd, yd)
mean5 = lcv5.mse_path_.mean(axis=1)
se5 = lcv5.mse_path_.std(axis=1, ddof=1) / np.sqrt(lcv5.mse_path_.shape[1])
i5 = int(np.argmin(mean5))
thr5 = mean5[i5] + se5[i5]
a5 = float(lcv5.alphas_[mean5 <= thr5].max())
print(round(float(lcv5.alpha_), 4), round(float(se5[i5]), 1), round(a5, 3), int(np.sum(mean5 <= thr5)))
# 0.0038 70.7 0.187 65
for a in [lcv.alpha_, alpha_1se]:
    coef = Lasso(alpha=a).fit(Xd, yd).coef_
    print(round(float(a), 3), int(np.sum(coef != 0)), np.round(coef, 1))
# 0.057 7 [  -0.  -188.6  521.2  292.4  -92.8   -0.  -220.9    0.   508.1   50.2]
# 0.403 4 [  0.   -0.  487.5 162.5  -0.   -0.  -84.8   0.  423.    0. ]

# %% [markdown]
# ## 4.8.8 Model selection without a validation set: AIC, AICc, BIC, and Mallows' Cp

# %% [markdown]
# The block repeats this for degrees 1 through 12, in the well-conditioned Legendre basis of Example 4-4 so that every degree is an exact fit, and prints the test MSE on the 16 held-out cars beside each criterion.

# %%
import numpy as np, seaborn as sns, statsmodels.api as sm
from statsmodels.tools.eval_measures import aicc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
mpg = sns.load_dataset("mpg").dropna(subset=["horsepower", "mpg"])
sub = mpg.sample(n=45, random_state=1)                    # the chapter's same 45 cars
Xc = sub[["horsepower"]].to_numpy(float); yc = sub["mpg"].to_numpy(float)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.35, random_state=0)
lo, hi = 46.0, 230.0
def legendre_features(Xcol, D):                           # the well-conditioned basis of Example 4-4
    t = 2 * (Xcol.ravel() - lo) / (hi - lo) - 1
    return np.polynomial.legendre.legvander(t, D)

n = len(yc_tr)                                                      # 29 training cars
full = sm.OLS(yc_tr, legendre_features(Xc_tr, 12)).fit()
sigma2 = full.ssr / (n - 13)                                        # error variance from the fullest model, for Cp
table = []
for D in range(1, 13):
    res = sm.OLS(yc_tr, legendre_features(Xc_tr, D)).fit()          # degree D fits k = D + 1 coefficients
    k = D + 1
    cp = (res.ssr + 2 * D * sigma2) / n                             # Mallows' Cp with d = D predictors
    test = mean_squared_error(yc_te, legendre_features(Xc_te, D) @ res.params)
    table.append((D, res.aic, aicc(res.llf, n, k), res.bic, cp))
    print(f"{D:2d} RSS {res.ssr:7.2f} AIC {res.aic:8.3f} AICc {aicc(res.llf, n, k):8.3f} BIC {res.bic:8.3f} Cp {cp:6.3f} test MSE {test:10.4g}")
T = np.array(table)
for j, name in [(1, "AIC"), (2, "AICc"), (3, "BIC"), (4, "Cp")]:
    print(name, "picks degree", int(T[np.argmin(T[:, j]), 0]))
print(round(sigma2, 3), round(np.log(n), 3))                       # 8.106 3.367
#  1 RSS  568.21 AIC  172.579 AICc  173.041 BIC  175.314 Cp 20.153 test MSE       31.2
#  2 RSS  282.90 AIC  154.355 AICc  155.315 BIC  158.456 Cp 10.873 test MSE       18.9
#  3 RSS  280.53 AIC  156.111 AICc  157.778 BIC  161.580 Cp 11.351 test MSE      17.03
#  4 RSS  278.98 AIC  157.950 AICc  160.559 BIC  164.787 Cp 11.856 test MSE      15.63
#  5 RSS  278.38 AIC  159.887 AICc  163.705 BIC  168.091 Cp 12.394 test MSE      18.77
#  6 RSS  268.69 AIC  160.860 AICc  166.193 BIC  170.431 Cp 12.619 test MSE      129.1
#  7 RSS  211.87 AIC  155.971 AICc  163.171 BIC  166.909 Cp 11.219 test MSE       2675
#  8 RSS  153.37 AIC  148.599 AICc  158.073 BIC  160.905 Cp  9.761 test MSE       8320
#  9 RSS  153.34 AIC  150.595 AICc  162.817 BIC  164.268 Cp 10.319 test MSE       6819
# 10 RSS  146.31 AIC  151.232 AICc  166.762 BIC  166.273 Cp 10.635 test MSE  2.867e+05
# 11 RSS  129.95 AIC  149.795 AICc  169.295 BIC  166.202 Cp 10.630 test MSE   1.41e+07
# 12 RSS  129.69 AIC  151.736 AICc  176.003 BIC  169.511 Cp 11.180 test MSE  4.265e+07
# AIC picks degree 8
# AICc picks degree 2
# BIC picks degree 2
# Cp picks degree 8
# 8.106 3.367

# %% [markdown]
# `LassoLarsIC` computes the Lasso solution path with the LARS algorithm, scores each candidate alpha on the path by AIC or BIC, and keeps the best.

# %%
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LassoLarsIC
Xd, yd = load_diabetes(return_X_y=True)
for crit in ["aic", "bic"]:
    m = LassoLarsIC(criterion=crit).fit(Xd, yd)       # noise variance estimated from an OLS fit; no cross-validation
    print(crit, round(float(m.alpha_), 3), int(np.sum(m.coef_ != 0)), len(m.alphas_), round(float(m.noise_variance_), 1))
    print([n for n, c in zip(load_diabetes().feature_names, m.coef_) if c != 0])
# aic 0.045 7 13 2932.7
# ['sex', 'bmi', 'bp', 's1', 's3', 's5', 's6']
# bic 0.045 7 13 2932.7
# ['sex', 'bmi', 'bp', 's1', 's3', 's5', 's6']

# %% [markdown]
# ## 4.9 Always beat a common-sense baseline

# %% [markdown]
# the 8-by-8 handwritten digits, binarized to detect a single digit (the digit 3) against all others so the positive class is only about a tenth of the data, and the diabetes dataset, whose target is a measure of disease progression one year aft

# %%
import numpy as np
from sklearn.datasets import load_digits, load_diabetes
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score

# classification: detect the digit 3 among the handwritten digits, one class vs rest
digits = load_digits()
Xc, yc = digits.data, (digits.target == 3).astype(int)
print(round(yc.mean(), 3))                                                                            # 0.102
print(round(cross_val_score(DummyClassifier(strategy="most_frequent"), Xc, yc, cv=5).mean(), 3))      # 0.898
print(round(cross_val_score(LogisticRegression(max_iter=5000), Xc, yc, cv=5).mean(), 3))              # 0.973

# regression: predict diabetes progression from ten baseline measurements
Xr, yr = load_diabetes(return_X_y=True)
print(round(cross_val_score(DummyRegressor(strategy="mean"), Xr, yr, cv=5, scoring="r2").mean(), 3))  # -0.028
print(round(cross_val_score(LinearRegression(), Xr, yr, cv=5, scoring="r2").mean(), 3))               # 0.482

# %% [markdown]
# ## 4.9.1 Is the gap real? Testing model against baseline

# %% [markdown]
# it shuffles the labels many times, refits the model on each shuffle, and records how often the shuffled data reaches the real score, returning a p-value.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, permutation_test_score

digits = load_digits()
Xc, yc = digits.data, (digits.target == 3).astype(int)
clf = LogisticRegression(max_iter=5000)

scores = cross_val_score(clf, Xc, yc, cv=5)
print(np.round(scores, 3))                                     # [0.992 0.939 0.986 0.986 0.961]
print("%.3f +/- %.3f" % (scores.mean(), scores.std()))         # 0.973 +/- 0.020

score, perm_scores, pval = permutation_test_score(
    clf, Xc, yc, cv=5, n_permutations=200, random_state=0, scoring="accuracy")
print("%.3f  %.3f  %.3f  %.4f" % (score, perm_scores.mean(), perm_scores.max(), pval))  # 0.973  0.896  0.899  0.0050

# %% [markdown]
# The block confirms the Wilson arithmetic for four of the forest's metrics (Example 4-6 objects carry over).

# %%
from statsmodels.stats.proportion import proportion_confint
# tp, fp, fn, tn carry over from Example 4-6: 7 8 22 157
for name, k, n in [("recall", tp, tp + fn), ("accuracy", tp + tn, tp + fp + fn + tn),
                   ("majority baseline", tn + fp, tp + fp + fn + tn), ("precision", tp, tp + fp)]:
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    print(f"{name:18s} {k:3d}/{n:3d} = {k/n:.3f}   95% Wilson [{lo:.3f}, {hi:.3f}]")
# recall               7/ 29 = 0.241   95% Wilson [0.122, 0.421]
# accuracy           164/194 = 0.845   95% Wilson [0.788, 0.889]
# majority baseline  165/194 = 0.851   95% Wilson [0.794, 0.894]
# precision            7/ 15 = 0.467   95% Wilson [0.248, 0.699]

# %% [markdown]
# The block bootstraps the AUC and, on the same resamples, the gap between the forest's accuracy and the majority baseline's, then repeats the AUC interval with `scipy.stats.bootstrap`.

# %%
from sklearn.metrics import roc_auc_score
from scipy import stats
# proba, pred, and y_te carry over from Example 4-6
y_arr = y_te.to_numpy(); n = len(y_arr)
rng = np.random.default_rng(0)
aucs, gaps = [], []
for _ in range(2000):
    idx = rng.integers(0, n, n)                   # one resample: n row indices drawn with replacement
    y_b = y_arr[idx]                              # labels and scores travel together, row by row
    aucs.append(roc_auc_score(y_b, proba[idx]))
    gaps.append((pred[idx] == y_b).mean() - (y_b == 0).mean())   # forest accuracy minus majority accuracy, same rows
print(round(roc_auc_score(y_arr, proba), 3), np.round(np.percentile(aucs, [2.5, 97.5]), 3), round(float(np.std(aucs, ddof=1)), 3))
# 0.744 [0.631 0.84 ] 0.054
print(np.round(np.percentile(gaps, [2.5, 97.5]), 3), round(float(np.mean(np.array(gaps) >= 0)), 3))
# [-0.046  0.036] 0.46
res = stats.bootstrap((y_arr, proba), roc_auc_score, paired=True, n_resamples=2000, random_state=0, method="percentile")
print(np.round([res.confidence_interval.low, res.confidence_interval.high], 3))   # [0.634 0.848]
bca = stats.bootstrap((y_arr, proba), roc_auc_score, paired=True, n_resamples=2000, random_state=0)   # default method: BCa
print(np.round([bca.confidence_interval.low, bca.confidence_interval.high], 3))   # [0.623 0.839]

# %% [markdown]
# A t interval for the mean squared error can be square rooted into an interval for the RMSE, as the next block does for the diabetes linear model of 4.9.

# %%
import numpy as np
from scipy import stats
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
Xr, yr = load_diabetes(return_X_y=True)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.3, random_state=0)
sq = (LinearRegression().fit(Xr_tr, yr_tr).predict(Xr_te) - yr_te) ** 2     # one squared error per test row
lo, hi = stats.t.interval(0.95, len(sq) - 1, loc=sq.mean(), scale=stats.sem(sq))   # t interval for the mean squared error
print(len(sq), round(float(np.sqrt(sq.mean())), 1), round(float(np.sqrt(lo)), 1), round(float(np.sqrt(hi)), 1))   # 133 55.7 48.2 62.2

# %% [markdown]
# Two worked cases on one split of the breast-cancer data show McNemar's test at both ends.

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from statsmodels.stats.contingency_tables import mcnemar
Xb, yb = load_breast_cancer(return_X_y=True)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.3, random_state=0, stratify=yb)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xb_tr, yb_tr)
rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2).fit(Xb_tr, yb_tr)
stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xb_tr, yb_tr)   # a one-split tree, deliberately weak

def disagreement_table(m1, m2):
    r1, r2 = m1.predict(Xb_te) == yb_te, m2.predict(Xb_te) == yb_te       # which test rows each model got right
    return np.array([[np.sum(r1 & r2), np.sum(r1 & ~r2)], [np.sum(~r1 & r2), np.sum(~r1 & ~r2)]])

for name, m2 in [("random forest", rf), ("stump", stump)]:
    T = disagreement_table(lr, m2)
    b, c = T[0, 1], T[1, 0]                                                # the two disagreement counts
    chi2, exact = mcnemar(T, exact=False, correction=True), mcnemar(T, exact=True)
    print(f"logistic {lr.score(Xb_te, yb_te):.3f} vs {name} {m2.score(Xb_te, yb_te):.3f}  table {T.tolist()}  b={b} c={c}")
    print(f"   chi2 {chi2.statistic:.3f} p {chi2.pvalue:.4f}   exact p {exact.pvalue:.4f}")
# logistic 0.959 vs random forest 0.953  table [[159, 5], [4, 3]]  b=5 c=4
#    chi2 0.000 p 1.0000   exact p 1.0000
# logistic 0.959 vs stump 0.889  table [[150, 14], [2, 5]]  b=14 c=2
#    chi2 7.562 p 0.0060   exact p 0.0042

# %% [markdown]
# Example 4-8 implements the 5x2cv paired t test in a dozen lines, then runs the ten-fold paired t test that Dietterich warned about, for contrast.

# %%
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

def paired_t_5x2cv(model_a, model_b, X, y, seed_shift=0):
    """Dietterich's 5x2cv paired t test on accuracy: five repetitions of two-fold CV with different shuffles."""
    s2, p_first = [], None
    for rep in range(5):
        halves = list(StratifiedKFold(2, shuffle=True, random_state=rep + seed_shift).split(X, y))
        p = []
        for train, test in halves:                          # each half trains once and is tested on once
            acc_a = clone(model_a).fit(X[train], y[train]).score(X[test], y[test])
            acc_b = clone(model_b).fit(X[train], y[train]).score(X[test], y[test])
            p.append(acc_a - acc_b)
        if p_first is None:
            p_first = p[0]                                   # the numerator uses only the very first difference
        p_bar = (p[0] + p[1]) / 2
        s2.append((p[0] - p_bar) ** 2 + (p[1] - p_bar) ** 2)
    t = p_first / np.sqrt(np.mean(s2))
    return t, 2 * stats.t.sf(abs(t), df=5), p_first

cv10 = StratifiedKFold(10, shuffle=True, random_state=0)
for name, m2 in [("random forest", rf), ("stump", stump)]:
    t, p, p11 = paired_t_5x2cv(lr, m2, Xb, yb)
    a1, a2 = cross_val_score(lr, Xb, yb, cv=cv10), cross_val_score(m2, Xb, yb, cv=cv10)
    naive = stats.ttest_rel(a1, a2)                          # the paired t test over ten folds, for contrast
    print(f"logistic vs {name}: 5x2cv t {t:.3f} p {p:.4f} p11 {p11:.4f} | 10-fold means {a1.mean():.4f} {a2.mean():.4f}, paired t {naive.statistic:.3f} p {naive.pvalue:.4f}")
# logistic vs random forest: 5x2cv t 3.169 p 0.0249 p11 0.0316 | 10-fold means 0.9772 0.9631, paired t 2.233 p 0.0525
# logistic vs stump: 5x2cv t 7.014 p 0.0009 p11 0.0807 | 10-fold means 0.9772 0.8875, paired t 10.029 p 0.0000
for name, m2 in [("random forest", rf), ("stump", stump)]:
    t, p, p11 = paired_t_5x2cv(lr, m2, Xb, yb, seed_shift=10)   # the same five shuffles, shifted by 10
    print(f"seed_shift=10 vs {name}: t {t:.3f} p {p:.4f} p11 {p11:.4f}")
# seed_shift=10 vs random forest: t 0.000 p 1.0000 p11 0.0000
# seed_shift=10 vs stump: t 3.596 p 0.0156 p11 0.0561

# %% [markdown]
# ## 4.9.2 Error analysis: read your model's mistakes

# %% [markdown]
# On the full ten-class digits problem, this time with a logistic regression, it turns a single error rate into a concrete list of which digits this model actually mixes up, extracted programmatically rather than eyeballed.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

digits = load_digits()
X, y = digits.data, digits.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
clf = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
pred = clf.predict(Xte)
print(round((pred == yte).mean(), 3))                       # 0.961

cm = confusion_matrix(yte, pred)
pairs = [(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j and cm[i, j] > 0]
for count, i, j in sorted(pairs, reverse=True)[:3]:
    print(f"true {i} predicted {j}: {count}")               # true 8 predicted 1: 4
                                                            # true 7 predicted 9: 2
                                                            # true 9 predicted 5: 1
print("misclassified per true digit:",
      [int(cm[i].sum() - cm[i, i]) for i in range(10)])     # [1, 1, 1, 2, 2, 3, 1, 3, 6, 1]

# %% [markdown]
# ## 4.9.3 Behavioral tests: perturbation, invariance, and directional expectation

# %% [markdown]
# Example 4-9 runs the directional and perturbation tests on the California housing data of 4.3.6 with a random forest, then shows the standard repair for a directional failure.

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
housing = fetch_california_housing(as_frame=True)
Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(housing.data, housing.target, test_size=0.2, random_state=0)
forest = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=0, n_jobs=2).fit(Xh_tr, yh_tr)
base = forest.predict(Xh_te)
print(len(Xh_te), round(forest.score(Xh_te, yh_te), 3))                       # 4128 0.793

# directional expectation: raising median income by 10 percent should never lower the predicted value
X_up = Xh_te.copy(); X_up["MedInc"] *= 1.10
delta = forest.predict(X_up) - base
fell = delta < -1e-9        # a drop beyond rounding; rows whose prediction does not move are counted next
print(int(np.sum(np.abs(delta) <= 1e-9)))                                     # 65  rows whose prediction does not change beyond rounding
print(int(np.sum(fell)), round(float(np.mean(fell)), 3), int(np.sum(delta < -0.01 * base)),
      round(float(delta.mean()), 3), round(float(delta.min()), 3))             # 514 0.125 288 0.138 -0.899

# perturbation: Gaussian noise of 1 percent (then 5 percent) of each feature's training standard deviation
rng = np.random.default_rng(0)
sd = Xh_tr.std().to_numpy()
for frac in [0.01, 0.05]:
    noisy = forest.predict(Xh_te + rng.standard_normal(Xh_te.shape) * sd * frac)
    moved = np.abs(noisy - base) / base                                        # relative change of each prediction
    print(frac, int(np.sum(moved > 0.05)), round(float(np.mean(moved > 0.05)), 3), int(np.sum(moved > 0.10)),
          round(float(np.median(moved)), 4))
# 0.01 551 0.133 150 0.0131
# 0.05 2250 0.545 1313 0.0577

# the fix for a directional failure: a monotonic constraint on MedInc (feature 0), the other seven unconstrained
for cst in [None, [1, 0, 0, 0, 0, 0, 0, 0]]:
    hgb = HistGradientBoostingRegressor(monotonic_cst=cst, random_state=0).fit(Xh_tr, yh_tr)
    d = hgb.predict(X_up) - hgb.predict(Xh_te)
    print(cst is not None, round(hgb.score(Xh_te, yh_te), 3), int(np.sum(d < 0)))
# False 0.837 330
# True 0.829 0

# %% [markdown]
# The block takes the logistic regression of 4.9.2, shifts every test image one pixel right and then one pixel left, zeroing the edge column that scrolls in, and counts changed predictions.

# %%
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
digits = load_digits()
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(digits.data, digits.target, test_size=0.3, random_state=0, stratify=digits.target)
clf = LogisticRegression(max_iter=5000).fit(Xd_tr, yd_tr)             # the classifier of 4.9.2
before = clf.predict(Xd_te)
imgs = Xd_te.reshape(-1, 8, 8)
print(np.round((imgs > 0).mean(axis=(0, 2)), 2))                        # share of inked pixels in each of the 8 rows
# [0.48 0.56 0.51 0.52 0.52 0.47 0.53 0.49]
right = np.roll(imgs, 1, axis=2); right[:, :, 0] = 0                    # every image one pixel to the right
left = np.roll(imgs, -1, axis=2); left[:, :, 7] = 0                     # every image one pixel to the left
for name, shifted, edge in [("right", right, imgs[:, :, 7]), ("left", left, imgs[:, :, 0])]:
    after = clf.predict(shifted.reshape(len(shifted), -1))
    print(name, int(np.sum(after != before)), round(float(np.mean(after != before)), 3),
          round(float((before == yd_te).mean()), 3), round(float((after == yd_te).mean()), 3),
          round(float((edge > 0).mean()), 4))                            # share of nonzero pixels in the discarded edge
# right 292 0.541 0.961 0.463 0.0292
# left 297 0.55 0.961 0.452 0.0025

# %% [markdown]
# ## 4.10 The no-free-lunch theorem

# %% [markdown]
# The averages over all 256 targets are the theorem's quantity; the averages over the six targets that copy or negate a single input bit are the structured case.

# %%
import itertools
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

inputs = np.array(list(itertools.product([0, 1], repeat=3)))   # the 8 points of {0,1}^3
seen, unseen = [0, 1, 2, 3, 4], [5, 6, 7]                       # 5 training inputs, 3 off-training-set inputs
learners = {"decision tree": lambda: DecisionTreeClassifier(random_state=0),
            "1-nearest neighbor": lambda: KNeighborsClassifier(n_neighbors=1)}
all_avg = {name: 0.0 for name in learners}
bit_avg = {name: 0.0 for name in learners}
for target in itertools.product([0, 1], repeat=8):             # every one of the 2^8 = 256 target functions
    f = np.array(target)
    one_bit = any(np.array_equal(f, inputs[:, j]) or np.array_equal(f, 1 - inputs[:, j]) for j in range(3))
    for name, make in learners.items():
        if len(set(f[seen])) == 1:                             # a one-class training set: predict that class
            pred = np.full(3, f[seen][0])
        else:
            pred = make().fit(inputs[seen], f[seen]).predict(inputs[unseen])
        acc = np.mean(pred == f[unseen])                       # off-training-set accuracy on this target
        all_avg[name] += acc / 256
        if one_bit:                                            # the 6 targets that copy or negate one input bit
            bit_avg[name] += acc / 6
for name in learners:
    print(f"{name:19s} all 256 targets: {all_avg[name]:.3f}   the 6 single-bit targets: {bit_avg[name]:.3f}")
# decision tree       all 256 targets: 0.500   the 6 single-bit targets: 1.000
# 1-nearest neighbor  all 256 targets: 0.500   the 6 single-bit targets: 0.667

# %% [markdown]
# To show inductive bias at work we need to know the true generating rule, so we build one target as an exact linear function of the inputs and the other from a known interaction and a sine wave; only then can we be certain which model's assu

# %%
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
# a KNOWN linear rule: the target is an exact weighted sum of the inputs plus small noise
X1 = rng.normal(0, 1, (600, 5))
y1 = X1 @ np.array([3, -2, 1, 0, 0]) + rng.normal(0, 0.5, 600)
# a KNOWN nonlinear rule: a sign interaction (adds 2 when x0 and x1 share a sign) and a sine wave
X2 = rng.normal(0, 1, (600, 5))
y2 = (X2[:, 0] * X2[:, 1] > 0) * 2.0 + np.sin(3 * X2[:, 2]) + rng.normal(0, 0.2, 600)

for name, X, y in [("linear", X1, y1), ("nonlinear", X2, y2)]:
    lin = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2").mean()
    rf = cross_val_score(RandomForestRegressor(random_state=0), X, y, cv=5, scoring="r2").mean()
    print(name, round(lin, 3), round(rf, 3))
# linear      0.981   0.945
# nonlinear  -0.017   0.658

# %% [markdown]
# ## 4.11 Interview questions and coding exercises
# Each coding exercise starter block runs as given; the full solutions are in Appendix H.

# %% [markdown]
# Coding exercise 1, Supervised versus unsupervised on the wine data: the starter block.
# %%
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score

X, y = load_wine(return_X_y=True)               # 178 wines, 13 measurements, 3 cultivars

def supervised_accuracy(X, y):
    """Stratified 70/30 split, StandardScaler + LogisticRegression, return the test accuracy."""
    ...

def clustering_scores(X, y):
    """Return the adjusted Rand score of 3-cluster KMeans on raw X and on standardized X."""
    ...

print(X.shape, np.round(X.max(0) - X.min(0), 1).max(), supervised_accuracy(X, y))

# %% [markdown]
# Coding exercise 2, Empirical risk against true risk on another 45 cars: the starter block.
# %%
import numpy as np, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

mpg = sns.load_dataset("mpg").dropna(subset=["horsepower", "mpg"])
sub = mpg.sample(n=45, random_state=2)          # a different 45 cars from the chapter's random_state=1
X = sub[["horsepower"]].to_numpy(float); y = sub["mpg"].to_numpy(float)

def train_and_test_mse(X, y, degree):
    """Split 65/35 with random_state=0, fit the degree-d polynomial pipeline, return (train MSE, test MSE)."""
    ...

print(sub.shape, train_and_test_mse(X, y, 1))

# %% [markdown]
# Coding exercise 3, Bias and variance against the amount of data: the starter block.
# %%
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

true_f = lambda x: np.sin(1.5 * x)              # the chapter's known truth; noise sd 0.3 on x in [-3, 3]
x_test = np.linspace(-3, 3, 60).reshape(-1, 1)  # 60 fixed test points

def bias2_and_variance(n, degree, reps=200, seed=0):
    """Fit the degree-d pipeline on reps fresh n-point samples; return (bias squared, variance) at x_test."""
    ...

print(x_test.shape, round(float(true_f(x_test).std()), 3), bias2_and_variance(25, 1))

# %% [markdown]
# Coding exercise 4, The Bayes floor with unequal priors: the starter block.
# %%
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

pi0, pi1 = 0.3, 0.7                             # class priors; class 0 is N(-1, 1), class 1 is N(+1, 1)

def bayes_boundary(pi0, pi1):
    """Solve pi0 N(x; -1, 1) = pi1 N(x; +1, 1) for x (by hand first, then return the number)."""
    ...

def bayes_error(boundary, pi0, pi1):
    """Return the Bayes error: the mass of each class that lands on the wrong side of the boundary."""
    ...

print(round(norm.cdf(-1), 4), bayes_boundary(pi0, pi1), bayes_error(0.0, pi0, pi1))

# %% [markdown]
# Coding exercise 5, Choosing the regularization strength by cross-validation: the starter block.
# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.model_selection import cross_val_score

dd = load_diabetes()                            # 442 patients, 10 features that we standardize next
Xs = StandardScaler().fit_transform(dd.data); yv = dd.target
alphas = np.logspace(-2, 2, 41)

def chosen_alphas(Xs, yv, alphas):
    """Return (Lasso alpha, number of nonzero Lasso coefficients, dropped feature names, Ridge alpha)."""
    ...

def cv_mse(model, Xs, yv):
    """Return the 5-fold cross-validated MSE of model."""
    ...

print(Xs.shape, dd.feature_names, chosen_alphas(Xs, yv, alphas))

# %% [markdown]
# Coding exercise 6, A validation curve over C: the starter block.
# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)      # 569 tumors, 30 features
Cs = np.logspace(-3, 3, 7)
pipe = make_pipeline(StandardScaler(), SVC(gamma=0.01))   # gamma fixed at the chapter's sweet spot

def curve_over_C(pipe, X, y, Cs):
    """Return (mean training accuracy, mean validation accuracy) per C from a 5-fold validation_curve."""
    ...

print(X.shape, Cs, curve_over_C(pipe, X, y, Cs))

# %% [markdown]
# Coding exercise 7, A full evaluation report on imbalanced data: the starter block.
# %%
import numpy as np, pandas as pd, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score,
                             precision_recall_curve)

df = sns.load_dataset("titanic"); feats = ["pclass", "age", "fare", "sibsp", "parch"]
died, surv = df[df.survived == 0], df[df.survived == 1]
imb = pd.concat([died, surv.sample(n=60, random_state=0)]).reset_index(drop=True)   # 60 survivors, about 10 percent (the chapter kept 95)
X, yv = imb[feats], imb["survived"]

def metric_report(y_true, pred, proba):
    """Return a dict with the four confusion counts and the eight metrics the task names, plus the majority baseline."""
    ...

def best_f1_threshold(model, Xtr, ytr):
    """Return the probability cutoff that maximizes F1 on out-of-fold probabilities of the training rows."""
    ...

print(imb.shape, round(yv.mean(), 3), metric_report(yv, yv, yv))

# %% [markdown]
# Coding exercise 8, Tuning three ways and exposing selection bias: the starter block.
# %%
import numpy as np
from scipy.stats import loguniform
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_wine(return_X_y=True)               # 178 wines, 13 measurements, 3 cultivars: data the chapter never tuned on
pipe = make_pipeline(StandardScaler(), SVC())
grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [1e-3, 1e-2, 1e-1, 1]}
dist = {"svc__C": loguniform(1e-1, 1e3), "svc__gamma": loguniform(1e-4, 1e0)}
big_grid = {"svc__C": np.logspace(-2, 3, 8), "svc__gamma": np.logspace(-4, 1, 8)}

def grid_and_random_best(pipe, X, y, grid, dist):
    """Return (grid best_score_, random best_score_) with 5 folds and random_state=0, 8 random samples."""
    ...

def nested_estimate(pipe, X, y, big_grid):
    """Return (inner best_score_ on all rows, outer 5-fold nested cross-validation mean) for the 8-by-8 grid."""
    ...

print(X.shape, len(grid["svc__C"]) * len(grid["svc__gamma"]), big_grid["svc__C"].size * big_grid["svc__gamma"].size, nested_estimate(pipe, X, y, big_grid))

# %% [markdown]
# Coding exercise 9, Beating the baseline is not luck: the starter block.
# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, permutation_test_score, StratifiedKFold

X, y = load_digits(return_X_y=True)
y8 = (y == 8).astype(int)                       # detect the digit 8: about ten percent positive

def baseline_and_model(X, y8):
    """Return (majority-class accuracy, logistic regression CV mean, CV standard deviation) with 5 stratified folds."""
    ...

def permutation_p_value(X, y8, n_permutations=200):
    """Return the permutation_test_score p-value for the logistic regression against shuffled labels."""
    ...

print(X.shape, round(y8.mean(), 3), baseline_and_model(X, y8))

# %% [markdown]
# Coding exercise 10, An end-to-end model-selection workflow: the starter block.
# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

Xb, yb = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.3, random_state=0, stratify=yb)
cands = {"baseline": DummyClassifier(strategy="most_frequent"),
         "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
         # ensemble=False keeps one SVC refit on all the training rows rather than averaging five fold models;
         # it is scikit-learn 1.9's replacement for the deprecated SVC(probability=True), which 6.5.7 will explain
         "rbf_svc": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(random_state=0), ensemble=False)),
         "rf": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2)}

def validation_auc(cands, Xtr, ytr):
    """Return {name: 5-fold cross-validated ROC AUC} for every candidate."""
    ...

def winner_test_report(cands, scores, Xtr, ytr, Xte, yte):
    """Pick the best real model by validation AUC, refit on all training rows, return (test AUC, log loss, Brier)."""
    ...

print(Xtr.shape, Xte.shape, list(cands), validation_auc(cands, Xtr, ytr))

# %% [markdown]
# Coding exercise 11, Learning curves for three classifiers: the starter block.
# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

Xb, yb = load_breast_cancer(return_X_y=True)
models = {"logreg C=1e-4": make_pipeline(StandardScaler(), LogisticRegression(C=1e-4, max_iter=5000)),
          "logreg C=1": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000)),
          "1-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))}

def curve_table(model, X, y):
    """Return (sizes, mean training accuracy, mean validation accuracy) at five sizes, 5 folds, shuffle=True, random_state=0."""
    ...

print(Xb.shape, round(yb.mean(), 3), curve_table(models["1-NN"], Xb, yb))

# %% [markdown]
# Coding exercise 12, Calibration repair for a LinearSVC: the starter block.
# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

Xb, yb = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.3, stratify=yb, random_state=0)

def calibrated_report(method, Xtr, ytr, Xte, yte):
    """Fit StandardScaler + CalibratedClassifierCV(LinearSVC, method, cv=5); return (bin means, bin fractions, Brier, log loss)."""
    ...

def compare_over_splits(n_splits=10):
    """Return the mean Brier score and mean log loss of each method over n_splits random 70/30 splits."""
    ...

print(len(ytr), len(yte), calibrated_report("sigmoid", Xtr, ytr, Xte, yte))

# %% [markdown]
# Coding exercise 13, Expected improvement, Optuna, and successive halving: the starter block.
# %%
import numpy as np
from scipy.stats import norm, loguniform
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def expected_improvement(mu, sigma, f_star):
    """Return EI = (mu - f_star) Phi(z) + sigma phi(z) with z = (mu - f_star) / sigma; work the two candidates by hand first."""
    ...

Xd, yd = load_digits(return_X_y=True)           # 1797 digits, 64 pixels, 10 classes
dist = {"svc__C": loguniform(1e-1, 1e3), "svc__gamma": loguniform(1e-4, 1e0)}

def optuna_study(Xd, yd, n_trials=25):
    """Return (best value, best params, best trial number, best value among the first ten trials) of a TPE study with seed 0."""
    ...

def halving_search(Xd, yd, dist):
    """Return (best_score_, n_candidates_, n_resources_, full-data CV of the winner) of HalvingRandomSearchCV(factor=3, cv=5, random_state=0)."""
    ...

print(Xd.shape, round(norm.cdf(2.0), 4), expected_improvement(0.980, 0.001, 0.978))

# %% [markdown]
# Coding exercise 14, Early stopping on other tumors and augmentation at three scales: the starter block.
# %%
import numpy as np, seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.metrics import log_loss, mean_squared_error

Xb, yb = load_breast_cancer(return_X_y=True)
X_grow, X_hold, y_grow, y_hold = train_test_split(Xb, yb, test_size=0.3, random_state=0, stratify=yb)

def early_stopping_run(fit_seed, sgd_seed):
    """Draw 40 tumors from X_grow with random_state=fit_seed, run 2000 warm-start epochs as in Example 4-5,
    and return (stopping epoch, weight norm there, test log loss there, test log loss at epoch 2000)."""
    ...

mpg = sns.load_dataset("mpg").dropna(subset=["horsepower", "mpg"])
sub = mpg.sample(n=45, random_state=1)          # the chapter's 45 cars and the 4.6.1 split
Xm = sub[["horsepower"]].to_numpy(float); ym = sub["mpg"].to_numpy(float)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.35, random_state=0)

def jittered_mse(scale, copies=20, seed=0):
    """Train the degree-15 pipeline on the training cars plus `copies` jittered copies; return (test MSE, train MSE)."""
    ...

print(X_grow.shape, X_hold.shape, Xm_tr.shape, early_stopping_run(1, 0), jittered_mse(1.0))

# %% [markdown]
# Coding exercise 15, Five percent of the labels, then even-to-odd transfer: the starter block.
# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading
from sklearn.neural_network import MLPClassifier

Xd, yd = load_digits(return_X_y=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.3, stratify=yd, random_state=0)
rng = np.random.default_rng(1)
unlabeled = rng.random(len(yd_tr)) < 0.95       # hide 95 percent of the training labels
y_semi = yd_tr.copy(); y_semi[unlabeled] = -1

def three_accuracies(Xd_tr, y_semi, unlabeled, Xd_te, yd_te):
    """Return the test accuracy of logistic regression on the labeled rows, SelfTrainingClassifier, and LabelSpreading(knn, 10)."""
    ...

def even_to_odd_transfer(n_labels):
    """Freeze an MLP body trained on the even digits; return (from-scratch accuracy, frozen-body accuracy) on the odd digits."""
    ...

print(int((y_semi != -1).sum()), len(yd_tr), len(np.unique(yd_tr[~unlabeled])), three_accuracies(Xd_tr, y_semi, unlabeled, Xd_te, yd_te))

# %% [markdown]
# Coding exercise 16, Double descent on another 45 cars: the starter block.
# %%
import numpy as np, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mpg = sns.load_dataset("mpg").dropna(subset=["horsepower", "mpg"])
sub = mpg.sample(n=45, random_state=2)          # a different 45 cars from Example 4-4's random_state=1
X = sub[["horsepower"]].to_numpy(float); y = sub["mpg"].to_numpy(float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.35, random_state=0)
lo, hi = 46.0, 230.0                            # horsepower's range in the full mpg data

def legendre_features(Xcol, D):
    """Degree-D Legendre basis of horsepower rescaled to [-1, 1] (as in Example 4-4)."""
    ...

def min_norm_sweep(X_tr, y_tr, X_te, y_te, degrees):
    """Return [(degree, train MSE, test MSE, coefficient norm)] using the minimum-norm fit at each degree."""
    ...

print(len(X_tr), len(np.unique(X_tr.ravel())), min_norm_sweep(X_tr, y_tr, X_te, y_te, [1, 3, 9]))

# %% [markdown]
# Coding exercise 17, Making the three averages disagree: the starter block.
# %%
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, top_k_accuracy_score

Xw, yw = load_wine(return_X_y=True)             # class counts 59, 71, 48

def thin_class(X, y, cls, keep, seed=0):
    """Return (X, y) with only `keep` randomly chosen rows of class cls retained."""
    ...

def averaged_scores(y_true, pred, proba):
    """Return (per-class F1, macro F1, weighted F1, micro F1, top-2 accuracy)."""
    ...

print(np.bincount(yw), thin_class(Xw, yw, 2, 10))

# %% [markdown]
# Coding exercise 18, Error analysis for a 3-nearest-neighbor classifier: the starter block.
# %%
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

Xd, yd = load_digits(return_X_y=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.3, random_state=0, stratify=yd)   # the 4.9.2 split
knn3 = KNeighborsClassifier(n_neighbors=3).fit(Xd_tr, yd_tr)
cm = confusion_matrix(yd_te, knn3.predict(Xd_te))

def top_confusions(cm, k=3):
    """Return the k largest off-diagonal entries as (count, true, predicted) tuples, largest first."""
    ...

def save_confusion_images(X, y_true, pred, true_digit, predicted_digit, path):
    """Save every test image of true_digit that was predicted as predicted_digit to path as one PNG strip."""
    ...

print(cm.shape, int(np.trace(cm)), top_confusions(cm))

# %% [markdown]
# Coding exercise 19, Adjusted R-squared at two sample sizes: the starter block.
# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

Xdb, ydb = load_diabetes(return_X_y=True)       # 442 patients, 10 features
small = np.random.default_rng(0).choice(len(ydb), size=40, replace=False)

def adj_r2(r2, n, p):
    """The adjusted R-squared of 4.7.1 for n rows and p features."""
    ...

def r2_with_and_without_noise(X, y, n_noise=10, seed=1):
    """Return (R2, adjusted R2) of OLS on X, and the same pair after appending n_noise Gaussian noise columns."""
    ...

print(Xdb.shape, Xdb[small].shape, r2_with_and_without_noise(Xdb[small], ydb[small]))

# %% [markdown]
# Coding exercise 20, McNemar's test for a model swap: the starter block.
# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.contingency_tables import mcnemar

Xb, yb = load_breast_cancer(return_X_y=True)

def disagreement_table(model_a, model_b, Xte, yte):
    """Return the 2-by-2 array [[both right, only A right], [only B right, both wrong]]."""
    ...

def mcnemar_report(split_seed):
    """Fit a scaled logistic regression and a 200-tree forest on a stratified 70/30 split with random_state=split_seed;
    return (b, c, continuity-corrected statistic, chi-square p, exact p)."""
    ...

print(Xb.shape, [mcnemar_report(s) for s in (1, 2, 3)])

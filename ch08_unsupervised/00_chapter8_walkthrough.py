# %% [markdown]
# Chapter 8 walkthrough. Every code block from the chapter, in book order.


# %% [markdown]

# ### Chapter 8: Unsupervised Learning
#
# Every model so far learned from labeled examples: inputs paired with the right answer. But most of the world's data has no labels, just raw observations with structure waiting to be found. Unsupervised learning finds that structure without being told what to look for. It splits into two great families. Clustering groups similar points together, answering "which things belong with which," and powers customer segmentation, image compression, and anomaly baselines. Dimensionality reduction compresses many features into a few while preserving what matters, which makes data visualizable, denoises it, and speeds up every downstream model. Around these sit anomaly detection, finding the points that fit no pattern, association rules, finding which items occur together, and topic modeling, finding the themes in text. The recurring challenge of the whole chapter is that without labels there is no single right answer and no accuracy score to optimize, so evaluation becomes subtler and judgment matters more.
#
# #### 8.1 Clustering
#
# Clustering partitions data into groups so that points in the same group are more similar to each other than to points in other groups, and it does this with no labels to guide it. The catch is that "similar" and even "how many groups" are choices, not givens, so different methods encode different ideas of what a cluster is: a ball around a center, a chain of nearby points, a dense region, a density peak, a Gaussian blob, or connectivity on a graph. This section covers k-means and its minibatch variant, hierarchical clustering, DBSCAN and HDBSCAN, Gaussian mixtures, spectral clustering, and mean-shift, the eight rows of Table 8-1. It starts with k-means.
#
# ##### 8.1.1 k-means
#
# k-means splits the data into a chosen number $k$ of clusters, each represented by its center, or centroid. The rule is simple: every point belongs to the nearest centroid, and every centroid sits at the mean of its points. Those two conditions are circular, the assignment depends on the centers and the centers depend on the assignment, so k-means alternates between them until they agree. Its uses run from customer segmentation to compressing an image's colors down to $k$ representative ones, and the scikit-learn user guide notes that it copes well as the number of samples grows (scikit-learn developers, 2026d, section 2.3.2).
#
# The algorithm, called Lloyd's algorithm after Lloyd (1982), minimizes the within-cluster sum of squares, also called inertia, the total squared distance from each point to its centroid:
#
# $$J = \sum_{c=1}^{k} \sum_{\mathbf{x} \in C_c} \lVert \mathbf{x} - \boldsymbol{\mu}_c \rVert^2,$$
#
# where $C_c$ is the set of points in cluster $c$ and $\boldsymbol{\mu}_c$ is its centroid. Starting from initial centers, it repeats two steps: the assignment step puts each point in the cluster of its nearest centroid, and the update step moves each centroid to the mean of its assigned points. Each step can only lower $J$ or leave it unchanged, so the process converges, though only to a local minimum, not necessarily the global one (Bishop, 2006, section 9.1). The name k-means appears in MacQueen (1967), whose version moves the nearest center after each point rather than after a full pass over the data (Bishop, 2006, section 9.1).
#
# One round by hand shows both steps at work. Take six points on a line, 1, 2, 3, 8, 9, and 10, and start two centers badly, at 1 and 4. The assignment step sends 1 and 2 to the center at 1, and 3, 8, 9, and 10 to the center at 4, since 3 is 1 away from 4 but 2 away from 1. With those centers, $J = 0^2 + 1^2 + 1^2 + 4^2 + 5^2 + 6^2 = 0 + 1 + 1 + 16 + 25 + 36 = 79$. The update step moves each center to the mean of its points, $(1 + 2)/2 = 1.5$ and $(3 + 8 + 9 + 10)/4 = 7.5$, and $J$ drops to $0.5^2 + 0.5^2 + 4.5^2 + 0.5^2 + 1.5^2 + 2.5^2 = 0.25 + 0.25 + 20.25 + 0.25 + 2.25 + 6.25 = 29.5$. In the second round the point 3 sits 1.5 from the first center and 4.5 from the second, so it changes clusters, and $J$ falls to 11.5 before the centers move; the new means, 2 and 9, bring it down to 4. A third assignment changes nothing, so the algorithm has converged, with centers 2 and 9 and $J = 4$. The next block runs the same three rounds.

# %%
import numpy as np
x = np.array([1.0, 2, 3, 8, 9, 10])                  # six points on a line
centers = np.array([1.0, 4.0])                       # two deliberately poor starting centers
for step in range(3):
    labels = np.abs(x[:, None] - centers[None]).argmin(1)          # assignment step
    J_assign = float(((x - centers[labels]) ** 2).sum())           # J after assigning
    centers = np.array([x[labels == j].mean() for j in range(2)])  # update step
    J_update = float(((x - centers[labels]) ** 2).sum())           # J after moving the centers
    print(step + 1, labels.tolist(), J_assign, centers.tolist(), J_update)
# 1 [0, 0, 1, 1, 1, 1] 79.0 [1.5, 7.5] 29.5
# 2 [0, 0, 0, 1, 1, 1] 11.5 [2.0, 9.0] 4.0
# 3 [0, 0, 0, 1, 1, 1] 4.0 [2.0, 9.0] 4.0

# %% [markdown]

# Each printed row is one round: the round number, the cluster of each point, $J$ after the assignment step, the new centers, and $J$ after the update step. The rows read 79.0 then 29.5, 11.5 then 4.0, and 4.0 then 4.0, the hand values, so each step lowers $J$ or leaves it alone, and the third round, which moves nothing, marks convergence.
#
# To watch it recover real groups we use the Palmer penguins dataset of Gorman, Williams, and Fraser (2014) as seaborn ships it: after dropping incomplete rows, 333 penguins of three species (Adelie, Chinstrap, and Gentoo) measured by bill length, bill depth, flipper length, and body mass. We standardize the four measurements so each counts equally, set the species labels aside, cluster with $k = 3$, and then score the clusters against the species we held out. Written out, the core is a few lines, in Example 8-1.
#
# Example 8-1. k-means from scratch: Lloyd's two alternating steps.

# %%
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
pen = sns.load_dataset("penguins").dropna()
feats = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = StandardScaler().fit_transform(pen[feats].to_numpy(float))   # standardize the four measurements
ytrue = pen["species"].map({s: i for i, s in enumerate(sorted(pen["species"].unique()))}).to_numpy()
rng = np.random.default_rng(0)

def kmeans_scratch(X, k, iters=100):
    centers = X[rng.choice(len(X), k, replace=False)]            # random initial centers
    for _ in range(iters):
        d = np.linalg.norm(X[:, None] - centers[None], axis=2)   # distance to each center
        labels = d.argmin(axis=1)                                 # assignment step
        centers = np.array([X[labels == j].mean(0) for j in range(k)])  # update step
    return labels, centers

labels, centers = kmeans_scratch(X, 3)
inertia = sum(((X[labels == j] - centers[j]) ** 2).sum() for j in range(3))   # the objective J, by hand
sk = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)             # the library version
print(round(adjusted_rand_score(ytrue, labels), 3),
      round(adjusted_rand_score(ytrue, sk.labels_), 3))                 # 0.806 0.799
ct = np.zeros((3, 3), dtype=int)
for a, b in zip(labels, sk.labels_):
    ct[a, b] += 1
n_disagree = int(ct.sum() - ct.max(axis=1).sum())   # match cluster ids, then count mismatches
print(f"{inertia:.3f} {sk.inertia_:.3f} {n_disagree}")                   # 370.770 370.766 1
for name in sorted(pen["species"].unique()):
    row = sk.labels_[pen["species"].to_numpy() == name]
    print(name, [int(np.sum(row == c)) for c in range(3)])
# Adelie [22, 0, 124]
# Chinstrap [63, 0, 5]
# Gentoo [0, 119, 0]

# %% [markdown]

# On the standardized penguin measurements both recover the three species at an adjusted Rand index near 0.80 (0.806 from scratch, 0.799 for scikit-learn), where 1.0 is a perfect match and 0 is random. The next print is the two inertias to three decimals and the number of penguins the two partitions place differently once cluster ids are matched: 370.770, 370.766, and 1. The runs reach nearly the same objective from partitions that differ by a single penguin, and that one penguin is why the adjusted Rand indices are 0.806 and 0.799. The adjusted Rand index is the standard way to score a clustering against known labels when you have them for evaluation. The score is not a perfect 1.0 because Adelie and Chinstrap overlap in these four measurements. The species rows that follow are counts against scikit-learn's three clusters: Adelie is [22, 0, 124], Chinstrap is [63, 0, 5], and Gentoo is [0, 119, 0]. So 22 Adelie penguins sit in the cluster that holds 63 Chinstraps, and 5 Chinstraps sit in the cluster that holds 124 Adelies. That is 27 of the 333 penguins, and 27/333 = 0.081. Gentoo's 119 birds occupy a cluster alone. The overlap is the ordinary texture of real data rather than the clean separation of textbook blobs.
#
# The hardest practical question is choosing $k$, since the algorithm cannot decide it. Two tools help. The elbow method plots inertia against $k$: inertia always falls as $k$ rises, since more centroids fit tighter, but it falls steeply until the natural number of clusters and then levels off, leaving a visible elbow. The silhouette score of Rousseeuw (1987) measures, for each point, how much closer it is to its own cluster than to the nearest other cluster, on a scale from $-1$ to $1$, and its average peaks at the best $k$. On clean, well-separated blobs the two agree, but on real data they often disagree, as the penguin sweep in the next listing shows, where a third signal, agreement with the known species, settles the choice.

# %%
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
pen = sns.load_dataset("penguins").dropna()
feats = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = StandardScaler().fit_transform(pen[feats].to_numpy(float))
ytrue = pen["species"].map({s: i for i, s in enumerate(sorted(pen["species"].unique()))}).to_numpy()
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    print(k, round(km.inertia_, 1), round(silhouette_score(X, km.labels_), 3),
          round(adjusted_rand_score(ytrue, km.labels_), 3))
#  k   inertia   silhouette   ARI-vs-species
#  2    552.7      0.531        0.649
#  3    370.8      0.446        0.799    <- inertia elbow; ARI peak = 3 species
#  4    293.9      0.398        0.645
#  5    228.5      0.374        0.578
#  6    200.3      0.368        0.553

# %% [markdown]

# Inertia drops sharply through $k=3$, the number of species, and then only crawls, so its elbow sits at three. The silhouette, though, peaks not at three but at $k=2$ with 0.531, because Gentoo penguins are large and separate cleanly while Adelie and Chinstrap overlap, so a purely geometric score sees two well-separated groups rather than three. The agreement-with-truth column breaks the tie: the adjusted Rand index peaks at 0.799 exactly at $k=3$, confirming the three species. This is what real data looks like. The elbow and the silhouette can disagree, and when they do you lean on whatever outside knowledge you have, here the known species count, because an internal score like the silhouette measures separation in the features, not the biology that produced them. Be clear about what made that tiebreaker possible: the penguins are a labeled benchmark, so the species column exists to check against. On data you actually need to cluster there is no such column, and the internal metrics of 8.1.6 are all you have; the ARI column here is a validation of the method, not a step you can reproduce on unlabeled data.
#
# Initialization matters too, because Lloyd's algorithm only finds a local minimum and a bad random start can converge to a poor one. The k-means++ initialization of Arthur and Vassilvitskii (2007) fixes this by choosing initial centers spread apart: the first is random, and each subsequent center is chosen with probability proportional to its squared distance from the nearest center already chosen, so the seeds rarely bunch together. On an easy problem it makes no difference, but on a harder problem with twelve clusters the gap is large; we keep this one synthetic, generated with `make_blobs`, precisely because we need to dial the cluster count up to a known twelve to expose the trap.

# %%
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
Xpp, _ = make_blobs(n_samples=1500, centers=12, cluster_std=0.6, random_state=1)
rand, pp = [], []
for s in range(30):                                  # 30 different single-start seeds
    rand.append(KMeans(n_clusters=12, init="random",    n_init=1, random_state=s).fit(Xpp).inertia_)
    pp.append(  KMeans(n_clusters=12, init="k-means++", n_init=1, random_state=s).fit(Xpp).inertia_)
print("random   ", round(np.mean(rand)), round(np.max(rand)))   # mean 1372, worst 2955
print("k-means++", round(np.mean(pp)),   round(np.max(pp)))     # mean  978, worst 1212

# %% [markdown]

# Random initialization averages an inertia of 1372 and in its worst run reaches a terrible 2955, stuck in a bad local minimum, while k-means++ averages 978 and never exceeds 1212, both lower and far more reliable. k-means++ is also scikit-learn's default initialization. Since version 1.4 the default `n_init='auto'` runs k-means++ once, and it runs ten restarts when `init='random'` or when `init` is a callable (one run as well when `init` is an array of centers). The scikit-learn 1.9.1 `KMeans` documentation states that rule and records that the default changed to `'auto'` in version 1.4 (scikit-learn developers, 2026d, KMeans API reference). A block in this chapter that wants several restarts passes `n_init=10` explicitly. The thirty-seed loop in this listing sets `n_init=1` so that the gap comes from the choice of seeds alone.
#
# When the dataset is too large for ordinary k-means, which must scan every point on every iteration, the minibatch variant scales it down to size. Minibatch k-means (Sculley, 2010) updates the centroids from a small random batch of points at each step rather than the whole dataset, exactly the stochastic idea of the minibatch gradient descent of 2.6.4, trading a slightly worse final inertia for updates that touch only a batch of rows. The 1,797 digit images in the next listing, scikit-learn's copy of the test set of the UCI Optical Recognition of Handwritten Digits data (Alpaydin and Kaynak, 1998), are too small to show a large speedup. The method is aimed at data large enough that scanning every row on every iteration is the cost. On the 1,797 standardized digit images it reaches an inertia of about 71,100 against full k-means's 69,400, a gap of roughly two percent, while touching only a small batch of points at each step. The scikit-learn user guide presents `MiniBatchKMeans` as a way to cut the computation time of k-means on very large numbers of samples, at the price of a final result that is usually only a little worse (scikit-learn developers, 2026d, section 2.3.2.2).

# %%
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
Xd = StandardScaler().fit_transform(load_digits().data)          # 1797 real 8-by-8 digit images
full = KMeans(n_clusters=10, n_init=10, random_state=0).fit(Xd).inertia_
mini = MiniBatchKMeans(n_clusters=10, n_init=10, batch_size=256, random_state=0).fit(Xd).inertia_
print(round(full), round(mini))                                  # 69405 71075

# %% [markdown]

# Full k-means prints an inertia of 69405 and the minibatch version 71075, the two percent that the batching costs us in exchange for scanning only a fraction of the data at each step.
#
# The same center-finding idea compresses a photograph's colors. scikit-learn's sample image `china.jpg` is 427 by 640 pixels with three color channels. On a 4,000-pixel sample, 4 colors leave a mean squared error of 449.3 and 16 colors leave 111.2.

# %%
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
img = load_sample_image("china.jpg")
pix = img.reshape(-1, 3).astype(float)
rng = np.random.default_rng(0)
sample = pix[rng.choice(len(pix), size=4000, replace=False)]
for k in (4, 16):
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(sample)
    rec = km.cluster_centers_[km.labels_]
    print(k, round(float(np.mean((sample - rec) ** 2)), 1))
# 4 449.3
# 16 111.2

# %% [markdown]

# The two prints are 449.3 and 111.2. They are the error on the sample, not on every pixel of the photograph.
#
# Color quantization is the name for this use of k-means: every pixel is replaced by the centroid of its cluster, so the photograph can be stored as a palette of $k$ colors and one small index per pixel. The full photograph holds $427 \times 640 = 273{,}280$ pixels, and the next block counts 96,615 distinct colors among them. Storage is the reason to bother. The original spends 3 bytes, or 24 bits, on every pixel, $273{,}280 \times 3 = 819{,}840$ bytes. With 16 colors each index needs $\log_2 16 = 4$ bits, so the indices fill $273{,}280 \times 4 / 8 = 136{,}640$ bytes and the palette $16 \times 3 = 48$ more, 136,688 bytes in all, a file $819{,}840 / 136{,}688 \approx 6.0$ times smaller. With 4 colors and 2-bit indices the ratio is about 12.0, and with 64 colors and 6-bit indices about 4.0. The error printed so far is the k-means objective in disguise. $J$ adds up $\lVert\mathbf{x} - \boldsymbol{\mu}_c\rVert^2$ over the pixels, and each squared distance is a sum over the red, green, and blue channels, so
#
# $$\text{MSE} = \frac{J}{3n},$$
#
# the mean squared error per channel value, where $n$ is the number of pixels. The block fits each palette twice, once on the 4,000-pixel sample and once on every pixel, and scores both on every pixel. Its last line tries the simpler alternative, a fixed palette that cuts each channel into four equal bins (0 to 63, 64 to 127, 128 to 191, and 192 to 255) and paints every pixel with the centers of its bins, $4^3 = 64$ colors chosen without looking at the photograph.

# %%
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
img = load_sample_image("china.jpg")
pix = img.reshape(-1, 3).astype(float)                  # all 273,280 pixels, one RGB row each
print(len(pix), len(np.unique(img.reshape(-1, 3), axis=0)))   # pixels, distinct colors
rng = np.random.default_rng(0)
sample = pix[rng.choice(len(pix), size=4000, replace=False)]  # the same 4,000-pixel sample
for k in (4, 16, 64):
    small = KMeans(n_clusters=k, n_init=3, random_state=0).fit(sample)   # palette from the sample
    full = KMeans(n_clusters=k, n_init=3, random_state=0).fit(pix)       # palette from every pixel
    err_small = np.mean((pix - small.cluster_centers_[small.predict(pix)]) ** 2)   # both scored
    err_full = np.mean((pix - full.cluster_centers_[full.labels_]) ** 2)           # on every pixel
    print(k, round(float(err_small), 1), round(float(err_full), 1), round(full.inertia_ / pix.size, 1))
grid = np.floor(pix / 64) * 64 + 31.5                   # fixed palette: 4 levels per channel, 4**3 = 64 colors
print(round(float(np.mean((pix - grid) ** 2)), 1))
# 273280 96615
# 4 457.7 456.3 456.3
# 16 115.6 114.6 114.6
# 64 39.7 37.3 37.3
# 366.3

# %% [markdown]

# The photograph has 273,280 pixels and 96,615 distinct colors. Scored on every pixel, the 4-color palette learned from the sample leaves a mean squared error of 457.7 and the palette learned from every pixel 456.3; at 16 colors the pair is 115.6 and 114.6, and at 64 colors 39.7 and 37.3. The sample's own errors, 449.3 and 111.2, were therefore a little optimistic, yet a palette fitted on 4,000 pixels, $4{,}000 / 273{,}280 \approx 1.5$ percent of the image, comes within 1.4, 1.0, and 2.4 of the full fit. The gap is largest at 64 colors, where the sample offers only $4{,}000 / 64 = 62.5$ pixels per centroid on average. The third number on each line, the full fit's inertia divided by the $3n = 819{,}840$ channel values, equals the second, as $\text{MSE} = J/(3n)$ says it must. The fixed 64-color grid leaves 366.3, almost ten times the 37.3 that k-means reaches with the same 64 colors, because the grid spreads its colors evenly over the whole cube of possible colors while k-means places its centroids where this photograph's pixels actually are. To experiment, rebuild the picture with `full.cluster_centers_[full.labels_].reshape(img.shape).astype("uint8")`, display it beside `img`, and look for bands of flat color in smoothly shaded regions at small $k$.
#
# Clusters can also become features for a supervised model. Each centroid marks a typical point of the data, and a row's similarity to every centroid becomes a new column, which gives a model that can only add up weighted columns a way to say "near here." Location shows why. In the California housing data (Pace and Barry, 1997), 20,640 districts from the 1990 census with the median house value in units of 100,000 dollars as the target, a linear model can use latitude and longitude only as two weighted columns, which describes a single tilted plane of price over the map, however the prices are actually arranged. One cure, which Géron (2023, Chapter 2, "Custom Transformers") builds for this dataset, runs k-means on the two coordinates alone and turns each district's distance to each centroid into the radial basis function (RBF) similarity of 6.5.3,
#
# $$\phi_c(\mathbf{x}) = \exp\big(-\gamma\,\lVert\mathbf{x} - \boldsymbol{\mu}_c\rVert^2\big), \qquad c = 1, \ldots, k,$$
#
# where $\mathbf{x}$ holds the district's latitude and longitude in degrees and $\gamma > 0$ sets the width of each bump. The similarity is 1 at the centroid and falls toward 0 with distance. With $\gamma = 1$, a district 0.5 degrees away scores $e^{-0.25} = 0.779$, one at 1 degree $e^{-1} = 0.368$, and one at 2 degrees $e^{-4} = 0.018$; with $\gamma = 4$ the same distances give 0.368, 0.018, and 0.000, a narrower bump. The ridge regression then sees the eight standardized columns $\mathbf{z}$ plus the $k$ similarities, so its prediction
#
# $$\hat y = \mathbf{w}^\top\mathbf{z} + \sum_{c=1}^{k} v_c\,\phi_c(\mathbf{x}) + b$$
#
# can raise or lower the price around each centroid separately, with one weight $v_c$ per bump. Example 8-2 builds this from the pieces of 3.6.7: `KMeans` inside a pipeline acts as a transformer whose `transform` returns the distances to the centroids, a `FunctionTransformer` applies the RBF, and a `ColumnTransformer` places the new columns beside the old ones. Because the whole chain is refit inside every fold of the five-fold cross-validation, the test districts of a fold never move the centroids.
#
# Example 8-2. k-means centroids of latitude and longitude turned into RBF similarity features for a ridge regression on the California housing districts, scored by five-fold cross-validated RMSE.

# %%
import numpy as np
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
df = fetch_california_housing(as_frame=True).frame        # 20,640 districts, 1990 census
X, y = df.drop(columns="MedHouseVal"), df["MedHouseVal"]   # target in units of $100,000
cv = KFold(n_splits=5, shuffle=True, random_state=0)

def cv_rmse(model):                                       # mean and spread of the five fold RMSEs
    rmse = -cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return round(float(rmse.mean()), 4), round(float(rmse.std()), 4)

def with_clusters(k, gamma):
    # k-means sees only Latitude and Longitude; KMeans.transform returns each district's
    # distance d to every centroid, and the RBF turns it into a similarity exp(-gamma * d**2)
    geo = make_pipeline(KMeans(n_clusters=k, n_init=10, random_state=0),
                        FunctionTransformer(lambda d: np.exp(-gamma * d ** 2)))
    prep = ColumnTransformer([("all8", StandardScaler(), list(X.columns)),
                              ("geo", geo, ["Latitude", "Longitude"])])
    return make_pipeline(prep, Ridge(alpha=1.0))         # refit inside every fold, so no leakage

print(cv_rmse(make_pipeline(StandardScaler(), Ridge(alpha=1.0))))   # the eight columns alone
for k, gamma in [(10, 1.0), (30, 1.0), (100, 1.0), (100, 4.0)]:
    print(k, gamma, cv_rmse(with_clusters(k, gamma)))
# (0.7263, 0.0115)
# 10 1.0 (0.7026, 0.009)
# 30 1.0 (0.6658, 0.0092)
# 100 1.0 (0.6577, 0.0093)
# 100 4.0 (0.6268, 0.0088)

# %% [markdown]

# Each line prints the mean and the standard deviation of the five fold RMSEs. The eight columns alone give 0.7263, about 72,600 dollars of error on a typical district, with a fold spread of 0.0115. Ten cluster similarities at $\gamma = 1$ lower it to 0.7026, thirty to 0.6658, and a hundred to 0.6577, and narrowing the bumps to $\gamma = 4$ with a hundred centroids reaches 0.6268, about 62,700 dollars, an improvement of $0.7263 - 0.6268 = 0.0995$, more than eight times the baseline's fold spread. Ten wide bumps can describe only the coarse geography; a hundred narrow ones can follow smaller pockets of price. One caution: the best $k$ and $\gamma$ were picked by looking at these same cross-validated scores, so 0.6268 carries the selection optimism that 4.8.6 measures, and an honest final number would come from a nested search or a held-out test set. To experiment, drop the `FunctionTransformer` so that the raw distances from `KMeans.transform` become the features: with thirty centroids that scores 0.6614, about the same as the RBF version's 0.6658, so on this data what matters most is giving the linear model several location columns that respond to distance, whichever shape they take.
#
# Clusters can also stretch a small labeling budget. Section 4.2.1 hid all but 38 of the 1,257 training digits' labels and chose the 38 at random. The alternative is to let k-means choose: cluster the unlabeled training digits into $k$ groups, ask a person to label only the digit nearest each centroid, the most typical member of its cluster, and then, optionally, copy each of those labels to every digit in its cluster. Géron (2023, Chapter 9, "Using Clustering for Semi-Supervised Learning") demonstrates both steps; the run here uses the split of 4.2.1 so that its numbers can be read against that section's. In symbols, the representative of cluster $c$ is the training digit closest to its centroid,
#
# $$r_c = \arg\min_{i}\ \lVert\mathbf{x}_i - \boldsymbol{\mu}_c\rVert,$$
#
# and propagation gives digit $i$ the label $\hat y_i = y_{r_{c(i)}}$, where $c(i)$ is the cluster of digit $i$. A propagated label is wrong whenever a digit's true class differs from its representative's, so the share of correct labels can be no higher than the cluster purity
#
# $$P = \frac{1}{n}\sum_{c=1}^{k}\max_{j}\ \big|\{\, i \in C_c : y_i = j \,\}\big|,$$
#
# the share of digits that belong to their cluster's most common class, and it equals $P$ exactly when every representative carries its cluster's majority label. For a feel of the bound, a cluster of 30 digits holding 27 sevens and 3 ones contributes at most 27 correct labels, however its representative is chosen. Random labels waste the budget in two ways that representatives tend to avoid: a random draw can miss a class altogether, as the 12-label draw of 4.2.1 missed one of the ten digits, and it can spend several labels on nearly identical images of a common shape. Example 8-3 labels the representatives for $k = 38$ and $k = 50$ (in the simulation the label is looked up in `y_tr`, standing in for the person), compares each with 20 random draws of the same size, and then trains on the propagated labels.
#
# Example 8-3. Labeling only the training digit nearest each k-means centroid, compared with random label budgets of the same size, and then copying each representative's label to its whole cluster.

# %%
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)   # the 4.2.1 split

def test_acc(rows, labels):                       # train on the chosen rows, score on the 540 test digits
    clf = LogisticRegression(max_iter=2000).fit(X_tr[rows], labels)
    return accuracy_score(y_te, clf.predict(X_te))

rng = np.random.default_rng(0)
for k in (38, 50):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_tr)   # clustering uses no labels
    reps = km.transform(X_tr).argmin(axis=0)      # for each centroid, the nearest training digit
    rand = []
    for _ in range(20):                           # 20 random label budgets of the same size
        r = rng.choice(len(X_tr), k, replace=False)
        rand.append(test_acc(r, y_tr[r]))
    print(k, len(set(y_tr[reps])), round(test_acc(reps, y_tr[reps]), 3),
          round(float(np.mean(rand)), 3), round(min(rand), 3), round(max(rand), 3))
    spread = y_tr[reps][km.labels_]               # each digit takes its cluster representative's label
    purity = sum(np.bincount(y_tr[km.labels_ == c]).max() for c in range(k)) / len(y_tr)
    print(k, round(float(np.mean(spread == y_tr)), 3), round(float(purity), 3),
          round(test_acc(np.arange(len(X_tr)), spread), 3))
print(round(test_acc(np.arange(len(X_tr)), y_tr), 3))   # all 1,257 true labels, for comparison
# 38 10 0.87 0.758 0.691 0.846
# 38 0.92 0.92 0.891
# 50 10 0.898 0.816 0.726 0.885
# 50 0.959 0.959 0.937
# 0.961

# %% [markdown]

# The first line for each $k$ shows how many of the ten classes the representatives cover, the test accuracy of a logistic regression trained on the representatives alone, and the mean, lowest, and highest accuracy over the 20 random draws. With 38 labels the representatives cover all ten digits and score 0.870, while random draws of 38 average 0.758 and range from 0.691 to 0.846, so none of the 20 matched the representatives; 4.2.1's single draw of 38 scored 0.794, and its self-trainer and label spreading reached 0.844 and 0.870 only by also using the unlabeled digits. With 50 labels the representatives score 0.898 against a random mean of 0.816 (0.726 to 0.885). The second line for each $k$ is the propagation step. With 38 clusters, 0.920 of the training digits receive the right label, exactly the purity 0.920, so every representative carried its cluster's majority class, and a classifier trained on all 1,257 propagated labels scores 0.891 on the test digits. With 50 clusters the propagated labels are 0.959 correct, again equal to the purity, and the classifier reaches 0.937, against 0.961 for the same model trained on all 1,257 true labels. Fifty labels typed by a person thus close $(0.937 - 0.816)/(0.961 - 0.816) \approx 0.83$ of the gap between the random mean and full supervision. The purity bound also names the risk: every minority member of a mixed cluster inherits a wrong label at once. To experiment, raise $k$ to 100: the representatives alone then score 0.939 and the purity rises to 0.971, yet the classifier trained on the propagated labels stays at 0.937, so here, past 50 clusters, extra clusters bought cleaner labels without a better model. Another experiment is to copy each label only to the members nearest their centroid and compare the accuracy of those labels with the purity.
#
# k-means has real limitations that follow directly from its design. Because it assigns points to the nearest center by straight-line distance, it implicitly assumes clusters are round and similarly sized, and it fails when they are not. Two interleaving crescents, synthetic because we need a known nonspherical shape to expose the failure cleanly, do not match its round clusters.

# %%
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
Xm, ym = make_moons(n_samples=400, noise=0.06, random_state=0)
labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(Xm)
agree = max((labels == ym).mean(), (labels != ym).mean())
print(round(adjusted_rand_score(ym, labels), 3), round(float(agree), 3))
# 0.274 0.762   (poor ARI; 0.762 of points match after aligning labels)

# %% [markdown]

# The adjusted Rand index of 0.274 is a poor match, and 0.762 of the points still agree with the true crescents after the two cluster ids are aligned. The straight cut slices both moons, because no spherical grouping matches their shape, as Figure 8-1 shows side by side with its success on round blobs. It is also sensitive to feature scale, so features must be standardized first, and it requires $k$ in advance. The density-based method in 8.1.3 handles exactly the nonspherical shapes that defeat k-means here.
#
# The moons failure is about shape; a second failure is about scale, and it is just as easy to trigger. Because k-means measures straight-line distance, a feature with a large numeric range dominates the sum of squares and drowns out the others, so the four penguin measurements, on very different scales, make the point directly.

# %%
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
pen = sns.load_dataset("penguins").dropna()
feats = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
Xraw = pen[feats].to_numpy(float)
ytrue = pen["species"].map({s: i for i, s in enumerate(sorted(pen["species"].unique()))}).to_numpy()
print(Xraw.std(0).round(1).tolist())                            # [5.5, 2.0, 14.0, 804.0]
raw = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Xraw)
scaled = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(Xraw))
print(round(adjusted_rand_score(ytrue, raw), 3))                # 0.316   (body mass dominates)
print(round(adjusted_rand_score(ytrue, scaled), 3))            # 0.799   (all four count equally)

# %% [markdown]

# Body mass in grams has a standard deviation near 804, while bill depth in millimeters is near 2, a gap of roughly four hundred to one. Since distance sums squared differences, body mass alone accounts for almost the entire distance and the three bill and flipper measurements barely register, so clustering the raw numbers is effectively a one-feature clustering on body mass and scores an adjusted Rand index of only 0.316. Standardizing each feature to zero mean and unit variance first, so every measurement contributes equally, lifts the score to 0.799. This is why standardizing is a mandatory first step for k-means and for every distance-based method, not an optional polish.
#
# ![Four round blobs correctly colored by k-means cluster on the left; the two-moons data on the right with a straight k-means split cutting across both crescents.](figures/fig_08_01_kmeans-blobs-moons.png)
#
# Figure 8-1. k-means succeeds on round, similarly sized blobs (left) but fails on nonspherical clusters like two moons (right), where it cuts across the true shapes.
#
# ##### 8.1.2 Hierarchical clustering
#
# Where k-means demands the number of clusters up front, hierarchical clustering builds the whole family of clusterings at once and lets you choose afterward. The common form, agglomerative clustering, works bottom up: start with every point as its own cluster, then repeatedly merge the two closest clusters, until everything is one cluster. The record of those merges is a tree called a dendrogram, and cutting the tree at a chosen height yields any number of clusters you like. This matters because you decide $k$ after seeing the structure rather than guessing it, because the tree reveals nested groupings, subclusters within clusters, which is exactly what taxonomies, gene-expression studies, and document hierarchies need, and because the result is deterministic, with no random initialization.
#
# The one real choice is what "closest" means when comparing two clusters, called the linkage criterion, and it changes everything. For clusters $A$ and $B$ with $|A|$ and $|B|$ points, single linkage uses the nearest pair across the two, complete linkage the farthest pair, and average linkage the mean over all $|A||B|$ pairs:
#
# $$d_{\text{single}}(A,B) = \min_{\mathbf{a} \in A,\, \mathbf{b} \in B} \lVert \mathbf{a} - \mathbf{b} \rVert, \qquad d_{\text{complete}}(A,B) = \max_{\mathbf{a} \in A,\, \mathbf{b} \in B} \lVert \mathbf{a} - \mathbf{b} \rVert, \qquad d_{\text{average}}(A,B) = \frac{1}{|A||B|}\sum_{\mathbf{a} \in A}\sum_{\mathbf{b} \in B} \lVert \mathbf{a} - \mathbf{b} \rVert.$$
#
# Ward linkage (Ward, 1963), the default of scikit-learn's `AgglomerativeClustering`, merges the two clusters whose union least increases the total within-cluster sum of squares, the same objective k-means minimizes. With $\boldsymbol{\mu}_A$ and $\boldsymbol{\mu}_B$ the two centroids, that increase is
#
# $$\Delta(A,B) = \frac{|A||B|}{|A| + |B|}\,\lVert \boldsymbol{\mu}_A - \boldsymbol{\mu}_B \rVert^2.$$
#
# Take three points on a line, 0, 2, and 6. Every rule first joins 0 and 2, the closest pair, at distance 2. To join $A = \{0, 2\}$ with $B = \{6\}$, the two cross distances are 6 and 4, so single linkage gives 4, complete linkage 6, and average linkage $(6 + 4)/2 = 5$. For Ward, $\boldsymbol{\mu}_A = 1$ and $\boldsymbol{\mu}_B = 6$, so $\Delta = \tfrac{2 \times 1}{3}(1 - 6)^2 = 50/3 \approx 16.667$. The sums of squares confirm it: $\{0, 2\}$ holds $1^2 + 1^2 = 2$ about its mean 1, $\{6\}$ holds 0, and the merged $\{0, 2, 6\}$, with mean $8/3$, holds $64/9 + 4/9 + 100/9 = 168/9 \approx 18.667$, an increase of $18.667 - 2 = 16.667$.

# %%
import numpy as np
A, B = np.array([0.0, 2.0]), np.array([6.0])          # the cluster {0, 2} and the lone point 6
gaps = np.abs(A[:, None] - B[None, :])                # every distance across the two clusters
print(gaps.min(), gaps.max(), gaps.mean())            # single, complete, average: 4.0 6.0 5.0
def sse(c):                                           # within-cluster sum of squares
    return float(((c - c.mean()) ** 2).sum())
delta = len(A) * len(B) / (len(A) + len(B)) * (A.mean() - B.mean()) ** 2       # Ward's formula
print(round(float(delta), 3), round(sse(np.concatenate([A, B])) - sse(A) - sse(B), 3))  # 16.667 16.667

# %% [markdown]

# The first line prints 4.0, 6.0, and 5.0, the single, complete, and average linkage distances, and the second prints 16.667 twice: Ward's formula and the direct increase in the within-cluster sum of squares agree.
#
# On the wine recognition data that scikit-learn bundles from the UCI Machine Learning Repository (Aeberhard and Forina, 1992), 178 wines from three cultivars described by 13 chemical measurements, the choice ranks clearly.
#
# Example 8-4. Agglomerative clustering of the standardized wine data with single, complete, average, and Ward linkage, scored against the three cultivars.

# %%
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
wine = load_wine()
X = StandardScaler().fit_transform(wine.data)     # 13 chemical measurements, standardized
ytrue = wine.target                                # 3 cultivars as ground truth
saved = {}
for linkage in ["single", "complete", "average", "ward"]:
    labels = AgglomerativeClustering(n_clusters=3, linkage=linkage).fit_predict(X)
    sizes = sorted(int(np.sum(labels == j)) for j in np.unique(labels))
    saved[linkage] = labels
    print(f"{linkage:9} {adjusted_rand_score(ytrue, labels):.3f}  {sizes}")
avg = saved["average"]
big = int(np.bincount(avg).argmax())
out_idx = np.where(avg != big)[0]
print(sorted(int(i) for i in out_idx), np.max(np.abs(X[out_idx]), axis=1).round(2).tolist())
print(np.bincount(ytrue[avg == big], minlength=3).tolist())
# single    -0.007  [1, 3, 174]
# complete  0.577  [51, 58, 69]
# average   -0.005  [1, 3, 174]
# ward      0.790  [56, 58, 64]
# [25, 59, 73, 121] [3.12, 3.68, 3.15, 3.16]
# [58, 68, 48]

# %% [markdown]

# Ward leads at 0.790 because the three cultivars form compact, roughly convex groups in standardized chemical space, the shape its variance criterion fits. Single and average linkage score $-0.007$ and $-0.005$. Both print cluster sizes 1, 3, and 174: 174 of the 178 wines stay together, and the other four are peeled off. The last two lines examine the average-linkage run. Its large cluster still holds every cultivar, 58, 68, and 48 wines, and the four wines it peels off, indices 25, 59, 73, and 121, each have a maximum absolute standardized measurement above 3, printed as 3.12, 3.68, 3.15, and 3.16. Cutting the tree into three clusters stops before the last two merges, so these four outliers are the last points average linkage absorbs. Single linkage peels off a different set of four, sharing only wine 73 with average linkage. With `saved["single"]` in place of `saved["average"]`, the same two lines print wines 69, 73, 78, and 95, whose largest standardized measurements are 3.6, 3.15, 2.55, and 4.37, and a large cluster of 59, 67, and 48 wines. Complete linkage lands at 0.577 with sizes 51, 58, and 69. Single linkage's nearest-pair rule is a strength when clusters are not compact, because it follows a chain of nearby points and traces shapes that variance-based methods cut apart. On the two moons, again synthetic so the crescent shape is known exactly, it is perfect.

# %%
from sklearn.datasets import make_moons
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
Xm, ym = make_moons(n_samples=300, noise=0.06, random_state=0)
for linkage in ["ward", "single"]:
    labels = AgglomerativeClustering(n_clusters=2, linkage=linkage).fit_predict(Xm)
    print(f"{linkage} {adjusted_rand_score(ym, labels):.3f}")
# ward 0.297 (cuts across the crescents)   single 1.000 (follows each crescent)

# %% [markdown]

# Single linkage scores a perfect 1.000 on the moons, separating the two crescents flawlessly by following each one as a chain, where Ward, like k-means, manages only 0.297. The lesson is that the linkage criterion encodes an assumption about cluster shape just as surely as k-means does, so you choose it to match your data: Ward for compact blobs, single linkage for stringy, connected shapes. The dendrogram itself, shown in Figure 8-2, is read by drawing a horizontal line: its height is the distance at which clusters merged, and the number of vertical lines it crosses is the number of clusters at that cut, so a tall gap between merges is a natural place to cut.
#
# scipy computes the tree directly, and cutting it is one function call. `linkage` performs the agglomerative merges and returns them as a matrix with one row per merge whose third column is the merge distance, `dendrogram` draws that matrix, and `fcluster` cuts the tree, either into a chosen number of clusters or at a chosen height.

# %%
import matplotlib
matplotlib.use("Agg")                                 # headless: the plot is saved to a file
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
wine = load_wine()
X = StandardScaler().fit_transform(wine.data)         # the standardized wine data of Example 8-4
Z = linkage(X, method="ward")                         # the full merge history: 177 merges of 178 wines
print(Z.shape, Z[-3:, 2].round(1))                    # (177, 4) [12.6 27.7 35.4]
lab3 = fcluster(Z, t=3, criterion="maxclust")         # cut the tree into exactly 3 clusters
print(f"{adjusted_rand_score(wine.target, lab3):.3f}")            # 0.790, the same Ward partition
print(len(np.unique(fcluster(Z, t=20, criterion="distance"))))    # cutting at height 20 also gives 3
top = fcluster(Z, t=2, criterion="maxclust")          # the two clusters that the last merge joins
a, b = X[top == 1], X[top == 2]
delta = len(a) * len(b) / (len(a) + len(b)) * np.sum((a.mean(0) - b.mean(0)) ** 2)   # Ward's increase
print(len(a), len(b), round(float(delta), 2), round(float(np.sqrt(2 * delta)), 1))  # 56 122 626.63 35.4
fig, ax = plt.subplots(figsize=(8, 3.5))
dendrogram(Z, truncate_mode="lastp", p=12, ax=ax)     # draw the tree, collapsed to its last 12 merges
ax.set_ylabel("merge distance (Ward)")
fig.savefig("wine_ward_dendrogram.png", dpi=110)

# %% [markdown]

# The merge matrix's last three distances are 12.6, 27.7, and 35.4: the two biggest jumps come at the very end, which is the tall-gap rule in numeric form, saying the data resists being fused below three groups. Cutting the tree into three clusters with `fcluster` reproduces the Ward partition that Example 8-4 scored, adjusted Rand index 0.790 against the cultivars, and cutting by height instead, at a distance of 20, lands in the wide gap between 12.6 and 27.7 and yields the same three clusters. The last line ties the heights to Ward's formula. Cutting into two clusters recovers the two groups that the final merge joins, 56 and 122 wines; their $\Delta$ is 626.63, and $\sqrt{2\Delta} = 35.4$ is exactly the last merge height. scipy reports Ward merges on this $\sqrt{2\Delta}$ scale, which for two single points is just their distance, since then $\Delta = \tfrac{1}{2}\lVert \mathbf{a} - \mathbf{b} \rVert^2$. The saved plot is a dendrogram of the wine data itself, truncated to its last twelve merges so the top-level structure stays readable; Figure 8-2 shows the same kind of tree drawn in full on a small blob dataset. The chief limitation of hierarchical clustering is cost. scipy's `linkage` documentation gives $O(n^2)$ time for the single, complete, average, and Ward methods and $O(n^2)$ memory for every method (SciPy community, 2026h), and its input of pairwise distances alone holds $\binom{n}{2} = n(n-1)/2$ numbers, 15,753 for the 178 wines. So it does not scale to very large datasets the way k-means does, and its greedy merges can never be undone.
#
# ![Dendrogram of nested merges with a horizontal cut line crossing four vertical branches, large vertical gaps marking the natural cut heights.](figures/fig_08_02_dendrogram.png)
#
# Figure 8-2. A dendrogram from Ward linkage. Cutting horizontally at a chosen height sets the number of clusters; the large vertical gaps mark natural cut points.
#
#
# Hierarchical clustering can take a distance matrix that is not Euclidean. Gower distance, the dissimilarity form of the general similarity coefficient of Gower (1971), is one way to mix numeric and categorical fields. For each feature it scores a pair in $[0, 1]$, then averages. A numeric feature contributes $|x-y|/R$ with $R$ the range of that feature. A categorical feature contributes 0 when the labels match and 1 when they do not. On the first two penguins in the seaborn table the average over four measurements plus island and sex is 0.211. Average-linkage clustering on that matrix, scored against species, reaches an adjusted Rand index of 0.543. One-hot encodings of island and sex, stacked on the standardized measurements and clustered with k-means, reach 0.553. The two are close here. The point of Gower is that each field contributes one term in $[0, 1]$, instead of a one-hot column whose scale depends on how the encoder and the standardizer treat it.
#
# Example 8-5. A Gower distance matrix built by hand from the penguins' four measurements, island, and sex, clustered with average linkage and compared with k-means on the standardized measurements plus one-hot island and sex.

# %%
import numpy as np
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
pen = sns.load_dataset("penguins").dropna()
num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
cat_cols = ["island", "sex"]
num = pen[num_cols].to_numpy(float)
ranges = num.max(0) - num.min(0)
D = np.zeros((len(pen), len(pen)))
for f in range(num.shape[1]):
    D += np.abs(num[:, f][:, None] - num[:, f][None, :]) / ranges[f]
for col in cat_cols:
    vals = pen[col].to_numpy()
    D += (vals[:, None] != vals[None, :]).astype(float)
D /= num.shape[1] + len(cat_cols)
np.fill_diagonal(D, 0.0)
y = pen["species"].to_numpy()
lab = AgglomerativeClustering(n_clusters=3, metric="precomputed", linkage="average").fit_predict(D)
oh = OneHotEncoder(sparse_output=False).fit_transform(pen[cat_cols])
Xk = np.column_stack([StandardScaler().fit_transform(num), oh])
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Xk)
print(round(float(D[0, 1]), 3), round(adjusted_rand_score(y, lab), 3), round(adjusted_rand_score(y, km), 3))
# 0.211 0.543 0.553

# %% [markdown]

# The three printed values are 0.211, 0.543, and 0.553.
#
# ##### 8.1.3 DBSCAN: density-based clustering
#
# DBSCAN, density-based spatial clustering of applications with noise (Ester, Kriegel, Sander, and Xu, 1996), defines a cluster not as a ball around a center but as a dense region of points. Wherever points are packed tightly together it grows a cluster; wherever they are sparse it leaves them out, labeling them noise. This single shift in definition gives DBSCAN three abilities k-means lacks: it discovers the number of clusters on its own, it finds clusters of any shape, and it explicitly flags outliers rather than forcing every point into a group. It is the natural tool for spatial data and for clustering in the presence of noise.
#
# The method rests on two parameters and three kinds of point. The radius $\varepsilon$ defines a neighborhood around each point, and `min_samples` is how many points must lie inside that radius for the point to count as dense. The count includes the point itself, as the scikit-learn 1.9.1 `DBSCAN` documentation states (scikit-learn developers, 2026d, DBSCAN API reference), so `min_samples = 5` means the point plus four others. A core point has at least `min_samples` points within $\varepsilon$, itself included. A border point lies within $\varepsilon$ of a core point but is not itself core. A noise point is neither. A cluster is then a maximal group of core points that are reachable from one another through overlapping neighborhoods, together with the border points attached to them, and everything left over is noise. Because clusters grow by following local density rather than radiating from a center, they bend around any shape, which is why DBSCAN separates the two moons perfectly.
#
# Example 8-6. DBSCAN on the two moons, given no number of clusters, against k-means with two clusters.

# %%
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
Xm, ym = make_moons(n_samples=400, noise=0.06, random_state=0)
db = DBSCAN(eps=0.2, min_samples=5).fit(Xm)               # no k given
n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print(n_clusters, f"{adjusted_rand_score(ym, db.labels_):.3f}")   # 2 1.000
print(round(adjusted_rand_score(ym, KMeans(2, n_init=10, random_state=0).fit_predict(Xm)), 3))  # 0.274

# %% [markdown]

# DBSCAN finds exactly two clusters without being told there are two, and matches the true crescents at ARI 1.000, against k-means at 0.274. Its second gift is outlier detection: points in sparse regions get the label $-1$ instead of being absorbed into a cluster.
#
# Example 8-7. DBSCAN as an outlier detector: three blobs plus 40 scattered points, with the noise labels counted among the injected points and the blob points.

# %%
import numpy as np                                        # continuing with the DBSCAN import of Example 8-6
from sklearn.datasets import make_blobs
rng = np.random.default_rng(0)
Xb, _ = make_blobs(n_samples=400, centers=3, cluster_std=0.5, random_state=0)
noise = rng.uniform(Xb.min(0) - 1, Xb.max(0) + 1, size=(40, 2))   # 40 scattered outliers
Xn = np.vstack([Xb, noise])
labels = DBSCAN(eps=0.4, min_samples=6).fit_predict(Xn)
print(int(np.sum(labels == -1)), "noise labels out of", len(Xn))
print(int(np.sum(labels[400:] == -1)), "of the 40 injected points")
print(int(np.sum(labels[:400] == -1)), "of the 400 blob points")
# 33 noise labels out of 440
# 23 of the 40 injected points
# 10 of the 400 blob points

# %% [markdown]

# Faced with three blobs and 40 scattered points, DBSCAN labels 33 of the 440 points as noise. Of those 33, 23 are among the 40 injected points, so 17 injected points land inside or next to a blob, and the other 10 noise labels are sparse points on the blob edges. Those noise labels are points k-means would have pulled into a cluster. This makes DBSCAN double as an anomaly detector, a theme 8.4 returns to.
#
# The price is sensitivity to $\varepsilon$, which must match the data's density. One way to choose it, proposed with the method by Ester et al. (1996), is the sorted k-distance plot: compute each point's distance to its `min_samples`-th nearest neighbor, counting the point itself so the nearest of those distances is zero, sort those distances, and look for the elbow where they shoot up, which separates in-cluster distances from outlier distances. Getting it wrong collapses the result. On these moons the distances run from 0.020 to 0.166, and the 90th percentile is 0.105. The radius 0.2 that separates the crescents sits above every one of those distances, which is why that setting flags no noise.
#
# Example 8-8. The sorted k-distances of the moons, each point's distance to its fifth nearest neighbor counting itself, printed at the minimum, the 90th percentile, and the maximum.

# %%
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors
Xm, _ = make_moons(n_samples=400, noise=0.06, random_state=0)
nn = NearestNeighbors(n_neighbors=5).fit(Xm)
dist, _ = nn.kneighbors(Xm)
kdist = np.sort(dist[:, -1])
print(f"{kdist[0]:.3f}", f"{float(np.quantile(kdist, 0.90)):.3f}", f"{kdist[-1]:.3f}")
# 0.020 0.105 0.166

# %% [markdown]

# The printed distances are 0.020, 0.105, and 0.166.

# %%
# continuing with the moons Xm of Example 8-8 and the DBSCAN imports of Example 8-6
for eps in [0.10, 0.20, 0.50]:
    lab = DBSCAN(eps=eps, min_samples=5).fit_predict(Xm)
    nc = len(set(lab)) - (1 if -1 in lab else 0)
    print(eps, nc, round(np.mean(lab == -1), 2))
# 0.1 4 0.02   (too small: 4 clusters, some points marked noise)
# 0.2 2 0.0    (right: 2 clusters, no noise)
# 0.5 1 0.0    (too large: 1 cluster)

# %% [markdown]

# At $\varepsilon = 0.10$ the radius is too small and the moons fragment into four pieces with stray noise; at $0.50$ it is so large that both moons fuse into one blob; only near $0.20$ does the structure appear. DBSCAN's other real weakness is clusters of differing density: a single $\varepsilon$ cannot be right for both a dense cluster and a sparse one at once. HDBSCAN, the hierarchical version of Campello, Moulavi, and Sander (2013), fixes this by effectively trying all values of $\varepsilon$, building a hierarchy of density levels and extracting the clusters that persist across the widest range, so it handles varying density and needs only a minimum cluster size rather than a radius. It has been built into scikit-learn as `sklearn.cluster.HDBSCAN` since version 1.3, released in June 2023, whose release notes describe it as very robust to its hyperparameter values and usable on a wide variety of data with little or no tuning (scikit-learn developers, 2023), so trying it costs nothing, and one construction shows exactly the case it exists for: two tight blobs standing close together next to one diffuse blob.

# %%
from sklearn.cluster import HDBSCAN     # continuing with the imports of Examples 8-6 and 8-7
Xv, yv = make_blobs(n_samples=[300, 300, 300], centers=[[0, 0], [3, 0], [9, 5]],
                    cluster_std=[0.4, 0.4, 2.2], random_state=0)   # two tight blobs + one diffuse
for eps in [0.3, 1.2]:                                   # no single radius fits both densities
    db2 = DBSCAN(eps=eps, min_samples=10).fit(Xv)
    nc = len(set(db2.labels_)) - (1 if -1 in db2.labels_ else 0)
    print("eps", eps, "->", nc, "clusters,", int((db2.labels_ == -1).sum()), "noise points")
hdb = HDBSCAN(min_cluster_size=20, copy=True).fit(Xv)               # no radius at all, only a minimum size
nc = len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0)
print("HDBSCAN ->", nc, "clusters,", int((hdb.labels_ == -1).sum()), "noise,",
      "ARI", round(adjusted_rand_score(yv, hdb.labels_), 3))
# eps 0.3 -> 2 clusters, 316 noise points   eps 1.2 -> 2 clusters, 21 noise points
# HDBSCAN -> 3 clusters, 51 noise, ARI 0.928

# %% [markdown]

# The two DBSCAN runs fail in opposite directions. At $\varepsilon = 0.3$ the small radius resolves the tight pair but calls the entire diffuse cluster noise, 316 points thrown away; at $\varepsilon = 1.2$ the large radius holds the diffuse cluster together but fuses the tight pair into one, two clusters again with almost no noise. HDBSCAN, told nothing but a minimum cluster size of 20, recovers all three groups at an adjusted Rand index of 0.928 with 51 genuine edge points set aside as noise, which is precisely the varying-density rescue the hierarchy of density levels buys. Like all distance-based methods, DBSCAN and HDBSCAN both degrade in very high dimensions where the curse of dimensionality of 6.1 flattens density. Figure 8-3 shows DBSCAN separating the moons and isolating outliers among blobs.
#
# ![The two moons each colored as its own DBSCAN cluster on the left; blobs on the right with scattered points marked as noise.](figures/fig_08_03_dbscan.png)
#
# Figure 8-3. DBSCAN clusters by density: it separates the nonspherical moons (left) and isolates scattered outliers as noise among blobs (right), neither of which k-means can do.
#
# ##### 8.1.4 Gaussian mixture models and the EM algorithm
#
# k-means makes hard decisions, every point belongs fully to one cluster, and round ones, every cluster is a sphere. A Gaussian mixture model relaxes both. It assumes the data was generated by a mixture of $k$ Gaussian distributions, and instead of assigning each point to one of them, it gives each point a probability of belonging to each, a soft assignment. Because each Gaussian carries a full covariance matrix, clusters can be elliptical and tilted rather than round. This buys three things: soft membership, useful when points genuinely sit between groups; flexible shapes; and a true generative model, one you can sample new data from. The algorithm that fits it, expectation-maximization (Dempster, Laird, and Rubin, 1977), is a general method for fitting models with hidden variables, so its reach goes far beyond clustering.
#
# Section 2.3.4 derived a mixture density and, with Bayes' theorem, the posterior probability of each component, which this subsection calls the responsibility. The model here says the probability of a point is a weighted sum of Gaussians,
#
# $$p(\mathbf{x}) = \sum_{c=1}^{k} \pi_c\, \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_c, \Sigma_c), \qquad \sum_{c=1}^{k}\pi_c = 1,$$
#
# where $\pi_c$ is the mixing weight, the prior probability of cluster $c$, and $\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_c, \Sigma_c)$ is the multivariate normal density of 6.3 with mean $\boldsymbol{\mu}_c$ and covariance $\Sigma_c$. The difficulty is that we know neither which Gaussian produced each point nor the Gaussians' parameters, a chicken-and-egg problem identical in spirit to k-means. Expectation-maximization breaks it by alternating two steps. The E-step, holding the parameters fixed, computes the responsibility of each cluster for each point, the posterior probability that point $i$ came from cluster $c$:
#
# $$\gamma_{ic} = \frac{\pi_c\, \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_c, \Sigma_c)}{\sum_{j=1}^{k} \pi_j\, \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_j, \Sigma_j)}.$$
#
# The M-step, holding the responsibilities fixed, reestimates each Gaussian as a responsibility-weighted average, where $N_c = \sum_i \gamma_{ic}$ is the effective number of points in cluster $c$:
#
# $$\pi_c = \frac{N_c}{n}, \qquad \boldsymbol{\mu}_c = \frac{1}{N_c}\sum_i \gamma_{ic}\mathbf{x}_i, \qquad \Sigma_c = \frac{1}{N_c}\sum_i \gamma_{ic}(\mathbf{x}_i - \boldsymbol{\mu}_c)(\mathbf{x}_i - \boldsymbol{\mu}_c)^\top.$$
#
# These are the ordinary formulas for a Gaussian's weight, mean, and covariance, except every point contributes in proportion to its responsibility rather than all or nothing. All three come from one objective. With the responsibilities held fixed, the M-step maximizes the expected complete-data log likelihood, the log likelihood the model would have if each point's cluster were known, averaged over the responsibilities (Bishop, 2006, section 9.3.1):
#
# $$Q = \sum_{i=1}^{n}\sum_{c=1}^{k}\gamma_{ic}\Big[\ln \pi_c + \ln \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_c, \Sigma_c)\Big], \qquad \ln \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \Sigma) = -\frac{d}{2}\ln 2\pi - \frac{1}{2}\ln\lvert\Sigma\rvert - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top\Sigma^{-1}(\mathbf{x} - \boldsymbol{\mu}),$$
#
# where $d$ is the number of features and $\lvert\Sigma\rvert$ is the determinant of $\Sigma$. Only the last term involves $\boldsymbol{\mu}_c$, and the quadratic-form rule of 2.2.8, applied with the symmetric matrix $\Sigma_c^{-1}$, gives its gradient as $\Sigma_c^{-1}(\mathbf{x}_i - \boldsymbol{\mu}_c)$. Setting the total to zero and multiplying by $\Sigma_c$,
#
# $$\nabla_{\boldsymbol{\mu}_c} Q = \Sigma_c^{-1}\sum_{i}\gamma_{ic}(\mathbf{x}_i - \boldsymbol{\mu}_c) = \mathbf{0} \quad\Longrightarrow\quad \sum_i \gamma_{ic}\mathbf{x}_i = N_c\,\boldsymbol{\mu}_c \quad\Longrightarrow\quad \boldsymbol{\mu}_c = \frac{1}{N_c}\sum_i \gamma_{ic}\mathbf{x}_i.$$
#
# The weights must sum to 1, so they need the Lagrange multiplier of 2.6.6. They enter $Q$ only through $\sum_c N_c \ln \pi_c$, and setting the derivative of $\sum_c N_c\ln\pi_c - \lambda\big(\sum_c \pi_c - 1\big)$ with respect to $\pi_c$ to zero gives $N_c/\pi_c = \lambda$, so $\pi_c = N_c/\lambda$. Summing over $c$ gives $1 = \sum_c N_c/\lambda = n/\lambda$, because each point's responsibilities add up to 1 and so the $N_c$ add up to $n$. Hence $\lambda = n$ and $\pi_c = N_c/n$. The covariance follows from setting the derivative with respect to $\Sigma_c$ to zero, which needs the derivative of a log determinant; Bishop (2006, section 9.2.2) obtains it from the maximum-likelihood covariance of a single Gaussian. Plug in three points on a line, 0, 1, and 4, whose responsibilities for one cluster are 0.9, 0.6, and 0.1. Then $N_c = 1.6$, $\pi_c = 1.6/3 \approx 0.533$, $\mu_c = (0.9 \times 0 + 0.6 \times 1 + 0.1 \times 4)/1.6 = 1.0/1.6 = 0.625$, and the variance is $(0.9 \times 0.625^2 + 0.6 \times 0.375^2 + 0.1 \times 3.375^2)/1.6 = 1.575/1.6 \approx 0.984$.
#
# The two steps repeat, and the log likelihood $\ell = \sum_i \ln p(\mathbf{x}_i)$ never decreases from one round to the next, a property Dempster, Laird, and Rubin (1977) proved for EM in general. The proof needs only the KL divergence of 2.5.3. Take any probabilities $q_{ic}$ over the clusters, positive and summing to 1 for each point. Because $\pi_c\,\mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_c, \Sigma_c) = p(\mathbf{x}_i)\,\gamma_{ic}$, each point's log likelihood splits in two:
#
# $$\ln p(\mathbf{x}_i) = \sum_{c} q_{ic}\ln\frac{\pi_c\,\mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_c, \Sigma_c)}{q_{ic}} + \sum_c q_{ic}\ln\frac{q_{ic}}{\gamma_{ic}}.$$
#
# The second sum is a KL divergence in nats, never negative and zero exactly when $q_{ic} = \gamma_{ic}$. So the first sum, totaled over the points and called $F(q, \theta)$ with $\theta$ standing for all the parameters, is a lower bound on $\ell(\theta)$ that touches it when $q$ equals the responsibilities. The E-step sets $q$ to the responsibilities at the old parameters $\theta_{\text{old}}$, so $F(q, \theta_{\text{old}}) = \ell(\theta_{\text{old}})$. The M-step maximizes $F(q, \theta)$ over $\theta$, which is the same as maximizing $Q$, since the two differ by $-\sum_{i,c} q_{ic}\ln q_{ic}$, a term free of $\theta$. Then
#
# $$\ell(\theta_{\text{new}}) \;\ge\; F(q, \theta_{\text{new}}) \;\ge\; F(q, \theta_{\text{old}}) \;=\; \ell(\theta_{\text{old}}),$$
#
# where the first inequality is the bound at the new parameters and the second is the M-step. This lower-bound form of the argument follows Bishop (2006, section 9.4). It does not promise a strict rise at every step: Example 8-9 prints 19 rises, then a plateau, and no fall. Its points are drawn from two known Gaussians, so the true groups are known, which a real unlabeled set would not tell us. Example 8-9 implements both steps from scratch.
#
# Example 8-9. Expectation-maximization for a Gaussian mixture, from scratch.

# %%
import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
rng = np.random.default_rng(0)
# two elongated, perpendicular Gaussians: the case k-means mangles
A = rng.multivariate_normal([0, 0], [[6, 0], [0, 0.3]], 300)
B = rng.multivariate_normal([1, 4], [[0.3, 0], [0, 6]], 300)
X = np.vstack([A, B]); ytrue = np.r_[np.zeros(300), np.ones(300)].astype(int)

def em_gmm(X, k, iters=60):
    r = np.random.default_rng(0)
    means = X[r.choice(len(X), k, replace=False)].astype(float)   # random initial means
    covs = [np.eye(2) for _ in range(k)]; weights = np.ones(k) / k; lls = []
    for _ in range(iters):
        resp = np.column_stack([weights[c] * mvn.pdf(X, means[c], covs[c]) for c in range(k)])  # E-step
        total = resp.sum(1); lls.append(np.log(total).sum()); resp /= total[:, None]            # responsibilities
        Nk = resp.sum(0); weights = Nk / len(X)                                                  # M-step
        for c in range(k):
            means[c] = (resp[:, c, None] * X).sum(0) / Nk[c]
            diff = X - means[c]
            covs[c] = (resp[:, c, None, None] * np.einsum("ni,nj->nij", diff, diff)).sum(0) / Nk[c] + 1e-6 * np.eye(2)
    return resp, lls, weights

resp, lls, weights = em_gmm(X, 2)
print(round(lls[0], 1), round(lls[4], 1), round(lls[-1], 1), all(np.diff(lls) >= -1e-6))
print(int(np.sum(np.diff(lls) > 1e-4)), int(np.sum(np.diff(lls) < -1e-6)))
# -8664.8 -2507.1 -2208.3 True   (never decreases: it rises, then plateaus)
# 19 0   (19 of the 59 steps rise by more than 0.0001; none fall)
gm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(X)
print(weights.round(3).tolist(), gm.weights_.round(3).tolist())
# [0.492, 0.508] vs [0.505, 0.495]   (same mixture; the component order is arbitrary)
print(f"{adjusted_rand_score(ytrue, resp.argmax(1)):.3f}",
      f"{adjusted_rand_score(ytrue, gm.predict(X)):.3f}")    # 0.846 0.840

# %% [markdown]

# The log likelihood climbs from $-8664.8$ to $-2208.3$ and never decreases. The next print is 19 and 0: 19 of the 59 steps raise it by more than 0.0001, and none fall. After those rises the value plateaus at convergence, which is what the tolerance inside `all(np.diff(lls) >= -1e-6)` allows. That check, that no step falls, is the standard check that an EM implementation is correct. It matches scikit-learn closely: the from-scratch mixing weights of $[0.492, 0.508]$ sit beside scikit-learn's $[0.505, 0.495]$, the same mixture with the component order flipped, since EM numbers its clusters arbitrarily, and the two recover the true labels at adjusted Rand indices of 0.846 and 0.840. The soft assignment is visible at the boundary between clusters.

# %%
gm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(X)
grid = np.array([[x, y] for x in np.linspace(-3, 4, 40) for y in np.linspace(-2, 6, 40)])
amb = grid[np.argmin(np.abs(gm.predict_proba(grid)[:, 0] - 0.5))]   # the most 50/50 point
print(amb.round(2).tolist(), gm.predict_proba([amb])[0].round(2).tolist())
# [0.41, -1.18] [0.5, 0.5]   a soft membership, not a forced 0/1

# %% [markdown]

# A point sitting between the two Gaussians is assigned 50 percent to each, a statement of genuine uncertainty that k-means, forced to pick one, cannot make. And because each Gaussian has its own covariance, the GMM fits the two elongated, differently oriented clusters that k-means mangles.

# %%
print(f"{adjusted_rand_score(ytrue, gm.predict(X)):.3f}")   # 0.840 (elliptical fits)
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
print(round(adjusted_rand_score(ytrue, km.labels_), 3))      # k-means 0.503 (spheres miss)

# %% [markdown]

# The GMM scores 0.840 against k-means' 0.503, because its elliptical Gaussians wrap the stretched clusters that k-means tries to cover with circles, as Figure 8-4 illustrates. To choose the number of components, GMMs use the Bayesian information criterion, which rewards fit but penalizes extra parameters, and is read by its minimum.

# %%
for k in range(1, 5):
    bic = GaussianMixture(n_components=k, covariance_type="full", random_state=0).fit(X).bic(X)
    print(k, round(bic))
#  1 5286    2 4487  <- lowest, the true number    3 4526    4 4565

# %% [markdown]

# BIC bottoms out at $k=2$, the true number, then rises as extra components add complexity without enough fit to justify it.
#
# The covariance_type option controls how much shape freedom each Gaussian gets, and with it how many parameters the model fits, which is exactly what BIC weighs. Sweeping the four settings on the same two elongated clusters lays the tradeoff bare.

# %%
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
rng = np.random.default_rng(0)
A = rng.multivariate_normal([0, 0], [[6, 0], [0, 0.3]], 300)   # elongated along x
B = rng.multivariate_normal([1, 4], [[0.3, 0], [0, 6]], 300)   # elongated along y
X = np.vstack([A, B]); ytrue = np.r_[np.zeros(300), np.ones(300)].astype(int)
for ct in ["spherical", "diag", "tied", "full"]:
    gm = GaussianMixture(n_components=2, covariance_type=ct, random_state=0).fit(X)
    means, weights = 2 * 2, 2 - 1
    cov = {"spherical": 2, "diag": 4, "tied": 3, "full": 6}[ct]
    n_params = means + cov + weights
    print(f"{ct:10} BIC {gm.bic(X):6.0f}  params {n_params:2d}  "
          f"ARI {adjusted_rand_score(ytrue, gm.predict(X)):.3f}")
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
print(f"k-means    ARI {adjusted_rand_score(ytrue, km):.3f}")
# spherical  BIC   5117  params  7  ARI 0.280
# diag       BIC   4475  params  9  ARI 0.846
# tied       BIC   5128  params  8  ARI 0.375
# full       BIC   4487  params 11  ARI 0.840
# k-means    ARI 0.503

# %% [markdown]

# The four settings trade parameters for flexibility. Spherical gives each component a single variance, so its clusters are round, and uses the fewest parameters, seven. Tied forces all components to share one full covariance, eight parameters. Diag gives each component its own per-axis variances, nine, allowing axis-aligned ellipses. Full gives each its own complete covariance, eleven, allowing tilted ellipses. On these two perpendicular elongated clusters the round and shared shapes fail, spherical at an adjusted Rand index of 0.280 and tied at 0.375, both near or below k-means at 0.503, because a single round or single shared shape cannot wrap two clusters stretched in different directions. Diag and full succeed, at 0.846 and 0.840. BIC is lowest for diag, 4475, just under full's 4487: because the two clusters happen to be axis aligned, a diagonal covariance already captures them and full's extra off-diagonal parameters add cost without improving the fit, so BIC correctly prefers the simpler model. The spherical setting is the Gaussian mixture form closest to k-means, one variance per component and round clusters, which is why it fails on the same elongated groups, at 0.280 against k-means at 0.503, not at a similar score.
#
# The Bayesian information criterion and the Akaike information criterion turn the penalty for extra parameters into formulas. With $\hat L$ the maximized likelihood, $p$ the number of free parameters, and $n$ the number of points,
#
# $$\mathrm{BIC} = -2\ln \hat L + p\ln n, \qquad \mathrm{AIC} = -2\ln \hat L + 2p.$$
#
# Lower is better. For two full covariance components in two dimensions, $p = 11$. On this sample the fitted log likelihood is $-2208.3$, and both formulas match `bic` and `aic`: BIC 4487.0 and AIC 4438.7. Because $\ln 600 > 2$, BIC penalizes the extra parameters harder than AIC.

# %%
ll = gm.score(X) * len(X)
p = 11
print(f"{ll:.1f}", f"{-2 * ll + p * np.log(len(X)):.1f}", f"{gm.bic(X):.1f}",
      f"{-2 * ll + 2 * p:.1f}", f"{gm.aic(X):.1f}")
# -2208.3 4487.0 4487.0 4438.7 4438.7

# %% [markdown]

# Finally, GMMs unify the section: k-means is exactly a Gaussian mixture in the limit where every cluster shares the same spherical covariance $\sigma^2 I$ with $\sigma^2 \to 0$ and the soft responsibilities harden to all or nothing (Bishop, 2006, section 9.3.2). k-means is a special case of the GMM, which is why the GMM is the more flexible and the more illuminating model. The EM algorithm seen here recurs throughout the book, in hidden Markov models in Chapter 9 and wherever a model has hidden variables to infer.
#
# ![Elongated axis-aligned clusters wrapped by GMM covariance ellipses on the left, and the same data split incorrectly by round k-means cells on the right.](figures/fig_08_04_gmm-vs-kmeans.png)
#
# Figure 8-4. A GMM fits elongated, axis-aligned clusters with per-cluster covariance (left). The generating covariances are diagonal, so the ellipses are not tilted. k-means, restricted to spheres, splits the same groups incorrectly (right).
#
# ##### 8.1.5 Spectral clustering
#
# Spectral clustering takes a completely different view: it treats the data as a graph and clusters using linear algebra. The recipe, in the forms of Shi and Malik (2000) and of Ng, Jordan, and Weiss (2001) that von Luxburg (2007) surveys, is to connect each point to its nearby neighbors, forming a similarity graph, then to compute the eigenvectors of a matrix built from that graph, and finally to run k-means on those eigenvectors. The payoff is that the eigenvectors unfold tangled, nonconvex shapes into a new space where the clusters become simple blobs, so spectral clustering separates structures that defeat k-means and even GMMs, like two interlocking moons or two concentric rings.
#
# The construction goes through the graph Laplacian. From the similarity graph, let $A$ be its adjacency matrix, with $A_{ij} = 1$ when points $i$ and $j$ are neighbors, and let $D$ be the diagonal degree matrix whose entry $D_{ii}$ is the number of neighbors of point $i$. The unnormalized graph Laplacian is
#
# $$L = D - A.$$
#
# This matrix has a remarkable property: it is symmetric and positive semidefinite, its smallest eigenvalue is always zero, and the number of zero eigenvalues equals the number of connected components in the graph (von Luxburg, 2007, Propositions 1 and 2). So the eigenvectors belonging to the smallest eigenvalues encode how the graph splits into pieces. Spectral clustering takes the eigenvectors of the $k$ smallest eigenvalues, uses each point's entries in them as a new $k$-dimensional coordinate, and clusters those coordinates with k-means. On the two-moons graph the eigenvalues tell the story outright.
#
# For a value on each node, $\mathbf{x}^\top L \mathbf{x}$ equals the sum over edges of $(x_i - x_j)^2$. A constant vector makes every difference zero, so it is an eigenvector with eigenvalue 0: one connected piece, one zero eigenvalue. On a path of three nodes, $\mathbf{x} = (1, 0, 0)$ touches one edge and the sum is $(1-0)^2 = 1$.

# %%
import numpy as np
A = np.array([[0.0, 1, 0], [1, 0, 1], [0, 1, 0]])
L = np.diag(A.sum(1)) - A
x = np.array([1.0, 0, 0])
print(np.round(L @ np.ones(3), 1).tolist(), f"{float(x @ L @ x):.1f}")
# [0.0, 0.0, 0.0] 1.0

# %% [markdown]

# The constant vector prints `[0.0, 0.0, 0.0]` and the quadratic form prints 1.0.
#
# Example 8-10. The unnormalized graph Laplacian of a ten-neighbor graph on the two moons: its five smallest eigenvalues, the gaps among the six smallest, and the count of zero eigenvalues.

# %%
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neighbors import kneighbors_graph
Xm, ym = make_moons(n_samples=400, noise=0.05, random_state=0)
A = kneighbors_graph(Xm, n_neighbors=10, mode="connectivity").toarray()
A = np.maximum(A, A.T)                         # symmetric adjacency matrix
L = np.diag(A.sum(1)) - A                       # unnormalized graph Laplacian, L = D - A
eigvals = np.sort(np.linalg.eigvalsh(L))
print(eigvals[:5].round(4))                     # [0.     0.     0.0177 0.0201 0.0634]
print(" ".join(f"{g:.4f}" for g in np.diff(eigvals[:6])))
# 0.0000 0.0177 0.0024 0.0432 0.0144
print(int(np.sum(eigvals < 1e-8)))              # 2  = number of connected components

# %% [markdown]

# The first line prints `[0.     0.     0.0177 0.0201 0.0634]`: two eigenvalues are zero to four decimals, correctly reporting two connected components. The printed gaps between the first six eigenvalues are 0.0000, 0.0177, 0.0024, 0.0432, and 0.0144, so the largest gap sits after the fourth eigenvalue, not after the second. The count of exact zeros is what picks two, because that count equals the number of connected components.
#
# The library version does not use $L$ itself. Two normalized Laplacians divide out the degrees,
#
# $$L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}, \qquad L_{\text{rw}} = D^{-1} L = I - D^{-1} A,$$
#
# and each has as many zero eigenvalues as the graph has connected components (von Luxburg, 2007, Proposition 4). They are tied by one fact: $\mathbf{u}$ is an eigenvector of $L_{\text{rw}}$ with eigenvalue $\lambda$ exactly when $D^{1/2}\mathbf{u}$ is an eigenvector of $L_{\text{sym}}$ with the same $\lambda$ (von Luxburg, 2007, Proposition 3). Normalizing matters when degrees differ. Von Luxburg (2007, section 8.5) notes that when most points have about the same degree the three Laplacians behave alike, that they differ considerably when the degrees are spread widely, and that in her assessment the arguments favor normalized spectral clustering with $L_{\text{rw}}$. scikit-learn's `SpectralClustering` takes that route: its 1.9.1 source describes it as clustering a projection of the normalized Laplacian, solves $L_{\text{sym}}\mathbf{x} = \lambda\mathbf{x}$, and recovers $\mathbf{u} = D^{-1/2}\mathbf{x}$, an eigenvector of $L_{\text{rw}}$ (scikit-learn developers, 2026d, SpectralClustering, source file sklearn/cluster/_spectral.py). The next block builds both on the same moons graph.

# %%
deg = A.sum(1)                                  # continuing with A and L from Example 8-10
D_half = np.diag(1 / np.sqrt(deg))              # D^(-1/2)
L_sym = D_half @ L @ D_half                     # symmetric normalized Laplacian
vals, vecs = np.linalg.eigh(L_sym)              # eigenvalues in ascending order
print(int(deg.min()), int(deg.max()))           # the degrees run from 10 to 18
print(vals[:4].round(4))                        # [0.     0.     0.0015 0.0017]
print(int(np.sum(vals < 1e-8)))                 # 2 zero eigenvalues, as for L
U = vecs[:, :4] / np.sqrt(deg)[:, None]         # u = D^(-1/2) x for the first four eigenvectors
L_rw = np.diag(1 / deg) @ L                     # random-walk normalized Laplacian
print(np.allclose(L_rw @ U, U * vals[:4]))      # True: each u is an eigenvector of L_rw

# %% [markdown]

# The degrees run from 10 to 18, so normalizing does change the matrix. $L_{\text{sym}}$ still has exactly two zero eigenvalues, its first four print as `[0.     0.     0.0015 0.0017]`, and the last line prints True: rescaling the first four eigenvectors of $L_{\text{sym}}$ by $D^{-1/2}$ gives eigenvectors of $L_{\text{rw}}$ with the same eigenvalues, the kind of vectors scikit-learn clusters. Clustering those eigenvectors recovers the moons exactly, and the same works for nested circles.

# %%
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.datasets import make_circles
from sklearn.metrics import adjusted_rand_score
for name, (Xd, yd) in {"moons":   make_moons(n_samples=400, noise=0.05, random_state=0),
                       "circles": make_circles(n_samples=400, noise=0.04, factor=0.5, random_state=0)}.items():
    sc = SpectralClustering(2, affinity="nearest_neighbors", n_neighbors=10, random_state=0).fit_predict(Xd)
    km = KMeans(2, n_init=10, random_state=0).fit_predict(Xd)
    print(name, f"{adjusted_rand_score(yd, sc):.3f}", f"{adjusted_rand_score(yd, km):.3f}")
# moons 1.000 0.274     circles 1.000 -0.002  (k-means cannot touch nested rings)

# %% [markdown]

# Besides the two result lines, the block prints scikit-learn's warning "Graph is not fully connected, spectral embedding may not work as expected" twice, once per shape. The warning means what it says: the ten-neighbor graph is not connected. For the moons that is the two-component split that Example 8-10 counted, so here the warning describes the structure spectral clustering exploits, not a failure of the code. Spectral clustering scores a perfect 1.000 on both shapes, where k-means manages 0.274 on the moons and an adjusted Rand index of $-0.002$, about zero, the chance level on the concentric circles, which it cannot touch because no center-based grouping separates an inner ring from an outer one. The reason spectral clustering succeeds is that the graph connects points through chains of neighbors, so two points on the same ring are linked even though they are far apart in straight-line distance, and the eigenvectors place linked points close together. The cost is scale: building an $n \times n$ graph and computing its eigenvectors is expensive, fine for thousands of points but impractical for millions, and you must still choose the number of clusters and the graph's neighborhood size. Spectral clustering is the bridge between clustering and the spectral graph theory and eigenvector methods that also underlie the PCA of the next section.
#
# ##### 8.1.6 Evaluating a clustering: internal and external metrics
#
# Clustering has no labels, which raises a question supervised learning never faces: with no right answer to compare against, how do you know a clustering is any good? There are two families of answer. Internal metrics judge a clustering using only the data and the cluster assignments, rewarding clusters that are compact and well separated, and they are what you use in real unlabeled work, including to choose the number of clusters. External metrics compare a clustering to a known true grouping, which you only have on a labeled benchmark, so they are used to validate an algorithm, never to score a clustering of data you actually have no labels for.
#
# The scikit-learn user guide describes three internal metrics (scikit-learn developers, 2026d, sections 2.3.11.5 to 2.3.11.7), and they often agree. The silhouette score of 8.1.1 (Rousseeuw, 1987), from $-1$ to $1$, measures for each point how much closer it is to its own cluster than to the nearest other one, higher being better. Its formula makes that comparison exact. For point $i$, let $a(i)$ be the mean distance from $i$ to the other points of its own cluster, and let $b(i)$ be the smallest, taken over the other clusters, of the mean distance from $i$ to that cluster's points, the distance to the nearest rival cluster. The point's silhouette is
#
# $$s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}},$$
#
# and the silhouette score of a clustering is the average of $s(i)$ over all points. Plugging in numbers shows the scale: a point whose own cluster sits 0.4 away on average while the nearest rival sits 1.2 away scores $s = (1.2 - 0.4)/\max(0.4, 1.2) = 0.8/1.2 \approx 0.67$, comfortably positive; a point equally far from both, $a = b$, scores exactly 0; and a point closer to the rival than to its own cluster, say $a = 1.2$ and $b = 0.4$, scores $(0.4 - 1.2)/1.2 \approx -0.67$, flagging a likely misassignment. The Davies-Bouldin index (Davies and Bouldin, 1979) measures the average ratio of within-cluster scatter to between-cluster distance, so lower is better. The Calinski-Harabasz index (Caliński and Harabasz, 1974) is the ratio of between-cluster to within-cluster variance, so higher is better. With $s_i$ the mean distance of cluster $i$'s points from its centroid $\boldsymbol\mu_i$,
#
# $$\mathrm{DB} = \frac{1}{k}\sum_i\max_{j\neq i}\frac{s_i+s_j}{\lVert\boldsymbol\mu_i-\boldsymbol\mu_j\rVert}.$$
#
# With $B$ the weighted scatter of the centroids about the grand mean and $W$ the pooled within-cluster sum of squares,
#
# $$\mathrm{CH} = \frac{\mathrm{tr}(B)/(k-1)}{W/(n-k)}.$$
#
# On the four blobs at $k = 4$ both formulas match the library, 0.590 and 3788.

# %%
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
X, _ = make_blobs(n_samples=1500, centers=4, cluster_std=0.8, random_state=0)
lab = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(X)
k = 4
cent = np.vstack([X[lab == j].mean(0) for j in range(k)])
s = np.array([np.linalg.norm(X[lab == j] - cent[j], axis=1).mean() for j in range(k)])
db = [max((s[i] + s[j]) / np.linalg.norm(cent[i] - cent[j]) for j in range(k) if j != i) for i in range(k)]
overall = X.mean(0)
trB = sum(int((lab == j).sum()) * np.sum((cent[j] - overall) ** 2) for j in range(k))
W = sum(((X[lab == j] - cent[j]) ** 2).sum() for j in range(k))
ch = (trB / (k - 1)) / (W / (len(X) - k))
print(f"{np.mean(db):.3f}", f"{ch:.0f}")
# 0.590 3788

# %% [markdown]

# Running all three across candidate values of $k$ on four synthetic blobs, where the true $k$ is known so the metrics can be checked, they point to the same answer.

# %%
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
X, y_true = make_blobs(n_samples=1500, centers=4, cluster_std=0.8, random_state=0)  # known k = 4
for k in [2, 3, 4, 5, 6]:
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    print(k, f"{silhouette_score(X, labels):.3f}",
             f"{davies_bouldin_score(X, labels):.3f}",
             f"{calinski_harabasz_score(X, labels):.0f}")
#  k  silhouette  Davies-Bouldin  Calinski-Harabasz
#  2    0.515        0.613            1881
#  3    0.540        0.671            2707
#  4    0.583        0.590            3788   <- best on all three metrics
#  5    0.497        0.800            3213
#  6    0.466        0.893            2914

# %% [markdown]

# The five printed rows agree on the answer. At the true number of clusters, four, the silhouette peaks at 0.583 after rising from 0.515, the Davies-Bouldin index bottoms out at 0.590 before climbing to 0.893, and the Calinski-Harabasz index peaks at 3788, so all three independently recover the right $k$, and each metric visibly worsens on both sides of it. When they agree like this you can trust the choice; when they disagree, the silhouette has the advantage that it can be read per point and per cluster, but it is not always the right guide to a labeled structure: on the penguins in 8.1.1 it peaks at two clusters while the species count is three.
#
# The silhouette can also be read per cluster. On the standardized penguins, k = 2 gives cluster means 0.486 and 0.612 (overall 0.531). k = 3 gives 0.305, 0.567, and 0.428 (overall 0.446). The third cluster is the less compact one, which is why the average falls even though one cluster stays tight.

# %%
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
pen = sns.load_dataset("penguins").dropna()
X = StandardScaler().fit_transform(pen[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])
for k in (2, 3):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    s = silhouette_samples(X, lab)
    means = [round(float(s[lab == j].mean()), 3) for j in range(k)]
    print(k, means, round(float(s.mean()), 3))
# 2 [0.486, 0.612] 0.531
# 3 [0.305, 0.567, 0.428] 0.446

# %% [markdown]

# External metrics enter when you do have a ground-truth grouping, typically to test whether a clustering algorithm recovers known classes on a benchmark. The adjusted Rand index of Hubert and Arabie (1985) counts how often pairs of points are grouped the same way in the clustering and the truth, corrected for chance so that a random clustering scores about zero rather than a misleadingly positive number. The correction is worth seeing exactly. Lay the two labelings out in a contingency table whose entry $n_{ij}$ counts the points that fall in cluster $i$ of the truth and cluster $j$ of the clustering, with row sums $a_i$, column sums $b_j$, and $n$ points in all. The raw agreement counts the pairs placed together in both labelings, $\sum_{ij}\binom{n_{ij}}{2}$; under random labeling with the same cluster sizes its expected value is $E = \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} \big/ \binom{n}{2}$, and its maximum is $M = \tfrac{1}{2}\bigl[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\bigr]$. The adjusted Rand index rescales the raw agreement between those two anchors,
#
# $$\text{ARI} = \frac{\sum_{ij}\binom{n_{ij}}{2} - E}{M - E},$$
#
# so identical labelings score 1 and chance-level agreement scores about 0. A four-point example shows the correction bite. Take the truth $[0, 0, 1, 1]$ and the clustering $[0, 1, 1, 1]$. The contingency table has $n_{00} = 1$, $n_{01} = 1$, $n_{11} = 2$, so the raw agreement is $\binom{2}{2} = 1$ (the one pair inside the size-2 overlap), the row sums give $\sum_i\binom{a_i}{2} = 1 + 1 = 2$, the column sums give $\sum_j\binom{b_j}{2} = 0 + 3 = 3$, and $\binom{4}{2} = 6$. Then $E = 2 \cdot 3/6 = 1$ and $M = (2 + 3)/2 = 2.5$, making $\text{ARI} = (1 - 1)/(2.5 - 1) = 0$ exactly: this clustering agrees with the truth no more than chance would, and `adjusted_rand_score([0,0,1,1], [0,1,1,1])` confirms it, printing 0.0. The mutual-information metrics, in their normalized and adjusted forms (Vinh, Epps, and Bailey, 2010), measure how much knowing the cluster tells you about the true class. And the homogeneity, completeness, and V-measure trio of Rosenberg and Hirschberg (2007) splits the question in two: homogeneity asks whether each cluster contains only one true class, completeness asks whether each true class lands in a single cluster, and the V-measure is their harmonic mean.

# %%
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score
pen = sns.load_dataset("penguins").dropna()
feats = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = StandardScaler().fit_transform(pen[feats].to_numpy(float))
y_true = pen["species"].map({s: i for i, s in enumerate(sorted(pen["species"].unique()))}).to_numpy()
rng = np.random.default_rng(0)
labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
print(round(adjusted_rand_score(y_true, labels), 3))          # 0.799   pairs grouped as in the truth
print(round(adjusted_mutual_info_score(y_true, labels), 3))   # 0.789
print(f"{v_measure_score(y_true, labels):.3f}")              # 0.790   homogeneity-completeness harmonic mean
# random labels score ~0 because these metrics are corrected for chance:
print(round(adjusted_rand_score(y_true, rng.integers(0, 3, len(y_true))), 3))   # -0.002

# %% [markdown]

# The k-means clustering recovers the three penguin species well, scoring an adjusted Rand index of 0.799 and a V-measure of 0.790, while a random labeling scores essentially zero, $-0.002$, which is the whole point of the chance correction: it strips out the agreement you would get by luck, so a clearly positive score means real structure. The discipline to remember is which metric to use when. On the unlabeled data you are actually clustering, you have only internal metrics, so use the silhouette to choose $k$ and judge quality; reserve the external metrics for the rare case where you have true labels and want to test how well an algorithm recovers them.
#
# ##### 8.1.7 Mean-shift: climbing a density estimate to its peaks
#
# Mean-shift is a clustering method that pictures the data as a landscape of density, high where points crowd together and low where they thin out, and lets every point walk uphill until it stands on a peak, called a mode. Points that finish on the same peak form one cluster, so the number of clusters is the number of peaks the walkers reach. It belongs to the density family with the DBSCAN and HDBSCAN of 8.1.3, and like them it finds the number of clusters itself rather than being told $k$. That is the reason to learn it: k-means needs $k$, and DBSCAN needs a radius and a minimum count, while mean-shift needs one number, the bandwidth, which sets how wide a neighborhood each point looks at while it climbs. The landscape is worth knowing in its own right as well. It is a kernel density estimate, and 8.4.6 uses one to score unseen shuttle records for anomalies.
#
# The landscape is the kernel density estimate that 2.4.15 built: it places a bump on every point and adds the bumps. In one dimension, with observations $x_1, \ldots, x_n$, a kernel $K$, and bandwidth $h$,
#
# $$\hat f_h(x) = \frac{1}{nh}\sum_{i=1}^{n} K\left(\frac{x-x_i}{h}\right),$$
#
# and the standard normal kernel is $K(u) = (2\pi)^{-1/2}\exp(-u^2/2)$. On the points 0, 1, and 4, with $h = 1$, the three bumps contribute $K(1) = 0.2420$, $K(0) = 0.3989$, and $K(-3) = 0.0044$ at $x = 1$, so the estimate there is $(0.2420 + 0.3989 + 0.0044)/3 = 0.2151$. scikit-learn's `KernelDensity` prints the same value.

# %%
from math import exp, pi, sqrt
import numpy as np
from sklearn.neighbors import KernelDensity
pts = np.array([0.0, 1.0, 4.0])
def gauss(u):
    return exp(-0.5 * u * u) / sqrt(2 * pi)
hand = sum(gauss(1.0 - xi) for xi in pts) / 3
sk = float(np.exp(KernelDensity(bandwidth=1.0).fit(pts.reshape(-1, 1)).score_samples([[1.0]])[0]))
print(round(hand, 4), round(sk, 4))
# 0.2151 0.2151

# %% [markdown]

# Both prints are 0.2151. The seaborn geyser table of 2.3.4, Old Faithful eruption durations in minutes, has 272 duration values, mean 3.488. A Gaussian kernel with bandwidth 0.2, read on a grid from 1.5 to 5.5, rises near 2.0 and again near 4.5, with a valley at 3.0.

# %%
import numpy as np
import seaborn as sns
from sklearn.neighbors import KernelDensity
dur = sns.load_dataset("geyser")["duration"].to_numpy(float)
grid = np.linspace(1.5, 5.5, 9)
dens = np.exp(KernelDensity(bandwidth=0.2).fit(dur.reshape(-1, 1)).score_samples(grid.reshape(-1, 1)))
print(len(dur), round(float(dur.mean()), 3))
print(" ".join(f"{v:.3f}" for v in dens))
# 272 3.488
# 0.102 0.450 0.134 0.037 0.136 0.404 0.551 0.180 0.003

# %% [markdown]

# The first line prints 272 and 3.488. The second line is the nine grid densities at 1.5, 2.0, and so on up to 5.5: they peak at 0.450 at 2.0 and at 0.551 at 4.5, and the fourth value, 0.037 at 3.0, is the valley between the two peaks.
#
# A climber moves by averaging. Take three points, $(0, 0)$ and $(0.2, 0.1)$ close together and $(3, 3)$ far away, set $h = 1$, and start a climber at $(0.5, 0.4)$. Each data point gets a weight from the Gaussian kernel of its squared distance to the climber. The squared distances are 0.41, 0.18, and 13.01, so the weights are $e^{-0.41/2} = 0.8146$, $e^{-0.18/2} = 0.9139$, and $e^{-13.01/2} = 0.0015$. The climber then moves to the weighted average of the three points. Its first coordinate becomes $(0.8146 \times 0 + 0.9139 \times 0.2 + 0.0015 \times 3)/(0.8146 + 0.9139 + 0.0015) = 0.1873/1.7300 = 0.108$, and its second becomes $(0.8146 \times 0 + 0.9139 \times 0.1 + 0.0015 \times 3)/1.7300 = 0.0959/1.7300 = 0.055$. One step takes the climber from $(0.5, 0.4)$ to $(0.108, 0.055)$, toward the close pair, while the far point, with weight 0.0015, has almost no say.
#
# In general, take data points $\mathbf{x}_1, \ldots, \mathbf{x}_n$ in $d$ dimensions and a bandwidth $h > 0$. The Gaussian mean-shift step replaces the climber's position $\mathbf{x}$ by the weighted average
#
# $$\mathbf{m}(\mathbf{x}) = \frac{\sum_{i=1}^{n} w_i(\mathbf{x})\,\mathbf{x}_i}{\sum_{i=1}^{n} w_i(\mathbf{x})}, \qquad w_i(\mathbf{x}) = \exp\left(-\frac{\lVert \mathbf{x} - \mathbf{x}_i \rVert^2}{2h^2}\right),$$
#
# and repeats until a step moves the climber less than a small tolerance. The move $\mathbf{m}(\mathbf{x}) - \mathbf{x}$ is the mean shift that gives the method its name. To see why it points uphill, write the Gaussian kernel density estimate in $d$ dimensions as $\hat f(\mathbf{x}) = c\sum_{i=1}^{n} w_i(\mathbf{x})$, where the constant $c = 1/\big(n(2\pi)^{d/2}h^d\big)$ does not depend on $\mathbf{x}$. The chain rule gives $\nabla w_i(\mathbf{x}) = w_i(\mathbf{x})(\mathbf{x}_i - \mathbf{x})/h^2$, so
#
# $$\nabla \hat f(\mathbf{x}) = \frac{c}{h^2}\sum_{i=1}^{n} w_i(\mathbf{x})(\mathbf{x}_i - \mathbf{x}) = \frac{c}{h^2}\Big(\sum_{i=1}^{n} w_i(\mathbf{x})\Big)\big(\mathbf{m}(\mathbf{x}) - \mathbf{x}\big),$$
#
# and dividing by $\hat f(\mathbf{x})$ cancels both $c$ and the sum of the weights:
#
# $$\mathbf{m}(\mathbf{x}) - \mathbf{x} = h^2\,\frac{\nabla \hat f(\mathbf{x})}{\hat f(\mathbf{x})} = h^2\,\nabla \log \hat f(\mathbf{x}).$$
#
# The mean shift is the gradient of the log density, scaled by $h^2$, so every step heads up the slope of the density estimate. The shift is zero only where the gradient is zero, and in practice the climbs end at modes, the tops of local peaks. Every data point starts a climber of its own, and points whose climbers end at the same mode share a cluster. On the three-point example the shift is $(0.108 - 0.5, 0.055 - 0.4) = (-0.392, -0.345)$. Example 8-11 computes the step, checks it against a numerical gradient of $\log \hat f$, and then keeps climbing until the steps vanish.
#
# Example 8-11. Gaussian mean-shift from scratch on three points: one step, a check against the numerical gradient of the log density, the flat-kernel step, and the climb to a mode.

# %%
import numpy as np
data = np.array([[0.0, 0.0], [0.2, 0.1], [3.0, 3.0]])   # two close points and one far point
x = np.array([0.5, 0.4])                                 # where the climber starts
h = 1.0                                                  # the bandwidth
w = np.exp(-((data - x) ** 2).sum(1) / (2 * h * h))      # Gaussian weight of each point
m = (w[:, None] * data).sum(0) / w.sum()                 # weighted mean: one mean-shift step
print(" ".join(f"{v:.4f}" for v in w), " ".join(f"{v:.3f}" for v in m))

def log_kde(p):                                          # log of the Gaussian KDE; the constant c drops out
    return np.log(np.exp(-((data - p) ** 2).sum(1) / (2 * h * h)).sum())

e = 1e-6
grad = np.array([(log_kde(x + e * u) - log_kde(x - e * u)) / (2 * e) for u in np.eye(2)])
print(" ".join(f"{v:.4f}" for v in m - x), " ".join(f"{v:.4f}" for v in h * h * grad))
flat = data[np.linalg.norm(data - x, axis=1) <= h].mean(0)   # flat kernel: plain mean within h
print(" ".join(f"{v:.3f}" for v in flat))
for it in range(1, 101):                                 # climb until a step moves less than 1e-6
    w = np.exp(-((data - x) ** 2).sum(1) / (2 * h * h))
    nxt = (w[:, None] * data).sum(0) / w.sum()
    if np.linalg.norm(nxt - x) < 1e-6:
        break
    x = nxt
print(it, " ".join(f"{v:.4f}" for v in x))
# 0.8146 0.9139 0.0015 0.108 0.055
# -0.3918 -0.3446 -0.3918 -0.3446
# 0.100 0.050
# 5 0.1003 0.0503

# %% [markdown]

# The first line prints the three weights, 0.8146, 0.9139, and 0.0015, and the new position, 0.108 and 0.055, the hand values. The second line prints the shift, $-0.3918$ and $-0.3446$, and then $h^2$ times the numerical gradient of $\log \hat f$, the same two numbers, so the derivation checks out. The third line, 0.100 and 0.050, is the step a flat kernel takes, which the next paragraph explains. The last line prints 5 and the climber's final position: on the fifth pass a step would have moved it less than 0.000001, so the climb ends at $(0.1003, 0.0503)$. That mode sits beside $(0.1, 0.05)$, the midpoint of the close pair, pulled a hair toward the far point.
#
# scikit-learn's `MeanShift` changes the weights. Its docstring in scikit-learn 1.9.1 describes it as "mean shift clustering using a flat kernel" (scikit-learn developers, 2026d, MeanShift API reference): every point within distance $h$ of the climber gets weight 1 and every other point gets weight 0, so each step moves the climber to the plain average of the points inside that radius. From $(0.5, 0.4)$ with $h = 1$, only the close pair is within reach, at distances 0.64 and 0.42 against 3.61 for the far point, so the flat step lands on their midpoint, $(0.100, 0.050)$, beside the Gaussian step's $(0.108, 0.055)$. The source code supplies the rest of the procedure. By default every data point starts a climber. A climb stops once a step moves no more than 0.001 times the bandwidth, or after `max_iter` steps, 300 by default. Any end point that lies within one bandwidth of a more crowded end point is dropped, and each data point takes the label of the nearest surviving center. The bandwidth itself comes from `estimate_bandwidth`, which multiplies `quantile` by the number of points to get a neighbor count $k$ and returns the distance from each point to its $k$-th nearest neighbor, counting the point itself as the first, averaged over all points. For the 1,500 points of the four blobs that 8.1.6 scored, quantile 0.15 means $k = 225$. The docstring cites Comaniciu and Meer (2002) for the method.
#
# Example 8-12. scikit-learn's `MeanShift` on the four blobs of 8.1.6, with the bandwidth from `estimate_bandwidth` checked against the mean distance to the 225th neighbor.

# %%
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
X, y_true = make_blobs(n_samples=1500, centers=4, cluster_std=0.8, random_state=0)  # the 8.1.6 blobs
bw = estimate_bandwidth(X, quantile=0.15, random_state=0)     # neighborhood size from the data itself
dk = NearestNeighbors(n_neighbors=225).fit(X).kneighbors(X)[0][:, -1]   # 0.15 x 1500 = 225 neighbors
print(round(bw, 2), round(float(dk.mean()), 2), abs(bw - dk.mean()) < 1e-9)   # 1.42 1.42 True
ms = MeanShift(bandwidth=bw).fit(X)
print(round(bw, 2), len(np.unique(ms.labels_)),
      round(adjusted_rand_score(y_true, ms.labels_), 3))      # bandwidth 1.42, 4 clusters, ARI 0.928

# %% [markdown]

# The first line prints 1.42 twice and then True: the bandwidth that `estimate_bandwidth` returns is exactly the mean distance to the 225th neighbor. With that bandwidth mean-shift reports exactly 4 clusters, the true number, and matches the generating labels at an adjusted Rand index of 0.928, all without ever being told $k$. The bandwidth is doing the work $k$ does elsewhere, and it is just as much a judgment call: reestimating it with `quantile=0.2` widens the neighborhoods enough to merge two blobs into one, dropping to 3 clusters and an ARI of 0.681.

# %%
bw2 = estimate_bandwidth(X, quantile=0.2, random_state=0)
ms2 = MeanShift(bandwidth=bw2).fit(X)
print(round(bw2, 2), len(np.unique(ms2.labels_)), round(adjusted_rand_score(y_true, ms2.labels_), 3))
# 1.75 3 0.681

# %% [markdown]

# The block prints a bandwidth of 1.75, 3 clusters, and 0.681. Some shapes defeat every bandwidth. The scikit-learn docstring describes mean-shift as a search for blobs in a smooth density, and the two moons of 8.1.1 and 8.1.3, which DBSCAN separated perfectly, are not blobs. Sweeping the quantile on them shows the trap.

# %%
from sklearn.datasets import make_moons        # continuing with the imports of Example 8-12
Xm, ym = make_moons(n_samples=400, noise=0.06, random_state=0)   # the moons of 8.1.1 and 8.1.3
for q in (0.1, 0.2, 0.3):
    b = estimate_bandwidth(Xm, quantile=q, random_state=0)
    lab = MeanShift(bandwidth=b).fit_predict(Xm)
    print(q, round(float(b), 2), len(np.unique(lab)), round(adjusted_rand_score(ym, lab), 3))
# 0.1 0.35 11 0.176
# 0.2 0.68 3 0.36
# 0.3 0.86 1 0.0

# %% [markdown]

# No bandwidth recovers the crescents. At quantile 0.1 the bandwidth is 0.35 and the moons break into 11 clusters, an adjusted Rand index of 0.176. At 0.2 the bandwidth is 0.68 and 3 clusters score 0.36. At 0.3 the bandwidth is 0.86 and everything merges into 1 cluster, a score of 0.0. The prints fit a simple picture: a crescent is a long ridge of density with no single peak, so a narrow neighborhood finds several small peaks along each crescent, while a neighborhood wide enough to smooth them away also spans the gap to the other moon. DBSCAN avoids the trap because it follows chains of dense points instead of climbing to peaks.
#
# Mean-shift is intuitive and needs no $k$, and it carries two costs. It favors clusters gathered around a peak, as the moons show. And it is slow: the scikit-learn docstring warns that its running time grows toward the square of the number of samples in higher dimensions, and that `estimate_bandwidth`, which takes time at least quadratic in the number of samples, scales even worse than the clustering itself. Those costs are why the practical defaults of 8.1.8 reach for DBSCAN or HDBSCAN when shapes are irregular, and keep mean-shift for blobs whose number you do not know.
#
# ##### 8.1.8 Choosing a clustering method, and a self-check
#
# The methods encode different ideas of a cluster, and Table 8-1 summarizes when each fits. There is no universally best clustering algorithm, the unsupervised echo of the no-free-lunch theorem, so the choice follows the shape of your data and your needs: do you know the number of clusters, are they round or stringy, do you need soft memberships or outlier detection, and how large is the dataset.
#
# Table 8-1. Choosing a clustering method.
#
# | Method | Cluster idea | Needs k? | Shapes | Notes |
# | --- | --- | --- | --- | --- |
# | k-means | ball around a center | yes | round, similar size | fast, scales; sensitive to scale and init |
# | Minibatch k-means | ball around a center | yes | round, similar size | k-means for huge data, slightly worse inertia |
# | Hierarchical | nested merges | no (cut tree) | depends on linkage | dendrogram; $O(n^2)$, no huge data |
# | DBSCAN | dense region | no | any shape | finds outliers; struggles with varying density |
# | HDBSCAN | dense region at every density level | no | any shape | handles varying density; needs only a minimum cluster size |
# | Mean-shift | density peak (mode) | no | roundish blobs | finds k itself; slow on large data |
# | GMM | Gaussian blob | yes | elliptical | soft assignment; generative; BIC picks k |
# | Spectral | graph connectivity | yes | any shape | great on nonconvex; $O(n^2)$, no huge data |
#
# The practical defaults: start with k-means for speed and round clusters, or minibatch k-means when the data is huge, switch to a GMM when clusters are elliptical or you need soft memberships, reach for DBSCAN or HDBSCAN when shapes are irregular or outliers matter, use spectral clustering for nonconvex shapes on modest data, mean-shift when you want a density method that picks $k$ for you, and hierarchical clustering when you want to see nested structure in a dendrogram. Evaluate with the internal metrics of 8.1.6, above all the silhouette, when you have no labels, and with the external metrics like the adjusted Rand index only when a ground truth is available.
#
# Try these before moving on, with full solutions in Appendix H.
#
# 1. What objective does k-means minimize, and why does it fail on nonspherical clusters?
# 2. How do the elbow and silhouette methods each suggest a value of $k$, and what should you do when they disagree, as they did on the penguins?
# 3. How does the linkage criterion change hierarchical clustering, and why does single linkage succeed on the two moons where Ward fails?
# 4. What are core, border, and noise points in DBSCAN, and what two abilities does DBSCAN have that k-means lacks?
# 5. Write the E-step and M-step of EM for a Gaussian mixture, and explain what a responsibility is.
# 6. In what precise sense is k-means a special case of a Gaussian mixture model?
# 7. Why can spectral clustering separate two concentric circles when k-means cannot, and what does the number of zero eigenvalues of the graph Laplacian tell you?
# 8. What is the difference between an internal and an external clustering metric, and which kind can you actually use on data you have no labels for?
# 9. Name one internal metric where higher is better and one where lower is better, and say what "corrected for chance" means for the adjusted Rand index.
# 10. Which clustering methods in this chapter discover the number of clusters themselves, and what does minibatch k-means trade away for its speed?
# 11. Penguin body mass in grams has a standard deviation near 800 while bill depth in millimeters is near 2. Why does running k-means on these raw features score far worse against the species than running it on standardized features, and what exactly does standardizing change?
# 12. A Gaussian mixture offers four covariance_type settings: spherical, diag, tied, and full. Which one reproduces the round-cluster assumption of k-means, and how do the four differ in the number of parameters they fit and the cluster shapes they allow?
# 13. In Gower distance, what does one categorical feature contribute when two records carry the same label, and what does it contribute when the labels differ?
# 14. Write the Gaussian mean-shift step for a point $\mathbf{x}$ with bandwidth $h$, and use the gradient of the kernel density estimate to show why each step heads uphill. Where do the climbs end?
# 15. scikit-learn's `MeanShift` uses a flat kernel and picks its bandwidth with `estimate_bandwidth`. What does one flat-kernel step compute, how does `quantile` become a bandwidth, and why did no quantile recover the two moons?
# 16. Quantizing the china photograph to 16 colors, how many bytes do the indices and the palette take, and how much smaller is that than the original? Why did a fixed grid of 64 colors leave about ten times the error of the 64 colors that k-means chose?
# 17. How do k-means centroids become features for a linear model on the California housing data, and how do they become a labeling shortcut on the digits? What limits the accuracy of labels copied from cluster representatives?
#
# #### 8.2 Dimensionality reduction: linear methods
#
# Real data often has hundreds or thousands of features, far more than the few independent factors that actually drive it. Dimensionality reduction compresses many features into a few while keeping what matters, and the linear methods in this section do it with rotations and projections, which keeps them fast, interpretable, and grounded in the linear algebra of Chapter 2. The payoffs are everywhere: a two-dimensional view of high-dimensional data, removal of noise and redundancy, faster and better-conditioned downstream models, and relief from the curse of dimensionality of 6.1. The workhorse of the section is principal component analysis.
#
# ##### 8.2.1 Principal component analysis
#
# PCA finds a new set of axes for the data, ordered so that the first captures as much of the variance as possible, the second captures as much of the remaining variance as possible while being perpendicular to the first, and so on. These axes are the principal components, and because the later ones carry little variance, you can drop them and keep only the first few, compressing the data with minimal loss. The intuition is to rotate the coordinate system to line up with the directions the data actually spreads in, then ignore the directions where it barely moves.
#
# The derivation is a clean optimization. Center the data so each feature has mean zero, and form the covariance matrix $C = \frac{1}{n}\mathbf{X}_c^\top \mathbf{X}_c$. The variance of the data projected onto a unit direction $\mathbf{w}$ is $\mathbf{w}^\top C \mathbf{w}$, so the first principal component is the direction maximizing it subject to $\lVert\mathbf{w}\rVert = 1$. Using a Lagrange multiplier from 2.6,
#
# $$\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^\top C \mathbf{w} - \lambda(\mathbf{w}^\top \mathbf{w} - 1), \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2C\mathbf{w} - 2\lambda \mathbf{w} = 0 \;\Rightarrow\; C\mathbf{w} = \lambda \mathbf{w}.$$
#
# The maximizing direction is an eigenvector of the covariance matrix, and the variance along it is $\mathbf{w}^\top C\mathbf{w} = \lambda\,\mathbf{w}^\top\mathbf{w} = \lambda$, the corresponding eigenvalue, so the first component is the eigenvector with the largest eigenvalue; call it $\mathbf{w}_1$ and its eigenvalue $\lambda_1$. The second component maximizes the same variance under one more constraint, that it be perpendicular to the first, $\mathbf{w}^\top\mathbf{w}_1 = 0$, and that constraint gets its own multiplier $\mu$:
#
# $$\mathcal{L}_2(\mathbf{w}, \lambda, \mu) = \mathbf{w}^\top C \mathbf{w} - \lambda(\mathbf{w}^\top \mathbf{w} - 1) - \mu\,\mathbf{w}^\top\mathbf{w}_1, \qquad \frac{\partial \mathcal{L}_2}{\partial \mathbf{w}} = 2C\mathbf{w} - 2\lambda \mathbf{w} - \mu\,\mathbf{w}_1 = 0.$$
#
# Multiplying that condition on the left by $\mathbf{w}_1^\top$ makes two of its terms vanish. Because $C$ is symmetric, $\mathbf{w}_1^\top C\mathbf{w} = (C\mathbf{w}_1)^\top\mathbf{w} = \lambda_1\mathbf{w}_1^\top\mathbf{w} = 0$, and $\mathbf{w}_1^\top\mathbf{w} = 0$ by the constraint, which leaves $-\mu\,\mathbf{w}_1^\top\mathbf{w}_1 = -\mu = 0$. With $\mu = 0$ the condition is again $C\mathbf{w} = \lambda\mathbf{w}$, so the second component is also an eigenvector of $C$, and since the variance it captures is its eigenvalue, the best choice perpendicular to $\mathbf{w}_1$ is the eigenvector whose eigenvalue is second largest. Adding one such perpendicularity constraint for each earlier component repeats the argument for the third component and every later one. So the principal components are exactly the eigenvectors of $C$, ordered by eigenvalue, and each eigenvalue is the variance captured by its component. On a correlated two-dimensional cloud this is concrete.

# %%
import numpy as np
rng = np.random.default_rng(0)
X2 = rng.multivariate_normal([0, 0], [[3.0, 1.5], [1.5, 1.0]], 800)
Xc = X2 - X2.mean(axis=0)               # center the data
C = (Xc.T @ Xc) / len(Xc)               # covariance of centered data
evals, evecs = np.linalg.eigh(C)
order = evals.argsort()[::-1]           # order components by descending variance
evals, evecs = evals[order], evecs[:, order]
print(evals.round(3))                   # [3.717 0.191]   variance along each PC
print(evecs[:, 0].round(3))             # [-0.879 -0.477] first PC = the major axis
print((evals / evals.sum()).round(3))   # [0.951 0.049]   explained variance ratio

# %% [markdown]

# The first component points along the cloud's major axis and carries 95.1 percent of the variance, leaving only 4.9 percent for the perpendicular direction, as Figure 8-5 shows. That ratio, an eigenvalue divided by the sum of all eigenvalues, is the explained variance ratio, the fraction of total variance each component holds.
#
# There is a second route to the same result through the singular value decomposition of 2.1. Writing the centered data as $\mathbf{X}_c = U \Sigma V^\top$, the columns of $V$ are the principal components and the eigenvalues are $\lambda_i = \sigma_i^2 / n$ from the singular values $\sigma_i$. Computing that SVD avoids forming the covariance matrix. Forming the covariance squares the condition number of 2.1.10: the singular values of $\mathbf{X}_c^\top\mathbf{X}_c$ are the squares $\sigma_i^2$, so $\kappa(\mathbf{X}_c^\top\mathbf{X}_c) = \sigma_{\max}^2/\sigma_{\min}^2 = \kappa(\mathbf{X}_c)^2$. On the cloud just fitted, whose covariance eigenvalues printed as 3.717 and 0.191, the covariance has $\kappa = 3.717/0.191 \approx 19.5$ while the centered data has $\kappa = \sqrt{19.5} \approx 4.4$. Because relative errors can grow by up to a factor of $\kappa$, squaring $\kappa$ doubles $\log_{10}\kappa$, the number of decimal digits that rounding can cost, so the SVD is the more accurate route when there are many features. Speed runs the other way on tall data. Since scikit-learn 1.5, the default `svd_solver='auto'` uses `covariance_eigh`, which forms the covariance, when there are fewer than 1,000 features and more than ten times as many rows. That cutoff is stated on the scikit-learn 1.9.1 `PCA` page, which also warns that, compared with the full SVD, this solver "effectively doubles the condition number" (scikit-learn developers, 2026i): the same squaring, read on the logarithmic scale of lost digits. The release notes for version 1.5.0 record that `auto` began selecting the new solver in that release (scikit-learn developers, 2026d, release notes for 1.5.0). The 800 by 2 fit in the next block prints `covariance_eigh`, and so does the digits fit of Example 8-13. The three explained-variance ratios in the next block still agree at four decimals.

# %%
import numpy as np
from sklearn.decomposition import PCA
rng = np.random.default_rng(0)                       # rebuild the same correlated 2D cloud
X2 = rng.multivariate_normal([0, 0], [[3.0, 1.5], [1.5, 1.0]], 800)
Xc = X2 - X2.mean(axis=0)
evals = np.linalg.eigvalsh((Xc.T @ Xc) / len(Xc))[::-1]   # eigenvalue route, descending
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)   # SVD route, never forms the covariance
svd_evr = S**2 / (S**2).sum()
sk = PCA(n_components=2).fit(X2)
print((evals / evals.sum()).round(4))           # eigen:        [0.9511 0.0489]
print(svd_evr.round(4))                         # SVD:          [0.9511 0.0489]
print(sk.explained_variance_ratio_.round(4))    # scikit-learn: [0.9511 0.0489]
print(sk._fit_svd_solver)                       # covariance_eigh

# %% [markdown]

# The three prints agree at four decimals: 0.9511 and 0.0489 from the eigenvalues, from the singular values, and from scikit-learn, and the solver line is `covariance_eigh`. That name is read from `_fit_svd_solver`, a private attribute: the leading underscore marks it as internal to scikit-learn, it is not among the attributes the `PCA` page documents (scikit-learn developers, 2026i), and it can be renamed or removed without notice, so it serves here only to show which solver `auto` picked in scikit-learn 1.9.1 and has no place in production code. The real value of PCA appears in high dimensions. The handwritten digits are 8 by 8 images, 64 pixels, but the pixels are highly correlated, so far fewer directions hold most of the variance.
#
# Example 8-13. PCA on the 64-pixel digits: the solver that the default `svd_solver='auto'` picks, and the number of components that reach 80, 90, and 95 percent of the variance.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
Xd, _ = load_digits(return_X_y=True)            # 1797 images, 64 pixels each
pca_d = PCA().fit(Xd)
print(pca_d._fit_svd_solver)                    # covariance_eigh
cum = np.cumsum(pca_d.explained_variance_ratio_)
for thr in [0.80, 0.90, 0.95]:
    print(int(thr * 100), int(np.argmax(cum >= thr)) + 1, "of 64")
# 80 -> 13 of 64    90 -> 21 of 64    95 -> 29 of 64

# %% [markdown]

# The solver name prints `covariance_eigh`, and the three lines name 13, 21, and 29 components.
#
# Just 21 of the 64 dimensions capture 90 percent of the variance, a threefold compression, and a standard way to choose how many components to keep is exactly this: pick the number that reaches a target cumulative explained variance, often 90 or 95 percent, the rule Géron (2023, Chapter 8, "Choosing the Right Number of Dimensions") states with a 95 percent target. The compression is lossy, and projecting to $k$ components and back measures the loss directly.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
Xd, _ = load_digits(return_X_y=True)
full = PCA().fit(Xd)
cum = np.cumsum(full.explained_variance_ratio_)
ev = full.explained_variance_ * (len(Xd) - 1) / len(Xd)   # eigenvalues of the 1/n covariance
for k in [2, 8, 16, 32]:
    p = PCA(n_components=k).fit(Xd)
    Xrec = p.inverse_transform(p.transform(Xd))     # project down to k, then back up
    print(k, round(np.mean((Xd - Xrec) ** 2), 3), round(cum[k - 1], 3), f"{ev[k:].sum():.3f}")
#  k    reconstruction MSE   variance kept   dropped eigenvalues
#  2       13.421              0.285            858.945
#  8        6.122              0.674            391.795
# 16        2.827              0.849            180.940
# 32        0.632              0.966             40.425

# %% [markdown]

# With 2 components the reconstruction is crude, keeping 28.5 percent of the variance, but by 32 components the reconstruction error has fallen to 0.632 and 96.6 percent of the variance is retained, a near-perfect rebuild from half the dimensions. The printed reconstruction error is a per-pixel mean square, and the last column shows where it comes from. That column is the sum of the eigenvalues of the $1/n$ covariance for the components left out, and dividing it by the 64 pixels gives the error at every $k$: $858.945/64 = 13.421$ at 2 components and $40.425/64 = 0.632$ at 32. That identity measures the loss of PCA's own truncation. That no other rank-$k$ reconstruction of the centered data can do better is a separate result, the Eckart-Young theorem of 2.1.9 (Eckart and Young, 1936), which makes the truncated SVD the best rank-$k$ approximation in squared error; applied to the centered data, it is what makes PCA the optimal linear compressor in the squared-error sense. Two practical notes carry across every use. Always center the data, and usually standardize it too, because PCA chases variance and an unscaled large-range feature would dominate the components for the same reason it dominated kNN in 6.1. And remember PCA is unsupervised and linear: it knows nothing about labels, so its top components are not guaranteed to be the most useful for prediction, a gap the LDA of 8.2.3 and the kernel PCA of 8.2.2 each address.
#
# Two variants of PCA come up often enough to name here, and the first is whitening. Ordinary PCA rotates the data onto its principal axes but leaves the components with their original, widely different variances: on the digits the first ten component variances run 178.9, 163.6, 141.7, 101.0, 69.5, and on down to about 37. Some methods want features that are uncorrelated and on a common scale, including the ICA fit in 8.2.4, which leaves FastICA at its scikit-learn 1.9.1 default `whiten='unit-variance'`. Setting whiten=True divides each component by its standard deviation, so every retained component comes out with unit variance while staying uncorrelated.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
Xd, _ = load_digits(return_X_y=True)
Zw = PCA(n_components=10, whiten=True, random_state=0).fit_transform(Xd)   # whitened
Zp = PCA(n_components=10, random_state=0).fit_transform(Xd)                # plain PCA
print(Zp.var(axis=0).round(1))   # [178.9 163.6 141.7 101.   69.5  59.1  51.9  44.   40.3  37. ]
print(Zw.var(axis=0).round(3))   # [0.999 0.999 0.999 0.999 0.999 0.999 0.999 0.999 0.999 0.999]

# %% [markdown]

# Plain PCA's ten components carry sharply decreasing variances from 178.9 down to 37.0, while the whitened components all sit at 1.0 (printed as 0.999 only because scikit-learn scales by the unbiased variance using $n-1$ while np.var here divides by $n$). Whitening is what you reach for when the next step needs a common scale as well as decorrelation.
#
# The second variant to know is really a habit of reading, the scree plot, the standard picture for choosing how many components to keep. It plots each component's explained variance ratio against its index, usually with the cumulative curve drawn on top, and you read it two ways: look for the elbow where the per-component bars flatten out, and read across to where the cumulative curve crosses your variance target.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
Xd, _ = load_digits(return_X_y=True)
evr = PCA().fit(Xd).explained_variance_ratio_    # variance fraction per component
cum = np.cumsum(evr)                             # cumulative variance kept
print(evr[:5].round(3))                          # [0.149 0.136 0.118 0.084 0.058]
print(int(np.argmax(cum >= 0.90)) + 1)           # 21   components reach 90 percent

# %% [markdown]

# The first digit component holds 14.9 percent of the variance, the next 13.6, 11.8, 8.4, and 5.8, and the bars tail off from there, with the cumulative curve crossing 90 percent at the 21st component, as Figure 8-5 shows. That plot is how the 21-of-64 count of Example 8-13 is read off by eye rather than looked up. The figure's left panel shows the same rotation geometrically on a two-dimensional cloud: PC1 aligns with the cloud's major axis and carries 95 percent of the variance, leaving the perpendicular PC2 the remaining 5 percent.
#
# One more relative deserves a mention because a later section leans on it. Truncated SVD computes the top singular directions of the matrix you hand it. PCA's directions are the singular directions of the centered matrix, so skipping the centering step changes them. On the uncentered digits the first truncated-SVD direction lines up with the data mean, absolute cosine 1.000, and its absolute cosine with any of PCA's first four components is at most 0.063. The next three truncated-SVD directions match PCA's first three at absolute cosines 1.000, 0.999, and 0.999. Skipping the centering step is still what keeps a sparse term-count matrix sparse: subtracting the column means would replace every stored zero with a nonzero value. Because it never centers, truncated SVD runs directly on a sparse term-document matrix, which is exactly why the latent semantic analysis of 8.6 uses it to pull topics out of text.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, TruncatedSVD
X = load_digits().data
mean = X.mean(0)
mean = mean / np.linalg.norm(mean)
pca = PCA(n_components=4).fit(X)
svd = TruncatedSVD(n_components=4, random_state=0).fit(X)

def abs_cos(a, b):
    return abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))

print(f"{abs_cos(svd.components_[0], mean):.3f}")
print(" ".join(f"{max(abs_cos(svd.components_[i], pca.components_[j]) for j in range(4)):.3f}" for i in range(4)))
# 1.000
# 0.063 1.000 0.999 0.999

# %% [markdown]

# ![A correlated point cloud with two perpendicular principal-axis arrows on the left; bars of per-component explained variance with a rising cumulative curve crossing 90 percent at component 21 on the right.](figures/fig_08_05_pca-axes-scree.png)
#
# Figure 8-5. PCA principal axes on a correlated two-dimensional cloud (left) and the scree plot of the 64-pixel digits (right).
#
#
# A loading is the weight a component gives one original feature: one entry of that component's row of `components_`, computed after the features have been standardized. Reading the largest weights is how you say what a component means. The signed loading times the square root of that component's variance matches the correlation between the feature and the scores, up to the factor $\sqrt{n/(n-1)}$ that comes from the unbiased variance scikit-learn stores. On the wine data, $n = 178$, that correction removes the gap. PC1 puts its largest weights on flavanoids (+0.42), total phenols (+0.39), and the diluted-wine OD280/OD315 ratio (+0.38). PC2 puts them on color intensity (+0.53), alcohol (+0.48), and proline (+0.36). Flipping the sign of PC1 and of its scores together leaves the reconstruction unchanged, because a loading vector is identified only up to sign.

# %%
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
wine = load_wine()
X = StandardScaler().fit_transform(wine.data)
pca = PCA(2).fit(X)
scores = pca.transform(X)
factor = np.sqrt(len(X) / (len(X) - 1))
gap = []
for j in range(2):
    scaled = pca.components_[j] * np.sqrt(pca.explained_variance_[j]) / factor
    corr = np.array([np.corrcoef(X[:, f], scores[:, j])[0, 1] for f in range(X.shape[1])])
    gap.append(np.max(np.abs(scaled - corr)))
print(" ".join(f"{v:.2f}" for v in pca.explained_variance_ratio_))
print(f"{gap[0]:.3f}", f"{gap[1]:.3f}")
for j in range(2):                                   # three largest weights, with their signs
    top = np.argsort(np.abs(pca.components_[j]))[::-1][:3]
    print(f"PC{j + 1}", " ".join(f"{wine.feature_names[i]} {pca.components_[j, i]:+.2f}" for i in top))
# 0.36 0.19
# 0.000 0.000
# PC1 flavanoids +0.42 total_phenols +0.39 od280/od315_of_diluted_wines +0.38
# PC2 color_intensity +0.53 alcohol +0.48 proline +0.36

# %% [markdown]

# The block prints the two explained variance ratios, 0.36 and 0.19, then the largest gap between the scaled loadings and the correlations for each component, 0.000 and 0.000, and last the three largest weights of each component with their signs: flavanoids +0.42, total phenols +0.39, and the OD280/OD315 ratio +0.38 on PC1, and color intensity +0.53, alcohol +0.48, and proline +0.36 on PC2.
#
# On the digits, a 10-component exact PCA and a randomized PCA agree on the explained-variance sum, both 0.7382. This matrix is too small for the randomized solver's speed advantage to be the point. The agreement is the check.

# %%
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
Xd = load_digits().data
a = PCA(n_components=10, svd_solver="full").fit(Xd)
b = PCA(n_components=10, svd_solver="randomized", random_state=0).fit(Xd)
print(round(float(a.explained_variance_ratio_.sum()), 4), round(float(b.explained_variance_ratio_.sum()), 4))
# 0.7382 0.7382

# %% [markdown]

# Both sums print 0.7382.
#
# Every PCA so far decomposed a matrix that fits easily in memory. Two variants exist for the cases where that stops being true. Randomized PCA finds the top $k$ components approximately: it first squeezes the data through a few random directions and then decomposes only that small sketch, which saves time when $k$ is small next to the number of features $p$. Incremental PCA reads the rows in batches and updates its components after each one, so the full matrix never has to sit in memory at once, the out-of-core idea of 3.9.1 applied to PCA. The test case is MNIST, 70,000 handwritten digits of $28 \times 28 = 784$ pixels each, created by Yann LeCun, Corinna Cortes, and Christopher Burges, which `fetch_openml("mnist_784")` downloads from OpenML on first use and caches (OpenML, 2014c). Held as 70,000 by 784 floating-point numbers of 8 bytes each it takes $70{,}000 \times 784 \times 8 = 439{,}040{,}000$ bytes, about 439 MB.
#
# The randomized idea rests on one observation. Multiplying the data matrix by a vector $\boldsymbol\omega$ forms $\mathbf{X}\boldsymbol\omega$, a weighted sum of the columns of $\mathbf{X}$, so the product always lies in the space those columns span. Take the rank-one matrix with rows $(1, 2)$, $(2, 4)$, and $(2, 4)$, whose columns are both multiples of $\mathbf{u} = (1, 2, 2)^\top$. For the arbitrary $\boldsymbol\omega = (0.5, -1)^\top$ the product is $(0.5 - 2,\ 1 - 4,\ 1 - 4)^\top = (-1.5, -3, -3)^\top = -1.5\,\mathbf{u}$: one random vector recovered the matrix's only direction. Any $\boldsymbol\omega$ not perpendicular to $(1, 2)$ would do, and a vector with continuous random entries is perpendicular to it with probability zero. When the data has $k$ strong directions and many weaker ones, a few more than $k$ random vectors catch the strong directions with high probability.
#
# Halko, Martinsson, and Tropp (2011) turn this into a two-stage method. Stage A finds an orthonormal basis for the range of the centered $n \times p$ matrix $\mathbf{X}_c$ (their Algorithm 4.1): draw a $p \times \ell$ matrix $\boldsymbol\Omega$ of independent standard normal entries, with $\ell = k + s$ for a small oversampling $s$ (Halko and coauthors call it $p$, a letter this section keeps for the number of features), form the sketch $\mathbf{Y} = \mathbf{X}_c\boldsymbol\Omega$, and orthonormalize its columns with the QR factorization of 2.1.12, $\mathbf{Y} = \mathbf{Q}\mathbf{R}$, so that $\mathbf{X}_c \approx \mathbf{Q}\mathbf{Q}^\top\mathbf{X}_c$. Stage B (their Algorithm 5.1) forms the small $\ell \times p$ matrix $\mathbf{B} = \mathbf{Q}^\top\mathbf{X}_c$, takes its exact SVD $\mathbf{B} = \tilde{\mathbf{U}}\boldsymbol\Sigma\mathbf{V}^\top$, and returns $\mathbf{X}_c \approx (\mathbf{Q}\tilde{\mathbf{U}})\boldsymbol\Sigma\mathbf{V}^\top$, whose first $k$ rows of $\mathbf{V}^\top$ are the approximate principal components. They found an oversampling of $s = 5$ or $s = 10$ often enough in practice (Section 1.3.2), and scikit-learn's default is $s = 10$ (`n_oversamples=10`; scikit-learn developers, 2026i). When the singular values fall off slowly, the weak directions leak into the sketch, and the remedy for that case (their Section 4.5) is $q$ steps of power iteration (their Algorithm 4.3): sketch $(\mathbf{X}_c\mathbf{X}_c^\top)^q\mathbf{X}_c\boldsymbol\Omega$ instead, a matrix with the same singular vectors whose singular values are $\sigma_j^{2q+1}$ (their equation 4.5). Their Algorithm 4.4, which they recommend and which Example 8-14 follows, orthonormalizes with a QR factorization after every multiplication by $\mathbf{X}_c$ or $\mathbf{X}_c^\top$; it equals Algorithm 4.3 in exact arithmetic at about the same cost and is far more accurate in floating-point arithmetic. With $q = 2$, a direction half as strong as another counts only $2^{-5} = 1/32$ as much in the sketch. scikit-learn's randomized solver uses $q = 4$, or $q = 7$ when $k$ is below a tenth of the smaller dimension of the data, and normalizes between the steps with an LU factorization when $q > 2$ (scikit-learn developers, 2026d, `randomized_svd` API reference). The arithmetic is cheap when $\ell$ is small. Each product of the $70{,}000 \times 784$ matrix with a $784 \times \ell$ matrix costs $n p \ell$ multiplications, and with $q$ power iterations the method makes $2q + 2$ such passes over the data: one for $\mathbf{Y}$, two per iteration, and one for $\mathbf{B}$. At $k = 154$, $\ell = 164$, and $q = 4$ that is $10 \times 70{,}000 \times 784 \times 164 \approx 9.0 \times 10^{10}$, about twice $n p^2 \approx 4.3 \times 10^{10}$, the order of the cost of one exact SVD (scikit-learn developers, 2026d, IncrementalPCA API reference); at $k = 10$, $\ell = 20$, and $q = 7$ it is $16 \times 70{,}000 \times 784 \times 20 \approx 1.8 \times 10^{10}$, about 0.4 of it. Example 8-14 writes both stages and the power iterations in NumPy and measures how well the sketch captures the variance, $\lVert\mathbf{X}_c\mathbf{V}_k\rVert_F^2 / \lVert\mathbf{X}_c\rVert_F^2$, where $\mathbf{V}_k$ holds the $k$ returned directions as columns; no $k$ directions can capture more than the exact top $k$.
#
# Example 8-14. Randomized SVD from scratch: a Gaussian sketch, QR, power iterations, and a small exact SVD.

# %%
import numpy as np
from sklearn.datasets import fetch_openml
X = fetch_openml("mnist_784", as_frame=False, parser="auto").data / 255.0   # 70,000 images, 784 pixels
Xc = X - X.mean(axis=0)                                  # center, as PCA does
s_exact = np.linalg.svd(Xc, compute_uv=False)            # all 784 exact singular values
total = float((s_exact ** 2).sum())                      # sum of squares of the centered data
cum = np.cumsum(s_exact ** 2) / total
k = int(np.argmax(cum >= 0.95)) + 1                      # components for the 95 percent cut
print(X.shape, k, round(float(cum[k - 1]), 4))

def randomized_svd_scratch(A, k, s=10, q=2, seed=0):  # s = oversampling
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((A.shape[1], k + s))     # p x (k + s) Gaussian test matrix
    Q, _ = np.linalg.qr(A @ omega)                       # orthonormal basis for the sketch Y = A omega
    for _ in range(q):                                   # power iterations, orthonormalized again each time
        Q, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Q)
    B = Q.T @ A                                          # the small (k + s) x p matrix
    _, sigma, Vt = np.linalg.svd(B, full_matrices=False) # exact SVD of the small matrix
    return sigma[:k], Vt[:k]

for q in (0, 1, 2, 4):
    sigma, Vt = randomized_svd_scratch(Xc, k, q=q)
    captured = float(((Xc @ Vt.T) ** 2).sum() / total)  # variance the k directions really capture
    worst = float(np.max(np.abs(sigma - s_exact[:k]) / s_exact[:k]))   # worst relative error, top k values
    print(q, round(captured, 4), round(worst, 3))
# (70000, 784) 154 0.9503
# 0 0.936 0.31
# 1 0.9475 0.136
# 2 0.9493 0.075
# 4 0.9501 0.033

# %% [markdown]

# MNIST needs 154 of its 784 components to reach 95 percent of the variance, and those exact components capture 0.9503. The sketch with no power iteration captures 0.9360, and its worst singular value among the top 154 is off by 31 percent of its true size (0.31). One power iteration lifts the captured share to 0.9475, two to 0.9493, and four to 0.9501, within 0.0002 of the exact 0.9503, while the worst relative error falls to 0.136, 0.075, and 0.033. MNIST is the slowly decaying case: `s_exact[153]` and `s_exact[154]`, the 154th and 155th singular values, are 40.740 and 40.634, so the 155th is 0.9974 of the 154th and nothing in the spectrum marks where the top 154 end; the power steps are what pull the sketch onto them. scikit-learn's documentation advises raising the oversampling before adding power iterations, since each iteration costs two more passes over the data (scikit-learn developers, 2026d, `randomized_svd` API reference), but on this spectrum oversampling helps far less. With no power iteration, raising $s$ from 10 to 50 moves the captured share only from 0.9360 to 0.9427, and with $q = 2$, dropping $s$ to 0 costs 0.0008 (0.9485 against 0.9493).
#
# Incremental PCA rests on an identity about scatter matrices, the centered sums $\mathbf{S} = \sum_i (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\top$, which are $n$ times the covariance of 8.2.1. If a block $A$ of $n_A$ rows with mean $\bar{\mathbf{x}}_A$ is joined by a block $B$ of $n_B$ rows with mean $\bar{\mathbf{x}}_B$, the scatter of the union is
#
# $$\mathbf{S}_{A \cup B} = \mathbf{S}_A + \mathbf{S}_B + \frac{n_A n_B}{n_A + n_B}(\bar{\mathbf{x}}_B - \bar{\mathbf{x}}_A)(\bar{\mathbf{x}}_B - \bar{\mathbf{x}}_A)^\top$$
#
# (Ross et al., 2008, Lemma 1). In one dimension, with $A = \{0, 2\}$ and $B = \{4\}$, the scatters are 2 and 0, the means 1 and 4, and the correction $\tfrac{2 \times 1}{3}(4 - 1)^2 = 6$, so the formula gives $2 + 0 + 6 = 8$, the scatter of $\{0, 2, 4\}$ around its mean 2, which is $4 + 0 + 4 = 8$. scikit-learn's `IncrementalPCA`, whose documentation cites Ross et al. (2008) for its method, decomposes the first batch on its own and then updates after each later batch by stacking three things: the old components scaled by their singular values, $\boldsymbol\Sigma_k\mathbf{V}_k^\top$; the new batch centered on its own mean; and one extra row, $\sqrt{n_A n_B/(n_A + n_B)}\,(\bar{\mathbf{x}}_A - \bar{\mathbf{x}}_B)^\top$. It takes the exact SVD of that $(k + n_B + 1) \times p$ matrix and keeps the top $k$ rows of $\mathbf{V}^\top$ (scikit-learn developers, 2026d, IncrementalPCA API reference and its linked source). For the stacked matrix $\mathbf{M}$, the product $\mathbf{M}^\top\mathbf{M}$ equals the right side of the identity with $\mathbf{S}_A$ replaced by its best rank-$k$ part $\mathbf{V}_k\boldsymbol\Sigma_k^2\mathbf{V}_k^\top$, so the only loss is the variance of earlier batches outside the $k$ kept directions. Memory per step is on the order of the batch size times $p$, which is why the documentation suggests feeding it from a NumPy memory map, an array whose rows stay in a file on disk until they are sliced. Example 8-15 compares the three solvers of `PCA` at 10 and 154 components on time and on the peak memory each fit allocates, measured with the standard library's `tracemalloc` met in Chapter 1, and then fits `IncrementalPCA` in 14 batches of 5,000 rows read from a file of raw bytes.
#
# Example 8-15. The three `PCA` solvers on MNIST at 10 and 154 components, with their captured variance, time, and peak allocated memory, then `IncrementalPCA` fitted in 14 batches read from a memory-mapped file.

# %%
import os, tempfile, time, tracemalloc
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, IncrementalPCA
X8 = fetch_openml("mnist_784", as_frame=False, parser="auto").data.astype(np.uint8)   # pixels 0 to 255
X = X8 / 255.0                                           # 70,000 x 784 float64 values
Xc = X - X.mean(axis=0)
total = float((Xc ** 2).sum())

def captured(components):                                # share of the variance the directions capture
    return float(((Xc @ components.T) ** 2).sum() / total)

for k in (10, 154):
    for solver in ("full", "covariance_eigh", "randomized"):
        tracemalloc.start()                              # count only what the fit itself allocates
        t0 = time.perf_counter()
        pca = PCA(n_components=k, svd_solver=solver, random_state=0).fit(X)
        seconds = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        print(k, solver, round(captured(pca.components_), 4), f"{seconds:.1f} s", f"{peak:.0f} MB")

with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "mnist_uint8.npy")
    np.save(path, X8)                                    # a file on disk stands in for data too big for memory
    disk = np.load(path, mmap_mode="r")                  # a memory map of the file: rows are read only when sliced
    ipca = IncrementalPCA(n_components=154)
    tracemalloc.start()
    t0 = time.perf_counter()
    for start in range(0, len(disk), 5000):              # 14 batches of 5,000 rows
        ipca.partial_fit(disk[start:start + 5000] / 255.0)
    seconds = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    size = os.path.getsize(path) / 1e6
    del disk                                             # release the file before the folder is removed
print("incremental", round(captured(ipca.components_), 4), f"{seconds:.1f} s", f"{peak:.0f} MB",
      f"file {size:.1f} MB", int(ipca.n_samples_seen_))
# 10 full 0.4892 2.8 s 888 MB
# 10 covariance_eigh 0.4892 0.2 s 25 MB
# 10 randomized 0.4892 0.4 s 473 MB
# 154 full 0.9503 2.8 s 888 MB
# 154 covariance_eigh 0.9503 0.2 s 25 MB
# 154 randomized 0.9501 1.3 s 716 MB
# incremental 0.9502 5.3 s 111 MB file 54.9 MB 70000
# (seconds and megabytes: one run on the reference machine)

# %% [markdown]

# At 10 components all three solvers capture 0.4892. The exact SVD takes 2.8 s and allocates 888 MB, about twice the 439 MB of the data itself; the randomized solver takes 0.4 s and allocates 473 MB; the covariance route takes 0.2 s and 25 MB. At 154 components the exact routes capture 0.9503 and the randomized one 0.9501, in 2.8 s, 0.2 s, and 1.3 s, with the randomized solver now allocating 716 MB. These are one run on the reference machine, and two lessons follow. First, randomized PCA beat the full SVD about seven times over at 10 components ($2.8/0.4 = 7.0$) but only about twice at 154 ($2.8/1.3 \approx 2.2$). The count of $2q + 2$ passes predicted that direction, though not the sizes: at 154 components the sketch does about twice the multiplications of $n p^2$ and still finishes first, so the count sets the trend rather than the exact times. Either way, the home of randomized PCA is a few components out of many features. Second, on MNIST the covariance route of 8.2.1 beat both, because 784 features make a $784 \times 784$ covariance cheap to form and to decompose; this is the shape, fewer than 1,000 features and more than ten times as many rows, for which scikit-learn's default picks it, and its documentation warns that the route becomes intractable when the number of features is large, since the covariance matrix itself must be built (scikit-learn developers, 2026i). Incremental PCA, reading the 54.9 MB file of raw bytes in 14 batches, captures 0.9502, within 0.0001 of exact, and its fit allocated at most 111 MB while never holding the 439 MB matrix; it took 5.3 s, the slowest of the four, because it runs 14 SVDs in place of one, the first of the $5{,}000 \times 784$ batch alone and the other 13 of $(154 + 5{,}000 + 1) \times 784$ stacked matrices. For text, `TruncatedSVD`, the LSA tool of 8.6, runs the same randomized algorithm by default (`algorithm="randomized"` with `n_iter=5`; scikit-learn developers, 2026d, TruncatedSVD API reference). To experiment, change the batch size to 1,000 and to 20,000 and watch peak memory trade against time, or pass `iterated_power=0` to the randomized `PCA` and compare its captured share with the $q = 0$ line of Example 8-14.
#
# ##### 8.2.2 Kernel PCA
#
# Linear PCA can only rotate the axes; it cannot bend them, so it fails on data whose structure is curved. Kernel PCA (Schölkopf, Smola, and Müller, 1998) fixes this with the same kernel trick that powered support vector machines in 6.5: it performs PCA in a high-dimensional feature space without ever visiting that space, by working through a kernel function. The result is nonlinear principal components that can unfold curved structure into something a straight line can handle. Where linear PCA eigendecomposes the $p \times p$ covariance matrix, with $p$ the number of features, kernel PCA eigendecomposes the $n \times n$ kernel matrix $K$ with entries $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$, where $k$ is a kernel such as the RBF $k(\mathbf{x},\mathbf{z}) = \exp(-\gamma\lVert\mathbf{x}-\mathbf{z}\rVert^2)$. The matrix that is diagonalized is the centered kernel $\tilde K = HKH$, with $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$ (Murphy, 2022, Section 20.4.6). On five generated points the raw RBF kernel sums to 12.700 and the centered kernel sums to 0. The effect is sharpest on concentric circles, which no rotation can separate.

# %%
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
rng = np.random.default_rng(0)
X = rng.normal(size=(5, 2))
K = rbf_kernel(X, gamma=0.5)
H = np.eye(5) - np.ones((5, 5)) / 5
Kc = H @ K @ H
print(f"{float(Kc.sum()):.6f}", f"{float(K.sum()):.3f}")
# -0.000000 12.700

# %% [markdown]

# The centered kernel prints -0.000000 and the raw kernel prints 12.700.

# %%
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
X, y = make_circles(n_samples=500, factor=0.3, noise=0.05, random_state=0)
raw = cross_val_score(LogisticRegression(), X, y, cv=5).mean()
lin = cross_val_score(make_pipeline(PCA(2), LogisticRegression()), X, y, cv=5).mean()
ker = cross_val_score(
    make_pipeline(KernelPCA(n_components=2, kernel="rbf", gamma=3), LogisticRegression()),
    X, y, cv=5).mean()
print(f"{raw:.3f}", f"{lin:.3f}", f"{ker:.3f}")
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=3).fit_transform(X)   # full-data fit, for the means below
# 0.446 (chance)   0.448 (linear PCA no help)   1.000 (RBF kernel PCA unfolds the rings)
print(round(float(kpca[y == 1, 0].mean()), 3), round(float(kpca[y == 0, 0].mean()), 3))
# 0.393 -0.393   the first kernel component alone separates inner ring from outer

# %% [markdown]

# Cross-validated inside pipelines, so each fold fits its own PCA or kernel PCA on that fold's training rows only, a linear classifier scores 0.446 on the raw circles, 0.448 after linear PCA, and 1.000 after RBF kernel PCA. The separate full-data kernel PCA, used only to look at the embedding, still separates the rings on its first component: the inner ring's points average $+0.393$ and the outer ring's $-0.393$. As with SVMs, the kernel width $\gamma$ must be tuned, since too small or too large a value fails to unfold the structure. The cost mirrors spectral clustering, to which kernel PCA is closely related: building and decomposing an $n \times n$ kernel matrix is impractical for very large datasets. Kernel PCA is the tool when you need PCA's compression but the data lies on a curved manifold that linear PCA cannot flatten.
#
# ##### 8.2.3 PCA versus LDA for dimensionality reduction
#
# Section 6.3 already derived LDA as a classifier, including the limit of $C-1$ components. PCA and LDA both project data onto fewer dimensions, and a favorite interview question asks how they differ. The answer is supervision. PCA is unsupervised: it finds the directions of maximum variance, ignoring any labels, so its top components capture how the data spreads but not necessarily how the classes differ. LDA, met as a classifier in 6.3, doubles as a supervised dimensionality reducer: it finds the directions that maximize the separation between class means relative to the spread within each class, using the labels directly. When the goal is to separate known classes, LDA's components are far more efficient, as projecting the ten-class digits down to a few dimensions shows.

# %%
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
X, y = load_digits(return_X_y=True)
for k in [2, 5, 9]:
    pca_s = cross_val_score(
        make_pipeline(StandardScaler(), PCA(n_components=k), LogisticRegression(max_iter=2000)),
        X, y, cv=5).mean()
    lda_s = cross_val_score(
        make_pipeline(StandardScaler(), LDA(n_components=k), LogisticRegression(max_iter=2000)),
        X, y, cv=5).mean()
    print(k, f"{pca_s:.3f}", f"{lda_s:.3f}")
# 2 0.535 0.650
# 5 0.771 0.861
# 9 0.829 0.906

# %% [markdown]

# The scaler, the projection, and the classifier all sit inside the cross-validation, so a test fold's labels never shape the LDA axes. At every dimension LDA still wins. At $k=9$ it reaches 0.906 against PCA's 0.829, because LDA packs the class-distinguishing information into its few axes while PCA's high-variance directions need not line up with the class boundaries at all. There is a hard limit worth knowing: LDA can produce at most $C-1$ components for $C$ classes, since the class means span only that many dimensions, which is why the digits' maximum is nine. The tradeoff is the usual one. Use LDA when you have labels and want the projection that best separates them, often as a preprocessing step before a classifier; use PCA when you have no labels, when you want to preserve the data's overall structure rather than class boundaries, or when LDA's assumption of Gaussian, equal-covariance classes does not hold. PCA is the more general tool; LDA is the sharper one when labels exist.
#
# ##### 8.2.4 ICA, factor analysis, and NMF
#
# PCA is not the only linear factorization. Three others answer different questions, and each is worth knowing. Independent component analysis, ICA, solves the cocktail-party problem: given several microphones each recording a mixture of several speakers, recover the individual voices (Hyvärinen and Oja, 2000). The mixture being undone is $\mathbf{x} = A\mathbf{s}$, with $\mathbf{s}$ the independent sources and $A$ an unknown mixing matrix. ICA looks for $W$ so that the coordinates of $W\mathbf{x}$ are independent. Where PCA finds uncorrelated components ordered by variance, ICA finds components that are statistically independent and maximally non-Gaussian, which is what lets it unmix signals that PCA cannot. Non-Gaussianity points at the sources because of the central limit theorem of 2.4.2: a sum of two independent random variables usually has a distribution closer to Gaussian than either of them, so a mixture looks more Gaussian than the sources inside it (Hyvärinen and Oja, 2000, Section 4.1). FastICA, the algorithm behind scikit-learn's `FastICA`, measures non-Gaussianity by negentropy. After whitening the data to $\mathbf{z}$, the whitening of 8.2.1, it looks for unit vectors $\mathbf{w}$ that maximize, for $y = \mathbf{w}^\top\mathbf{z}$,
#
# $$J(y) = H(y_{\text{gauss}}) - H(y) \;\approx\; k\,\bigl[\,\mathbb{E}\{G(y)\} - \mathbb{E}\{G(\nu)\}\,\bigr]^2, \qquad G(u) = \log\cosh u,$$
#
# where $H$ is differential entropy, $y_{\text{gauss}}$ is a Gaussian variable with the same variance as $y$, $\nu$ is a standard normal variable, and $k$ is a positive constant. Negentropy is zero for a Gaussian and positive for anything else. FastICA maximizes the squared bracket on the right, since the constant $k$ does not move the maximum, using the `logcosh` choice of $G$ that is scikit-learn's default (Hyvärinen and Oja, 2000, equations 22, 24, 25, and 26). Kurtosis, the older measure of the same thing, makes the idea concrete: for a unit-variance $y$ it is $\operatorname{kurt}(y) = \mathbb{E}\{y^4\} - 3$, which is 0 for a Gaussian. A square wave that sits at $+1$ half the time and at $-1$ the other half has $\mathbb{E}\{y^4\} = 1$ and kurtosis $-2$, and a unit-variance sine wave $\sqrt{2}\sin t$ has $\mathbb{E}\{y^4\} = 4 \cdot \tfrac{3}{8} = 1.5$ and kurtosis $-1.5$, both far from a Gaussian's 0. The sources in Example 8-16 are a tone and a square wave of this kind, plus a little noise, generated rather than recorded because scoring the unmixing requires knowing the true signals.
#
# Factor analysis writes $\mathbf{x} = \Lambda \mathbf{f} + \mathbf{e}$, with $\mathbf{f}$ a few shared factors and $\mathbf{e}$ private noise whose variance may differ by feature. It is the statistician's cousin of PCA: it models the observed features as a few hidden factors plus per-feature noise, separating shared structure from individual noise. With factors of zero mean and identity covariance, and noise independent of them, the covariance the model fits is
#
# $$\Sigma = \Lambda\Lambda^\top + \Psi,$$
#
# where $\Psi$ is the diagonal matrix of the private noise variances. Bishop (2006, Section 12.2.4) calls the columns of $\Lambda$ the factor loadings and the diagonal entries of $\Psi$ the uniquenesses, and scikit-learn's `FactorAnalysis` reports that diagonal as `noise_variance_`. In Example 8-17, the first feature listens to the two factors with weights $(0.9, 0)$ and carries noise with standard deviation 0.1, so its variance splits into $0.9^2 + 0^2 = 0.81$ shared and $0.1^2 = 0.01$ private, 0.82 in all. The fifth has weights $(0.5, 0.5)$ and noise standard deviation 2.0, so $0.5^2 + 0.5^2 = 0.5$ shared against $2.0^2 = 4.0$ private, 4.5 in all.
#
# Nonnegative matrix factorization, NMF, adds one constraint that changes everything: it factors the data into two matrices whose entries are all nonnegative. With $X$ the $n \times p$ data, $W$ an $n \times k$ matrix, and $H$ a $k \times p$ matrix, scikit-learn's `NMF` by default solves
#
# $$\min_{W \ge 0,\; H \ge 0}\; \tfrac{1}{2}\,\lVert X - WH \rVert_F^2,$$
#
# where $\lVert \cdot \rVert_F$ is the Frobenius norm, the square root of the sum of squared entries, and the inequalities hold entry by entry; its default coordinate descent solver updates $W$ and $H$ in turn (scikit-learn developers, 2026d, NMF API reference). For a small case, $X = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$ is matched exactly by $W = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ and $H = \begin{bmatrix} 1 & 2 \end{bmatrix}$, a loss of 0, while $W = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ with the same $H$ gives $WH = \begin{bmatrix} 1 & 2 \\ 1 & 2 \end{bmatrix}$, a residual of $\begin{bmatrix} 0 & 0 \\ 1 & 2 \end{bmatrix}$, and a loss of $\tfrac{1}{2}(1 + 4) = 2.5$. Because nothing can cancel, the parts can only add up, and NMF tends to discover parts-based components: Lee and Seung (1999) showed it learning parts of faces, and groups of related words from encyclopedia articles, where PCA learned holistic components. That word-group behavior is why 8.6 uses it for topic modeling. All three deserve a demonstration: first ICA and NMF together, then factor analysis against PCA.
#
# Example 8-16. FastICA against PCA on a tone and a square wave mixed into two recordings, then the signs of NMF and PCA components on the digits.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA, NMF, PCA
Xd, _ = load_digits(return_X_y=True)
rng = np.random.default_rng(0)
t = np.linspace(0, 8, 2000)
S = np.c_[np.sin(2 * t), np.sign(np.sin(3 * t))] + 0.05 * rng.normal(size=(2000, 2))  # tone + square wave
Xmix = S @ np.array([[1.0, 1.0], [0.5, 2.0]]).T            # what two microphones record
def mcorr(R):                                              # best |corr| of recovered with true sources
    return np.mean([max(abs(np.corrcoef(R[:, i], S[:, j])[0, 1]) for i in range(2)) for j in range(2)])
print(round(mcorr(FastICA(n_components=2, random_state=0).fit_transform(Xmix)), 3))   # 0.999 unmixed
print(round(mcorr(PCA(n_components=2).fit_transform(Xmix)), 3))                        # 0.949 still blended
nmf = NMF(n_components=16, init="nndsvda", max_iter=1000, random_state=0).fit(Xd)
print((nmf.components_ >= 0).all(), (PCA(16).fit(Xd).components_ < 0).any())           # True True

# %% [markdown]

# ICA recovers the original signals almost perfectly at a correlation of 0.999, where PCA, which only removes correlation, leaves them blended at 0.949, and NMF's components stay nonnegative while PCA's contain negatives. The nonnegativity holds by construction of the factorization. It does not, by itself, prove that the parts are interpretable. What the signs show is that NMF cannot cancel, while PCA can.
#
# Factor analysis needs its own check, and the check must be synthetic by necessity: proving that the model separates shared signal from private noise requires knowing the true noise level of every feature, which only manufactured data provides. We build five features from two hidden factors, give four of them small private noise with standard deviation 0.1, and drown the fifth in noise with standard deviation 2.0, so its true noise variance is $2.0^2 = 4.0$ against the others' $0.01$.
#
# Example 8-17. Factor analysis against PCA on five generated features driven by two hidden factors, one feature drowned in private noise.

# %%
import numpy as np
from sklearn.decomposition import FactorAnalysis, PCA
rng = np.random.default_rng(0)
F = rng.normal(size=(1000, 2))                          # two hidden factors drive everything
load = np.array([[0.9, 0.0], [0.8, 0.3], [0.1, 0.9],   # how strongly each feature
                 [0.0, 0.8], [0.5, 0.5]])               # listens to each factor
noise_sd = np.array([0.1, 0.1, 0.1, 0.1, 2.0])          # feature 5 is almost all private noise
X = F @ load.T + rng.normal(size=(1000, 5)) * noise_sd
fa = FactorAnalysis(n_components=2, random_state=0).fit(X)
rounded = fa.noise_variance_.round(2)
print("[" + " ".join(f"{v:.2f}" if v < 1 else f"{v:.1f}" for v in rounded) + "]")
# [0.01 0.01 0.01 0.01 4.0]
p = PCA(n_components=2).fit(X)
print(p.components_[0].round(2), p.explained_variance_ratio_.round(2))
# [0.15 0.18 0.15 0.12 0.95] [0.64 0.2]   PC1 points at the noisy feature

# %% [markdown]

# Factor analysis reads the structure back almost exactly: its `noise_variance_` reports 0.01 for each of the four clean features and 4.0 for the noisy one, the very values the data was built with, so its two factors are left carrying the shared signal. PCA has no concept of per-feature noise; chasing raw variance, its first component loads 0.95 on the noisy fifth feature and only 0.12 to 0.18 on the informative ones, mistaking the loudest column for the most important, the same failure mode that sinks principal components regression in 8.2.5. The contrast is the lesson: PCA decomposes by variance, ICA by independence, factor analysis by shared-versus-noise, and NMF by additive parts, and the right one depends on what structure you are trying to recover.
#
# ##### 8.2.5 Dimensionality reduction for prediction: PCR and PLS
#
# When dimensionality reduction is a step toward prediction rather than an end in itself, a subtle trap appears. Principal components regression, PCR, is the obvious approach: run PCA, then regress the target on the top components. But PCA chooses its components by variance alone, ignoring the target, so it can throw away a low-variance direction that happens to be exactly what predicts $y$. Partial least squares, PLS, fixes this by finding a unit vector $w$ that maximizes the covariance of $Xw$ with $y$, so its directions are predictive by construction. For centered columns that covariance is $w^\top X^\top y / n$, which a unit vector maximizes by pointing along $X^\top y$; with standardized columns each weight is therefore proportional to that feature's correlation with the target, which is how James et al. (2013, Section 6.3.2) describe the first PLS direction. The difference is stark on generated columns: one low-variance feature that really drives $y$, one high-variance feature that is pure noise, and eight more noise columns. The true pattern has to be known to see which direction each method keeps.

# %%
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
rng = np.random.default_rng(0)
x_big = rng.normal(0, 10, 400)            # high variance, but irrelevant to y
x_small = rng.normal(0, 1, 400)           # low variance, the real signal
X = np.column_stack([x_big, x_small, rng.normal(0, 5, (400, 8))])
y = 3.0 * x_small + rng.normal(0, 0.5, 400)
pcr = make_pipeline(PCA(n_components=1), LinearRegression())     # PCA then regress
print(round(cross_val_score(pcr, X, y, cv=5, scoring="r2").mean(), 3))                     # -0.008 (chases noise)
print(round(cross_val_score(PLSRegression(n_components=1), X, y, cv=5, scoring="r2").mean(), 3))  # 0.929 (finds signal)
print(f"{cross_val_score(make_pipeline(PCA(10), LinearRegression()), X, y, cv=5, scoring='r2').mean():.3f}")  # 0.970
pc1 = PCA(n_components=1).fit(X).components_[0]                  # where each method points
w1 = PLSRegression(n_components=1).fit(X, y).x_weights_[:, 0]
print(round(cross_val_score(PLSRegression(n_components=1, scale=False), X, y, cv=5, scoring="r2").mean(), 3))
# 0.044
print(np.abs(pc1[:2]).round(2).tolist(), np.abs(w1[:2]).round(2).tolist())
# [0.99, 0.01] [0.08, 0.98]   PCA's component points at the noise, PLS's at the signal

# %% [markdown]

# With a single component PCR scores $-0.008$, worse than guessing the mean, because its first component points almost entirely at the high-variance noise feature: the printed weights show PCA loading the noise feature at 0.99 against the signal's 0.01. PLS with one component scores 0.929, because it weights the signal feature 0.98 and the noise only 0.08, having used the target to find the right direction. `PLSRegression` scales the columns by default. The same fit with `scale=False` scores 0.044, so part of the gap versus PCR, whose pipeline does not rescale, is that default scaling. PCR does eventually recover, reaching 0.970 once it is allowed all ten components and the regression can pick out the useful one, but that defeats the purpose of reducing dimensions. The lesson is that when the reduction serves prediction, use a supervised method like PLS; unsupervised PCA components are not guaranteed to be the predictive ones, the same warning that distinguished PCA from LDA in 8.2.3.
#
# ##### 8.2.6 The high-dimensional regime and random projections
#
# A recurring theme deserves its own statement: what happens when the number of features $p$ exceeds the number of samples $n$, the regime of genomics, text, and many modern problems. Section 5.3.3 already ran this regime. Here ordinary least squares breaks completely, because with more features than data points there are infinitely many weight vectors that fit the training data perfectly, and the one it picks generalizes terribly. Regularization is what rescues it, and which kind depends on the signal. The rows in the next block are generated with `make_regression`, 90 samples and 400 features of which 10 are informative, so the sparse pattern is known.

# %%
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
X, y, coef = make_regression(n_samples=90, n_features=400, n_informative=10, noise=10.0, random_state=0, coef=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=1/3, random_state=0)   # 60 train rows, 400 features
ols = LinearRegression().fit(Xtr, ytr)
lasso = LassoCV(cv=5, max_iter=5000, random_state=0).fit(Xtr, ytr)
print(f"{r2_score(ytr, ols.predict(Xtr)):.3f}", f"{r2_score(yte, ols.predict(Xte)):.3f}")     # 1.000 -0.019
print(f"{r2_score(ytr, lasso.predict(Xtr)):.3f}", f"{r2_score(yte, lasso.predict(Xte)):.3f}")   # 1.000 0.971
kept = np.flatnonzero(np.abs(lasso.coef_) > 1e-8)
truth = np.flatnonzero(np.abs(coef) > 0)
print(int(kept.size), int(np.intersect1d(kept, truth).size), int(truth.size), f"{float(lasso.alpha_):.3f}")
# 57 9 10 0.354

# %% [markdown]

# Ordinary least squares achieves a perfect training $R^2$ of 1.000 and a test $R^2$ of $-0.019$, the signature of fitting noise, while the L1 regularization of 5.3 drives many coefficients to exactly zero. It does not isolate the ten genuine features: the fit keeps 57 of the 400 coefficients, and 9 of the 10 informative ones are among those 57, at `alpha` 0.354, with a test $R^2$ of 0.971. This is the same lesson as 5.3 stated as a regime: in $p \gg n$ problems, regularization is not optional, and sparse penalties shine when the true signal is sparse.
#
# There is also a strikingly cheap alternative to PCA when you only need to shrink dimensions while preserving distances: project the data onto a random lower-dimensional subspace. The Johnson-Lindenstrauss lemma, in the form proved by Dasgupta and Gupta (2003, Theorem 2.1), says that once the target dimension is at least $4\ln n/(\varepsilon^2/2 - \varepsilon^3/3)$, any $n$ points can be mapped into it by some map $f$ that keeps every squared distance inside a factor $1 \pm \varepsilon$. Their proof finds $f$ by random projection, and scikit-learn's `johnson_lindenstrauss_min_dim` returns this bound as a safe number of components for its random projections (scikit-learn developers, 2026d, johnson_lindenstrauss_min_dim API reference). For these 500 points and $\varepsilon = 0.1$ the bound is 5,326 components, more than the original 2,000. At $\varepsilon = 0.2$ it is 1,434. The next block is a milder check: the mean relative change of the unsquared distances on 500 generated points in 2,000 dimensions, not that worst-case guarantee on squared distances. Empirically the preservation is excellent.

# %%
import numpy as np
from sklearn.random_projection import GaussianRandomProjection, johnson_lindenstrauss_min_dim
print(int(johnson_lindenstrauss_min_dim(500, eps=0.1)),
      int(johnson_lindenstrauss_min_dim(500, eps=0.2)))
# 5326 1434
rng = np.random.default_rng(0)
Xhd = rng.normal(size=(500, 2000))             # 500 points in 2000 dimensions
def distortion(k):
    Xp = GaussianRandomProjection(n_components=k, random_state=0).fit_transform(Xhd)
    i, j = rng.integers(0, 500, 3000), rng.integers(0, 500, 3000)
    keep = i != j
    d_hi = np.linalg.norm(Xhd[i[keep]] - Xhd[j[keep]], axis=1)
    d_lo = np.linalg.norm(Xp[i[keep]] - Xp[j[keep]], axis=1)
    return round(float(np.mean(np.abs(d_lo - d_hi) / d_hi)), 3)
print(distortion(100), distortion(300))        # 0.055, 0.033   distances barely change

# %% [markdown]

# Projecting 2000-dimensional data down to 100 dimensions changes pairwise distances by only about 5 percent on average, and to 300 dimensions by about 3 percent, yet a random projection requires no training at all, just a random matrix, where PCA must compute eigenvectors of the data. The lemma's formal dimension bound is conservative, often larger than the original dimension for tight distortions, but in practice random projection is a fast, data-free way to compress very high-dimensional data before a distance-based method, the kind of trick that makes large-scale similarity search and the vector databases of 26.3 feasible.
#
# ##### 8.2.7 What we built, and a self-check
#
# Linear dimensionality reduction rotates and projects. PCA, the workhorse, finds the variance-maximizing directions as eigenvectors of the covariance matrix, equivalently through the SVD, compressing the digits to 21 of 64 components for 90 percent of the variance. Kernel PCA unfolds curved structure that linear PCA cannot, separating concentric circles a linear classifier then split perfectly. LDA beats PCA for class separation when labels exist, and PLS beats PCR when the reduction serves prediction, both because they use the target. ICA, factor analysis, and NMF decompose by independence, shared structure, and additive parts respectively. On the 70,000 MNIST images, randomized PCA captured 0.9501 of the variance with 154 components against the exact 0.9503, and incremental PCA captured 0.9502 while reading the data from disk in batches. And in the $p \gg n$ regime, regularization is mandatory, while random projection offers a training-free way to shrink dimensions with little distortion. The next section turns to nonlinear methods built for visualization, t-SNE and UMAP.
#
# Try these before moving on, with full solutions in Appendix H.
#
# 1. Derive why the first principal component is the top eigenvector of the covariance matrix.
# 2. What is the explained variance ratio, and how do you use it to choose the number of components?
# 3. How does kernel PCA differ from PCA, and why can it separate concentric circles?
# 4. When should you use LDA rather than PCA, and what is LDA's limit on the number of components?
# 5. Why can PCR miss a predictive signal that PLS captures?
# 6. In the $p \gg n$ regime, why does ordinary least squares fail and what fixes it?
# 7. What does the Johnson-Lindenstrauss lemma promise, and what is the practical advantage of a random projection over PCA?
# 8. Whitening and the scree plot are two ways of using PCA, and truncated SVD is a close relative that skips centering. What does whitening change about the transformed components, and why does truncated SVD fit a large sparse term-document matrix?
# 9. On standardized data, what is a PCA loading, and why can you flip a component's sign without changing the reconstruction?
# 10. Randomized PCA and incremental PCA both target data that an exact PCA finds costly. On MNIST at the 95 percent cut of 154 components, how much variance did each capture, why did the from-scratch sketch need power iterations, and when does each method earn its place?
#
# #### 8.3 Dimensionality reduction: nonlinear and manifold learning
#
# The linear methods of 8.2 can only rotate and project, so they flatten curved structure rather than unrolling it. Manifold learning methods bend. They rest on the manifold hypothesis: that high-dimensional data, like the 64-pixel images of digits, actually lies on a much lower-dimensional curved surface embedded in that high space, the way a sheet of paper is two-dimensional even when crumpled into three. These methods try to discover and flatten that surface, and they are used above all for one thing, visualization, producing the striking two-dimensional maps of high-dimensional data that appear throughout machine learning. How well each method preserves structure is measurable, by embedding the digits to 2D and checking whether the classes stay separated, and 8.3.3 runs that measurement across every method in this section, collecting the results in Table 8-2 once each method has been introduced.
#
# ##### 8.3.1 t-SNE
#
# t-SNE, t-distributed stochastic neighbor embedding, is the method behind most of those two-dimensional maps. Its goal is to preserve local neighborhoods: points close in the high-dimensional space should stay close in the map, even at the expense of global distances. It works in three steps. First it converts high-dimensional distances into probabilities, where the probability that point $i$ would pick point $j$ as a neighbor falls off as a Gaussian with distance, so nearby points have high similarity. Second, in the low-dimensional map it defines similar probabilities but using a Student-t distribution, whose heavy tails give distant points room and solve the crowding problem, the tendency for everything to pile up in the middle when many dimensions collapse to two. Third, it moves the map points by gradient descent to minimize the Kullback-Leibler divergence of 2.5.3 between the high-dimensional and low-dimensional probability distributions, so the two neighbor structures match as closely as possible.
#
# Each step has an exact formula, and together they are the whole algorithm (van der Maaten and Hinton, 2008, Sections 2 and 3). Write $\mathbf{x}_i$ for the high-dimensional points and $\mathbf{y}_i$ for their positions in the low-dimensional map, with $n$ points in all. The conditional probability that point $i$ picks point $j$ as a neighbor is a Gaussian in their distance, normalized over $i$'s other candidates,
#
# $$p_{j|i} = \frac{\exp\!\bigl(-\lVert \mathbf{x}_i - \mathbf{x}_j \rVert^2 / 2\sigma_i^2\bigr)}{\sum_{k \neq i} \exp\!\bigl(-\lVert \mathbf{x}_i - \mathbf{x}_k \rVert^2 / 2\sigma_i^2\bigr)}, \qquad p_{i|i} = 0,$$
#
# where $\sigma_i$ is a per-point bandwidth: each point gets its own, so dense regions use small neighborhoods and sparse regions large ones. These are symmetrized into the joint similarities the algorithm actually matches,
#
# $$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n},$$
#
# which guarantees every point contributes to the objective even if nothing picks it as a neighbor. In the map, the similarity between $\mathbf{y}_i$ and $\mathbf{y}_j$ uses the heavy-tailed Student-t distribution with one degree of freedom instead of a Gaussian,
#
# $$q_{ij} = \frac{\bigl(1 + \lVert \mathbf{y}_i - \mathbf{y}_j \rVert^2\bigr)^{-1}}{\sum_{k \neq l} \bigl(1 + \lVert \mathbf{y}_k - \mathbf{y}_l \rVert^2\bigr)^{-1}},$$
#
# whose slowly decaying tail is what lets moderately distant pairs sit far apart in the map, curing the crowding. The map positions are then chosen by gradient descent to minimize the KL divergence between the two distributions,
#
# $$C = \operatorname{KL}(P \,\|\, Q) = \sum_{i \neq j} p_{ij} \ln \frac{p_{ij}}{q_{ij}},$$
#
# which punishes hardest the pairs that are close in the data ($p_{ij}$ large) but far in the map ($q_{ij}$ small), exactly the neighborhood-preserving asymmetry intended. The gradient of that cost, when $p_{ij}$ and $q_{ij}$ are the joint probabilities, is $4\sum_{j\neq i}(p_{ij}-q_{ij})(\mathbf{y}_i-\mathbf{y}_j)(1+\lVert\mathbf{y}_i-\mathbf{y}_j\rVert^2)^{-1}$. On three fixed map points the formula gives the first point's gradient as $(-0.406, 0.144)$. A central finite difference of the cost, which moves that point $10^{-5}$ each way along each axis, gives the same two numbers, and the two estimates differ by less than $10^{-9}$.

# %%
import numpy as np
Y = np.array([[0.0, 0.0], [1.0, 0.2], [0.3, 0.9]])
P = np.array([[0.0, 0.45, 0.05], [0.45, 0.0, 0.05], [0.05, 0.05, 0.0]])
P = P / P.sum()
d2 = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
num = 1.0 / (1.0 + d2)
np.fill_diagonal(num, 0.0)
Q = num / num.sum()
grad = np.zeros_like(Y)
for i in range(3):
    for j in range(3):
        if i == j:
            continue
        grad[i] += 4.0 * (P[i, j] - Q[i, j]) * (Y[i] - Y[j]) * num[i, j]
print(f"{grad[0, 0]:.3f}", f"{grad[0, 1]:.3f}")
# -0.406 0.144

def kl(Y):                                   # the cost C = KL(P || Q) for map positions Y
    d2 = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    w = 1.0 / (1.0 + d2)
    np.fill_diagonal(w, 0.0)
    Qy = w / w.sum()
    m = P > 0                                # every pair i != j
    return float((P[m] * np.log(P[m] / Qy[m])).sum())

step = 1e-5
fd = []
for c in range(2):                           # move point 0 a little both ways along each axis
    E = np.zeros_like(Y)
    E[0, c] = step
    fd.append((kl(Y + E) - kl(Y - E)) / (2 * step))
gap = max(abs(fd[0] - grad[0, 0]), abs(fd[1] - grad[0, 1]))
print(f"{fd[0]:.3f}", f"{fd[1]:.3f}", gap < 1e-9)
# -0.406 0.144 True

# %% [markdown]

# The first line prints the formula's $-0.406$ and $0.144$. The second prints the finite difference, the same two numbers at three decimals, and True, because the larger of the two gaps between the estimates is below $10^{-9}$.
#
# Perplexity, roughly the number of neighbors each point considers, is the parameter that sets the neighborhood size. Typical values lie between 5 and 50 (van der Maaten and Hinton, 2008, p. 2582), and it sets the bandwidths $\sigma_i$. Formally the perplexity of point $i$'s neighbor distribution is $\operatorname{Perp}(P_i) = 2^{H(P_i)}$, where $H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$ is its Shannon entropy from 2.5.1, so perplexity is the entropy read as an effective neighbor count: a distribution spread evenly over 30 neighbors has perplexity 30. For each point, $\sigma_i$ is found by binary search so that $\operatorname{Perp}(P_i)$ equals the value you chose. It balances local against global structure: too small and clusters shatter into fragments, too large and they blur together. The starting layout matters too. By default scikit-learn's `TSNE` starts from principal component coordinates, `init='pca'`, the default since version 1.2, and its documentation notes that this start is usually more globally stable than the alternative, `init='random'` (scikit-learn developers, 2026d, TSNE API reference). How well the map keeps the large-scale arrangement depends largely on that start: Kobak and Linderman (2021) found that t-SNE and UMAP preserve global structure equally well when both start from an informative initialization, and equally poorly when both start at random.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
rng = np.random.default_rng(0)
X, y = load_digits(return_X_y=True)
idx = rng.choice(len(X), 1000, replace=False); X, y = X[idx], y[idx]   # subset for speed
for perp in [5, 30, 50]:
    emb = TSNE(n_components=2, perplexity=perp, random_state=0).fit_transform(X)
    print(perp, round(cross_val_score(KNeighborsClassifier(10), emb, y, cv=5).mean(), 3))
# 5 0.98    30 0.975    50 0.974

# %% [markdown]

# On the digits the loop prints 0.98 at perplexity 5 (0.980, with Python dropping the trailing zero), 0.975 at 30, and 0.974 at 50, dipping slightly as perplexity grows and the embedding favors coarser structure. These are the companion notebook's values in the book's reference environment (Python 3.13.14 and scikit-learn 1.9.1 on an Apple M1 Max). Even with random_state fixed, a different setup can change the third decimal: the book's earlier test setup, Python 3.12 on Windows, printed 0.982, 0.976, and 0.973 from the same code. The crucial cautions, the classic t-SNE misreadings that Wattenberg, Viégas, and Johnson (2016) illustrate, are these: the size of a cluster in a t-SNE plot is meaningless, the gaps between well-separated clusters may not mean anything either, the algorithm is stochastic, so runs that are not seeded can give different pictures, and at low perplexity it can make clumps out of pure noise. t-SNE is a tool for seeing local neighborhood structure, not for measuring distances, not for counting clusters reliably, and not for producing features to feed a downstream model.
#
# ##### 8.3.2 UMAP
#
# UMAP, uniform manifold approximation and projection (McInnes, Healy, and Melville, 2018), is the modern alternative, and on the digits it will match t-SNE in the side-by-side scores of 8.3.3, 0.980 against 0.975 on the 2D kNN. It is grounded in topology rather than probability: it builds a fuzzy graph connecting each point to its nearest neighbors, then optimizes a low-dimensional layout whose own neighbor graph matches it as closely as possible. Compared with t-SNE it has two practical advantages and one caveat. The first advantage is speed. Its authors report that it ran faster than every other method they timed on all of their test datasets except one very small one, and that it embedded the full set of 3 million GoogleNews word vectors in about 200 minutes, where a multicore Barnes-Hut t-SNE, the kind of t-SNE that scikit-learn's `TSNE` runs by default, needed several days (McInnes, Healy, and Melville, 2018, Section 5.4). The second is two intuitive knobs: `n_neighbors` trades local detail against global shape, and `min_dist` controls how tightly points pack within a cluster. The caveat is that much of any extra global arrangement it keeps, compared with t-SNE, comes from its informative default start, a spectral layout of the neighbor graph: as 8.3.1 noted, Kobak and Linderman (2021) found UMAP started at random preserving global structure as poorly as t-SNE started at random, so the relative positions of clusters still should not be read as measurements. For these reasons UMAP is a common choice for large maps and is what 8.7 uses to map the digits, though the same cautions about overreading the plot apply.
#
# The fuzzy graph and the layout each have a formula (McInnes, Healy, and Melville, 2018, Sections 3 and 4). For each point $\mathbf{x}_i$ and its $k$ nearest neighbors ($k$ is `n_neighbors`, 15 by default), let $\rho_i$ be the distance to the nearest neighbor and choose the bandwidth $\sigma_i$ so that $\sum_{j} \exp\bigl(-\max(0, d(\mathbf{x}_i, \mathbf{x}_j) - \rho_i)/\sigma_i\bigr) = \log_2 k$, the sum running over those $k$ neighbors. The directed edge from $i$ to a neighbor $j$ gets the membership strength
#
# $$w_{j|i} = \exp\!\left(-\frac{\max\bigl(0,\, d(\mathbf{x}_i, \mathbf{x}_j) - \rho_i\bigr)}{\sigma_i}\right),$$
#
# so the nearest neighbor always gets weight 1, and the two directions combine as $w_{ij} = w_{j|i} + w_{i|j} - w_{j|i}\,w_{i|j}$, the probability that at least one of the two edges exists. In the map, the similarity of $\mathbf{y}_i$ and $\mathbf{y}_j$ is the smooth curve
#
# $$v_{ij} = \frac{1}{1 + a\,\lVert \mathbf{y}_i - \mathbf{y}_j \rVert^{2b}},$$
#
# whose constants $a$ and $b$ are fitted to `min_dist`, and the layout minimizes the fuzzy set cross entropy between the two graphs,
#
# $$C = \sum_{i<j} \left[\, w_{ij} \ln \frac{w_{ij}}{v_{ij}} + (1 - w_{ij}) \ln \frac{1 - w_{ij}}{1 - v_{ij}} \right].$$
#
# Its gradient pulls each pair together with a force weighted by $w_{ij}$ and pushes it apart with a force weighted by $1 - w_{ij}$. With the defaults, `min_dist` 0.1 and a `spread` of 1.0, the fitted constants are $a = 1.577$ and $b = 0.895$. Take a neighbor at distance 1.5 from a point whose nearest neighbor sits at $\rho_i = 1.0$, with $\sigma_i = 0.5$: its membership is $e^{-(1.5 - 1.0)/0.5} = e^{-1} = 0.368$. Placed at map distance 1 the edge has $v = 1/(1 + 1.577) = 0.388$ and costs only 0.001, while pulled in to 0.5 it costs 0.214 and pushed out to 2.0 it costs 0.135, so the cost is smallest where the map similarity matches the membership. The next block computes these numbers with umap-learn's own curve fit; the package installs as umap-learn, version 0.5.12 in the book's reference environment, but imports as umap.

# %%
import numpy as np
from umap.umap_ import find_ab_params                  # pip install umap-learn
a, b = find_ab_params(1.0, 0.1)                        # spread 1.0 and min_dist 0.1, UMAP's defaults
print(round(float(a), 3), round(float(b), 3))
rho, sigma, d = 1.0, 0.5, 1.5                          # nearest-neighbor distance, bandwidth, distance to j
w = np.exp(-max(0.0, d - rho) / sigma)                 # high-dimensional membership of the edge
for dist in [0.5, 1.0, 2.0]:
    v = 1.0 / (1.0 + a * dist ** (2 * b))              # low-dimensional similarity at this map distance
    ce = w * np.log(w / v) + (1 - w) * np.log((1 - w) / (1 - v))   # this edge's cross-entropy term
    print(dist, round(float(w), 3), round(float(v), 3), round(float(ce), 3))
# 1.577 0.895
# 0.5 0.368 0.687 0.214
# 1.0 0.368 0.388 0.001
# 2.0 0.368 0.155 0.135

# %% [markdown]

# The first line prints the fitted constants, 1.577 and 0.895. Each later line prints a map distance, the membership 0.368, the map similarity at that distance (0.687 at 0.5, 0.388 at 1.0, and 0.155 at 2.0), and the edge's cost (0.214, 0.001, and 0.135).
#
# Trustworthiness penalizes points that become neighbors in the map without having been neighbors in the original space. With $N_i^k$ the $k$ map-neighbors of $i$ and $r(i,j)$ the rank of $j$ in the original space,
#
# $$T(k) = 1 - \frac{2}{nk(2n-3k-1)}\sum_i\sum_{j\in N_i^k}\max(0, r(i,j)-k).$$
#
# This is the form scikit-learn's `trustworthiness` function computes, with $n$ the number of points (scikit-learn developers, 2026d, trustworthiness API reference). A perfect map scores 1. Six points on a line, with the embedding swapping the third and fourth, score 0.833 at $k=1$ and 0.800 at $k=2$.

# %%
import numpy as np
from sklearn.manifold import trustworthiness
pts = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [10.0]])
emb = np.array([[0.0], [1.0], [3.0], [2.0], [4.0], [10.0]])
print(f"{trustworthiness(pts, emb, n_neighbors=1):.3f}",
      f"{trustworthiness(pts, emb, n_neighbors=2):.3f}")
# 0.833 0.800

# %% [markdown]

# On the same 1,000-digit subset used for t-SNE, computing UMAP and scoring it produces the trustworthiness value that Table 8-2 in 8.3.3 will report.

# %%
import warnings
warnings.filterwarnings("ignore")               # silence numba/UMAP convergence notices
import numpy as np
import umap                                      # pip install umap-learn
from sklearn.datasets import load_digits
from sklearn.manifold import trustworthiness
rng = np.random.default_rng(0)
X, y = load_digits(return_X_y=True)
idx = rng.choice(len(X), 1000, replace=False)
X = X[idx]                                       # same subset as t-SNE, for speed
emb = umap.UMAP(n_components=2, random_state=0).fit_transform(X)   # 64D down to 2D
print(round(trustworthiness(X, emb, n_neighbors=10), 3))          # 0.987

# %% [markdown]

# The embedding earns a trustworthiness of 0.987, essentially tied with t-SNE's 0.991 and far above PCA's 0.825. Fixing random_state makes the layout reproducible; without it UMAP, like t-SNE, gives a different picture each run.
#
# ##### 8.3.3 MDS, Isomap, LLE, autoencoders, and self-organizing maps
#
# This subsection covers five more methods: three classical ones built on distances or neighbor graphs, MDS, Isomap, and LLE, then autoencoders and the self-organizing map. Multidimensional scaling, MDS, preserves all pairwise distances as faithfully as a low-dimensional layout allows, minimizing the raw stress $\sum_{i<j}(d_{ij}-\lVert\mathbf{y}_i-\mathbf{y}_j\rVert)^2$. Classical MDS double-centers the squared distances, $B=-\tfrac12 J D^{(2)} J$ with $J=I-\tfrac1n\mathbf{1}\mathbf{1}^\top$, and reads the coordinates from the top eigenvectors of $B$; on Euclidean distances this reproduces PCA (Murphy, 2022, Section 20.4.4). On a generated 30 by 4 cloud those coordinates match PCA, absolute correlation 1.000 on each of the first two axes; it captures global structure but, scoring 0.665 here, misses the fine local separation t-SNE achieves. Isomap (Tenenbaum, de Silva, and Langford, 2000) improves on it for curved data by measuring distance along the manifold rather than straight through space: it builds a nearest-neighbor graph and uses shortest-path, geodesic, distances, so it can unroll a shape like a Swiss roll that straight-line distances would fold together, reaching 0.688 on the digits.
#
# Locally linear embedding, LLE (Roweis and Saul, 2000), is a third neighbor-graph method. Instead of measuring geodesic distances, it first reconstructs each point from its nearest neighbors, choosing the weights $w_{ij}$ that minimize $\sum_i\lVert\mathbf{x}_i-\sum_j w_{ij}\mathbf{x}_j\rVert^2$, with each point's weights summing to 1 and nonzero only on its neighbors, and then finds the low-dimensional layout that best preserves those same reconstruction weights, another way to keep the manifold's local structure while flattening it. On the digits that local focus pays off, lifting the 2D kNN to 0.837, the best of the three, though still well short of t-SNE and UMAP.

# %%
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
rng = np.random.default_rng(0)
Z = rng.normal(size=(30, 4))
D2 = pairwise_distances(Z) ** 2
n = len(Z)
J = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * J @ D2 @ J
evals, evecs = np.linalg.eigh(B)
idx = np.argsort(evals)[::-1]
mds = evecs[:, idx[:2]] * np.sqrt(np.maximum(evals[idx[:2]], 0))
pca = PCA(2).fit_transform(Z)
c0 = abs(float(np.corrcoef(mds[:, 0], pca[:, 0])[0, 1]))
c1 = abs(float(np.corrcoef(mds[:, 1], pca[:, 1])[0, 1]))
print(f"{c0:.3f}", f"{c1:.3f}")
# 1.000 1.000

# %% [markdown]

# The two correlations print 1.000 and 1.000.
#
# With all six methods now introduced, the loop of Example 8-18 produces every number in Table 8-2: embed the same 1,000-digit subset with each method, score a 10-class kNN on the 2D coordinates by cross-validation, and measure trustworthiness against the original 64-dimensional neighborhoods.
#
# Example 8-18. Six two-dimensional embeddings of the same 1,000 digits, each scored by the cross-validated accuracy of a 10-neighbor kNN and by trustworthiness.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE, trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import umap
rng = np.random.default_rng(0)
X, y = load_digits(return_X_y=True)
idx = rng.choice(len(X), 1000, replace=False); X, y = X[idx], y[idx]   # same subset as 8.3.1
methods = {"PCA":    PCA(n_components=2),
           "MDS":    MDS(n_components=2, init="random", random_state=0),
           "Isomap": Isomap(n_components=2, n_neighbors=10),
           "LLE":    LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=0),
           "t-SNE":  TSNE(n_components=2, perplexity=30, random_state=0),
           "UMAP":   umap.UMAP(n_components=2, random_state=0)}
for name, m in methods.items():
    emb = m.fit_transform(X)
    knn = cross_val_score(KNeighborsClassifier(10), emb, y, cv=5).mean()
    print(f"{name:7} kNN {knn:.3f}  trustworthiness {trustworthiness(X, emb, n_neighbors=10):.3f}")
# PCA     kNN 0.618  trustworthiness 0.825
# MDS     kNN 0.665  trustworthiness 0.848
# Isomap  kNN 0.688  trustworthiness 0.856
# LLE     kNN 0.837  trustworthiness 0.886
# t-SNE   kNN 0.975  trustworthiness 0.991
# UMAP    kNN 0.980  trustworthiness 0.987

# %% [markdown]

# The printed six lines are collected in Table 8-2. The MDS entry passes `init="random"` on purpose. A random start is the default in scikit-learn 1.9.1, but leaving the option unset makes it warn that the default will change to `'classical_mds'` in version 1.10; naming it silences the warning and keeps the MDS row, 0.665 and 0.848, reproducible after that change.
#
# Table 8-2. How faithfully each method preserves the digit structure in a 2D embedding (kNN accuracy on the embedding, and trustworthiness in [0, 1]).
#
# | Method | kNN accuracy in 2D | Trustworthiness | Type |
# | --- | --- | --- | --- |
# | PCA | 0.618 | 0.825 | linear |
# | MDS | 0.665 | 0.848 | nonlinear, global |
# | Isomap | 0.688 | 0.856 | nonlinear, manifold |
# | LLE | 0.837 | 0.886 | nonlinear, manifold |
# | t-SNE | 0.975 | 0.991 | nonlinear, local |
# | UMAP | 0.980 | 0.987 | nonlinear, local |
#
# The ordering tells the family's story in one screen. The linear projection trails, the global-distance methods MDS and Isomap improve on it modestly, the locally focused LLE gains real ground, and the two local-neighborhood methods t-SNE and UMAP dominate: a 10-class kNN on the PCA map scores 0.618, while on the t-SNE and UMAP maps it reaches 0.975 and 0.980, because the nonlinear methods place each digit's images in a tight, well-separated island that PCA's flat projection smears together, as Figure 8-6 shows. Trustworthiness, which measures how many of each point's true neighbors remain neighbors after embedding, tells the same story. On a curled-up manifold it is the local neighborhoods, not the global distances, that carry the class structure.
#
# Autoencoders, previewed here and developed in Chapter 24 (24.3), take the neural-network route: an encoder network compresses the input to a small bottleneck layer and a decoder reconstructs it, and the bottleneck is a learned nonlinear embedding. A linear autoencoder trained with squared error recovers the projection onto the subspace of the leading principal components, so the reconstruction matches PCA's, while the learned directions are a basis for that subspace rather than a unique list of principal axes (Baldi and Hornik, 1989). Nonlinear units make autoencoders the nonlinear generalization of PCA, and they scale to the largest datasets and integrate naturally with the deep learning to come.
#
# One classic method rounds out the family from an older tradition: the self-organizing map, or SOM, also called a Kohonen network after Teuvo Kohonen, who introduced it (Kohonen, 1982). A self-organizing map is a small grid of prototype vectors, say 5 by 5, that learns to spread itself over the data so that each grid cell comes to represent one patch of the data and neighboring cells represent neighboring patches. It is a clustering method and a manifold method at once: like k-means it summarizes the data with prototypes, and like t-SNE it preserves neighborhoods, because the prototypes are tied to fixed positions on a 2D grid, so mapping each point to its nearest prototype lays the whole dataset out on a small map whose geometry means something. A SOM puts high-dimensional data on one readable screen, and it remains a simple way to get a topology-preserving map with nothing but NumPy.
#
# Training is a neighborly version of the k-means update. Each prototype $\mathbf{w}_j$ owns a fixed grid position $\mathbf{r}_j$, its row and column. One training step draws a random data point $\mathbf{x}$, finds the best matching unit, the prototype closest to it, $b = \arg\min_j \lVert \mathbf{x} - \mathbf{w}_j \rVert$, and then moves the winner and every other prototype toward $\mathbf{x}$, each by an amount that decays with its grid distance from the winner:
#
# $$\mathbf{w}_j \leftarrow \mathbf{w}_j + \eta(t)\, h_{bj}(t)\, (\mathbf{x} - \mathbf{w}_j), \qquad h_{bj}(t) = \exp\!\left(-\frac{\lVert \mathbf{r}_b - \mathbf{r}_j \rVert^2}{2\,\sigma(t)^2}\right),$$
#
# where $\eta(t)$ is a learning rate and $\sigma(t)$ a neighborhood width, both decaying over training so the map organizes coarsely first and fine-tunes locally later. The neighborhood function $h_{bj}$ is the one ingredient k-means lacks, and it is what makes the map topological: dragging grid neighbors along with each winner forces cells that are adjacent on the grid to end up representing adjacent regions of the data. Plugging in numbers with $\eta = 0.5$ and $\sigma = 1$: the winner itself has $h = e^{0} = 1$ and moves half the distance to $\mathbf{x}$; a cell one grid step away has $h = e^{-1/2} = 0.61$ and moves 30 percent of the way; a cell three steps away has $h = e^{-4.5} = 0.011$ and barely stirs. Example 8-19 trains a 5 by 5 map on the standardized penguin measurements in 32 lines, then labels each cell with the majority species of the penguins that map to it, using the species only for this readout, never for training.
#
# Example 8-19. A self-organizing map from scratch on the penguins.

# %%
import numpy as np
import seaborn as sns
import collections
from sklearn.preprocessing import StandardScaler
pen = sns.load_dataset("penguins").dropna()
feats = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = StandardScaler().fit_transform(pen[feats].to_numpy(float))
species = pen["species"].to_numpy()                     # used only to label the finished map
rng = np.random.default_rng(0)
rows = cols = 5
W = rng.normal(0, 0.1, size=(rows * cols, X.shape[1]))  # one prototype per grid cell
gy, gx = np.divmod(np.arange(rows * cols), cols)
grid = np.column_stack([gy, gx]).astype(float)          # each unit's fixed (row, col) position
bmu0 = ((X[:, None, :] - W[None]) ** 2).sum(2).argmin(1)
print(round(np.sqrt(((X - W[bmu0]) ** 2).sum(1)).mean(), 3))   # 1.715 quantization error at start
steps = 20000
for t in range(steps):
    x = X[rng.integers(len(X))]                         # one random penguin
    b = np.argmin(((W - x) ** 2).sum(1))                # best matching unit
    frac = t / steps
    eta = 0.5 * (0.01 / 0.5) ** frac                    # learning rate decays 0.5 -> 0.01
    sig = 2.5 * (0.3 / 2.5) ** frac                     # neighborhood width decays 2.5 -> 0.3
    h = np.exp(-((grid - grid[b]) ** 2).sum(1) / (2 * sig * sig))
    W += eta * h[:, None] * (x - W)                     # neighbors move with the winner
bmu = ((X[:, None, :] - W[None]) ** 2).sum(2).argmin(1)
print(round(np.sqrt(((X - W[bmu]) ** 2).sum(1)).mean(), 3))    # 0.524 quantization error after
cells = []
for j in range(rows * cols):                            # majority species initial per cell, "." if empty
    sp = species[bmu == j]
    cells.append(collections.Counter(sp).most_common(1)[0][0][0] if len(sp) else ".")
for r in range(rows):
    print(" ".join(cells[r * cols:(r + 1) * cols]))

# %% [markdown]

# The quantization error, the average distance from each penguin to its best matching unit, falls from 1.715 on the random initial grid to 0.524 after training, so the prototypes have genuinely moved onto the data. The printed map is the topology paying off:
#
# ```
# G G G G G
# A . A . .
# A A C C C
# A A A C A
# A A A A A
# ```
#
# Gentoo (G) fills the entire top row, Adelie (A) holds the connected lower-left block, and Chinstrap (C) sits in a block on the middle right, each species occupying one contiguous territory even though the map never saw a label. The empty cells (.) fall on the second row, exactly the border between Gentoo, which is well separated in these measurements, and the other two species; and the one stray A inside Chinstrap territory sits where Adelie and Chinstrap genuinely overlap, the same confusion k-means made in 8.1.1. The experiment worth running is to hold $\sigma$ near zero from the start: the neighborhood function then never couples the cells, so the SOM degenerates into plain 25-means clustering, and you should expect the species to scatter across the grid instead of forming territories, a prediction worth checking because it shows exactly what the neighborhood update buys. The from-scratch loop is the whole method; for production work the minisom package wraps the same algorithm with more options.
#
# ![Two scatters of the digits in 2D: PCA smears the ten class colors into overlapping clouds, while t-SNE separates them into distinct islands.](figures/fig_08_06_pca-vs-tsne-digits.png)
#
# Figure 8-6. The same 64-dimensional digits in 2D: PCA (left) smears the classes together, while t-SNE (right) places each digit in a distinct island.
#
# ##### 8.3.4 What we built, and a self-check
#
# Manifold methods unroll curved structure for visualization. On the digits, t-SNE and UMAP placed each class in a clean island, lifting a 2D kNN from PCA's 0.618 to 0.975 and 0.980, because they preserve local neighborhoods where linear projection cannot. t-SNE matches a probability distribution over neighbors by minimizing KL divergence, with perplexity as its local-global knob, and must never be read for cluster sizes or distances. UMAP does the same neighborhood-preserving job and is a common choice when the map has to be fast. MDS, Isomap, and LLE preserve global distances, geodesic distances, and local reconstruction weights respectively, and autoencoders generalize PCA nonlinearly through a neural bottleneck. The self-organizing map spreads a grid of prototypes over the data, and its neighborhood update keeps grid neighbors representing data neighbors, cutting the penguins' quantization error from 1.715 to 0.524 while laying the three species out in contiguous territories. The next section turns from compression to finding the points that fit no pattern at all: anomaly detection.
#
# Try these before moving on, with full solutions in Appendix H.
#
# 1. What is the manifold hypothesis, and why does it motivate nonlinear dimensionality reduction?
# 2. Describe the three steps of t-SNE, and explain what the Student-t distribution fixes.
# 3. What does perplexity control, and what are three things you must not read from a t-SNE plot?
# 4. How does UMAP differ from t-SNE in method and in practical advantages?
# 5. How does Isomap measure distance differently from MDS, and why does that help on curved data?
# 6. In what sense is an autoencoder a nonlinear generalization of PCA?
# 7. UMAP and LLE were both introduced as neighbor-graph manifold methods. Roughly what trustworthiness does UMAP reach on the 2D digit embedding, and how does LLE decide where to place each point in the low-dimensional map?
# 8. In a self-organizing map, what does the neighborhood function $h_{bj}$ do during the update, and what does the map degenerate into if the neighborhood width $\sigma$ is held near zero from the start?
#
# #### 8.4 Anomaly detection
#
# Anomaly detection finds the points that do not belong, the fraudulent transaction among millions of normal ones, the failing machine, the network intrusion, the corrupted record. It is mostly an unsupervised problem because anomalies are rare and usually unlabeled: you cannot train a classifier on examples you do not have, so instead you model what normal looks like and flag whatever departs from it. This section covers four approaches, each defining departure differently, and compares them on a real benchmark: the Statlog (Shuttle) data, loaded from OpenML. The UCI description (doi 10.24432/C5WS31) says the set has 58,000 rows and nine numerical attributes, the first of them time, and a class coded 1 Rad Flow through 7 Bpv Open, with about 80 percent of the rows in class 1. The page thanks NASA for the use of the shuttle data. It does not call the columns sensor readings or the rare classes faults. The OpenML copy used here has nine numeric columns, A1 through A9, plus the class. Class 1 holds 45,586 of the 58,000 rows, and 45,586/58,000 = 0.786. Classes 2, 3, 6, and 7 hold 50, 171, 10, and 13 rows, which sum to 244. We treat class 1 as the common state and those four classes as the rare states. The two remaining classes, 4 and 5, are deliberately excluded: with 8,903 and 3,267 records they are alternative operating modes too common to count as anomalies, and folding them in would turn the task into ordinary classification rather than rare-event detection. We subsample the normals so the four detectors run quickly. Because the class labels exist, each method can be scored by ROC AUC, but the labels are used only for scoring, never for fitting.
#
# ##### 8.4.1 Statistical methods
#
# The oldest idea is to fit a probability distribution to the normal data and flag the low-probability points. In one dimension this is the familiar z-score: a point more than, say, three standard deviations from the mean, $|x - \mu|/\sigma > 3$, is an outlier. In many dimensions the generalization is the Mahalanobis distance from 6.3, $\sqrt{(\mathbf{x}-\boldsymbol\mu)^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)}$, which measures distance from the center in units of the data's own spread in every direction at once. A small synthetic cloud shows what that adds. Take 20 points whose first coordinate is a standard normal draw and whose second coordinate is 0.8 times the first plus 0.2 times a second, independent normal draw, so the two coordinates rise and fall almost together, and compare the first point's z-score on one axis with its Mahalanobis distance.

# %%
import numpy as np
rng = np.random.default_rng(0)
cloud = rng.normal(size=(20, 2))
cloud[:, 1] = 0.8 * cloud[:, 0] + 0.2 * cloud[:, 1]    # the second coordinate tracks the first
mu = cloud.mean(0)
cov = np.cov(cloud, rowvar=False, bias=True)           # 1/n covariance, matching std() below
diff = cloud[0] - mu                                   # the first point, measured from the center
z = diff / cloud.std(0)                                # its z-score on each axis separately
r = np.corrcoef(cloud, rowvar=False)[0, 1]             # how tightly the two coordinates move together
print(np.round(z, 3), round(float(r), 3))
print(f"{z[0]:.3f}", f"{float(np.sqrt(diff @ np.linalg.inv(cov) @ diff)):.3f}")
# [0.484 0.351] 0.988
# 0.484 0.965

# %% [markdown]

# The first line prints the point's z-scores on the two axes, 0.484 and 0.351, and the correlation of the two coordinates, 0.988. The second line prints the first coordinate's z-score, 0.484, next to the point's Mahalanobis distance, 0.965, about twice as large. Neither z-score is unusual, but the correlation makes the cloud a thin diagonal band. A point whose first coordinate sits 0.484 standard deviations above the mean would lie on the band with its second coordinate about $0.988 \times 0.484 = 0.478$ standard deviations up, and this point's second coordinate is only 0.351 up. It sits slightly off the band, in the direction where the cloud has almost no spread, and the Mahalanobis distance, which measures each direction in units of the spread along it, counts that small miss heavily. A z-score on one axis cannot see it.
#
# Scikit-learn's elliptic envelope applies the same distance to real data. It fits a robust Gaussian with the FastMCD algorithm named in the scikit-learn 1.9.1 `EllipticEnvelope` docstring, estimating the location and covariance from a subset of the points rather than from all of them, and then flags the points with large Mahalanobis distance. It is fast and interpretable, and it assumes the normal data is a single Gaussian blob, an assumption that fits the shuttle's compact normal state well, as Example 8-20 shows, though it breaks whenever normal data splits into several separated clusters. Example 8-20 loads the data, counts the rows in each class, builds a sample of 3,000 randomly drawn normal rows plus the 244 rare-class rows, standardizes it, and scores the envelope.
#
# Example 8-20. The Statlog (Shuttle) data from OpenML counted by class, then a sample of 3,000 normal rows and the 244 rare-class rows, standardized and scored by a robust elliptic envelope.

# %%
import warnings, numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)   # hide a benign determinant notice from the robust fit
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_auc_score
# Statlog (Shuttle) via OpenML: nine numeric columns A1..A9; class 1 is the common state.
shuttle = fetch_openml(name="shuttle", version=1, as_frame=False)
Xall = shuttle.data.astype(float)
yall = shuttle.target.astype(int)
cls, cnt = np.unique(yall, return_counts=True)          # rows in each of the seven classes
print({int(c): int(n) for c, n in zip(cls, cnt)})
# {1: 45586, 2: 50, 3: 171, 4: 8903, 5: 3267, 6: 10, 7: 13}
normal = yall == 1                       # the dominant normal operating state
anom = np.isin(yall, [2, 3, 6, 7])       # four rare classes, 2, 3, 6, and 7
rng = np.random.default_rng(0)
idx = rng.choice(np.where(normal)[0], 3000, replace=False)   # subsample normals for speed
X = np.vstack([Xall[idx], Xall[anom]])
y = np.r_[np.zeros(3000), np.ones(anom.sum())].astype(int)   # 1 = anomaly (used only to score)
X = StandardScaler().fit_transform(X)                        # standardize the nine features
contam = float(y.mean())                 # about 0.075: roughly 7.5% anomalies
ee = EllipticEnvelope(contamination=contam, random_state=0).fit(X)   # one robust Gaussian
print(round(roc_auc_score(y, -ee.decision_function(X)), 3))         # 0.951  (normal state is one blob)

# %% [markdown]

# The block prints the seven class counts quoted at the start of 8.4 and then the envelope's ROC AUC. The Gaussian method scores 0.951, and here it is the strongest of the four, because the shuttle's normal operating state really is one compact cloud in these nine numeric columns, exactly the shape a robust ellipse captures. Statistical methods are the right first try when the normal data really is unimodal, as it is here, and a poor fit when it splits into several separated clusters, where the ellipse also swallows the empty gaps between them and calls the points there normal.
#
# ##### 8.4.2 Isolation Forest
#
# Isolation Forest turns the problem inside out: instead of modeling normal points, it isolates anomalies. It builds many random trees, each splitting on a random feature at a random threshold, and measures how many splits it takes to isolate each point. Anomalies, sitting alone in sparse regions, get cut off in just a few splits, while normal points buried in dense clusters take many. The average path length to isolate a point becomes its anomaly score. The score is $s = 2^{-E[h(x)]/c(n)}$ with $c(n) = 2H_{n-1} - 2(n-1)/n$ and $H_m$ the harmonic number. For $n = 256$, $c(n) = 10.249$. A path of length 4 scores 0.763, a path of length 10 scores 0.508, and a path equal to $c(n)$ scores 0.5. The method assumes nothing about the shape of the normal data. Its cuts are parallel to the axes, and those cuts leave artifacts between clusters. Hariri, Carrasco Kind, and Brunner (2021) show the artifacts on anomaly-score maps and trace them to the branching rule. A two-blob check makes the same point with numbers.

# %%
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
B, _ = make_blobs(n_samples=[200, 200], centers=[[0, 0], [10, 10]], cluster_std=1.0, random_state=0)
iso_b = IsolationForest(n_estimators=500, random_state=0).fit(B)
pts = np.array([[0.0, 10.0], [10.0, 0.0], [-7.07, -7.07], [5.0, 5.0]])
print(iso_b.score_samples(pts).round(3))
# [-0.582 -0.608 -0.728 -0.663]

# %% [markdown]

# Lower `score_samples` means more anomalous. The empty corners (0, 10) and (10, 0), each 10 units from both centers, score -0.582 and -0.608. The outside point (-7.07, -7.07), also about 10 units from the nearest center, scores -0.728, and (5, 5), only about 7.07 units from each center, scores -0.663. The gaps between the blobs look more normal than points that are as far, or closer.

# %%
from sklearn.ensemble import IsolationForest   # continuing on X, y, contam, and ee of Example 8-20
from sklearn.metrics import precision_score
def harm(m):
    return float(np.sum(1.0 / np.arange(1, m + 1)))
cn = 2 * harm(255) - 2 * 255 / 256
print(f"{cn:.3f}", f"{2 ** (-4 / cn):.3f}", f"{2 ** (-10 / cn):.3f}")
# 10.249 0.763 0.508
iso = IsolationForest(contamination=contam, random_state=0).fit(X)
auc = roc_auc_score(y, -iso.score_samples(X))              # lower isolation score = more anomalous
caught = int(((iso.predict(X) == -1) & (y == 1)).sum())    # how many true anomalies were flagged
print(round(auc, 3), caught, "of 244 anomalies caught")    # 0.934  166 of 244 anomalies caught
iso_flag = iso.predict(X) == -1
ee_flag = ee.predict(X) == -1
print(f"{precision_score(y, iso_flag):.3f}", f"{precision_score(y, ee_flag):.3f}")
print(int(iso_flag.sum()), int(ee_flag.sum()), int((iso_flag & ee_flag).sum()))
# 0.680 0.570
# 244 244 157

# %% [markdown]

# Isolation Forest scores 0.934, a close second here, catching 166 of the 244 true anomalies by their short isolation paths. It is fast, scales to large high-dimensional data, and needs little tuning beyond the expected contamination fraction, which is why it is a widely used general-purpose detector. On this shuttle sample the Gaussian still leads on ROC AUC. Isolation Forest's flags are the more precise of the two: each detector flags 244 points, the count the contamination setting dictates, and only 157 of those points are flagged by both. Isolation Forest's precision is 0.680, which is 166 of the 244 true anomalies, and the ellipse's is 0.570, which is 139 of 244.
#
# ##### 8.4.3 Local Outlier Factor
#
# Local Outlier Factor, LOF, introduced by Breunig, Kriegel, Ng, and Sander (2000), defines an anomaly by local density rather than global position. It compares each point's density, how tightly its neighbors surround it, to the densities of those neighbors; a point whose neighborhood is much sparser than its neighbors' is an outlier. The ratio is the local outlier factor, around 1 for normal points and well above 1 for outliers. Its strength is catching local anomalies: a point that is not far from the data overall but is sparse relative to its immediate surroundings, which global methods miss.
#
# Breunig and coauthors make this precise in a chain of definitions (2000, Section 4, Definitions 3 to 7). Fix a neighbor count $k$, write $d(p, o)$ for the distance between points $p$ and $o$, and let $\mathrm{kdist}(o)$ be the distance from $o$ to its $k$th nearest neighbor. The reachability distance from $p$ to a neighbor $o$ is $\max\{\mathrm{kdist}(o), d(p,o)\}$, the true distance except that it never drops below $o$'s own neighborhood radius. The local reachability density of $p$ is the reciprocal of the mean reachability distance from $p$ to its $k$ nearest neighbors, and the LOF of $p$ is the mean of its neighbors' densities divided by its own. Take the points 0, 1, 2, and 10 on a line with $k = 1$. Each of 0, 1, and 2 has a nearest neighbor 1 away whose own kdist is 1, so each of their reachability distances is 1, each density is 1, and each LOF is 1. The point 10's nearest neighbor is 2, which is 8 away, so its reachability distance is $\max\{1, 8\} = 8$, its density is 1/8, and its LOF is $1/(1/8) = 8$. Scikit-learn stores the negative of the ratio.

# %%
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
Xlof = np.array([[0.0], [1.0], [2.0], [10.0]])
print(np.round(LocalOutlierFactor(n_neighbors=1).fit(Xlof).negative_outlier_factor_, 1))
# [-1. -1. -1. -8.]

# %% [markdown]

# It prints $-1, -1, -1, -8$, the hand result with the sign flipped: the far point is an 8 and the others sit at 1.

# %%
from sklearn.neighbors import LocalOutlierFactor   # continuing on X, y, and contam of Example 8-20
lof = LocalOutlierFactor(n_neighbors=20, contamination=contam).fit(X)
print(round(roc_auc_score(y, -lof.negative_outlier_factor_), 3))   # 0.7  (scored by local density)

# %% [markdown]

# LOF scores 0.700, the weakest of the four on this data, and the reason is instructive. The rare shuttle classes do not sit alone in sparse gaps, they arrive in their own small clusters, so each rare-class point's nearest neighbors are mostly other points from the same rare classes, which makes its local density look unremarkable and keeps its outlier factor near 1. LOF shines instead when anomalies are genuinely isolated inside otherwise dense, varying-density data, where a single global threshold fails but a local comparison succeeds. The tradeoff is its locality parameter, the number of neighbors, and its cost on large data, since it is distance-based like the kNN it descends from.
#
# ##### 8.4.4 One-Class SVM
#
# The one-class SVM adapts the support vector machine of 6.5 to anomaly detection. Instead of separating two classes, it learns a boundary that encloses most of the normal data in kernel feature space, treating points outside the boundary as anomalies,  with a parameter $\nu$ setting roughly what fraction may fall outside. One primal is $\min \tfrac12\lVert w\rVert^2 + \tfrac{1}{\nu n}\sum_i \xi_i - \rho$, subject to $w\cdot\phi(x_i)\ge \rho-\xi_i$ and $\xi_i\ge 0$. The slack weight is $1/(\nu n)$. At $\nu = 0.1$ and $n = 100$ that weight is $0.1$. With an RBF kernel it can wrap a curved, multi-cluster region.

# %%
from sklearn.svm import OneClassSVM   # continuing on X, y, and contam of Example 8-20
ocsvm = OneClassSVM(nu=contam, gamma=0.1).fit(X)         # RBF boundary around the normal region
print(round(roc_auc_score(y, -ocsvm.decision_function(X)), 3))     # 0.842

# %% [markdown]

# It scores 0.842, third here, and it is the most sensitive of the four to its kernel and $\nu$ settings: a different gamma can swing the number substantially, which is both its flexibility and its fragility. It scales worst to large data, so it is best on smaller, high-dimensional problems, the same niche where SVMs excel as classifiers.
#
# ##### 8.4.5 Choosing a detector
#
# Table 8-3 summarizes the four on the one-class shuttle sample: the robust Gaussian led at 0.951, Isolation Forest was second at 0.934, the one-class SVM third at 0.842, and LOF last at 0.700. That order is not a law. Treating shuttle classes 1, 4, and 5 together as normal, a 3,000-row sample with 2,362, 462, and 176 rows, and keeping the same 244 rare-class rows, the four detectors still rank the same way, at 0.877, 0.844, 0.728, and 0.676. Several normal modes do not by themselves send the Gaussian to the bottom on this data. ROC AUC and precision also disagree on the one-class sample: the Gaussian wins the first while Isolation Forest wins the second, 0.680 against 0.570 among the flagged points as 8.4.2 showed, a reminder to score anomaly detectors by the metric that matches how you will act on the flags.

# %%
# continuing with Xall, yall, and anom from Example 8-20
normal3 = np.isin(yall, [1, 4, 5])
rng = np.random.default_rng(0)
idx3 = rng.choice(np.where(normal3)[0], 3000, replace=False)
print({int(c): int(np.sum(yall[idx3] == c)) for c in (1, 4, 5)})
X3 = np.vstack([Xall[idx3], Xall[anom]])
y3 = np.r_[np.zeros(3000), np.ones(int(anom.sum()))].astype(int)
X3 = StandardScaler().fit_transform(X3)
contam3 = float(y3.mean())
ee3 = EllipticEnvelope(contamination=contam3, random_state=0).fit(X3)
iso3 = IsolationForest(contamination=contam3, random_state=0).fit(X3)
oc3 = OneClassSVM(nu=contam3, gamma=0.1).fit(X3)
lof3 = LocalOutlierFactor(n_neighbors=20, contamination=contam3).fit(X3)
print(round(roc_auc_score(y3, -ee3.decision_function(X3)), 3),
      round(roc_auc_score(y3, -iso3.score_samples(X3)), 3),
      round(roc_auc_score(y3, -oc3.decision_function(X3)), 3),
      round(roc_auc_score(y3, -lof3.negative_outlier_factor_), 3))
# {1: 2362, 4: 462, 5: 176}
# 0.877 0.844 0.728 0.676

# %% [markdown]

# Table 8-3. Anomaly detection methods compared.
#
# | Method | Defines anomaly as | AUC here | Best when |
# | --- | --- | --- | --- |
# | Gaussian / elliptic envelope | far in Mahalanobis distance | 0.951 | normal data is one Gaussian blob |
# | Isolation Forest | easy to isolate by random splits | 0.934 | general purpose, large or high-dimensional data |
# | Local Outlier Factor | low density vs neighbors | 0.700 | local outliers, varying-density data |
# | One-Class SVM | outside a learned boundary | 0.842 | small, high-dimensional data |
#
# The practical defaults: reach for Isolation Forest first as a fast, robust general detector, use LOF when anomalies are local or densities vary, use a one-class SVM on small high-dimensional data, and use the Gaussian method only when the normal data is genuinely unimodal. DBSCAN from 8.1.3 is a fifth option whenever its noise points coincide with the anomalies. Across all of them, set the contamination or $\nu$ parameter to your best estimate of the anomaly rate, and evaluate with whatever few labeled anomalies you can gather.
#
# ##### 8.4.6 Outlier detection versus novelty detection, and a self-check
#
# Every detector so far was fit on the full dataset, anomalies included, and asked to flag those same points. That is outlier detection: the training data is already contaminated, and the model marks the odd ones out in-sample. A different and often more useful setup is novelty detection, where the model is trained on data known to be clean, all normal, and then judges new, unseen points as normal or novel. The distinction is not cosmetic. It changes what the model learns and how you must call it.
#
# Outlier detection suits the exploratory case: a pile of records with an unknown few bad ones, find them now. Novelty detection suits the monitoring case: you have a clean baseline of healthy behavior and want to raise an alarm when future readings drift away from it, a failing engine, a fraud pattern, a fresh intrusion. Fitting on clean data usually gives a sharper boundary, because the anomalies are not there to distort the model's idea of normal.
#
# scikit-learn wires this distinction into its API. Isolation Forest and the one-class SVM support both modes naturally: fit on whatever data you have, then call predict or the scoring methods on new points. Local Outlier Factor is stricter. By default it does outlier detection only, exposing its scores through negative_outlier_factor_ on the training data and offering no way to score new points. To use LOF for novelty detection you must construct it with `novelty=True`. After that, `fit_predict` is unavailable and raises `AttributeError`. `predict`, `decision_function`, and `score_samples` still run on the training rows, and `negative_outlier_factor_` is still filled in. The scikit-learn 1.9.0 `LocalOutlierFactor` docstring says those three methods should be used only on new unseen data, because the results may differ from the standard LOF scores. That switch is easy to forget and a common source of confusion.
#
# The shuttle data shows the payoff. We train on 1,500 clean normal records only, then score a held-out mix of 1,500 unseen normals and the 244 rare-class rows, continuing with Xall, yall, normal, and anom from Example 8-20.
#
# Example 8-21. Outlier mode against novelty mode on the shuttle: LOF fit on a held-out mix of 1,500 unseen normal rows and the 244 rare-class rows, then LOF, a one-class SVM, and Isolation Forest fit on 1,500 clean normal rows and scored on that mix.

# %%
import numpy as np                    # continuing with Xall, yall, normal, anom from Example 8-20
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(1)
norm_idx = rng.permutation(np.where(normal)[0])
train_idx, test_norm = norm_idx[:1500], norm_idx[1500:3000]
Xtr = Xall[train_idx]                                   # clean normal-only training set
Xte = np.vstack([Xall[test_norm], Xall[anom]])          # unseen normals + the rare-class rows
yte = np.r_[np.zeros(1500), np.ones(anom.sum())].astype(int)
sc = StandardScaler().fit(Xtr)                          # scaler learned on clean data only
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

lof_out = LocalOutlierFactor(n_neighbors=20).fit(Xte)               # outlier mode: fit on the mix
print(round(roc_auc_score(yte, -lof_out.negative_outlier_factor_), 3))          # 0.52
lof_nov = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Xtr)  # novelty mode: fit on clean
print(round(roc_auc_score(yte, -lof_nov.decision_function(Xte)), 3))            # 0.995
ocsvm = OneClassSVM(nu=0.05, gamma=0.1).fit(Xtr)
print(round(roc_auc_score(yte, -ocsvm.decision_function(Xte)), 3))             # 0.995
iforest = IsolationForest(random_state=0).fit(Xtr)
print(round(roc_auc_score(yte, -iforest.score_samples(Xte)), 3))              # 0.953

# %% [markdown]

# The four printed numbers are 0.52, 0.995, 0.995, and 0.953. LOF run as an outlier detector on the contaminated mix scores 0.52, barely better than a coin flip, because the rare-class records cluster together and hide in each other's company. The very same algorithm switched to novelty mode with novelty=True and trained on clean normals scores 0.995: knowing what pure normal looks like, it flags the unseen rare-class rows almost perfectly. The one-class SVM matches it at 0.995 and Isolation Forest reaches 0.953, both trained on clean data and scoring new points. The lesson is that when you have a clean baseline, use it: train on normal-only data and detect novelties, rather than pouring everything into one contaminated pile and hoping the model can still see the difference.
#
# The same split supports three more scores, each built on a model the book has already met. The first is the PCA reconstruction error of 8.2.1: fit PCA on the clean rows, project each test row onto the kept components, and score it by the squared distance between the row and its projection. A few directions fitted to normal data rebuild normal rows well and unusual rows poorly, so a large residual flags an anomaly. The second and third are density scores: the Gaussian mixture of 8.1.4 with a single component, and the Gaussian kernel density estimate of 2.4.15 with bandwidth 1, both fit on the clean training rows. Each scores a test row by its negative log density, so a row in a region the clean data seldom visits gets a high score. The block continues with `Xtr`, `Xte`, and `yte` from Example 8-21: the 1,500 clean training rows and the held-out mix of 1,500 unseen normals and the 244 rare-class rows.

# %%
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
for kcomp in (2, 4, 6):
    p = PCA(n_components=kcomp).fit(Xtr)
    err = ((Xte - p.inverse_transform(p.transform(Xte))) ** 2).sum(1)   # squared residual per test row
    print(kcomp, round(float(roc_auc_score(yte, err)), 3),
          round(float(np.median(err[yte == 0])), 3), round(float(np.median(err[yte == 1])), 3))
gm = GaussianMixture(n_components=1, random_state=0).fit(Xtr)
kd = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(Xtr)
print(round(float(roc_auc_score(yte, -gm.score_samples(Xte))), 3),
      round(float(roc_auc_score(yte, -kd.score_samples(Xte))), 3))
# 2 0.995 1.43 223.263
# 4 0.995 0.611 123.934
# 6 0.609 0.004 0.005
# 0.995 0.995

# %% [markdown]

# Each of the first three lines gives a number of components, the ROC AUC of the reconstruction error used as the anomaly score, and the median squared residual of the normal and of the rare-class test rows. Scoring each test row by its PCA reconstruction error gives ROC AUC 0.995 at 2 components and at 4, then 0.609 at 6. The medians show why. At two components the rare-class rows' median residual is 223.263 against 1.43 for the normal rows, but at six components it falls to 0.005, next to the normal rows' 0.004: the extra components rebuild the rare-class rows as well as the normal ones, and the score loses its contrast. The last line shows the one-component mixture and the kernel density estimate both at 0.995.
#
# Try these before moving on, with full solutions in Appendix H.
#
# 1. Why is anomaly detection usually unsupervised, and how does that shape the methods?
# 2. How does the Mahalanobis distance generalize the z-score, and when does the Gaussian approach fail?
# 3. Explain how Isolation Forest scores a point, and why anomalies get short isolation paths.
# 4. What does Local Outlier Factor compare, and what kind of anomaly does it catch that a global method misses?
# 5. How does a one-class SVM decide what is anomalous, and what is its main practical drawback?
# 6. What is the difference between outlier detection and novelty detection, and what must you pass to scikit-learn's LocalOutlierFactor to use it for novelty detection?
#
# #### 8.5 Association rules
#
# Association rule mining answers a different unsupervised question: which items tend to appear together? Given a pile of transactions, each a basket of items, it finds rules of the form "if a basket contains A, it tends to contain B." This is market basket analysis, behind product recommendations, cross-selling, and store layout, and often introduced with the story that diapers and beer are bought together. The eight baskets below are a toy for the arithmetic, not that story. A rule $A \rightarrow B$ is judged by three metrics. Support is how often the items co-occur, the fraction of all baskets containing both. Confidence is the conditional probability $P(B \mid A)$, how often $B$ appears when $A$ does. Lift compares that to chance:
#
# $$\text{support}(X) = \frac{\#\{\text{baskets} \supseteq X\}}{N}, \quad \text{confidence}(A\rightarrow B) = \frac{\text{support}(A \cup B)}{\text{support}(A)}, \quad \text{lift}(A\rightarrow B) = \frac{\text{confidence}(A\rightarrow B)}{\text{support}(B)}.$$
#
# Lift is the key one: above 1 means the items are bought together more than chance would predict, exactly 1 means they are independent, and below 1 means buying one makes the other less likely. Working the classic rule on a small set of baskets makes the metrics concrete.
#
# Example 8-22. Support, confidence, and lift of the rule diapers to beer on eight toy baskets, with a one-sided Fisher exact test.

# %%
baskets = [{"bread", "milk"}, {"bread", "diapers", "beer", "eggs"}, {"milk", "diapers", "beer", "cola"},
           {"bread", "milk", "diapers", "beer"}, {"bread", "milk", "diapers", "cola"},
           {"diapers", "beer"}, {"bread", "milk", "diapers"}, {"beer", "diapers"}]
N = len(baskets)
def support(items):   return sum(set(items) <= b for b in baskets) / N   # fraction of baskets with all items
def confidence(A, B): return support(A | B) / support(A)                 # P(B | A)
def lift(A, B):       return confidence(A, B) / support(B)               # vs chance
A, B = {"diapers"}, {"beer"}
print(round(support(A), 3), round(support(B), 3), round(support(A | B), 3))   # 0.875 0.625 0.625
print(round(confidence(A, B), 3))    # 0.714   P(beer | diapers)
print(round(lift(A, B), 3))          # 1.143   0.143 above 1
from scipy.stats import fisher_exact
both = sum({"diapers", "beer"} <= b for b in baskets)
dia_only = sum(("diapers" in b) and ("beer" not in b) for b in baskets)
beer_only = sum(("beer" in b) and ("diapers" not in b) for b in baskets)
neither = sum(("diapers" not in b) and ("beer" not in b) for b in baskets)
table = [[both, dia_only], [beer_only, neither]]
print(table, round(fisher_exact(table, alternative="greater").pvalue, 3))
# [[5, 2], [0, 1]] 0.375

# %% [markdown]

# Diapers appear in 87.5 percent of baskets and beer in 62.5 percent; given diapers, beer appears 71.4 percent of the time, and the lift of 1.143 is 0.143 above 1, so beer shows up with diapers more often than independence predicts. On eight baskets that ratio is not a test against chance: the printed Fisher exact test on the diapers-by-beer table gives a one-sided p value of 0.375. Real rule mining uses thousands of baskets and a minimum support. Mining all such rules sounds like it would require checking every possible itemset, an exponential explosion, but the Apriori algorithm prunes the search with one observation, the Apriori principle: if an itemset is infrequent, every larger set containing it must be infrequent too, so once a set falls below the minimum support, all its supersets can be skipped. On the eight baskets, with support at least 0.3, cola at 0.250 and eggs at 0.125 are dropped before any pair is built. Of the 6 pairs among the four frequent items, beer with bread and beer with milk are 0.250 and drop. Bread, diapers, and milk still have support 0.375, because every pair inside that triple was frequent. Apriori finds the frequent itemsets above a support threshold, then forms rules above a confidence threshold; the FP-growth algorithm searches by a different route, which the next block checks on these baskets. Leverage and conviction are two more readings of the same rule. Leverage is $\mathrm{supp}(A\cup B)-\mathrm{supp}(A)\mathrm{supp}(B)$. Conviction is $(1-\mathrm{supp}(B))/(1-\mathrm{conf}(A\rightarrow B))$. For diapers to beer those are $0.625 - 0.875\times 0.625 = 0.0781$ and $(1-0.625)/(1-5/7) = 1.3125$. A level-by-level search, joining frequent sets and dropping any candidate with an infrequent subset, finds 9 itemsets at support 0.3, the same count the library reports next.
#
# Example 8-23. A level-by-level Apriori search from scratch at minimum support 0.3, with the leverage and conviction of diapers to beer.

# %%
from itertools import combinations
items = sorted(set().union(*baskets))
level = [frozenset([it]) for it in items if sum(it in b for b in baskets) / len(baskets) >= 0.3]
found = list(level)
k = 2
while level:
    cands = []
    for a, b in combinations(level, 2):
        u = a | b
        if len(u) != k:
            continue
        if all(frozenset(sub) in level for sub in combinations(u, k - 1)):
            cands.append(u)
    level = [c for c in set(cands) if sum(c <= b for b in baskets) / len(baskets) >= 0.3]
    found.extend(level)
    k += 1
print(len(found), round(0.625 - 0.875 * 0.625, 4), round((1 - 0.625) / (1 - 5 / 7), 4))
# 9 0.0781 1.3125

# %% [markdown]

# The line prints 9, 0.0781, and 1.3125. Example 8-23 reuses the eight `baskets` of Example 8-22.
#
# In practice you do not compute these by hand, you mine them with a library. The mlxtend package one-hot encodes the baskets with `TransactionEncoder`, finds the frequent itemsets with `apriori`, and turns them into ranked rules with `association_rules`.

# %%
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
te = TransactionEncoder()                              # sets are iterable, so baskets pass straight in
onehot = pd.DataFrame(te.fit_transform(baskets), columns=te.columns_)   # baskets x items, True/False
freq = apriori(onehot, min_support=0.3, use_colnames=True)             # frequent itemsets, support >= 0.3
rules = association_rules(freq, metric="confidence", min_threshold=0.6) # rules with confidence >= 0.6
one = lambda s: s.map(len) == 1
r = rules[one(rules["antecedents"]) & one(rules["consequents"])].copy() # keep single-item -> single-item
r["rule"] = (r["antecedents"].map(lambda s: list(s)[0]) + " -> "
             + r["consequents"].map(lambda s: list(s)[0]))
order = r.sort_values(["lift", "confidence", "rule"],               # the rule name breaks exact ties,
                      ascending=[False, False, True])              # so every run prints one order
for _, x in order.iterrows():
    print(f"{x['rule']:<16} sup={x.support:.2f}  conf={x.confidence:.2f}  lift={x.lift:.2f}")

# %% [markdown]

# This prints all six single-item rules that clear the thresholds, ordered by lift, then by confidence, then by name, so rules that tie on both numbers always print in the same order: `bread -> milk` and `milk -> bread` at support 0.50, confidence 0.80, lift 1.28; `beer -> diapers` at 0.62, 1.00, 1.14 and its mirror `diapers -> beer` at 0.62, 0.71, 1.14; then `bread -> diapers` and `milk -> diapers` at 0.50, 0.80, 0.91. These are exactly the numbers we computed by hand for the diapers-and-beer rule, now produced automatically for every pair. FP-growth mines the identical frequent itemsets by a different route, compressing the data into a tree instead of generating candidates level by level, which avoids building a fresh candidate list at every itemset size.

# %%
alt = fpgrowth(onehot, min_support=0.3, use_colnames=True)             # same support threshold
print(len(freq), set(map(frozenset, alt["itemsets"])) == set(map(frozenset, freq["itemsets"])))   # 9 True

# %% [markdown]

# This prints 9 and `True`: mlxtend's `apriori` found nine frequent itemsets, the count of Example 8-23, and FP-growth returns the same nine, so the rules built from them are identical. Sorting the mined rules by lift gives Table 8-4.
#
# Table 8-4. Mined association rules, support at least 0.3 and confidence at least 0.6, ranked by lift.
#
# | Rule | Support | Confidence | Lift |
# | --- | --- | --- | --- |
# | bread → milk | 0.50 | 0.80 | 1.28 |
# | milk → bread | 0.50 | 0.80 | 1.28 |
# | beer → diapers | 0.62 | 1.00 | 1.14 |
# | diapers → beer | 0.62 | 0.71 | 1.14 |
# | bread → diapers | 0.50 | 0.80 | 0.91 |
# | milk → diapers | 0.50 | 0.80 | 0.91 |
#
# The strongest association is bread with milk at lift 1.28, and beer implies diapers with perfect confidence 1.00, every beer basket also held diapers, yet its lift is only 1.14 because diapers are so common that the cooccurrence is partly expected. A rule and its mirror share support and lift and differ only in confidence, so both directions appear only when both clear the confidence threshold. Bread and milk do, which is why they show up as two rows at lift 1.28. Diapers to bread and diapers to milk do not: their confidence is $0.50/0.875 = 0.571$, below 0.6, so Table 8-4 has bread to diapers and milk to diapers without those mirrors. The bread-to-diapers rule has high confidence 0.80 but lift below 1, a warning that high confidence alone can mislead: diapers are frequent enough that 80 percent looks impressive but is actually slightly less than chance. This is why lift, not confidence, is the metric to trust when ranking rules.
#
# The eight baskets are a teaching toy. Real market basket analysis runs the identical pipeline over thousands of receipts. The Bread Basket dataset on OpenML records 20,507 item purchases across 9,465 transactions at a bakery in Edinburgh, with columns for the transaction id and the item bought. We group the rows into baskets, one-hot encode them, and mine the rules exactly as before, only lowering the support threshold to 1 percent because no single item fills a large share of a real bakery's baskets.

# %%
import pandas as pd
from sklearn.datasets import fetch_openml
from mlxtend.frequent_patterns import apriori, association_rules
raw = fetch_openml(data_id=43422, as_frame=True, parser="auto").frame   # The Bread Basket, a UK bakery
raw["Item"] = raw["Item"].str.strip("'").str.strip()                    # clean stray quotes
raw = raw[raw["Item"].str.lower() != "none"]                            # drop non-purchase rows
print(f"{raw['Transaction'].nunique()} baskets, {raw['Item'].nunique()} distinct items")
basket = raw.assign(v=True).pivot_table(index="Transaction", columns="Item",
                                        values="v", aggfunc="any", fill_value=False)   # one-hot
freq = apriori(basket, min_support=0.01, use_colnames=True)             # items in >= 1% of baskets
rules = association_rules(freq, metric="lift", min_threshold=1.0)       # positively associated pairs
one = lambda s: s.map(len) == 1
r = rules[one(rules["antecedents"]) & one(rules["consequents"])].copy()
r["rule"] = (r["antecedents"].map(lambda s: list(s)[0]) + " -> "
             + r["consequents"].map(lambda s: list(s)[0]))
for _, x in r.sort_values(["lift", "rule"], ascending=[False, True]).head(6).iterrows():   # name breaks ties
    print(f"{x['rule']:<26} sup={x.support:.3f}  conf={x.confidence:.2f}  lift={x.lift:.2f}")

# %% [markdown]

# The first line reports 9,465 baskets and 94 distinct items. The top six rules by lift are all sensible bakery pairings: cake and hot chocolate go together most strongly at lift 1.88 (in both directions), tea and cake at lift 1.60, and coffee and toast at lift 1.47. That last pair is the most actionable: `Toast -> Coffee` has confidence 0.70, meaning 70 percent of baskets with toast also contain coffee, so the two belong side by side on the menu. The other columns show how thin real baskets are. Cake with hot chocolate appears in only 1.1 percent of baskets (support 0.011), and the other two pairs in 2.4 percent (0.024). A rule and its mirror share support and lift, but confidence divides that shared support by the antecedent's own support, so the two directions differ: 0.11 from cake to hot chocolate against 0.20 back, 0.17 from tea to cake against 0.23 back, and only 0.05 from coffee to toast against 0.70 back. The coffee and toast pair has the same numerator both ways, so the much smaller confidence from coffee means coffee appears in far more baskets than toast. None of these rules were told to the algorithm; they fall out of the cooccurrence counts alone, which is the whole point of unsupervised rule mining.
#
# ##### 8.5.1 What we built, and a self-check
#
# Association rule mining turns piles of baskets into ranked if-then rules. Three metrics judge a rule: support, the fraction of baskets holding the itemset; confidence, the conditional probability $P(B \mid A)$; and lift, confidence divided by $B$'s base rate, the one to trust because it corrects for how common the consequent is. On the eight toy baskets the diapers-to-beer rule worked out by hand to support 0.625, confidence 0.714, and lift 1.143, and the mlxtend pipeline reproduced those numbers automatically for every pair, with FP-growth mining the identical nine frequent itemsets by a faster route and the Apriori principle, infrequent itemsets have infrequent supersets, keeping the search tractable. The bread-to-diapers rule showed why confidence misleads, 0.80 confidence but lift below 1, and the same pipeline scaled to the 9,465 real bakery baskets, surfacing cake with hot chocolate at lift 1.88 and toast implying coffee at confidence 0.70 with no supervision at all. The next section applies the same unsupervised spirit to text, finding the themes that run through documents.
#
# Try these before moving on, with full solutions in Appendix H.
#
# 1. Define support, confidence, and lift, and explain why lift is more trustworthy than confidence for ranking rules.
# 2. A rule has confidence 0.8 but lift 0.9. What does that tell you?
# 3. State the Apriori principle and explain how it prunes the search for frequent itemsets.
# 4. In the mlxtend pipeline, what does TransactionEncoder produce, and what do apriori and association_rules each take as input and return as output?
# 5. Apriori and FP-growth returned the identical frequent itemsets on the eight baskets. How do the two algorithms differ in how they search, and when would you prefer FP-growth?
#
# #### 8.6 Topic modeling
#
# Topic modeling is unsupervised structure-finding for text: given a collection of documents, discover the hidden themes running through them, where each topic is a distribution over words, like {space, nasa, orbit, launch}, and each document is a mixture of topics. No labels are given; the themes emerge from word cooccurrence. It powers document organization, search, and recommendation, and gives a compact set of features for downstream text models. Two classical methods dominate, and they map directly onto the factorizations of 8.2.
#
# Latent semantic analysis, LSA, is the linear-algebra approach: build the term-document matrix, weight it by TF-IDF so common words count less, and apply the truncated SVD of 8.2.1. The resulting components are latent semantic dimensions, directions in word space that capture themes, and LSA is essentially PCA for text. It is fast and it denoises, but its components carry signed weights and are not probabilities, which makes them harder to read. Latent Dirichlet allocation, LDA, is the probabilistic approach, and a quick warning: this LDA is Latent Dirichlet Allocation, unrelated to the Linear Discriminant Analysis of 6.3 despite sharing the letters. It is a generative model, introduced by Blei, Ng, and Jordan (2003), that assumes each document was written by first picking a mixture of topics and then drawing each word from one of those topics, with Dirichlet priors on both mixtures. A Dirichlet distribution is a distribution over probability vectors, lists of nonnegative numbers that sum to 1, and its parameter sets how concentrated the draws are; the end of this section gives its density and shows draws from it. Written out for $K$ topics and a vocabulary of $V$ words, the story is four sampling statements:
#
# $$\boldsymbol\phi_k \sim \mathrm{Dirichlet}(\eta, \ldots, \eta), \qquad \boldsymbol\theta_d \sim \mathrm{Dirichlet}(\alpha, \ldots, \alpha),$$
#
# $$z_{dn} \sim \mathrm{Categorical}(\boldsymbol\theta_d), \qquad w_{dn} \sim \mathrm{Categorical}(\boldsymbol\phi_{z_{dn}}).$$
#
# Each topic $k = 1, \ldots, K$ draws $\boldsymbol\phi_k$, its probabilities for the $V$ words, and each document $d$ draws $\boldsymbol\theta_d$, its probabilities for the $K$ topics. Then for each word position $n = 1, \ldots, N_d$ of document $d$, a hidden topic $z_{dn}$ in $\{1, \ldots, K\}$ is drawn from $\boldsymbol\theta_d$, and the word $w_{dn}$ in $\{1, \ldots, V\}$ is drawn from that topic's distribution. A categorical draw is one roll of a weighted die, the one-trial case of the multinomial, which is the name Blei, Ng, and Jordan use. Their Section 3 states the process for one document, starting with a draw of the document's length that plays no further part and is left out here, and their Section 5.4 adds the Dirichlet prior on the topics. The numbers $\alpha$ and $\eta$ are the two Dirichlet parameters, which scikit-learn calls `doc_topic_prior` and `topic_word_prior`; both default to $1/K$. Summing over the hidden topic gives the probability that a word of document $d$ is word $v$:
#
# $$p(w_{dn} = v \mid \boldsymbol\theta_d, \boldsymbol\phi_{1:K}) = \sum_{k=1}^{K} \theta_{dk}\,\phi_{kv}.$$
#
# With two topics, a document mix $\boldsymbol\theta_d = (0.7, 0.3)$, and a word whose probability is 0.05 under topic 1 and 0.001 under topic 2, the word's probability in that document is $0.7 \times 0.05 + 0.3 \times 0.001 = 0.0353$. Fitting LDA runs the story backward: from the words alone it infers the topic-word and document-topic distributions, and because everything is a non-negative probability, its topics are interpretable lists of words, much like the parts-based NMF of 8.2.4, which we also run here as a third method. Running all three on real newsgroup posts from four distinct subjects, messages from sci.space, rec.sport.hockey, comp.graphics, and talk.politics.mideast, shows how they differ.
#
# Example 8-24. LDA on four newsgroups: UMass coherence counted from the doc-term matrix, a scan of three to six topics by coherence and held-out perplexity, a four-seed check of four against five topics, and the top words of the four-topic fit.

# %%
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
cats = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.mideast"]
docs = fetch_20newsgroups(subset="train", categories=cats,
                          remove=("headers", "footers", "quotes")).data
cv = CountVectorizer(max_df=0.5, min_df=10, stop_words="english")
dtm = cv.fit_transform(docs)                      # 2341 documents by 3760 words
feats = cv.get_feature_names_out()

# UMass coherence, counted straight from the doc-term matrix with NumPy
B = (dtm > 0).astype(int)                         # 1 if a word appears in a document
dfreq = np.asarray(B.sum(axis=0)).ravel()         # documents containing each word
CO = (B.T @ B).toarray()                          # co-document counts, word by word
def umass(top_ids):                               # higher (less negative) = more coherent
    s = 0.0
    for m in range(1, len(top_ids)):
        for l in range(m):
            s += np.log((CO[top_ids[m], top_ids[l]] + 1) / dfreq[top_ids[l]])
    return s

# choosing the number of topics: coherence (full corpus) vs held-out perplexity
tr, te = train_test_split(dtm, test_size=0.25, random_state=0)
for k in (3, 4, 5, 6):
    full = LatentDirichletAllocation(n_components=k, max_iter=20,
                                     learning_method="batch", random_state=0).fit(dtm)
    coh = np.mean([umass(c.argsort()[::-1][:10]) for c in full.components_])
    split = LatentDirichletAllocation(n_components=k, max_iter=20,
                                      learning_method="batch", random_state=0).fit(tr)
    print(f"k={k}: mean coherence={coh:.1f}   held-out perplexity={split.perplexity(te):.0f}")
# k=3: mean coherence=-71.0   held-out perplexity=2054
# k=4: mean coherence=-65.4   held-out perplexity=2103   <- best of this one seed
# k=5: mean coherence=-65.6   held-out perplexity=2185
# k=6: mean coherence=-66.1   held-out perplexity=2205
for seed in range(4):
    pair = []
    for k in (4, 5):
        full_s = LatentDirichletAllocation(
            n_components=k, max_iter=20, learning_method="batch", random_state=seed).fit(dtm)
        pair.append(np.mean([umass(c.argsort()[::-1][:10]) for c in full_s.components_]))
    print(seed, f"{pair[0]:.2f}", f"{pair[1]:.2f}")
# 0 -65.44 -65.59
# 1 -68.91 -58.88
# 2 -55.36 -57.01
# 3 -72.97 -56.90

lda = LatentDirichletAllocation(n_components=4, max_iter=20,
                                learning_method="batch", random_state=0).fit(dtm)
for i, comp in enumerate(lda.components_):
    print("topic", i, round(umass(comp.argsort()[::-1][:10]), 1),
          [feats[j] for j in comp.argsort()[::-1][:8]])
# topic 0 -70.2 ['people','armenian','said','armenians','israel','turkish','jews','israeli']
# topic 1 -62.3 ['like','don','just','think','know','good','does','time']   (generic filler)
# topic 2 -72.7 ['space','image','edu','data','nasa','graphics','available','program']
# topic 3 -56.6 ['10','team','25','11','55','12','game','play']   (hockey scores)

# %% [markdown]

# LDA recovers readable themes with no labels: topic 0 is the Middle East politics group, topic 2 the technical space-and-graphics posts, topic 3 the hockey games (its top words include the score numbers 10, 25, 11), and topic 1 is a generic conversational topic of filler words such as like, don, and just, a common outcome, since real corpora contain chatter that belongs to no subject. The topics are not a perfect one-to-one match to the four source groups, space and graphics merge and a junk topic appears, which is normal: topic models surface themes that require human interpretation, not clean labels.
#
# The fitted model hands back the quantities of the story. Normalizing each row of `components_` gives a topic's estimated word distribution $\hat{\boldsymbol\phi}_k$, and `transform` gives each post's estimated topic mix $\hat{\boldsymbol\theta}_d$, so the word probability above can be read off a real post.

# %%
# continuing with lda, dtm, docs, and feats from Example 8-24
phi = lda.components_ / lda.components_.sum(axis=1, keepdims=True)   # row k: topic k's word distribution
theta = lda.transform(dtm)                                           # row d: post d's topic mix
lens = np.asarray(dtm.sum(axis=1)).ravel()                           # counted words in each post
d = int(lens.argmax())                                               # the longest post
team = list(feats).index("team")
print(int(lens[d]), docs[d][:24])
print(" ".join(f"{v:.3f}" for v in theta[d]), " ".join(f"{v:.4f}" for v in phi[:, team]))
print(f"{float(theta[d] @ phi[:, team]):.4f}", f"{float((theta[d] @ phi).sum()):.6f}")
print(lda.doc_topic_prior_, lda.topic_word_prior_)
# 5197 Archive-name: hockey-faq
# 0.016 0.000 0.149 0.835 0.0000 0.0022 0.0000 0.0108
# 0.0090 1.000000
# 0.25 0.25

# %% [markdown]

# The longest post holds 5,197 words that survive the vectorizer's filters, and it is the rec.sport.hockey FAQ. Its estimated mix is 0.016, 0.000, 0.149, and 0.835, mostly topic 3, the hockey topic, and the word team has probabilities 0.0000, 0.0022, 0.0000, and 0.0108 under the four topics. The formula gives $0.016 \times 0.0000 + 0.000 \times 0.0022 + 0.149 \times 0.0000 + 0.835 \times 0.0108 = 0.0090$, the first number on the third line, and the second number, 1.000000, confirms that the mixture is itself a distribution over the 3,760 words. The last line shows scikit-learn's two priors at their default, $1/K = 0.25$.
#
# The per-topic UMass coherence puts a number on how readable each topic is. It is the topic coherence that Mimno, Wallach, Talley, Leenders, and McCallum (2011, equation 1) defined, and it is read straight off the doc-term matrix, with no held-out data. List a topic's $M$ most probable words in descending order of weight, $v_1, \ldots, v_M$ (here $M = 10$), let $D(v)$ be the number of documents that contain word $v$, and let $D(v, v')$ be the number that contain both $v$ and $v'$. Then
#
# $$C_{\mathrm{UMass}} = \sum_{m=2}^{M}\sum_{l=1}^{m-1}\log\frac{D(v_m, v_l) + 1}{D(v_l)},$$
#
# a sum over all $M(M-1)/2 = 45$ pairs in which each word is scored against every word ranked above it. Without the $+1$, the fraction would be the share of the posts containing $v_l$ that also contain $v_m$; the authors add the 1 to avoid taking the logarithm of zero for a pair that never shares a post. Since $D(v_m, v_l) \le D(v_l)$, each term is at most $\log\big(1 + 1/D(v_l)\big)$, barely above zero, and it falls far below zero when the two words seldom appear together, so a higher (less negative) total means the topic's words really do appear together in the same posts. One pair from the hockey topic shows the arithmetic. Its second word is team and its seventh is game; team appears in 223 posts, 71 of which also contain game, so that pair adds $\log(72/223) = -1.131$ to the topic's total of $-56.6$.

# %%
team, game = list(feats).index("team"), list(feats).index("game")   # continuing with feats, dfreq, and CO of Example 8-24
print(int(dfreq[team]), int(CO[game, team]), f"{float(np.log((CO[game, team] + 1) / dfreq[team])):.3f}")
# 223 71 -1.131

# %% [markdown]

# The line prints 223, 71, and $-1.131$.
#
# The hockey topic scores best at -56.6 and the merged space-and-graphics topic worst at -72.7, which matches the intuition that a blended topic reads less cleanly. Averaging coherence over the topics is one way to compare topic counts. On random_state 0 the scan is best at four topics (mean -65.4, against -71.0 at three, -65.6 at five, and -66.1 at six). That margin over five is 0.2 on the one-decimal print. The next block of prints, seeds 0 to 3 at four and at five topics, is -65.44, -68.91, -55.36, -72.97 against -65.59, -58.88, -57.01, -56.90. The means are $(-65.44-68.91-55.36-72.97)/4 = -65.67$ and $(-65.59-58.88-57.01-56.90)/4 = -59.60$. Four wins on seeds 0 and 2. Five wins on seeds 1 and 3. On this corpus coherence does not separate four topics from five.
#
# Perplexity measures something else: how well the fitted model predicts the words of documents it has not seen. Blei, Ng, and Jordan (2003, Section 7.1) define it over a set of held-out documents as
#
# $$\mathrm{perplexity} = \exp\left(-\frac{\sum_{d} \log p(\mathbf{w}_d)}{\sum_{d} N_d}\right),$$
#
# where both sums run over the held-out documents $d$, $N_d$ is the number of words in document $d$, $\mathbf{w}_d$ is those words, and $p(\mathbf{w}_d)$ is the probability the model gives them, with the document's topic mix averaged over its Dirichlet prior (their equation 3). The exponent is minus the average log probability per word, so perplexity is the reciprocal of the geometric mean of the per-word probabilities, and lower is better. Two anchors fix the scale. A model that gave each of this corpus's 3,760 vocabulary words probability $1/3{,}760$ would have $\log p(\mathbf{w}_d) = -N_d \log 3{,}760$ and a perplexity of exactly 3,760, and a model that predicted every word with certainty would score 1. Blei, Ng, and Jordan (2003, Section 5.1) call the exact average over the topic mix intractable, so scikit-learn's `perplexity` uses in its place the approximate log likelihood that its `score` method returns, which its docstring calls an approximate bound. The next block checks that arithmetic on the four-topic split.

# %%
split4 = LatentDirichletAllocation(n_components=4, max_iter=20,      # continuing with tr and te from the topic-count scan of Example 8-24
                                   learning_method="batch", random_state=0).fit(tr)
bound, n_words = split4.score(te), te.sum()       # approximate log likelihood, held-out word count
print(te.shape[0], int(n_words), f"{float(bound):.1f}")
print(f"{float(np.exp(-bound / n_words)):.1f}", f"{float(split4.perplexity(te)):.1f}")
# 586 58100 -444539.2
# 2103.3 2103.3

# %% [markdown]

# The held-out quarter holds 586 posts and 58,100 counted words, the approximate log likelihood is $-444{,}539.2$, and $\exp(444{,}539.2/58{,}100) = 2103.3$, which is what `perplexity` prints and, rounded, the 2103 of the scan.
#
# Perplexity disagrees with coherence on this corpus. It keeps favoring the smallest model, rising from 2054 at three topics to 2205 at six, so the two scores do not pick the same model; the evidence is the perplexity and coherence columns of the topic-count scan in Example 8-24. Chang, Boyd-Graber, Gerrish, Wang, and Blei (2009) report a related tension measured with people rather than formulas: in their user studies, topic models with better held-out likelihood sometimes produced topics that people judged less meaningful. Coherence, which rewards co-occurring words, is the metric aimed at interpretability, and Mimno and coauthors found that it agreed with expert ratings of topic quality better than topic size did, so we read the four-topic model.
#
# LSA and NMF, the other two methods, start from TF-IDF weights rather than raw counts. Scikit-learn's TF-IDF multiplies a raw count by $\ln((1+n)/(1+\mathrm{df}))+1$, where $n$ is the number of documents and $\mathrm{df}$ is how many documents contain the term, then divides each row by its Euclidean length when `norm='l2'`. On three short documents, `red wine fruit`, `red fruit fruit`, and `white wine`, the unnormalized weight of a term that appears once and in two documents is $\ln(4/3)+1 = 1.288$. The word `white` appears in one document, so its weight is $\ln(2)+1 = 1.693$. After L2 normalization the first row has entries 0.577 and length 1. Section 15.2 will derive the plain weighting $\ln(N/\mathrm{df})$ that this smoothed form adjusts, with its own worked numbers.

# %%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
docs = ["red wine fruit", "red fruit fruit", "white wine"]
vec = TfidfVectorizer(norm=None).fit(docs)
raw = vec.transform(docs).toarray()
l2 = TfidfVectorizer(norm="l2").fit(docs).transform(docs).toarray()
print(list(vec.get_feature_names_out()))
print(np.round(raw[0], 3).tolist(), round(float(np.linalg.norm(l2[0])), 3))
# ['fruit', 'red', 'white', 'wine']
# [1.288, 1.288, 0.0, 1.288] 1.0

# %% [markdown]

# The block prints the vocabulary ['fruit', 'red', 'white', 'wine'], the unnormalized first row [1.288, 1.288, 0.0, 1.288], and 1.0, the length of the first row after L2 normalization. LSA on the newsgroup posts, with this weighting, is messier than LDA.

# %%
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
cats = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.mideast"]
docs = fetch_20newsgroups(subset="train", categories=cats,
                          remove=("headers", "footers", "quotes")).data
tfidf = TfidfVectorizer(max_df=0.5, min_df=10, stop_words="english")
X = tfidf.fit_transform(docs)
feats = tfidf.get_feature_names_out()
lsa = TruncatedSVD(n_components=4, random_state=0).fit(X)
print(round(lsa.explained_variance_ratio_.sum(), 3))     # 0.028  (text is sparse and high-dimensional)
for i, comp in enumerate(lsa.components_):
    print("topic", i, [feats[j] for j in np.abs(comp).argsort()[::-1][:8]])
# topic 0: don, just, like, space, people, know, think, israel   (common words dominate)
# topic 1: israel, armenian, armenians, jews, israeli, turkish, thanks, people   (politics)
# topic 2: game, team, hockey, space, play, players, season, games   (hockey, with space bleeding in)
# topic 3: israel, armenian, space, armenians, israeli, turkish, arab, jews   (politics and space blended)

# %% [markdown]

# LSA's four components explain only 2.8 percent of the variance, because text is extremely high dimensional and sparse, and its first component is dominated by common words (don, just, like) rather than a clear theme, while later components do recover the politics and hockey themes but with signed weights that blend subjects, one direction mixing Israel, Armenia, and space together. Its components are directions in word space, not word distributions, so a large negative weight is as meaningful as a large positive one, which is why we rank by absolute value and why the result reads less cleanly than LDA. NMF offers a third route: factorize the same TF-IDF matrix into nonnegative parts, as 8.2.4 introduced, so each topic is again an additive list of words, but through matrix factorization rather than a probability model.

# %%
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
cats = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.mideast"]
docs = fetch_20newsgroups(subset="train", categories=cats,
                          remove=("headers", "footers", "quotes")).data
tfidf = TfidfVectorizer(max_df=0.5, min_df=10, stop_words="english")
X = tfidf.fit_transform(docs)
feats = tfidf.get_feature_names_out()
nmf = NMF(n_components=4, init="nndsvda", max_iter=400, random_state=0).fit(X)
for i, comp in enumerate(nmf.components_):
    print("topic", i, [feats[j] for j in comp.argsort()[::-1][:8]])
# topic 0: space, thanks, graphics, know, program, files, image, nasa
# topic 1: israel, israeli, jews, arab, arabs, jewish, peace, lebanese
# topic 2: game, team, hockey, play, players, season, games, nhl
# topic 3: armenian, armenians, turkish, people, armenia, turkey, genocide, turks

# %% [markdown]

# NMF produced the cleanest topics of the three here, four crisp themes with no filler topic at all: one for space and graphics together, one for Israel, one for hockey, and one for Armenia and Turkey. It even split the politics group into its two distinct conflicts where LDA had folded them together. The three methods sit on a spectrum. LSA is the fastest and a direct application of the SVD, but its signed components are the hardest to read. LDA is the probabilistic default, giving proper word-and-topic distributions and a principled generative story at the cost of slower inference. NMF sits between them: like LDA it keeps every weight nonnegative and builds topics only by addition, so its topics are interpretable additive lists, but it is a deterministic matrix factorization like LSA rather than a probability model, which makes it fast and often strikingly clean on TF-IDF text. Fitting LDA means inferring the hidden $\boldsymbol\theta_d$, $\boldsymbol\phi_k$, and $z_{dn}$ from the words alone. Blei, Ng, and Jordan (2003, Section 5.1) call the exact posterior intractable and list Laplace, variational, and Markov chain Monte Carlo approximations, and Chapter 9 teaches all three, with the conjugate Dirichlet and multinomial pair in 9.3, Markov chain Monte Carlo and its Gibbs special case in 9.4.1 and 9.4.2, variational inference in 9.4.4, and the Laplace approximation in 9.4.5. scikit-learn's `LatentDirichletAllocation` takes the variational route: its documentation names the method online variational Bayes and bases the implementation on Hoffman, Blei, and Bach (2010) and Hoffman, Blei, Wang, and Paisley (2013).
#
# A Dirichlet distribution is a distribution on vectors of nonnegative numbers that sum to 1, the same shape as a document's topic mix. For a parameter vector $\boldsymbol\alpha = (\alpha_1, \ldots, \alpha_K)$ with positive entries, its density at a mix $\boldsymbol\theta$ is
#
# $$\mathrm{Dir}(\boldsymbol\theta; \boldsymbol\alpha) = \frac{\Gamma\big(\sum_{k=1}^{K} \alpha_k\big)}{\prod_{k=1}^{K} \Gamma(\alpha_k)}\prod_{k=1}^{K} \theta_k^{\alpha_k-1},$$
#
# where $\Gamma$ is the gamma function of 2.1.6 and the fraction in front makes the density integrate to 1 over all mixes (Blei, Ng, and Jordan, 2003, equation 1). With $\boldsymbol\alpha = (1, 1, 1)$ the fraction is $\Gamma(3)/\Gamma(1)^3 = 2$ and every exponent is 0, so the density is 2 at every three-part mix: the uniform distribution on the triangle of mixes, whose area, measured in the first two coordinates, is 1/2. With $\boldsymbol\alpha = (2, 2, 2)$ the fraction is $\Gamma(6)/\Gamma(2)^3 = 120$, and at the mix $(0.2, 0.3, 0.5)$ the density is $120 \times 0.2 \times 0.3 \times 0.5 = 3.6$. A small shared $\alpha$ piles each draw onto one coordinate. A larger $\alpha$ spreads the draw. Two thousand draws with three equal parameters show the difference: at $\alpha = 0.2$ the largest coordinate averages 0.813 and exceeds 0.8 in 0.582 of the draws. At $\alpha = 2$ the largest coordinate averages 0.534 and exceeds 0.8 in only 0.021 of the draws.

# %%
import numpy as np
from scipy.stats import dirichlet
rng = np.random.default_rng(0)
for a in (0.2, 2.0):
    s = dirichlet.rvs([a, a, a], size=2000, random_state=rng)
    print(a, round(float(s.max(1).mean()), 3), round(float(np.mean(s.max(1) > 0.8)), 3))
# 0.2 0.813 0.582
# 2.0 0.534 0.021
print(round(float(dirichlet.pdf([0.2, 0.3, 0.5], [1.0, 1.0, 1.0])), 3),
      round(float(dirichlet.pdf([0.2, 0.3, 0.5], [2.0, 2.0, 2.0])), 3))   # the two densities by hand
# 2.0 3.6

# %% [markdown]

# The first two lines print 0.2, 0.813, 0.582 and then 2.0, 0.534, 0.021, and the last prints the two densities, 2.0 and 3.6. These draws are what the $\boldsymbol\theta_d$ line of LDA's story makes, and the four-topic fit used scikit-learn's default $\alpha = 1/K = 0.25$.
#
# ##### 8.6.1 What we built, and a self-check
#
# Topic modeling found the themes in 2,341 newsgroup posts with no labels. LDA, the probabilistic model in which each document mixes topics and each topic distributes over words, recovered the Middle East politics, space-and-graphics, and hockey themes plus one generic filler topic, and UMass coherence, computed straight from word cooccurrence in the documents, both ranked the topics by readability, hockey best at $-56.6$, and on random_state 0 was highest at four topics, a margin smaller than the seed-to-seed spread reported above, where held-out perplexity kept preferring fewer and illustrated why word-prediction quality is a poor guide to interpretability. LSA, truncated SVD on TF-IDF, was fastest but its signed components blended subjects, while NMF's additive parts gave the cleanest four themes of all. Topic modeling by meaning rather than by word counts, which embeds the posts and clusters the vectors, waits for the sentence embeddings of 26.2. This section stops at bag-of-words topics. The chapter's lab now puts the two halves, clustering and dimensionality reduction, to work on real business and image data.
#
# Try these before moving on, with full solutions in Appendix H.
#
# 1. What is a topic in topic modeling, and in what sense is a document a mixture of topics?
# 2. How does LSA relate to PCA and the SVD, and why are its components hard to interpret?
# 3. Why are LDA's topics more interpretable than LSA's?
# 4. Why did the topic model produce a generic filler topic rather than four clean subject topics?
# 5. UMass topic coherence is computed only from the documents themselves, with no held-out data. How is it defined for a topic's top words, and why is one seed's average a weak guide to the number of topics on this corpus?
# 6. On this corpus perplexity keeps favoring fewer topics, while one seed of coherence is highest at four and the mean over four seeds is higher at five. What does that say about using either number to choose how many topics to fit?
# 7. NMF and LDA both give non-negative, additive topics, yet NMF is not a probabilistic model. What is the core difference between them, and how does each differ from LSA?
# 8. In scikit-learn's TF-IDF, what happens to the weight of a term when it appears in more documents, and what does a small Dirichlet $\alpha$ do to a topic mix?
# 9. Write LDA's generative process for $K$ topics and a vocabulary of $V$ words, saying what each symbol is, and give the probability that one word of document $d$ is word $v$ once $\boldsymbol\theta_d$ and the topics are known. What does it give for the word team in the hockey FAQ?
# 10. Perplexity is $\exp\big(-\sum_d \log p(\mathbf{w}_d) / \sum_d N_d\big)$. Why is lower better, what would a model score that gave each of the 3,760 vocabulary words probability $1/3{,}760$, and what does scikit-learn use in place of the exact $\log p(\mathbf{w}_d)$?
#
# #### 8.7 Hands-on lab: customer segmentation and digit visualization
#
# This lab applies the chapter's two halves to two real tasks. Part A, Example 8-25, segments customers with clustering and turns the clusters into business-readable personas, a common commercial use of unsupervised learning. Part B, Example 8-26, visualizes high-dimensional images by projecting them to two dimensions with four methods, a common research use. The full script is `code/ch08_unsupervised/07_lab.py`.
#
# ##### 8.7.1 Part A: customer segmentation
#
# The data is the Wholesale Customers dataset, 440 real wholesale clients of a Portuguese distributor (from the UCI repository, loaded here through OpenML), each described by annual spending in six product categories: fresh produce, milk, grocery, frozen, detergents and paper, and delicatessen. The goal is to group the clients into segments a sales team can act on. The dataset carries two attributes that are not spending: the sales channel, Horeca versus Retail, which OpenML exposes as the target, and the client's region of Portugal, Lisbon, Oporto, or elsewhere. We set both aside, fit k-means on the six spends alone, and use the held-out channel label only at the end, as an external check. Because the spends are heavily right-skewed, running from single digits to over a hundred thousand (column minimums 3, 55, 3, 25, 3, and 3, maximum 112,151), we take log1p of each category before standardizing so a single large account does not dominate the distances.
#
# Example 8-25. Customer segmentation on the wholesale spends.

# %%
import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
ds = fetch_openml("wholesale-customers", as_frame=True, parser="auto")
# OpenML names the columns V1..V7: V1 = Region (1 Lisbon, 2 Oporto, 3 other) and
# V2..V7 = the six annual spends (Fresh, Milk, Grocery, Frozen, Detergents_Paper,
# Delicassen); the frame repeats the region as a categorical Region column, and the
# target is Channel. We keep only the six spends, dropping Region and holding out Channel.
spend = ds.data[["V2", "V3", "V4", "V5", "V6", "V7"]].copy()
spend.columns = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
channel = ds.target.astype(int).to_numpy()    # 1 = Horeca (hotel/restaurant/cafe), 2 = Retail
raw = spend.to_numpy(float)
print(raw.min(0).astype(int).tolist(), raw.max(0).astype(int).tolist(), int((raw < 100).sum()), int((raw < 100).any(1).sum()))
# [3, 55, 3, 25, 3, 3] [112151, 73498, 92780, 60869, 40827, 47943] 95 79
X = np.log1p(raw)                              # spends run from 3 to 112151
Xs = StandardScaler().fit_transform(X)
for k in range(2, 8):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
    print(k, f"{silhouette_score(Xs, lab):.3f}")
# 2 0.290   3 0.259   4 0.188   5 0.195   6 0.193   7 0.196
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Xs)
print(round(adjusted_rand_score(channel, km.labels_), 3))   # 0.514  vs the held-out channel label
prof = spend.copy(); prof["segment"] = km.labels_
print(prof["segment"].value_counts().sort_index().tolist())
# [252, 188]
print(prof.groupby("segment").median().round(0).astype(int))
# segment 0 (n=252): Fresh 10348  Milk 1871  Grocery  2406  Frozen 2225  Deterg  328  Delic  750
# segment 1 (n=188): Fresh  5406  Milk 7690  Grocery 11527  Frozen 1062  Deterg 4495  Delic 1388
lab2 = km.labels_
lab3 = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Xs)
print(sorted(int((lab3 == j).sum()) for j in range(3)))
fresh_id = int(np.argmax([np.median(raw[lab2 == j, 0]) for j in range(2)]))
for j in range(3):
    m = lab3 == j
    med = np.median(raw[m], axis=0).round(0).astype(int)
    print(j, int(m.sum()), int(((lab2 == fresh_id) & m).sum()), med.tolist())
# [81, 147, 212]
# 0 81 9 [1454, 6264, 10487, 388, 4196, 436]
# 1 147 32 [12126, 7184, 9965, 2005, 3378, 2005]
# 2 212 211 [9635, 1598, 2151, 2151, 274, 686]

# %% [markdown]

# The silhouette is highest at two clusters (0.290), drops to 0.259 at three, then reads 0.188, 0.195, 0.193, and 0.196 from four through seven, so it flattens near 0.19 after the drop. Two is still the number the data supports. That is a real result, not the tidy four-way split a toy dataset would hand you: the strongest structure in wholesale spending is a single split into two kinds of buyer. The recovered pair lines up with the dataset's held-out sales channel at an adjusted Rand index of 0.514, which means k-means rediscovered most of the Horeca-versus-Retail distinction from spending patterns alone, without ever seeing the channel label. But the clustering is only half the job; the value comes from profiling each segment by its typical spend and naming it, as in Table 8-5.
#
# Table 8-5. Wholesale-customer segments from k-means, profiled by median annual spend per category.
#
# | Segment | Size | Fresh | Milk | Grocery | Frozen | Detergents/Paper | Delicatessen | Persona |
# | --- | --- | --- | --- | --- | --- | --- | --- | --- |
# | 0 | 252 | 10,348 | 1,871 | 2,406 | 2,225 | 328 | 750 | fresh-and-frozen buyers (hotels, restaurants, cafes) |
# | 1 | 188 | 5,406 | 7,690 | 11,527 | 1,062 | 4,495 | 1,388 | grocery-and-detergents buyers (retailers) |
#
# Each row is a persona a business can act on. Segment 0 pours its money into fresh produce and frozen goods but buys little grocery, detergents, or paper, the buying pattern of hotels, restaurants, and cafes that cook from raw ingredients. Segment 1's largest medians are grocery at 11,527 and milk at 7,690. Fresh at 5,406 is still above detergents and paper at 4,495. The pattern is still a retail restocking basket, with grocery and milk in front. This profiling step, translating cluster medians into plain-language descriptions, is what makes segmentation useful, and it is where the practitioner's judgment matters more than the algorithm. Forcing $k=3$ (silhouette 0.259) does not split the fresh buyers. The prints show sizes 81, 147, and 212. Of the 252 fresh-and-frozen clients, 211 stay in the cluster of 212. The grocery side splits. The cluster of 147 has medians Fresh 12,126, Milk 7,184, and Grocery 9,965, and $147 - 32 = 115$ of its clients come from the two-cluster grocery segment. The cluster of 81 has medians Fresh 1,454 and Grocery 10,487, grocery-heavy accounts with little fresh spend.
#
# ##### 8.7.2 Part B: visualizing the digits
#
# The second task takes the 64-dimensional handwritten digits and projects them to two dimensions four ways, from linear to nonlinear, scoring each by trustworthiness, how well it preserves each point's true neighbors.
#
# Example 8-26. Four two-dimensional embeddings of the digits, scored by trustworthiness.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, trustworthiness
import umap
rng = np.random.default_rng(0)
Xd, yd = load_digits(return_X_y=True)
idx = rng.choice(len(Xd), 1000, replace=False); Xd, yd = Xd[idx], yd[idx]
emb = {"PCA": PCA(2).fit_transform(Xd),
       "Kernel PCA": KernelPCA(n_components=2, kernel="rbf", gamma=0.001).fit_transform(Xd),
       "t-SNE": TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(Xd),
       "UMAP": umap.UMAP(n_components=2, random_state=0).fit_transform(Xd)}
for name, e in emb.items():
    print(name, round(trustworthiness(Xd, e, n_neighbors=10), 3))
# PCA 0.825   Kernel PCA 0.82   t-SNE 0.991   UMAP 0.987   (t-SNE and UMAP best preserve neighbors)

# %% [markdown]

# Run on its own, as in the lab notebook, Example 8-26 prints a `UserWarning` from umap-learn before its four scores: `n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.` It is a notice, not an error. Because the block fixes the seed with `random_state=0`, umap-learn 0.5.12 resets `n_jobs` from its default of -1 (all cores) to 1, a single job, and the message prints the value after that reset, which is why it names 1 twice. The UMAP block of 8.3.2 hides notices like this one with `warnings.filterwarnings("ignore")`. The progression is clear in Figure 8-7: linear PCA and kernel PCA, both near 0.82 (0.825 and 0.820), smear the ten digit classes together because no linear projection and no single fixed kernel untangles them, while t-SNE and UMAP, at 0.991 and 0.987 (umap-learn 0.5.12 in the reference environment), place each digit in its own well-separated island. PCA, t-SNE, and UMAP repeat their Table 8-2 scores, because Example 8-26 draws the same 1,000 digits with the same seed and the same settings as 8.3; the lab's new entry is kernel PCA. This is the practical recommendation distilled: use PCA for fast linear compression and as a denoising preprocessor, but reach for t-SNE or UMAP when the goal is to actually see the structure of high-dimensional data. Example 8-26 runs PCA and UMAP separately on the raw 64 pixels. Figures 8-6 and 8-7 are drawn from all 1,797 digits, not from that 1,000-point subset, and the figure's kernel PCA standardizes the pixels and uses gamma 0.015.
#
# ![Four scatter panels of the digits in 2D: PCA and kernel PCA overlap the ten class colors, while t-SNE and UMAP separate them into clean islands.](figures/fig_08_07_four-embeddings-digits.png)
#
# Figure 8-7. The 64-dimensional digits in 2D by four methods: PCA and kernel PCA (top) overlap the classes, while t-SNE and UMAP (bottom) separate all ten cleanly.
#
# Another recipe cuts with PCA first and then maps with UMAP. The next block runs it on the same 1,000 digits for 5, 20, and 40 components and scores every map against the original 64 pixels, so any neighbors that the PCA step disturbs count against it.

# %%
import warnings
warnings.filterwarnings("ignore")                # silence numba/UMAP notices, as in 8.3.2
import numpy as np
import umap
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
rng = np.random.default_rng(0)
Xd, yd = load_digits(return_X_y=True)
idx = rng.choice(len(Xd), 1000, replace=False); Xd, yd = Xd[idx], yd[idx]   # Example 8-26's subset
direct = umap.UMAP(n_components=2, random_state=0).fit_transform(Xd)
print("raw 64", round(trustworthiness(Xd, direct, n_neighbors=10), 3))
for m in (5, 20, 40):
    pca = PCA(n_components=m, random_state=0).fit(Xd)                     # cut to m components first
    emb = umap.UMAP(n_components=2, random_state=0).fit_transform(pca.transform(Xd))
    print(m, round(float(pca.explained_variance_ratio_.sum()), 3),
          round(trustworthiness(Xd, emb, n_neighbors=10), 3))              # scored against all 64 pixels
# raw 64 0.987
# 5 0.544 0.963
# 20 0.894 0.988
# 40 0.988 0.986

# %% [markdown]

# UMAP on the raw 64 pixels scores 0.987. After PCA to 20 components, which keep 0.894 of the variance, the map scores 0.988, and after 40 components (0.988 of the variance) 0.986, both within 0.001 of the raw map. After only 5 components, which keep 0.544 of the variance, the score drops to 0.963, because UMAP then builds its neighbor graph in a space that has discarded $1 - 0.544 = 0.456$ of the variance. On 64 features the PCA step therefore costs nothing in quality once it keeps most of the variance, and it gains nothing either; its use is on much wider data, where it shrinks every distance the neighbor search computes from one term per original feature to one term per component kept. To experiment, swap the 1,000 digits for 5,000 MNIST images from `fetch_openml("mnist_784")` and time UMAP on the 784 raw pixels against UMAP after PCA to 50 components.
#
# ##### 8.7.3 What we built, and a self-check
#
# The two halves of the chapter solved two real problems. Clustering plus profiling turned 440 raw wholesale-customer records into two named personas, fresh-buying caterers and grocery-buying retailers, with the silhouette choosing two segments and the result matching the dataset's held-out sales channel at ARI 0.514, and the interpretation, not the algorithm, was the deliverable. Dimensionality reduction turned unviewable 64-dimensional digits into a 2D map, where t-SNE and UMAP at trustworthiness near 0.99 vastly outperformed linear PCA at 0.82 for revealing structure. Both are unsupervised: no labels guided the segmentation or the embedding, yet both surfaced structure a human can act on, which is the whole promise of unsupervised learning.
#
# Try these before moving on, with full solutions in Appendix H.
#
# 1. On the wholesale customers the silhouette was highest at two clusters. How does the silhouette score choose $k$, and why is its peak, not the largest $k$ you try, the number to take?
# 2. The two clusters were profiled by their median spend per category and named caterers and retailers. Why is this profiling step, rather than the clustering itself, what makes a segmentation useful to a business?
# 3. On the digits, t-SNE scored 0.991 and PCA 0.825 on trustworthiness. What does a trustworthiness of 0.991 tell you about the 2D map, and what would a score near 0.5 mean?
# 4. Why do t-SNE and UMAP separate the ten digit classes when PCA and kernel PCA cannot? What property of the data are the nonlinear methods exploiting that a linear projection ignores?
#
# #### 8.8 Interview questions and coding exercises
#
# Unsupervised learning is tested differently from supervised: interviewers probe whether you understand that there is no accuracy to optimize, that each method encodes an assumption about structure, and that the choice of method and of $k$ is a judgment call. The questions that follow are grouped by topic. Most of them join two or more sections or work from the numbers the chapter printed, so they extend the section self-checks rather than repeat them. Model answers to them, together with solutions to the exercises from each self-check, are collected in Appendix H.
#
# ##### Clustering
#
# 1. Show that neither step of Lloyd's algorithm can raise the inertia $J$, and explain why that guarantees convergence but not the best clustering. What do k-means++ and `n_init` each do about the second problem, judging by the twelve-blob runs of 8.1.1?
# 2. On the penguins the inertia elbow pointed to three clusters, the silhouette peaked at two, and the adjusted Rand index peaked at three, while on the four generated blobs of 8.1.6 all three internal metrics agreed on four. Explain both results, and say which of these signals you could still use on data with no labels.
# 3. A dataset holds two tight clusters close together, one diffuse cluster, and some scattered junk points. Compare k-means, DBSCAN, and HDBSCAN on it, using what 8.1.3 printed for DBSCAN at two radii.
# 4. Match the E-step and M-step of EM for a Gaussian mixture to the two steps of Lloyd's algorithm. What changes when the assignments become soft, and which quantity is each algorithm guaranteed never to make worse?
# 5. Write the BIC and the AIC, count the free parameters of a two-component mixture with full covariances in two dimensions, and say why BIC penalizes extra parameters harder than AIC on the 600 points of 8.1.4.
# 6. Spectral clustering, kernel PCA, and Isomap each build an $n \times n$ matrix from neighbor or kernel similarities. What does each do with its matrix, why can all three handle curved shapes that k-means and linear PCA cannot, and what cost do they share?
#
# ##### Dimensionality reduction
#
# 1. PCA's reconstruction error on the digits at 32 components is 0.632 per pixel. Show that it equals the sum of the dropped eigenvalues divided by the 64 pixels, and explain why that makes a target on explained variance and a budget on reconstruction error the same rule.
# 2. What is the relationship between PCA and the SVD?
# 3. You will train a classifier on the digits and want to cut the 64 pixels to a few features first. Compare PCA and LDA for that job with the chapter's numbers, and say what you would use instead of LDA if the target were a continuous quantity rather than a class.
# 4. When would you use kernel PCA or t-SNE instead of PCA?
# 5. Table 8-2 scores six embeddings by trustworthiness and by 2D kNN accuracy. What does each measure, why do they rank the methods almost the same, and which one could you compute on data with no labels?
# 6. A gene-expression study has 20,000 features and 200 samples. Explain geometrically why ordinary least squares breaks down here, then compare three fixes, ridge, lasso, and PCA followed by regression, saying what each assumes about the data and when each would be the wrong choice.
#
# ##### Other unsupervised methods
#
# 1. On the shuttle data the robust Gaussian led on ROC AUC, 0.951 against Isolation Forest's 0.934, yet among the 244 points each one flagged, Isolation Forest caught 166 true anomalies and the ellipse 139. Which would you use to feed an alert queue that analysts review by hand, and what does the disagreement say about choosing a metric?
# 2. Isolation Forest and DBSCAN can both flag outliers. Compare how each decides that a point is anomalous, and explain why Isolation Forest scored the empty corners between two blobs as more normal than an outside point just as far from the nearest blob.
# 3. On the bakery receipts `Toast -> Coffee` has confidence 0.70 and lift 1.47, while on the eight toy baskets bread to diapers has confidence 0.80 and lift 0.91. Explain how the rule with the higher confidence can be the negative association, and what more you would need before acting on any rule mined from eight baskets.
# 4. LDA gives every newsgroup post a mix of topics, while k-means on the posts' TF-IDF vectors would give every post exactly one cluster. Compare the two ways of organizing the 2,341 posts, saying what LDA's mixture buys and what it costs, and relate the contrast to the GMM and k-means of 8.1.4.
#
# ##### Coding exercises
#
# The interview questions ask whether you can explain unsupervised learning; these exercises ask whether you can run it. Each one takes a method from this chapter to new data or new settings and checks it, either against a trusted library, a known ground-truth label used only for scoring, or a value you can reason about by hand, the same discipline every worked example in the chapter followed. Most of them use real datasets bundled with scikit-learn or seaborn, or the 20 Newsgroups text corpus. Exercises 3 and 6 use `make_moons` and `make_circles` because the shape has to be known, exercise 9 uses ten baskets made up for the exercise so the metrics can be checked by hand, and exercise 11 uses generated signals because the true sources must be known to score the unmixing. Little needs downloading, and each runs in seconds on a CPU. Write each solution yourself, then compare against the worked version and its stated output in Appendix H (the `Solutions.md` companion).
#
# 1. k-means from scratch with a convergence check. On the wine dataset (`load_wine`), standardize the 13 chemical features, set the cultivar labels aside, and implement Lloyd's algorithm yourself: alternate the assignment step (each point to its nearest centroid) and the update step (each centroid to the mean of its points), and stop when no centroid moves farther than a tolerance of $10^{-6}$ instead of running a fixed number of iterations. Cluster with $k=3$ from five random starts (seeds 0 to 4), print the number of iterations and the final inertia of each run, keep the run with the lowest inertia, and confirm it matches the inertia of scikit-learn's `KMeans(n_init=10)` to three decimals. Report the adjusted Rand index of the kept run against the cultivars, and say what the runs that stop at a higher inertia show about Lloyd's algorithm.

# %%
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(load_wine().data)
print(X.shape)
def lloyd(X, k, tol=1e-6, seed=0):
    ...
print(lloyd(X[:4], 2))

# %% [markdown]

# The starter prints `(178, 13)` and `None`.
#
#
# 2. Choosing $k$ with internal and external metrics. On the wine dataset (`load_wine`), standardize the 13 chemical features and cluster with `KMeans` for $k$ from 2 to 6. For each $k$ print the silhouette score, the Davies-Bouldin index, the Calinski-Harabasz index, and the adjusted Rand index against the true cultivar labels. Report which $k$ each metric favors, and confirm that the three internal metrics and the external one all agree here.

# %%
from sklearn.datasets import load_wine
wine = load_wine()
print(wine.data.shape)
def scores(X, y, k):
    ...
print(scores(wine.data, wine.target, 3))

# %% [markdown]

# The starter prints `(178, 13)` and `None`.
#
#
# 3. Cluster shapes under noise. Generate two interleaving crescents with `make_moons(n_samples=400, random_state=0)` at three noise levels, 0.06, 0.08, and 0.10, and cluster each five ways at $k=2$ with the settings held fixed: k-means, Ward-linkage agglomerative, single-linkage agglomerative, DBSCAN (`eps=0.2`, `min_samples=5`), and spectral clustering with a 10-nearest-neighbor graph. Score each against the true crescent labels with the adjusted Rand index. The 0.06 level is the setting of 8.1.1 and 8.1.3, so it checks your code against the chapter. Explain from the other two levels why the three methods that follow the crescents at 0.06 break down as the noise grows while k-means hardly changes, and print the cluster sizes to see what single linkage does at 0.10.

# %%
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=400, noise=0.10, random_state=0)
print(X.shape)
def cluster_all(X, y):
    ...
print(cluster_all(X, y))

# %% [markdown]

# The starter prints `(400, 2)` and `None`.
#
#
# 4. Gaussian mixtures and BIC. On the standardized penguins (the four body measurements of 8.1.1), fit a `GaussianMixture` with `covariance_type="full"` for $k$ from 1 to 5 and report the BIC for each, confirming it bottoms out at the true three species. Then, holding $k=3$, sweep the four covariance types (spherical, diag, tied, full) and print the adjusted Rand index of each against the species alongside k-means, and say which shape assumptions recover the species and which do not.

# %%
import seaborn as sns
from sklearn.preprocessing import StandardScaler
pen = sns.load_dataset("penguins").dropna()
X = StandardScaler().fit_transform(pen[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])
print(X.shape)
def bic_sweep(X):
    ...
print(bic_sweep(X))

# %% [markdown]

# The starter prints `(333, 4)` and `None`.
#
#
# 5. PCA three ways. On the digits (`load_digits`), center the 64-pixel data and compute the explained-variance ratio of the top three components three different ways: from the eigenvalues of the covariance matrix, from the singular values of the centered data, and from scikit-learn's `PCA`. Confirm all three agree to four decimals. Then report how many components reach 99 percent cumulative variance, and for $k \in \{4, 12, 48\}$ project down to $k$ components and back, print the reconstruction mean squared error, and check it against the sum of the dropped eigenvalues of the $1/n$ covariance divided by the 64 pixels, the identity 8.2.1 showed at 32 components.

# %%
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
print(X.shape)
def pca_three_ways(X):
    ...
print(pca_three_ways(X))

# %% [markdown]

# The starter prints `(1797, 64)` and `None`.
#
#
# 6. Kernel PCA and the kernel width. Section 8.2.2 unfolded two concentric circles with RBF kernel PCA at $\gamma = 3$ and warned that a width too small or too large fails. Test that warning on the same circles: put a two-component RBF `KernelPCA` and a logistic regression in one pipeline and measure its 5-fold cross-validated accuracy for $\gamma \in \{0.1, 0.3, 1, 3, 10, 30, 100\}$. Report which widths unfold the rings, and explain why both ends fail. For the small end, expand $e^{-\gamma d^2}$, where $d$ is the distance between two points, to first order in $\gamma$ and ask what centering the kernel leaves.

# %%
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=500, factor=0.3, noise=0.05, random_state=0)
print(X.shape)
def cv_accuracy(X, y, gamma):
    ...
print(cv_accuracy(X, y, 3))

# %% [markdown]

# The starter prints `(500, 2)` and `None`.
#
#
# 7. PCA versus LDA for class separation. On the wine data (`load_wine`, three cultivars), reduce the standardized features to $k \in \{1, 2\}$ dimensions with unsupervised `PCA` and with supervised `LinearDiscriminantAnalysis` (which uses the labels), keeping the scaler and the projection inside a pipeline, and compare a logistic regression's 5-fold accuracy on each projection. Report which projection wins at each $k$ and why, then ask LDA for three components and explain the error it raises.

# %%
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)
print(X.shape, len(set(y)))
def compare(X, y, k):
    ...
print(compare(X, y, 1))

# %% [markdown]

# The starter prints `(178, 13) 3` and `None`.
#
#
# 8. Novelty detection on the digits. Treat the images of one digit as the normal class. Train three novelty detectors on a clean sample of that digit only, Isolation Forest, a one-class SVM, and Local Outlier Factor with `novelty=True`, then score a held-out mix of unseen normals and a handful of other-digit images by ROC AUC (the labels used only to score, never to fit). Report the three AUCs and explain why fitting on clean data gives a sharper boundary than pouring everything into one contaminated pile.

# %%
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
print(int((y == 1).sum()))
def novelty(X, y):
    ...
print(novelty(X, y))

# %% [markdown]

# The starter prints `182` and `None`.
#
#
# 9. Association-rule metrics from scratch. The starter holds ten coffee-shop baskets, a list made up for this exercise so that every number can be checked by hand. Implement `support`, `confidence`, and `lift` from their definitions with no library. Compute all three for the rules coffee to cake, coffee to milk, and tea to scone; show that the two coffee rules have the same confidence but different lifts, and that tea to scone has the highest lift. Then run the one-sided Fisher exact test of 8.5 on coffee to cake and on tea to scone. Say in a comment that a lift above 1 on ten baskets describes those baskets and is not by itself a test against chance, and explain in a comment why lift, not confidence, is the metric to rank rules by.

# %%
baskets = [{"coffee", "cake"}, {"coffee", "cake", "milk"}, {"tea", "scone"}, {"tea", "scone", "milk"},
           {"coffee", "milk"}, {"coffee", "cake", "sugar"}, {"tea", "milk", "sugar"},
           {"coffee", "milk", "sugar"}, {"tea", "scone", "cake"}, {"coffee", "cake", "milk"}]
print(len(baskets))
def lift(A, B):
    ...
print(lift({"coffee"}, {"cake"}))

# %% [markdown]

# The starter prints `10` and `None`.
#
#
# 10. Miniproject: topic modeling on four other newsgroups. Fetch the 20 Newsgroups training posts from four subjects that 8.6 did not use (`rec.autos`, `sci.med`, `comp.sys.mac.hardware`, `soc.religion.christian`) with headers, footers, and quotes removed. Build a count matrix with `CountVectorizer` using the settings of 8.6 (`max_df=0.5`, `min_df=10`, English stop words) and fit a 4-topic `LatentDirichletAllocation`, printing each topic's top words. Then fit a 4-topic `NMF` on the TF-IDF version and print its top words too. Compare the two: report which source groups each method recovers, search the posts for a word of any topic made of odd tokens to find where that topic comes from, and say how far the topics are from a one-to-one match with the four groups.

# %%
from sklearn.datasets import fetch_20newsgroups
docs = fetch_20newsgroups(subset="train", categories=["rec.autos"], remove=("headers", "footers", "quotes")).data
print(len(docs))
def topics(docs):
    ...
print(topics(docs[:5]))

# %% [markdown]

# The starter prints `594` and `None`.
#
#
#
# 11. ICA and a Gaussian source. Mix a sine wave and a square wave with a new 2 by 2 matrix, unmix with `FastICA` and with `PCA`, and score each by the mean of the best absolute correlations with the true sources. Then replace both sources with independent Gaussian noise, mix them with the same matrix, and compare the two scores again.

# %%
import numpy as np
from sklearn.decomposition import FastICA
t = np.linspace(0, 8, 2000)
S = np.c_[np.sin(2 * t), np.sign(np.sin(3 * t))]
print(S.shape)
def unmix(X):
    ...
print(unmix(S))

# %% [markdown]

# The starter prints `(2000, 2)` and `None`.
#
# 12. Factor analysis on wine. On the standardized wine features, fit `FactorAnalysis(n_components=2)` and print the three features with the largest `noise_variance_`.

# %%
from sklearn.datasets import load_wine
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
wine = load_wine()
X = StandardScaler().fit_transform(wine.data)
print(X.shape)
def noisiest(X):
    ...
print(noisiest(X))

# %% [markdown]

# The starter prints `(178, 13)` and `None`.
#
# 13. HDBSCAN and DBSCAN on real data. Section 8.1.3 compared the two on generated blobs of different density. Run the comparison on the standardized penguins: `DBSCAN(min_samples=10)` at two radii, $\varepsilon = 0.5$ and $\varepsilon = 0.8$, against `HDBSCAN(min_cluster_size=20, copy=True)`. Report clusters, noise counts, and adjusted Rand index against species, count each species in each HDBSCAN cluster, and explain why HDBSCAN and DBSCAN at the larger radius, which choose the number of clusters themselves, find two groups rather than the three species, and what goes wrong at the smaller radius.

# %%
import seaborn as sns
from sklearn.preprocessing import StandardScaler
pen = sns.load_dataset("penguins").dropna()
X = StandardScaler().fit_transform(pen[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])
print(X.shape)
def compare(X):
    ...
print(compare(X))

# %% [markdown]

# The starter prints `(333, 4)` and `None`.
#
# 14. k-means++ seeding. Implement the D-squared seeding step and compare 30 single-start runs with random seeding on the standardized digits at $k = 10$. Report the mean, the worst, and the standard deviation of the inertia for each method, and explain why the large gap that k-means++ opened on the twelve blobs of 8.1.1 does or does not appear here.

# %%
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(load_digits().data)
print(X.shape)
def seed_once(X, k):
    ...
print(seed_once(X[:20], 3))

# %% [markdown]

# The starter prints `(1797, 64)` and `None`.
#
# 15. Cutting a dendrogram. Section 8.1.2 cut the Ward tree of the wine data, where a cut at three clusters and a cut in the wide gap between 12.6 and 27.7 gave the same partition. Repeat the test on the standardized penguins: build the Ward linkage, print the last five merge heights, and cut the tree twice, with `maxclust` at three clusters and at a height midway across the largest gap among those five heights. Report how many clusters each cut gives, whether the partitions match, and the adjusted Rand index of each against species, and say what the result tells you about the tall-gap rule.

# %%
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
pen = sns.load_dataset("penguins").dropna()
X = StandardScaler().fit_transform(pen[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])
Z = linkage(X, method="ward")
print(Z.shape)
def cut(Z):
    ...
print(cut(Z))

# %% [markdown]

# The starter prints `(332, 4)` and `None`.
#
# 16. Mean-shift bandwidth. On the standardized penguins, estimate the bandwidth at quantiles 0.1, 0.2, and 0.3, run `MeanShift` with `bin_seeding=True` for each, and report the number of clusters and the adjusted Rand index against species.

# %%
import seaborn as sns
from sklearn.cluster import estimate_bandwidth
from sklearn.preprocessing import StandardScaler
pen = sns.load_dataset("penguins").dropna()
X = StandardScaler().fit_transform(pen[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])
print(round(float(estimate_bandwidth(X, quantile=0.2, random_state=0)), 3))
def clusters_at(X, q):
    ...
print(clusters_at(X, 0.2))

# %% [markdown]

# The starter prints `1.369` and `None`.
#
# Each exercise follows the chapter's loop: build the method or metric from its definition, then hold it up against a library or a known label that computes the same thing a different way. When they agree you have both a correct implementation and evidence you understood the idea.
#
# ##### Coding drills
#
# The drills are a bank of short single-idea problems grouped by the chapter's topics. Each is a one-liner or a few lines. Write it yourself, predict the exact value it should print, then check both the code and the output against the worked solutions in Appendix H (the `Solutions.md` companion). The habit to build is reaching for the right scikit-learn call for each unsupervised task and always confirming the result against something you can reason about, a known label, a shape, or a hand count.
#
# Clustering: fit `KMeans(3)` on the standardized iris features and print its inertia; print the silhouette score of that clustering; add 30 scattered points to three blobs, run `DBSCAN`, and count how many points get the noise label $-1$; cluster the iris features with Ward-linkage `AgglomerativeClustering` and print the adjusted Rand index against the species; print the same score for the k-means labels; fit a 3-component `GaussianMixture` and print `predict_proba` for the first point to see its soft membership.
#
# Linear dimensionality reduction: fit `PCA` on the standardized breast cancer measurements (`load_breast_cancer`, 30 features) and print the explained-variance ratio of the top three components; print how many components are needed to reach 90 percent cumulative variance; on the digits, fit `PCA(n_components=5, whiten=True)` and print the per-component variances to confirm they are all 1; transform the digits with `TruncatedSVD(n_components=4)` and print the output shape; fit `LinearDiscriminantAnalysis` on the digits and print the number of components it produced, confirming it equals classes minus one.
#
# Manifold learning: embed a 400-image digit subset to 2D with `TSNE` and print the embedding shape; print its trustworthiness at 10 neighbors; do the same for an `Isomap` embedding; for an `MDS` embedding; and for a plain `PCA(2)` embedding, so the four trustworthiness numbers can be ranked.
#
# Anomaly detection: on one blob polluted with 20 scattered points, run `IsolationForest(contamination=0.1)` and count how many points it flags as $-1$; do the same with `LocalOutlierFactor`; and with `EllipticEnvelope`; on a small 1D array with one obvious spike, compute z-scores and count how many exceed 2 in absolute value.
#
# Association rules and topic modeling: on a tiny list of baskets, compute the support of the pair {a, b} by hand; compute the confidence of a to b; compute the lift of a to b; build a doc-term matrix from four toy sentences with `CountVectorizer` and print its shape; fit a 2-component `NMF` on it and confirm every entry of its components is nonnegative.
#
# With unsupervised learning in hand, Chapter 9 turns to making uncertainty itself a first-class citizen.

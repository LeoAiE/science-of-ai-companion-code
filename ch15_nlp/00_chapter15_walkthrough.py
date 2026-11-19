"""Chapter 15: Natural Language Processing

Section-aligned walkthrough: every code block from this chapter of the book,
in book order. Generated from the chapter slice.
"""

# %% [markdown]
# ## 15.1 The classical NLP pipeline

# %%
text = "The CEOs were running quickly to the important meetings."
tokens = "".join(c for c in text.lower() if c.isalpha() or c == " ").split()   # lowercase, keep letters
stop = {"the", "to", "were", "a", "an", "of", "and", "in", "is"}
no_stop = [w for w in tokens if w not in stop]
def stem(w):                                          # crude suffix stripping (the Porter-stemmer idea)
    for suf in ("ing", "ly", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w
print("tokens   :", tokens)
print("no stop  :", no_stop)
print("stemmed  :", [stem(w) for w in no_stop])
# tokens drop the period; stop words removed; stemmed: quickly->quick, ceos->ceo (but running->runn shows stemming is crude)

# %%
import spacy                                          # the standard industrial NLP library
nlp = spacy.load("en_core_web_sm")                    # small English pipeline: tagger, parser, lemmatizer
doc = nlp("The CEOs were running quickly to the important meetings.")
print(f"{'token':10} {'lemma':9} {'POS':6} dependency -> head")
for t in doc:
    if not t.is_punct:
        print(f"{t.text:10} {t.lemma_:9} {t.pos_:6} {t.dep_:9} -> {t.head.text}")

# %%
import numpy as np
labels = ["O", "B-PER", "I-PER"]
E = np.array([[3.0, 0.0, 0.0],   # the   -> O          (emission scores: what a local classifier believes)
              [3.0, 0.0, 0.0],   # ceo   -> O
              [0.0, 2.0, 2.5],   # tim   -> leans I-PER over B-PER
              [0.0, 1.0, 3.0],   # cook  -> I-PER
              [3.0, 0.0, 0.0]])  # spoke -> O
print("greedy per-token tags :", [labels[i] for i in E.argmax(1)])   # ['O','O','I-PER','I-PER','O']: I-PER after O is ILLEGAL
NEG = -1e9
T = np.zeros((3, 3)); T[0, 2] = NEG                   # forbid the O -> I-PER transition
start = np.array([0.0, 0.0, NEG])                     # cannot start with I-PER
def viterbi(E, T, start):                             # decode the best LEGAL whole-sequence labeling
    n, k = E.shape; dp = start + E[0]; back = np.zeros((n, k), int)
    for t in range(1, n):
        scores = dp[:, None] + T; back[t] = scores.argmax(0); dp = scores.max(0) + E[t]
    path = [int(dp.argmax())]
    for t in range(n - 1, 0, -1): path.append(int(back[t, path[-1]]))
    return [labels[i] for i in path[::-1]]
print("CRF (Viterbi) tags    :", viterbi(E, T, start))   # ['O','O','B-PER','I-PER','O']: a valid, correct entity span
def path_score(seq):                                  # start + emissions + transitions, the exponent in p(y|x)
    s = start[seq[0]] + E[0, seq[0]]
    for t in range(1, len(seq)):
        s += T[seq[t - 1], seq[t]] + E[t, seq[t]]
    return s
win = (0, 0, 1, 2, 0)                                 # O, O, B-PER, I-PER, O
Z, n_legal = 0.0, 0
for seq in np.ndindex(3, 3, 3, 3, 3):                 # every labeling, 3^5 = 243
    s = path_score(seq)
    if s > -1e8:                                      # the forbidden transitions are -1e9 and add nothing to Z
        Z += np.exp(s); n_legal += 1
print(f"legal paths {n_legal} of {3**5}, Z={Z:.0f}, p(win)={np.exp(path_score(win)) / Z:.3f}")

# %% [markdown]
# ## 15.1.1 Edit distance

# %%
def edit_distance(a, b):
    prev = list(range(len(b) + 1))               # D[0][j] = j
    for i, ca in enumerate(a, 1):
        cur = [i]                                 # D[i][0] = i
        for j, cb in enumerate(b, 1):
            sub = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + sub))
        prev = cur
    return prev[-1]
print(edit_distance("kitten", "sitting"))        # 3: k->s, e->i, insert g

# %% [markdown]
# ## 15.1.2 Text before tokens

# %%
import unicodedata, html, re
a, b = "caf\u00e9", "cafe\u0301"                 # one code point, or e plus a combining accent
print(a == b, len(a), len(b))
print(unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b),
      len(unicodedata.normalize("NFC", b)))
raw = "A great film.<br />Not slow."
clean = re.sub(r"<[^>]+>", " ", html.unescape(raw))
print(clean)

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import html, re, unicodedata
from collections import Counter
from datasets import load_dataset
reviews = load_dataset("stanfordnlp/imdb", split="train")["text"]    # 25,000 labeled reviews
TAG = re.compile(r"</?[A-Za-z][^<>]*>")              # a tag opens with a letter: <br />, <i>, </i>
def clean(t):
    t = TAG.sub(" ", t)                               # a tag becomes a space, never glue
    t = html.unescape(t)                              # then entities: &#8230; becomes an ellipsis
    return unicodedata.normalize("NFC", t)
crude = re.compile(r"<[^>]+>")                        # any bracket up to the next >
print(f"{len(reviews)} reviews: {sum('<br />' in r for r in reviews)} with <br />, "
      f"{sum(bool(re.search(r'&(#[0-9]+|[A-Za-z]+);', r)) for r in reviews)} with an entity, "
      f"{sum(not unicodedata.is_normalized('NFC', r) for r in reviews)} not NFC")
print("changed by clean:", sum(clean(r) != r for r in reviews),
      " crude pattern differs:", sum(crude.sub(" ", r) != TAG.sub(" ", r) for r in reviews))
raw = Counter(w for r in reviews for w in r.lower().split())
cleaned = Counter(w for r in reviews for w in clean(r).lower().split())
glued = sorted((w for w in raw if "<br" in w or "/>" in w), key=lambda w: -raw[w])
print(f"whitespace types: raw {len(raw)}, cleaned {len(cleaned)}, raw types holding markup {len(glued)}",
      glued[:4])

# %%
import spacy
text = "Dr. Smith paid 3.5 dollars in the U.S. on Friday. It was worth it."
print("split on '. ':", text.split(". "))
print("split on '.'  :", [s for s in text.split(".") if s.strip()])
parser = spacy.load("en_core_web_sm")                 # sentence ends come from the dependency parse
rules = spacy.blank("en"); rules.add_pipe("sentencizer")   # punctuation rules over spaCy's tokens
print("parser       :", [s.text for s in parser(text).sents])
print("sentencizer  :", [s.text for s in rules(text).sents])
for i, r in enumerate(clean(r) for r in reviews[:5]):
    print(f"review {i}: split {len([s for s in r.split('. ') if s.strip()])}, "
          f"parser {len(list(parser(r).sents))}, sentencizer {len(list(rules(r).sents))}")

# %% [markdown]
# ## 15.1.3 What we built, and a self-check

# %% [markdown]
# ## 15.2 Text representation: from counts to embeddings

# %%
import numpy as np
docs = ["the cat sat on the mat", "the dog sat on the log", "the cats and dogs are pets"]
vocab = sorted(set(w for d in docs for w in d.split()))
tf = np.array([[d.split().count(w) for w in vocab] for d in docs], float)
df = (tf > 0).sum(0)                                  # in how many documents each word appears
idf = np.log(len(docs) / df)                          # rare words get a high idf, ubiquitous ones ~0
tfidf = tf * idf
print(f"'the' in {int(df[vocab.index('the')])}/3 docs -> idf = {idf[vocab.index('the')]:.2f} -> TF-IDF weight 0 (uninformative)")
print(f"'cat' in {int(df[vocab.index('cat')])}/3 docs -> idf = {idf[vocab.index('cat')]:.2f} -> kept (informative)")
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(norm=None, smooth_idf=True).fit(docs)   # smoothed idf = ln((1+N)/(1+df)) + 1
idf_s = dict(zip(vec.get_feature_names_out(), vec.idf_))
print(f"smoothed idf('the')={idf_s['the']:.3f}, smoothed idf('cat')={idf_s['cat']:.3f}")
# plain idf: 'the' = log(3/3) = 0.00, 'cat' = log(3/1) = 1.10; smoothed: 1.000 and 1.693

# %%
import numpy as np
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
texts = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes")).data[:3000]
counts = Counter(w for t in texts for w in t.lower().split() if w.isalpha())
ranked = [c for _, c in counts.most_common()]
print("Zipf's law (freq * rank should be roughly constant):")
for r in [1, 10, 100, 1000]:
    print(f"  rank {r:>4}: freq {ranked[r-1]:>6}, freq*rank = {ranked[r-1]*r:,}")
ranks = np.arange(1, 2001); freqs = np.array(ranked[:2000])
print(f"  log-log slope = {np.polyfit(np.log(ranks), np.log(freqs), 1)[0]:.2f}  (Zipf predicts about -1)")
# freq*rank stays near 26k-49k across ranks; the log-log slope is -1.02, the power law Zipf predicts

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
texts = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes")).data[:3000]
cv = CountVectorizer(max_features=600, stop_words="english", min_df=10)
X = cv.fit_transform(texts); words = cv.get_feature_names_out()
cooc = (X.T @ X).toarray().astype(float); np.fill_diagonal(cooc, 0)   # word-word cooccurrence
emb = TruncatedSVD(n_components=50, random_state=0).fit_transform(np.log1p(cooc))
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
def neighbors(word, k=5):
    i = list(words).index(word); sims = emb @ emb[i]
    return [words[j] for j in sims.argsort()[::-1][1:k+1]]
for w in ["space", "god"]:
    print(f"nearest neighbors of '{w}': {neighbors(w)}")
# 'space' -> ['nasa','technology','technical','systems','box'];  'god' -> ['claim','christian','bible','course','understand']
# similar-meaning words cluster with NO labels (the distributional hypothesis); word2vec and GloVe learn richer versions

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
from gensim.models import FastText
sentences = [[w for w in t.lower().split() if w.isalpha()] for t in texts]   # the same 3000 posts, tokenized
ft = FastText(sentences, vector_size=50, window=5, min_count=5, seed=0, workers=1, epochs=5)
print("'spacecraft'  in the trained vocabulary:", "spacecraft" in ft.wv.key_to_index)
print("'spacecrafts' in the trained vocabulary:", "spacecrafts" in ft.wv.key_to_index)   # never seen
print(f"cosine(spacecraft, spacecrafts) = {ft.wv.similarity('spacecraft', 'spacecrafts'):.3f}")
print("nearest to the misspelling 'spacecrft':", [w for w, _ in ft.wv.most_similar("spacecrft", topn=3)])

# %% [markdown]
# ## 15.2.1 The analogy that made embeddings famous

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-50")           # 400k words as 50-dim vectors, trained on 6B tokens
def rank_all(pos, neg, k=4):                         # same query most_similar builds, with nothing excluded
    vecs = [glove[w] / np.linalg.norm(glove[w]) for w in pos]
    vecs += [-glove[w] / np.linalg.norm(glove[w]) for w in neg]
    q = np.mean(vecs, axis=0); q = q / np.linalg.norm(q)
    sims = glove.get_normed_vectors() @ q
    idx = np.argsort(-sims)[:k]
    return [(glove.index_to_key[int(i)], float(sims[int(i)])) for i in idx]
for label, pos, neg in [("king - man + woman", ["king", "woman"], ["man"]),
                        ("paris - france + italy", ["paris", "italy"], ["france"])]:
    top = glove.most_similar(positive=pos, negative=neg, topn=3)
    print(f"{label:24} most_similar -> " + ", ".join(f"{w} {s:.3f}" for w, s in top))
    print(f"{'':24} unfiltered   -> " + ", ".join(f"{w} {s:.3f}" for w, s in rank_all(pos, neg)))
wm = glove["woman"] - glove["man"]; qk = glove["queen"] - glove["king"]
off = float(np.dot(wm, qk) / (np.linalg.norm(wm) * np.linalg.norm(qk)))
print(f"offset cosine={off:.3f}, lengths {np.linalg.norm(wm):.2f} and {np.linalg.norm(qk):.2f}")

# %% [markdown]
# ## 15.2.2 How word2vec and GloVe learn their vectors

# %%
import numpy as np
sigmoid = lambda z: 1 / (1 + np.exp(-z))
dot_pos, dot_neg = 2.0, 0.5                           # center.true-context and center.negative-sample dot products
loss = -np.log(sigmoid(dot_pos)) - np.log(sigmoid(-dot_neg))
print(f"sigma(2.0)={sigmoid(2.0):.3f}, sigma(-0.5)={sigmoid(-0.5):.3f}, skip-gram loss={loss:.3f}")
X, xmax, alpha = 30.0, 100.0, 0.75
f = (X / xmax) ** alpha                               # GloVe weighting, capped at 1 for X >= xmax
resid = 1.5 + 0.2 + 0.1 - np.log(X)                   # w_i . w~_j + b_i + b~_j - log X_ij
print(f"f(30)={f:.3f}, residual={resid:.3f}, GloVe term=f*resid^2={f * resid ** 2:.3f}")

# %%
import math
sig = lambda z: 1 / (1 + math.exp(-z))
print(f"PMI toy log(5) = {math.log(5):.3f}")
print(f"sigma(2)-1 = {sig(2.0) - 1:.3f}, sigma(0.5) = {sig(0.5):.3f}")

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
texts = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes")).data[:3000]
cv = CountVectorizer(max_features=600, stop_words="english", min_df=10)
X = cv.fit_transform(texts); words = list(cv.get_feature_names_out())
cooc = (X.T @ X).toarray().astype(float); np.fill_diagonal(cooc, 0)
def ppmi(M, alpha=0.75):                              # max(PMI, 0), context counts raised to alpha
    pwc = M / M.sum()
    pw = M.sum(1, keepdims=True) / M.sum()
    pc = M.sum(0, keepdims=True) ** alpha; pc = pc / pc.sum()
    with np.errstate(divide="ignore"):
        return np.maximum(np.log(pwc / (pw * pc)), 0)  # a zero count gives log 0 = -inf, clipped to 0
def neighbors(E, word, k=5):
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    i = words.index(word); return [words[j] for j in (E @ E[i]).argsort()[::-1][1:k + 1]]
P = ppmi(cooc)
print(f"PPMI keeps {np.mean(P > 0):.3f} of the cells")
for name, M in [("log1p", np.log1p(cooc)), ("PPMI", P)]:
    E = TruncatedSVD(n_components=50, random_state=0).fit_transform(M)
    print(f"{name:5} space -> {neighbors(E, 'space')}  god -> {neighbors(E, 'god')}")

# %%
from collections import Counter
sents = [[w for w in t.lower().split() if w.isalpha()] for t in texts]   # the FastText tokens of 15.2
counts = Counter(w for s in sents for w in s)
vocab = [w for w, c in counts.most_common() if c >= 5]; idx = {w: i for i, w in enumerate(vocab)}
V, d, K, window, epochs, B = len(vocab), 50, 5, 2, 5, 256
pairs = []                                            # (center, context) for each context within 2 words
for s in sents:
    ids = [idx[w] for w in s if w in idx]
    for i, c in enumerate(ids):
        for j in range(max(0, i - window), min(len(ids), i + window + 1)):
            if j != i: pairs.append((c, ids[j]))
pairs = np.array(pairs)
noise = np.array([counts[w] for w in vocab]) ** 0.75; noise /= noise.sum()   # word2vec's 3/4 power
cdf = np.cumsum(noise)
rng = np.random.default_rng(0)
W = (rng.random((V, d)) - 0.5) / d                    # center vectors v, a small random start
C = np.zeros((V, d))                                  # context vectors u, a zero start
sig = lambda z: 1 / (1 + np.exp(-z))
print(f"vocabulary {V}, training pairs {len(pairs)}")
step, total = 0, epochs * len(pairs)
for epoch in range(epochs):
    loss, order = 0.0, rng.permutation(len(pairs))   # shuffle, so one batch is not one document
    for b in range(0, len(pairs), B):
        lr = 0.025 * max(1e-4, 1 - step / total); step += B          # linear decay, as word2vec does
        c_id, o_id = pairs[order[b:b + B]].T
        n_id = np.minimum(np.searchsorted(cdf, rng.random((len(c_id), K))), V - 1)   # K negatives each
        v, u_o, u_n = W[c_id], C[o_id], C[n_id]
        s_o = sig((v * u_o).sum(1)); s_n = sig(np.einsum("bd,bkd->bk", v, u_n))
        loss += -(np.log(s_o).sum() + np.log(1 - s_n).sum())        # J summed over the batch
        g_o, g_n = s_o - 1, s_n                       # the coefficients sigma(u_o.v)-1 and sigma(u_n.v)
        grad_v = g_o[:, None] * u_o + np.einsum("bk,bkd->bd", g_n, u_n)
        np.add.at(C, o_id, -lr * g_o[:, None] * v)    # add.at sums the updates of a repeated word
        np.add.at(C, n_id, -lr * g_n[:, :, None] * v[:, None, :])
        np.add.at(W, c_id, -lr * grad_v)
    print(f"epoch {epoch + 1}: mean loss per pair {loss / len(pairs):.3f}")
def near(E, word, k=5):
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return [vocab[j] for j in (En @ En[idx[word]]).argsort()[::-1][1:k + 1]]
for w in ["space", "god"]:
    print(f"scratch SGNS {w:5} -> {near(W, w)}")

# %%
pc = Counter(map(tuple, pairs.tolist()))              # #(w, c), each pair type's count
n_w = np.bincount(pairs[:, 0], minlength=V)           # #(w), how often w was a center
n_c = np.bincount(pairs[:, 1], minlength=V)           # #(c), how often c was a context
keys = np.array([k for k, n in pc.items() if n >= 10]); n_wc = np.array([pc[tuple(k)] for k in keys])
a, c = keys[:, 0], keys[:, 1]
learned = (W[a] * C[c]).sum(1)                        # the trained dot products w . c
plain = np.log(n_wc * len(pairs) / (n_w[a] * n_c[c])) - np.log(K)    # PMI - log k, equation 7
smooth = np.log(n_wc / (n_w[a] * noise[c])) - np.log(K)               # same, with the 3/4-power p(c)
print(f"{len(keys)} pair types seen at least 10 times")
print(f"corr(w.c, PMI - log k) = {np.corrcoef(plain, learned)[0, 1]:.3f}, "
      f"with the 3/4-power p(c) = {np.corrcoef(smooth, learned)[0, 1]:.3f}")
print(f"means: learned {learned.mean():.3f}, target {smooth.mean():.3f}")

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
from gensim.models import Word2Vec
w2v = Word2Vec(sents, vector_size=50, window=2, min_count=5, sg=1, negative=5,
               seed=0, workers=1, epochs=5)            # sg=1 is skip-gram; one worker keeps the run repeatable
print(f"gensim defaults: alpha {w2v.alpha}, sample {w2v.sample}, shrink_windows {w2v.shrink_windows}")
for w in ["space", "god"]:
    print(f"gensim SGNS  {w:5} -> {[x for x, _ in w2v.wv.most_similar(w, topn=5)]}")

# %% [markdown]
# ## 15.2.3 BM25, the ranking function behind search and RAG

# %%
import numpy as np
def bm25(tf, dlen, avgdl, N, n_q, k=1.5, b=0.75):
    idf = np.log((N - n_q + 0.5) / (n_q + 0.5) + 1)   # smoothed inverse document frequency
    return idf * (tf * (k + 1)) / (tf + k * (1 - b + b * dlen / avgdl))
for tf in [1, 3, 10, 100]:                            # one document of length 100 in a collection averaging 80
    print(f"tf={tf:>3}: BM25={bm25(tf, 100, 80, 10, 1):.3f}   raw tf-idf={tf * np.log(10 / 1):.1f}")

# %% [markdown]
# ## 15.2.4 Bag of n-grams and the hashing trick

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import numpy as np, zlib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
tr = ds["train"].shuffle(seed=0).select(range(800))
te = ds["test"].shuffle(seed=0).select(range(400))
Xtr, ytr = list(tr["text"]), np.array(tr["label"])
Xte, yte = list(te["text"]), np.array(te["label"])
def acc(pipe):
    return float((pipe.fit(Xtr, ytr).predict(Xte) == yte).mean())
uni = acc(make_pipeline(TfidfVectorizer(max_features=5000), LogisticRegression(max_iter=1000)))
bi = acc(make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=5000), LogisticRegression(max_iter=1000)))
hashed = acc(make_pipeline(HashingVectorizer(ngram_range=(1, 2), n_features=2**18),
                           LogisticRegression(max_iter=1000)))
hv = HashingVectorizer(ngram_range=(1, 2), n_features=2**18)
sgd = SGDClassifier(loss="log_loss", random_state=0)
for i in range(0, len(Xtr), 200):
    sgd.partial_fit(hv.transform(Xtr[i:i+200]), ytr[i:i+200], classes=np.array([0, 1]))
n = len(CountVectorizer(ngram_range=(1, 2)).fit(Xtr).vocabulary_)
D = 2 ** 18
print(f"unigram TF-IDF {uni:.4f}  bigram TF-IDF {bi:.4f}  hashed logistic {hashed:.4f}")
print(f"hashed partial_fit {float((sgd.predict(hv.transform(Xte)) == yte).mean()):.4f}")
print(f"bigram features {n}, expected colliding pairs {n * (n - 1) / (2 * D):.1f}")
print("crc32 buckets", [zlib.crc32(s.encode()) % 8 for s in ("not", "good", "not good")])

# %% [markdown]
# ## 15.2.5 What we built, and a self-check

# %% [markdown]
# ## 15.3 Modern tokenization

# %%
from collections import Counter
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3, "newer": 2}   # word -> frequency
vocab = {" ".join(w) + " </w>": f for w, f in corpus.items()}           # split each word into chars + end marker
def get_pairs(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        syms = word.split()
        for i in range(len(syms) - 1): pairs[(syms[i], syms[i+1])] += freq
    return pairs
def merge(pair, vocab):                          # merge only a whole adjacent pair, never across a symbol boundary
    out = {}
    for word, freq in vocab.items():
        syms, merged, i = word.split(), [], 0
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == pair[0] and syms[i + 1] == pair[1]:
                merged.append(pair[0] + pair[1]); i += 2
            else:
                merged.append(syms[i]); i += 1
        out[" ".join(merged)] = freq
    return out
merges = []
for _ in range(10):                                   # repeatedly merge the most frequent adjacent pair
    pairs = get_pairs(vocab)
    if not pairs: break
    best = pairs.most_common(1)[0][0]; merges.append(best); vocab = merge(best, vocab)
print("learned merges:", ", ".join("+".join(m) for m in merges))
# w+e, s+t, st+</w>, n+e, ne+we, l+o, newe+st</w>, lo+w, low+</w>, r+</w>: frequent endings and whole words become single tokens

# %%
def tokenize(word):
    syms = list(word) + ["</w>"]
    for a, b in merges:                               # apply the learned merges in order
        i = 0
        while i < len(syms) - 1:
            if syms[i] == a and syms[i+1] == b: syms[i:i+2] = [a + b]
            else: i += 1
    return syms
for w in ["newest", "lowest", "coldest"]:
    print(f"{w:8} ({'in corpus' if w in corpus else 'UNSEEN':9}) -> {tokenize(w)}")
# newest -> ['newest</w>'] (one token); lowest -> ['lo','we','st</w>']; coldest -> ['c','o','l','d','e','st</w>']
# unseen words reuse the learned 'st</w>' ending. 'c' is not in the training alphabet; tokenize never checks a vocabulary.

# %% [markdown]
# ## 15.3.1 What real tokenizers produce

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
bert = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")   # WordPiece: continuation pieces get ##
gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")           # byte-level BPE: leading space becomes G-with-stroke
for w in ["tokenization", "unhappiness", "coldest"]:
    print(f"BERT  {w:13} -> {bert.tokenize(w)}")
print("GPT-2 'tokenization is powerful' ->", gpt2.tokenize("tokenization is powerful"))
print("GPT-2 'coldest' ->", gpt2.tokenize("coldest"))
odd = "i love \U0001F999 and \u2603 snow"
print("BERT odd ->", bert.tokenize(odd))
print("GPT-2 round trip", gpt2.decode(gpt2.encode(odd)) == odd)

# %% [markdown]
# ## 15.3.2 Bytes, and training a real tokenizer

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings("ignore")
import json
from transformers import AutoTokenizer
from tokenizers import pre_tokenizers
gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")
bert = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
alphabet = pre_tokenizers.ByteLevel.alphabet()        # one printable stand-in for each of the 256 bytes
print(len(alphabet), pre_tokenizers.ByteLevel(add_prefix_space=False).pre_tokenize_str(" a"))
model = json.loads(gpt2.backend_tokenizer.to_str())["model"]
print(f"GPT-2 vocabulary {len(gpt2)} = 256 bytes + {len(model['merges'])} merges + 1 special {gpt2.eos_token}")
for s in ["café", "東京に行く", "llama \U0001F999"]:
    print(f"{s!r}: {len(s)} chars, {len(s.encode('utf-8'))} bytes | GPT-2 {gpt2.tokenize(s)} | BERT {bert.tokenize(s)}")

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, decoders
raw = load_dataset("Trelis/tiny-shakespeare")         # 1.2 million characters of train, 119,020 of test
train_text = raw["train"]["Text"]
bpe = Tokenizer(models.BPE())
bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe.decoder = decoders.ByteLevel()
bpe.train_from_iterator(train_text, trainers.BpeTrainer(
    vocab_size=5000, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<|endoftext|>"]))
uni = Tokenizer(models.Unigram())
uni.pre_tokenizer = pre_tokenizers.Metaspace()        # SentencePiece's visible space marker, U+2581
uni.decoder = decoders.Metaspace()
uni.train_from_iterator(train_text, trainers.UnigramTrainer(
    vocab_size=5000, special_tokens=["<unk>"], unk_token="<unk>"))
merges = json.loads(bpe.to_str())["model"]["merges"]
print("first BPE merges:", [" ".join(m) if isinstance(m, list) else m for m in merges[:8]])
for w in ["wherefore", "unhappiness", "tokenization"]:
    print(f"{w:12} BPE {bpe.encode(w).tokens}  unigram {uni.encode(w).tokens}  GPT-2 {gpt2.tokenize(w)}")
test = " ".join(" ".join(raw["test"]["Text"]).split()); n_words = len(test.split())   # one space between words
for name, n in [("BPE 5k", len(bpe.encode(test).ids)), ("unigram 5k", len(uni.encode(test).ids)),
                ("GPT-2 50k", len(gpt2(test)["input_ids"]))]:
    print(f"{name:10}: {n} tokens for {n_words} test words, {n / n_words:.3f} tokens per word")
odd = "llama \U0001F999"   # the llama emoji, outside the training text
enc = uni.encode(odd)
print("BPE round trip:", bpe.decode(bpe.encode(odd).ids) == odd,
      "| unigram:", enc.tokens, "ids", enc.ids, "unk id", uni.token_to_id("<unk>"))

# %% [markdown]
# ## 15.3.3 What we built, and a self-check

# %% [markdown]
# ## 15.4 Contextual embeddings

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased").eval()
def bank_vector(sentence):
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state[0]    # one contextual vector per token
    pos = int((enc["input_ids"][0] == tok.convert_tokens_to_ids("bank")).nonzero()[0, 0])
    return hidden[pos].numpy()
def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
v_r1 = bank_vector("i sat down by the river bank to relax")
v_r2 = bank_vector("the boat drifted slowly near the muddy river bank")
v_money = bank_vector("i deposited my paycheck at the bank this morning")
print(f"river bank vs river bank (same sense)      = {cos(v_r1, v_r2):.3f}")     # 0.802 (high)
print(f"river bank vs money bank (different sense) = {cos(v_r1, v_money):.3f}")   # 0.703 (lower): context changes the vector

# %%
emb = model.get_input_embeddings().weight               # the model's own STATIC token-embedding table
static_bank = emb[tok.convert_tokens_to_ids("bank")].detach().numpy()   # one fixed row, whatever the sentence
print(f"static embedding cosine(river bank, money bank) = {cos(static_bank, static_bank):.3f}")   # 1.000 by construction
print(f"contextual cosine                               = {cos(v_r1, v_money):.3f}")   # 0.703: distinguishes the senses
def embed_bank(sentence):
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        e = model.embeddings(input_ids=enc["input_ids"])[0]
    pos = int((enc["input_ids"][0] == tok.convert_tokens_to_ids("bank")).nonzero()[0, 0])
    return e[pos].numpy(), pos
e1, p1 = embed_bank("i sat down by the river bank to relax")
e2, p2 = embed_bank("the boat drifted slowly near the muddy river bank")
e3, p3 = embed_bank("i deposited my paycheck at the bank this morning")
print(f"positions {p1}, {p2}, {p3}")
print(f"embedding-module cosine, positions {p1} and {p3} = {cos(e1, e3):.3f}")
print(f"embedding-module cosine, both at {p3}           = {cos(e2, e3):.3f}")
# the table row is identical (cosine 1.000). Position embeddings make different positions 0.961 and the same position 1.000. Sense is not in that layer.

# %% [markdown]
# ## 15.4.1 One encoder, many languages

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
xnli = load_dataset("facebook/xnli", "all_languages", split="test")     # 5,010 pairs in 15 languages
seen, rows = set(), []
for p in xnli["premise"]:                             # each premise repeats for three hypotheses
    if p["en"] not in seen: seen.add(p["en"]); rows.append(p)
rows = rows[:300]
langs = ["en", "de", "fr", "ru", "hi", "zh", "ar", "sw"]
toks = {n: AutoTokenizer.from_pretrained(m) for n, m in [("GPT-2", "openai-community/gpt2"),
        ("BERT", "google-bert/bert-base-uncased"), ("XLM-R", "FacebookAI/xlm-roberta-base")]}
count = lambda t, s: len(t(s, add_special_tokens=False)["input_ids"])
base = {n: np.mean([count(t, r["en"]) for r in rows]) for n, t in toks.items()}
print("tokens per sentence on 300 parallel XNLI premises (ratio to English); BERT [UNK] share")
for lang in langs:
    texts = [r[lang] for r in rows]
    cells = [f"{n} {np.mean([count(t, s) for s in texts]):5.1f} "
             f"({np.mean([count(t, s) for s in texts]) / base[n]:.2f})" for n, t in toks.items()]
    ids = np.array([i for s in texts for i in toks["BERT"](s, add_special_tokens=False)["input_ids"]])
    print(f"{lang}: " + "  ".join(cells) + f"  [UNK] {np.mean(ids == toks['BERT'].unk_token_id):.3f}")

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
fill = pipeline("fill-mask", model="FacebookAI/xlm-roberta-base", device="cpu")
for s in ["The capital of France is <mask>.", "La capitale de la France est <mask>.",
          "Die Hauptstadt von Frankreich ist <mask>."]:
    print(f"{s:42} -> {[(r['token_str'], round(r['score'], 3)) for r in fill(s, top_k=3)]}")

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import torch
torch.set_num_threads(2)
from sentence_transformers import SentenceTransformer
multi = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
english = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")   # its model card: language en
en = [r["en"] for r in rows]
E = {name: m.encode(en, normalize_embeddings=True) for name, m in [("multilingual", multi), ("English-only", english)]}
print("top-1 retrieval of the English source among 300, from each translation")
for lang in ["de", "fr", "ru", "hi", "zh", "sw"]:
    out = []
    for name, m in [("multilingual", multi), ("English-only", english)]:
        T = m.encode([r[lang] for r in rows], normalize_embeddings=True)
        out.append(f"{name} {np.mean((T @ E[name].T).argmax(1) == np.arange(len(rows))):.3f}")
    print(f"  {lang}: " + "   ".join(out))

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
from sklearn.linear_model import LogisticRegression
def pair_features(prem, hyp):                         # both vectors, their gap, and their product
    u = multi.encode(prem, normalize_embeddings=True); v = multi.encode(hyp, normalize_embeddings=True)
    return np.hstack([u, v, np.abs(u - v), u * v])
dev = load_dataset("facebook/xnli", "en", split="validation")          # 2,490 English pairs with labels
clf = LogisticRegression(max_iter=2000).fit(pair_features(dev["premise"], dev["hypothesis"]), dev["label"])
test = xnli.select(range(1000))
print("NLI accuracy on 1,000 XNLI test pairs, classifier trained on English only (chance 0.333)")
for lang in ["en", "de", "fr", "hi", "zh", "sw"]:
    prem = [p[lang] for p in test["premise"]]
    hyp = [h["translation"][h["language"].index(lang)] for h in test["hypothesis"]]
    print(f"  {lang}: {np.mean(clf.predict(pair_features(prem, hyp)) == np.array(test['label'])):.3f}")

# %% [markdown]
# ## 15.4.2 What we built, and a self-check

# %% [markdown]
# ## 15.5 Classic NLP tasks and evaluating generated text

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups(subset="train", categories=["rec.sport.hockey", "sci.med"],
                          remove=("headers", "footers", "quotes"))
clf = make_pipeline(TfidfVectorizer(stop_words="english", max_features=5000), LogisticRegression(max_iter=1000))
print(f"CV accuracy = {cross_val_score(clf, data.data, data.target, cv=5).mean():.3f}")   # 0.961: TF-IDF + logistic is a strong, cheap text baseline

# %%
import numpy as np
from collections import Counter
def ngrams(toks, n): return Counter(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
def bleu2(cand, ref, max_n=2):                        # BLEU with n up to 2; standard BLEU-4 uses n up to 4
    c, r = cand.split(), ref.split(); ps = []
    for n in range(1, max_n + 1):
        cg, rg = ngrams(c, n), ngrams(r, n)
        if not cg: return 0.0
        ps.append(sum(min(cg[g], rg[g]) for g in cg) / sum(cg.values()))   # clipped n-gram precision
    if min(ps) == 0: return 0.0
    bp = min(1.0, np.exp(1 - len(r) / max(len(c), 1)))      # brevity penalty
    return bp * np.exp(np.mean(np.log(ps)))                 # geometric mean of the precisions
def rouge1(cand, ref):                                # ROUGE-1: unigram RECALL against the reference
    cg, rg = ngrams(cand.split(), 1), ngrams(ref.split(), 1)
    return sum(min(cg[g], rg[g]) for g in rg) / sum(rg.values())
ref = "the cat is sitting on the mat"
cands = {"near-copy": "the cat is resting on the mat", "paraphrase": "the feline rests upon the rug",
         "unrelated": "a dog ran across the park today"}
for name, cand in cands.items():
    print(f"  {name:10}: BLEU-2={bleu2(cand, ref):.3f}  ROUGE-1={rouge1(cand, ref):.3f}")

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import torch
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased").eval()
def sent_vec(s):
    with torch.no_grad():
        return model(**tok(s, return_tensors="pt")).last_hidden_state[0].mean(0).numpy()   # mean-pooled
def token_vecs(s):                                    # one contextual vector per real token (no [CLS]/[SEP])
    enc = tok(s, return_tensors="pt")
    with torch.no_grad():
        h = model(**enc).last_hidden_state[0]
    return torch.nn.functional.normalize(h[1:-1], dim=1)
def bertscore_f1(cand, ref):                          # real BERTScore: greedy token matching BOTH ways
    C, R = token_vecs(cand), token_vecs(ref)
    sim = C @ R.T                                     # cosine matrix: candidate tokens x reference tokens
    p = sim.max(dim=1).values.mean().item()           # precision: each candidate token's best match
    r = sim.max(dim=0).values.mean().item()           # recall: each reference token's best match
    return 2 * p * r / (p + r)
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
vr = sent_vec(ref)
for name, cand in cands.items():
    print(f"  {name:10}: mean-pool={cos(vr, sent_vec(cand)):.3f}  BERTScore-F1={bertscore_f1(cand, ref):.3f}")

# %%
from sacrebleu.metrics import BLEU                    # sacreBLEU (Post, 2018): BLEU with its settings fixed and printed
from rouge_score import rouge_scorer                  # Google's Python reimplementation of the ROUGE-1.5.5 Perl package
from bert_score import score as bert_score            # the BERTScore authors' own package
bleu = BLEU(effective_order=True)                     # BLEU-4, 13a tokenizer, exp smoothing, 0 to 100 scale
toy = BLEU(max_ngram_order=2, smooth_method="none", tokenize="none", effective_order=True)   # Example 15-11's settings
rouge = rouge_scorer.RougeScorer(["rouge1"])          # no stemming, like rouge1 of Example 15-11
for name, cand in cands.items():
    b, t = bleu.sentence_score(cand, [ref]), toy.sentence_score(cand, [ref])   # one sentence, not a test set
    r = rouge.score(ref, cand)["rouge1"]              # reference (the target) first, then the candidate
    print(f"  {name:10}: sacreBLEU={b.score:4.1f}  toy settings={t.score:4.1f}  "
          f"ROUGE-1 R={r.recall:.3f} P={r.precision:.3f} F1={r.fmeasure:.3f}")
print(" ", bleu.get_signature())                      # the settings a paper reports next to its BLEU
period = "the cat is resting on the mat."             # the near-copy with a final period
print(f"  with a period: hand BLEU-2={bleu2(period, ref):.3f}  13a tokens={bleu.tokenizer(period)!r}")
for layers in (6, None):                              # 6: DistilBERT's last layer, as bertscore_f1 used; None: the default
    (P, R, F), h = bert_score(list(cands.values()), [ref] * len(cands), model_type="distilbert-base-uncased",
                              num_layers=layers, return_hash=True)
    print(f"  BERTScore-F1={[round(f, 3) for f in F.tolist()]}  {h}")

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
t5_tok = AutoTokenizer.from_pretrained("google-t5/t5-small")
t5 = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small").eval()
article = ("The city council met on Tuesday to debate the new bicycle lanes. Supporters argued the lanes "
           "would cut traffic and improve safety, while opponents worried about losing parking spaces. "
           "After two hours of discussion, the council voted seven to two to approve the plan, with "
           "construction scheduled to begin next spring.")
reference = "the council approved the new bicycle lanes after a long debate"
ids = t5_tok("summarize: " + article, return_tensors="pt").input_ids   # T5 takes its task as a text prefix
with torch.no_grad():
    out_ids = t5.generate(ids, max_new_tokens=60, num_beams=4)         # beam search (15.5.1); 60 leaves room to stop
    cut_ids = t5.generate(ids, max_new_tokens=25, num_beams=4)         # a 25-token cap, too short for this summary
summary = t5_tok.decode(out_ids[0], skip_special_tokens=True).lower().strip()
cut = t5_tok.decode(cut_ids[0], skip_special_tokens=True).lower().strip()
print("T5 summary:", summary)
print(f"BLEU-2={bleu2(summary, reference):.3f}  ROUGE-1={rouge1(summary, reference):.3f}  "
      f"BERTScore-F1={bertscore_f1(summary, reference):.3f}")
for name, o in (("room for 60", out_ids), ("cap of 25", cut_ids)):     # position 0 holds the decoder start token
    print(f"{name}: {o.shape[1] - 1} new tokens, ends with the end token: {int(o[0, -1]) == t5_tok.eos_token_id}")
print(f"capped: {cut}")
print(f"capped BLEU-2={bleu2(cut, reference):.3f}")

# %% [markdown]
# ## 15.5.1 Decoding strategies: how a model turns probabilities into text

# %%
probs = [0.50, 0.20, 0.15, 0.10, 0.05]
total, kept = 0.0, 0
for x in probs:
    total += x
    kept += 1
    if total >= 0.9:
        break
print(kept, round(total, 2))                 # 4 tokens, mass 0.95

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import torch
from transformers import pipeline, logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
gen = pipeline("text-generation", model="openai-community/gpt2", device="cpu")
prompt = "The future of AI is"
def decode(name, **kw):
    torch.manual_seed(2)                       # make a sampled run reproducible here
    text = gen(prompt, max_new_tokens=30, pad_token_id=50256, **kw)[0]["generated_text"]
    print(f"[{name}] {text}")
decode("greedy",    do_sample=False)
decode("beam-5",    do_sample=False, num_beams=5)
decode("temp-0.7",  do_sample=True, temperature=0.7, top_k=0, top_p=1.0)
decode("top_k-50",  do_sample=True, top_k=50)
decode("top_p-0.9", do_sample=True, top_p=0.9, top_k=0)

# %% [markdown]
# ## 15.5.2 Perplexity: the intrinsic language-model metric

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()
def perplexity(ids):
    with torch.no_grad():
        ce = model(ids, labels=ids).loss.item()   # mean over the N-1 scored tokens
    return ce, float(torch.exp(torch.tensor(ce)))
for s in ["The cat sat on the mat.", "Colorless green ideas sleep furiously."]:
    ce, ppl = perplexity(tok(s, return_tensors="pt").input_ids)
    print(f"CE={ce:.3f} nats  PPL={ppl:.1f}   {s!r}")
fluent = "the cat sat on the mat"
ids = tok(fluent, return_tensors="pt").input_ids
scr = ids[:, [5, 0, 3, 2, 1, 4]]                 # the same six token ids, order destroyed
ce_f, ppl_f = perplexity(ids)
ce_s, ppl_s = perplexity(scr)
print(f"same tokens, real order  CE={ce_f:.3f} PPL={ppl_f:.1f}")
print(f"same tokens, reordered   CE={ce_s:.3f} PPL={ppl_s:.1f}  {tok.convert_ids_to_tokens(scr[0].tolist())}")
text = "Colorless green ideas sleep furiously."
ids = tok(text, return_tensors="pt").input_ids
with torch.no_grad():
    logp = torch.log_softmax(model(ids).logits[0, :-1], dim=-1)
labels = ids[0, 1:]
pieces = tok.convert_ids_to_tokens(labels.tolist())
nats = [-float(logp[i, labels[i]]) for i in range(len(labels))]
print("Chomsky nats:", ", ".join(f"{a} {b:.3f}" for a, b in zip(pieces, nats)))

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import re, numpy as np
from collections import Counter
from datasets import load_dataset
raw = load_dataset("Trelis/tiny-shakespeare")
lines = [l.strip().lower() for l in "".join(raw["train"]["Text"]).splitlines()]
lines = [l for l in lines if l and not l.endswith(":")]      # keep dialogue, drop speaker tags
train, test = lines[:2000], lines[2000:2200]                 # 2,000 lines to count, 200 held out
toks = lambda l: ["<s>"] + re.findall(r"[a-z']+", l) + ["</s>"]  # <s> starts a line, </s> ends it
uni, bi = Counter(), Counter()
for l in train:
    t = toks(l); uni.update(t); bi.update(zip(t, t[1:]))     # unigram and bigram counts
V = len(uni) + 1                                             # word types + one unknown type
prob = lambda u, w: (bi[(u, w)] + 1) / (uni[u] + V)          # add-one: never zero
def prep(ls, scramble=False):
    rng = np.random.default_rng(0); out = []
    for l in ls:
        t = [w if w in uni else "<unk>" for w in re.findall(r"[a-z']+", l)]
        if scramble: rng.shuffle(t)                          # same words, order destroyed
        out.append(["<s>"] + t + ["</s>"])               # line-final words now have a successor
    return out
def ppl(ls):                                                 # PPL = exp(mean negative log prob)
    lp, n, zeros = 0.0, 0, 0
    for t in ls:
        for u, w in zip(t, t[1:]):
            lp += np.log(prob(u, w)); n += 1
            zeros += (bi[(u, w)] == 0)
    return np.exp(-lp / n), zeros, n
p_real, z, n = ppl(prep(test))
p_scr, _, _ = ppl(prep(test, scramble=True))
tot = sum(uni.values())
u_ce = -np.mean([np.log((uni[w] + 1) / (tot + V)) for t in prep(test) for w in t[1:]])
print(f"V = {V} word types, {tot} training tokens, {n} held-out bigrams")
print(f"C('of','the') = {bi[('of','the')]}, C('of') = {uni['of']} -> p(the|of) = {prob('of','the'):.4f}")
print(f"unseen held-out bigrams: {z} of {n} (unsmoothed model: probability 0, perplexity infinite)")
print(f"add-one bigram PPL: real order = {p_real:.1f}, scrambled = {p_scr:.1f}")
print(f"add-one unigram PPL (order-blind) = {np.exp(u_ce):.1f}")

# %%
train, test = lines[:20000], lines[20000:20200]          # ten times the training lines, the next 200 held out
uni, bi = Counter(), Counter()
for l in train:
    t = toks(l); uni.update(t); bi.update(zip(t, t[1:]))
V = len(uni) + 1                                         # prob() and prep() now read the new counts
p_real20, _, _ = ppl(prep(test))
p_scr20, _, _ = ppl(prep(test, scramble=True))
print(f"V = {V}; add-one bigram PPL: real order = {p_real20:.1f}, scrambled = {p_scr20:.1f}, "
      f"ratio = {p_scr20 / p_real20:.2f} (2,000 lines: {p_scr / p_real:.2f})")

# %% [markdown]
# ## 15.5.3 Neural named entity recognition as token classification

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import pipeline, logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
model_id = "dbmdz/bert-large-cased-finetuned-conll03-english"
sentence = "Hugging Face was founded in New York City by Clement Delangue."
raw = pipeline("token-classification", model=model_id, aggregation_strategy="none", device="cpu")
print("raw:", [(e["word"], e["entity"]) for e in raw(sentence)])
ner = pipeline("token-classification", model=model_id, aggregation_strategy="simple", device="cpu")
for e in ner(sentence):
    print(f'{e["word"]:20} {e["entity_group"]:4} {e["score"]:.3f}')

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import torch, torch.nn.functional as F
tagger, ner_tok = ner.model, ner.tokenizer                     # the pipeline's own model and tokenizer
words = sentence.replace(".", " .").split()                   # 12 words; the period is its own word
enc = ner_tok(words, is_split_into_words=True, return_tensors="pt")
with torch.no_grad():
    out = tagger(**enc, output_hidden_states=True)
h = out.hidden_states[-1][0]                                   # one contextual vector per piece
z = h @ tagger.classifier.weight.T + tagger.classifier.bias    # z_t = W h_t + b, by hand
print("W:", tuple(tagger.classifier.weight.shape), " pieces:", h.shape[0],
      " same as the model's logits:", torch.allclose(z, out.logits[0], atol=1e-4))
names = [tagger.config.id2label[k] for k in range(z.shape[1])]
t = ner_tok.convert_ids_to_tokens(enc["input_ids"][0]).index("Face")
print("Face logits:", ", ".join(f"{n} {v:.2f}" for n, v in zip(names, z[t].tolist())))
print(f"p(I-ORG | Face) = {torch.softmax(z[t], -1)[tagger.config.label2id['I-ORG']]:.4f}")
gold = ["I-ORG", "I-ORG", "O", "O", "O", "I-LOC", "I-LOC", "I-LOC", "O", "I-PER", "I-PER", "O"]
y, prev = [], None
for wid in enc.word_ids():                                     # word index of each piece; None for [CLS], [SEP]
    y.append(-100 if wid is None or wid == prev else tagger.config.label2id[gold[wid]])
    prev = wid
y = torch.tensor(y)
keep = y != -100
by_hand = -torch.log_softmax(z[keep], -1)[torch.arange(int(keep.sum())), y[keep]].mean()
print("labels:", y.tolist())
print(f"scored {int(keep.sum())} of {len(y)} pieces; cross_entropy {F.cross_entropy(z, y):.4f}, "
      f"mean over the scored pieces {by_hand:.4f}")

# %% [markdown]
# ## 15.5.4 Training a tagger: a CRF and a fine-tuned transformer on CoNLL-2003

# %%
def spans(tags):                                      # (start, end, type) for each entity in IOB2 tags
    out, start, kind = [], None, None
    for i, t in enumerate(list(tags) + ["O"]):        # a final O closes an entity that runs to the end
        if start is not None and not (t.startswith("I-") and t[2:] == kind):
            out.append((start, i, kind)); start = None
        if t.startswith("B-") or (t.startswith("I-") and start is None):   # an I- after O also opens one
            start, kind = i, t[2:]
    return out
def entity_f1(gold_seqs, pred_seqs):
    g = {(k,) + s for k, seq in enumerate(gold_seqs) for s in spans(seq)}
    p = {(k,) + s for k, seq in enumerate(pred_seqs) for s in spans(seq)}
    tp = len(g & p); prec = tp / max(len(p), 1); rec = tp / max(len(g), 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-12)
gold = ["B-ORG", "I-ORG", "O", "O", "O", "B-LOC", "I-LOC", "I-LOC", "O", "B-PER", "I-PER", "O"]
pred = gold[:7] + ["O"] + gold[8:]                    # City tagged O, the wrong boundary of 15.5.3
print(spans(pred), [round(x, 3) for x in entity_f1([gold], [pred])])

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import time
import sklearn_crfsuite
from datasets import load_dataset
conll = load_dataset("eriktks/conll2003", revision="refs/convert/parquet")   # Tjong Kim Sang and De Meulder (2003)
names = conll["train"].features["ner_tags"].feature.names                    # O, B-PER, I-PER, B-ORG, ...
print({k: len(v) for k, v in conll.items()}, names)
def word_features(sent, i):
    w = sent[i]
    f = {"bias": 1.0, "lower": w.lower(), "suffix3": w[-3:], "suffix2": w[-2:],
         "is_title": w.istitle(), "is_upper": w.isupper(), "is_digit": w.isdigit(),
         "shape": "".join("X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c
                          for c in w)[:6]}
    for off in (-1, 1):                               # the neighbors' words and capitalization
        j = i + off
        if 0 <= j < len(sent):
            f.update({f"{off}:lower": sent[j].lower(), f"{off}:is_title": sent[j].istitle()})
        else:
            f["BOS" if off < 0 else "EOS"] = True
    return f
def featurize(split):
    X = [[word_features(s, i) for i in range(len(s))] for s in conll[split]["tokens"]]
    y = [[names[t] for t in tags] for tags in conll[split]["ner_tags"]]
    return X, y
Xtr, ytr = featurize("train"); Xte, yte = featurize("test")
bio = [[names[t] for t in tags] for split in conll for tags in conll[split]["ner_tags"]]
opens = sum(t.startswith("I-") and (i == 0 or s[i - 1][2:] != t[2:]) for s in bio for i, t in enumerate(s))
n_test = sum(len(s) for s in yte)
print(f"I- tags that open an entity: {opens} in {len(bio)} sentences; "
      f"O is {sum(t == 'O' for s in yte for t in s) / n_test:.3f} of the {n_test} test tokens")
crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100,
                           all_possible_transitions=True)   # L1 and L2 penalties on the weights lambda_k
t0 = time.time(); crf.fit(Xtr, ytr); secs = time.time() - t0
pred = crf.predict(Xte)
acc = sum(a == b for s, q in zip(yte, pred) for a, b in zip(s, q)) / n_test
p, r, f1 = entity_f1(yte, pred)
print(f"CRF trained in {secs:.0f} s on {len(Xtr)} sentences, {len(crf.state_features_)} state features")
print(f"test token accuracy {acc:.4f}; entity precision {p:.4f} recall {r:.4f} F1 {f1:.4f}")
for a, b in [("O", "I-PER"), ("B-PER", "I-PER"), ("I-PER", "I-PER"), ("B-PER", "I-ORG")]:
    print(f"transition weight {a:>5} -> {b}: {crf.transition_features_[(a, b)]:+.3f}")

# %% [markdown]
# Example 15-19, the fine-tune of distilbert-base-cased, is a long run; it runs as code/ch15_nlp/08_ner_token_classifier.py.

# %% [markdown]
# ## 15.5.5 Extractive question answering

# %%
import numpy as np
toks = ["the", "lab", "opened", "in", "1998", "downtown"]      # a six-token passage, constructed
start = np.array([0.2, 1.0, -0.5, 0.1, 2.5, 0.3])              # hand-set start logits
end = np.array([0.1, 2.6, -0.3, 0.0, 2.2, 0.8])                # hand-set end logits
print("separate argmax:", toks[start.argmax()], "to", toks[end.argmax()])   # the end comes before the start
spans = [(i, j) for i in range(6) for j in range(i, min(i + 3, 6))]          # i <= j, at most 3 tokens
i, j = max(spans, key=lambda s: start[s[0]] + end[s[1]])
p = lambda z: np.exp(z) / np.exp(z).sum()
print(f"best legal span {toks[i:j + 1]}, score {start[i] + end[j]:.1f}, "
      f"probability {p(start)[i] * p(end)[j]:.3f}")

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import re, string, warnings; warnings.filterwarnings("ignore")
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from collections import Counter
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
torch.set_num_threads(2)
squad = load_dataset("rajpurkar/squad", split="validation")     # 10,570 questions on Wikipedia paragraphs
qa_id = "distilbert/distilbert-base-cased-distilled-squad"
tok = AutoTokenizer.from_pretrained(qa_id)
reader = AutoModelForQuestionAnswering.from_pretrained(qa_id).eval()
def answer(question, context, max_len=15):
    enc = tok(question, context, truncation="only_second", max_length=384, stride=128,
              return_overflowing_tokens=True, return_offsets_mapping=True, padding=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping"); enc.pop("overflow_to_sample_mapping")
    with torch.no_grad():
        out = reader(**enc)
    best = (-1e9, "")
    for w in range(len(offsets)):                     # one window per 384-token slice of the passage
        ctx = [k for k, s in enumerate(enc.sequence_ids(w)) if s == 1]   # passage tokens only
        lo, hi = ctx[0], ctx[-1]
        s_log, e_log = out.start_logits[w], out.end_logits[w]
        for a in torch.topk(s_log[lo:hi + 1], 20).indices + lo:          # the 20 best starts and ends
            for b in torch.topk(e_log[lo:hi + 1], 20).indices + lo:
                if a <= b < a + max_len:
                    score = float(s_log[a] - s_log[lo:hi + 1].logsumexp(0) + e_log[b] - e_log[lo:hi + 1].logsumexp(0))
                    if score > best[0]:
                        best = (score, context[int(offsets[w][a][0]):int(offsets[w][b][1])])
    return best[1]
def normalize(s):                                     # the SQuAD v1.1 rule: lowercase, drop punctuation,
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))   # the articles a, an,
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())               # the, and extra spaces
def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    same = sum((Counter(p) & Counter(g)).values())
    return 0.0 if same == 0 else 2 * same / (len(p) + len(g))
rows = squad.select(range(200))
preds = [answer(r["question"], r["context"]) for r in rows]
em = np.mean([max(normalize(p) == normalize(g) for g in r["answers"]["text"]) for p, r in zip(preds, rows)])
fs = np.mean([max(f1(p, g) for g in r["answers"]["text"]) for p, r in zip(preds, rows)])
print(f"first 200 questions: exact match {100 * em:.1f}, F1 {100 * fs:.1f}")
for r, p in list(zip(rows, preds))[:3]:
    print(f"  Q: {r['question']}  ->  {p!r}  (gold {sorted(set(r['answers']['text']))})")
try:
    pipeline("question-answering", model=qa_id)       # the pipeline that earlier releases offered
except KeyError as err:
    print("KeyError:", str(err).split(".")[0])

# %%
long_ctx = max(dict.fromkeys(squad["context"]), key=len)
n_tok = len(tok(long_ctx)["input_ids"])
enc = tok("What is it?", long_ctx, truncation="only_second", max_length=384, stride=128,
          return_overflowing_tokens=True)
print(f"longest passage: {len(long_ctx.split())} words, {n_tok} tokens, {len(enc['input_ids'])} windows "
      f"of at most 384 tokens overlapping by 128")

# %% [markdown]
# The same reader on all 10,570 validation questions runs as code/ch15_nlp/09_squad_full_validation.py.

# %% [markdown]
# ## 15.5.6 Natural language inference and zero-shot classification
#
# The three listings of this section run as code/ch15_nlp/10_zero_shot_nli.py (bart-large-mnli, about 400 million parameters).

# %% [markdown]
# ## 15.5.7 Information extraction: relations, keyphrases, and entity links

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings("ignore")
import spacy
from datasets import load_dataset
nlp = spacy.load("en_core_web_sm")
squad = load_dataset("rajpurkar/squad", split="validation")
paragraphs = list(dict.fromkeys(squad["context"]))[:5]          # the first five distinct Wikipedia paragraphs
def phrase(tok, chunks):                              # the noun chunk around a head word, plus its numbers
    words = set(chunks.get(tok.i, [tok]))             # and compound, modifier, and apposition subtrees
    words |= {t for c in tok.children if c.dep_ in ("nummod", "compound", "amod", "appos") for t in c.subtree}
    words = sorted(words, key=lambda t: t.i)
    return " ".join(t.text for t in words if not t.is_punct and not (t is words[0] and t.pos_ == "DET"))
def triples(doc):
    chunks = {t.i: nc for nc in doc.noun_chunks for t in nc}    # token index -> its noun chunk
    out, noun = [], lambda t: t.pos_ in ("NOUN", "PROPN", "NUM")  # skip pronouns such as which and this
    for verb in (t for t in doc if t.pos_ in ("VERB", "AUX")):
        subj = [c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass") and noun(c)]
        objs = [c for c in verb.children if c.dep_ in ("dobj", "attr") and noun(c)]
        objs += [g for p in verb.children if p.dep_ == "prep" for g in p.children if g.dep_ == "pobj" and noun(g)]
        for s in subj:
            for o in objs:
                rel = verb.lemma_ + ("" if o.head == verb else " " + o.head.text)   # "defeat", "play in"
                out.append((phrase(s, chunks), rel, phrase(o, chunks)))
    return out
found = [t for p in paragraphs for t in triples(nlp(p))]
sentences = sum(len(list(nlp(p).sents)) for p in paragraphs)
print(f"{len(found)} triples from {sentences} sentences in {len(paragraphs)} paragraphs")
for t in found[:8]:
    print("  ", t)

# %%
import numpy as np, networkx as nx
from sklearn.datasets import fetch_20newsgroups
space = fetch_20newsgroups(subset="train", categories=["sci.space"], remove=("headers", "footers", "quotes"))
post = max(space.data[:200], key=len)                 # the longest of the first 200 sci.space posts
doc = nlp(" ".join(post.split()))
cand = [t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "PROPN", "ADJ") and t.is_alpha and len(t) > 2]
G = nx.Graph()
for a, b in zip(cand, cand[1:]):                      # cooccurrence window of 2 over the candidate words
    if a != b: G.add_edge(a, b)
nodes = list(G.nodes); A = nx.to_numpy_array(G, nodelist=nodes)
M = A / A.sum(0, keepdims=True)                       # column-normalized adjacency, as in 20.2.1
r = np.full(len(nodes), 1 / len(nodes))
for it in range(100):                                 # power iteration with damping d = 0.85
    r_new = 0.15 / len(nodes) + 0.85 * M @ r
    if np.abs(r_new - r).sum() < 1e-10: break
    r = r_new
pr = nx.pagerank(G, alpha=0.85, tol=1e-12)
print(f"{len(doc)} tokens, {len(nodes)} candidate words, {G.number_of_edges()} edges, "
      f"converged in {it} iterations, max gap to networkx {max(abs(r[i] - pr[n]) for i, n in enumerate(nodes)):.1e}")
top = [nodes[i] for i in np.argsort(-r)[:len(nodes) // 3]]   # Mihalcea and Tarau keep the top third
print("top words:", top[:8])
words = [t.lemma_.lower() for t in doc]
phrases, run = [], []
for w in words + [""]:                                # join adjacent top words into phrases
    if w in top and w not in run: run.append(w)       # a repeated word starts a new phrase
    else:
        if len(run) > 1: phrases.append(" ".join(run))
        run = [w] if w in top else []
score = lambda p: sum(r[nodes.index(w)] for w in p.split())
print("top phrases:", sorted(set(phrases), key=lambda p: -score(p))[:5])

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
from sentence_transformers import SentenceTransformer
kb = {"Q90": "Paris: capital and most populous city in France",          # Wikidata labels and English
      "Q830149": "Paris: city in and county seat of Lamar County, Texas, United States",   # descriptions,
      "Q47899": "Paris Hilton: American media personality (born 1981)"}  # read 5 November 2026
mentions = ["We took the train from Lyon to Paris to see the Louvre.",   # three constructed sentences
            "She grew up in Paris, Texas, and still drives a pickup truck.",
            "Paris posted a new selfie and thanked her fans for the birthday wishes."]
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
K = enc.encode(list(kb.values()), normalize_embeddings=True)
S = enc.encode(mentions, normalize_embeddings=True)
for s, row in zip(mentions, S @ K.T):
    print(f"{list(kb)[int(row.argmax())]:8} {np.round(row, 3)}  {s}")

# %% [markdown]
# ## 15.5.8 What we built, and a self-check

# %% [markdown]
# ## 15.6 Hands-on lab: a sentiment classifier up the representation ladder

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
                          pipeline)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Real data: rotten_tomatoes movie reviews, label 1 = positive, 0 = negative.
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
tr = ds["train"].shuffle(seed=0).select(range(800))   # small slices keep the lab fast
te = ds["test"].shuffle(seed=0).select(range(400))
Xtr, ytr = tr["text"], np.array(tr["label"])
Xte, yte = te["text"], np.array(te["label"])

# Rung 1: TF-IDF counts, no English stop list. That list deletes not, no, never, nothing, but, very, too.
m = make_pipeline(TfidfVectorizer(max_features=5000),
                  LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
acc_tfidf = (m.predict(Xte) == yte).mean()

# Rung 2: frozen DistilBERT contextual embeddings, mean-pooled -> logistic regression.
tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
enc_model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased").eval()

def embed(texts, bs=32):
    vecs = []
    for k in range(0, len(texts), bs):
        enc = tok(list(texts[k:k+bs]), return_tensors="pt",
                  padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            h = enc_model(**enc).last_hidden_state         # (batch, tokens, 768)
        mask = enc["attention_mask"].unsqueeze(-1)
        vecs.append(((h * mask).sum(1) / mask.sum(1)).numpy())   # average the real tokens
    return np.vstack(vecs)

lr = LogisticRegression(max_iter=1000).fit(embed(Xtr), ytr)
acc_emb = (lr.predict(embed(Xte)) == yte).mean()

# Rung 3: fine-tune distilbert-base-uncased on these 800 reviews. Two epochs, AdamW, lr 2e-5.
torch.manual_seed(0)                                  # seed first: from_pretrained draws the new head's weights
ft = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2)
torch.manual_seed(0)
opt = torch.optim.AdamW(ft.parameters(), lr=2e-5)
ft.train()
Xtr_l = list(Xtr)
for epoch in range(2):
    order = np.random.default_rng(epoch).permutation(len(Xtr_l))
    for i in range(0, len(Xtr_l), 16):
        idx = order[i:i + 16]
        enc = tok([Xtr_l[j] for j in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=64)
        enc["labels"] = torch.tensor(ytr[idx])
        opt.zero_grad()
        ft(**enc).loss.backward()
        opt.step()
ft.eval()
correct = 0
with torch.no_grad():
    for i in range(0, len(Xte), 16):
        enc = tok(list(Xte)[i:i + 16], return_tensors="pt", padding=True,
                  truncation=True, max_length=64)
        pred = ft(**enc).logits.argmax(-1).tolist()
        correct += sum(int(p == int(y)) for p, y in zip(pred, yte[i:i + 16]))
acc_ft = correct / len(yte)

# The off-the-shelf SST-2 checkpoint is a warning, not a rung: it saw much of this test set.
def _norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())
sst_rows = load_dataset("stanfordnlp/sst2", split="train")
seen = {_norm(s) for s in sst_rows["sentence"]}
mask = np.array([_norm(s) in seen for s in Xte])
sst = pipeline("sentiment-analysis",
               model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
               truncation=True, device="cpu")
sst_pred = np.array([1 if r["label"] == "POSITIVE" else 0
                     for r in sst(list(Xte), batch_size=32)])

print(f"1. TF-IDF + logistic regression         : {acc_tfidf:.3f}")
print(f"2. DistilBERT embeddings + logistic reg : {acc_emb:.3f}")
print(f"3. fine-tuned on these 800 reviews      : {acc_ft:.3f}")
print(f"SST-2 checkpoint, all / overlap / clean : "
      f"{(sst_pred == yte).mean():.3f} / {(sst_pred[mask] == yte[mask]).mean():.3f} / "
      f"{(sst_pred[~mask] == yte[~mask]).mean():.3f}  "
      f"({int(mask.sum())} overlapping of {len(yte)})")

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
m_stop = make_pipeline(TfidfVectorizer(max_features=5000, stop_words="english"),
                       LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
print(f"rung 1 correct: {int((m.predict(Xte) == yte).sum())} of {len(yte)}; "
      f"with the English stop list: {(m_stop.predict(Xte) == yte).mean():.3f}")
full_test = ds["test"]["text"]
print(f"SST-2 training rows: {len(sst_rows)}; full test sentences found there: "
      f"{sum(_norm(s) in seen for s in full_test)} of {len(full_test)}")

# %% [markdown]
# ## 15.6.1 What we built, and a self-check

# %% [markdown]
# ## 15.7 Interview questions and coding exercises

# %% [markdown]
# ## Pipeline and representation

# %% [markdown]
# ## Tokenization and context

# %% [markdown]
# ## Tasks and evaluation

# %% [markdown]
# ## Coding exercises

# %%
from sklearn.datasets import fetch_20newsgroups
texts = fetch_20newsgroups(subset="train", categories=["sci.space"],
                           remove=("headers", "footers", "quotes")).data
def preprocess(text):
    ...
print(len(texts), preprocess(texts[0]))

# %%
import numpy as np
E = np.array([[0., 1, 2], [0, 0.5, 3], [3, 0, 0], [0, 1.8, 2.2], [0, 0.5, 3]])
def viterbi(E):
    ...
print(E.shape, viterbi(E))

# %%
docs = ["the cat sat on the mat", "the dog sat on the log",
        "the cats and dogs are pets", "happy dog and happy cat too"]
def smoothed_idf(docs):
    ...
print(len(docs), smoothed_idf(docs))

# %%
from sklearn.datasets import fetch_20newsgroups
texts = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes")).data[:3000]
def zipf_slope(texts):
    ...
print(len(texts), zipf_slope(texts))

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import re
from collections import Counter
from datasets import load_dataset
raw = load_dataset("Trelis/tiny-shakespeare")
lines = [l.strip().lower() for l in "".join(raw["train"]["Text"]).splitlines()]
lines = [l for l in lines if l and not l.endswith(":")]          # dialogue lines, as in Example 15-14
train_words = Counter(w for l in lines[:2000] for w in re.findall(r"[a-z']+", l))
held_out = [w for l in lines[2000:2200] for w in re.findall(r"[a-z']+", l)]
def tokens_per_word(merges, words):
    ...
print(len(train_words), len(held_out), tokens_per_word([], held_out))

# %%
import math
def skipgram_and_glove():
    ...
print(round(1 / (1 + math.exp(-1.2)), 3), skipgram_and_glove())

# %%
import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-50")
def analogy(pos, neg):
    ...
print(glove.vector_size, analogy(["king", "woman"], ["man"]))

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
def vectors_by_layer(sentence, word):
    ...
print(tok.convert_tokens_to_ids("plant"), vectors_by_layer("the green plant", "plant"))

# %%
ref = "the boy kicked the ball into the net"
def bleu2(cand, ref):
    ...
print(ref.split(), bleu2("the boy kicked the ball over the net", ref))

# %%
from sklearn.datasets import fetch_20newsgroups
cats = ["sci.space", "rec.autos", "rec.sport.hockey", "comp.graphics"]
data = fetch_20newsgroups(subset="train", categories=cats,
                          remove=("headers", "footers", "quotes"))
def precision_at_1(data):
    ...
print(len(data.data), len(set(data.target)), precision_at_1(data))

# %% [markdown]
# ## Coding drills

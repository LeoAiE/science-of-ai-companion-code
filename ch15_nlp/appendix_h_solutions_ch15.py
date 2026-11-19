"""Appendix H solutions for Chapter 15, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

# %% [markdown]
# ## 15.7 coding exercises

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
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

# %% [markdown]
# 1. Classical preprocessing pipeline: vocabulary reduction on 20newsgroups

# %%
# 1. Classical preprocessing pipeline: vocabulary reduction on 20newsgroups
texts = fetch_20newsgroups(subset="train", categories=["sci.space"],
                           remove=("headers", "footers", "quotes")).data
stop = {"the","a","an","of","and","to","in","is","it","that","this","for",
        "on","with","as","was","are","be","by","or","at","from","but","not"}
def stem(w):
    for suf in ("ing","ly","ed","es","s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w
def preprocess(t):
    toks = "".join(c for c in t.lower() if c.isalpha() or c == " ").split()
    return [stem(w) for w in toks if w not in stop]
raw = set(w for t in texts for w in
          "".join(c for c in t.lower() if c.isalpha() or c == " ").split())
proc = set(w for t in texts for w in preprocess(t))
print(len(raw), len(proc), round(1 - len(proc)/len(raw), 3))
# expected output: 17727 15665 0.116

# %% [markdown]
# 2. CRF Viterbi decoding on a new sentence: an illegal start and two names

# %%
# 2. CRF Viterbi decoding on a new sentence: an illegal start and two names
labels = ["O", "B-PER", "I-PER"]; tokens = ["marie","curie","met","pierre","curie"]
E = np.array([[0.,1,2],[0,.5,3],[3,0,0],[0,1.8,2.2],[0,.5,3]])
print("greedy:", [labels[i] for i in E.argmax(1)])
NEG = -1e9; T = np.zeros((3,3)); T[0,2] = NEG; start = np.array([0.,0.,NEG])
def viterbi(E, T=T, start=start):              # the starter's viterbi(E), with this exercise's T and start
    n, k = E.shape; dp = start + E[0]; back = np.zeros((n,k), int)
    for t in range(1, n):
        sc = dp[:,None] + T; back[t] = sc.argmax(0); dp = sc.max(0) + E[t]
    path = [int(dp.argmax())]
    for t in range(n-1, 0, -1): path.append(int(back[t, path[-1]]))
    return path[::-1]
vit = viterbi(E); print("viterbi:", [labels[i] for i in vit])
spans = []
for w, i in zip(tokens, vit):
    if labels[i] == "B-PER": spans.append([w])
    elif labels[i] == "I-PER": spans[-1].append(w)
print("spans:", [" ".join(s) for s in spans])
def path_score(seq):                           # start + emissions + transitions, as in 15.1
    s = start[seq[0]] + E[0, seq[0]]
    for t in range(1, len(seq)): s += T[seq[t-1], seq[t]] + E[t, seq[t]]
    return s
legal = [s for s in (path_score(q) for q in np.ndindex(3,3,3,3,3)) if s > -1e8]
Z = np.exp(legal).sum()
print(f"legal {len(legal)} of 243, Z={Z:.0f}, p(viterbi)={np.exp(path_score(vit))/Z:.3f}")
# expected output: greedy: ['I-PER', 'I-PER', 'O', 'I-PER', 'I-PER']
# expected output: viterbi: ['B-PER', 'I-PER', 'O', 'B-PER', 'I-PER']
# expected output: spans: ['marie curie', 'pierre curie']
# expected output: legal 89 of 243, Z=224690, p(viterbi)=0.593
# The greedy path's raw emission total is 13.2 against the legal path's 11.8, but its two illegal moves
# each add the -1e9 penalty, so Viterbi returns two names and e^11.8 / Z gives the legal path 0.593.

# %% [markdown]
# 3. TF-IDF from scratch matches scikit-learn

# %%
# 3. TF-IDF from scratch matches scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
docs = ["the cat sat on the mat", "the dog sat on the log",
        "the cats and dogs are pets", "happy dog and happy cat too"]
vec = TfidfVectorizer(); sk = vec.fit_transform(docs).toarray()
vocab = list(vec.get_feature_names_out())
tf = np.array([[d.split().count(w) for w in vocab] for d in docs], float)
def smoothed_idf(docs):                        # the starter's function: ln((1+N)/(1+df)) + 1 for each word of vocab
    df = np.array([sum(w in d.split() for d in docs) for w in vocab])
    return np.log((1 + len(docs))/(1 + df)) + 1
idf = smoothed_idf(docs)
tfidf = tf * idf; tfidf /= np.linalg.norm(tfidf, axis=1, keepdims=True)
print(f"{np.abs(tfidf - sk).max():.6f}")
print(f"{idf[vocab.index('the')]:.3f}", f"{idf[vocab.index('cat')]:.3f}")
# expected output: 0.000000
# expected output: 1.223 1.511

# %% [markdown]
# 4. Zipf's law on posts the chapter never counted (the test split)

# %%
# 4. Zipf's law on posts the chapter never counted (the test split)
texts_t = fetch_20newsgroups(subset="test",
                             remove=("headers","footers","quotes")).data[:3000]
def zipf_slope(texts):                         # the starter's function: log-log slope over the top 2000 ranks
    ranked = np.array([c for _, c in
        Counter(w for t in texts for w in t.lower().split() if w.isalpha()).most_common()])
    ranks = np.arange(1, 2001)
    return float(np.polyfit(np.log(ranks), np.log(ranked[:2000]), 1)[0]), ranked
slope, ranked = zipf_slope(texts_t)
for r in [1, 10, 100, 1000]:
    print(f"rank {r:>4}: freq {ranked[r-1]:>6}  freq*rank {ranked[r-1]*r}")
print("slope", round(slope, 2))
print("types", len(ranked), "seen once", round(float((ranked == 1).mean()), 3))
# expected output: rank    1: freq  27484  freq*rank 27484
# expected output: rank   10: freq   4972  freq*rank 49720
# expected output: rank  100: freq    417  freq*rank 41700
# expected output: rank 1000: freq     43  freq*rank 43000
# expected output: slope -1.02
# expected output: types 24431 seen once 0.452
# The slope matches the -1.02 of the training posts, and almost half the distinct words occur only once.

# %% [markdown]
# 5. Byte-pair encoding learned from real text: tokens per word as the merges grow

# %%
# 5. Byte-pair encoding learned from real text: tokens per word as the merges grow
import re
from datasets import load_dataset
raw = load_dataset("Trelis/tiny-shakespeare")
lines = [l.strip().lower() for l in "".join(raw["train"]["Text"]).splitlines()]
lines = [l for l in lines if l and not l.endswith(":")]           # dialogue lines, as in Example 15-14
train_words = Counter(w for l in lines[:2000] for w in re.findall(r"[a-z']+", l))
held_out = [w for l in lines[2000:2200] for w in re.findall(r"[a-z']+", l)]
def get_pairs(v):
    p = Counter()
    for w, f in v.items():
        s = w.split()
        for i in range(len(s)-1): p[(s[i], s[i+1])] += f
    return p
def merge(pair, v):
    out = {}
    for w, f in v.items():
        syms, merged, i = w.split(), [], 0
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == pair[0] and syms[i + 1] == pair[1]:
                merged.append(pair[0] + pair[1]); i += 2
            else:
                merged.append(syms[i]); i += 1
        out[" ".join(merged)] = f
    return out
def bpe_tokenize(word, merge_list):             # apply the learned merges in order
    syms = list(word) + ["</w>"]
    for a, b in merge_list:
        i = 0
        while i < len(syms)-1:
            if syms[i] == a and syms[i+1] == b: syms[i:i+2] = [a+b]
            else: i += 1
    return syms
def tokens_per_word(merge_list, words):
    return sum(len(bpe_tokenize(w, merge_list)) for w in words) / len(words)
vb = {" ".join(w) + " </w>": f for w, f in train_words.items()}
learned = []
for _ in range(300):
    pairs = get_pairs(vb)
    if not pairs: break
    best = pairs.most_common(1)[0][0]; learned.append(best); vb = merge(best, vb)
print(len(train_words), len(held_out), "first merges:", ", ".join("+".join(m) for m in learned[:6]))
for k in [0, 50, 100, 200, 300]:
    print(f"{k:3d} merges: {tokens_per_word(learned[:k], held_out):.3f} tokens per held-out word")
unseen = sorted({w for w in held_out if w not in train_words and w.isalpha()}, key=lambda w: (-len(w), w))
print(len(unseen), "held-out word types never seen in training")
for w in unseen[:3]:
    print(w, bpe_tokenize(w, learned))
alphabet = {c for w in train_words for c in w}
print("held-out words with an unseen character:", sum(any(c not in alphabet for c in w) for w in held_out))
# expected output: 2544 1270 first merges: e+</w>, t+h, s+</w>, t+</w>, d+</w>, o+u
# expected output:   0 merges: 5.243 tokens per held-out word
# expected output:  50 merges: 3.324 tokens per held-out word
# expected output: 100 merges: 2.787 tokens per held-out word
# expected output: 200 merges: 2.280 tokens per held-out word
# expected output: 300 merges: 2.046 tokens per held-out word
# expected output: 98 held-out word types never seen in training
# expected output: multitudinous ['mu', 'l', 'ti', 'tu', 'd', 'in', 'ou', 's</w>']
# expected output: apprehended ['a', 'p', 'pre', 'h', 'en', 'd', 'ed</w>']
# expected output: fundamental ['f', 'un', 'd', 'a', 'm', 'en', 'ta', 'l</w>']
# expected output: held-out words with an unseen character: 0
# With no merges a word costs its letters plus the end marker, 5.243 tokens on average; 300 merges cut that to
# 2.046, with the steepest fall in the first 50. The unseen words split into learned pieces such as 'pre', 'en',
# and 'ed</w>', and no held-out word has a character outside the training alphabet, so nothing is unknown.

# %% [markdown]
# 6. Skip-gram-with-negative-sampling and GloVe objectives on new numbers

# %%
# 6. Skip-gram-with-negative-sampling and GloVe objectives on new numbers
sigmoid = lambda z: 1/(1+np.exp(-z))
def sgns_loss(pos, negs):                      # one true context, any number of negatives
    return -np.log(sigmoid(pos)) - sum(np.log(sigmoid(-d)) for d in negs)
def glove_term(X, dot, bi, bj, xmax=100.0, alpha=0.75):
    f = min(1.0, (X/xmax)**alpha); resid = dot + bi + bj - np.log(X)
    return f, resid, f*resid**2
pairs6 = [(12.0, 2.0, 0.3, -0.1), (250.0, 6.0, 0.4, 0.2)]   # (count X, dot product, b_i, b_j)
def skipgram_and_glove():                      # the starter's function: both objectives on this exercise's numbers
    return sgns_loss(1.2, [0.8, -0.3]), [glove_term(*p) for p in pairs6]
loss, terms = skipgram_and_glove()
print(f"{sigmoid(1.2):.3f} {sigmoid(-0.8):.3f} {sigmoid(0.3):.3f} {loss:.3f}")
for (X, *_), (f, resid, term) in zip(pairs6, terms):
    print(f"X={X:g}: {f:.3f} {resid:.3f} {term:.3f}")
# expected output: 0.769 0.310 0.574 1.989
# expected output: X=12: 0.204 -0.285 0.017
# expected output: X=250: 1.000 1.079 1.163
# The rare pair gets weight 0.204, and dot + b_i + b_j = 2.0 + 0.3 - 0.1 = 2.2 falls short of log 12 = 2.485
# (residual -0.285), so a gradient step raises its dot product. The frequent pair is capped at weight 1, and its
# 6.0 + 0.4 + 0.2 = 6.6 exceeds log 250 = 5.521 (residual 1.079), so a gradient step lowers its dot product.

# %% [markdown]
# 7. Real GloVe vectors: analogies and meaning-not-spelling similarity

# %%
# 7. Real GloVe vectors: analogies and meaning-not-spelling similarity
import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-50")   # ~66 MB download, cached after first run
def analogy(pos, neg):                         # the starter's function: nearest word, query words excluded, and its cosine
    return glove.most_similar(positive=pos, negative=neg, topn=1)[0]
for lbl, pos, neg in [("king-man+woman", ["king","woman"], ["man"]),
                      ("walking-walk+swim", ["walking","swim"], ["walk"]),
                      ("better-good+bad", ["better","bad"], ["good"])]:
    w, s = analogy(pos, neg)
    print(f"{lbl:20} -> {w} {s:.3f}")
print(f"cat/dog {glove.similarity('cat','dog'):.3f}  cat/democracy {glove.similarity('cat','democracy'):.3f}")
# expected output: king-man+woman       -> queen 0.852
# expected output: walking-walk+swim    -> swimming 0.807
# expected output: better-good+bad      -> worse 0.901
# expected output: cat/dog 0.922  cat/democracy 0.037

# %% [markdown]
# 8. Where DistilBERT's layers separate two senses of "plant"

# %%
# 8. Where DistilBERT's layers separate two senses of "plant"
import torch
from transformers import AutoTokenizer, AutoModel
tk = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased").eval()
def vectors_by_layer(sentence, word):
    enc = tk(sentence, return_tensors="pt")
    with torch.no_grad():
        states = model(**enc, output_hidden_states=True).hidden_states   # embedding output + one per layer
    pos = int((enc["input_ids"][0] == tk.convert_tokens_to_ids(word)).nonzero()[0,0])
    return [s[0, pos].numpy() for s in states], pos
cos = lambda a, b: float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)))
f1, p1 = vectors_by_layer("the car plant hired two hundred new workers", "plant")
f2, p2 = vectors_by_layer("the chemical plant closed after the explosion", "plant")
g1, p3 = vectors_by_layer("the green plant needed water every morning", "plant")
print("positions", p1, p2, p3, " hidden states", len(f1))
for layer in range(len(f1)):
    same, diff = cos(f1[layer], f2[layer]), cos(f1[layer], g1[layer])
    print(f"state {layer}: same {same:.3f}  diff {diff:.3f}  gap {same - diff:+.3f}")
# expected output: positions 3 3 3  hidden states 7
# expected output: state 0: same 1.000  diff 1.000  gap +0.000
# expected output: state 1: same 0.888  diff 0.842  gap +0.046
# expected output: state 2: same 0.920  diff 0.796  gap +0.125
# expected output: state 3: same 0.926  diff 0.788  gap +0.137
# expected output: state 4: same 0.930  diff 0.817  gap +0.113
# expected output: state 5: same 0.903  diff 0.816  gap +0.087
# expected output: state 6: same 0.833  diff 0.688  gap +0.145
# At the embedding output each vector is the one table row for "plant" plus the position 3 embedding, passed
# through a layer normalization, so the three vectors are identical and the gap is exactly 0. The context-mixing
# layers open the gap; it is largest after the last layer (0.145), with layer 3 close behind (0.137).

# %% [markdown]
# 9. BLEU-2 and ROUGE-1 from scratch on new sentences, checked against the packages

# %%
# 9. BLEU-2 and ROUGE-1 from scratch on new sentences, checked against the packages
def ngrams(t, n): return Counter(tuple(t[i:i+n]) for i in range(len(t)-n+1))
def bleu2(cand, ref, max_n=2):                 # n up to 2; standard BLEU-4 uses n up to 4
    c, r = cand.split(), ref.split(); ps = []
    for n in range(1, max_n+1):
        cg, rg = ngrams(c, n), ngrams(r, n)
        if not cg: return 0.0
        ps.append(sum(min(cg[g], rg[g]) for g in cg)/sum(cg.values()))
    if min(ps) == 0: return 0.0
    return min(1.0, np.exp(1 - len(r)/max(len(c),1))) * np.exp(np.mean(np.log(ps)))
def rouge1(cand, ref):
    c, r = Counter(cand.split()), Counter(ref.split())
    return sum(min(c[w], r[w]) for w in r)/sum(r.values())
ref = "the boy kicked the ball into the net"
cands9 = {"near-copy":"the boy kicked the ball over the net",
          "short":"the boy kicked the ball",
          "paraphrase":"a youngster booted the football into the goal",
          "unrelated":"my train leaves the station at nine"}
for name, cand in cands9.items():
    print(f"{name:10} BLEU-2 {bleu2(cand,ref):.3f}  ROUGE-1 {rouge1(cand,ref):.3f}")
print(f"brevity penalty of the short candidate: {np.exp(1 - 8/5):.3f}")
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
toy = BLEU(max_ngram_order=2, smooth_method="none", tokenize="none", effective_order=True)
rs = rouge_scorer.RougeScorer(["rouge1"])
print("packages agree:",
      all(abs(toy.sentence_score(c, [ref]).score/100 - bleu2(c, ref)) < 1e-9 and
          abs(rs.score(ref, c)["rouge1"].recall - rouge1(c, ref)) < 1e-9 for c in cands9.values()))
# expected output: near-copy  BLEU-2 0.791  ROUGE-1 0.875
# expected output: short      BLEU-2 0.549  ROUGE-1 0.625
# expected output: paraphrase BLEU-2 0.231  ROUGE-1 0.375
# expected output: unrelated  BLEU-2 0.000  ROUGE-1 0.125
# expected output: brevity penalty of the short candidate: 0.549
# expected output: packages agree: True
# The short candidate's two precisions are both 1, so its BLEU-2 is the brevity penalty e^(1-8/5) alone.
# The paraphrase keeps the meaning but shares only "the", "into", and "into the", so it scores below the
# short candidate, which never says where the ball went, on both metrics.

# %% [markdown]
# 10. Miniproject: BM25 vs TF-IDF retrieval, precision@1 on 20newsgroups

# %%
# 10. Miniproject: BM25 vs TF-IDF retrieval, precision@1 on 20newsgroups
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
cats = ["sci.space","rec.autos","rec.sport.hockey","comp.graphics"]
train = fetch_20newsgroups(subset="train", categories=cats,
                           remove=("headers","footers","quotes"))
test = fetch_20newsgroups(subset="test", categories=cats,
                          remove=("headers","footers","quotes"))
def precision_at_1(data, stop=frozenset()):    # the starter's function: corpus sampled from data, queries from test
    rng = np.random.default_rng(0)
    ci = rng.choice(len(data.data), 400, replace=False)
    corpus = [data.data[i] for i in ci]; cy = data.target[ci]
    qi = rng.choice(len(test.data), 100, replace=False)
    queries = [test.data[i] for i in qi]; qy = test.target[qi]
    tok = lambda d: [w for w in d.lower().split() if w.isalpha() and w not in stop]
    ctok = [tok(d) for d in corpus]; Nd = len(ctok); avgdl = np.mean([len(d) for d in ctok])
    df = Counter(w for d in ctok for w in set(d))
    def bm25(qt, k=1.5, b=0.75):
        sc = np.zeros(Nd)
        for j, d in enumerate(ctok):
            dc = Counter(d); dl = len(d)
            for w in set(qt):
                if df.get(w,0) == 0: continue
                idf = np.log((Nd-df[w]+0.5)/(df[w]+0.5)+1)
                sc[j] += idf*(dc[w]*(k+1))/(dc[w] + k*(1-b+b*dl/avgdl))
        return sc
    bm25_hits = sum(int(cy[int(bm25(tok(q)).argmax())] == y) for q, y in zip(queries, qy))
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    C = vec.fit_transform(corpus); Q = vec.transform(queries); sims = (Q @ C.T).toarray()
    tfidf_hits = sum(int(cy[int(sims[i].argmax())] == qy[i]) for i in range(len(queries)))
    return bm25_hits/len(queries), tfidf_hits/len(queries)
p_bm25, p_tfidf = precision_at_1(train)
print(f"BM25 P@1 {p_bm25:.3f}  TF-IDF P@1 {p_tfidf:.3f}")
print(f"BM25 P@1 with the English stop list {precision_at_1(train, ENGLISH_STOP_WORDS)[0]:.3f}")
# expected output: BM25 P@1 0.630  TF-IDF P@1 0.680
# expected output: BM25 P@1 with the English stop list 0.650
# Both retrievers are far above the 0.25 chance level of four balanced categories: BM25 retrieves a post of the
# query's category for 63 of the 100 queries and TF-IDF for 68. The two sides are not preprocessed alike. Only the
# TF-IDF side drops English stop words and keeps 5,000 terms, and with the same stop list BM25 reaches 0.650,
# still 3 queries behind. On this sample TF-IDF wins, so the run does not show the advantage on heavily repeated
# terms that 15.2.3 describes for BM25.

# %% [markdown]
# ## 15.7 coding drills

# %%
import numpy as np
from collections import Counter

# %% [markdown]
# Preprocessing and normalization

# %%
# Preprocessing and normalization
text = "The CEOs Were Running Quickly."
toks = "".join(c for c in text.lower() if c.isalpha() or c == " ").split()
print(toks)                                                    # expected output: ['the', 'ceos', 'were', 'running', 'quickly']
print([w for w in toks if w not in {"the","were","to"}])       # expected output: ['ceos', 'running', 'quickly']
def stem(w):
    for s in ("ing","ly","ed","es","s"):
        if w.endswith(s) and len(w)-len(s) >= 3: return w[:-len(s)]
    return w
print([stem(w) for w in ["running","quickly","ceos"]])         # expected output: ['runn', 'quick', 'ceo']
words = "the dogs were running and the dog runs".split()
print(len(set(words)), len(set(stem(w) for w in words if w not in {"the","were","to"})))   # expected output: 7 4

# %% [markdown]
# Counts, TF-IDF, and Zipf

# %%
# Counts, TF-IDF, and Zipf
docs = ["the cat sat", "the dog ran", "the cat ran"]
vocab = sorted(set(w for d in docs for w in d.split()))
tf = np.array([[d.split().count(w) for w in vocab] for d in docs], float)
print(vocab, tf[0].astype(int).tolist())                       # expected output: ['cat', 'dog', 'ran', 'sat', 'the'] [1, 0, 0, 1, 1]
idf = np.log(len(docs)/(tf > 0).sum(0))
print({w: round(float(i), 2) for w, i in zip(vocab, idf)})     # expected output: {'cat': 0.41, 'dog': 1.1, 'ran': 0.41, 'sat': 1.1, 'the': 0.0}
a, b = tf[0], tf[2]
print(round(float(a @ b/(np.linalg.norm(a)*np.linalg.norm(b))), 3))   # expected output: 0.667
print([f*r for r, f in enumerate([300,150,100], 1)])           # expected output: [300, 300, 300]

# %% [markdown]
# Tokenization and BPE

# %%
# Tokenization and BPE
v = {"l o w </w>":5, "l o w e r </w>":2, "n e w </w>":6}
def get_pairs(v):
    p = Counter()
    for w, f in v.items():
        s = w.split()
        for i in range(len(s)-1): p[(s[i], s[i+1])] += f
    return p
print(get_pairs(v).most_common(1)[0])                          # expected output: (('w', '</w>'), 11)
def merge(pair, v):
    out = {}
    for w, f in v.items():
        syms, merged, i = w.split(), [], 0
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == pair[0] and syms[i + 1] == pair[1]:
                merged.append(pair[0] + pair[1]); i += 2
            else:
                merged.append(syms[i]); i += 1
        out[" ".join(merged)] = f
    return out
print(list(merge(("l","o"), v).keys())[0])                     # expected output: lo w </w>
print(len(list("coldest")) + 1)                                # expected output: 8
def apply_merges(word, merges):
    syms = list(word) + ["</w>"]
    for a, b in merges:                                        # each merge, in the order it was learned
        i = 0
        while i < len(syms) - 1:
            if syms[i] == a and syms[i+1] == b: syms[i:i+2] = [a + b]
            else: i += 1
    return syms
print(apply_merges("lower", [("l","o"), ("lo","w"), ("e","r"), ("er","</w>")]))   # expected output: ['low', 'er</w>']

# %% [markdown]
# Embedding objectives and analogies

# %%
# Embedding objectives and analogies
sig = lambda z: 1/(1+np.exp(-z))
print(round(float(sig(2.0)), 3), round(float(sig(-0.5)), 3))   # expected output: 0.881 0.378
print(round(float(-np.log(sig(2.0)) - np.log(sig(-0.5))), 3))  # expected output: 1.101
f = lambda x, xmax=100, a=0.75: (x/xmax)**a if x < xmax else 1.0
print(round(f(30), 3), round(f(150), 3))                       # expected output: 0.405 1.0
king, man, woman, queen = (np.array(v) for v in ([.9,.8],[.8,.1],[.2,.9],[.4,1.6]))
g = king - man + woman
print(round(float(g @ queen/(np.linalg.norm(g)*np.linalg.norm(queen))), 3))   # expected output: 0.998

# %% [markdown]
# Ranking and evaluation metrics

# %%
# Ranking and evaluation metrics
def bm25_term(tf, dl, avgdl, N, nq, k=1.5, b=0.75):
    idf = np.log((N-nq+0.5)/(nq+0.5)+1)
    return idf*(tf*(k+1))/(tf + k*(1-b+b*dl/avgdl))
print(round(float(bm25_term(3, 100, 80, 10, 1)), 3))           # expected output: 3.125
c, r = "the cat".split(), "the cat is here".split()
print(round(float(min(1.0, np.exp(1 - len(r)/len(c)))), 3))    # expected output: 0.368
cand, ref = Counter("the cat ran fast".split()), Counter("the cat sat".split())
print(round(sum(min(cand[w], ref[w]) for w in ref)/sum(ref.values()), 3))   # expected output: 0.667

# %% [markdown]
# Decoding and perplexity

# %%
# Decoding and perplexity
def softmax_T(z, T=1.0):
    e = np.exp((z - z.max())/T); return e/e.sum()
z = np.array([2.0, 1.0])
print([round(float(softmax_T(z, T)[0]), 3) for T in (0.5, 1.0, 2.0)])   # expected output: [0.881, 0.731, 0.622]
def topk(p, k):
    idx = p.argsort()[::-1][:k]; m = np.zeros_like(p); m[idx] = p[idx]; return m/m.sum()
print([round(float(x), 3) for x in topk(np.array([0.5,0.3,0.15,0.05]), 2)])   # expected output: [0.625, 0.375, 0.0, 0.0]
print(int(np.array([0.1,0.7,0.2]).argmax()))                   # expected output: 1
print(round(float(np.exp(4.5025)), 1), round(float(np.exp(4.503)), 1))  # expected output: 90.2 90.3

# %% [markdown]
# # Section 26.3.1 Example 26-9: product quantization of real MiniLM embeddings
#
# MiniLM (sentence-transformers/all-MiniLM-L6-v2, on the CPU) embeds every unique paragraph of SQuAD's train and
# validation splits, 20,958 of them, at unit length (Rajpurkar et al., 2016; rajpurkar/squad on the Hugging Face Hub),
# and the first 500 validation questions serve as queries. Exact search by brute force gives each query's top 10, and
# faiss.IndexPQ with M = 16 and M = 64 subvectors of one byte each compresses the 384-dimensional vectors to 16 and 64
# bytes; the listing prints recall@10 against the exact top 10. The run takes about 13 minutes on the reference
# machine (one run), most of it the embedding, which is why the listing lives here rather than in the walkthrough.
# Tested with sentence-transformers 6.2.0, faiss-cpu 1.15.1, datasets 5.1.0, and torch 2.15.0 (CPU).

# %% [markdown]
# Example 26-9. Product quantization of real MiniLM embeddings of 20,958 SQuAD paragraphs.

# %%
import warnings; warnings.filterwarnings("ignore")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"               # Mac: torch and faiss each ship libomp
import numpy as np, torch
torch.set_num_threads(2)                                  # two threads, the thread count of the book's builds
import faiss
faiss.omp_set_num_threads(1)                              # one thread: reproducible codebooks
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Every unique SQuAD paragraph of the train and validation splits, in split order
paras, seen = [], set()
for split in ("train", "validation"):
    for c in load_dataset("rajpurkar/squad", split=split)["context"]:
        if c not in seen:
            seen.add(c)
            paras.append(c)
questions = load_dataset("rajpurkar/squad", split="validation")["question"][:500]

st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
X = st.encode(paras, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
Xq = st.encode(questions, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
print(f"{len(paras)} unique paragraphs embedded to {X.shape}; {len(questions)} validation questions as queries")

k, d = 10, X.shape[1]
E = np.argsort(Xq @ X.T, axis=1)[:, ::-1][:, :k]          # the exact top 10 by brute force
for M in (16, 64):
    pq = faiss.IndexPQ(d, M, 8, faiss.METRIC_INNER_PRODUCT)   # M subvectors of d/M dimensions, one byte each
    pq.train(X)
    pq.add(X)
    _, I = pq.search(Xq, k)
    rec = np.mean([len(set(a) & set(b)) / k for a, b in zip(I, E)])
    print(f"IndexPQ M {M}: {pq.code_size} bytes per vector against {4 * d} for float32, "
          f"{4 * d // pq.code_size} times smaller; recall@{k} against exact search {rec:.2f}")

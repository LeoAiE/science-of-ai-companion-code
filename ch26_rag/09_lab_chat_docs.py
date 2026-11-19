# %% [markdown]
# # Section 26.9 Lab: chat with your documents (real SQuAD store)
#
# Example 26-27 of the book: the embedder sentence-transformers/all-MiniLM-L6-v2, the generator
# Qwen/Qwen2.5-0.5B-Instruct on the CPU in 32-bit floats, and the Wikipedia paragraphs of the
# rajpurkar/squad validation split, searched by NumPy cosine similarity. The starter of the lab task
# that follows loads the cross-encoder cross-encoder/ms-marco-MiniLM-L6-v2 of Example 26-16.
# The constructed corpora elsewhere in the chapter are labeled illustrations; Section 26.3 uses
# synthetic vectors for the approximate nearest-neighbor mechanic.

# %%
import warnings; warnings.filterwarnings("ignore")
import torch, numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
etok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
emodel = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
def embed(texts):
    b = etok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad(): out = emodel(**b).last_hidden_state
    m = b.attention_mask.unsqueeze(-1).float()
    return torch.nn.functional.normalize((out * m).sum(1) / m.sum(1), dim=1).numpy()
llm = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct",
               device="cpu", dtype=torch.float32)             # the CPU and 32-bit floats, as in Example 25-1

from datasets import load_dataset
np.random.seed(0); torch.manual_seed(0)

# REAL document store: SQuAD validation Wikipedia paragraphs, deterministic
sq = load_dataset("rajpurkar/squad", split="validation").select(range(400))
docs, seen = [], set()
for row in sq:
    if row["context"] not in seen:
        seen.add(row["context"]); docs.append(row["context"])
    if len(docs) >= 40: break                      # a ceiling; the first 400 rows yield 22 unique paragraphs
chunks = []                                        # sentence-chunk the real paragraphs
for d in docs:
    for s in d.replace("\n", " ").split(". "):
        s = s.strip()
        if len(s) > 15:
            chunks.append(s.rstrip(".") + ".")
index = embed(chunks)
print(f"ingested {len(docs)} real Wikipedia paragraphs -> {len(chunks)} chunks -> index {index.shape}")

def answer(question, k=3):
    top = np.argsort(index @ embed([question])[0])[::-1][:k]
    context = "\n".join(f"- {chunks[i]}" for i in top)
    prompt = (f"Context:\n{context}\n\nAnswer the question using ONLY the context above. "
              f"If the answer is not in the context, reply exactly 'I don't know from the documents.'\n"
              f"Question: {question}")
    out = llm([{"role": "user", "content": prompt}], max_new_tokens=40,
              do_sample=False)[0]["generated_text"][-1]["content"].strip()
    return out, [int(i) for i in top]

questions = [
    "Which NFL team represented the AFC at Super Bowl 50?",   # covered -> correct
    "What is the price of a Tesla Model 3?",                  # NOT in the corpus -> refuse
    "Where did Super Bowl 50 take place?",                    # covered, but the model still slips
]
for q in questions:
    a, used = answer(q)
    print(f"\nQ: {q}\n   retrieved chunks {used}\n   A: {a}")

print("\nAll chat-with-docs lab checks ran.")

# %% [markdown]
# ## 26.9 Your turn: rerank the lab's top 10 chunks with the cross-encoder (starter)
#
# The starter continues from Example 26-27 and prints 87, the number of chunks in the lab's store; the full solution is in Appendix H.

# %%
from transformers import AutoModelForSequenceClassification   # continuing with Example 26-27's store
ce_tok = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2")
ce = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2").eval()

def rerank_top3(question, n=10):
    # take the lab's top n chunks by cosine, score every (question, chunk) pair with the
    # cross-encoder, and return the ids of the three chunks it scores highest
    ...

print(len(chunks))


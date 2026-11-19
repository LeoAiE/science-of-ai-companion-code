"""Section 16.10.1: fine-tune distilbert-base-uncased on all of Rotten Tomatoes with the Transformers Trainer.
Run from this folder. Two epochs over the 8,530 training sentences take tens of minutes on the reference CPU,
so the run lives here rather than in the walkthrough. torch.set_num_threads(2) pins the thread count.
"""
# %% [markdown]
# ## Example 16-17. Loading and tokenizing Rotten Tomatoes for the Trainer

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
import os, warnings; warnings.filterwarnings("ignore")
os.environ["HF_HUB_VERBOSITY"] = "error"              # keep the Hub's notices out of the printed output
import numpy as np, torch
from datasets import load_dataset, disable_progress_bars
from transformers import AutoTokenizer, logging as hf_logging
disable_progress_bars(); hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
torch.set_num_threads(2)                              # the last digits of a CPU fine-tune depend on it

rt = load_dataset("cornell-movie-review-data/rotten_tomatoes")   # Pang and Lee (2005); label 1 = positive
tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
enc = rt.map(lambda batch: tok(batch["text"], truncation=True),  # ids and attention mask, no padding yet
             batched=True, remove_columns=["text"])
lengths = np.array([len(ids) for ids in enc["train"]["input_ids"]])
print({split: enc[split].num_rows for split in enc})
print(f"labels {rt['train'].features['label'].names}; positive share of train "
      f"{np.mean(list(rt['train']['label'])):.3f}")
print(f"tokens per training sentence, [CLS] and [SEP] included: mean {lengths.mean():.1f}, "
      f"longest {lengths.max()}")

# %% [markdown]
# ## Example 16-18. Fine-tuning DistilBERT on all 8,530 training sentences with the Trainer

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
import math, tempfile
from transformers import (AutoModelForSequenceClassification, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)
from transformers.trainer_callback import PrinterCallback

set_seed(0)                                           # before from_pretrained: it draws the new head's weights
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2,
    id2label={0: "neg", 1: "pos"}, label2id={"neg": 0, "pos": 1})
n_all = sum(p.numel() for p in model.parameters())
n_new = sum(p.numel() for name, p in model.named_parameters() if not name.startswith("distilbert."))
print(f"{n_all:,} parameters; {n_new:,} of them in the new classification head")

def accuracy(eval_pred):                              # one metric needs no extra library
    logits, labels = eval_pred
    return {"accuracy": float((logits.argmax(-1) == labels).mean())}

args = TrainingArguments(
    output_dir=tempfile.mkdtemp(),                    # nothing is saved, but the Trainer wants a folder
    num_train_epochs=2, learning_rate=2e-5, weight_decay=0.01,   # the 15.6 loop's epochs, rate, and decay
    per_device_train_batch_size=16, per_device_eval_batch_size=64,
    eval_strategy="epoch", save_strategy="no", logging_steps=100, logging_first_step=True,
    seed=0, disable_tqdm=True, report_to="none", use_cpu=True)
trainer = Trainer(model=model, args=args, train_dataset=enc["train"], eval_dataset=enc["validation"],
                  data_collator=DataCollatorWithPadding(tok),   # pads each batch to its own longest sentence
                  compute_metrics=accuracy)
trainer.remove_callback(PrinterCallback)              # print one line per log entry, not every log dict
result = trainer.train()

opt = trainer.optimizer.optimizer                     # the torch optimizer inside Accelerate's wrapper
print(f"{result.global_step} steps = 2 epochs x {math.ceil(enc['train'].num_rows / 16)} batches of 16; "
      f"optimizer {type(opt).__name__}, prepared by {type(trainer.accelerator).__name__}")
print("weight decay per parameter group:", [(g["weight_decay"], len(g["params"])) for g in opt.param_groups])
for log in trainer.state.log_history:
    if "loss" in log:
        print(f"  step {log['step']:4d}: training loss {log['loss']:.4f}, learning rate {log['learning_rate']:.3e}")
    elif "eval_accuracy" in log:
        print(f"  end of epoch {log['epoch']:.0f}: validation loss {log['eval_loss']:.4f}, "
              f"accuracy {log['eval_accuracy']:.4f}")
print(f"training took {result.metrics['train_runtime'] / 60:.1f} min")

# %% [markdown]
# ## Example 16-19. Scoring the fine-tuned model once on the test split

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
from transformers import pipeline

test = trainer.evaluate(enc["test"], metric_key_prefix="test")      # the test split, scored once
n_test = enc["test"].num_rows
print(f"test accuracy on all {n_test:,} sentences: {test['test_accuracy']:.4f} "
      f"({round(test['test_accuracy'] * n_test)} correct)")
lab = enc["test"].shuffle(seed=0).select(range(400))                # the 400 test sentences of the 15.6 lab
print(f"on the 400 test sentences of the 15.6 lab: {trainer.evaluate(lab)['eval_accuracy']:.4f}")
clf = pipeline("sentiment-analysis", model=trainer.model, tokenizer=tok, device="cpu")
for txt in ["This book is wonderful.", "That was a complete waste of time."]:
    r = clf(txt)[0]
    print(f"  {txt!r} -> {r['label']} ({r['score']:.2f})")

# %% [markdown]
# ## Experiment: the same Trainer on the 800 training sentences of the 15.6 lab
#
# The experiment paragraph of Section 16.10.1 asks whether the gain over the 15.6 fine-tune comes from the data or
# from the loop. This cell keeps every setting of Example 16-18, including the seed set before
# `from_pretrained`, and trains on the 800 rows the lab drew, `enc["train"].shuffle(seed=0).select(range(800))`.
# It then scores the full test split and the 400 test sentences of the lab, as Example 16-19 does.

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
set_seed(0)                                           # the same seed before from_pretrained as the full run
model_800 = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2,
    id2label={0: "neg", 1: "pos"}, label2id={"neg": 0, "pos": 1})
trainer_800 = Trainer(model=model_800, args=args,
                      train_dataset=enc["train"].shuffle(seed=0).select(range(800)),   # the 15.6 lab's rows
                      eval_dataset=enc["validation"], data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=accuracy)
trainer_800.remove_callback(PrinterCallback)
result_800 = trainer_800.train()
print(f"{result_800.global_step} steps = 2 epochs x {math.ceil(800 / 16)} batches of 16")
test_800 = trainer_800.evaluate(enc["test"], metric_key_prefix="test")
print(f"test accuracy on all {n_test:,} sentences: {test_800['test_accuracy']:.4f} "
      f"({round(test_800['test_accuracy'] * n_test)} correct)")
print(f"on the 400 test sentences of the 15.6 lab: {trainer_800.evaluate(lab)['eval_accuracy']:.4f}")

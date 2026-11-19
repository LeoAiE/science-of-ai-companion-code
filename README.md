# Companion Code

Runnable, notebook-style code for *The AI Practitioner's Handbook: From Math to Agents*, organized one folder per chapter. Every code example in the book lives here as code you can actually run, alongside extra examples for practice.

## Structure

```
code/
├── ch01_python/           # Chapter 1: language, DSA, the library ecosystem, Hugging Face
├── ch02_mathematics/      # Chapter 2: linear algebra, calculus, probability, ...
├── ...                    # one folder per chapter, added as the book is written
├── requirements.txt
└── README.md
```

## Format: notebook-style scripts

Each file is a plain Python script written in the **percent** notebook format. Cells are marked with `# %%` for code and `# %% [markdown]` for prose. This format is deliberately dual-purpose:

- **As a notebook:** open the file in VS Code or Jupyter and the `# %%` cells render as runnable notebook cells with Run buttons, exactly like a `.ipynb`.
- **As a script:** run the whole file from a terminal, for example `python ch01_python/02_library_ecosystem.py`.

We use this format because it is both notebook-friendly for exploration and a normal `.py` file that can be version-controlled cleanly and tested by running it.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Plots are saved as `.png` files next to the script that makes them, so the scripts run headlessly; inside a notebook the same figures display inline.

## Tested with

Python 3.13.14 (conda env aibook), with numpy 2.5.4, pandas 2.3.3, scipy 1.18.1, matplotlib 3.11.2, scikit-learn 1.9.1, torch 2.15.0, torchvision 0.29.2, transformers 5.22.1, and datasets 5.1.0 (verified 2026-11-18).

The Hugging Face examples download small models and datasets from the Hub the first time they run, so they need an internet connection on first use and then cache locally.

# %% [markdown]
# # Section 21.7 Lab: spectrogram CNN and Whisper (ESC-50)
#
# The code of Example 21-26 in Book.md: real audio (ESC-50 clips and LibriSpeech utterances) and a real model (whisper-tiny.en).

# %%
import warnings; warnings.filterwarnings("ignore")
import io
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, spectrogram
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Audio
import torch
import torch.nn as nn
np.random.seed(0); torch.manual_seed(0)
sr = 16000                                                   # we resample every clip to 16 kHz

# PART 1: real ESC-50 environmental sounds -> log-mel spectrograms -> a small CNN
CLASSES = ["dog", "rain", "crying_baby", "siren", "clock_tick"]
lab = {c: i for i, c in enumerate(CLASSES)}
esc = (load_dataset("ashraq/esc50", split="train")
       .cast_column("audio", Audio(decode=False))           # decode bytes ourselves (no torchcodec)
       .filter(lambda r: r["category"] in lab))             # keep 5 categories, ~40 clips each

def mel_filterbank(n_mels, n_fft, sr):
    hz2mel = lambda h: 2595 * np.log10(1 + h / 700); mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    fr = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2))
    fb = np.zeros((n_mels, len(fr)))
    for i in range(n_mels):
        lo, mid, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((fr - lo) / (mid - lo + 1e-9), (hi - fr) / (hi - mid + 1e-9)), 0, None)
    return fb

FB = mel_filterbank(64, 1024, sr)
def logmel(clip, frames=160):
    _, _, Sxx = spectrogram(clip, fs=sr, nperseg=1024, noverlap=512)  # power spectrogram (freq, time)
    M = np.log(FB @ Sxx + 1e-9)                                       # mel filterbank -> log
    if M.shape[1] >= frames:
        M = M[:, :frames]
    else:
        # pad with this clip's own floor, not with 0, which is louder than every real cell
        M = np.pad(M, ((0, 0), (0, frames - M.shape[1])), constant_values=M.min())
    return (M - M.mean()) / (M.std() + 1e-9)                          # per-clip standardize

specs, y = [], []
for row in esc:
    wav, srate = sf.read(io.BytesIO(row["audio"]["bytes"]))           # decode the raw .wav bytes
    if wav.ndim > 1: wav = wav.mean(axis=1)                           # to mono
    wav = resample_poly(wav, sr, srate)                              # 44.1 kHz -> 16 kHz
    specs.append(logmel(wav.astype(np.float32))); y.append(lab[row["category"]])
specs = np.stack(specs)[:, None, :, :]; y = np.array(y)
print(f"PART 1: {len(y)} real ESC-50 clips, {len(CLASSES)} classes {CLASSES}")
print(f"        each a {specs.shape[2]}x{specs.shape[3]} log-mel spectrogram (an image)")

Xtr, Xte, ytr, yte = train_test_split(specs, y, test_size=0.3, random_state=0, stratify=y)
Xtr = torch.tensor(Xtr, dtype=torch.float32); ytr = torch.tensor(ytr)
Xte = torch.tensor(Xte, dtype=torch.float32); yte = torch.tensor(yte)

class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.LazyLinear(32), nn.ReLU(),
                                  nn.Linear(32, len(CLASSES)))
    def forward(self, x): return self.head(self.conv(x))

net = AudioCNN(); opt = torch.optim.Adam(net.parameters(), lr=0.003, weight_decay=1e-4)
for ep in range(30):
    net.train(); p = torch.randperm(len(Xtr))
    for b in range(0, len(p), 16):
        idx = p[b:b + 16]
        loss = nn.functional.cross_entropy(net(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
net.eval()
with torch.no_grad():
    acc = (net(Xte).argmax(1) == yte).float().mean().item()
    trn = (net(Xtr).argmax(1) == ytr).float().mean().item()
print(f"  spectrogram-CNN train accuracy = {trn:.2f}, test accuracy = {acc:.2f}  (chance = {1/len(CLASSES):.2f})")

import re
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device="cpu")
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                  split="validation").cast_column("audio", Audio(decode=False))
def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()
def wer(ref, hyp):
    r, h = norm(ref), norm(hyp); d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0] = np.arange(len(r) + 1); d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (r[i-1] != h[j-1]))
    return d[len(r), len(h)] / len(r)
print("\nPART 2: Whisper-tiny transcription of 2 LibriSpeech clips:")
for k in range(2):
    a = ds[k]["audio"]; wav, srate = sf.read(io.BytesIO(a["bytes"])) if a.get("bytes") else sf.read(a["path"])
    hyp = asr({"raw": np.asarray(wav, np.float32), "sampling_rate": srate})["text"].strip()
    print(f"  clip {k}: WER {wer(ds[k]['text'], hyp):.2f}  ->  '{hyp[:60].strip()}...'")
print("\none lab, both modes: a CNN that CLASSIFIES real environmental sound from spectrograms,")
print("and a transformer that TRANSCRIBES real speech.")

print("\nAll audio-lab checks ran.")


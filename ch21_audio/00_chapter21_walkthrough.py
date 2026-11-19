"""Chapter 21: Audio & Speech AI

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from Book.md; keep the book as the source.
"""

# %% [markdown]
# ## MFCCs: the classic compact features

# %% [markdown]
# MFCCs fed the hidden Markov models of classical speech recognition, the next section's starting point, and they remain a strong lightweight feature for audio classification when deep models are overkill.

# %%
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sr = 16000                                              # sample rate: 16k samples per second
dur = 1.0
t = np.linspace(0, dur, int(sr * dur), endpoint=False)

wave = 0.6 * np.sin(2 * np.pi * 440 * t) + 0.4 * np.sin(2 * np.pi * 660 * t)
print(f"waveform: {len(wave)} samples at {sr} Hz = {dur} s of audio")
print(f"Nyquist frequency = sr/2 = {sr//2} Hz (the highest frequency this sample rate can represent)")

spectrum = np.abs(rfft(wave)); freqs = rfftfreq(len(wave), 1 / sr)
peaks = freqs[np.argsort(spectrum)[::-1][:2]]
print(f"\ntwo strongest frequencies found by the FFT = {sorted(int(p) for p in np.round(peaks))} Hz "
      f"(matches the 440 and 660 we put in)")

sr_low = 16000
high = np.sin(2 * np.pi * 14000 * np.linspace(0, 1, sr_low, endpoint=False))
sp = np.abs(rfft(high)); fr = rfftfreq(sr_low, 1 / sr_low)
alias = int(round(fr[np.argmax(sp)]))
print(f"\na 14000 Hz tone sampled at {sr_low} Hz appears at {alias} Hz (= {sr_low}-14000): that is ALIASING;")
print("real systems low-pass filter before sampling to prevent it.")

chirp = signal.chirp(t, f0=200, f1=4000, t1=dur, method="linear")   # frequency sweeps 200->4000 Hz
f, tt, Sxx = signal.spectrogram(chirp, fs=sr, nperseg=256)
peak_track = f[np.argmax(Sxx, axis=0)]                  # dominant frequency at each time slice
print(f"\nspectrogram shape = {Sxx.shape} (frequencies x time frames)")
print(f"dominant frequency rises over time: start ~{int(peak_track[1])} Hz -> end ~{int(peak_track[-2])} Hz")
print("the spectrogram turns 1D audio into a 2D image that CNNs (21.4) can classify.")

def hz_to_mel(h): return 2595 * np.log10(1 + h / 700)
def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)
def mfcc(frame, sr, n_filters=26, n_coeff=13):
    mag = np.abs(rfft(frame)) ** 2                      # power spectrum
    fr = rfftfreq(len(frame), 1 / sr)
    mel_pts = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_filters + 2))
    fb = np.zeros((n_filters, len(fr)))
    for i in range(n_filters):                          # triangular mel filters
        lo, mid, hi = mel_pts[i], mel_pts[i + 1], mel_pts[i + 2]
        fb[i] = np.clip(np.minimum((fr - lo) / (mid - lo + 1e-9), (hi - fr) / (hi - mid + 1e-9)), 0, None)
    energies = np.log(fb @ mag + 1e-9)                  # log mel energies
    from scipy.fft import dct
    return dct(energies, norm="ortho")[:n_coeff]        # DCT, keep first 13
coeffs = mfcc(wave[:512], sr)
print(f"\nMFCC of one frame: {np.round(coeffs, 2)}")
print(f"13 numbers summarize a 512-sample frame: this is the classic speech feature (21.2).")

fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(t[:400], wave[:400]); ax[0].set_title("waveform (two tones)"); ax[0].set_xlabel("seconds")
ax[1].pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-9), shading="auto")
ax[1].set_title("spectrogram (rising chirp)"); ax[1].set_ylabel("Hz"); ax[1].set_xlabel("seconds")
fig.tight_layout(); fig.savefig("01_sound_as_data.png", dpi=110)
print("\nsaved figure: 01_sound_as_data.png")
print("\nAll sound-as-data checks ran.")

# %% [markdown]
# ## The mel-spectrogram, and the same features on real speech

# %% [markdown]
# MFCCs are simply this log-mel spectrogram with one more DCT applied on top.

# %%
import numpy as np
import librosa

# a REAL speech clip: a LibriSpeech sentence bundled with librosa (downloads once, then caches)
y, sr = librosa.load(librosa.example("libri1"), sr=None)
print(f"real speech: {len(y)} samples at {sr} Hz = {len(y) / sr:.2f} s")
print(f"amplitude in float32, range {y.min():.2f}..{y.max():.2f} "
      f"(decoded floats; librosa's libri1 file is Ogg Vorbis, not 16-bit PCM)")

# the STFT window-length tradeoff: short window = sharp in time, blurry in frequency; long = reverse
for n_fft in [256, 2048]:
    hop = n_fft // 4
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    print(f"  n_fft={n_fft:4d}: STFT {S.shape} (freq x time), "
          f"freq resolution {sr / n_fft:5.1f} Hz, time step {1000 * hop / sr:4.1f} ms")

# log-mel spectrogram: THE modern CNN/transformer input. 64 mel bands, energies in decibels.
M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=512, htk=True)
Mdb = librosa.power_to_db(M, ref=np.max)
print(f"\nlog-mel spectrogram = {Mdb.shape} (mel bands x frames), {Mdb.min():.0f}..{Mdb.max():.0f} dB")

# MFCCs of the whole clip: 13 coefficients per frame, the classic compact speech feature
mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512, htk=True)
print(f"MFCC matrix = {mf.shape} (coeffs x frames)")
print(f"MFCC of the middle frame: {np.round(mf[:, mf.shape[1] // 2], 1)}")

# pitch: the fundamental frequency F0 (perceived pitch), distinct from formant resonances
f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=300, sr=sr)
f0v = f0[~np.isnan(f0)]
print(f"\nvoiced fundamental F0 median {np.median(f0v):.0f} Hz "
      f"({np.percentile(f0v, 10):.0f}..{np.percentile(f0v, 90):.0f} Hz over voiced frames): a low male voice")
print("All real-audio feature checks ran.")

# %% [markdown]
# ## 21.1.4 The discrete Fourier transform

# %%
import numpy as np
from scipy.fft import dct
import librosa

N = 8
n = np.arange(N)
k = n.reshape(-1, 1)
F = np.exp(-2j * np.pi * k * n / N)
x = np.sin(2 * np.pi * np.arange(N) / N)
print(f"DFT matrix vs np.fft.fft, max abs diff {np.max(np.abs(F @ x - np.fft.fft(x))):.1e}")

sr, N = 16000, 512
t = np.arange(N)
print("bin width Hz", sr / N)

def frac_outside(windowed):
    P = np.abs(np.fft.rfft(windowed)) ** 2
    peak = int(np.argmax(P))
    main = P[max(0, peak - 1):peak + 2].sum()
    return peak, 1 - main / P.sum()

def report(freq):
    tone = np.sin(2 * np.pi * freq * t / sr)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * t / N)
    rect = frac_outside(tone)
    han = frac_outside(tone * hann)
    print(f"  {freq} Hz: rect bin {rect[0]} outside {rect[1]:.4f}; "
          f"Hann bin {han[0]} outside {han[1]:.4f}")

report(440)
report(453.125)

y, sr_y = librosa.load(librosa.example("libri1"), sr=None)
for n_fft, hop in ((256, 64), (2048, 512)):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    print(f"STFT n_fft={n_fft} hop={hop} shape {S.shape}, 1+floor(L/H)={1 + len(y) // hop}")

M = librosa.feature.melspectrogram(y=y, sr=sr_y, n_mels=26, n_fft=2048, hop_length=512, htk=True)
E = np.log(M[:, M.shape[1] // 2] + 1e-9)
mels = len(E)
q = np.arange(mels)
j = np.arange(mels)
raw = np.cos(np.pi * q[:, None] * (j[None, :] + 0.5) / mels) @ E
scale = np.full(mels, np.sqrt(2 / mels))
scale[0] = np.sqrt(1 / mels)
ortho = raw * scale
ref = dct(E, type=2, norm="ortho")
print(f"type II DCT vs scipy ortho, max abs diff {np.max(np.abs(ortho - ref)):.1e}")

# %% [markdown]
# ## 21.1.5 Decibels and resampling

# %%
import io
import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset, Audio
from scipy.signal import resample_poly

y, sr_y = librosa.load(librosa.example("libri1"), sr=None)
speech = librosa.resample(y, orig_sr=sr_y, target_sr=16000)
ds = load_dataset("ashraq/esc50", split="train").cast_column("audio", Audio(decode=False))
for row in ds:
    if row["category"] == "rain":
        rain, rsr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        if getattr(rain, "ndim", 1) > 1:
            rain = rain.mean(axis=1)
        break
noise = resample_poly(rain, 16000, rsr)
L = min(len(speech), len(noise))
speech, noise = speech[:L], noise[:L]
Ps, Pn = np.mean(speech ** 2), np.mean(noise ** 2)
for target in (20, 10, 0):
    g = np.sqrt(Ps / (Pn * 10 ** (target / 10)))
    snr = 10 * np.log10(Ps / np.mean((g * noise) ** 2))
    print(f"target {target} dB, gain {g:.4f}, achieved {snr:.1f} dB")

clip = rain[:int(rsr)]
spec = np.abs(np.fft.rfft(clip)) ** 2
freqs = np.fft.rfftfreq(len(clip), 1 / rsr)
new_nyq = (rsr / 3) / 2
hi = spec[freqs > new_nyq].sum() / spec.sum()
naive, poly = clip[::3], resample_poly(clip, 1, 3)
print(f"rain {rsr} Hz, power above {new_nyq:.0f} Hz: {hi:.4f}")
print(f"naive length {len(naive)}, polyphase length {len(poly)}")

# %% [markdown]
# ## 21.1.6 Delta features

# %%
import numpy as np
import librosa

y, sr = librosa.load(librosa.example("libri1"), sr=None)
mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512, htk=True)
d = librosa.feature.delta(mf, width=5)
frame, Nwin = 100, 2
num = sum(n * (mf[:, frame + n] - mf[:, frame - n]) for n in range(1, Nwin + 1))
den = 2 * sum(n ** 2 for n in range(1, Nwin + 1))
print(f"delta vs N=2 formula, max abs {np.max(np.abs(d[:, frame] - num / den)):.1e}")
stacked = np.vstack([mf, d, librosa.feature.delta(mf, order=2, width=5)])
print("13 + delta + double delta shape", stacked.shape)

# %% [markdown]
# ## Whisper

# %% [markdown]
# Nothing about the architecture changes between these jobs; only the prompt tokens do, exactly the prompt-conditioning idea from the transformer decoder of Chapter 16.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np

def ctc_collapse(frames):
    out = []
    for ch in frames:
        if not out or ch != out[-1]: out.append(ch)     # 1. merge consecutive repeats
    return "".join(c for c in out if c != "_")           # 2. drop the blanks
frame_seq = "hh_eee_lll_lloo"                            # what the net emitted, one symbol per frame
print(f"frame-level emission : {frame_seq}")
print(f"CTC-decoded text     : '{ctc_collapse(frame_seq)}'")
print("repeats merge and blanks vanish, so the 15-frame emission becomes the word 'hello'.")
print("the blank lets the net REPEAT a letter (e.g. 'll') by separating them with a blank.")

import io
import soundfile as sf
from transformers import pipeline
from datasets import load_dataset, Audio

asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device="cpu")
# decode=False avoids datasets' torchcodec dependency; we read the audio with soundfile ourselves
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                  split="validation").cast_column("audio", Audio(decode=False))
def load_wav(a):
    if a.get("bytes"): wav, srate = sf.read(io.BytesIO(a["bytes"]))
    else: wav, srate = sf.read(a["path"])
    return np.asarray(wav, dtype=np.float32), srate

import re
def norm(s):                                             # standard WER normalization: lowercase, strip punctuation
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()
def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    # word-level edit distance (Levenshtein)
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0] = np.arange(len(r) + 1); d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (r[i-1] != h[j-1]))
    return d[len(r), len(h)] / len(r)

print("\nWhisper-tiny transcription vs reference on 3 LibriSpeech clips:")
wers = []
for k in range(3):
    sample = ds[k]
    wav, srate = load_wav(sample["audio"])
    hyp = asr({"raw": wav, "sampling_rate": srate})["text"].strip()
    ref = sample["text"]
    w = wer(ref, hyp); wers.append(w)
    print(f"  clip {k}: WER {w:.2f}")
    print(f"    ref : {ref[:70].lower()}...")
    print(f"    hyp : {hyp[:70].lower()}...")
print(f"\nmean word error rate = {np.mean(wers):.2f} (after normalizing case/punctuation; 0 = perfect).")
print("remaining errors are mostly 'mr.' vs 'mister' style normalization, not real mistakes; the")
print("TINY model already reads clean speech well, and larger Whisper models near human transcribers.")

print("\nAll speech-recognition checks ran.")

# %% [markdown]
# ## 21.2.2 Phonemes and pronunciation lexicons

# %% [markdown]
# Example 21-4. A pronunciation lexicon: the CMU Pronouncing Dictionary, its homophones, and a phoneme error rate.

# %%
import re
import numpy as np
import nltk
nltk.download("cmudict", quiet=True)                  # the CMU Pronouncing Dictionary, fetched once into ~/nltk_data
from nltk.corpus import cmudict

lex = cmudict.dict()                                  # word -> list of pronunciations, each a list of ARPAbet symbols
print(f"CMUdict in NLTK: {len(lex)} words, {sum(len(p) for p in lex.values())} pronunciations, "
      f"{sum(len(p) > 1 for p in lex.values())} words with more than one")
phonemes = sorted({re.sub(r"\d", "", s) for prons in lex.values() for pr in prons for s in pr})
print(f"{len(phonemes)} phonemes once the stress digits 0, 1 and 2 are removed")
for w in ["to", "too", "two", "read", "tomato"]:
    print(f"  {w:7s}", [" ".join(p) for p in lex[w]])

by_sound = {}                                         # invert the lexicon: pronunciation -> words
for w, prons in lex.items():
    for pr in prons:
        by_sound.setdefault(" ".join(pr), []).append(w)
groups = [ws for ws in by_sound.values() if len(ws) > 1]
print(f"homophone groups {len(groups)}; T UW1 is {by_sound['T UW1']}")
print(f"the largest group has {max(map(len, groups))} words: {max(groups, key=len)}")

def edit_distance(ref, hyp):                          # Example 21-3's recursion, over any two symbol lists
    d = np.zeros((len(ref) + 1, len(hyp) + 1), int)
    d[:, 0] = np.arange(len(ref) + 1); d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (ref[i-1] != hyp[j-1]))
    return int(d[len(ref), len(hyp)])

ref, hyp = lex["tomato"]
e = edit_distance(ref, hyp)
print(f"PER of [{' '.join(hyp)}] against [{' '.join(ref)}] = {e}/{len(ref)} = {e / len(ref):.3f}")
print("PER of read's two pronunciations =", round(edit_distance(*lex["read"]) / len(lex["read"][0]), 3))

sentence = "The spectrogram turns sound into an image."      # the sentence MMS-TTS speaks in Example 21-8
for w in re.findall(r"[a-z']+", sentence.lower()):
    print(f"  {w:12s}", " ".join(lex[w][0]) if w in lex else None)

# %% [markdown]
# ## 21.2.3 wav2vec 2.0

# %%
import math
import torch
log_probs = torch.log(torch.tensor([[[0.4, 0.6]], [[0.4, 0.6]]]))   # (frames T = 2, batch 1, symbols); index 0 is the blank
ctc = torch.nn.CTCLoss(blank=0, reduction="sum")
loss = ctc(log_probs, torch.tensor([[1]]), torch.tensor([2]), torch.tensor([1]))   # target "a", 2 frames, 1 label
print(f"CTCLoss on the hand case of Section 21.2 = {loss.item():.4f};  -ln 0.84 = {-math.log(0.84):.4f}")

# %% [markdown]
# Example 21-5. wav2vec 2.0 transcribing the clips of Example 21-3, decoded with Example 21-3's CTC collapse.

# %%
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
torch.manual_seed(0)                                       # seed before loading, as every model listing here does
torch.set_num_threads(2)
w2v_id = "facebook/wav2vec2-base-960h"
w2v_proc = Wav2Vec2Processor.from_pretrained(w2v_id)
w2v = Wav2Vec2ForCTC.from_pretrained(w2v_id).eval()
cfg, tk = w2v.config, w2v_proc.tokenizer
print(f"{w2v_id}: {sum(p.numel() for p in w2v.parameters())} parameters, {cfg.vocab_size} output symbols, "
      f"blank {tk.pad_token} (id {tk.pad_token_id}), word boundary '{tk.word_delimiter_token}'")
hop, field = 1, 1
for kernel, stride in zip(cfg.conv_kernel, cfg.conv_stride):   # the convolution stack that reads the raw waveform
    field += (kernel - 1) * hop                            # receptive field grows by (kernel - 1) input hops
    hop *= stride                                          # the hop multiplies by each stride
print(f"kernels {cfg.conv_kernel}, strides {cfg.conv_stride}: hop {hop} samples = {1000 * hop / 16000:.0f} ms, "
      f"receptive field {field} samples = {1000 * field / 16000:.0f} ms")
print(f"pretraining loss settings: {cfg.num_negatives} distractors, temperature {cfg.contrastive_logits_temperature}, "
      f"{cfg.num_codevector_groups} codebooks of {cfg.num_codevectors_per_group} entries")

print("\nwav2vec 2.0 on the three clips of Example 21-3 (Whisper's WER from Example 21-3 beside it):")
for k in range(3):
    wav, srate = load_wav(ds[k]["audio"])
    with torch.no_grad():
        logits = w2v(w2v_proc(wav, sampling_rate=srate, return_tensors="pt").input_values).logits[0]
    toks = tk.convert_ids_to_tokens(logits.argmax(-1).tolist())     # greedy: the best symbol in every 20 ms frame
    frames = "".join("_" if t == tk.pad_token else t for t in toks)  # write the blank as Example 21-3's underscore
    hyp = ctc_collapse(frames).replace("|", " ").strip()
    same = hyp == w2v_proc.batch_decode(logits.argmax(-1)[None])[0].strip()
    start = next(i for i, c in enumerate(frames) if c != "_")
    print(f"  clip {k}: {len(wav)} samples, {len(frames)} frames, 1 + (N - {field}) // {hop} = {1 + (len(wav) - field) // hop}; "
          f"same as the processor's decode: {same}")
    print(f"    frames {start} to {start + 59}: {frames[start:start + 60]}")
    print(f"    hyp: {hyp[:70].lower()}...")
    print(f"    WER wav2vec 2.0 {wer(ds[k]['text'], hyp):.3f}   whisper-tiny.en {wers[k]:.3f}")

# %% [markdown]
# ## 21.2.4 Decoding with a language model

# %% [markdown]
# Example 21-6. CTC prefix beam search with a character language model, tuned on half of the clips and tested on the other half.

# %%
import collections
import math
import re
import nltk

def prefix_beam_search(logp, syms, lm=None, lam=0.0, beta=0.0, beam=16, blank=0):
    """CTC prefix beam search over frame log probabilities logp (T x V). Every prefix keeps two log
    probabilities: of the paths that spell it and end in a blank, and of those that end in its last symbol."""
    def lse(a, b):                                          # log(e^a + e^b), safe when either is -inf
        if a == -math.inf: return b
        if b == -math.inf: return a
        return max(a, b) + math.log1p(math.exp(-abs(a - b)))
    index = {c: i for i, c in enumerate(syms)}
    beams = {"": (0.0, -math.inf)}
    for row in logp.tolist():
        nxt = collections.defaultdict(lambda: (-math.inf, -math.inf))
        for pre, (pb, pnb) in beams.items():
            b, nb = nxt[pre]
            nxt[pre] = (lse(b, lse(pb, pnb) + row[blank]), nb)          # a blank: the text stays pre
            if pre:                                                      # the last symbol again: it merges
                b, nb = nxt[pre]
                nxt[pre] = (b, lse(nb, pnb + row[index[pre[-1]]]))
            for i, c in enumerate(syms):
                if i == blank or len(c) > 1 or row[i] < -12:             # skip <s>, </s>, <unk>, hopeless symbols
                    continue
                bonus = lam * lm(pre, c) + beta if lm else 0.0           # shallow fusion, once per new symbol
                via = pb if pre and c == pre[-1] else lse(pb, pnb)       # a doubled letter needs a blank between
                b, nb = nxt[pre + c]
                nxt[pre + c] = (b, lse(nb, via + row[i] + bonus))
        beams = dict(sorted(nxt.items(), key=lambda kv: -lse(*kv[1]))[:beam])
    best = max(beams, key=lambda p: lse(*beams[p]))
    return best, lse(*beams[best])

toy = np.log(np.array([[0.6, 0.4], [0.6, 0.4]]))           # the T = 2 case: blank 0.6, "a" 0.4 at both frames
greedy_toy = ctc_collapse("".join("_a"[i] for i in toy.argmax(1)))
text_toy, score_toy = prefix_beam_search(toy, ["_", "a"])
print(f"toy: greedy decode '{greedy_toy}', prefix beam search '{text_toy}' with p = {math.exp(score_toy):.2f}")

nltk.download("gutenberg", quiet=True)                     # 18 public domain books from Project Gutenberg
from nltk.corpus import gutenberg
corpus = re.sub(r" +", " ", re.sub(r"[^A-Z' ]+", " ", gutenberg.raw().upper())).replace(" ", "|")
ORDER, K = 5, 0.1                                          # a 5-gram over 28 symbols, add-0.1 smoothing
tables = [collections.defaultdict(collections.Counter) for _ in range(ORDER)]
for gram, n in collections.Counter(corpus[i:i + ORDER] for i in range(len(corpus) - ORDER + 1)).items():
    tables[ORDER - 1][gram[:-1]][gram[-1]] += n
for L in range(ORDER - 1, 0, -1):                          # shorter contexts: sum the longer ones that end in them
    for ctx, nexts in tables[L].items():
        tables[L - 1][ctx[1:]].update(nexts)

def char_lm(prefix, c):
    """log p(c | the last ORDER - 1 symbols), backing off to the longest context the corpus contains."""
    ctx = ("|" + prefix)[-(ORDER - 1):]                    # an utterance starts after a word boundary
    for L in range(len(ctx), -1, -1):
        nexts = tables[L].get(ctx[len(ctx) - L:])
        if nexts:
            return math.log((nexts[c] + K) / (sum(nexts.values()) + 28 * K))
    return math.log(1 / 28)

refs = [ds[k]["text"].replace(" ", "|") for k in range(ds.num_rows)]
bits = -sum(char_lm(r[:i], r[i]) for r in refs for i in range(len(r))) / sum(map(len, refs)) / math.log(2)
print(f"LM text: {len(gutenberg.fileids())} books, {len(corpus)} symbols; on the {len(refs)} reference "
      f"transcripts {bits:.2f} bits per symbol (uniform over 28 symbols: {math.log2(28):.2f})")
words = corpus.split("|")
runs = {tuple(words[i:i + 6]) for i in range(len(words) - 5)}    # every six word run of the LM text
shared = sum(any(tuple(r.split("|")[i:i + 6]) in runs for i in range(len(r.split("|")) - 5)) for r in refs)
print(f"reference transcripts that share a six word run with the LM text: {shared} of {len(refs)}")

logps = []                                                 # the acoustic model once, over all clips
for k in range(ds.num_rows):
    wav, srate = load_wav(ds[k]["audio"])
    with torch.no_grad():
        logits = w2v(w2v_proc(wav, sampling_rate=srate, return_tensors="pt").input_values).logits[0]
    logps.append(torch.log_softmax(logits, -1).numpy())
print(f"{len(logps)} clips, {sum(map(len, logps))} frames = {sum(map(len, logps)) * 0.02:.1f} s of audio")

def corpus_wer(clips, decode):                             # total edits over total reference words
    errors = words = 0
    for k in clips:
        r, h = norm(ds[k]["text"]), norm(decode(logps[k]).replace("|", " "))
        errors += edit_distance(r, h); words += len(r)
    return errors / words, errors, words

syms = w2v_proc.tokenizer.convert_ids_to_tokens(list(range(w2v.config.vocab_size)))
greedy = lambda lp: w2v_proc.batch_decode(torch.tensor(lp.argmax(-1))[None])[0]
tune, test = range(0, 36), range(36, ds.num_rows)
print("\ngrid on the tuning clips 0 to 35 (beam 16):")
grid = {}
for lam in (0.25, 0.5, 1.0):
    for beta in (0.0, 1.0, 2.0):
        grid[lam, beta] = corpus_wer(tune, lambda lp: prefix_beam_search(lp, syms, char_lm, lam, beta)[0])
        print(f"  lambda {lam:4.2f}, beta {beta:3.1f}: WER {grid[lam, beta][0]:.4f} ({grid[lam, beta][1]} errors)")
lam, beta = min(grid, key=lambda g: grid[g][1])
print(f"chosen on the tuning clips: lambda {lam}, beta {beta}")
for name, part in (("tuning clips 0 to 35", tune), (f"test clips 36 to {ds.num_rows - 1}", test)):
    g = corpus_wer(part, greedy)
    b = corpus_wer(part, lambda lp: prefix_beam_search(lp, syms)[0])
    f = corpus_wer(part, lambda lp: prefix_beam_search(lp, syms, char_lm, lam, beta)[0])
    print(f"{name} ({g[2]} words): greedy {g[0]:.4f} ({g[1]}), beam {b[0]:.4f} ({b[1]}), "
          f"beam + LM {f[0]:.4f} ({f[1]})")

import difflib
print("\nwhere the language model changed the test transcripts (beam -> beam + LM):")
for k in test:
    b = norm(prefix_beam_search(logps[k], syms)[0].replace("|", " "))
    f = norm(prefix_beam_search(logps[k], syms, char_lm, lam, beta)[0].replace("|", " "))
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, b, f).get_opcodes():
        if op != "equal":
            print(f"  clip {k}: '{' '.join(b[i1:i2])}' -> '{' '.join(f[j1:j2])}'")
for w in ("SOMEONE", "SOME|ONE", "BRION", "BRIAN"):
    print(f"'{w}' as a whole word in the LM text: {corpus.count('|' + w + '|')} times")

# %% [markdown]
# ## 21.2.5 Whisper on long recordings

# %% [markdown]
# Example 21-7. Whisper on 62 seconds of speech: sequential long-form decoding, 30-second chunks, and rain.

# %%
import time
import librosa
import torch
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

parts = [load_wav(ds[k]["audio"])[0] for k in range(5)]   # clips 0 to 4 of Example 21-3's dataset, back to back
long_wav = np.concatenate(parts)
long_ref = " ".join(ds[k]["text"] for k in range(5))
audio_s = len(long_wav) / 16000
print(f"clips 0 to 4 joined: {audio_s:.2f} s, {len(norm(long_ref))} reference words; "
      f"the clips meet at {np.round(np.cumsum([len(p) / 16000 for p in parts])[:-1], 2)} s")

whisper = {}
for name in ("openai/whisper-tiny.en", "openai/whisper-base.en"):
    torch.manual_seed(0)
    torch.set_num_threads(2)
    proc = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name).eval()
    whisper[name] = proc, model
    feats = proc(long_wav, sampling_rate=16000, return_tensors="pt", truncation=False,
                 padding="longest", return_attention_mask=True)       # keep every frame: no cut at 30 s
    t0 = time.perf_counter()
    with torch.no_grad():                                             # sequential long form: window by window
        out = model.generate(feats.input_features, attention_mask=feats.attention_mask,
                             return_timestamps=True, return_segments=True)
    rtf = (time.perf_counter() - t0) / audio_s
    text = proc.batch_decode(out["sequences"], skip_special_tokens=True)[0]
    segs = out["segments"][0]
    print(f"\n{name}: {sum(p.numel() for p in model.parameters())} parameters, features {tuple(feats.input_features.shape)}")
    print(f"  sequential long form: {len(segs)} segments, WER {wer(long_ref, text):.3f}, real time factor {rtf:.2g}")
    print("    segments (start, end) in s:", [(round(float(s["start"]), 2), round(float(s["end"]), 2)) for s in segs])
    print(f"    first segment:{proc.decode(segs[0]['tokens'], skip_special_tokens=True)}")
    chunker = pipeline("automatic-speech-recognition", model=model, tokenizer=proc.tokenizer,
                       feature_extractor=proc.feature_extractor, device="cpu")   # stay on the CPU
    t0 = time.perf_counter()
    res = chunker({"raw": long_wav, "sampling_rate": 16000}, chunk_length_s=30, return_timestamps=True)
    rtf = (time.perf_counter() - t0) / audio_s
    print(f"  pipeline in 30 s chunks: {len(res['chunks'])} pieces, WER {wer(long_ref, res['text']):.3f}, "
          f"real time factor {rtf:.2g}; first piece {res['chunks'][0]['timestamp']}")
try:
    chunker({"raw": long_wav, "sampling_rate": 16000})                # no chunks and no timestamps
except ValueError as err:
    print("\nwithout chunks or timestamps the pipeline refuses:", str(err).split(" which ")[0])

esc = load_dataset("ashraq/esc50", split="train").cast_column("audio", Audio(decode=False))
rain = esc.filter(lambda c: c == "rain", input_columns="category").select(range(5))
tiny_proc, tiny = whisper["openai/whisper-tiny.en"]
sot, nocap = tiny_proc.tokenizer.convert_tokens_to_ids(["<|startoftranscript|>", "<|nocaptions|>"])

def transcribe(w):                                        # whisper-tiny.en, greedy decoding, on the CPU
    f = tiny_proc(w, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        return tiny_proc.batch_decode(tiny.generate(f), skip_special_tokens=True)[0].strip()

def no_speech_prob(w):                                    # p(<|nocaptions|>) right after <|startoftranscript|>
    f = tiny_proc(w, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        logits = tiny(input_features=f, decoder_input_ids=torch.tensor([[sot]])).logits[0, -1]
    return torch.softmax(logits, -1)[nocap].item()

def gate_fraction(w):                                     # the energy gate of Example 21-22 (Section 21.6)
    rms = librosa.feature.rms(y=w, frame_length=400, hop_length=160)[0]
    db = 20 * np.log10(rms + 1e-9)
    return float(np.mean(db > db.max() - 30))

print("\nwhisper-tiny.en on five ESC-50 rain clips, then on three speech clips:")
for row in rain:
    w, s = sf.read(io.BytesIO(row["audio"]["bytes"]))
    w16 = resample_poly(w.mean(axis=1) if w.ndim > 1 else w, 16000, s).astype(np.float32)
    print(f"  {row['filename']}: gate keeps {gate_fraction(w16):.2f}, p(no speech) {no_speech_prob(w16):.2f}, "
          f"hears '{transcribe(w16)[:40]}'")
for k in range(3):
    print(f"  speech clip {k}: gate keeps {gate_fraction(parts[k]):.2f}, p(no speech) {no_speech_prob(parts[k]):.2f}, "
          f"hears '{transcribe(parts[k])[:40]}'")

# %% [markdown]
# ## Voice cloning

# %% [markdown]
# Consent, watermarking of synthetic audio, and deepfake detection are therefore central concerns. Section 21.3.4 marks synthetic speech with a simple keyed watermark and detects it.

# %%
import numpy as np
import librosa
import soundfile as sf

# a REAL instrument recording: a 5.33 s solo trumpet bundled with librosa (downloads once, caches)
y, sr = librosa.load(librosa.example("trumpet"), sr=None)
print(f"real audio: a solo trumpet, {len(y)} samples at {sr} Hz = {len(y) / sr:.2f} s")

S = np.abs(librosa.stft(y, n_fft=512, hop_length=128))     # magnitude only (phase thrown away)
print(f"magnitude spectrogram shape = {S.shape} (freq bins x time frames); phase is GONE")

def spectral_match(a, b):
    Sa = np.abs(librosa.stft(a, n_fft=512, hop_length=128))
    Sb = np.abs(librosa.stft(b, n_fft=512, hop_length=128))
    m = min(Sa.shape[1], Sb.shape[1])
    return np.corrcoef(Sa[:, :m].ravel(), Sb[:, :m].ravel())[0, 1]

print("\nGriffin-Lim reconstruction quality (spectrogram correlation with the original):")
for n_iter in [1, 8, 64]:
    recon = librosa.griffinlim(S, n_iter=n_iter, hop_length=128, rng=0)   # rng=0: seeded random start phases
    print(f"  {n_iter:3d} iterations: spectral match = {spectral_match(y, recon):.3f}")
best = librosa.griffinlim(S, n_iter=64, hop_length=128, rng=0)
sf.write("03_griffinlim_recon.wav", best, sr)
print("wrote 03_griffinlim_recon.wav: a real trumpet rebuilt from magnitude alone (listen to it).")

# REAL text-to-speech: MMS-TTS English, Meta's end-to-end VITS model (Hub card: 36.3M parameters, CC-BY-NC 4.0)
import torch
from transformers import VitsModel, AutoTokenizer, pipeline
tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
tts = VitsModel.from_pretrained("facebook/mms-tts-eng")
torch.manual_seed(0)                                       # VITS samples noise: seed it for a repeatable voice
text = "The spectrogram turns sound into an image."
with torch.no_grad():
    wav = tts(**tok(text, return_tensors="pt")).waveform[0].numpy()
sr_tts = tts.config.sampling_rate
sf.write("03_tts_synth.wav", wav, sr_tts)
print(f"\nreal TTS: VITS (facebook/mms-tts-eng) spoke {len(wav)} samples at {sr_tts} Hz "
      f"= {len(wav)/sr_tts:.2f} s -> 03_tts_synth.wav")

# close the loop: does an independent listener understand the synthetic speech?
asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device="cpu")
heard = asr({"raw": wav.astype(np.float32), "sampling_rate": sr_tts})["text"].strip()
print(f"whisper-tiny hears: '{heard}'")

print("\nAll TTS checks ran.")

# %% [markdown]
# ## 21.3.2 Inside the acoustic model and the vocoder

# %%
import torch
phonemes = torch.arange(16, dtype=torch.float32).reshape(4, 4)   # four phoneme vectors of four numbers each
D = torch.tensor([2, 2, 3, 1])                                  # predicted durations, in spectrogram frames
for alpha in (1.0, 1.3, 0.5):                                   # alpha > 1 slows the speech down
    d = torch.floor(D * alpha + 0.5).long()                     # round half up, as Ren et al. (2019) do
    frames = torch.repeat_interleave(phonemes, d, dim=0)        # phoneme i is copied d[i] times
    print(f"alpha {alpha}: durations {d.tolist()}, frames {tuple(frames.shape)}, "
          f"phoneme of each frame {(frames[:, 0] // 4).long().tolist()}")

# %% [markdown]
# Example 21-9. A HiFi-GAN vocoder and Griffin-Lim inverting the same real log-mel spectrogram.

# %%
from transformers import SpeechT5FeatureExtractor, SpeechT5HifiGan
torch.manual_seed(0)
torch.set_num_threads(2)
fe = SpeechT5FeatureExtractor.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval()
print(f"HiFi-GAN vocoder: {sum(p.numel() for p in vocoder.parameters())} parameters; features at {fe.sampling_rate} Hz, "
      f"{fe.num_mel_bins} mel bands from {fe.fmin} to {fe.fmax} Hz, window {fe.win_length} ms, hop {fe.hop_length} ms")

speech, _ = librosa.load(librosa.example("libri1"), sr=16000)
speech = speech[:int(7.96 * 16000)]                     # the first 7.96 s of libri1: two phrases of real speech
logmel = fe(audio_target=speech, sampling_rate=16000, return_tensors="pt").input_values[0]
print(f"log mel spectrogram {tuple(logmel.shape)} (frames x bands), values {logmel.min():.2f} to {logmel.max():.2f} (log10)")
with torch.no_grad():
    hifi = vocoder(logmel).numpy()                      # the neural vocoder: spectrogram in, waveform out
mag = librosa.feature.inverse.mel_to_stft((10.0 ** logmel.numpy()).T, sr=16000, n_fft=1024, power=1.0,
                                          fmin=fe.fmin, fmax=fe.fmax)   # undo the mel bands, then guess the phase
gl = librosa.griffinlim(mag, n_iter=32, hop_length=256, win_length=1024, rng=0)
sf.write("03_hifigan_libri1.wav", hifi, 16000)
sf.write("03_griffinlim_libri1.wav", gl, 16000)
print(f"waveforms: original {len(speech)}, HiFi-GAN {len(hifi)}, Griffin-Lim {len(gl)} samples")
print(f"spectral match with the original: HiFi-GAN {spectral_match(speech, hifi[:len(speech)]):.3f}, "
      f"Griffin-Lim {spectral_match(speech, gl[:len(speech)]):.3f}")
for name, audio in (("original", speech), ("HiFi-GAN", hifi), ("Griffin-Lim", gl)):
    print(f"  whisper-tiny.en hears the {name}: {transcribe(audio.astype(np.float32))}")

# %% [markdown]
# ## 21.3.3 Audio tokens: neural codecs and residual vector quantization

# %% [markdown]
# Example 21-10. Residual vector quantization from scratch, with k-means codebooks on libri1's log-mel frames.

# %%
from sklearn.cluster import KMeans
y22, sr22 = librosa.load(librosa.example("libri1"), sr=None)
Z = librosa.power_to_db(librosa.feature.melspectrogram(y=y22, sr=sr22, n_mels=64, n_fft=2048, hop_length=512,
                                                       htk=True), ref=np.max).T   # Example 21-2's frames, one row each
fit, held = Z[:320], Z[320:]                            # learn codebooks on the first half, measure on the second
print(f"{len(Z)} frames of 64 log mel values; codebooks fitted on {len(fit)}, measured on {len(held)}")
residual_fit, residual_held = fit.copy(), held.copy()
for stage in range(1, 5):
    km = KMeans(n_clusters=16, n_init=4, random_state=0).fit(residual_fit)     # 16 codewords: 4 bits per stage
    residual_fit -= km.cluster_centers_[km.predict(residual_fit)]
    residual_held -= km.cluster_centers_[km.predict(residual_held)]             # nearest codeword, subtract it
    print(f"  {stage} stage(s), {4 * stage:2d} bits per frame: mean squared error "
          f"{np.mean(residual_fit ** 2):6.2f} on the fitted frames, {np.mean(residual_held ** 2):6.2f} dB^2 held out")
print(f"variance of the held-out frames {np.var(held):.2f} dB^2 (the error of sending nothing but the mean)")

# %% [markdown]
# Example 21-11. EnCodec turning libri1 into audio tokens at three bitrates.

# %%
from transformers import AutoProcessor, EncodecModel
torch.manual_seed(0)
torch.set_num_threads(2)
codec = EncodecModel.from_pretrained("facebook/encodec_24khz").eval()
codec_proc = AutoProcessor.from_pretrained("facebook/encodec_24khz")
c = codec.config
frame_rate = c.sampling_rate / int(np.prod(c.upsampling_ratios))
print(f"EnCodec 24 kHz: {sum(p.numel() for p in codec.parameters())} parameters, upsampling ratios {c.upsampling_ratios}, "
      f"hop {int(np.prod(c.upsampling_ratios))} samples = {frame_rate:.0f} frames per second, "
      f"codebooks of {c.codebook_size} entries ({np.log2(c.codebook_size):.0f} bits each)")
y24 = librosa.resample(y22, orig_sr=sr22, target_sr=c.sampling_rate)          # libri1 at the codec's rate
inputs = codec_proc(raw_audio=y24, sampling_rate=c.sampling_rate, return_tensors="pt")
for kbps in (1.5, 6.0, 24.0):
    with torch.no_grad():
        enc = codec.encode(inputs["input_values"], inputs["padding_mask"], bandwidth=kbps)
        out = codec.decode(enc.audio_codes, enc.audio_scales, inputs["padding_mask"])[0][0, 0].numpy()
    n_q, steps = enc.audio_codes.shape[-2], enc.audio_codes.shape[-1]
    print(f"  {kbps:4.1f} kbps: codes {tuple(enc.audio_codes.shape)} = {n_q} codebooks x {steps} steps, "
          f"{frame_rate:.0f} x {n_q} x {np.log2(c.codebook_size):.0f} = {frame_rate * n_q * np.log2(c.codebook_size):.0f} bit/s, "
          f"spectral match {spectral_match(y24, out[:len(y24)]):.3f}")
print("first 8 steps of the first two codebooks:", enc.audio_codes[0, 0, :2, :8].tolist())

# %% [markdown]
# ## 21.3.4 Marking synthetic speech

# %% [markdown]
# Example 21-12. A keyed watermark on synthetic and on real speech, detected by correlation and scored by the equal error rate.

# %%
def key(seed, n):                                        # the secret: a seeded sequence of +1 and -1
    return np.random.default_rng(seed).choice([-1.0, 1.0], size=n)

def rho(signal, k):                                      # normalized correlation of a recording with a key
    return float(signal @ k / (np.linalg.norm(signal) * np.linalg.norm(k)))

def eer(genuine, impostor):                              # the threshold sweep of the speaker trials in 21.4.2
    genuine, impostor = np.array(genuine), np.array(impostor)
    ths = np.sort(np.concatenate([genuine, impostor]))
    far = np.array([(impostor >= t).mean() for t in ths])
    frr = np.array([(genuine < t).mean() for t in ths])
    i = np.argmin(np.abs(far - frr))
    return (far[i] + frr[i]) / 2

alpha = 0.005                                            # the mark's amplitude
def noisy(s, i):                                         # add noise 20 dB below the recording's power
    return s + np.random.default_rng(7000 + i).normal(0, np.sqrt(np.mean(s ** 2) / 100), len(s))
def through(rate):                                       # resample down to rate and back up to 16 kHz
    return lambda s, i: librosa.resample(librosa.resample(s, orig_sr=16000, target_sr=rate),
                                         orig_sr=rate, target_sr=16000)[:len(s)]
attacks = {"clean": lambda s, i: s, "noise at 20 dB SNR": noisy, "8 kHz and back": through(8000),
           "4 kHz and back": through(4000)}

x = wav.astype(np.float64)                               # Example 21-8's MMS-TTS sentence: synthetic speech
p = key(1234, len(x))
snr = 10 * np.log10(np.mean(x ** 2) / alpha ** 2)
print(f"MMS-TTS sentence: {len(x)} samples, mark SNR {snr:.1f} dB, so rho should be near "
      f"10^(-SNR/20) = {10 ** (-snr / 20):.4f}; chance level 1/sqrt(N) = {1 / np.sqrt(len(x)):.4f}")
for name, attack in attacks.items():
    print(f"  {name:19s}: rho marked {rho(attack(x + alpha * p, 0), p):.4f}, unmarked {rho(attack(x, 0), p):.4f}")

hosts = [load_wav(ds[k]["audio"])[0].astype(np.float64) for k in range(ds.num_rows)]   # 73 real LibriSpeech recordings
snrs = [10 * np.log10(np.mean(h ** 2) / alpha ** 2) for h in hosts]
print(f"\n{len(hosts)} real recordings, {min(map(len, hosts)) / 16000:.2f} to {max(map(len, hosts)) / 16000:.2f} s; "
      f"mark SNR {min(snrs):.1f} to {max(snrs):.1f} dB, median {np.median(snrs):.1f}")
for name, attack in attacks.items():
    genuine, impostor = [], []
    for i, h in enumerate(hosts):                        # every trial has its own recording and its own key
        k_marked, k_other = key(i, len(h)), key(1000 + i, len(h))
        genuine.append(rho(attack(h + alpha * k_marked, i), k_marked))
        impostor.append(rho(attack(h, i), k_other))
    print(f"  {name:19s}: marked median {np.median(genuine):.4f}, lowest {min(genuine):.4f}; "
          f"unmarked highest {max(impostor):.4f}; EER {eer(genuine, impostor):.2%}")

# %% [markdown]
# ## Pretrained audio models and speaker recognition

# %% [markdown]
# Example 21-13 runs the AudioSet-fine-tuned AST checkpoint MIT/ast-finetuned-audioset-10-10-0.4593, which tags a clip with AudioSet labels with no training on our data at all.

# %%
import io
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio
from scipy.signal import resample_poly
from transformers import pipeline

tagger = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593", device="cpu")
CLASSES = ["dog", "siren", "crying_baby"]
ds = (load_dataset("ashraq/esc50", split="train")
      .cast_column("audio", Audio(decode=False))
      .filter(lambda r: r["category"] in CLASSES))
picked = {}
for row in ds:                                            # keep the first clip of each category
    picked.setdefault(row["category"], row)
for cat in CLASSES:
    wav, srate = sf.read(io.BytesIO(picked[cat]["audio"]["bytes"]))
    if wav.ndim > 1: wav = wav.mean(axis=1)
    wav16 = resample_poly(wav, 16000, srate).astype(np.float32)   # AST expects 16 kHz input
    top = tagger({"raw": wav16, "sampling_rate": 16000}, top_k=3)
    print(f"  true '{cat}':  AST top-3 = " + ", ".join(f"{t['label']} {t['score']:.2f}" for t in top))

# %% [markdown]
# a same-speaker pair should score high, a different-speaker pair low.

# %%
import numpy as np
import librosa
clips, who = [], []
for spk, name in enumerate(["libri1", "libri2", "libri3"]):     # three real readers
    s, _ = librosa.load(librosa.example(name), sr=16000)
    step = int(1.5 * 16000)
    for k in range(len(s) // step):
        clips.append(s[k * step:(k + 1) * step]); who.append(spk)
who = np.array(who)
emb = np.array([librosa.feature.mfcc(y=c, sr=16000, n_mfcc=13).mean(axis=1) for c in clips])
emb = (emb - emb.mean(0)) / emb.std(0)                    # standardize each MFCC dimension
emb /= np.linalg.norm(emb, axis=1, keepdims=True)         # unit length: dot product = cosine

same, diff = [], []                                       # verification trials: every segment pair
for i in range(len(emb)):
    for j in range(i + 1, len(emb)):
        (same if who[i] == who[j] else diff).append(float(emb[i] @ emb[j]))
same, diff = np.array(same), np.array(diff)
print(f"trial pairs: {len(same)} same-speaker, {len(diff)} different-speaker")
print(f"mean cosine: same {same.mean():.2f}, different {diff.mean():.2f}")

ths = np.sort(np.concatenate([same, diff]))               # sweep the accept threshold
far = np.array([(diff >= t).mean() for t in ths])         # false accepts: impostors let in
frr = np.array([(same < t).mean() for t in ths])          # false rejects: genuine turned away
k = np.argmin(np.abs(far - frr))
print(f"EER = {(far[k] + frr[k]) / 2:.2%} at threshold {ths[k]:.2f} (FAR {far[k]:.2%}, FRR {frr[k]:.2%})")

# %% [markdown]
# The mechanics of the metric, though, are exactly these dozen lines.

# %%
import io
import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset, Audio
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
np.random.seed(0)

# Real environmental sounds: ESC-50, restricted to a 5-class subset (chance = 1/5)
CLASSES = ["dog", "rain", "crying_baby", "siren", "clock_tick"]
ds = load_dataset("ashraq/esc50", split="train")        # 2000 clips, 50 classes, 5 s each
ds = ds.cast_column("audio", Audio(decode=False))       # no torchcodec: keep raw bytes
ds = ds.filter(lambda r: r["category"] in CLASSES)      # -> 200 clips, 5 classes
label = {c: i for i, c in enumerate(CLASSES)}

def mfcc_features(y, sr):
    m = librosa.feature.mfcc(y=y.astype(np.float32), sr=sr, n_mfcc=13)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])   # 26-d mean+std summary

X, y = [], []
for row in ds:
    wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))     # decode the raw WAV bytes
    if wav.ndim > 1:
        wav = wav.mean(axis=1)                               # stereo -> mono
    X.append(mfcc_features(wav, sr)); y.append(label[row["category"]])
X = np.array(X); y = np.array(y)
print(f"ESC-50 subset: {len(X)} real clips, {len(CLASSES)} classes, {X.shape[1]}-d MFCC features (sr={sr} Hz)")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr)
print(f"\naudio classification test accuracy = {clf.score(Xte, yte):.2f}  (chance = {1/len(CLASSES):.2f})")
print("well above chance. The misses are specific confusions, not proof the classes are inseparable.")
cm = confusion_matrix(yte, clf.predict(Xte))             # rows: true class, columns: predicted class
print("confusion matrix, rows true and columns predicted, in the order", CLASSES)
print(cm)

E = normalize(StandardScaler().fit_transform(X))        # standardize, then unit-normalize
correct = 0
for i in range(len(E)):
    sims = E @ E[i]; sims[i] = -1                        # cosine similarity to all others
    correct += (y[int(np.argmax(sims))] == y[i])        # nearest neighbor's class
print(f"\nembedding retrieval: nearest neighbor shares the query's class {correct/len(E)*100:.0f}% of the time")
print(f"(chance {100/len(CLASSES):.0f}%). The same features drive both classification and 'find similar sounds'.")

print("\nAll audio-classification checks ran.")

# %% [markdown]
# ## SpecAugment: augmenting the spectrogram

# %% [markdown]
# It costs almost nothing to implement, a pair of array assignments on the existing mel matrix.

# %%
import numpy as np
import librosa
rng = np.random.default_rng(0)

y, sr = librosa.load(librosa.example("libri1"), sr=None)
M = librosa.power_to_db(librosa.feature.melspectrogram(
    y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=512, htk=True), ref=np.max)
aug = M.copy()
f0 = int(rng.integers(0, M.shape[0] - 8))               # frequency mask: erase 8 mel bands
aug[f0:f0 + 8, :] = M.min()
t0 = int(rng.integers(0, M.shape[1] - 40))              # time mask: erase 40 frames
aug[:, t0:t0 + 40] = M.min()
frac = (8 * M.shape[1] + 40 * M.shape[0] - 8 * 40) / M.size
print(f"SpecAugment on the {M.shape} log-mel image: masked mel bands {f0}-{f0 + 7} "
      f"and frames {t0}-{t0 + 39} ({frac:.0%} of the image erased)")

# %% [markdown]
# ## 21.4.4 Splits that keep a recording on one side

# %%
import io
import numpy as np
import soundfile as sf
import librosa
from collections import Counter
from datasets import load_dataset, Audio
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

CLASSES = ["dog", "rain", "crying_baby", "siren", "clock_tick"]
ds = (load_dataset("ashraq/esc50", split="train")
      .cast_column("audio", Audio(decode=False))
      .filter(lambda r: r["category"] in CLASSES))
label = {c: i for i, c in enumerate(CLASSES)}
src = Counter(ds["src_file"])
multi = sum(1 for v in src.values() if v > 1)
clips = sum(v for v in src.values() if v > 1)
print(f"recordings {len(src)}, with more than one clip {multi}, clips from those {clips}")
print("fold counts", dict(sorted(Counter(ds["fold"]).items())))

def feats(y, sr):
    m = librosa.feature.mfcc(y=y.astype(np.float32), sr=sr, n_mfcc=13)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])

X, y, g = [], [], []
for row in ds:
    wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    X.append(feats(wav, sr))
    y.append(label[row["category"]])
    g.append(row["src_file"])
X, y, g = np.array(X), np.array(y), np.array(g)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
scores = cross_val_score(clf, X, y, cv=GroupKFold(5), groups=g)
print("group", np.round(scores, 3), "mean", round(float(scores.mean()), 3))

# %% [markdown]
# ## 21.4.5 Transfer learning: CLAP and a pretrained speaker embedder

# %% [markdown]
# Example 21-17. CLAP zero-shot classification and a linear probe on all 50 ESC-50 categories with the official folds.

# %%
import time
from transformers import ClapModel, ClapProcessor
from sklearn.model_selection import PredefinedSplit, cross_val_score
torch.manual_seed(0)
torch.set_num_threads(2)
clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
clap_proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
print(f"CLAP: {sum(p.numel() for p in clap.parameters())} parameters, audio at {clap_proc.feature_extractor.sampling_rate} Hz")

esc = load_dataset("ashraq/esc50", split="train").cast_column("audio", Audio(decode=False))
cats = sorted(set(esc["category"]))
A, M, labels, folds, batch = [], [], [], [], []
t0 = time.perf_counter()
for row in esc:                                           # all 2000 clips of the 50 categories
    w, s = sf.read(io.BytesIO(row["audio"]["bytes"]))
    w = w.mean(axis=1) if w.ndim > 1 else w
    M.append(mfcc_features(w, s))                         # Example 21-15's 26-number fingerprint
    batch.append(resample_poly(w, 48000, s).astype(np.float32))   # CLAP reads 48 kHz audio
    labels.append(cats.index(row["category"])); folds.append(row["fold"])
    if len(batch) == 25:
        with torch.no_grad():
            A.append(clap.get_audio_features(**clap_proc(audio=batch, sampling_rate=48000,
                                                         return_tensors="pt")).pooler_output.numpy())
        batch = []
A, M, labels, folds = np.concatenate(A), np.array(M), np.array(labels), np.array(folds)
print(f"{len(A)} clips embedded to {A.shape[1]} numbers each in {time.perf_counter() - t0:.0f} s")

prompts = [f"the sound of a {c.replace('_', ' ')}" for c in cats]
with torch.no_grad():
    T = clap.get_text_features(**clap_proc(text=prompts, return_tensors="pt", padding=True)).pooler_output.numpy()
sims = A @ T.T                                            # cosine similarity: both sides have unit length
five = np.isin(labels, [cats.index(c) for c in CLASSES]) # Example 21-15's five classes
five_cols = [cats.index(c) for c in CLASSES]
print(f"zero shot, no training: 50 categories {np.mean(sims.argmax(1) == labels):.3f}; "
      f"Example 21-15's five {np.mean(np.array(five_cols)[sims[five][:, five_cols].argmax(1)] == labels[five]):.3f}")

def official_folds(X, y, f):                              # train on four folds, test on the fifth, five times
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    return cross_val_score(probe, X, y, cv=PredefinedSplit(f - 1))
for name, X in (("frozen CLAP", A), ("MFCC 26", M)):
    s50, s5 = official_folds(X, labels, folds), official_folds(X[five], labels[five], folds[five])
    print(f"linear probe on {name:11s}: 50 categories {np.round(s50, 3)} mean {s50.mean():.3f}; "
          f"five classes mean {s5.mean():.3f}")

# %% [markdown]
# Example 21-18. x-vectors from a pretrained WavLM model on the speaker trials of Section 21.4.2.

# %%
from transformers import AutoFeatureExtractor, WavLMForXVector
torch.manual_seed(0)
torch.set_num_threads(2)
xfe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
xvec = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
print(f"WavLM x-vector model: {sum(p.numel() for p in xvec.parameters())} parameters, {xvec.config.xvector_output_dim}-number embedding")

segs, who = [], []                                        # the 1.5 s segments of the EER block of Section 21.4.2
for spk, name in enumerate(["libri1", "libri2", "libri3"]):
    s, _ = librosa.load(librosa.example(name), sr=16000)
    step = int(1.5 * 16000)
    for k in range(len(s) // step):
        segs.append(s[k * step:(k + 1) * step]); who.append(spk)
who = np.array(who)
with torch.no_grad():
    E = xvec(**xfe(segs, sampling_rate=16000, return_tensors="pt", padding=True)).embeddings
E = torch.nn.functional.normalize(E, dim=-1).numpy()     # unit length, so a dot product is a cosine
same = [float(E[i] @ E[j]) for i in range(len(E)) for j in range(i + 1, len(E)) if who[i] == who[j]]
diff = [float(E[i] @ E[j]) for i in range(len(E)) for j in range(i + 1, len(E)) if who[i] != who[j]]
print(f"{len(same)} same-speaker and {len(diff)} different-speaker trials; mean cosine same {np.mean(same):.2f}, "
      f"different {np.mean(diff):.2f}")
print(f"EER with x-vectors {eer(same, diff):.2%}, against 15.76% for the mean MFCC embedding of Section 21.4.2")

# %% [markdown]
# ## 21.5 Music generation

# %% [markdown]
# choose symbolic notes for control and efficiency, or learned audio tokens for full-fidelity sound, exactly as you choose text tokens or image pixels elsewhere.

# %%
import numpy as np
import soundfile as sf
from collections import defaultdict, Counter
from music21 import corpus
rng = np.random.default_rng(0)

# REAL scores: J.S. Bach's chorales, bundled as scores inside the music21 library.
paths = corpus.getComposer("bach")
melodies, other_first = [], 0
for p in paths:
    sc = corpus.parse(p)
    sop = [part for part in sc.parts if part.partName == "Soprano"]   # pick the soprano by its name
    if not sop: continue                                   # skip a score with no part named Soprano
    line = [n.pitch.midi for n in sop[0].recurse().notes if n.isNote]
    if len(line) >= 20:
        melodies.append(line)
        other_first += sc.parts[0].partName != "Soprano"  # parts[0] can be a horn, trumpet, or oboe part
    if len(melodies) == 50: break                          # 50 chorales keep the run fast
vocab = sorted(set(n for m in melodies for n in m))
def mean_abs_interval(mels): return np.mean([abs(m[i+1]-m[i]) for m in mels for i in range(len(m)-1)])
print(f"Bach chorales: {len(melodies)} soprano lines, {sum(len(m) for m in melodies)} notes, "
      f"pitch vocabulary {len(vocab)} (MIDI {vocab[0]}..{vocab[-1]})")
print(f"scores that list another part before the Soprano: {other_first}")
print(f"corpus style: mean |interval| between consecutive notes = {mean_abs_interval(melodies):.2f} semitones")
print("(small = smooth, stepwise melodies, the style we want the model to imitate)")

trans = defaultdict(Counter)
for m in melodies:
    for i in range(len(m) - 2):
        trans[(m[i], m[i+1])][m[i+2]] += 1                # count: which pitch follows each PAIR
starts = [(m[0], m[1]) for m in melodies]
def generate(length=32):
    a, b = starts[rng.integers(len(starts))]              # begin the way a real chorale begins
    out = [a, b]
    for _ in range(length - 2):
        nxt = trans.get((a, b))
        if not nxt: c = int(rng.choice(vocab))            # unseen context: fall back to random
        else:
            choices, counts = zip(*nxt.items()); c = int(rng.choice(choices, p=np.array(counts)/sum(counts)))
        out.append(c); a, b = b, c
    return out
gen = [generate() for _ in range(300)]
rand = [[int(rng.choice(vocab)) for _ in range(32)] for _ in range(300)]   # random same-vocabulary baseline
print(f"\nmean |interval|:  Bach corpus {mean_abs_interval(melodies):.2f}  |  "
      f"Markov-generated {mean_abs_interval(gen):.2f}  |  random-in-vocab {mean_abs_interval(rand):.2f}")
print("the Markov model reproduces Bach's smooth, mostly stepwise melodic motion; random notes leap.")

sr = 16000; note_dur = 0.25
melody = generate(16)
audio = np.concatenate([0.4 * np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12)
                                     * np.linspace(0, note_dur, int(sr * note_dur), endpoint=False))
                        for m in melody])
sf.write("05_generated_melody.wav", audio, sr)
print(f"\nrendered a 16-note generated melody to 05_generated_melody.wav (MIDI pitches {melody})")

print("\nAll music-generation checks ran.")

# %% [markdown]
# ## 21.5.1 Pitch, MIDI numbers, and harmonics

# %%
import numpy as np
for m in (69, 60, 81):
    print(m, round(440 * 2 ** ((m - 69) / 12), 2))
sr = 16000
f0 = 261.63
t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
x = (np.sin(2 * np.pi * f0 * t)
     + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
     + 0.25 * np.sin(2 * np.pi * 3 * f0 * t))
spec = np.abs(np.fft.rfft(x))
freqs = np.fft.rfftfreq(len(x), 1 / sr)
top = np.argsort(spec)[::-1][:5]
print("bin width Hz", sr / len(t))
print("strongest bins Hz", np.round(freqs[top], 1))

# %% [markdown]
# ## 21.5.2 A neural model of chorale melodies

# %% [markdown]
# Example 21-20. An LSTM over Bach soprano lines, split by melody group, against a smoothed order-2 Markov model.

# %%
import collections
import copy
import difflib
import math
import torch
import torch.nn as nn
from music21 import corpus
from sklearn.model_selection import GroupShuffleSplit

lines, seen, names, expanded = [], set(), [], []
for path in corpus.getComposer("bach"):                   # every Bach score in the music21 corpus
    score = corpus.parse(path)
    sop = [part for part in score.parts if part.partName == "Soprano"]   # Example 21-19's selection rule
    if not sop:
        continue
    line = tuple(n.pitch.midi for n in sop[0].recurse().notes if n.isNote)
    if len(line) < 20 or line in seen:                    # at least 20 notes, and no exact duplicate
        continue
    seen.add(line); lines.append(list(line))
    names.append(path.stem)                               # the score's name, such as bwv112.5
    expanded.append([n.pitch.midi for n in sop[0].expandRepeats().recurse().notes if n.isNote])   # repeats played out

def shape(line):                                          # intervals after merging repeated notes: the tune
    tune = [line[0]] + [b for a, b in zip(line, line[1:]) if b != a]
    return [b - a for a, b in zip(tune, tune[1:])]       # unchanged when the melody is transposed
shapes = [shape(l) for l in lines]
n = len(lines)
sim = np.eye(n)
for i in range(n):
    for j in range(i + 1, n):
        sim[i, j] = sim[j, i] = difflib.SequenceMatcher(None, shapes[i], shapes[j], autojunk=False).ratio()
group = list(range(n))                                    # melodies at similarity 0.9 or more share a group
def root(a):
    while group[a] != a:
        a = group[a]
    return a
for i in range(n):
    for j in range(i + 1, n):
        if sim[i, j] >= 0.9:
            group[root(i)] = root(j)
group = np.array([root(i) for i in range(n)])
sizes = np.bincount(np.unique(group, return_inverse=True)[1])
print(f"{n} soprano lines, {sum(map(len, lines))} notes, {len(sizes)} melody groups "
      f"({int((sizes == 1).sum())} of one line, the largest of {sizes.max()})")

shuffled = np.random.default_rng(0).permutation(n)       # the split this listing does not use: by line
held, kept = shuffled[:n // 5], shuffled[n // 5:]
print(f"a split by line would leave {int((sim[np.ix_(held, kept)].max(axis=1) >= 0.9).sum())} of its {len(held)} "
      f"test lines with a training line at similarity 0.9 or more")

trval, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(np.zeros(n), groups=group))
fit_pos, val_pos = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0).split(
    np.zeros(len(trval)), groups=group[trval]))
fit, val = trval[fit_pos], trval[val_pos]                 # validation groups come out of the training side
for name, part in (("train", fit), ("validation", val), ("test", test)):
    print(f"  {name:10s}: {len(part):3d} lines in {len(set(group[part])):3d} groups")
print(f"highest similarity of a test line to any training or validation line: {sim[np.ix_(test, trval)].max():.3f}")

vocab = sorted({p for l in lines for p in l})
idx = {p: i + 1 for i, p in enumerate(vocab)}             # token 0 is the start symbol
V = len(vocab) + 1
print(f"pitch vocabulary {len(vocab)} (MIDI {vocab[0]}..{vocab[-1]}) plus a start symbol: V = {V}")

def markov_nll(train_ids, eval_ids, k):                   # order 2, add-k smoothing over the pitch vocabulary
    counts = collections.defaultdict(collections.Counter)
    for i in train_ids:
        s = [0, 0] + [idx[p] for p in lines[i]]
        for t in range(2, len(s)):
            counts[s[t - 2], s[t - 1]][s[t]] += 1
    nll = notes = 0
    for i in eval_ids:
        s = [0, 0] + [idx[p] for p in lines[i]]
        for t in range(2, len(s)):
            c = counts.get((s[t - 2], s[t - 1]), collections.Counter())
            nll -= math.log((c[s[t]] + k) / (sum(c.values()) + k * (V - 1))); notes += 1
    return nll / notes, counts
k_best = min((0.01, 0.1, 1.0), key=lambda k: markov_nll(fit, val, k)[0])
markov_test, counts = markov_nll(trval, test, k_best)     # refit on train plus validation, score the test groups
print(f"\nMarkov order 2, add-{k_best} smoothing (chosen on validation): test perplexity {math.exp(markov_test):.2f}")

class MelodyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 32)                    # an embedding table, never a raw index into a tensor
        self.lstm = nn.LSTM(32, 128, batch_first=True)
        self.out = nn.Linear(128, V)
    def forward(self, x, state=None):
        h, state = self.lstm(self.emb(x), state)
        return self.out(h), state

def batch(ids):                                           # inputs start with token 0; targets are the next notes
    L = max(len(lines[i]) for i in ids)
    X = torch.zeros(len(ids), L, dtype=torch.long)
    Y = torch.full((len(ids), L), -100, dtype=torch.long)
    for r, i in enumerate(ids):
        s = [0] + [idx[p] for p in lines[i]]
        X[r, :len(s) - 1] = torch.tensor(s[:-1]); Y[r, :len(s) - 1] = torch.tensor(s[1:])
    return X, Y

def mean_nll(model, ids):
    X, Y = batch(ids)
    with torch.no_grad():
        return nn.functional.cross_entropy(model(X)[0].reshape(-1, V), Y.reshape(-1), ignore_index=-100).item()

torch.manual_seed(0)
torch.set_num_threads(2)
lstm = MelodyLSTM()
opt = torch.optim.Adam(lstm.parameters(), lr=3e-3)
order = torch.Generator().manual_seed(0)
best = (math.inf, 0, None)
for epoch in range(1, 31):
    lstm.train()
    perm = fit[torch.randperm(len(fit), generator=order).numpy()]
    for b in range(0, len(perm), 16):
        X, Y = batch(perm[b:b + 16])
        loss = nn.functional.cross_entropy(lstm(X)[0].reshape(-1, V), Y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    lstm.eval()
    v = mean_nll(lstm, val)
    if v < best[0]:
        best = (v, epoch, copy.deepcopy(lstm.state_dict()))
    if epoch % 5 == 0:
        print(f"  epoch {epoch:2d}: validation perplexity {math.exp(v):.2f}")
lstm.load_state_dict(best[2])
print(f"LSTM ({sum(p.numel() for p in lstm.parameters())} parameters), best validation epoch {best[1]}: "
      f"test perplexity {math.exp(mean_nll(lstm, test)):.2f}")

gen = torch.Generator().manual_seed(0)
openings = [lines[i][:2] for i in fit]
def sample_markov(first, length=32):
    s = [idx[p] for p in first]
    while len(s) < length:
        c = counts.get((s[-2], s[-1]), collections.Counter())
        probs = torch.tensor([(c[t] + k_best) if t else 0.0 for t in range(V)])
        s.append(int(torch.multinomial(probs, 1, generator=gen)))
    return [vocab[t - 1] for t in s]
def sample_lstm(first, length=32):
    s = [0] + [idx[p] for p in first]
    with torch.no_grad():
        logits, state = lstm(torch.tensor([s]))
        while len(s) < length + 1:
            probs = torch.softmax(logits[0, -1], -1); probs[0] = 0.0   # never emit the start symbol
            s.append(int(torch.multinomial(probs, 1, generator=gen)))
            logits, state = lstm(torch.tensor([[s[-1]]]), state)
    return [vocab[t - 1] for t in s[1:]]
picks = torch.randint(len(openings), (300,), generator=gen).tolist()
markov_samples = [sample_markov(openings[i]) for i in picks]
lstm_samples = [sample_lstm(openings[i]) for i in picks]
print(f"\nmean |interval| in semitones: corpus {mean_abs_interval(lines):.2f}, Markov samples "
      f"{mean_abs_interval(markov_samples):.2f}, LSTM samples {mean_abs_interval(lstm_samples):.2f}")
melody = lstm_samples[0][:16]
tone = lambda m: 0.4 * np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * np.arange(4000) / 16000)
sf.write("05_lstm_melody.wav", np.concatenate([tone(m) for m in melody]), 16000)
print(f"wrote 05_lstm_melody.wav, 16 notes of one LSTM sample: {melody}")

# The 0.9 rule compared the tunes as the scores write them. Play every repeat sign out and compare again.
tunes = [shape(e) for e in expanded]
in_val = set(val.tolist())
print("\ntest lines whose tune, with repeats played out, equals a training or validation tune:")
leaked = []
for i in test:
    partners = [j for j in trval if tunes[j] == tunes[i]]
    if partners:
        j = partners[0]; leaked.append(i)
        print(f"  {names[i]} ({len(lines[i])} notes) = {names[j]} ({len(lines[j])} notes, "
              f"{'validation' if j in in_val else 'train'}), similarity as written {sim[i, j]:.3f}")
rest = np.array([i for i in test if i not in leaked])
print(f"test perplexity on the other {len(rest)} test lines: Markov {math.exp(markov_nll(trval, rest, k_best)[0]):.2f}, "
      f"LSTM {math.exp(mean_nll(lstm, rest)):.2f}")

# %% [markdown]
# ## 21.5.3 Text to music with MusicGen

# %% [markdown]
# Example 21-21. MusicGen small generating five seconds of music from a text prompt.

# %%
import time
from transformers import AutoProcessor, MusicgenForConditionalGeneration
torch.manual_seed(0)
torch.set_num_threads(2)
mg_proc = AutoProcessor.from_pretrained("facebook/musicgen-small")
musicgen = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").eval()
cfg = musicgen.config
K, fr, sr_mg = cfg.decoder.num_codebooks, cfg.audio_encoder.frame_rate, cfg.audio_encoder.sampling_rate
count = lambda m: sum(p.numel() for p in m.parameters())
print(f"MusicGen small: {count(musicgen)} parameters = {cfg.text_encoder.model_type} text encoder {count(musicgen.text_encoder)}"
      f" + decoder {count(musicgen.decoder)} + EnCodec {count(musicgen.audio_encoder)} + projection {count(musicgen.enc_to_dec_proj)}")
print(f"decoder: {cfg.decoder.num_hidden_layers} layers, {K} codebooks of {cfg.decoder.vocab_size} entries, "
      f"{fr} frames per second of {sr_mg} Hz audio; guidance scale {musicgen.generation_config.guidance_scale}")

pad = musicgen.generation_config.pad_token_id            # the delay pattern for 8 steps, from the library itself
_, mask = musicgen.decoder.build_delay_pattern_mask(torch.full((K, 1), pad), pad, max_length=8)
print("delay pattern (P padding, x a token the decoder predicts), one row per codebook:")
for row in mask.tolist():
    print("   ", " ".join("P" if v == pad else "x" for v in row))

prompt = "upbeat acoustic folk with a fiddle"
inputs = mg_proc(text=[prompt], padding=True, return_tensors="pt")
torch.manual_seed(0)                                     # sampling draws random numbers: seed it
t0 = time.perf_counter()
with torch.no_grad():
    audio = musicgen.generate(**inputs, do_sample=True, max_new_tokens=256)[0, 0].numpy()
seconds = time.perf_counter() - t0
sf.write("05_musicgen_folk.wav", audio, sr_mg)
print(f"256 decoder steps -> {256 - (K - 1)} frames x {sr_mg // fr} samples = {len(audio)} samples = "
      f"{len(audio) / sr_mg:.2f} s of audio, written to 05_musicgen_folk.wav; generation took {seconds:.2g} s")

texts = [prompt, "heavy metal with distorted electric guitars", "a slow solo piano ballad",
         "rain falling on a roof", "a dog barking"]
with torch.no_grad():                                    # CLAP (Section 21.4.5) as an independent listener
    a = clap.get_audio_features(**clap_proc(audio=[resample_poly(audio, 48000, sr_mg).astype(np.float32)],
                                            sampling_rate=48000, return_tensors="pt")).pooler_output
    t = clap.get_text_features(**clap_proc(text=texts, return_tensors="pt", padding=True)).pooler_output
for text, s in sorted(zip(texts, (a @ t.T)[0].tolist()), key=lambda ts: -ts[1]):
    print(f"  CLAP similarity {s:.3f}  '{text}'")

# %% [markdown]
# ## 21.6 Voice AI in practice

# %% [markdown]
# Example 21-22 gates speech by energy and clusters speech turns by speaker from MFCC embeddings.

# %%
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
np.random.seed(0)

# VAD on real speech: librosa.example("libri1") is a ~15 s LibriSpeech reading
y, sr = librosa.load(librosa.example("libri1"), sr=16000)
print(f"real speech: libri1, {len(y)/sr:.2f} s at {sr} Hz")
win = int(0.025 * sr); hop = int(0.010 * sr)            # 25 ms frames, 10 ms hop
rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0]
rms_db = 20 * np.log10(rms + 1e-9)
thresh = rms_db.max() - 30                              # speech = within 30 dB of the peak
speech = rms_db > thresh
t = librosa.frames_to_time(np.arange(len(speech)), sr=sr, hop_length=hop)

segs = []                                               # collapse per-frame flags into segments
i = 0
while i < len(speech):
    if speech[i]:
        j = i
        while j < len(speech) and speech[j]:
            j += 1
        end = t[min(j, len(t) - 1)]
        if segs and t[i] - segs[-1][1] < 0.3:           # merge gaps shorter than 0.3 s
            segs[-1][1] = end
        else:
            segs.append([t[i], end])
        i = j
    else:
        i += 1
segs = [(a, b) for a, b in segs if b - a > 0.2]         # drop blips shorter than 0.2 s
print(f"VAD found {len(segs)} speech segments (the silences are the pauses between phrases):")
for a, b in segs:
    print(f"   {a:5.2f}-{b:5.2f} s  ({b - a:.2f} s of speech)")
frac = sum(b - a for a, b in segs) / (len(y) / sr)      # merged-segment share of the clip
print(f"  speech = {sum(b - a for a, b in segs):.2f} s of {len(y)/sr:.2f} s  (fraction {frac:.2f})")

# Diarization on THREE real speakers: libri1/libri2/libri3 are different readers
clips, who = [], []
for spk, name in enumerate(["libri1", "libri2", "libri3"]):
    s, _ = librosa.load(librosa.example(name), sr=16000)
    step = int(1.5 * 16000)                             # 1.5 s segments
    for k in range(len(s) // step):
        clips.append(s[k * step:(k + 1) * step]); who.append(spk)
who = np.array(who)
emb = np.array([librosa.feature.mfcc(y=c, sr=16000, n_mfcc=13).mean(axis=1) for c in clips])
emb = StandardScaler().fit_transform(emb)
clusters = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(emb)
purity = sum(Counter(who[clusters == c]).most_common(1)[0][1] for c in set(clusters)) / len(who)
print(f"\ndiarization: {len(clips)} segments from 3 real speakers (libri1/2/3), clustered by mean-MFCC embedding")
print(f"  cluster purity = {purity:.2f}  (each voice gives a consistent embedding, so it clusters)")

print("\nAll voice-AI checks ran.")

# %% [markdown]
# Example 21-23 mixes the libri1 reading with ESC-50 rain at 10 dB SNR and runs spectral subtraction in front of the energy gate of Example 21-22.

# %%
import io
import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset, Audio
from scipy.signal import resample_poly

sr = 16000
speech, _ = librosa.load(librosa.example("libri1"), sr=sr)      # the clean reading of Example 21-22
esc = load_dataset("ashraq/esc50", split="train").cast_column("audio", Audio(decode=False))
for row in esc:                                                  # the first rain clip, as in Section 21.1.5
    if row["category"] == "rain":
        rain, rsr = sf.read(io.BytesIO(row["audio"]["bytes"])); name = row["filename"]; break
if rain.ndim > 1: rain = rain.mean(axis=1)
rain = resample_poly(rain, sr, rsr)                              # 44.1 kHz -> 16 kHz
noise = np.tile(rain, int(np.ceil(len(speech) / len(rain))))[:len(speech)]   # repeat the 5 s of rain
g = np.sqrt(np.mean(speech ** 2) / (np.mean(noise ** 2) * 10 ** (10 / 10)))  # the 21.1.5 gain for 10 dB
noisy = speech + g * noise

def snr_db(clean, x):                                            # all that is not the clean speech is error
    return 10 * np.log10(np.sum(clean ** 2) / np.sum((x - clean) ** 2))

def gate(x):                                                     # Example 21-22's rule on 25 ms frames every
    rms = librosa.feature.rms(y=x, frame_length=400, hop_length=160)[0]   # 10 ms: within 30 dB of the loudest
    db = 20 * np.log10(rms + 1e-9)
    return db > db.max() - 30

truth = gate(speech)                                             # the clean clip's flags are the reference
print(f"{name} repeated to {len(speech) / sr:.2f} s and mixed at {snr_db(speech, noisy):.1f} dB SNR")
print(f"gate flags {truth.mean():.2f} of the frames of clean speech, {gate(noisy).mean():.2f} of speech in rain")

Y = librosa.stft(noisy, n_fft=400, hop_length=160)              # the same 25 ms frames, every 10 ms
mag, phase = np.abs(Y), np.angle(Y)
energy = (mag ** 2).sum(axis=0)
quiet = energy <= np.percentile(energy, 10)                      # the quietest 10 percent of the frames
N = mag[:, quiet].mean(axis=1, keepdims=True)                    # noise magnitude estimate, one per bin
print(f"noise estimated from {quiet.sum()} quiet frames, {np.mean(~truth[quiet]):.2f} of them pauses in the clean clip")

for alpha, floor in ((1.0, 0.0), (2.0, 0.02)):                   # plain subtraction; oversubtraction + floor
    S = np.maximum(mag - alpha * N, floor * mag)                 # subtract, never below floor x the noisy value
    enhanced = librosa.istft(S * np.exp(1j * phase), hop_length=160, length=len(noisy))   # noisy phase kept
    flags = gate(enhanced)
    print(f"subtract {alpha:.0f} x noise, floor {floor:.2f}: SNR {snr_db(speech, enhanced):.2f} dB, "
          f"gate flags {flags.mean():.2f}, agrees with the clean gate on {np.mean(flags == truth):.2f}")
    sf.write(f"06_enhanced_x{alpha:.0f}.wav", enhanced, sr)      # listen for the residual "musical noise"
sf.write("06_speech_in_rain.wav", noisy, sr)

# %% [markdown]
# ## Inside the transducer: streaming alignment

# %% [markdown]
# Example 21-24 checks the hand case, implements the recursion, and confirms it against brute-force enumeration on a lattice small enough to enumerate.
#
# Example 21-24. The RNN-T forward recursion, checked against brute-force enumeration of all 20 alignments of a 4-frame, 3-token lattice.

# %%
import numpy as np
from itertools import combinations
from math import comb
rng = np.random.default_rng(0)

# first check the hand-worked T=2, U=1 case from the prose
path1 = 0.6 * (1 - 0.1) * (1 - 0.2)
path2 = (1 - 0.6) * 0.5 * (1 - 0.2)
print(f"hand case: path1 {path1:.3f} + path2 {path2:.3f} = {path1 + path2:.3f}")

T, U = 4, 3
p_emit = rng.uniform(0.2, 0.8, (T + 2, U + 2))            # p(next label | t, u); indexed [t][u]

def p_blank(t, u): return 1.0 - p_emit[t, u]

alpha = np.zeros((T + 1, U + 1)); alpha[1, 0] = 1.0       # forward lattice, frames 1-indexed
for t in range(1, T + 1):
    for u in range(0, U + 1):
        if t == 1 and u == 0: continue
        s = 0.0
        if t > 1: s += alpha[t - 1, u] * p_blank(t - 1, u)   # a blank moved us to the next frame
        if u > 0: s += alpha[t, u - 1] * p_emit[t, u - 1]    # a label moved us up the transcript
        alpha[t, u] = s
p_forward = alpha[T, U] * p_blank(T, U)                   # final blank consumes the last frame

total = 0.0                                               # brute force: interleave T-1 blanks, U labels
for label_pos in combinations(range(T + U - 1), U):
    t, u, prob = 1, 0, 1.0
    for i in range(T + U - 1):
        if i in label_pos: prob *= p_emit[t, u]; u += 1
        else: prob *= p_blank(t, u); t += 1
    total += prob * p_blank(T, U)
print(f"T={T} frames, U={U} labels: {comb(T + U - 1, U)} alignments")
print(f"forward algorithm p(y|x) = {p_forward:.6f}   brute-force sum = {total:.6f}")

# %% [markdown]
# ## Wake words and barge-in

# %% [markdown]
# We plant a 0.7-second stretch of the libri1 reading inside a 30-second stream built from three concatenated readers and slide the detector along it in 50-millisecond steps.

# %%
import numpy as np
import librosa

sr = 16000
libs = {n: librosa.load(librosa.example(n), sr=sr)[0] for n in ["libri1", "libri2", "libri3"]}
wake = libs["libri1"][int(3.5 * sr):int(4.2 * sr)]        # a 0.7 s stretch of reader 1: the "wake word"
stream = np.concatenate([libs["libri2"][:10 * sr], libs["libri1"][:10 * sr], libs["libri3"][:10 * sr]])
# the template recurs at 10 s + 3.5 s = 13.5 s into the 30 s stream

def mfcc_vec(x):
    v = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)[1:].ravel()   # drop c0 (loudness), keep the shape
    return v / (np.linalg.norm(v) + 1e-9)

tv = mfcc_vec(wake)
win, hop = len(wake), int(0.05 * sr)                      # slide in 50 ms steps
sims = np.array([mfcc_vec(stream[s0:s0 + win]) @ tv
                 for s0 in range(0, len(stream) - win, hop)])
t = np.arange(len(sims)) * hop / sr
best = int(np.argmax(sims))
away = np.abs(t - t[best]) > 0.5                          # everything half a second from the hit
print(f"windows scored: {len(sims)}")
print(f"detector fires at {t[best]:.2f} s, similarity {sims[best]:.2f}  (true location 13.50 s)")
print(f"highest similarity anywhere else: {sims[away].max():.2f}")

# %% [markdown]
# ## 21.7 Hands-on lab: spectrogram CNN classifier and Whisper transcription

# %% [markdown]
# Example 21-26 is the lab, training a CNN to classify sounds from log-mel spectrograms and running Whisper to transcribe speech.

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

# %% [markdown]
# ## 21.8 Coding exercise starters

# %%
import numpy as np

def strongest_frequencies(x, sr, k):
    """Return the k strongest frequencies, in hertz, in the real FFT of x."""
    ...

sr = 16000
t = np.arange(sr) / sr                                   # one second of sample times
mix = 0.7 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)
print(len(mix), mix.dtype)
print(strongest_frequencies(mix, sr, 2))

# %%
import librosa

y, sr = librosa.load(librosa.example("libri1"), sr=None)

def window_tradeoff(y, sr):
    """Return the STFT shapes at n_fft 512 and 4096."""
    ...

print(len(y), sr)
print(window_tradeoff(y, sr))

# %%
import librosa

y, sr = librosa.load(librosa.example("trumpet"), sr=None)
frame = y[8000:8000 + 2048]                              # one 2048-sample frame of the trumpet

def frame_mfcc(frame, sr, n_filters=40, n_coeff=13):
    """Return the first n_coeff MFCCs of one frame, computed from scratch."""
    ...

print(len(y), sr, len(frame))
print(frame_mfcc(frame, sr))

# %%
import numpy as np

# the hand case of Section 21.2: T = 2 frames, symbol 0 is the blank and symbol 1 is "a"
P = np.array([[0.4, 0.6],
              [0.4, 0.6]])                               # P[t, k] = p(symbol k at frame t)
target = [1]

def ctc_forward(P, target, blank=0):
    """Return p(target | x), the sum over all alignments, by the alpha recursion."""
    ...

print(P.shape, P.sum(axis=1))
print(ctc_forward(P, target))

# %%
import re
from datasets import load_dataset, Audio

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                  split="validation").cast_column("audio", Audio(decode=False))

def norm(s):                                             # the normalization of Example 21-3
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()

def error_counts(ref, hyp):
    """Return (S, D, I, N) for two token lists, read off a backtrace of the edit distance table."""
    ...

print(ds.num_rows, [len(norm(ds[k]["text"])) for k in range(3)])
print(error_counts(norm(ds[0]["text"]), norm(ds[0]["text"])))

# %%
import numpy as np
import librosa

y, sr = librosa.load(librosa.example("trumpet"), sr=None)
S = np.abs(librosa.stft(y, n_fft=512, hop_length=128))   # magnitude only, as in Example 21-8

def griffin_grid(S, y, iterations=(1, 8, 32)):
    """Spectral match for momentum 0 and 0.99 crossed with init "random" and None."""
    ...

print(len(y), sr, S.shape)
print(griffin_grid(S, y))

# %%
from datasets import load_dataset, Audio

ds = load_dataset("ashraq/esc50", split="train").cast_column("audio", Audio(decode=False))
TEN = ["cat", "cow", "crow", "frog", "hen", "insects", "pig", "rooster", "sheep", "chirping_birds"]
sub = ds.filter(lambda r: r["category"] in TEN)          # 400 clips that Example 21-15 never used

def fingerprint_scores(sub):
    """Official-fold and random-split accuracy of the 26-number fingerprint, and retrieval."""
    ...

print(ds.num_rows, sub.num_rows, sorted(set(sub["fold"])))
print(fingerprint_scores(sub))

# %%
from music21 import corpus

paths = corpus.getComposer("bach")

def bass_intervals(paths):
    """Mean absolute bass interval on 50 chorales, plus a Markov sample."""
    ...

print(len(paths))
print(bass_intervals(paths))

# %%
import io
import soundfile as sf
import librosa
from datasets import load_dataset, Audio

speech, sr = librosa.load(librosa.example("libri1"), sr=16000)
esc = load_dataset("ashraq/esc50", split="train").cast_column("audio", Audio(decode=False))
row = next(r for r in esc if r["category"] == "rain")    # the first rain clip, as in Section 21.1.5
rain, rain_sr = sf.read(io.BytesIO(row["audio"]["bytes"]))

def gate_fractions(speech, rain, rain_sr, snrs=(20, 10, 0)):
    """For each SNR, the fraction of frames each gate flags and its agreement with the clean gate."""
    ...

print(len(speech), sr, row["filename"], rain.shape, rain_sr)
print(gate_fractions(speech, rain, rain_sr))

# %%
import librosa

clips = [librosa.load(librosa.example(name), sr=16000)[0] for name in ("libri1", "libri2", "libri3")]

def choose_speaker_count(clips, ks=range(2, 7)):
    """Silhouette score for each k, the chosen k, and the cluster purity at that k."""
    ...

print([len(c) for c in clips], [len(c) // 24000 for c in clips])
print(choose_speaker_count(clips))

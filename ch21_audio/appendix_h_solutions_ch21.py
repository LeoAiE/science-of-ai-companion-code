"""Appendix H solutions for Chapter 21, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 21.8 coding exercises

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, re
import librosa
from scipy.fft import rfft, rfftfreq, dct

# %% [markdown]
# 1. FFT peaks and aliasing at two sample rates

# %%
# 1. FFT peaks and aliasing at two sample rates
sr = 16000; t = np.linspace(0, 1.0, sr, endpoint=False)
wave = 0.7*np.sin(2*np.pi*440*t) + 0.3*np.sin(2*np.pi*1000*t)
spec = np.abs(rfft(wave)); fr = rfftfreq(len(wave), 1/sr)
peaks = sorted(int(round(f)) for f in fr[np.argsort(spec)[::-1][:2]])
def alias_of(f, rate):                                     # where a one-second FFT at this rate puts a tone of f Hz
    x = np.sin(2*np.pi*f*np.arange(rate)/rate)
    return int(round(rfftfreq(rate, 1/rate)[np.argmax(np.abs(rfft(x)))]))
print("peaks:", peaks, "Nyquist:", sr//2, "15000 Hz aliases to:", alias_of(15000, sr))
print("at 22050 Hz: Nyquist:", 22050//2, "15000 Hz aliases to:", alias_of(15000, 22050))
# expected output: peaks: [440, 1000] Nyquist: 8000 15000 Hz aliases to: 1000
# expected output: at 22050 Hz: Nyquist: 11025 15000 Hz aliases to: 7050

# %% [markdown]
# 2. STFT window tradeoff on real LibriSpeech

# %%
# 2. STFT window tradeoff on real LibriSpeech
y, srr = librosa.load(librosa.example("libri1"), sr=None)
for n_fft in [512, 4096]:
    hop = n_fft // 4
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    print(f"n_fft={n_fft}: {S.shape} freqres={srr/n_fft:.1f}Hz step={1000*hop/srr:.1f}ms")
# expected output: n_fft=512: (257, 2557) freqres=43.1Hz step=5.8ms
# expected output: n_fft=4096: (2049, 320) freqres=5.4Hz step=46.4ms

# %% [markdown]
# 3. MFCCs from scratch, checked against librosa on the same frame

# %%
# 3. MFCCs from scratch, checked against librosa on the same frame
yt, st = librosa.load(librosa.example("trumpet"), sr=None)
hz2mel = lambda h: 2595*np.log10(1 + h/700)
mel2hz = lambda m: 700*(10**(m/2595) - 1)
def my_mfcc(frame, sr, n_filters=40, n_coeff=13):
    mag = np.abs(rfft(frame))**2                          # power spectrum, no window
    fq = rfftfreq(len(frame), 1/sr)
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr/2), n_filters+2))
    fb = np.zeros((n_filters, len(fq)))
    for i in range(n_filters):                            # triangular mel filters with peak height 1
        lo, mid, hi = pts[i], pts[i+1], pts[i+2]
        fb[i] = np.clip(np.minimum((fq-lo)/(mid-lo+1e-9), (hi-fq)/(hi-mid+1e-9)), 0, None)
    return dct(np.log(fb @ mag + 1e-10), norm="ortho")[:n_coeff]   # natural log, then the DCT
frame = yt[8000:8000+2048]                                # the starter's 2048-sample frame
mine = my_mfcc(frame, st)
print("mfcc:", [round(float(v), 2) for v in mine], "shape:", mine.shape)
# librosa with the same choices: one 2048-sample frame, no window (boxcar), 40 HTK filters of peak height 1
lib = librosa.feature.mfcc(y=frame, sr=st, n_mfcc=13, n_fft=2048, hop_length=2048, center=False,
                           window="boxcar", n_mels=40, htk=True, mel_norm=None)
ratio = lib[:, 0] / mine                                  # one ratio per coefficient
print("librosa shape:", lib.shape, "ratio librosa/mine from", round(float(ratio.min()), 3),
      "to", round(float(ratio.max()), 3), "10/ln(10):", round(10/np.log(10), 3))
print("all defaults:", librosa.feature.mfcc(y=frame, sr=st, n_mfcc=13).shape)  # librosa frames the samples itself
# expected output: mfcc: [14.87, 13.04, -13.09, -3.36, 1.61, -1.41, -1.57, -1.01, -1.5, -0.5, 1.44, 2.79, 2.74] shape: (13,)
# expected output: librosa shape: (13, 1) ratio librosa/mine from 4.343 to 4.343 10/ln(10): 4.343
# expected output: all defaults: (13, 5)

# %% [markdown]
# 4. The CTC forward algorithm

# %%
# 4. The CTC forward algorithm
import torch
def ctc_forward(P, target, blank=0):
    """P[t, c] is frame t's probability of symbol c; returns p(target | x) summed over all alignments."""
    ext = [blank]
    for c in target:
        ext += [c, blank]                                 # blank-extended labels: _ y1 _ y2 ... _
    T, L = P.shape[0], len(ext)
    a = np.zeros((T, L))
    a[0, 0], a[0, 1] = P[0, ext[0]], P[0, ext[1]]         # start on the first blank or the first label
    for t in range(1, T):
        for s in range(L):
            v = a[t-1, s] + (a[t-1, s-1] if s > 0 else 0.0)   # stay put, or move one place on
            if s > 1 and ext[s] != blank and ext[s] != ext[s-2]:
                v += a[t-1, s-2]                          # skip the blank between two different labels
            a[t, s] = v * P[t, ext[s]]
    return a[-1, -1] + a[-1, -2]                          # end on the final blank or on the last label
hand = np.array([[0.4, 0.6], [0.4, 0.6]])                 # Section 21.2: blank 0.4 and "a" 0.6 at both frames
print("hand case p('a') =", round(float(ctc_forward(hand, [1])), 4))
rng = np.random.default_rng(0)
logits = rng.normal(size=(6, 4))                          # a random lattice: T = 6 frames, 4 symbols
P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
p = ctc_forward(P, [1, 2, 2])                             # the repeated 2 forces a blank between them
loss = torch.nn.CTCLoss(blank=0, reduction="sum")(torch.log(torch.tensor(P)).unsqueeze(1),
       torch.tensor([[1, 2, 2]]), torch.tensor([6]), torch.tensor([3]))
print(f"T=6, target [1, 2, 2]: p = {p:.6f}, -ln p = {-np.log(p):.6f}, CTCLoss = {loss.item():.6f}")
# expected output: hand case p('a') = 0.84
# expected output: T=6, target [1, 2, 2]: p = 0.000473, -ln p = 7.656312, CTCLoss = 7.656312

# %% [markdown]
# 5. Word error rate by error type, and the character error rate, on Whisper's transcripts

# %%
# 5. Word error rate by error type, and the character error rate, on Whisper's transcripts
import io, soundfile as sf
from datasets import load_dataset, Audio
from transformers import pipeline
def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()   # the chapter's normalization
def edit_counts(r, h):
    """Align token lists r (reference) and h (hypothesis) and return the counts S, D, I."""
    d = np.zeros((len(r)+1, len(h)+1), int)
    d[:, 0] = np.arange(len(r)+1); d[0, :] = np.arange(len(h)+1)
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i, j] = min(d[i-1, j]+1, d[i, j-1]+1, d[i-1, j-1] + (r[i-1] != h[j-1]))
    S = D = I = 0; i, j = len(r), len(h)
    while i > 0 or j > 0:                                 # walk back along one cheapest path
        if i > 0 and j > 0 and d[i, j] == d[i-1, j-1] + (r[i-1] != h[j-1]):
            S += int(r[i-1] != h[j-1]); i, j = i-1, j-1   # a match or a substitution
        elif i > 0 and d[i, j] == d[i-1, j] + 1:
            D += 1; i -= 1                                # a reference token the hypothesis dropped
        else:
            I += 1; j -= 1                                # an extra hypothesis token
    assert S + D + I == d[len(r), len(h)]                 # the three counts add up to the final cell
    return S, D, I
torch.manual_seed(0); torch.set_num_threads(2)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
ls = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                  split="validation").cast_column("audio", Audio(decode=False))
for k in range(3):
    w, s = sf.read(io.BytesIO(ls[k]["audio"]["bytes"]))
    hyp = asr({"raw": np.asarray(w, np.float32), "sampling_rate": s})["text"]
    r, h = norm(ls[k]["text"]), norm(hyp)
    S, D, I = edit_counts(r, h)
    rc, hc = list(" ".join(r)), list(" ".join(h))         # characters of the normalized text, spaces included
    print(f"clip {k}: N={len(r)} S={S} D={D} I={I} WER {(S+D+I)/len(r):.3f} "
          f"CER {sum(edit_counts(rc, hc))/len(rc):.3f}")
# expected output: clip 0: N=17 S=1 D=0 I=0 WER 0.059 CER 0.045
# expected output: clip 1: N=10 S=1 D=0 I=0 WER 0.100 CER 0.065
# expected output: clip 2: N=32 S=0 D=0 I=0 WER 0.000 CER 0.000

# %% [markdown]
# 6. Griffin-Lim: the original algorithm against fast Griffin-Lim, random against zero initial phase

# %%
# 6. Griffin-Lim: the original algorithm against fast Griffin-Lim, random against zero initial phase
S = np.abs(librosa.stft(yt, n_fft=512, hop_length=128))   # magnitude only; phase gone
def smatch(a, b):
    Sa = np.abs(librosa.stft(a, n_fft=512, hop_length=128))
    Sb = np.abs(librosa.stft(b, n_fft=512, hop_length=128))
    m = min(Sa.shape[1], Sb.shape[1])
    return np.corrcoef(Sa[:, :m].ravel(), Sb[:, :m].ravel())[0, 1]
for mom in (0.0, 0.99):                                   # 0 is the original algorithm, 0.99 librosa's default
    for init in ("random", None):                         # None starts every bin at phase zero
        m = [round(float(smatch(yt, librosa.griffinlim(S, n_iter=n, hop_length=128, momentum=mom,
                                                           init=init, rng=0))), 3) for n in (1, 8, 32)]
        print(f"momentum {mom}, init {init}: 1, 8, 32 iterations {m}")
# expected output: momentum 0.0, init random: 1, 8, 32 iterations [0.921, 0.968, 0.988]
# expected output: momentum 0.0, init None: 1, 8, 32 iterations [0.864, 0.953, 0.983]
# expected output: momentum 0.99, init random: 1, 8, 32 iterations [0.921, 0.981, 0.998]
# expected output: momentum 0.99, init None: 1, 8, 32 iterations [0.864, 0.973, 0.997]

# %% [markdown]
# 7. Miniproject: ten other ESC-50 categories under the official folds

# %%
# 7. Miniproject: ten other ESC-50 categories under the official folds
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, PredefinedSplit
TEN = ["cat", "cow", "crow", "frog", "hen", "insects", "pig", "rooster", "sheep", "chirping_birds"]
lab = {c: i for i, c in enumerate(TEN)}
esc = load_dataset("ashraq/esc50", split="train").cast_column("audio", Audio(decode=False))
ds = esc.filter(lambda r: r["category"] in TEN)            # 2000 real clips -> 400 clips in 10 categories
def feat(w, sr):
    m = librosa.feature.mfcc(y=w.astype(np.float32), sr=sr, n_mfcc=13)
    return np.concatenate([m.mean(1), m.std(1)])           # the 26-number mean and std summary
X, Y, F, src = [], [], [], []
for row in ds:
    w, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
    if w.ndim > 1: w = w.mean(1)                           # stereo -> mono
    X.append(feat(w, sr)); Y.append(lab[row["category"]]); F.append(row["fold"]); src.append(row["src_file"])
X, Y, F, src = np.array(X), np.array(Y), np.array(F), np.array(src)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
folds = cross_val_score(clf, X, Y, cv=PredefinedSplit(F - 1))   # test on each official fold in turn
print("clips:", len(X), "official folds:", np.round(folds, 3), "mean:", round(float(folds.mean()), 3))
Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=0.4, random_state=0, stratify=Y)
print("random 40% split:", round(clf.fit(Xtr, ytr).score(Xte, yte), 3), "chance:", round(1/len(TEN), 2))
E = normalize(StandardScaler().fit_transform(X))           # standardize, then unit-normalize
hit = hit_other = 0
for i in range(len(E)):
    s = E @ E[i]; s[i] = -1
    hit += int(Y[int(np.argmax(s))] == Y[i])               # any other clip may be the neighbor
    s[src == src[i]] = -1                                  # now drop clips cut from the query's recording
    hit_other += int(Y[int(np.argmax(s))] == Y[i])
print("retrieval same-class, all 400 clips:", round(100*hit/len(E)), "%; other recordings only:",
      round(100*hit_other/len(E)), "%")
# expected output: clips: 400 official folds: [0.662 0.612 0.675 0.612 0.662] mean: 0.645
# expected output: random 40% split: 0.688 chance: 0.1
# expected output: retrieval same-class, all 400 clips: 77 %; other recordings only: 63 %

# %% [markdown]
# 8. Bass lines of the 50 Bach chorales of Example 21-19

# %%
# 8. Bass lines of the 50 Bach chorales of Example 21-19
from collections import defaultdict, Counter
from music21 import corpus
rng = np.random.default_rng(0)
sops, basses = [], []
for pth in corpus.getComposer("bach"):
    sc = corpus.parse(pth)
    sop = [p for p in sc.parts if p.partName == "Soprano"]   # Example 21-19's selection rule
    if not sop:
        continue
    line = [n.pitch.midi for n in sop[0].recurse().notes if n.isNote]
    if len(line) < 20:
        continue
    bass = [p for p in sc.parts if p.partName == "Bass"][0]  # every selected chorale has one
    sops.append(line); basses.append([n.pitch.midi for n in bass.recurse().notes if n.isNote])
    if len(sops) == 50:
        break
mai = lambda ms: np.mean([abs(m[i+1]-m[i]) for m in ms for i in range(len(m)-1)])
trans = defaultdict(Counter)
for m in basses:
    for i in range(len(m)-2):
        trans[(m[i], m[i+1])][m[i+2]] += 1
vocab = sorted(set(n for m in basses for n in m))
starts = [(m[0], m[1]) for m in basses]
def gen(n=32):
    a, b = starts[int(rng.integers(len(starts)))]
    out = [a, b]
    for _ in range(n-2):
        nx = trans.get((a, b))
        if not nx:
            c = int(rng.choice(vocab))
        else:
            ch, ct = zip(*nx.items()); c = int(rng.choice(ch, p=np.array(ct)/sum(ct)))
        out.append(c); a, b = b, c
    return out
g = [gen() for _ in range(300)]
rd = [[int(rng.choice(vocab)) for _ in range(32)] for _ in range(300)]
print("chorales:", len(basses), "bass vocabulary:", len(vocab), "soprano corpus:", round(float(mai(sops)), 2))
print("mean|interval| bass corpus:", round(float(mai(basses)), 2), "gen:", round(float(mai(g)), 2),
      "random:", round(float(mai(rd)), 2))
# expected output: chorales: 50 bass vocabulary: 27 soprano corpus: 1.84
# expected output: mean|interval| bass corpus: 3.21 gen: 3.28 random: 8.94

# %% [markdown]
# 9. The energy gate in rain, and one fix

# %%
# 9. The energy gate in rain, and one fix
from scipy.signal import resample_poly
yv, srv = librosa.load(librosa.example("libri1"), sr=16000)
rain_row = next(r for r in esc if r["category"] == "rain")  # the first rain clip of ESC-50
rain, rs = sf.read(io.BytesIO(rain_row["audio"]["bytes"]))
if rain.ndim > 1: rain = rain.mean(1)
rain = resample_poly(rain, 16000, rs)
rain = np.tile(rain, int(np.ceil(len(yv)/len(rain))))[:len(yv)]   # repeat it to the length of the speech
def frame_db(x):                                           # 25 ms frames, 10 ms hop, RMS in decibels
    return 20*np.log10(librosa.feature.rms(y=x, frame_length=400, hop_length=160)[0] + 1e-9)
clean = frame_db(yv); ref = clean > clean.max() - 30       # the gate's decisions on the clean speech
print(rain_row["filename"], "clean speech fraction:", round(float(ref.mean()), 2))
Ps, Pn = np.mean(yv**2), np.mean(rain**2)
for snr in (20, 10, 0):
    d = frame_db(yv + np.sqrt(Ps / (Pn * 10**(snr/10))) * rain)   # the gain of Section 21.1.5
    peak = d > d.max() - 30                                # within 30 dB of the loudest frame
    floor = d > np.percentile(d, 10) + 6                   # 6 dB above the 10th percentile frame
    print(f"{snr} dB: peak gate {peak.mean():.2f} (agrees {np.mean(peak == ref):.2f}); "
          f"floor gate {floor.mean():.2f} (agrees {np.mean(floor == ref):.2f})")
# expected output: 1-17367-A-10.wav clean speech fraction: 0.82
# expected output: 20 dB: peak gate 1.00 (agrees 0.82); floor gate 0.74 (agrees 0.92)
# expected output: 10 dB: peak gate 1.00 (agrees 0.82); floor gate 0.59 (agrees 0.77)
# expected output: 0 dB: peak gate 1.00 (agrees 0.82); floor gate 0.14 (agrees 0.32)

# %% [markdown]
# 10. Diarization with the speaker count chosen by silhouette score

# %%
# 10. Diarization with the speaker count chosen by silhouette score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
clips, who = [], []
for spk, name in enumerate(["libri1", "libri2", "libri3"]):
    s, _ = librosa.load(librosa.example(name), sr=16000)
    step = int(1.5*16000)                                  # 1.5 s segments
    for k in range(len(s)//step): clips.append(s[k*step:(k+1)*step]); who.append(spk)
who = np.array(who)
emb = np.array([librosa.feature.mfcc(y=c, sr=16000, n_mfcc=13).mean(1) for c in clips])
emb = StandardScaler().fit_transform(emb)
sil = {k: silhouette_score(emb, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb))
       for k in range(2, 7)}                               # one score per candidate speaker count
print("silhouette for k = 2 to 6:", [round(float(sil[k]), 3) for k in range(2, 7)])
best = max(sil, key=sil.get)
cl = KMeans(n_clusters=best, n_init=10, random_state=0).fit_predict(emb)
pur = sum(Counter(who[cl == k]).most_common(1)[0][1] for k in set(cl)) / len(who)
print("chosen k:", best, "segments:", len(clips), "purity:", round(pur, 2))
# expected output: silhouette for k = 2 to 6: [0.231, 0.28, 0.238, 0.227, 0.206]
# expected output: chosen k: 3 segments: 29 purity: 0.9

# %% [markdown]
# ## 21.8 coding drills

# %%
import numpy as np, re
import librosa
yt, st = librosa.load(librosa.example("trumpet"), sr=None)

# %% [markdown]
# Signals and features

# %%
# Signals and features
print(int(2.5*16000))                                      # expected output: 40000
print(44100//2)                                            # expected output: 22050
print(np.abs(librosa.stft(yt, n_fft=1024, hop_length=256)).shape)   # expected output: (513, 460)
print(round(2595*np.log10(1 + 1000/700), 1))               # expected output: 1000.0
print(round(700*(10**(1000/2595) - 1), 1))                 # expected output: 1000.0
print(librosa.feature.mfcc(y=yt, sr=st, n_mfcc=13).shape)  # expected output: (13, 230)

# %% [markdown]
# Recognition

# %%
# Recognition
def ctc_decode(f, blank="_"):
    out = []
    for ch in f:
        if not out or ch != out[-1]: out.append(ch)
    return "".join(c for c in out if c != blank)
print(ctc_decode("aa__bb__b"))                             # expected output: abb
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r)+1, len(h)+1), int)
    d[:, 0] = np.arange(len(r)+1); d[0, :] = np.arange(len(h)+1)
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i, j] = min(d[i-1, j]+1, d[i, j-1]+1, d[i-1, j-1] + (r[i-1] != h[j-1]))
    return d[len(r), len(h)] / len(r)
print(round(wer("play some jazz", "please play some jazz music"), 2))   # expected output: 0.67

# %% [markdown]
# Synthesis

# %%
# Synthesis
S = np.abs(librosa.stft(yt, n_fft=512, hop_length=128))
rec = librosa.griffinlim(S, n_iter=16, hop_length=128, rng=0)
Sr = np.abs(librosa.stft(rec, n_fft=512, hop_length=128))
m = min(S.shape[1], Sr.shape[1])
print(round(float(np.corrcoef(S[:, :m].ravel(), Sr[:, :m].ravel())[0, 1]), 2))  # expected output: 0.99

# %% [markdown]
# Classification and embeddings

# %%
# Classification and embeddings
mm = librosa.feature.mfcc(y=yt, sr=st, n_mfcc=13)
print(np.concatenate([mm.mean(1), mm.std(1)]).shape)       # expected output: (26,)
a, b = np.array([1., 0, 0]), np.array([0.6, 0.8, 0])
print(round(float(a@b / (np.linalg.norm(a)*np.linalg.norm(b))), 3))   # expected output: 0.6

# %% [markdown]
# Music

# %%
# Music
deg = [0, 2, 4, 5, 7]
print(np.mean([abs(deg[i+1]-deg[i]) for i in range(len(deg)-1)]))     # expected output: 1.75
print(round(440*2**((69-69)/12), 1))                       # expected output: 440.0

# %% [markdown]
# Voice systems

# %%
# Voice systems
yv, srv = librosa.load(librosa.example("libri2"), sr=16000)
db = 20*np.log10(librosa.feature.rms(y=yv, frame_length=400, hop_length=160)[0] + 1e-9)
print(round(float((db > db.max()-20).mean()), 2))          # expected output: 0.58

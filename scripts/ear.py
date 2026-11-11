#!/usr/bin/env python3
"""ear.py: listen to a film's sound without ears.

Turns audio into what can be judged by eye and by number:
  * a labelled spectrogram per window (time in seconds, loudness, every event marked)
  * every event (onset) described and classed: THUD, TAP, BELL (a tonal ring), VOICE (pitched, voice-like),
    NOTE (a short tone), WHOOSH, BOOM, with the numbers behind the call
  * per stem: counts, density, the voice's count and seconds, music key/mode and tempo, silences, loud spots
  * optional: what a speech model hears in the voice (faster-whisper, local), to catch mumbling
  * optional (--tags): what a trained listener calls each event and every 2 s of the mix, in AudioSet's words
    ("Walk, footsteps", "Ding", "Humming", "Speech"), via ear_tags.py (MIT AST, ONNX, local)

usage:
  python3 ear.py FILE [--stems DIR] [--out DIR] [--win 15] [--from S] [--to S] [--words] [--tags]
  python3 ear.py --samples a.wav b.wav ...      (class each sample on its own: test foley before mixing)

FILE can be a wav, mp3, mp4 (its audio) or raw .f32 (48 kHz mono float32). Stems are the *.wav in --stems.
Needs numpy and matplotlib; decoding and loudness go through ffmpeg.

--words needs faster-whisper and soundfile, --tags needs onnxruntime and the model (see ear_tags.py). When those
live in another Python (a venv), point EAR_PYTHON at its interpreter; by default the current one is used.
EAR_LANG sets the language --words listens for (default "en").
"""
import json, math, os, re, subprocess, sys, tempfile

import numpy as np

SR = 22050
NFFT = 2048
HOP = 256
HELPER_PY = os.environ.get('EAR_PYTHON') or sys.executable  # runs faster-whisper and ear_tags.py


# ── decoding and loudness ────────────────────────────────────────────────────────────────────────
def load(path, t0=None, t1=None):
    if path.endswith('.f32'):
        x = np.fromfile(path, dtype=np.float32).astype(np.float64)
        n = int(len(x) * SR / 48000)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)  # crude resample, fine for analysis
        if t0 is not None or t1 is not None:
            x = x[int((t0 or 0) * SR): int(t1 * SR) if t1 else None]
        return x
    cmd = ['ffmpeg', '-v', 'error']
    if t0:
        cmd += ['-ss', str(t0)]
    cmd += ['-i', path]
    if t1:
        cmd += ['-t', str(t1 - (t0 or 0))]
    cmd += ['-vn', '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def loudness(path, t0=None, t1=None):
    """Momentary loudness every 100 ms plus the integrated summary, from ffmpeg's ebur128."""
    cmd = ['ffmpeg', '-v', 'verbose', '-nostats']
    if t0:
        cmd += ['-ss', str(t0)]
    cmd += ['-i', path]
    if t1:
        cmd += ['-t', str(t1 - (t0 or 0))]
    cmd += ['-vn', '-af', 'ebur128=peak=true:framelog=verbose', '-f', 'null', '-']
    err = subprocess.run(cmd, capture_output=True, text=True).stderr
    ts, ms = [], []
    summ = {}
    for line in err.splitlines():
        if 'Parsed_ebur128' in line:
            # ffmpeg pads the value to six places, so a quiet "M:-120.7" has no space after the colon
            m = re.search(r'\bt:\s*([0-9.]+).*?\bM:\s*(-?[0-9.]+|-?inf|nan)', line)
            if m:
                ts.append(float(m.group(1)))
                ms.append(float(m.group(2)))
        # the summary lines; only the number and its unit, since ffmpeg may print other messages on the same line
        m = re.match(r'\s*(I|LRA|Peak):\s*(-?[0-9.]+|-?inf|nan) (LUFS|LU|dBFS)', line)
        if m:
            summ[m.group(1)] = f'{m.group(2)} {m.group(3)}'
    return np.array(ts), np.array(ms), summ


# ── spectra ──────────────────────────────────────────────────────────────────────────────────────
def stft_power(x, nfft=NFFT, hop=HOP):
    x = np.pad(x, (nfft // 2, nfft // 2))  # centred: frame i sits on sample i * hop, so pictures and times line up
    w = np.hanning(nfft)
    n = 1 + (len(x) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(x[idx] * w, axis=1)) ** 2  # frames × bins


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_bank(nb=128, fmin=30, fmax=11000, nfft=NFFT):
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    m = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), nb + 2)
    f = 700 * (10 ** (m / 2595) - 1)
    fb = np.zeros((nb, len(freqs)))
    for i in range(nb):
        a, b, c = f[i], f[i + 1], f[i + 2]
        fb[i] = np.clip(np.minimum((freqs - a) / (b - a), (c - freqs) / (c - b)), 0, None)
    return fb, f[1:-1]


def onsets(P, fb):
    """Spectral flux on the log-mel spectrum, peak-picked against a local median."""
    L = np.log10(P @ fb.T + 1e-10)
    flux = np.maximum(0, np.diff(L, axis=0)).sum(axis=1)
    flux = np.concatenate([[0], flux])
    k = max(1, int(0.02 * SR / HOP))
    flux = np.convolve(flux, np.ones(k) / k, mode='same')
    half = int(0.75 * SR / HOP)
    out = []
    last = -1e9
    gap = int(0.06 * SR / HOP)
    for i in range(1, len(flux) - 1):
        if flux[i] < flux[i - 1] or flux[i] < flux[i + 1]:
            continue
        lo, hi = max(0, i - half), min(len(flux), i + half)
        med = np.median(flux[lo:hi])
        mad = np.median(np.abs(flux[lo:hi] - med)) + 1e-9
        if flux[i] > med + 6 * mad and flux[i] > 0.02 * flux.max() and i - last >= gap:
            out.append(i)
            last = i
    return np.array(out) * HOP / SR


def yin_f0(seg, fmin=70, fmax=900):
    """Pitch (Hz) and aperiodicity (0 = perfectly periodic) over a short segment; None when unpitched."""
    W = min(len(seg), 2048)
    if W < 512:
        return None, 1.0
    x = seg[:W] - seg[:W].mean()
    tmax = min(int(SR / fmin), W // 2)
    tmin = int(SR / fmax)
    d = np.array([np.sum((x[:W - tmax] - x[t:t + W - tmax]) ** 2) for t in range(tmax)])
    cm = np.ones_like(d)
    cm[1:] = d[1:] * np.arange(1, tmax) / (np.cumsum(d[1:]) + 1e-12)
    rng = cm[tmin:tmax]
    if not len(rng):
        return None, 1.0
    below = np.where(rng < 0.15)[0]
    i = (below[0] if len(below) else int(np.argmin(rng))) + tmin
    while i + 1 < tmax and cm[i + 1] < cm[i]:
        i += 1
    ap = float(cm[i])
    if ap > 0.35:
        return None, ap
    return SR / i, ap


def note_name(f):
    if not f:
        return '-'
    m = int(round(69 + 12 * math.log2(f / 440)))
    return ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][m % 12] + str(m // 12 - 1)


def tonal_peaks(seg, start, dur):
    """How tonal a stretch is: prominence (dB) of its strongest narrow peak over the local floor, and that peak's Hz."""
    a, b = int(start * SR), int((start + dur) * SR)
    s = seg[a:b]
    if len(s) < 256:
        return 0.0, 0.0
    n = 1 << int(math.ceil(math.log2(max(len(s), 2048))))
    p = np.abs(np.fft.rfft(s * np.hanning(len(s)), n)) ** 2
    f = np.fft.rfftfreq(n, 1 / SR)
    band = (f > 150) & (f < 10000)
    pb, fbnd = p[band], f[band]
    if not len(pb) or pb.max() <= 0:
        return 0.0, 0.0
    i = int(np.argmax(pb))
    w = max(8, int(300 / (SR / n)))
    floor = np.median(pb[max(0, i - w): i + w]) + 1e-20
    return 10 * np.log10(pb[i] / floor), float(fbnd[i])


def coarse_bands(seg):
    """Power in 24 log-spaced bands, 60 Hz to 10 kHz: a spectrum coarse enough to be steady for steady noise."""
    sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 2048)) ** 2
    fr = np.fft.rfftfreq(2048, 1 / SR)
    edges = np.geomspace(60, 10000, 25)
    return np.array([sp[(fr >= lo) & (fr < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])


def describe(x, t):
    """Measure the event that starts at t (seconds) in x."""
    a = max(0, int((t - 0.01) * SR))
    seg = x[a: a + int(1.2 * SR)]
    if len(seg) < 256:
        return None
    h = int(0.005 * SR)
    env = np.sqrt(np.convolve(seg ** 2, np.ones(h) / h, mode='same') + 1e-20)
    db = 20 * np.log10(env + 1e-12)
    pk = int(np.argmax(db[: int(0.15 * SR)])) if len(db) > int(0.15 * SR) else int(np.argmax(db))
    peak_db = float(db[pk])
    # A sound that stops also changes the spectrum. When the 50 ms before were louder than anything in the 150 ms
    # after, this is the tail of that sound letting go, not a new event.
    w = int(0.05 * SR)
    before = x[max(0, a - w): a]
    before_db = 10 * np.log10(np.mean(before ** 2) + 1e-20) if len(before) >= h else -200.0
    if before_db > peak_db + 3:
        return None
    # A sound that carries on is not a new event either: something already sounding at a comparable level, the
    # same coarse spectrum after as before, and no step up of 3 dB or more from one 20 ms to the next. (Without
    # this, the fade-out of a tone or the middle of a swell counts as an event whenever nothing louder is around.)
    nxt = x[a + int(0.01 * SR): a + int(0.01 * SR) + w]
    if len(before) == w and len(nxt) == w and before_db > peak_db - 20:
        b0, b1 = coarse_bands(before), coarse_bands(nxt)
        alike = float(b0 @ b1 / (np.linalg.norm(b0) * np.linalg.norm(b1) + 1e-20))
        q = int(0.02 * SR)
        run = x[max(0, a - q): a + int(0.15 * SR)]
        lv = [10 * np.log10(np.mean(run[i: i + q] ** 2) + 1e-20) for i in range(0, len(run) - q + 1, q)]
        if alike >= 0.9 and (len(lv) < 2 or max(np.diff(lv)) < 3):
            return None
    # The detector fires when the analysis window first touches the sound, up to 50 ms early. The event starts
    # where its own envelope comes within 20 dB of its peak.
    rise = np.where(db[: pk + 1] >= peak_db - 20)[0]
    t = (a + (int(rise[0]) if len(rise) else pk)) / SR
    after = np.where(db[pk:] < peak_db - 30)[0]
    ring = (after[0] / SR) if len(after) else len(db[pk:]) / SR
    ring_open = not len(after)
    head = seg[pk: pk + int(0.06 * SR)]
    if len(head) < 128:
        return None
    sp = np.abs(np.fft.rfft(head * np.hanning(len(head)), 4096)) ** 2
    fr = np.fft.rfftfreq(4096, 1 / SR)
    band = (fr > 60) & (fr < 10000)
    ps, fs = sp[band] + 1e-20, fr[band]
    centroid = float((ps * fs).sum() / ps.sum())
    flatness = float(np.exp(np.mean(np.log(ps))) / np.mean(ps))
    tone_head, f_head = tonal_peaks(seg, pk / SR, 0.06)
    tone_tail, f_tail = tonal_peaks(seg, pk / SR + 0.1, 0.2) if ring > 0.12 else (0.0, 0.0)
    f0, ap = yin_f0(seg[pk: pk + int(0.2 * SR)])
    low = float(ps[fs < 200].sum() / ps.sum())
    return dict(t=round(t, 3), peak_db=round(peak_db, 1), ring=round(ring, 3), ring_open=ring_open,
                centroid=round(centroid), flatness=round(flatness, 3), tone_head=round(tone_head, 1),
                f_head=round(f_head), tone_tail=round(tone_tail, 1), f_tail=round(f_tail),
                f0=round(f0, 1) if f0 else None, aperiodicity=round(ap, 2), low_share=round(low, 2))


def classify(e):
    """A plain call a listener would make, with the reason."""
    ringing = e['ring'] > 0.15 and e['tone_tail'] >= 14 and e['f_tail'] > 700
    if ringing:
        return 'BELL', f"rings {e['ring']*1000:.0f} ms on a clear tone near {e['f_tail']} Hz ({note_name(e['f_tail'])})"
    if e['f0'] and 75 <= e['f0'] <= 750 and e['aperiodicity'] < 0.2 and e['ring'] > 0.12 and e['centroid'] < 3500:
        return 'VOICE', f"pitched {e['f0']} Hz ({note_name(e['f0'])}), {e['ring']*1000:.0f} ms, voice-like range"
    if e['low_share'] > 0.6 and e['ring'] > 0.3:
        return 'BOOM', f"{e['low_share']*100:.0f}% of the energy under 200 Hz, {e['ring']*1000:.0f} ms"
    if e['tone_head'] >= 18 and e['flatness'] < 0.05:
        return 'NOTE', f"a short tone near {e['f_head']} Hz ({note_name(e['f_head'])}), {e['ring']*1000:.0f} ms"
    if e['flatness'] > 0.25 and e['ring'] > 0.25:
        return 'WHOOSH', f"noise for {e['ring']*1000:.0f} ms, centroid {e['centroid']} Hz"
    if e['centroid'] < 1000 and e['ring'] < 0.16:
        return 'THUD', f"short and low: centroid {e['centroid']} Hz, {e['ring']*1000:.0f} ms, no ring"
    if e['ring'] < 0.09:
        return 'TAP', f"short and bright: centroid {e['centroid']} Hz, {e['ring']*1000:.0f} ms"
    return 'OTHER', f"centroid {e['centroid']} Hz, {e['ring']*1000:.0f} ms, flatness {e['flatness']}"


# ── music ────────────────────────────────────────────────────────────────────────────────────────
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def key_and_mode(P):
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    ok = (f > 60) & (f < 2000)
    pc = (np.round(12 * np.log2(f[ok] / 440)) + 9) % 12
    ch = np.zeros(12)
    e = P[:, ok].sum(axis=0)
    for i in range(12):
        ch[i] = e[pc == i].sum()
    if ch.sum() <= 0:
        return None
    ch = ch / ch.sum()
    best = []
    for k in range(12):
        for mode, prof in (('major', MAJ), ('minor', MIN)):
            r = np.corrcoef(ch, np.roll(prof, k))[0, 1]
            best.append((r, f'{NAMES[k]} {mode}'))
    best.sort(reverse=True)
    return best[:3]


def tempo(x):
    P = stft_power(x, 1024, 512)
    flux = np.maximum(0, np.diff(np.log10(P + 1e-10), axis=0)).sum(axis=1)
    flux = flux - flux.mean()
    ac = np.correlate(flux, flux, mode='full')[len(flux) - 1:]
    fps = SR / 512
    lags = np.arange(len(ac)) / fps
    ok = (lags > 60 / 180) & (lags < 60 / 50)
    if not ok.any():
        return None, 0
    i = np.argmax(ac[ok])
    lag = lags[ok][i]
    strength = ac[ok][i] / (ac[0] + 1e-9)
    return 60 / lag, float(strength)


# ── drawing ──────────────────────────────────────────────────────────────────────────────────────
QUIET_STEMS = ('music', 'score', 'low', 'amb', 'ambience')  # their events are not marked in the pictures
SHORT = {'BELL': 'Be', 'VOICE': 'V', 'NOTE': 'N', 'THUD': 'Th', 'TAP': 'Ta', 'WHOOSH': 'W', 'BOOM': 'Bo', 'OTHER': '?'}
COLORS = {'BELL': '#ff4d6d', 'VOICE': '#ffd166', 'NOTE': '#7bdff2', 'THUD': '#9ef01a', 'TAP': '#c77dff',
          'WHOOSH': '#adb5bd', 'BOOM': '#f77f00', 'OTHER': '#ffffff'}


def draw(path_png, x, t0, events, loud, stems_rms, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    P = stft_power(x)
    fb, fc = mel_bank()
    M = 10 * np.log10(P @ fb.T + 1e-10)
    dur = len(x) / SR
    rows = 3 if stems_rms else 2
    fig, ax = plt.subplots(rows, 1, figsize=(max(12, dur * 1.2), 7 + (2 if stems_rms else 0)), sharex=True,
                           gridspec_kw={'height_ratios': [4, 1.4] + ([1.6] if stems_rms else [])})
    vmax = np.percentile(M, 99.5)
    ax[0].imshow(M.T, origin='lower', aspect='auto', cmap='magma', vmin=vmax - 70, vmax=vmax,
                 extent=[t0, t0 + dur, 0, len(fc)])
    ticks = [100, 200, 500, 1000, 2000, 4000, 8000]
    ax[0].set_yticks([np.argmin(np.abs(fc - f)) for f in ticks])
    ax[0].set_yticklabels([f'{f//1000}k' if f >= 1000 else str(f) for f in ticks])
    ax[0].set_ylabel('Hz')
    for k, e in enumerate(events):
        c = COLORS.get(e['class'], '#fff')
        ax[0].plot([e['t'], e['t']], [len(fc) * 0.86, len(fc)], color=c, lw=1.4)
        ax[0].text(e['t'], len(fc) * (0.84 - 0.06 * (k % 3)), SHORT.get(e['class'], '?') + (e.get('stem', '')[:1].lower()),
                   color=c, fontsize=8, ha='center', va='top', weight='bold')
    if len(loud[0]):
        ax[1].plot(loud[0], loud[1], color='#2f55d4', lw=1.2)
        top_ = float(np.nanmax(loud[1])) if np.isfinite(loud[1]).any() else -5.0
        ax[1].set_ylim(-50, max(-5, top_ + 2))
        ax[1].axhline(-14, color='#999', lw=0.6, ls='--')
    ax[1].set_ylabel('LUFS (M)')
    if stems_rms:
        for name, (tt_, rr) in stems_rms.items():
            ax[2].plot(tt_, rr, lw=1.1, label=name)
        ax[2].set_ylim(-60, 0)
        ax[2].set_ylabel('stem dBFS')
        ax[2].legend(loc='upper right', fontsize=8, ncol=6)
    for a in ax:
        a.set_xlim(t0, t0 + dur)
        a.grid(axis='x', color='#ffffff33' if a is ax[0] else '#0002', lw=0.5)
        a.set_xticks(np.arange(math.ceil(t0), t0 + dur + 0.01, 1.0))
    ax[-1].set_xlabel('seconds')
    handles = [plt.Line2D([0], [0], color=c, lw=3, label=k) for k, c in COLORS.items()]
    fig.legend(handles=handles, loc='upper center', ncol=len(COLORS), fontsize=8, frameon=False)
    ax[0].set_title(title, fontsize=10, pad=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path_png, dpi=90)
    plt.close(fig)


def rms_track(x, hop_s=0.05):
    h = int(hop_s * SR)
    n = len(x) // h
    r = np.sqrt((x[: n * h].reshape(n, h) ** 2).mean(axis=1) + 1e-12)
    return np.arange(n) * hop_s, 20 * np.log10(r)


# ── words ────────────────────────────────────────────────────────────────────────────────────────
def words(path, t0=None, t1=None):
    """What a speech model hears: catches mumbling (words where there should be a hum) and counts utterances."""
    tmp = os.path.join(tempfile.mkdtemp(prefix='ear-'), 'words.wav')
    cmd = ['ffmpeg', '-v', 'error', '-y'] + (['-ss', str(t0)] if t0 else []) + ['-i', path] + \
          (['-t', str(t1 - (t0 or 0))] if t1 else []) + ['-vn', '-ac', '1', '-ar', '16000', tmp]
    subprocess.run(cmd, check=True)
    code = r'''
import json, sys
from faster_whisper import WhisperModel
m = WhisperModel("small", device="cpu", compute_type="int8")
import soundfile as sf
audio, _ = sf.read(sys.argv[1], dtype="float32")
segs, info = m.transcribe(audio, language=sys.argv[2], word_timestamps=True, vad_filter=False, beam_size=5,
                          condition_on_previous_text=False, no_speech_threshold=0.9)
out = []
for s in segs:
    out.append(dict(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip(),
                    logprob=round(s.avg_logprob, 2), no_speech=round(s.no_speech_prob, 2)))
print(json.dumps(out))
'''
    r = subprocess.run([HELPER_PY, '-c', code, tmp, os.environ.get('EAR_LANG', 'en')], capture_output=True, text=True)
    os.remove(tmp)
    os.rmdir(os.path.dirname(tmp))
    if r.returncode != 0:
        return [dict(error=r.stderr.strip()[-400:])]
    segs = json.loads(r.stdout.strip().splitlines()[-1])
    for s in segs:
        s['start'] = round(s['start'] + (t0 or 0), 2)
        s['end'] = round(s['end'] + (t0 or 0), 2)
    return segs


# ── a trained listener (AudioSet labels) ────────────────────────────────────────────────────────
TAGS_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ear_tags.py')


def tags(path, wins, topk=3):
    """AudioSet labels per window [(start, dur)], from ear_tags.py (run by EAR_PYTHON); [] when unavailable."""
    if not wins:
        return []
    r = subprocess.run([HELPER_PY, TAGS_PY, path, json.dumps([[round(a, 3), round(d, 3)] for a, d in wins]), str(topk)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write('ear_tags failed: ' + r.stderr[-400:] + '\n')
        return []
    return json.loads(r.stdout.strip().splitlines()[-1])


def heard(tg):
    return ', '.join(f'{n} {p:.2f}' for n, p in tg[:2])


# ── reports ──────────────────────────────────────────────────────────────────────────────────────
def analyse(path, t0=None, t1=None):
    x = load(path, t0, t1)
    P = stft_power(x)
    fb, _ = mel_bank()
    evs = []
    for t in onsets(P, fb):
        e = describe(x, t)
        if e and e['peak_db'] > -60:
            e['class'], e['why'] = classify(e)
            e['t'] = round(e['t'] + (t0 or 0), 3)
            evs.append(e)
    return x, P, evs


def silences(x, t0, thresh=-55, min_s=0.5):
    tt_, rr = rms_track(x, 0.05)
    out, start = [], None
    for t, r in zip(tt_, rr):
        if r < thresh and start is None:
            start = t
        elif r >= thresh and start is not None:
            if t - start >= min_s:
                out.append((round(float(start + t0), 2), round(float(t + t0), 2)))
            start = None
    if start is not None and tt_[-1] - start >= min_s:
        out.append((round(float(start + t0), 2), round(float(tt_[-1] + t0), 2)))
    return out


def hist(evs):
    h = {}
    for e in evs:
        h[e['class']] = h.get(e['class'], 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1]))


def voice_summary(evs, dur):
    v = [e for e in evs if e['class'] == 'VOICE']
    merged = []
    for e in v:
        if merged and e['t'] - (merged[-1]['t'] + merged[-1]['ring']) < 0.2:
            merged[-1]['ring'] = e['t'] + e['ring'] - merged[-1]['t']
        else:
            merged.append(dict(t=e['t'], ring=e['ring']))
    secs = sum(m['ring'] for m in merged)
    return dict(utterances=len(merged), seconds=round(float(secs), 1), per_minute=round(len(merged) / max(dur, 1e-9) * 60, 1),
                times=[round(float(m['t']), 2) for m in merged])


def main():
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0 if args else 2)

    def opt(name, default=None, cast=str):
        if name in args:
            i = args.index(name)
            v = args[i + 1]
            del args[i:i + 2]
            return cast(v)
        return default

    if args and args[0] == '--samples':
        print('| file | class | why | ring ms | centroid | tone tail dB | f0 |')
        print('|---|---|---|---|---|---|---|')
        for p in args[1:]:
            x = load(p)
            hit = np.where(np.abs(x) > 0.05 * (np.abs(x).max() + 1e-12))[0]
            e = describe(x, (hit[0] / SR + 0.01) if len(hit) else 0.01)
            if not e:
                print(f'| {os.path.basename(p)} | (too short) | | | | | |')
                continue
            c, why = classify(e)
            print(f"| {os.path.basename(p)} | **{c}** | {why} | {e['ring']*1000:.0f} | {e['centroid']} | {e['tone_tail']} | {e['f0'] or '-'} |")
        return

    out = opt('--out', 'ear-out')
    stems_dir = opt('--stems')
    win = opt('--win', 15.0, float)
    t_from = opt('--from', None, float)
    t_to = opt('--to', None, float)
    want_words = '--words' in args
    if want_words:
        args.remove('--words')
    want_tags = '--tags' in args
    if want_tags:
        args.remove('--tags')
    path = args[0]
    os.makedirs(out, exist_ok=True)

    x, P, evs = analyse(path, t_from, t_to)
    t0 = t_from or 0.0
    dur = len(x) / SR
    lt, lm, summ = loudness(path, t_from, t_to)
    lt = lt + t0

    stems = {}
    if stems_dir:
        for fn in sorted(os.listdir(stems_dir)):
            if fn.endswith('.wav'):
                sx, _, sev = analyse(os.path.join(stems_dir, fn), t_from, t_to)
                stems[fn[:-4]] = (sx, sev)

    timeline = []
    if want_tags:
        for name, (sx, sev) in stems.items():
            if name in QUIET_STEMS:
                continue
            todo = sev[:150]
            if len(sev) > 150:
                sys.stderr.write(f'tags: {name} has {len(sev)} events, tagging the first 150\n')
            res = tags(os.path.join(stems_dir, name + '.wav'), [(e['t'] - 0.03, min(e['ring'], 0.8) + 0.1) for e in todo])
            for e, r_ in zip(todo, res):
                e['heard'] = r_['tags']
        if not stems:
            res = tags(path, [(e['t'] - 0.03, min(e['ring'], 0.8) + 0.1) for e in evs[:150]])
            for e, r_ in zip(evs, res):
                e['heard'] = r_['tags']
        step = 2.0
        timeline = tags(path, [(t0 + k * step, step) for k in range(max(1, int(math.ceil((dur - 0.5) / step))))])

    rep = [f'# ear: {os.path.basename(path)}', '']
    rep.append(f"Length {dur:.2f} s from {t0:.2f} s. Integrated {summ.get('I', '?')}, LRA {summ.get('LRA', '?')}, true peak {summ.get('Peak', '?')}.")
    rep.append('')
    k = key_and_mode(P)
    bpm, strength = tempo(x)
    if k:
        rep.append(f"Key guess (whole mix): {k[0][1]} ({k[0][0]:.2f}), then {k[1][1]} ({k[1][0]:.2f}), {k[2][1]} ({k[2][0]:.2f}).")
    if bpm:
        rep.append(f"Pulse: about {bpm:.0f} bpm (strength {strength:.2f}; under 0.15 means no steady beat).")
    fr = np.fft.rfftfreq(NFFT, 1 / SR)
    tot = P.sum() + 1e-20
    bands = [(20, 120, 'sub/low'), (120, 500, 'low-mid'), (500, 2000, 'mid'), (2000, 5000, 'presence (harsh zone)'), (5000, 11025, 'air')]
    rep.append('Energy by band: ' + ', '.join(f"{n} {100 * P[:, (fr >= a) & (fr < b)].sum() / tot:.0f}%" for a, b, n in bands) + '.')
    sil = silences(x, t0)
    rep.append(f"Silences (< -55 dBFS for 0.5 s+): {sil if sil else 'none'}.")
    heard_ = np.where(np.isfinite(lm))[0]  # ffmpeg reports no loudness (nan) for digital silence
    if len(heard_):
        top = heard_[np.argsort(lm[heard_], kind='stable')[-5:][::-1]]
        rep.append('Loudest moments (momentary LUFS): ' + ', '.join(f'{lt[i]:.1f} s {lm[i]:.1f}' for i in top) + '.')
    rep.append('')
    rep.append(f"Events in the mix: {len(evs)} ({hist(evs)}); density {len(evs) / dur:.2f}/s.")
    rep.append('')
    for name, (sx, sev) in stems.items():
        rep.append(f"## stem {name}")
        rep.append(f"{len(sev)} events, {hist(sev)}, density {len(sev) / max(dur, 1e-9):.2f}/s.")
        if name.lower().startswith('voice') or name.lower().startswith('vox'):
            rep.append(f"Voice: {voice_summary(sev, dur)}")
        sk = key_and_mode(stft_power(sx))
        if name.lower().startswith(('music', 'score')) and sk:
            rep.append(f"Key guess: {sk[0][1]} ({sk[0][0]:.2f}), {sk[1][1]} ({sk[1][0]:.2f}).")
        rep.append('')
        if any('heard' in e for e in sev):
            h = {}
            for e in sev:
                if e.get('heard'):
                    h[e['heard'][0][0]] = h.get(e['heard'][0][0], 0) + 1
            rep.append(f"Heard as (top label per event): {dict(sorted(h.items(), key=lambda kv: -kv[1]))}")
        rep.append('')
        rep.append('| t | class | why | heard as | peak dB | ring ms |')
        rep.append('|---|---|---|---|---|---|')
        for e in sev:
            rep.append(f"| {e['t']:.2f} | {e['class']} | {e['why']} | {heard(e.get('heard', []))} | {e['peak_db']} | {e['ring']*1000:.0f} |")
        rep.append('')
    if not stems:
        rep.append('| t | class | why | heard as | peak dB | ring ms |')
        rep.append('|---|---|---|---|---|---|')
        for e in evs:
            rep.append(f"| {e['t']:.2f} | {e['class']} | {e['why']} | {heard(e.get('heard', []))} | {e['peak_db']} | {e['ring']*1000:.0f} |")
        rep.append('')
    if timeline:
        rep.append('## what a trained listener hears, every 2 s (AudioSet labels, whole mix)')
        for w in timeline:
            rep.append(f"- {w['t']:.0f}–{w['t'] + w['dur']:.0f} s: " + ', '.join(f'{n} {p:.2f}' for n, p in w['tags']))
        rep.append('')
    if want_words:
        src = os.path.join(stems_dir, 'voice.wav') if stems_dir and os.path.exists(os.path.join(stems_dir, 'voice.wav')) else path
        segs = words(src, t_from, t_to)
        rep.append(f"## what a speech model hears ({os.path.basename(src)})")
        if not segs:
            rep.append('- nothing')
        for s in segs:
            if 'error' in s:
                rep.append(f"- could not run: {s['error']}")
                continue
            # a speech model always writes something; a high no-speech score means it made the words up
            doubt = ' (probably not speech)' if s['no_speech'] > 0.6 else ''
            rep.append(f"- {s['start']:.2f}–{s['end']:.2f} s: \"{s['text']}\"{doubt}  [confidence {s['logprob']}, no-speech {s['no_speech']}]")
        rep.append('')

    stems_rms = {n: (lambda tr: (tr[0] + t0, tr[1]))(rms_track(sx)) for n, (sx, _) in stems.items()}
    n_win = max(1, math.ceil((dur - 0.5) / win))  # a tail under half a second gets no picture of its own
    for i in range(n_win):
        a, b = t0 + i * win, min(t0 + dur, t0 + (i + 1) * win)
        seg = x[int((a - t0) * SR): int((b - t0) * SR)]
        marked = evs if not stems else [dict(e, stem=n) for n, (_, sev) in stems.items() if n not in QUIET_STEMS for e in sev]
        ev_w = sorted([e for e in marked if a <= e['t'] < b], key=lambda e: e['t'])
        lw = (lt[(lt >= a) & (lt < b)], lm[(lt >= a) & (lt < b)])
        sw = {n: (tr[0][(tr[0] >= a) & (tr[0] < b)], tr[1][(tr[0] >= a) & (tr[0] < b)]) for n, tr in stems_rms.items()}
        draw(os.path.join(out, f'ear-{i:02d}-{a:05.1f}-{b:05.1f}.png'), seg, a, ev_w, lw, sw,
             f'{os.path.basename(path)}  {a:.1f}–{b:.1f} s   Be bell · V voice · N note · Th thud · Ta tap · W whoosh · Bo boom (+ stem initial; music/low/amb not marked)')
    with open(os.path.join(out, 'ear.md'), 'w') as f:
        f.write('\n'.join(rep))
    with open(os.path.join(out, 'events.json'), 'w') as f:
        json.dump(dict(mix=evs, stems={n: sev for n, (_, sev) in stems.items()}), f)
    print('\n'.join(rep[:14]))
    print(f'\nwrote {out}/ear.md, events.json and {n_win} pictures')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""ear_tags.py: what a trained listener calls a sound, in AudioSet's own words ("Footsteps", "Bell", "Humming").

The model is MIT's Audio Spectrogram Transformer fine-tuned on Google AudioSet (527 labels), as ONNX:
config.json and onnx/model_quantized.onnx from Xenova/ast-finetuned-audioset-10-10-0.4593 on Hugging Face,
both placed flat in ~/.cache/ear-models/ast (or the folder EAR_MODEL_DIR names). The README has the two
download commands. Features are Kaldi fbanks, written here in numpy to match transformers' ASTFeatureExtractor.

Needs numpy and onnxruntime; decoding goes through ffmpeg:
  ear_tags.py FILE '[[start, dur], ...]' [topk]      → JSON list, one entry per window
  ear_tags.py FILE --every 2 [topk]                   → windows of 2 s across the file
"""
import json, os, subprocess, sys

import numpy as np

MODEL_DIR = os.environ.get('EAR_MODEL_DIR') or os.path.expanduser('~/.cache/ear-models/ast')
SR = 16000


def load16k(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def kaldi_mel_banks(num_bins=128, nfft=512, low=20.0, high=SR / 2):
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)
    lo, hi = mel(low), mel(high)
    delta = (hi - lo) / (num_bins + 1)
    fft_mel = mel(np.arange(nfft // 2) * SR / nfft)
    b = np.arange(num_bins)[:, None]
    left, center, right = lo + b * delta, lo + (b + 1) * delta, lo + (b + 2) * delta
    up = (fft_mel[None, :] - left) / (center - left)
    down = (right - fft_mel[None, :]) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down))
    return np.pad(banks, ((0, 0), (0, 1)))  # (128, 257)


BANKS = kaldi_mel_banks()


def fbank(x):
    flen, shift, nfft = 400, 160, 512
    if len(x) < flen:
        x = np.pad(x, (0, flen - len(x)))
    n = 1 + (len(x) - flen) // shift
    fr = x[np.arange(flen)[None, :] + shift * np.arange(n)[:, None]].copy()
    fr -= fr.mean(axis=1, keepdims=True)
    prev = np.concatenate([fr[:, :1], fr[:, :-1]], axis=1)
    fr = fr - 0.97 * prev
    fr *= np.hanning(flen)[None, :]
    spec = np.abs(np.fft.rfft(fr, nfft, axis=1)) ** 2
    mel = spec @ BANKS.T
    return np.log(np.maximum(mel, np.finfo(np.float32).eps))


def features(x, mean=-4.2677393, std=4.5689974, max_len=1024):
    f = fbank(x)
    if len(f) < max_len:
        f = np.pad(f, ((0, max_len - len(f)), (0, 0)))
    else:
        f = f[:max_len]
    return ((f - mean) / (std * 2)).astype(np.float32)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    for need in ('config.json', 'model_quantized.onnx'):
        if not os.path.exists(os.path.join(MODEL_DIR, need)):
            sys.exit(f'ear_tags: {need} is missing in {MODEL_DIR}. See "Optional: the trained listener" in the README.')
    import onnxruntime as ort
    path = sys.argv[1]
    x = load16k(path)
    if sys.argv[2] == '--every':
        step = float(sys.argv[3])
        topk = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        wins = [[round(t, 2), step] for t in np.arange(0, len(x) / SR, step)]
    else:
        wins = json.loads(sys.argv[2])
        topk = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    labels = json.load(open(os.path.join(MODEL_DIR, 'config.json')))['id2label']
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(os.path.join(MODEL_DIR, 'model_quantized.onnx'), so, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    out = []
    for a, d in wins:
        seg = x[int(max(0, a) * SR): int((a + d) * SR)]
        logits = sess.run(None, {name: features(seg)[None]})[0][0]
        p = 1 / (1 + np.exp(-logits))
        top = np.argsort(p)[::-1][:topk]
        out.append(dict(t=a, dur=d, tags=[[labels[str(i)], round(float(p[i]), 2)] for i in top]))
    print(json.dumps(out))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""eye.py: measure how a film moves, so taste can be compared in numbers.

Per film:
  * cuts (shots per minute, shot lengths)
  * the camera: global pan speed and zoom speed per frame (phase correlation, whole frame and four quadrants),
    how much of the time it moves, and how every move eases (time to peak, smoothness, jerk)
  * holds: stretches where almost nothing changes
  * motion energy over time (how busy the picture is), and its rhythm
  * a picture per film: speeds, zoom, energy and cuts over time, plus a contact sheet at 1 frame per second

usage:
  python3 eye.py FILM.mp4 [FILM2.mp4 ...] --out DIR        (one summary table across all films, in DIR/eye.md)

Needs numpy and matplotlib; frames come from ffmpeg at a small size (256 px wide), so a long film is quick.
"""
import json, math, os, subprocess, sys

import numpy as np

W_SMALL = 256  # analysis width (height follows the film's aspect)


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width,height,r_frame_rate:format=duration', '-of', 'json', path],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j['streams'][0]
    a, b = s['r_frame_rate'].split('/')
    return int(s['width']), int(s['height']), float(a) / float(b), float(j['format']['duration'])


def frames(path, w, h):
    W = W_SMALL
    H = int(round(h * W / w / 2)) * 2
    vf = f'scale={W}:{H}:flags=area,format=gray'
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-vf', vf, '-f', 'rawvideo', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = W * H
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, dtype=np.uint8).reshape(H, W).astype(np.float32)
    p.wait()


def phase_shift(a, b, win):
    A = np.fft.rfft2((a - a.mean()) * win)
    B = np.fft.rfft2((b - b.mean()) * win)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-6
    r = np.fft.irfft2(R, s=a.shape)
    i = np.unravel_index(np.argmax(r), r.shape)
    peak = float(r[i])
    dy, dx = i
    H, W = a.shape

    def sub(c, m, axis):
        lo = r[((dy - 1) % H, dx)] if axis == 0 else r[(dy, (dx - 1) % W)]
        hi = r[((dy + 1) % H, dx)] if axis == 0 else r[(dy, (dx + 1) % W)]
        # Between two pixels the peak of a whitened correlation is a sinc, not a parabola (Foroosh et al., 2002):
        # one neighbour is positive, the other negative, and the positive one's share gives the offset. A parabola
        # reads a 0.3 px shift as 0.2 px, which made slow zooms come out a fifth too slow. When both neighbours
        # have the same sign the peak is blurred and the parabola is the safer fit.
        if hi > 0 >= lo:
            off = hi / (hi + peak)
        elif lo > 0 >= hi:
            off = -lo / (lo + peak)
        else:
            den = lo - 2 * peak + hi
            off = 0.5 * (lo - hi) / den if abs(den) > 1e-9 else 0.0
        v = c + off
        return v - m if v > m / 2 else v

    return sub(dx, W, 1), sub(dy, H, 0), peak


def analyse(path):
    w, h, fps, dur = probe(path)
    pan, zoom, energy, peaks = [], [], [], []
    prev = None
    win = None
    qwin = None
    for f in frames(path, w, h):
        if prev is None:
            H, W = f.shape
            win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
            qwin = np.outer(np.hanning(H // 2), np.hanning(W // 2)).astype(np.float32)
            prev = f
            continue
        dx, dy, pk = phase_shift(prev, f, win)
        H, W = f.shape
        qs = []
        for (y0, x0) in ((0, 0), (0, W // 2), (H // 2, 0), (H // 2, W // 2)):
            qa, qb = prev[y0:y0 + H // 2, x0:x0 + W // 2], f[y0:y0 + H // 2, x0:x0 + W // 2]
            qx, qy, _ = phase_shift(qa, qb, qwin)
            qs.append((qx, qy))
        # divergence: zooming in, the right half's content moves right and the left half's left. phase_shift gives
        # the shift that lays the new frame back on the old one, the opposite way, hence left minus right (+ = in).
        dxl = (qs[0][0] + qs[2][0]) / 2
        dxr = (qs[1][0] + qs[3][0]) / 2
        dyt = (qs[0][1] + qs[1][1]) / 2
        dyb = (qs[2][1] + qs[3][1]) / 2
        z = ((dxl - dxr) / (W / 2) + (dyt - dyb) / (H / 2)) / 2
        pan.append(math.hypot(dx, dy) / W)  # fraction of the frame width per frame
        zoom.append(z)
        energy.append(float(np.mean(np.abs(f - prev))) / 255.0)
        peaks.append(pk)
        prev = f
    pan, zoom, energy, peaks = map(np.array, (pan, zoom, energy, peaks))
    return dict(path=path, w=w, h=h, fps=fps, dur=dur, pan=pan, zoom=zoom, energy=energy, peaks=peaks)


def smooth(x, k):
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode='same')


def summarise(a):
    fps = a['fps']
    pan_s = a['pan'] * fps * 100  # % of width per second
    zoom_s = a['zoom'] * fps * 100  # % scale per second
    e = a['energy']
    # cuts: a frame change far above its neighbourhood with a weak phase peak
    med = np.median(e) + 1e-6
    cut = (e > max(0.08, 6 * med)) & (a['peaks'] < 0.25)
    idx = np.where(cut)[0]
    cuts = []
    for i in idx:
        if not cuts or i - cuts[-1] > fps * 0.2:
            cuts.append(int(i))
    shots = np.diff([0] + cuts + [len(e)]) / fps
    ok = ~cut
    k = max(1, int(fps * 0.1))
    ps = smooth(np.where(ok, pan_s, 0), k)
    zs = smooth(np.where(ok, np.abs(zoom_s), 0), k)
    cam = np.maximum(ps, zs * 0.5)
    moving = cam > 2.0
    holds = (e < 0.004) & ok & ~moving  # a slow move changes little per frame, but it is not a hold
    # moves: runs of moving frames of 0.25 s or more
    moves = []
    i = 0
    n = len(cam)
    while i < n:
        if moving[i]:
            j = i
            while j < n and moving[j]:
                j += 1
            if (j - i) / fps >= 0.25:
                seg = cam[i:j]
                pk = float(np.mean(np.where(seg >= 0.97 * seg.max())[0]))  # the middle of the top, not its first frame
                acc = np.diff(seg) * fps
                jerk = np.diff(acc) * fps if len(acc) > 2 else np.array([0.0])
                moves.append(dict(t=round(i / fps, 2), dur=round((j - i) / fps, 2), peak=round(float(seg.max()), 1),
                                  t_peak=round(pk / max(1, len(seg) - 1), 2),
                                  smooth=round(float(seg.mean() / (seg.max() + 1e-9)), 2),
                                  jerk=round(float(np.median(np.abs(jerk)) / (seg.max() + 1e-9)), 1)))
            i = j
        else:
            i += 1
    hold_runs = []
    i = 0
    while i < n:
        if holds[i]:
            j = i
            while j < n and holds[j]:
                j += 1
            if (j - i) / fps >= 0.3:
                hold_runs.append(round((j - i) / fps, 2))
            i = j
        else:
            i += 1
    es = smooth(e, int(fps * 0.5))
    ac = np.correlate(es - es.mean(), es - es.mean(), mode='full')[len(es) - 1:]
    lags = np.arange(len(ac)) / fps
    okl = (lags > 0.3) & (lags < 4)
    rhythm = float(lags[okl][np.argmax(ac[okl])]) if okl.any() and ac[0] > 0 else None
    mv = [m for m in moves]
    return dict(
        film=os.path.basename(a['path']), dur=round(a['dur'], 1), fps=round(fps, 2), size=f"{a['w']}x{a['h']}",
        cuts=len(cuts), shots_per_min=round(len(cuts) / a['dur'] * 60, 1), shot_median=round(float(np.median(shots)), 2),
        cam_moving=round(float(moving.mean()) * 100), cam_speed_med=round(float(np.median(cam[moving])) if moving.any() else 0, 1),
        cam_speed_p90=round(float(np.percentile(cam[moving], 90)) if moving.any() else 0, 1),
        zoom_med=round(float(np.median(zs[zs > 0.5])) if (zs > 0.5).any() else 0, 1),
        moves=len(mv), move_dur_med=round(float(np.median([m['dur'] for m in mv])), 2) if mv else 0,
        t_peak_med=round(float(np.median([m['t_peak'] for m in mv])), 2) if mv else 0,
        smooth_med=round(float(np.median([m['smooth'] for m in mv])), 2) if mv else 0,
        jerk_med=round(float(np.median([m['jerk'] for m in mv])), 1) if mv else 0,
        holds=len(hold_runs), hold_share=round(float(holds.mean()) * 100), hold_med=round(float(np.median(hold_runs)), 2) if hold_runs else 0,
        energy_med=round(float(np.median(e)) * 1000, 1), energy_p90=round(float(np.percentile(e, 90)) * 1000, 1),
        rhythm_s=round(rhythm, 2) if rhythm else None,
        _cuts=cuts, _cut_frames=cut, _moves=moves)


def draw(a, s, png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fps = a['fps']
    t = np.arange(len(a['pan'])) / fps
    fig, ax = plt.subplots(3, 1, figsize=(16, 7), sharex=True)
    k = max(1, int(fps * 0.1))
    ok = ~s['_cut_frames']  # across a cut the two frames share nothing, so its "shift" is noise: leave it out
    pan_s = smooth(np.where(ok, a['pan'] * fps * 100, 0), k)
    ax[0].plot(t, pan_s, lw=1, color='#2f55d4')
    ax[0].set_ylabel('pan %w/s')
    ax[0].set_ylim(0, max(10, float(pan_s.max()) * 1.2))
    ax[1].plot(t, smooth(np.where(ok, a['zoom'] * fps * 100, 0), k), lw=1, color='#e07a00')
    ax[1].axhline(0, color='#999', lw=0.5)
    ax[1].set_ylabel('zoom %/s (+in)')
    ax[2].plot(t, a['energy'] * 1000, lw=0.8, color='#333')
    ax[2].set_ylabel('change')
    for c in s['_cuts']:
        for x in ax:
            x.axvline(c / fps, color='#d00', lw=0.6, alpha=0.6)
    ax[2].set_xlabel('seconds (red = cut)')
    ax[0].set_title(f"{s['film']}: {s['shots_per_min']} cuts/min, camera moving {s['cam_moving']}% of the time, "
                    f"median move {s['move_dur_med']} s peaking at {s['t_peak_med']}, holds {s['hold_share']}%", fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=80)
    plt.close(fig)


def contact(path, png, dur, cols=10):
    n = int(min(dur, 120))
    tmp = png + '.tmp.png'
    w, h, _, _ = probe(path)
    th = 120 if w >= h else 180
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', path, '-vf', f'fps=1,scale=-2:{th},tile={cols}x{math.ceil(n / cols)}',
                    '-frames:v', '1', tmp], check=True)
    os.replace(tmp, png)


def main():
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0 if args else 2)
    out = 'eye-out'
    if '--out' in args:
        i = args.index('--out')
        out = args[i + 1]
        del args[i:i + 2]
    os.makedirs(out, exist_ok=True)
    rows = []
    for p in args:
        a = analyse(p)
        s = summarise(a)
        base = os.path.splitext(os.path.basename(p))[0]
        draw(a, s, os.path.join(out, f'{base}-motion.png'))
        contact(p, os.path.join(out, f'{base}-sheet.png'), a['dur'])
        with open(os.path.join(out, f'{base}-moves.json'), 'w') as f:
            json.dump(dict(cuts_s=[round(c / a['fps'], 2) for c in s['_cuts']], moves=s['_moves']), f)
        rows.append({k: v for k, v in s.items() if not k.startswith('_')})
        print(json.dumps(rows[-1]))
    keys = list(rows[0].keys())
    md = ['| ' + ' | '.join(keys) + ' |', '|' + '---|' * len(keys)]
    for r in rows:
        md.append('| ' + ' | '.join(str(r[k]) for k in keys) + ' |')
    legend = """
cuts/shots: hard cuts found; shot_median in s. cam_moving: % of frames where the camera pans or zooms (> 2 % of the
width per second). cam_speed: % of the width per second while moving. zoom_med: |zoom| % per second while zooming.
moves: camera moves of 0.25 s+; t_peak_med: where in a move the speed peaks (0.5 = symmetric ease, < 0.5 = fast start,
slow settle); smooth_med: mean/peak speed (high = flat, robotic constant speed; ~0.5-0.65 = eased); jerk_med: how
jittery the speed curve is (lower = smoother). holds: still stretches of 0.3 s+, hold_share % of frames.
energy: mean frame change x1000 (busy-ness); rhythm_s: the strongest repeat in the busy-ness (a beat, if any).
"""
    with open(os.path.join(out, 'eye.md'), 'w') as f:
        f.write('\n'.join(md) + '\n' + legend)
    print(f'wrote {out}/eye.md')


if __name__ == '__main__':
    main()

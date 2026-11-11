#!/usr/bin/env python3
"""Self-test: build a film whose motion and sound are known, then check that ear.py and eye.py report them.

    python3 tests/selftest.py            (needs ffmpeg, numpy, matplotlib; takes about half a minute)

What is planted (see examples/make_test_clip.sh):
  picture  a 4 s pan with a cosine ease (top speed 78.5 % of the width a second, exactly halfway; mean/peak 2/pi),
           a cut at 4 s, a 4 s hold, a cut at 8 s, a 4 s zoom in that starts at 15 % a second
  sound    a bell at 1.0 s (1320 Hz), a thud at 3.0 s, a tap at 5.0 s, a voice-like tone at 7.0 s (220 Hz),
           a whoosh at 9.0 s

Exit code 0 when every check passes, 1 otherwise.
"""
import json, math, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []


def check(name, got, want, tol):
    ok = got is not None and abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got} (planted {want}, tolerance {tol})")
    if not ok:
        fails.append(name)


def same(name, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got}" + ('' if ok else f' (planted {want})'))
    if not ok:
        fails.append(name)


def run(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{' '.join(cmd[:3])} ... failed:\n{r.stderr[-2000:]}")


with tempfile.TemporaryDirectory() as tmp:
    clip = os.path.join(tmp, 'testclip.mp4')
    run('bash', os.path.join(ROOT, 'examples', 'make_test_clip.sh'), clip)
    run(sys.executable, os.path.join(ROOT, 'scripts', 'ear.py'), clip, '--out', os.path.join(tmp, 'ear'))
    run(sys.executable, os.path.join(ROOT, 'scripts', 'eye.py'), clip, '--out', os.path.join(tmp, 'eye'))

    print('ear.py')
    events = json.load(open(os.path.join(tmp, 'ear', 'events.json')))['mix']
    same('classes in order', [e['class'] for e in events], ['BELL', 'THUD', 'TAP', 'VOICE', 'WHOOSH'])
    for e, t in zip(events, [1.0, 3.0, 5.0, 7.0, 9.0]):
        check(f"{e['class']} starts at (s)", round(e['t'], 2), t, 0.03)
    if len(events) == 5:
        check('bell tone (Hz)', events[0]['f_tail'], 1320, 15)
        check('voice pitch (Hz)', events[3]['f0'], 220, 5)

    print('eye.py')
    eye = json.load(open(os.path.join(tmp, 'eye', 'testclip-moves.json')))
    same('number of cuts', len(eye['cuts_s']), 2)
    for got, want in zip(eye['cuts_s'], [4.0, 8.0]):
        check('cut at (s)', got, want, 0.07)
    same('number of camera moves', len(eye['moves']), 2)
    if len(eye['moves']) == 2:
        pan, zoom = eye['moves']
        check('pan top speed (% of width a second)', pan['peak'], 78.5, 2.0)
        check('pan peaks at (share of the move)', pan['t_peak'], 0.5, 0.05)
        check('pan mean/peak speed', pan['smooth'], round(2 / math.pi, 2), 0.04)
        check('zoom top speed (% a second)', round(zoom['peak'] * 2, 1), 15.0, 1.5)
        check('zoom peaks at (share of the move)', zoom['t_peak'], 0.0, 0.15)

print('\nall checks pass' if not fails else f'\n{len(fails)} check(s) failed: ' + '; '.join(fails))
sys.exit(1 if fails else 0)

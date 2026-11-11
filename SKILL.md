---
name: film-ear-eye
description: Measure how a video sounds and moves when you cannot watch or hear it. Use before calling any film, reel, animation or edit "done", when comparing your cut against reference films, when asked whether the pacing, camera easing, sound design or mix is right, or when a sound effect might be read as the wrong thing (a bell, a voice, a buzz). Produces numbers, labelled spectrograms and motion plots from any mp4, wav or mp3.
---

# Film ear + eye

You cannot hear a mix or feel a camera move. These two tools turn both into numbers and pictures you can read,
so "it sounds fine" and "the pacing works" become claims you can check.

- `scripts/ear.py`: every sound event in the mix, timed and classed (THUD, TAP, BELL, VOICE, NOTE, WHOOSH, BOOM),
  with the numbers behind each call; loudness (LUFS, range, true peak), silences, key, pulse, energy by band; one
  labelled spectrogram per window.
- `scripts/eye.py`: cuts, how much the camera moves, how fast, how every move eases, holds, busy-ness; one motion
  plot and a contact sheet per film; one table across films.

Both need `ffmpeg`, `numpy` and `matplotlib`. Nothing leaves the machine.

## When to run them

1. **Before you call a film done.** Always both. Report the numbers, not an impression.
2. **Before you start**, on two or three reference films the owner likes. Their table is the target.
3. **On each sound effect before it goes in the mix** (`ear.py --samples`), to catch a "soft click" that is really
   a bell.

## Run

```bash
python3 scripts/eye.py refs/a.mp4 refs/b.mp4 out/mine.mp4 --out report/eye
python3 scripts/ear.py out/mine.mp4 --out report/ear              # the mix
python3 scripts/ear.py out/mine.mp4 --stems stems/ --out report/ear   # plus each stem (*.wav) on its own
python3 scripts/ear.py --samples sfx/click.wav sfx/pop.wav        # class samples before mixing
```

Then **read the outputs yourself**: `report/eye/eye.md` (the table), `report/eye/<film>-motion.png`,
`report/ear/ear.md`, and every `report/ear/ear-*.png`. Look at the pictures; the table alone hides where in the film
a problem sits.

## Reading eye.py

| column | meaning | what to look for |
|---|---|---|
| `shots_per_min`, `shot_median` | hard cuts | compare with the references; a dialogue piece and a music edit differ by 3× |
| `cam_moving` | % of frames where the camera pans or zooms | far below the references = static, dead frames |
| `cam_speed_med`, `cam_speed_p90` | % of the width per second while moving | p90 far above the references = whip moves |
| `zoom_med` | % per second while zooming, + is in | fast zoom-ins read as cheap |
| `t_peak_med` | where in a move the speed peaks | 0.5 = symmetric ease; near 0 = starts at full speed (no ease-in); near 1 = slams to a stop |
| `smooth_med` | mean speed / peak speed | ~0.5–0.65 = eased; > 0.85 = constant speed, robotic |
| `jerk_med` | jitter of the speed curve | high = stutter, dropped frames, or a hand-keyed move |
| `hold_share`, `hold_med` | stretches where nothing changes | a punchline needs a hold; a film with none never breathes |
| `rhythm_s` | strongest repeat in busy-ness | matches the beat when cut to music |

`<film>-moves.json` lists every cut and every move (`t`, `dur`, `peak`, `t_peak`, `smooth`, `jerk`) so you can name
the exact move that is wrong.

## Reading ear.py

- **Event table**: `t`, class, *why* it got that class, peak dB, ring time. Check each event against what you meant
  to put there. A UI click classed BELL will be heard as a bell. Something classed VOICE that is not a voice will
  be heard as someone humming.
- **Loudness line**: integrated LUFS, loudness range, true peak. Typical social target: about −14 LUFS, true peak
  ≤ −1 dBFS.
- **Silences**: stretches under −55 dBFS for 0.5 s or more. In a finished film an unplanned one is a hole.
- **Energy by band**: a "presence" share that dwarfs the rest is harshness; almost no low end is thin.
- **Spectrogram**: each event is marked at its time with its class. Vertical lines = clicks; horizontal lines =
  tones; a soft cloud = noise or a whoosh.

Options: `--words` adds what a local speech model hears in the voice (to catch mumbled lines); `--tags` adds what a
trained listener calls each event. Both are optional and need extra packages; see the README.

## Limits: say these when you report

- An event's "peak dB" is its level in the first 150 ms. A sound that swells (a whoosh) reads low there.
- Classes are rules on measured numbers, not a judgement of quality. They tell you what a sound will be *taken for*.
- Camera motion is measured from the whole picture. A big object crossing a still frame reads as a small pan.
  Flat or repeating pictures (stripes, a plain wall) give weak estimates; the motion plot shows it as noise.
- Neither tool sees design, acting or whether a joke lands. Good numbers are necessary, never sufficient.

## Check the tools themselves

`python3 tests/selftest.py` builds a 12 s film with planted motion and sounds and checks both reports against it
(17 checks). Run it after changing either script.

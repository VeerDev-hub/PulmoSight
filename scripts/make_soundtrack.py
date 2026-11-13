"""Synthesise the PulmoSight trailer soundtrack (46 s ambient score) with numpy only.

    python scripts/make_soundtrack.py            (writes assets/trailer/soundtrack.wav)
"""
import sys
import wave

import numpy as np

SR, DUR = 44100, 46.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(7)
hz = lambda midi: 440.0 * 2 ** ((midi - 69) / 12)

# A minor progression, one chord every 5.75 s: Am F C G Am F C G
CHORDS = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)] * 2
SEG = DUR / len(CHORDS)


def envelope(start, length, attack, release):
    e = np.zeros(N)
    i0, i1 = int(start * SR), min(N, int((start + length) * SR))
    x = (np.arange(i0, i1) / SR) - start
    e[i0:i1] = np.clip(x / attack, 0, 1) * np.clip((length - x) / release, 0, 1)
    return e


def lowpass(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x); acc = 0.0
    for i in range(len(x)):          # one-pole filter
        acc = (1 - a) * x[i] + a * acc; y[i] = acc
    return y


left = np.zeros(N); right = np.zeros(N)

# Pads: three detuned voices per chord note, soft harmonics, slow attack, overlapping release
for k, chord in enumerate(CHORDS):
    env = envelope(k * SEG, SEG + 1.6, 1.8, 2.2)
    for j, note in enumerate(chord):
        f = hz(note)
        for det, pan in ((-0.004, 0.25), (0.0, 0.5), (0.0045, 0.75)):
            ph = rng.uniform(0, 2 * np.pi)
            v = sum(np.sin(2 * np.pi * f * (1 + det) * h * t + ph * h) / h ** 1.6 for h in (1, 2, 3, 4))
            v *= env * 0.045
            left += v * (1 - pan); right += v * pan
    bass = np.sin(2 * np.pi * hz(chord[0] - 12) * t) * envelope(k * SEG, SEG + 0.8, 0.6, 1.2) * 0.11
    left += bass; right += bass

# Plucked arpeggio (from the "inside the network" scene to the end card), 96 BPM eighth notes, ping-pong echo
step = 60 / 96 / 2
pl = np.zeros(N); pr = np.zeros(N)
s = 15.6
while s < 42.0:
    k = min(int(s // SEG), len(CHORDS) - 1)
    pattern = [0, 1, 2, 1, 2, 0, 2, 1]
    note = CHORDS[k][pattern[int(round(s / step)) % 8]] + 12
    i0 = int(s * SR); L = int(1.2 * SR); x = np.arange(L) / SR
    tone = (np.sin(2 * np.pi * hz(note) * x) + 0.3 * np.sin(4 * np.pi * hz(note) * x)) * np.exp(-x * 5.5) * 0.05
    seg = slice(i0, min(N, i0 + L)); n = seg.stop - seg.start
    pl[seg] += tone[:n]; pr[seg] += tone[:n] * 0.7
    s += step
d = int(0.234 * SR)
for rep in range(1, 5):                  # echoes alternate left/right
    g = 0.42 ** rep
    src = pl if rep % 2 else pr
    (pr if rep % 2 else pl)[d * rep:] += src[:-d * rep] * g
left += pl; right += pr

# Impacts with a noise riser before each: logo (11.0 s), ensemble orb (32.5 s), end card (42.2 s)
for at in (11.0, 32.5, 42.2):
    i0 = int(at * SR); L = int(2.5 * SR); x = np.arange(L) / SR
    boom = np.sin(2 * np.pi * (55 - 20 * x / 2.5) * x) * np.exp(-x * 2.2) * 0.32
    seg = slice(i0, min(N, i0 + L)); n = seg.stop - seg.start
    left[seg] += boom[:n]; right[seg] += boom[:n]
    r0 = int((at - 1.6) * SR); RL = int(1.6 * SR)
    noise = lowpass(rng.normal(0, 1, RL), 2500) * (np.linspace(0, 1, RL) ** 2.5) * 0.05
    left[r0:r0 + RL] += noise; right[r0:r0 + RL] += noise[::-1] * 0.9

# Master: gentle low-pass on everything, fades, normalise to -3 dBFS peak
left, right = lowpass(left, 9000), lowpass(right, 9000)
fade = np.clip(t / 2.0, 0, 1) * np.clip((DUR - t) / 3.5, 0, 1)
stereo = np.stack([left * fade, right * fade], axis=1)
stereo *= 10 ** (-3 / 20) / np.abs(stereo).max()

out = sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parent.parent / "assets/trailer/soundtrack.wav")
with wave.open(out, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((stereo * 32767).astype("<i2").tobytes())
print("wrote", out, f"{DUR:.0f} s")

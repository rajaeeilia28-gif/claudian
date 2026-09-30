"""V2 score and sound design: 30 s, 120 BPM, A minor -> C major on the brand impact.

Built for small speakers: lean low end (kick with a mid "knock" and click, bass on offbeats with
harmonics instead of a sub sine), accents on every cut with mid/high content, linear loudness
normalisation (static gain only) to -14 LUFS, 48 kHz / 24-bit export plus stems.

Usage: v2_music.py [out_dir]   -> score_v2.wav, stems/music.wav, stems/accents.wav
"""
import os
import sys
import wave

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "qa")))
import loudness as L  # noqa: E402
import v2_spec as S  # noqa: E402

SR = 48000
N = int(round(S.DURATION * SR))
BEAT = 0.5
rng = np.random.default_rng(2610)


# ---------------------------------------------------------------- dsp helpers
def tt(n):
    return np.arange(n) / SR


def lp(x, fc, order=2):
    return sosfilt(butter(order, min(fc, SR * 0.45), "low", fs=SR, output="sos"), x, axis=0)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x, axis=0)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, min(hi, SR * 0.45)], "band", fs=SR, output="sos"), x, axis=0)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def saw(freq, n, phase=0.0):
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,))
    dt = f / SR
    ph = (phase + np.cumsum(dt)) % 1.0
    y = 2 * ph - 1
    m = ph < dt
    x = ph[m] / dt[m]
    y[m] -= x + x - x * x - 1
    m = ph > 1 - dt
    x = (ph[m] - 1) / dt[m]
    y[m] -= x * x + x + x + 1
    return y


def adsr(n, a, d, s, r):
    t = tt(n)
    e = np.where(t < a, t / max(a, 1e-4), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    rn = int(r * SR)
    if 0 < rn < n:
        e[-rn:] *= np.linspace(1, 0, rn)
    return e


def stereo(x, pan=0.0):
    ang = (pan + 1) * np.pi / 4
    return np.stack([x * np.cos(ang), x * np.sin(ang)], 1) * np.sqrt(2)


class Bus:
    def __init__(self):
        self.x = np.zeros((N, 2))

    def add(self, sig, t0, gain=1.0, pan=0.0):
        if sig.ndim == 1:
            sig = stereo(sig, pan)
        i = int(round(t0 * SR))
        if i >= N:
            return
        n = min(len(sig), N - i)
        self.x[i:i + n] += sig[:n] * gain


def reverb_ir(seconds, decay, bright, seed):
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    ir = r.standard_normal((n, 2)) * np.exp(-tt(n) / decay)[:, None]
    ir = lp(ir, bright)
    k = int(0.01 * SR)
    ir[:k] *= np.linspace(0, 1, k)[:, None]
    return ir / np.sqrt((ir ** 2).sum(0))


def reverb(x, ir, wet):
    y = np.stack([fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)
    return x * (1 - wet) + y * wet


def pingpong(x, d, fb, taps):
    out = x.copy()
    k = int(d * SR)
    for j in range(1, taps + 1):
        sh = np.zeros_like(x)
        sh[k * j:] = x[:len(x) - k * j]
        out += (sh[:, ::-1] if j % 2 else sh) * fb ** j
    return out


# ---------------------------------------------------------------- instruments
def kick(level=1.0):
    n = int(0.32 * SR)
    t = tt(n)
    f = 58 + 120 * np.exp(-t / 0.028)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.12)
    knock = np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.035) * 0.55
    click = hp(rng.standard_normal(n), 2500) * np.exp(-t / 0.004) * 0.5
    return hp(body + knock + click, 38) * level


def clap():
    n = int(0.4 * SR)
    t = tt(n)
    z = bp(rng.standard_normal(n), 900, 5000)
    e = np.zeros(n)
    for o in (0.0, 0.010, 0.021):
        i = int(o * SR)
        e[i:] += np.exp(-t[:n - i] / 0.008)
    e += np.exp(-t / 0.12) * 0.5
    return z * e * 0.55


def hat(decay=0.02):
    n = int(max(0.08, decay * 6) * SR)
    t = tt(n)
    return hp(rng.standard_normal(n), 7000, 4) * np.exp(-t / decay) * 0.45


def snare(pitch=1.0):
    n = int(0.22 * SR)
    t = tt(n)
    tone = np.sin(2 * np.pi * 200 * pitch * t) * np.exp(-t / 0.05)
    z = bp(rng.standard_normal(n), 1500, 9000) * np.exp(-t / 0.07)
    return (tone * 0.5 + z * 0.8) * 0.6


def bass(freq, dur):
    n = int(dur * SR)
    t = tt(n)
    y = 0.55 * saw(freq, n) + 0.35 * saw(freq * 1.006, n, 0.3) + 0.3 * np.sin(2 * np.pi * freq * t)
    return lp(y, 750) * adsr(n, 0.004, 0.12, 0.55, 0.03)


def pluck(freq, dur=0.55):
    n = int(dur * SR)
    t = tt(n)
    y = np.zeros(n)
    for k in range(1, 14):
        if freq * k > 12000:
            break
        y += np.sin(2 * np.pi * freq * k * t + k) / k ** 1.1 * np.exp(-t * (5 + k * 5.5))
    return y * np.minimum(1, t / 0.002) * 0.35


def pad(midis, dur, cutoff=2400, attack=0.3, release=0.5):
    n = int(dur * SR)
    out = np.zeros((n, 2))
    for m in midis:
        for v, det in enumerate((-0.11, -0.04, 0.04, 0.11)):
            s = saw(mtof(m) * 2 ** (det / 12), n, rng.random())
            out += stereo(s, pan=(v - 1.5) / 1.5 * 0.7)
    out = lp(out, cutoff) / (len(midis) * 4)
    return out * adsr(n, attack, 1, 1, release)[:, None]


def stab(midis, dur=1.4, cutoff=3200):
    n = int(dur * SR)
    t = tt(n)
    y = sum(saw(mtof(m), n, rng.random()) + 0.5 * saw(mtof(m) * 1.004, n, rng.random()) for m in midis)
    y = lp(y / len(midis), cutoff) * np.exp(-t / 0.55) * np.minimum(1, t / 0.003)
    return y


def bell(freq, dur=3.0, idx=1.6):
    n = int(dur * SR)
    t = tt(n)
    i = idx * np.exp(-t / 0.5)
    y = np.sin(2 * np.pi * freq * t + i * np.sin(2 * np.pi * freq * 3.5 * t))
    y += 0.35 * np.sin(2 * np.pi * freq * 2 * t) * np.exp(-t / 0.7)
    return y * np.exp(-t / (dur * 0.33)) * np.minimum(1, t / 0.002) * 0.3


def crash(dur=1.6, decay=0.55):
    n = int(dur * SR)
    t = tt(n)
    z = hp(rng.standard_normal((n, 2)), 3500) + 0.5 * bp(rng.standard_normal((n, 2)), 6000, 11000)
    return z * (np.exp(-t / decay) * np.minimum(1, t / 0.0015))[:, None] * 0.4


def accent_hit():
    """Cut accent: pitched tom body with harmonics + bright transient; attack < 2 ms."""
    n = int(0.45 * SR)
    t = tt(n)
    f = 128 + 45 * np.exp(-t / 0.03)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = (np.sin(ph) + 0.45 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)) * np.exp(-t / 0.18)
    tr = bp(rng.standard_normal(n), 2000, 8000) * np.exp(-t / 0.012)
    y = body * 0.8 + tr * 1.0
    return y * np.minimum(1, t / 0.0008)


def riser(dur, f0=400, f1=9000):
    n = int(dur * SR)
    z = rng.standard_normal((n, 2))
    out = np.zeros((n, 2))
    blocks = 40
    edges = np.linspace(0, n, blocks + 1).astype(int)
    for b in range(blocks):
        fc = f0 * (f1 / f0) ** (b / blocks)
        seg = bp(z, max(60, fc * 0.7), min(20000, fc * 1.4))
        out[edges[b]:edges[b + 1]] = seg[edges[b]:edges[b + 1]]
    return out * ((tt(n) / dur) ** 2)[:, None]


# ---------------------------------------------------------------- arrangement
CHORDS = {  # bass midi, pad voicing, arp notes
    "Am": (45, [57, 60, 64, 67], [69, 72, 76, 79]),
    "F": (41, [53, 57, 60, 64], [65, 69, 72, 76]),
    "C": (48, [55, 60, 64, 67], [67, 72, 76, 79]),
    "G": (43, [55, 59, 62, 67], [67, 71, 74, 79]),
}
TIMELINE = [(0, 4, "Am"), (4, 6, "Am"), (6, 8, "F"), (8, 10, "C"), (10, 12, "G"), (12, 14, "Am"),
            (14, 16, "F"), (16, 18, "C"), (18, 20, "G"), (20, 22, "Am"), (22, 23, "F"), (23, 26, "G"),
            (26, 30, "C")]
GROOVE = (S.LIFT_T, 23.0)
BUILD = (23.0, 25.4)


def build():
    drums, bassb, music, acc = Bus(), Bus(), Bus(), Bus()
    kicks = []

    # pad + arp through every section (no silent passage)
    for t0, t1, name in TIMELINE:
        bm, pv, arp = CHORDS[name]
        dur = t1 - t0
        if t0 < S.LIFT_T:
            music.add(pad(pv, dur + 0.4, cutoff=1400, attack=0.8), t0, 0.13)
        elif t0 >= S.IMPACT_T:
            music.add(pad([48, 55, 60, 64, 67, 74], dur, cutoff=2600, attack=0.02, release=1.6), t0, 0.24)
        else:
            music.add(pad(pv, dur + 0.3, cutoff=1800 if t0 < 8 else (2200 if t0 < 12 else (2600 if t0 < 23 else 3400))),
                      t0, 0.17 if t0 < 8 else (0.24 if t0 < 12 else (0.31 if t0 < 23 else 0.40)))
        step = 0.25 if t0 < S.LIFT_T else 0.125
        g = 0.06 if t0 < S.LIFT_T else (0.09 if t0 < 8 else (0.14 if t0 < 12 else 0.22))
        if t0 >= S.IMPACT_T:
            continue
        for k, s in enumerate(np.arange(t0, t1, step)):
            note = arp[[0, 1, 2, 3, 2, 1, 3, 2][k % 8]] + (12 if (k // 8) % 2 else 0)
            music.add(pluck(mtof(note)), float(s), g * (1.0 if k % 4 == 0 else 0.7), pan=0.3 * np.sin(k))

    # intro texture and swell into the lift
    for s in np.arange(1.0, S.LIFT_T, 0.125):
        drums.add(hat(0.012), float(s), 0.05 + 0.06 * (s - 1) / 3, pan=0.25)
    swell = crash(1.0, 0.25)[::-1] * 0.8
    acc.add(swell, S.LIFT_T - 1.0, 0.7)

    # groove: kick on every beat, clap on 2 and 4, offbeat hats and bass
    for b in np.arange(S.LIFT_T, BUILD[1], BEAT):
        b = float(b)
        drums.add(kick(), b, 0.26 if b < 8.0 else (0.34 if b < 12.0 else 0.42))
        kicks.append(b)
        beat = int(round((b - S.LIFT_T) / BEAT)) % 4
        if beat in (1, 3) and 8.0 <= b < GROOVE[1]:
            drums.add(clap(), b, 0.42)
        if beat == 3 and 12.0 <= b < GROOVE[1]:
            drums.add(hat(0.09), b + 0.25, 0.14, pan=-0.2)
        drums.add(hat(), b + 0.25, 0.20 if b < 8.0 else 0.34, pan=0.2)
        drums.add(hat(0.012), b + 0.125, 0.16, pan=-0.3)
        drums.add(hat(0.012), b + 0.375, 0.16, pan=-0.3)
    for t0, t1, name in TIMELINE:
        if t1 <= S.LIFT_T or t0 >= S.IMPACT_T:
            continue
        root = mtof(CHORDS[name][0])
        for k, s in enumerate(np.arange(max(t0, S.LIFT_T), min(t1, BUILD[1]), BEAT)):
            f = root * (2 if k % 4 == 3 else 1)
            bassb.add(bass(f, 0.24), float(s) + 0.25, 0.22 if s < 8.0 else 0.30)

    # lift at 4.0 s: crash + chord stab
    acc.add(crash(1.8, 0.6), S.LIFT_T, 0.45)
    acc.add(stereo(stab([57, 60, 64, 69], 1.2)), S.LIFT_T, 0.40)
    # cut accents
    for a in S.ACCENTS:
        acc.add(accent_hit(), a, 0.30)
        acc.add(crash(0.6, 0.12), a, 0.22)
    # build 23 - 25.4: snare roll, riser; 25.4 - 26.0 pad + riser only (no silent break)
    for s in np.arange(BUILD[0], BUILD[1], 0.25):
        drums.add(snare(1.0), float(s), 0.18 + 0.12 * (s - BUILD[0]) / 2.4)
    for s in np.arange(24.0, BUILD[1], 0.125):
        drums.add(snare(1.05), float(s), 0.12 + 0.2 * (s - 24.0) / 1.4)
    rz = riser(S.IMPACT_T - BUILD[0] - 0.05) * 0.32
    dip = np.ones(len(rz))
    k0 = int((S.IMPACT_T - 0.6 - BUILD[0]) * SR)
    dip[k0:] = np.linspace(1, 0.3, len(dip) - k0)
    acc.add(rz * dip[:, None], BUILD[0], 1.0)
    # brand impact at 26.0 s: body + crash + C major stab + bell arpeggio
    acc.add(kick(1.0), S.IMPACT_T, 0.45)
    acc.add(accent_hit(), S.IMPACT_T, 0.40)
    acc.add(crash(3.0, 0.9), S.IMPACT_T, 0.55)
    acc.add(stereo(stab([48, 55, 60, 64, 67, 72], 2.2, 3600)), S.IMPACT_T, 1.0)
    acc.add(stereo(stab([60, 64, 67, 72, 76], 1.6, 5000)), S.IMPACT_T, 0.45)
    for j, m in enumerate([72, 76, 79, 84]):
        acc.add(bell(mtof(m), 3.2), S.IMPACT_T + 0.25 * j, 0.35, pan=(-0.3, 0.3, -0.15, 0.15)[j])

    # 0.6 s before the brand impact: pad eases back to 70 % (no silent break)
    i0, i1 = int((S.IMPACT_T - 0.6) * SR), int(S.IMPACT_T * SR)
    music.x[i0:i1] *= np.linspace(1.0, 0.7, i1 - i0)[:, None]

    # sidechain: light ducking of bass and pad by the kick
    sc = np.ones(N)
    for k in kicks:
        i = int(k * SR)
        m = min(int(0.3 * SR), N - i)
        sc[i:i + m] = np.minimum(sc[i:i + m], 1 - 0.55 * np.exp(-tt(m) / 0.07))
    bassb.x *= sc[:, None]

    small = reverb_ir(0.9, 0.25, 9000, 3)
    hall = reverb_ir(2.8, 0.8, 7000, 5)
    drums_mix = reverb(drums.x, small, 0.10)
    tonal_mix = bassb.x + reverb(pingpong(music.x, 0.375, 0.28, 4), hall, 0.25)
    acc_mix = reverb(acc.x, hall, 0.22)
    return drums_mix, tonal_mix, acc_mix


def mono_low(x, fc=120):
    """Keep everything below fc in mono (side channel high-passed)."""
    m = (x[:, 0] + x[:, 1]) / 2
    s = hp((x[:, 0] - x[:, 1]) / 2, fc, 4)
    return np.stack([m + s, m - s], 1)


def soft(x, thr):
    """Per-bus saturation (sound design): transparent below thr, rounds peaks above it."""
    return thr * np.tanh(x / thr)


def master(drums_mix, tonal_mix, acc_mix, target=-14.0, tp_max=-1.2):
    """Saturate the percussive buses just enough that the statically normalised mix meets the
    true-peak ceiling. Loudness normalisation itself is a single static gain."""
    fade = np.clip((29.8 - tt(N)) / 0.6, 0, 1)[:, None]
    prep = [hp(mono_low(x), 30, 2) * fade for x in (drums_mix, tonal_mix, acc_mix)]
    thr = max(np.abs(prep[0]).max(), np.abs(prep[2]).max())
    for _ in range(40):
        d, a = soft(prep[0], thr), soft(prep[2], thr)
        mix = d + prep[1] + a
        gain = 10 ** ((target - L.integrated(mix)) / 20)
        if L.true_peak_db(mix * gain) <= tp_max:
            break
        thr *= 0.9
    music = (d + prep[1]) * gain
    accents = a * gain
    return music + accents, [music, accents], gain


def write24(path, x):
    x = np.clip(x + (rng.random(x.shape) - rng.random(x.shape)) / 2 ** 23, -1, 1 - 2 ** -23)
    ints = np.round(x * (2 ** 23 - 1)).astype("<i4")
    data = ints.reshape(-1, 1).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(SR)
        w.writeframes(data)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(os.path.join(HERE, "..", "out", "audio"))
    os.makedirs(os.path.join(out, "stems"), exist_ok=True)
    mix, stems, gain = master(*build())
    write24(os.path.join(out, "score_v2.wav"), mix)
    write24(os.path.join(out, "stems", "music.wav"), stems[0])
    write24(os.path.join(out, "stems", "accents.wav"), stems[1])
    print(f"gain {20 * np.log10(gain):+.2f} dB | integrated {L.integrated(mix):.2f} LUFS | "
          f"true peak {L.true_peak_db(mix):.2f} dBTP | LRA {L.lra(mix):.1f} LU")

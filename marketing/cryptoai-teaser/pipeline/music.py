"""Original score + sound design for the Crypto AI teaser, synthesized from scratch.

120 BPM (beat = 0.5 s). Structure mirrors spot.py:
  0-3    market noise swell, hard cut to silence
  3-6    bell, heartbeat, riser
  6-46   groove (Am F C G), breaks under the text cards
  46-50  hush for the tagline, reverse swell
  50-59  impact + resolving C chord under the logo
"""
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt

SR = 48000
DUR = 59.0
N = int(SR * DUR)
BEAT = 0.5
rng = np.random.default_rng(2026)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tt(n):
    return np.arange(n) / SR


def lp(x, fc, order=2):
    return sosfilt(butter(order, min(fc, SR * 0.45), "low", fs=SR, output="sos"), x, axis=0)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x, axis=0)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, min(hi, SR * 0.45)], "band", fs=SR, output="sos"), x, axis=0)


def saw(freq, n, phase0=0.0):
    """PolyBLEP band-limited saw; freq may be scalar or per-sample array."""
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,))
    dt = f / SR
    ph = (phase0 + np.cumsum(dt)) % 1.0
    y = 2 * ph - 1
    m1 = ph < dt
    x = ph[m1] / dt[m1]
    y[m1] -= x + x - x * x - 1
    m2 = ph > 1 - dt
    x = (ph[m2] - 1) / dt[m2]
    y[m2] -= x * x + x + x + 1
    return y


def env_adsr(n, a, d, s, r, hold):
    t = tt(n)
    e = np.where(t < a, t / max(a, 1e-4), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    rel = t > hold
    e[rel] *= np.exp(-(t[rel] - hold) / max(r, 1e-4))
    return e


class Bus:
    def __init__(self):
        self.x = np.zeros((N, 2))

    def add(self, sig, t0, gain=1.0, pan=0.0):
        i = int(round(t0 * SR))
        if i >= N:
            return
        if sig.ndim == 1:
            l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
            sig = np.stack([sig * l * 1.414, sig * r * 1.414], 1)
        if i < 0:
            sig = sig[-i:]
            i = 0
        n = min(len(sig), N - i)
        self.x[i:i + n] += sig[:n] * gain


def reverb_ir(seconds, decay, bright=6000, seed=1):
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = tt(n)
    ir = r.standard_normal((n, 2)) * np.exp(-t / decay)[:, None]
    ir = lp(ir, bright)
    ir[: int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))[:, None]
    return ir / np.sqrt((ir ** 2).sum(0))


def reverb(x, ir, mix=0.3):
    wet = np.stack([fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], 1)
    return x * (1 - mix) + wet * mix


def delay_pp(x, d, fb=0.4, n=6):
    out = x.copy()
    k = int(d * SR)
    for j in range(1, n + 1):
        g = fb ** j
        sh = np.zeros_like(x)
        sh[k * j:] = x[: len(x) - k * j]
        if j % 2:
            sh = sh[:, ::-1]
        out += sh * g
    return out


# ------------------------------------------------------------------ instruments
def kick(level=1.0):
    n = int(0.55 * SR)
    t = tt(n)
    f = 46 + 110 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.28)
    click = hp(rng.standard_normal(n), 2500) * np.exp(-t / 0.004) * 0.35
    return np.tanh((body + click) * 1.6) * level


def clap():
    n = int(0.45 * SR)
    t = tt(n)
    z = bp(rng.standard_normal(n), 900, 4200)
    e = np.zeros(n)
    for o in (0.0, 0.011, 0.022):
        i = int(o * SR)
        e[i:] += np.exp(-(t[: n - i]) / 0.009)
    e += np.exp(-t / 0.13) * 0.55
    return z * e * 0.6


def hat(open_=False):
    n = int((0.35 if open_ else 0.08) * SR)
    t = tt(n)
    z = hp(rng.standard_normal(n), 7500, 4)
    return z * np.exp(-t / (0.12 if open_ else 0.022)) * 0.5


def snare(pitch=1.0):
    n = int(0.25 * SR)
    t = tt(n)
    tone = np.sin(2 * np.pi * 190 * pitch * t) * np.exp(-t / 0.05)
    z = bp(rng.standard_normal(n), 1500, 9000) * np.exp(-t / 0.08)
    return (tone * 0.5 + z * 0.8) * 0.7


def pluck(freq, dur=0.7):
    n = int(dur * SR)
    t = tt(n)
    y = np.zeros(n)
    for k in range(1, 14):
        if freq * k > 12000:
            break
        y += np.sin(2 * np.pi * freq * k * t + k) / k ** 1.1 * np.exp(-t * (4 + k * 5.5))
    y *= np.minimum(1, t / 0.002)
    return y * 0.35


def bell(freq, dur=4.0, idx=3.0):
    n = int(dur * SR)
    t = tt(n)
    I = idx * np.exp(-t / 0.6)
    y = np.sin(2 * np.pi * freq * t + I * np.sin(2 * np.pi * freq * 3.5 * t))
    y += 0.4 * np.sin(2 * np.pi * freq * 2.0 * t) * np.exp(-t / 0.8)
    return y * np.exp(-t / (dur * 0.35)) * np.minimum(1, t / 0.003) * 0.3


def supersaw(freq, n, voices=7, detune=0.14):
    out = np.zeros((n, 2))
    for v in range(voices):
        d = (v - (voices - 1) / 2) / ((voices - 1) / 2)
        f = freq * 2 ** (d * detune / 12)
        s = saw(f, n, phase0=rng.random())
        pan = d * 0.8
        out[:, 0] += s * np.cos((pan + 1) * np.pi / 4)
        out[:, 1] += s * np.sin((pan + 1) * np.pi / 4)
    return out / voices


def sub(freq, n):
    t = tt(n)
    return np.sin(2 * np.pi * freq * t) * 0.9 + lp(saw(freq, n), 260) * 0.25


def noise_riser(dur, f0=300, f1=9000, curve=2.0):
    n = int(dur * SR)
    t = tt(n) / dur
    z = rng.standard_normal((n, 2))
    out = np.zeros((n, 2))
    # sweep a band through blocks so the filter centre can move
    blocks = 40
    edges = np.linspace(0, n, blocks + 1).astype(int)
    filt_full = [bp(z, max(40, f0 * (f1 / f0) ** (b / blocks) * 0.7),
                    min(20000, f0 * (f1 / f0) ** (b / blocks) * 1.4)) for b in range(blocks)]
    for b in range(blocks):
        out[edges[b]:edges[b + 1]] = filt_full[b][edges[b]:edges[b + 1]]
    return out * (t ** curve)[:, None]


def whoosh(dur=0.9, f0=400, f1=3000, level=0.5):
    n = int(dur * SR)
    t = tt(n) / dur
    r = noise_riser(dur, f0, f1, 1.0)
    e = np.sin(np.pi * np.clip(t, 0, 1)) ** 2
    return r * e[:, None] / np.maximum(t, 1e-3)[:, None] ** 1.0 * level * 0.35


def boom(dur=4.0, f0=70, f1=28, level=1.0):
    n = int(dur * SR)
    t = tt(n)
    f = f1 + (f0 - f1) * np.exp(-t / 0.35)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 1.1)
    hit = lp(rng.standard_normal(n), 1800) * np.exp(-t / 0.18) * 0.6
    crack = hp(rng.standard_normal(n), 3000) * np.exp(-t / 0.02) * 0.3
    return np.tanh((body * 1.2 + hit + crack) * 1.4) * level


# ------------------------------------------------------------------ arrangement
CHORDS = {  # (bass midi, pad notes, arp notes)
    "Am": (33, [57, 60, 64, 67], [69, 72, 76, 79]),
    "F": (29, [53, 57, 60, 64], [65, 69, 72, 76]),
    "C": (36, [55, 60, 62, 64], [67, 72, 74, 76]),
    "G": (31, [55, 59, 62, 67], [67, 71, 74, 79]),
}
PROG = ["Am", "F", "C", "G"]
BREAKS = [(22.0, 24.0), (30.0, 32.0), (36.0, 38.5)]


def chord_at(t):
    return PROG[int((t - 6.0) // 4) % 4]


def in_break(t):
    return any(a <= t < b for a, b in BREAKS)


def build():
    drums, bass, pad, arp, fx, intro, hush = Bus(), Bus(), Bus(), Bus(), Bus(), Bus(), Bus()
    kicks = []

    # ---------- 0-3: the loud market
    t_end = 2.95
    n = int(t_end * SR)
    t = tt(n)
    swell = (t / t_end) ** 1.8
    z = rng.standard_normal((n, 2))
    chatter = bp(z, 350, 3200) * (0.55 + 0.45 * np.sin(2 * np.pi * (9 + 7 * t / t_end) * t)[:, None])
    intro.add(chatter * swell[:, None] * 0.55, 0.0)
    drone = supersaw(mtof(33), n, 5, 0.3) + supersaw(mtof(45), n, 5, 0.25) * 0.5
    intro.add(lp(drone, 900) * (0.2 + swell[:, None]) * 0.5, 0.0)
    tcur = 0.25
    while tcur < t_end - 0.02:
        rate = 8 + 70 * (tcur / t_end) ** 2
        f = float(rng.uniform(500, 3400))
        d = float(rng.uniform(0.015, 0.05))
        m = int(d * SR)
        ping = np.sin(2 * np.pi * f * tt(m)) * np.exp(-tt(m) / (d * 0.4))
        if rng.random() < 0.25:
            ping = np.sign(ping) * 0.6  # digital square blip
        intro.add(ping * 0.22 * (0.3 + tcur / t_end), tcur, pan=float(rng.uniform(-0.9, 0.9)))
        tcur += float(rng.exponential(1 / rate))
    for g0 in np.arange(1.2, t_end - 0.1, 0.23):
        m = int(float(rng.uniform(0.02, 0.07)) * SR)
        glitch = np.round(rng.standard_normal(m) * 3) / 3 * 0.25
        intro.add(hp(glitch, 1200) * ((g0 / t_end) ** 2), float(g0), pan=float(rng.uniform(-1, 1)))
    ix = intro.x
    ix[: int(t_end * SR)] *= np.minimum(1, tt(int(t_end * SR)) / 0.6)[:, None]
    ix[int(t_end * SR):] = 0
    tail = reverb(ix, reverb_ir(2.5, 0.5, 5000, 3), 1.0) * 0.35
    tail[: int(t_end * SR)] = 0
    intro.x = ix + tail

    # ---------- 3-6: listen closer
    hush.add(reverb(np.stack([bell(mtof(76), 5.0)] * 2, 1), reverb_ir(4, 1.4, 7000, 5), 0.55) * 0.9, 3.25)
    hush.add(reverb(np.stack([bell(mtof(69), 5.0, 2.0)] * 2, 1), reverb_ir(4, 1.4, 7000, 6), 0.6) * 0.5, 3.75)
    for hb in (4.0, 4.28, 5.0, 5.28):
        k = kick(0.3 if hb % 1 < 0.1 else 0.2)
        hush.add(lp(k, 180), hb)
    rz = noise_riser(1.45, 250, 9000, 2.2) * 0.35
    n2 = int(1.45 * SR)
    up = np.sin(2 * np.pi * np.cumsum(220 * 2 ** (2.2 * tt(n2) / 1.45)) / SR) * (tt(n2) / 1.45) ** 2 * 0.12
    hush.add(rz + up[:, None], 4.45)
    pn = int(3.0 * SR)
    padin = lp(supersaw(mtof(57), pn) + supersaw(mtof(64), pn), 700) * \
        np.minimum(1, tt(pn) / 2.5)[:, None] * 0.35
    hush.add(padin, 2.95 + 0.05)

    # ---------- 6-46: groove
    for bar in np.arange(6.0, 46.0, BEAT):
        b = float(bar)
        if in_break(b):
            continue
        k = kick()
        drums.add(k, b, 0.95)
        kicks.append(b)
        beat_in_bar = int(round((b - 6.0) / BEAT)) % 4
        if b >= 10.0:
            drums.add(hat(), b + 0.25, 0.55, pan=0.25)
            if beat_in_bar == 3 and rng.random() < 0.5:
                drums.add(hat(True), b + 0.25, 0.35, pan=-0.2)
        if b >= 16.0 and beat_in_bar in (1, 3):
            drums.add(clap(), b, 0.8)
        if b >= 10.0:
            for s16 in (0.125, 0.375):
                drums.add(hat(), b + s16, 0.18, pan=-0.35)
    # fills leading out of breaks
    for a, e in BREAKS:
        for j, s in enumerate(np.arange(e - 0.5, e, 0.125)):
            drums.add(snare(1 + j * 0.05), float(s), 0.35 + j * 0.12)
        drums.add(whoosh(1.2, 300, 5000, 0.9), a - 0.1)
        drums.add(noise_riser(e - a - 0.1, 400, 8000, 3.0) * 0.18, a + 0.1)
    # montage roll + riser 44-46
    for j, s in enumerate(np.arange(44.0, 46.0, 0.125)):
        drums.add(snare(1 + j * 0.03), float(s), 0.25 + 0.5 * j / 16)
    for s in np.arange(45.0, 46.0, 0.0625):
        drums.add(snare(1.6), float(s), 0.45)
    drums.add(noise_riser(2.0, 500, 12000, 2.5) * 0.45, 44.0)
    for c in np.arange(42.0, 46.0, 0.5):
        drums.add(whoosh(0.25, 2000, 8000, 0.25), float(c) - 0.05)

    # bass + pad + arp per chord (4 s each)
    for ci, t0 in enumerate(np.arange(6.0, 46.0, 4.0)):
        t0 = float(t0)
        name = chord_at(t0)
        bm, pads, arps = CHORDS[name]
        n4 = int(4.0 * SR)
        bass.add(sub(mtof(bm + 12), n4) * env_adsr(n4, 0.01, 1, 1, 0.05, 3.95), t0, 0.55)
        pd = sum(supersaw(mtof(m), n4) for m in pads) / len(pads)
        bright = 1400 if t0 < 16 else 3200
        pd = lp(pd, bright)
        pd *= env_adsr(n4, 0.25, 1, 1, 0.4, 3.7)[:, None]
        pad.add(pd, t0, 0.55)
        if t0 >= 10.0:
            for s in np.arange(t0, t0 + 4.0, 0.125):
                s = float(s)
                if in_break(s) or s >= 46.0:
                    continue
                step = int(round((s - t0) / 0.125))
                note = arps[[0, 1, 2, 3, 2, 1, 3, 2][step % 8]] + (12 if step % 16 >= 12 else 0)
                vel = 0.8 if step % 4 == 0 else 0.55
                arp.add(pluck(mtof(note)), s, vel, pan=0.35 * np.sin(step))

    # break treatment: duck bass/pad under the text cards
    tvec = tt(N)
    duck = np.ones(N)
    for a, e in BREAKS:
        m = (tvec >= a) & (tvec < e)
        duck[m] = 0.35
    duck = lp(duck, 8)
    bass.x *= duck[:, None]
    pad.x *= (0.5 + 0.5 * duck)[:, None]

    # sidechain pump from kicks
    sc = np.ones(N)
    for k in kicks:
        i = int(k * SR)
        m = min(int(0.4 * SR), N - i)
        sc[i:i + m] = np.minimum(sc[i:i + m], 1 - 0.75 * np.exp(-tt(m) / 0.09))
    bass.x *= sc[:, None]
    pad.x *= (0.35 + 0.65 * sc)[:, None]
    arp.x *= (0.6 + 0.4 * sc)[:, None]

    # hard stop at 46.0
    stop = int(46.0 * SR)
    for bus in (drums, bass, pad, arp):
        bus.x[stop:] = 0

    # ---------- 46-50: tagline hush
    n5 = int(4.2 * SR)
    g = lp(sum(supersaw(mtof(m), n5) for m in [55, 60, 62, 67]) / 4, 900)
    g *= (np.minimum(1, tt(n5) / 1.2) * np.exp(-np.maximum(0, tt(n5) - 3.0) / 0.6))[:, None]
    hush.add(g * 0.35, 46.0)
    hush.add(np.stack([bell(mtof(74), 4.0, 1.5)] * 2, 1) * 0.5, 46.3)
    hush.add(np.stack([bell(mtof(79), 4.0, 1.5)] * 2, 1) * 0.35, 47.05)
    for hb in (47.0, 48.0, 49.0):
        hush.add(lp(kick(0.35), 160), hb)
    rev_n = int(1.2 * SR)
    rs = hp(rng.standard_normal((rev_n, 2)), 2000) * ((tt(rev_n) / 1.2) ** 3)[:, None] * 0.35
    hush.add(rs, 48.8)
    hush.add(noise_riser(1.2, 200, 6000, 3.0) * 0.25, 48.8)

    # ---------- 50-59: impact + resolution
    fx.add(boom(5.0, 75, 26, 1.0), 50.0, 0.95)
    fx.add(kick(1.0), 50.0, 0.8)
    n6 = int(9.0 * SR)
    fin = sum(supersaw(mtof(m), n6) for m in [48, 55, 60, 62, 64, 67]) / 6
    fin = lp(fin, 2600)
    fin *= (np.minimum(1, tt(n6) / 0.05) * np.exp(-tt(n6) / 5.0))[:, None]
    fx.add(fin, 50.0, 0.5)
    fx.add(sub(mtof(36), n6)[:, None] * np.exp(-tt(n6) / 3.0)[:, None] * [1, 1], 50.0, 0.4)
    for j, m in enumerate([72, 76, 79, 84]):
        fx.add(np.stack([bell(mtof(m), 5.0, 1.2)] * 2, 1), 50.9 + j * 0.25, 0.28)

    # ---------- mix
    big = reverb_ir(3.2, 0.9, 6500, 9)
    small = reverb_ir(1.2, 0.35, 8000, 10)
    mix_ = np.zeros((N, 2))
    mix_ += reverb(drums.x, small, 0.12)
    mix_ += bass.x * 0.9
    mix_ += reverb(pad.x, big, 0.35) * 0.8
    mix_ += reverb(delay_pp(arp.x, 0.375, 0.35, 5), big, 0.25) * 0.55
    mix_ += intro.x * 0.9
    mix_ += reverb(hush.x, big, 0.3)
    mix_ += reverb(fx.x, big, 0.3)
    mix_ = hp(mix_, 25)
    # final fade
    fade = np.clip((58.6 - tvec) / 1.6, 0, 1)
    mix_ *= fade[:, None]
    mix_ /= np.abs(mix_).max() + 1e-9
    mix_ = np.tanh(mix_ * 1.6) / np.tanh(1.6)
    return (mix_ * 0.89).astype(np.float32)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "score.wav"
    wavfile.write(out, SR, build())
    print("wrote", out)

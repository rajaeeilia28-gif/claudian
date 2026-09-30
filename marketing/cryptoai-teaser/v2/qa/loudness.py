"""ITU-R BS.1770-4 / EBU R128 measurements (integrated loudness, momentary, LRA, true peak)."""
import numpy as np
from scipy.signal import lfilter, resample_poly

SR = 48000


def k_weight(x, sr=SR):
    if sr != 48000:
        raise ValueError("K-weighting coefficients are for 48 kHz")
    y = lfilter([1.53512485958697, -2.69169618940638, 1.19839281085285],
                [1, -1.69065929318241, 0.73248077421585], x, axis=0)
    return lfilter([1.0, -2.0, 1.0], [1, -1.99004745483398, 0.99007225036621], y, axis=0)


def _blocks(y, win, hop):
    out = []
    for s in range(0, len(y) - win + 1, hop):
        p = (y[s:s + win] ** 2).mean(0).sum()
        out.append(p)
    return np.array(out)


def integrated(x, sr=SR):
    y = k_weight(np.asarray(x, np.float64), sr)
    p = _blocks(y, int(0.4 * sr), int(0.1 * sr))
    lk = -0.691 + 10 * np.log10(p + 1e-15)
    p = p[lk > -70]
    rel = -0.691 + 10 * np.log10(p.mean()) - 10
    p = p[-0.691 + 10 * np.log10(p) > rel]
    return float(-0.691 + 10 * np.log10(p.mean()))


def momentary(x, sr=SR):
    y = k_weight(np.asarray(x, np.float64), sr)
    win, hop = int(0.4 * sr), int(0.1 * sr)
    p = _blocks(y, win, hop)
    t = (np.arange(len(p)) * hop + win) / sr
    return t, -0.691 + 10 * np.log10(p + 1e-15)


def lra(x, sr=SR):
    y = k_weight(np.asarray(x, np.float64), sr)
    p = _blocks(y, int(3.0 * sr), int(0.1 * sr))
    st = -0.691 + 10 * np.log10(p + 1e-15)
    st = st[st > -70]
    rel = -0.691 + 10 * np.log10((10 ** ((st + 0.691) / 10)).mean()) - 20
    st = st[st > rel]
    return float(np.percentile(st, 95) - np.percentile(st, 10))


def true_peak_db(x):
    x = np.asarray(x, np.float64)
    up = resample_poly(x, 4, 1, axis=0)
    return float(20 * np.log10(np.abs(up).max() + 1e-15))

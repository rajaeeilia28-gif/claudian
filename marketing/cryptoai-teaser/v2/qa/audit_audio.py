"""V2 audio audit against the approved spec (section 8).

Usage: audit_audio.py <mix.wav | video.mp4> [--json out.json]
"""
import json
import os
import subprocess
import sys
import wave

import numpy as np
from scipy.signal import butter, sosfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline")))
import loudness as L  # noqa: E402
import v2_spec as S  # noqa: E402

SR = 48000


def load(path):
    if path.endswith(".wav"):
        with wave.open(path) as w:
            n, sw, ch = w.getnframes(), w.getsampwidth(), w.getnchannels()
            raw = np.frombuffer(w.readframes(n), np.uint8)
        if sw == 3:
            b = raw.reshape(-1, 3)
            ints = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
            ints = np.where(ints >= 2 ** 23, ints - 2 ** 24, ints)
            return (ints / 2 ** 23).reshape(-1, ch).astype(np.float64)
        return (raw.view("<i2") / 32768.0).reshape(-1, ch)
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a", "-f", "f32le", "-ac", "2",
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32).reshape(-1, 2).astype(np.float64)


def band(x, lo=None, hi=None):
    if lo and hi:
        sos = butter(4, [lo, hi], "band", fs=SR, output="sos")
    elif lo:
        sos = butter(4, lo, "high", fs=SR, output="sos")
    else:
        sos = butter(4, hi, "low", fs=SR, output="sos")
    return sosfilt(sos, x, axis=0)


def db(x):
    return 10 * np.log10((np.asarray(x) ** 2).mean() + 1e-15)


def seg(x, a, b):
    return x[int(a * SR):int(b * SR)]


def onset(x, t, win=0.03):
    """First crossing of 50 % of the local 2-8 kHz envelope peak around t (offset in s)."""
    hb = band(x.mean(1), 2000, 8000)
    e = np.convolve(hb ** 2, np.ones(24) / 24, "same")
    i0, i1 = int((t - win) * SR), int((t + win) * SR)
    seg_ = e[i0:i1]
    base = np.median(e[int((t - 0.12) * SR):int((t - 0.04) * SR)])
    thr = base + 0.5 * (seg_.max() - base)
    return (i0 + int(np.argmax(seg_ > thr))) / SR - t


def audit(path):
    x = load(path)
    m = x.mean(1)
    r = {"file": os.path.basename(path), "duration_s": round(len(x) / SR, 3)}
    r["integrated_lufs"] = round(L.integrated(x), 2)
    r["true_peak_dbtp"] = round(L.true_peak_db(x), 2)
    r["lra_lu"] = round(L.lra(x), 2)
    r["sample_peak_dbfs"] = round(float(20 * np.log10(np.abs(x).max() + 1e-15)), 2)
    g = seg(x, S.LIFT_T, 23.0)
    low = band(g, hi=120)
    r["groove_energy_below_120hz_pct"] = round(100 * (low ** 2).sum() / ((g ** 2).sum() + 1e-15), 1)
    gm = g.mean(1)
    r["groove_crest_db"] = round(float(20 * np.log10(np.abs(gm).max() / np.sqrt((gm ** 2).mean()))), 1)
    ph = band(x, 300, 12000)
    for name, t in (("lift_4s", S.LIFT_T), ("impact_26s", S.IMPACT_T)):
        r[f"{name}_rise_above_300hz_db"] = round(db(seg(ph, t, t + 0.5)) - db(seg(ph, t - 0.55, t - 0.05)), 2)
        r[f"{name}_rise_fullband_db"] = round(db(seg(x, t, t + 0.5)) - db(seg(x, t - 0.55, t - 0.05)), 2)
    L_, R_ = x[:, 0], x[:, 1]
    r["stereo_correlation"] = round(float(np.corrcoef(L_, R_)[0, 1]), 3)
    r["mono_loss_db"] = round(db((L_ + R_) / 2) - 10 * np.log10(((L_ ** 2).mean() + (R_ ** 2).mean()) / 2), 2)
    r["accent_onsets_ms"] = {str(t): round(1000 * onset(x, t), 1) for t in S.ACCENTS + [S.IMPACT_T]}
    # 4.0 s is the lift (no hard cut) and is preceded by an intentional reverse swell: report only
    r["lift_onset_ms_info"] = round(1000 * onset(x, S.LIFT_T), 1)
    t, M = L.momentary(x)
    def sec(a, b):
        s = (t > a + 0.4) & (t <= b)
        return [round(float(M[s].min()), 1), round(float(np.median(M[s])), 1), round(float(M[s].max()), 1)]
    r["momentary_min_med_max"] = {"intro_0_4": sec(0, 4), "groove_4_23": sec(4, 23), "build_23_26": sec(23, 26),
                                  "impact_26_28": sec(26, 28), "outro_28_30": sec(28, 29.8)}
    r["silent_gaps_over_150ms_before_29_5s"] = int(sum(1 for k in range(len(M)) if 0.5 < t[k] < 29.5 and M[k] < -45))
    tail = seg(x, 29.8, 30.0)
    r["tail_last_200ms_dbfs"] = round(db(tail), 1)
    checks = {
        "integrated -14 +-0.5": abs(r["integrated_lufs"] + 14) <= 0.5,
        "true peak <= -1 dBTP": r["true_peak_dbtp"] <= -1.0,
        "LRA 5-8 LU": 5 <= r["lra_lu"] <= 8,
        "groove <120 Hz <= ~50 %": r["groove_energy_below_120hz_pct"] <= 55,
        "lift +3 dB above 300 Hz": r["lift_4s_rise_above_300hz_db"] >= 3,
        "impact +3 dB above 300 Hz": r["impact_26s_rise_above_300hz_db"] >= 3,
        "groove crest >= 10 dB": r["groove_crest_db"] >= 10,
        "stereo correlation >= 0.5": r["stereo_correlation"] >= 0.5,
        "mono loss <= 1.5 dB": r["mono_loss_db"] <= 1.5,
        "accents within +-10 ms": all(abs(v) <= 10 for v in r["accent_onsets_ms"].values()),
        "no silent gap": r["silent_gaps_over_150ms_before_29_5s"] == 0,
    }
    r["checks"] = {k: bool(v) for k, v in checks.items()}
    r["pass"] = bool(all(checks.values()))
    return r


if __name__ == "__main__":
    res = audit(sys.argv[1])
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))

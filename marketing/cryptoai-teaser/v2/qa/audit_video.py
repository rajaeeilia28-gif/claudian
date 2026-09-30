"""Frame-by-frame and safe-zone audit of a rendered V2 video (full film or preview segment).

Usage: audit_video.py <video.mp4> <h|v> <t_offset_s> [--overlays dir]

Reports: resolution/fps/frame count, cuts vs. the planned hard cuts, unexpected jumps, black frames,
large-area flashes (Harding/ITU-style approximation), bright detail outside the safe area and
bright detail touching the frame edge during holds.
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline")))
import v2_spec as S  # noqa: E402


def probe(path):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    line = [ln for ln in out.splitlines() if "Video:" in ln][0]
    return line.strip()


def frames(path, W, H):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(W * H * 3)
        if len(b) < W * H * 3:
            break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)


def lin(u8):
    c = u8.astype(np.float32) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4) @ np.array([0.2126, 0.7152, 0.0722],
                                                                                       np.float32)


def detail_mask(rgb):
    """Bright, high-contrast pixels (text, numbers, pills) - smooth glows are excluded."""
    g = rgb.max(2).astype(np.float32) / 255
    local = g - cv2.GaussianBlur(g, (0, 0), 3)
    return (g > 0.62) & (local > 0.18)


def audit(path, fmt, t_off, overlays=None):
    W, H = S.SIZE[fmt]
    x0, y0, x1, y1 = S.SAFE[fmt]
    res = {"file": os.path.basename(path), "stream": probe(path), "fmt": fmt}
    prev_small = prev_lum = None
    diffs, lums, outside, edge_touch, black = [], [], [], [], []
    n = 0
    for i, f in enumerate(frames(path, W, H)):
        n += 1
        t = t_off + i / S.FPS
        small = cv2.resize(f, (W // 4, H // 4), interpolation=cv2.INTER_AREA).astype(np.int16)
        lum = lin(cv2.resize(f, (W // 10, H // 10), interpolation=cv2.INTER_AREA))
        if prev_small is not None:
            diffs.append((t, float(np.abs(small - prev_small).mean())))
            d = lum - prev_lum
            dark = np.minimum(lum, prev_lum) < 0.8
            lums.append((t, float(((d > 0.1) & dark).mean()), float(((d < -0.1) & dark).mean())))
        prev_small, prev_lum = small, lum
        if lum.mean() < 0.0008:
            black.append(round(t, 3))
        if i % 3 == 0:
            dm = detail_mask(f)
            out = dm.copy()
            out[y0:y1, x0:x1] = False
            outside.append((t, int(out.sum())))
            border = np.zeros_like(dm)
            border[:3, :] = border[-3:, :] = True
            border[:, :3] = border[:, -3:] = True
            edge_touch.append((t, int((dm & border).sum())))
            if overlays and i % 30 == 0:
                ov = f[:, :, ::-1].copy()
                cv2.rectangle(ov, (x0, y0), (x1, y1), (0, 200, 255), 2)
                ov[out] = (0, 0, 255)
                os.makedirs(overlays, exist_ok=True)
                cv2.imwrite(os.path.join(overlays, f"{fmt}_{t:06.2f}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 80])
    res["frames"] = n
    res["duration_s"] = round(n / S.FPS, 3)
    # cuts: large jumps compared with the local median
    dv = np.array([d for _, d in diffs])
    cuts = []
    for k, (t, d) in enumerate(diffs):
        local = np.median(dv[max(0, k - 30):k + 30])
        if d > 8 and d > 5 * max(local, 0.3):
            cuts.append(round(t, 3))
    planned = [c for c in S.HARD_CUTS if t_off <= c < t_off + n / S.FPS]
    res["cuts_detected"] = cuts
    res["cuts_planned"] = planned
    res["cuts_missing"] = [c for c in planned if not any(abs(c - x) <= 1 / S.FPS + 1e-6 for x in cuts)]
    res["cuts_unexpected"] = [x for x in cuts if not any(abs(c - x) <= 1 / S.FPS + 1e-6 for c in planned)]
    # flashes: opposing large-area transitions (>= 25 % of the screen) within one second
    ev = [(t, 1 if up > 0.25 else -1) for t, up, dn in lums if max(up, dn) > 0.25]
    res["large_area_transitions"] = ev
    res["flash_pass"] = sum(1 for k in range(1, len(ev)) if ev[k][1] != ev[k - 1][1]) <= 3
    res["black_frames_s"] = [black[0], black[-1]] if black else []
    res["black_frame_count"] = len(black)
    res["max_detail_px_outside_safe"] = max(outside, key=lambda v: v[1]) if outside else None
    res["frames_with_detail_outside_safe"] = sum(1 for _, v in outside if v > 40)
    res["max_detail_px_on_edge"] = max(edge_touch, key=lambda v: v[1]) if edge_touch else None
    res["frames_with_detail_on_edge"] = sum(1 for _, v in edge_touch if v > 20)
    res["outside_safe_times"] = [round(t, 2) for t, v in outside if v > 40][:60]
    return res


if __name__ == "__main__":
    ov = sys.argv[sys.argv.index("--overlays") + 1] if "--overlays" in sys.argv else None
    print(json.dumps(audit(sys.argv[1], sys.argv[2], float(sys.argv[3]), ov), indent=2, ensure_ascii=False))

"""Data and zoom audit from the per-frame usage manifest written by v2_render.py (<video>.uses.json).

Checks: only approved 28.09.2026 frames, forbidden frames/regions never visible, the 97,4 % tile never
enlarged, per-shot zoom limits. Elements below 50 % opacity and the non-target cells during the dive
(2.8-4.0 s, motion-blurred fly-through) are not zoom-rated.
"""
import json
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline")))
import v2_spec as S  # noqa: E402


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def audit(path):
    m = json.load(open(path))
    fmt = m["fmt"]
    limits = dict(S.ZOOM_LIMITS[fmt], s7_system=S.ZOOM_LIMITS[fmt]["s1_system"])
    issues, frames_used, max_zoom = [], set(), {}
    for fi, uses in m["frames"].items():
        t = int(fi) / S.FPS
        for u in uses:
            n, rect, z, op, shot = u["frame"], u["rect"], u["zoom"], u["opacity"], u["shot"]
            frames_used.add(n)
            if n not in S.APPROVED_FRAMES:
                issues.append(f"{t:.3f}s: Bild {n} nicht freigegeben")
            if n in S.FORBIDDEN_FRAMES:
                issues.append(f"{t:.3f}s: gesperrtes Bild {n}")
            for name, fset, reg in S.FORBIDDEN_REGIONS:
                if n in fset and overlaps(rect, reg):
                    issues.append(f"{t:.3f}s: {name} sichtbar (Bild {n})")
            for name, fset, reg in S.NO_HERO_REGIONS:
                if n in fset and overlaps(rect, reg) and z > 1.0 + 1e-6 and op >= 0.5:
                    issues.append(f"{t:.3f}s: {name} vergrößert ({z:.2f}x)")
            dive_other = shot == "s1_system" and t >= 2.8 and n != 70
            if op >= 0.5 and not dive_other:
                max_zoom[shot] = max(max_zoom.get(shot, 0), z)
                if z > limits.get(shot, 1.0) + 1e-3:
                    issues.append(f"{t:.3f}s: Zoom {z:.3f}x > Grenze {limits.get(shot)} ({shot}, Bild {n})")
    return {"fmt": fmt, "frames_checked": len(m["frames"]), "source_frames_used": sorted(frames_used),
            "max_zoom_per_shot": {k: round(v, 3) for k, v in sorted(max_zoom.items())},
            "zoom_limits": limits, "issues": issues[:50], "issue_count": len(issues), "pass": not issues}


if __name__ == "__main__":
    print(json.dumps(audit(sys.argv[1]), indent=2, ensure_ascii=False))

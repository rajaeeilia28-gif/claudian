"""V2 render entry points.

  v2_render.py stills  <h|v> <t> [t ...]          preview stills  -> ../out/stills/
  v2_render.py preview <h|v> <t0> <t1> <out.mp4>  preview segment + usage manifest (<out>.uses.json)
  v2_render.py final   <h|v> <out.mp4>            refuses to run until R1 is present
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_spec as S  # noqa: E402
from v2_spot import SpotV2  # noqa: E402

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
_SP = None


def _init(fmt, preview):
    global _SP
    cv2.setNumThreads(1)
    _SP = SpotV2(fmt, preview=preview)


def _work(i):
    img, uses = _SP.frame(i)
    return i, img.tobytes(), uses


def video(fmt, out, t0, t1, preview=True, workers=4):
    W, H = S.SIZE[fmt]
    frames = range(int(round(t0 * S.FPS)), int(round(t1 * S.FPS)))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(S.FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "15",
           "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-movflags", "+faststart", out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    manifest = {}
    start = time.time()
    with mp.Pool(workers, initializer=_init, initargs=(fmt, preview)) as pool:
        for k, (i, buf, uses) in enumerate(pool.imap(_work, frames, chunksize=4)):
            ff.stdin.write(buf)
            manifest[i] = uses
            if k % 240 == 0:
                print(f"{fmt} {k}/{len(frames)} {time.time() - start:.0f}s", flush=True)
    ff.stdin.close()
    ff.wait()
    with open(out + ".uses.json", "w") as fh:
        json.dump({"fmt": fmt, "fps": S.FPS, "preview": preview, "frames": manifest}, fh)
    print("done", out, f"{time.time() - start:.0f}s")


def stills(fmt, times):
    sp = SpotV2(fmt, preview=True)
    d = os.path.join(OUT, "stills")
    os.makedirs(d, exist_ok=True)
    for t in times:
        img, _ = sp.frame(int(round(t * S.FPS)))
        cv2.imwrite(os.path.join(d, f"{fmt}_{t:06.3f}.png"), img[:, :, ::-1])


if __name__ == "__main__":
    mode, fmt = sys.argv[1], sys.argv[2]
    if mode == "stills":
        stills(fmt, [float(x) for x in sys.argv[3:]])
    elif mode == "preview":
        video(fmt, sys.argv[5], float(sys.argv[3]), float(sys.argv[4]), preview=True)
    elif mode == "final":
        SpotV2(fmt, preview=False)  # raises while R1 is missing
        video(fmt, sys.argv[3], 0.0, S.DURATION, preview=False)
    else:
        raise SystemExit(__doc__)

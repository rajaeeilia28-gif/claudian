"""Render the teaser: `render.py video h out.mp4` or `render.py stills h 6.2 16.5 ...`."""
import multiprocessing as mp
import os
import subprocess
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spot import DURATION, FPS, Spot  # noqa: E402

cv2.setNumThreads(1)
_SP = None


def _init(fmt):
    global _SP
    cv2.setNumThreads(1)
    _SP = Spot(fmt)


def _work(i):
    return _SP.frame(i).tobytes()


def video(fmt, out, t0=0.0, t1=DURATION, workers=4):
    sp = Spot(fmt)
    W, H = sp.W, sp.H
    frames = range(int(round(t0 * FPS)), int(round(t1 * FPS)))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-tune", "film",
           "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
           "-color_trc", "bt709", "-movflags", "+faststart", out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    start = time.time()
    with mp.Pool(workers, initializer=_init, initargs=(fmt,)) as pool:
        for k, buf in enumerate(pool.imap(_work, frames, chunksize=4)):
            ff.stdin.write(buf)
            if k % 300 == 0:
                print(f"{k}/{len(frames)} frames  {time.time() - start:.0f}s", flush=True)
    ff.stdin.close()
    ff.wait()
    print("done", out, f"{time.time() - start:.0f}s")


def stills(fmt, times, outdir):
    sp = Spot(fmt)
    os.makedirs(outdir, exist_ok=True)
    for t in times:
        img = sp.frame(int(round(t * FPS)))
        cv2.imwrite(os.path.join(outdir, f"{fmt}_{t:06.2f}.png"), img[:, :, ::-1])


if __name__ == "__main__":
    mode, fmt = sys.argv[1], sys.argv[2]
    if mode == "video":
        out = sys.argv[3]
        t0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
        t1 = float(sys.argv[5]) if len(sys.argv) > 5 else DURATION
        video(fmt, out, t0, t1)
    else:
        stills(fmt, [float(x) for x in sys.argv[3:]], os.path.join(os.path.dirname(__file__), "..", "stills"))

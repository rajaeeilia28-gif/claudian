"""One-time asset prep: source frames, keyed logo, Inter font.

Usage: python3 prepare.py path/to/original.mp4
"""
import io
import os
import subprocess
import sys
import urllib.request
import zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INTER_URL = "https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip"


def frames(video):
    out = os.path.join(ROOT, "src")
    os.makedirs(out, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                    os.path.join(out, "%04d.png")], check=True)


def logo():
    # The recording ends on the logo over black; average the static tail to beat compression noise.
    ims = [cv2.imread(os.path.join(ROOT, "src", f"{n:04d}.png")).astype(np.float32)
           for n in range(876, 901)]
    im = np.mean(ims, 0)
    bg = np.median(im[200:300, 600:700].reshape(-1, 3), 0)
    c = np.clip(im[325:550, 850:1080] - bg, 0, None)
    a = np.clip((c.max(2) - 6) / 32, 0, 1)
    a = a * a * (3 - 2 * a)
    c2 = cv2.resize(c, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    a2 = np.clip(cv2.resize(a, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4), 0, 1)
    c2 = np.clip(cv2.bilateralFilter(c2.astype(np.float32), 9, 18, 5), 0, 255)
    rgba = np.dstack([c2[:, :, ::-1] / 255.0, a2]).astype(np.float32)
    np.save(os.path.join(HERE, "logo_rgba.npy"), rgba)


def fonts():
    dst = os.path.join(ROOT, "fonts")
    if os.path.isdir(os.path.join(dst, "extras", "ttf")):
        return
    data = urllib.request.urlopen(INTER_URL).read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(dst)


if __name__ == "__main__":
    frames(sys.argv[1])
    logo()
    fonts()
    print("assets ready")

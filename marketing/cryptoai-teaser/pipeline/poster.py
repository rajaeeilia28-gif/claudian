"""Key visuals: YouTube thumbnail (16:9) and Reel/TikTok cover (9:16)."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import (BRAND, GRAY, SRC, WHITE, affine, aurora, blit, blit_persp,  # noqa: E402
                 card_homography, finish, glow_layer, src_layer)
from spot import Spot, TextBlock  # noqa: E402


def poster(fmt, out):
    sp = Spot(fmt)
    W, H = sp.W, sp.H
    c = np.zeros((H, W, 3), np.float32)
    c += aurora(12.0, W, H, 1.3)
    if fmt == "h":
        plate = src_layer(SRC.get(600), 95, 155, 1825, 1011, radius=18)
        pc, s, rx, ry = (1370, 560), 0.6, 10, -24
        tx, logo_y, logo_h = 470, 370, 200
    else:
        plate = src_layer(SRC.get(600), 95, 155, 935, 1011, radius=18)
        pc, s, rx, ry = (540, 1430), 0.98, 24, -10
        tx, logo_y, logo_h = 540, 400, 250
    word = TextBlock([[("Crypto AI", {"fill": WHITE})]], 128, weight="SemiBold", tracking=-0.02)
    soon = TextBlock([[("Bald verfügbar.", {"fill": GRAY})]], 60, weight="Medium", tracking=-0.01)
    gl, gp = glow_layer(plate.w, plate.h, BRAND, 90, radius=18)
    blit_persp(c, gl, card_homography(plate.w + 2 * gp, plate.h + 2 * gp, pc[0], pc[1] + 40, s, rx, ry),
               opacity=0.55)
    blit_persp(c, plate, card_homography(plate.w, plate.h, pc[0], pc[1], s, rx, ry))
    sp._logo_glow(c, tx, logo_y, logo_h * 1.9, 0.5)
    blit(c, sp.logo, affine(logo_h / sp.logo.h, tx, logo_y, sp.logo.w / 2, sp.logo.h / 2))
    wy = logo_y + logo_h * 0.5 + 140
    word.draw(c, 10.0, tx, wy, 0.0, None, whole_line=True)
    soon.draw(c, 10.0, tx, wy + 110, 0.0, None, whole_line=True)
    img = finish(c, 0)
    cv2.imwrite(out, img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])


if __name__ == "__main__":
    poster("h", sys.argv[1])
    poster("v", sys.argv[2])

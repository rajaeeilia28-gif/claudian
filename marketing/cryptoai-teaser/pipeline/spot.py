"""Timeline for the Crypto AI teaser (16:9 master and 9:16 social cut)."""
import math
import os

import cv2
import numpy as np

from lib import (BRAND, DIM, GRAY, MINT, SRC, WARM, WHITE, Layer, affine, aurora, blit,
                 blit_persp, camera, card_homography, ease_in, ease_io, ease_io5, ease_out,
                 finish, glow_layer, grad_color, lin, mix, src_layer, text_line)

FPS = 60
DURATION = 59.0
HERE = os.path.dirname(os.path.abspath(__file__))

# Beat grid: 120 BPM, one bar = 2 s. Cuts land on beats.
T_LOUD = (0.8, 2.78)
T_LISTEN = (3.25, 5.55)
T_HERO = (6.0, 10.0)
T_STATS = [(10.0, 12.0), (12.0, 14.0), (14.0, 16.0)]
T_DATA = (16.0, 22.0)
T_MONEY_TXT = (22.0, 24.0)
T_FLOW = (24.0, 27.0)
T_OI = (27.0, 30.0)
T_BIG_TXT = (30.0, 32.0)
T_MACRO = (32.0, 36.0)
T_RISK_TXT = (36.0, 38.5)
T_RISK = (38.5, 42.0)
T_MONTAGE = (42.0, 46.0)
T_TAG = (46.0, 50.0)
T_END = (50.0, DURATION)

def _mask(shape, x0, y0, x1, y1):
    m = np.zeros(shape[:2], np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


NOISE_FRAMES = [70, 220, 290, 345, 600, 710, 800, 140]


class TextBlock:
    """Centered multi-line headline with Apple-style staggered word reveals."""

    def __init__(self, lines, size, weight="Bold", tracking=-0.022, gap=1.14, features=None):
        self.size = size
        self.gap = gap
        self.lines = []
        for segs in lines:
            layer, words, (asc, desc, pad) = text_line(segs, size, weight, tracking)
            self.lines.append(dict(layer=layer, words=words, asc=asc, pad=pad,
                                   width=layer.w - 2 * pad))

    @property
    def width(self):
        return max(L["width"] for L in self.lines)

    def draw(self, canvas, t, cx, cy, t_in, t_out=None, stagger=0.085, line_in=None,
             dur=0.8, rise=34, blur0=14.0, exit_dur=0.32, scale=1.0, opacity=1.0,
             sweep=None, whole_line=False):
        n = len(self.lines)
        span = self.size * self.gap * (n - 1)
        for li, L in enumerate(self.lines):
            base_y = cy - span / 2 + li * self.size * self.gap
            left = cx - L["width"] / 2
            start = line_in[li] if line_in else t_in + li * 0.28
            items = [(L["layer"], -L["pad"])] if whole_line else L["words"]
            for wi, (wl, off) in enumerate(items):
                st = start + wi * stagger
                p = ease_out((t - st) / dur, 3)
                if p <= 0:
                    continue
                q = ease_in((t - t_out) / exit_dur, 2) if t_out is not None else 0.0
                op = p * (1 - q) * opacity
                if op <= 0.003:
                    continue
                dy = (1 - p) * rise - q * rise * 0.45
                bl = (1 - p) * blur0 + q * blur0 * 0.7
                sc = scale * (1 + (1 - p) * 0.035)
                ox = left + off
                oy = base_y - (L["pad"] + L["asc"] - 0.36 * self.size)
                X = cx + (ox - cx) * sc
                Y = base_y + (oy - base_y) * sc + dy
                lay = wl
                if sweep is not None:
                    lay = _sweep(wl, sweep, ox - left, L["width"])
                blit(canvas, lay, [[sc, 0, X], [0, sc, Y]], opacity=op, blur=bl)


def _sweep(layer, u, x_off, total_w, width=0.10, strength=0.85):
    """Add a travelling diagonal highlight band across a text layer."""
    h, w = layer.a.shape
    xs = (np.arange(w, dtype=np.float32) + x_off) / max(total_w, 1)
    ys = np.arange(h, dtype=np.float32)[:, None] / max(h, 1)
    d = xs[None, :] - u + (ys - 0.5) * 0.18
    band = np.exp(-(d / width) ** 2) * strength
    rgb = layer.rgb + band[..., None] * layer.a[..., None] * (1 - layer.rgb * 0.6)
    return Layer(np.minimum(rgb, layer.a[..., None]), layer.a)


class Spot:
    def __init__(self, fmt="h"):
        self.fmt = fmt
        self.W, self.H = (1920, 1080) if fmt == "h" else (1080, 1920)
        self.v = fmt == "v"
        self._num_cache = {}
        self._build_text()
        self._fcache = {}
        yy, xx = np.mgrid[0:1080, 0:1920].astype(np.float32)
        edge = np.minimum.reduce([xx, yy, 1919 - xx, 1079 - yy]) / 110.0
        self.edge = (np.clip(edge, 0, 1) ** 1.6)[..., None]
        rgba = np.load(os.path.join(HERE, "logo_rgba.npy"))
        self.logo = Layer(rgba[..., :3], rgba[..., 3])

    # ------------------------------------------------------------ assets
    def _build_text(self):
        v = self.v
        head = 96 if v else 112
        W = {"fill": WHITE}
        B = {"fill": BRAND}
        self.t_loud = TextBlock([[("Der Markt ist laut.", W)]], head)
        self.t_listen = TextBlock([[("Hör genauer ", W), ("hin.", B)]], head)
        self.t_money = TextBlock([[("Folge dem ", W), ("Geld.", {"fill": MINT})]], head)
        self.t_big = TextBlock([[("Sieh das ", W), ("große Bild.", B)]], head)
        if v:
            self.t_risk = TextBlock([[("Nicht nur Markt.", W)], [("Auch Risiko.", {"fill": WARM})]], head)
            self.t_tag = TextBlock([[("Klarheit ist", W)], [("das neue ", W), ("Alpha.", B)]], 104)
        else:
            self.t_risk = TextBlock([[("Nicht nur Markt.", W)], [("Auch Risiko.", {"fill": WARM})]], head)
            self.t_tag = TextBlock([[("Klarheit ist das neue ", W), ("Alpha.", B)]], 108)
        lab = 60 if v else 68
        self.stat_labels = [
            TextBlock([[("Kennzahlen.", W)]], lab, weight="SemiBold", tracking=-0.015),
            TextBlock([[("Modelle.", W)]], lab, weight="SemiBold", tracking=-0.015),
            TextBlock([[("klares Urteil.", W)]], lab, weight="SemiBold", tracking=-0.015),
        ]
        self.word = TextBlock([[("Crypto AI", W)]], 96 if not v else 92, weight="SemiBold",
                              tracking=-0.02)
        self.soon = TextBlock([[("Bald verfügbar.", {"fill": GRAY})]], 44, weight="Medium",
                              tracking=-0.01)
        fs = 18 if not v else 19
        if v:
            foot = [[("Alle Werte: Aufnahmen der Crypto-AI-Plattform vom 28.09.2026.", {"fill": DIM})],
                    [("Keine Anlageberatung. Krypto-Assets sind mit hohen Risiken verbunden.", {"fill": DIM})]]
        else:
            foot = [[("Alle Werte: Aufnahmen der Crypto-AI-Plattform vom 28.09.2026 · keine Live-Daten.  "
                      "Keine Anlageberatung. Krypto-Assets sind mit hohen Risiken verbunden.", {"fill": DIM})]]
        self.foot = TextBlock(foot, fs, weight="TextRegular", tracking=0.0, gap=1.5)

    def number(self, s, size):
        key = (s, size)
        if key not in self._num_cache:
            lay, _, (asc, desc, pad) = text_line([(s, {"fill": BRAND})], size, "Bold", -0.035,
                                                features=["tnum"])
            self._num_cache[key] = (lay, asc, pad)
        return self._num_cache[key]

    # ------------------------------------------------------------ frame
    def frame(self, i):
        t = i / FPS
        c = np.zeros((self.H, self.W, 3), np.float32)
        vig = True
        if t < T_HERO[0]:
            self.intro(c, t)
        elif t < T_STATS[0][0]:
            self.hero(c, t - T_HERO[0])
        elif t < T_DATA[0]:
            self.stats(c, t)
        elif t < T_DATA[1]:
            self.data(c, t - T_DATA[0])
        elif t < T_FLOW[0]:
            self.card(c, t, self.t_money, T_MONEY_TXT)
        elif t < T_FLOW[1]:
            self.flow(c, t - T_FLOW[0])
        elif t < T_OI[1]:
            self.oi(c, t - T_OI[0])
        elif t < T_MACRO[0]:
            self.card(c, t, self.t_big, T_BIG_TXT)
        elif t < T_MACRO[1]:
            self.macro(c, t - T_MACRO[0])
        elif t < T_RISK[0]:
            self.risk_text(c, t)
        elif t < T_RISK[1]:
            self.risk(c, t - T_RISK[0])
        elif t < T_MONTAGE[1]:
            self.montage(c, t - T_MONTAGE[0])
        elif t < T_END[0]:
            self.tagline(c, t)
        else:
            self.endcard(c, t - T_END[0])
        return finish(c, i, vig)

    # ------------------------------------------------------------ helpers
    def bg(self, c, t, amount=1.0):
        c += aurora(t, self.W, self.H, amount)

    def feather(self, arr):
        """In portrait the 16:9 source is narrower than the frame: fade its edges to black."""
        return (arr * self.edge).astype(np.uint8) if self.v else arr

    def srcf(self, n):
        if not self.v:
            return SRC.get(n)
        if n not in self._fcache:
            if len(self._fcache) > 24:
                self._fcache.clear()
            self._fcache[n] = self.feather(self.clean(SRC.get(n), n))
        return self._fcache[n]

    @staticmethod
    def clean(src, n):
        """Portrait framings cross UI headings the 16:9 cut avoids; retouch them out."""
        if 752 <= n <= 830:  # risk lab: repeated "Nicht nur Markt. Auch Risiko." heading
            src = cv2.inpaint(src, _mask(src.shape, 120, 178, 860, 280), 9, cv2.INPAINT_TELEA)
        if n != 70:  # web-app header chips read as clutter in tight portrait framings
            m = _mask(src.shape, 30, 25, 290, 135) | _mask(src.shape, 1440, 75, 1900, 130)
            src = cv2.inpaint(src, m, 7, cv2.INPAINT_TELEA)
        return src

    def flash(self, c, k, gain=0.9, lift=0.06):
        if k > 0:
            c *= 1 + gain * k
            c += lift * k

    def dof(self, img, fx, fy, rx, ry, sigma):
        if sigma < 0.3:
            return img
        bl = cv2.GaussianBlur(img, (0, 0), sigma)
        H, W = img.shape[:2]
        yy, xx = np.ogrid[0:H, 0:W]
        d = np.sqrt(((xx - fx) / rx) ** 2 + ((yy - fy) / ry) ** 2)
        m = np.clip((d - 0.75) / 0.6, 0, 1).astype(np.float32)[..., None]
        return img * (1 - m) + bl * m

    def card(self, c, t, block, span, **kw):
        self.bg(c, t, 0.55)
        t0, t1 = span
        block.draw(c, t, self.W / 2, self.H / 2, t0 + 0.08, t1 - 0.36, **kw)

    # ------------------------------------------------------------ shots
    def intro(self, c, t):
        W, H = self.W, self.H
        if t < 3.0:
            # "the market is loud": flickering, out-of-focus fragments of data behind the line
            dens = ease_in(lin(t, 0.2, 2.9), 2)
            if dens > 0 and t < 2.96:
                self.fragments(c, t, dens)
            self.t_loud.draw(c, t, W / 2, H / 2, T_LOUD[0], T_LOUD[1] - 0.2, exit_dur=0.22)
        else:
            self.bg(c, t, 0.35 * lin(t, 3.0, 4.5))
            self.t_listen.draw(c, t, W / 2, H / 2, T_LISTEN[0], T_LISTEN[1], dur=1.0, stagger=0.16,
                               sweep=mix(-0.3, 1.3, lin(t, 4.1, 5.1)) if 4.0 < t < 5.2 else None)

    def fragments(self, c, t, dens):
        step = int(t * 15)
        rng = np.random.default_rng(step * 7919 + 13)
        n = int(2 + dens * 9)
        for _ in range(n):
            f = int(rng.choice(NOISE_FRAMES))
            src = SRC.get(f)
            w = int(rng.integers(260, 700))
            h = int(rng.integers(110, 340))
            x0 = int(rng.integers(40, 1920 - w - 40))
            y0 = int(rng.integers(40, 1080 - h - 40))
            crop = src[y0:y0 + h, x0:x0 + w]
            if crop.mean() < 10:
                continue
            lay = src_layer(src, x0, y0, x0 + w, y0 + h, feather=min(w, h) * 0.3)
            sc = float(rng.uniform(0.9, 2.3))
            px = float(rng.uniform(0.05, 0.95)) * self.W
            py = float(rng.uniform(0.08, 0.92)) * self.H
            M = affine(sc, px, py, w / 2, h / 2, float(rng.uniform(-3, 3)))
            blit(c, lay, M, opacity=float(rng.uniform(0.18, 0.5)) * dens,
                 blur=float(rng.uniform(3, 10)))

    def hero(self, c, t):
        # Push-in on the live verdict; flash-resolve on the drop.
        src = SRC.get(70).copy()
        # light sweep over "Bullisch"
        u = lin(t, 1.3, 2.6)
        if 0 < u < 1:
            x0, y0, x1, y1 = 690, 560, 1210, 680
            reg = src[y0:y1, x0:x1].astype(np.float32)
            lum = reg.mean(2, keepdims=True) / 255.0
            m = np.clip((lum - 0.18) / 0.35, 0, 1)
            xs = np.arange(x1 - x0, dtype=np.float32)[None, :, None]
            ys = np.arange(y1 - y0, dtype=np.float32)[:, None, None]
            pos = mix(-120, (x1 - x0) + 120, u)
            band = np.exp(-(((xs - pos) + (ys - 60) * 0.35) / 55) ** 2)
            reg = reg + band * m * 150
            src[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)
        k = ease_io(t / 4.0)
        if self.v:
            zoom = mix(1.30, 1.46, k)
            cx, cy = 944, mix(545, 525, k)
        else:
            zoom = mix(1.04, 1.30, k)
            cx, cy = 960, mix(545, 520, k)
        img = camera(self.feather(src), cx, cy, zoom, self.W, self.H, sharpen=0.35)
        f = 1 - ease_out(t / 0.5, 3)
        if f > 0.01:
            img = cv2.GaussianBlur(img, (0, 0), 0.3 + 16 * f)
        c += img
        self.flash(c, f, gain=1.1, lift=0.10)

    def stats(self, c, t):
        W, H = self.W, self.H
        self.bg(c, t, 0.9)
        items = [(241, True), (99, True), (1, False)]
        for idx, ((t0, t1), (val, count)) in enumerate(zip(T_STATS, items)):
            if not (t0 <= t < t1):
                continue
            lt = t - t0
            size = 300 if not self.v else 300
            if count:
                shown = int(round(val * ease_out((lt - 0.04) / 0.95, 4)))
            else:
                shown = val
            lay, asc, pad = self.number(str(max(shown, 0)), size)
            p = ease_out(lt / 0.55, 3)
            q = ease_in((lt - (t1 - t0 - 0.24)) / 0.24, 2)
            sc = 1 + (1 - p) * 0.08 - q * 0.02
            cy = H * (0.44 if not self.v else 0.45)
            dy = (1 - p) * 40 - q * 60
            if not count and 0.35 < lt < 1.3:
                lay = _sweep(lay, mix(-0.3, 1.3, lin(lt, 0.35, 1.3)), 0, lay.w, width=0.25)
            ox = W / 2 - lay.w / 2
            oy = cy - (pad + asc - 0.36 * size)
            X = W / 2 + (ox - W / 2) * sc
            Y = cy + (oy - cy) * sc + dy
            blit(c, lay, [[sc, 0, X], [0, sc, Y]], opacity=p * (1 - q),
                 blur=(0.6, (1 - p) * 16 + q * 22))
            self.stat_labels[idx].draw(c, t, W / 2, cy + size * 0.62, t0 + 0.2, t1 - 0.24,
                                       exit_dur=0.24, dur=0.6, blur0=10)

    def data(self, c, t):
        W, H = self.W, self.H
        self.bg(c, T_DATA[0] + t, 0.9)
        n = 430 + t * 30
        n = int(n) if n < 541 else 600
        src = SRC.get(n)
        x0, y0, x1, y1 = 95, 155, 1825, 1011
        if self.v:
            x0, x1 = 95, 935
        plate = src_layer(src, x0, y0, x1, y1, radius=18)
        w, h = plate.w, plate.h
        pa = ease_out(t / 3.2, 3)
        rx = 34 * (1 - pa)
        ry = (-12 if not self.v else -6) * (1 - pa)
        s0 = mix(0.74, 0.93, pa) if not self.v else mix(0.95, 1.12, pa)
        cx0 = W / 2
        cy0 = H / 2 + mix(180, 12, pa)
        pb = ease_io5((t - 3.0) / 3.0)
        # point of interest: the 50 / 29 / 0 vote count
        poi = np.array([520 - x0, 390 - y0], np.float64)
        s = mix(s0, 1.62 if not self.v else 1.36, pb)
        p0 = np.array([cx0, cy0]) + (poi - [w / 2, h / 2]) * s0
        pt = p0 + (np.array([W / 2, H / 2]) - p0) * pb
        cc = pt - (poi - [w / 2, h / 2]) * s
        op = ease_out(t / 0.45, 2)
        gl, gp = glow_layer(w, h, BRAND, 70, radius=18)
        Hg = card_homography(w + 2 * gp, h + 2 * gp, cc[0], cc[1] + 30 * s, s, rx, ry)
        blit_persp(c, gl, Hg, opacity=0.42 * op * (1 - pb))
        Hm = card_homography(w, h, cc[0], cc[1], s, rx, ry)
        blit_persp(c, plate, Hm, opacity=op)

    def flow(self, c, t):
        # Rack focus: net flow figure -> options open-interest heat map.
        k = ease_io5(t / 3.0)
        kf = ease_io5((t - 0.7) / 1.5)
        if self.v:
            cx, cy, zoom = mix(310, 1360, k), mix(250, 514, k), mix(2.35, 1.12, k)
        else:
            cx, cy, zoom = mix(430, 1300, k), mix(300, 515, k), mix(1.78, 1.38, k)
        img = camera(self.srcf(345), cx, cy, zoom, self.W, self.H, sharpen=0.4)
        fsx, fsy = mix(300, 1360, kf), mix(250, 514, kf)
        fx = self.W / 2 + (fsx - cx) * zoom
        fy = self.H / 2 + (fsy - cy) * zoom
        rx, ry = mix(300, 520, kf) * zoom, mix(140, 360, kf) * zoom
        img = self.dof(img, fx, fy, rx, ry, 6.5)
        f = 1 - ease_out(t / 0.35, 3)
        c += img
        self.flash(c, f * 0.6)

    def oi(self, c, t):
        W, H = self.W, self.H
        self.bg(c, T_OI[0] + t, 0.8)
        src = SRC.get(290)
        k = ease_io(t / 3.0)
        ry = mix(18, -8, k)
        rx = mix(10, 4, k)
        op = ease_out(t / 0.3, 2)
        if self.v:
            # portrait: chart card on top, open-interest total + exchange split below
            chart = src_layer(src, 95, 156, 1117, 658, radius=16)
            stats = src_layer(src, 1150, 150, 1895, 650, feather=34)
            s = mix(0.98, 1.04, t / 3)
            layers = [(chart, H / 2 - 330, s, 0.0), (stats, H / 2 + 340, s * 1.3, 0.35)]
        else:
            plate = src_layer(src, 60, 138, 1900, 690, feather=46)
            s = mix(1.02, 1.12, t / 3)
            layers = [(plate, H / 2, s, 0.0)]
        for plate, cy, sc, delay in layers:
            o = op * ease_out((t - delay) / 0.4, 2)
            gl, gp = glow_layer(plate.w, plate.h, (0.2, 0.35, 1.0), 80, radius=20)
            Hg = card_homography(plate.w + 2 * gp, plate.h + 2 * gp, W / 2, cy + 40, sc, rx, ry)
            blit_persp(c, gl, Hg, opacity=0.25 * o)
            Hm = card_homography(plate.w, plate.h, W / 2, cy, sc, rx, ry * (1.25 if delay else 1),
                                 focal=2000)
            blit_persp(c, plate, Hm, opacity=o)

    def macro(self, c, t):
        # Pull back from the regime headline to the whole macro desk as it builds.
        n = 668 + t * 30
        n = int(n) if n < 700 else 710
        k = ease_io5(lin(t, 0.15, 3.3))
        if self.v:
            cx, cy, zoom = mix(323, 523, k), mix(420, 583, k), mix(2.7, 1.26, k)
        else:
            cx, cy, zoom = mix(300, 960, k), mix(320, 585, k), mix(2.3, 1.08, k)
            zoom *= 1 + 0.03 * lin(t, 3.3, 4.0)
        img = camera(self.srcf(n), cx, cy, zoom, self.W, self.H, sharpen=0.4)
        c += img
        self.bg(c, T_MACRO[0] + t, 0.25)
        self.flash(c, (1 - ease_out(t / 0.35, 3)) * 0.6)

    def risk_text(self, c, t):
        self.bg(c, t, 0.5)
        t0, t1 = T_RISK_TXT
        self.t_risk.draw(c, t, self.W / 2, self.H / 2, t0 + 0.08, t1 - 0.34,
                         line_in=[t0 + 0.08, t0 + 0.95])

    def risk(self, c, t):
        n = 752 + t * 30
        n = int(n) if n < 777 else 800
        k = ease_io5(lin(t, 0.2, 3.3))
        if self.v:
            cx, cy, zoom = mix(370, 1497, k), mix(560, 610, k), mix(2.2, 1.8, k)
        else:
            cx, cy, zoom = mix(372, 1225, k), mix(560, 640, k), mix(2.2, 1.45, k)
        img = camera(self.srcf(n), cx, cy, zoom, self.W, self.H, sharpen=0.4)
        c += img
        self.flash(c, (1 - ease_out(t / 0.35, 3)) * 0.6)

    MONTAGE = [
        # (frame, cx, cy, zoom)  - detail close-ups cut on every beat
        (70, 950, 540, 1.75),
        (600, 470, 380, 2.0),
        (345, 1340, 560, 2.0),
        (710, 1300, 760, 2.1),
        (290, 700, 470, 2.1),
        (600, 1275, 470, 2.2),
        (800, 360, 560, 2.3),
        (800, 1560, 600, 2.1),
    ]

    MONTAGE_V = [
        (70, 944, 540, 1.3),
        (600, 518, 470, 1.35),
        (345, 1130, 560, 1.9),
        (710, 1120, 705, 1.9),
        (290, 600, 470, 1.9),
        (600, 1110, 470, 2.0),
        (800, 370, 560, 2.2),
        (800, 1497, 600, 1.85),
    ]

    def montage(self, c, t):
        idx = min(int(t / 0.5), 7)
        lt = t - idx * 0.5
        n, cx, cy, zoom = (self.MONTAGE_V if self.v else self.MONTAGE)[idx]
        d = 1 if idx % 2 == 0 else -1
        z = zoom * (1 + 0.07 * ease_out(lt / 0.5, 2))
        img = camera(self.srcf(n), cx + d * 30 * lt, cy, z, self.W, self.H, sharpen=0.45)
        c += img
        self.flash(c, (1 - ease_out(lt / 0.16, 2)) * 0.55)

    def tagline(self, c, t):
        self.bg(c, t, 0.3 * lin(t, 46.0, 48.0))
        t0, t1 = T_TAG
        self.t_tag.draw(c, t, self.W / 2, self.H / 2, t0 + 0.25, t1 - 0.45, stagger=0.12,
                        dur=0.95, exit_dur=0.4,
                        line_in=[t0 + 0.25, t0 + 0.65] if self.v else None,
                        sweep=mix(-0.2, 1.25, lin(t, 47.6, 48.8)) if 47.5 < t < 48.9 else None)

    def endcard(self, c, t):
        W, H = self.W, self.H
        fade = 1 - ease_io(lin(t, 7.4, 8.6))
        if fade <= 0:
            return
        base = np.zeros_like(c)
        base += aurora(T_END[0] + t, W, H, 0.55)
        logo_h = 250 if not self.v else 300
        ly = H * (0.385 if not self.v else 0.40)
        p = ease_out(t / 1.4, 4)
        s = logo_h / self.logo.h * (1.22 - 0.22 * p)
        # glow bloom behind the logo
        g_amt = 0.55 * ease_out(t / 0.25, 2) * (0.75 + 0.25 * math.exp(-t * 1.5)) * \
            (1 + 0.06 * math.sin(t * 2.2))
        self._logo_glow(base, W / 2, ly, logo_h * 1.9, g_amt)
        lay = self.logo
        u = lin(t, 1.1, 2.1)
        if 0 < u < 1:
            lay = _sweep(self.logo, mix(-0.3, 1.3, u), 0, self.logo.w, width=0.14, strength=0.9)
        M = affine(s, W / 2, ly, self.logo.w / 2, self.logo.h / 2)
        blit(base, lay, M, opacity=ease_out(t / 0.4, 2), blur=18 * (1 - p) ** 2)
        wy = ly + logo_h * 0.5 + (120 if not self.v else 130)
        self.word.draw(base, t, W / 2, wy, 0.7, None, dur=0.9, rise=26, blur0=12, whole_line=True)
        self.soon.draw(base, t, W / 2, wy + (82 if not self.v else 86), 1.7, None, dur=0.9, rise=18,
                       blur0=8, whole_line=True)
        fy = H - (48 if not self.v else 150)
        self.foot.draw(base, t, W / 2, fy, 2.4, None, dur=1.0, rise=0, blur0=0, whole_line=True,
                       opacity=0.9)
        f = 1 - ease_out(t / 0.5, 3)
        self.flash(base, f * 0.5, gain=0.5, lift=0.12)
        c += base * fade

    def _logo_glow(self, c, cx, cy, r, amount):
        if amount <= 0:
            return
        H, W = c.shape[:2]
        lw, lh = 160, int(160 * H / W)
        yy, xx = np.mgrid[0:lh, 0:lw].astype(np.float32)
        xx = xx / lw * W
        yy = yy / lh * H
        d = np.sqrt(((xx - cx) / r) ** 2 + ((yy - cy) / (r * 0.8)) ** 2)
        g = np.exp(-d ** 2 * 1.6)
        u = np.clip((xx - (cx - r)) / (2 * r), 0, 1)
        col = grad_color(BRAND, u)
        out = cv2.resize(g[..., None] * col, (W, H), interpolation=cv2.INTER_CUBIC)
        c += out * amount * 0.35

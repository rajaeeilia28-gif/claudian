"""V2 renderer: 30 s product film, 16:9 and 9:16, built from the approved 28.09.2026 recording only.

Every frame also returns a usage record (source frame, visible source rect, zoom, opacity) that the
data/zoom audit consumes. R1 (Risk Lab 4K) is not built yet: preview mode shows marked placeholders,
final mode refuses to render without it.
"""
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V1_PIPELINE = os.path.normpath(os.path.join(HERE, "..", "..", "pipeline"))
sys.path.insert(0, V1_PIPELINE)
from lib import SRC, Layer, blit, blit_persp, ease_out, finish, text_line  # noqa: E402

import v2_spec as S  # noqa: E402

R1_DIR = os.path.normpath(os.path.join(HERE, "..", "assets", "r1"))
PAGE_BLACK = np.array([2, 4, 4], np.float32) / 255.0   # measured around the masked headers


# ---------------------------------------------------------------- curves
def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def sine(x):
    return 0.5 - 0.5 * math.cos(math.pi * clamp01(x))


def quint(x):
    x = clamp01(x)
    return 16 * x ** 5 if x < 0.5 else 1 - (-2 * x + 2) ** 5 / 2


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def hermite(u, m):
    """0->1 with zero start velocity and end velocity m (monotonic for m >= 0)."""
    u = clamp01(u)
    return u * u * (3 - 2 * u) + m * u * u * (u - 1)


def mix(a, b, k):
    return a + (b - a) * k


def mixv(a, b, k):
    return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)


def mixr(a, b, k):
    return tuple(a[j] + (b[j] - a[j]) * k for j in range(4))


# ---------------------------------------------------------------- geometry helpers
def T(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], np.float64)


def Sc(s, cx=0.0, cy=0.0):
    return np.array([[s, 0, cx - s * cx], [0, s, cy - s * cy], [0, 0, 1]], np.float64)


def cam3(z, C, Q):
    """Map source/plane point C to canvas point Q with isotropic zoom z."""
    return np.array([[z, 0, Q[0] - z * C[0]], [0, z, Q[1] - z * C[1]], [0, 0, 1]], np.float64)


def tilt_h(rect, rx_deg, focal=S.FOCAL):
    """Rotate the plane about the horizontal axis through the rect centre and project."""
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a = math.radians(rx_deg)
    src, dst = [], []
    for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
        yy = y - cy
        zc = -yy * math.sin(a)
        k = focal / (focal + zc)
        src.append((x, y))
        dst.append((cx + (x - cx) * k, cy + yy * math.cos(a) * k))
    return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst)).astype(np.float64)


def rounded_mask(w, h, r, ss=4):
    m = np.zeros((h * ss, w * ss), np.uint8)
    r4 = int(r * ss)
    cv2.rectangle(m, (r4, 0), (w * ss - r4, h * ss), 255, -1)
    cv2.rectangle(m, (0, r4), (w * ss, h * ss - r4), 255, -1)
    for cx, cy in [(r4, r4), (w * ss - r4, r4), (r4, h * ss - r4), (w * ss - r4, h * ss - r4)]:
        cv2.circle(m, (cx, cy), r4, 255, -1, lineType=cv2.LINE_AA)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def rect_ramp(w, h, feather):
    """Alpha that is 1 inside and ramps to 0 over `feather` px at every edge."""
    if feather <= 0:
        return np.ones((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.minimum.reduce([xx + 0.5, yy + 0.5, w - xx - 0.5, h - yy - 0.5])
    return smooth_arr(d / feather)


def smooth_arr(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def match_black(img, target, b=None):
    """Lift the crop's background colour to the slab surface; brighter UI colours stay put."""
    if b is None:
        ring = np.concatenate([img[:6].reshape(-1, 3), img[-6:].reshape(-1, 3),
                               img[:, :6].reshape(-1, 3), img[:, -6:].reshape(-1, 3)])
        b = np.median(ring, 0)
    t = np.array(target, np.float32)
    return np.clip(t + (img - b) * (1 - t) / np.maximum(1 - b, 1e-3), 0, 1)


def background(img, k):
    """Smooth background estimate from non-detail pixels only (text, bars and pills excluded)."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    op = cv2.morphologyEx(img, cv2.MORPH_OPEN, ker)
    detail = (img - op).max(2)
    w = (detail < 0.05).astype(np.float32)
    w = cv2.erode(w, np.ones((9, 9), np.uint8))
    sig = k / 3
    den = cv2.GaussianBlur(w, (0, 0), sig)
    num = cv2.GaussianBlur(img * w[..., None], (0, 0), sig)
    fallback = cv2.GaussianBlur(op, (0, 0), sig)
    ok = (den > 0.05)[..., None]
    return np.where(ok, num / np.maximum(den, 1e-3)[..., None], fallback), detail


def flatten_bg(img, target, k):
    """Replace the low-frequency UI background (page/card glow) by the slab colour."""
    bg, _ = background(img, k)
    t = np.array(target, np.float32)
    return np.clip(t + (img - bg) * (1 - t) / np.maximum(1 - bg, 1e-3), 0, 1)


def visible_rect(C, z, Q, W, H):
    return (C[0] - Q[0] / z, C[1] - Q[1] / z, C[0] + (W - Q[0]) / z, C[1] + (H - Q[1]) / z)


def inter(a, b):
    r = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return r if r[2] > r[0] and r[3] > r[1] else None


FULL = (0, 0, 1920, 1080)


class SpotV2:
    def __init__(self, fmt="h", preview=True):
        self.fmt = fmt
        self.W, self.H = S.SIZE[fmt]
        self.preview = preview
        self.r1 = self._load_r1()
        if not preview and self.r1 is None:
            raise RuntimeError("R1 (Risk Lab 4K) fehlt in v2/assets/r1 – finaler Render nicht möglich.")
        self._src_cache = {}
        yy, xx = np.mgrid[0:1080, 0:1920].astype(np.float32)
        self.edge = smooth_arr(np.minimum.reduce([xx, yy, 1919 - xx, 1079 - yy]) / 110.0)[..., None]
        rgba = np.load(os.path.join(V1_PIPELINE, "logo_rgba.npy"))
        self.logo = Layer(rgba[..., :3], rgba[..., 3])
        ys, xs = np.where(rgba[..., 3] > 0.5)
        self.logo_box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        self._build_system()
        self._build_text()
        self.uses = []
        self._rec = True
        self._mask_cache = {}

    # ------------------------------------------------------------ sources
    def _load_r1(self):
        return None if not os.path.isdir(R1_DIR) or not os.listdir(R1_DIR) else R1_DIR

    def src(self, n, feather=False):
        """Approved 1080p frame as float RGB, app header masked; optional 9:16 edge feather."""
        if n not in S.APPROVED_FRAMES:
            raise ValueError(f"Bild {n} ist nicht freigegeben")
        key = (n, feather)
        if key not in self._src_cache:
            if len(self._src_cache) > 40:
                self._src_cache.clear()
            im = SRC.get(n).astype(np.float32) / 255.0
            for x0, y0, x1, y1 in S.HEADER_MASKS.get(n, []):
                im[y0:y1, x0:x1] = PAGE_BLACK
            if feather:
                im = im * self.edge
            self._src_cache[key] = im
        return self._src_cache[key]

    def use(self, shot, n, rect, zoom, opacity=1.0):
        if self._rec and rect is not None and opacity > 0.01:
            self.uses.append(dict(shot=shot, frame=n, rect=[round(v, 1) for v in rect],
                                  zoom=round(float(zoom), 4), opacity=round(float(opacity), 3)))

    # ------------------------------------------------------------ text
    def text(self, s, px, weight, color, track=0.0):
        lay, _, (asc, desc, pad) = text_line([(s, {"fill": color})], px, weight, track)
        return dict(layer=lay, asc=asc, pad=pad, px=px)

    def place_text(self, canvas, tx, cx, cy, opacity=1.0, dy=0.0, H3=None, left=None, top=None):
        """Draw text centred on (cx, cy) (cap-height centre) or with cap-top-left at (left, top)."""
        lay = tx["layer"]
        if left is not None:
            ox = left - tx["pad"]
            oy = top - (tx["pad"] + tx["asc"] - 0.727 * tx["px"])
        else:
            ox = cx - lay.w / 2
            oy = cy - (tx["pad"] + tx["asc"] - 0.364 * tx["px"])
        A = T(ox, oy + dy)
        if H3 is None:
            blit(canvas, lay, A[:2], opacity=opacity)
        else:
            blit_persp(canvas, lay, H3 @ A, opacity=opacity)

    def _build_text(self):
        f = self.fmt
        wm = S.SYSTEM[f]["wordmark"]
        self.t_wordmark = self.text(S.WORDMARK, wm["text_px"], "SemiBold", S.TEXT_C, wm["track"])
        ec = S.ENDCARD[f]
        self.t_word = self.text(S.WORDMARK, ec["word"][1], "SemiBold", S.TEXT_C, ec["word"][2])
        self.t_desc = self.text(S.DESCRIPTOR, ec["desc"][1], "Regular", S.DESC_C, -0.005)
        soon_c = tuple(v * 0.85 for v in S.TEXT_C)
        self.t_soon = self.text(S.SOON, ec["soon"][1], "SemiBold", soon_c, ec["soon"][2])
        self.t_disc = [self.text(line, ec["disc"][1], "TextRegular", S.DISC_C, 0.0)
                       for line in S.DISCLAIMER_LINES[f]]
        self.t_r1 = self.text("R1 ausstehend (Vorschau)", 18 if f == "h" else 20, "SemiBold",
                              (1.0, 0.55, 0.1), 0.02)

    # ------------------------------------------------------------ system object
    def _build_system(self):
        f = self.fmt
        sp = S.SYSTEM[f]
        x0, y0, x1, y1 = sp["slab"]
        self.slab_rect = sp["slab"]
        sw, sh = x1 - x0, y1 - y0
        a = rounded_mask(sw, sh, sp["radius"])
        inner = np.zeros_like(a)
        inner[1:-1, 1:-1] = rounded_mask(sw - 2, sh - 2, sp["radius"] - 1)
        ring = np.clip(a - inner, 0, 1)
        rgb = np.ones((sh, sw, 3), np.float32) * np.array(S.SLAB, np.float32)
        cw, chh = sw / sp["cols"], sh / sp["rows"]
        div = np.zeros((sh, sw), np.float32)
        for k in range(1, sp["cols"]):
            div[:, int(round(k * cw))] = 1
        for k in range(1, sp["rows"]):
            div[int(round(k * chh)), :] = 1
        rgb = rgb * (1 - S.BORDER_A * ring[..., None]) + S.BORDER_A * ring[..., None]
        rgb = rgb * (1 - S.DIVIDER_A * div[..., None]) + S.DIVIDER_A * div[..., None]
        self.slab = Layer(rgb * a[..., None], a)

        self.cells = {}
        for idx, label in enumerate(sp["order"]):
            c, r = idx % sp["cols"], idx // sp["cols"]
            cr = (x0 + round(c * cw), y0 + round(r * chh), x0 + round((c + 1) * cw), y0 + round((r + 1) * chh))
            box = (cr[0] + sp["pad"], cr[1] + sp["content_top"], cr[2] - sp["pad"], cr[3] - sp["pad"])
            lab = self.text(label, sp["label_px"], "SemiBold", S.LABEL_C, sp["label_track"])
            n, crop, scale = S.CELL_CROPS[f][label]
            cell = dict(rect=cr, box=box, label=lab, label_xy=(cr[0] + sp["pad"], cr[1] + sp["pad"]),
                        frame=n, crop=crop, scale=scale, order=idx)
            cw_, ch_ = (crop[2] - crop[0]) * scale, (crop[3] - crop[1]) * scale
            cell["dst"] = ((box[0] + box[2]) / 2 - cw_ / 2, (box[1] + box[3]) / 2 - ch_ / 2)
            if label not in ("MARKT", "RISIKO"):
                cell["layer"] = self._cell_layer(n, crop, scale)
            self.cells[label] = cell

        # dive into MARKT: end state equals Shot 2's first frame (F70 at zoom 1.0)
        m = self.cells["MARKT"]
        self.m_b = PAGE_BLACK
        crop = m["crop"]
        s_m = m["scale"]
        self.A_m = T(m["dst"][0], m["dst"][1]) @ Sc(s_m) @ T(-crop[0], -crop[1])
        hero = S.HERO[f]
        c0 = np.array([hero["c0"][0], hero["c0"][1], 1.0])
        p_end = self.A_m @ c0
        self.dive = dict(Z_end=1.0 / s_m, C_end=(p_end[0], p_end[1]), Q_end=(self.W / 2, self.H / 2),
                         C_pre=((x0 + x1) / 2, (y0 + y1) / 2), Q_pre=((x0 + x1) / 2, (y0 + y1) / 2))
        Z28 = 1.03
        v2 = (hero["z1"] - hero["z0"]) / hero["z0"] * 3 / 3.5          # Shot 2 initial relative zoom speed
        self.dive["m_z"] = v2 * 1.2 / math.log(self.dive["Z_end"] / Z28)
        # pan so that F70's centre moves at Shot 2's initial speed when the dive lands
        pan_end = (-(hero["c1"][1] - hero["c0"][1]) * 3 / 3.5)        # canvas px/s of a fixed source point
        dq = self.dive["Q_end"][1] - self.dive["Q_pre"][1]
        dc = self.dive["C_end"][1] - self.dive["C_pre"][1]
        denom = dq - self.dive["Z_end"] * dc
        self.dive["m_p"] = max(0.0, pan_end * 1.2 / denom) if abs(denom) > 1e-6 else 0.0
        inv = np.linalg.inv(cam3(self.dive["Z_end"], self.dive["C_end"], self.dive["Q_end"]))
        pts = [inv @ np.array([x, y, 1.0]) for x, y in [(0, 0), (self.W, 0), (0, self.H), (self.W, self.H)]]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        self.win_end = (min(xs) - 40, min(ys) - 40, max(xs) + 40, max(ys) + 40)
        self.m_feather_end = 110.0 if f == "v" else 0.0
        self.markt_over = self._markt_overview()

    def _markt_overview(self):
        """F70 as shown in the MARKT cell: blue band fades out elliptically, text stays intact."""
        orig = self.src(70)
        t = np.array(S.SLAB, np.float32)
        fld = S.MARKT_FIELD
        yy, xx = np.mgrid[0:1080, 0:1920].astype(np.float32)
        d = np.sqrt(((xx - fld["c"][0]) / fld["r"][0]) ** 2 + ((yy - fld["c"][1]) / fld["r"][1]) ** 2)
        m_e = 1 - smooth_arr((d - fld["inner"]) / (1 - fld["inner"]))
        bg, _ = background(orig, S.FLATTEN_KERNEL)
        detail = (orig - bg).max(2)
        box = np.zeros((1080, 1920), np.float32)
        for x0, y0, x1, y1 in S.MARKT_TEXT_BOXES:
            box[y0:y1, x0:x1] = 1
        box = cv2.GaussianBlur(box, (0, 0), 2)
        m_p = box * smooth_arr(detail / 0.22)
        m = np.maximum(m_e, m_p)[..., None]
        return (t + m * (orig - t)).astype(np.float32)

    def _cell_layer(self, n, crop, scale):
        x0, y0, x1, y1 = crop
        im = flatten_bg(self.src(n)[y0:y1, x0:x1].copy(), S.SLAB, S.FLATTEN_KERNEL)
        w, h = int(round((x1 - x0) * scale)), int(round((y1 - y0) * scale))
        if scale != 1.0:
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        a = rect_ramp(w, h, S.CROP_FEATHER)
        return Layer(im * a[..., None], a)

    def plane_P(self, rx, Z, C, Q):
        return cam3(Z, C, Q) @ tilt_h(self.slab_rect, rx)

    def draw_system(self, canvas, shot, P, Z, o_all, slab_op, cell_op, label_op, label_dy, markt):
        """Slab, cells and labels on the plane P. `markt` holds the MARKT window state."""
        blit_persp(canvas, self.slab, P @ T(self.slab_rect[0], self.slab_rect[1]), opacity=slab_op)
        for label, cell in self.cells.items():
            if label == "MARKT":
                self._draw_markt(canvas, P, Z, o_all, markt, shot)
            elif label == "RISIKO":
                if self.r1 is None and self.preview:
                    self._r1_placeholder(canvas, P, cell, o_all * cell_op[label])
            else:
                A = T(cell["dst"][0], cell["dst"][1])
                blit_persp(canvas, cell["layer"], P @ A, opacity=o_all * cell_op[label])
                self.use(shot, cell["frame"], cell["crop"], cell["scale"] * Z, o_all * cell_op[label])
            if label == "RISIKO" and self.r1 is None:
                continue
            lx, ly = cell["label_xy"]
            self.place_text(canvas, cell["label"], 0, 0, opacity=o_all * label_op[label],
                            dy=label_dy[label], H3=P, left=lx, top=ly)

    def _r1_placeholder(self, canvas, P, cell, op):
        if op <= 0.01:
            return
        x0, y0, x1, y1 = cell["box"]
        w, h = x1 - x0, y1 - y0
        a = np.zeros((h, w), np.float32)
        for k in range(0, w, 14):
            a[0:2, k:k + 7] = 1
            a[h - 2:h, k:k + 7] = 1
        for k in range(0, h, 14):
            a[k:k + 7, 0:2] = 1
            a[k:k + 7, w - 2:w] = 1
        col = np.array([1.0, 0.55, 0.1], np.float32)
        blit_persp(canvas, Layer(a[..., None] * col, a), P @ T(x0, y0), opacity=op)
        self.place_text(canvas, self.t_r1, (x0 + x1) / 2, (y0 + y1) / 2, opacity=op,
                        H3=P)

    def _draw_markt(self, canvas, P, Z, o_all, st, shot):
        """MARKT window: F70 through the plane, with source crop and plane window both expanding."""
        cell = self.cells["MARKT"]
        src = self.src(70)
        lam = st["lift"]
        if lam > 0:
            src = src + lam * (self.markt_over - src)
        cx0, cy0, cx1, cy1 = st["src_rect"]
        fx, fy = st["src_feather"]
        yy, xx = np.ogrid[0:1080, 0:1920]
        dx = np.minimum(xx - cx0, cx1 - xx).astype(np.float32)
        dy = np.minimum(yy - cy0, cy1 - yy).astype(np.float32)
        ax = smooth_arr(dx / fx) if fx > 0.5 else (dx >= 0).astype(np.float32)
        ay = smooth_arr(dy / fy) if fy > 0.5 else (dy >= 0).astype(np.float32)
        a_s = ax * ay
        rgba = np.dstack([src * a_s[..., None], a_s]).astype(np.float32)
        Hm = P @ self.A_m
        out = cv2.warpPerspective(rgba, Hm, (self.W, self.H), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        wx0, wy0, wx1, wy1 = st["win"]
        pts = np.array([[wx0, wy0], [wx1, wy0], [wx1, wy1], [wx0, wy1]], np.float64)
        ph = (P @ np.hstack([pts, np.ones((4, 1))]).T).T
        poly = (ph[:, :2] / ph[:, 2:3] * 2).astype(np.int32)
        big = np.zeros((self.H * 2, self.W * 2), np.uint8)
        cv2.fillPoly(big, [poly], 255, lineType=cv2.LINE_AA)
        win = cv2.resize(big, (self.W, self.H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if st.get("win_soft", 0) > 0.5:
            win = cv2.GaussianBlur(win, (0, 0), st["win_soft"])
        np.clip(out, 0, None, out=out)
        k = (win * o_all)[..., None]
        canvas *= 1 - np.minimum(out[..., 3:4], 1) * k
        canvas += out[..., :3] * k
        z_src = cell["scale"] * Z
        self.use(shot, 70, (cx0, cy0, cx1, cy1), z_src, o_all)

    # ------------------------------------------------------------ frame
    def frame(self, i, mblur=True):
        self.uses = []
        t = i / S.FPS
        c = np.zeros((self.H, self.W, 3), np.float32)
        if t < 4.0:
            if mblur and t >= 2.8:
                acc = np.zeros_like(c)
                offs = np.linspace(-0.25, 0.25, 6) / S.FPS
                for k, o in enumerate(offs):
                    self._rec = k == len(offs) // 2
                    sub = np.zeros_like(c)
                    self.s1(sub, t + o)
                    acc += sub
                self._rec = True
                c = acc / len(offs)
            else:
                self.s1(c, t)
        elif t < 7.5:
            self.s2(c, t)
        elif t < 12.0:
            self.s3(c, t)
        elif t < 15.5:
            self.s4(c, t)
        elif t < 19.0:
            self.s5(c, t)
        elif t < 24.4:
            self.r1_pending(c, t)
        else:
            self.s7(c, t)
        return finish(c, i, vig=False), list(self.uses)

    # ------------------------------------------------------------ Shot 1
    def s1_state(self, t):
        d = self.dive
        o_in = ease_out(t / 0.5, 3)
        s_in = 0.98 + 0.02 * o_in
        Zp = s_in * (1 + 0.03 * sine(t / 2.8))
        rx = S.TILT_DEG * (1 - sine((t - 1.6) / 1.6))
        if t < 2.8:
            Z, C, Q = Zp, d["C_pre"], d["Q_pre"]
        else:
            u = (t - 2.8) / 1.2
            Z28 = 1.03
            Z = Z28 * math.exp(hermite(u, d["m_z"]) * math.log(d["Z_end"] / Z28))
            wp = hermite(u, d["m_p"])
            C, Q = mixv(d["C_pre"], d["C_end"], wp), mixv(d["Q_pre"], d["Q_end"], wp)
        e_w = smooth((t - 3.0) / 0.9)
        e_f = smooth((t - 3.0) / 1.0)          # source edges: soft while opening, final exactly at 4.0 s
        cell = self.cells["MARKT"]
        s_m = cell["scale"]
        soft = 90 * math.sin(math.pi * e_f)
        markt = dict(lift=1 - smooth((t - 3.1) / 0.5),
                     src_rect=mixr(cell["crop"], (0, 0, 1920, 1080), e_w),
                     src_feather=(max(mix(S.MARKT_FEATHER[0] / s_m, self.m_feather_end, e_f), soft),
                                  max(mix(S.MARKT_FEATHER[1] / s_m, self.m_feather_end, e_f), soft)),
                     win=mixr(cell["rect"], self.win_end, e_w),
                     win_soft=28 * math.sin(math.pi * e_w))
        fade = 1 - smooth((t - 3.1) / 0.5)
        dim = sine((t - 2.0) / 0.8)
        cell_op, label_op, label_dy = {}, {}, {}
        for label, cl in self.cells.items():
            p = ease_out((t - 0.5 - 0.08 * cl["order"]) / 0.6, 3)
            other = label != "MARKT"
            cell_op[label] = fade * (1 - 0.45 * dim) if other else 1.0
            label_op[label] = p * fade * ((1 - 0.55 * dim) if other else 1.0)
            label_dy[label] = (1 - p) * 10
        return dict(o_in=o_in, Z=Z, C=C, Q=Q, rx=rx, markt=markt, fade=fade,
                    cell_op=cell_op, label_op=label_op, label_dy=label_dy)

    def s1(self, c, t):
        st = self.s1_state(t)
        P = self.plane_P(st["rx"], st["Z"], st["C"], st["Q"])
        self.draw_system(c, "s1_system", P, st["Z"], st["o_in"], st["o_in"] * st["fade"], st["cell_op"],
                         st["label_op"], st["label_dy"], st["markt"])
        self._wordmark(c, t)

    def _wordmark(self, c, t):
        wm = S.SYSTEM[self.fmt]["wordmark"]
        p = ease_out((t - 0.3) / 0.6, 3)
        q = smooth((t - 2.6) / 0.4)
        op = p * (1 - q)
        if op <= 0.01:
            return
        dy = (1 - p) * 10
        lb = self.logo_box
        s = wm["logo_px"] / (lb[3] - lb[1])
        lw = (lb[2] - lb[0]) * s
        tw = self.t_wordmark["layer"].w - 2 * self.t_wordmark["pad"]
        total = lw + wm["gap"] + tw
        x_logo = self.W / 2 - total / 2
        A = T(x_logo, wm["cy"] - wm["logo_px"] / 2 + dy) @ Sc(s) @ T(-lb[0], -lb[1])
        blit(c, self.logo, A[:2], opacity=op)
        self.place_text(c, self.t_wordmark, x_logo + lw + wm["gap"] + tw / 2, wm["cy"], opacity=op, dy=dy)

    # ------------------------------------------------------------ Shot 2
    def s2(self, c, t):
        h = S.HERO[self.fmt]
        u = ease_out((t - 4.0) / 3.5, 3)
        z = mix(h["z0"], h["z1"], u)
        C = mixv(h["c0"], h["c1"], u)
        Q = (self.W / 2, self.H / 2)
        self._camera(c, 70, C, z, Q, feather=self.fmt == "v", shot="s2_hero")

    def _camera(self, c, n, C, z, Q, feather=False, shot="", sigma_map=None):
        src = self.src(n, feather)
        M = cam3(z, C, Q)[:2]
        interp = cv2.INTER_CUBIC if z > 1.001 else cv2.INTER_LINEAR
        out = cv2.warpAffine(src, M, (self.W, self.H), flags=interp, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
        np.clip(out, 0, 1, out=out)
        c += out
        self.use(shot, n, inter(visible_rect(C, z, Q, self.W, self.H), FULL), z)
        return out

    # ------------------------------------------------------------ Shot 3
    def s3(self, c, t):
        d = S.DATA
        k = int(math.floor((t - 7.5) * 30 + 1e-6))
        n = d["first"] + k if d["first"] + k <= d["last"] else d["hold"]
        p = d[self.fmt]
        z = mix(p["z0"], p["z1"], sine((t - 7.5) / 4.5))
        self._plate(c, n, p["rect"], p["radius"], p["src_c"], z, p["dst_c"], "s3_data")

    def _plate(self, c, n, rect, radius, C, z, Q, shot):
        x0, y0, x1, y1 = rect
        src = self.src(n)[y0:y1, x0:x1]
        key = (x1 - x0, y1 - y0, radius)
        if key not in self._mask_cache:
            self._mask_cache[key] = rounded_mask(*key)
        a = self._mask_cache[key]
        lay = Layer(src * a[..., None], a)
        A = cam3(z, (C[0] - x0, C[1] - y0), Q)
        blit(c, lay, A[:2])
        vis = inter(visible_rect(C, z, Q, self.W, self.H), rect)
        self.use(shot, n, vis, z)

    # ------------------------------------------------------------ Shot 4
    def s4(self, c, t):
        fl = S.FLOW
        p = fl[self.fmt]
        a, b = p["a"], p["b"]
        if t < fl["t_move"][0]:
            u = sine((t - fl["t_hold_a"][0]) / (fl["t_hold_a"][1] - fl["t_hold_a"][0]))
            C, z, Q = mixv(a["c"], a["c_end"], u), a["z"], a["dst"]
        else:
            u = quint((t - fl["t_move"][0]) / (fl["t_move"][1] - fl["t_move"][0]))
            C = mixv(a["c_end"], b["c"], u)
            z = math.exp(mix(math.log(a["z"]), math.log(b["z"]), u))
            Q = mixv(a["dst"], b["dst"], u)
        img = np.zeros_like(c)
        self._camera(img, 345, C, z, Q, feather=self.fmt == "v", shot="s4_flow")
        f = smooth((t - fl["t_focus"][0]) / (fl["t_focus"][1] - fl["t_focus"][0]))
        yy, xx = np.ogrid[0:self.H, 0:self.W]
        fa = fl["focus_a"]
        acx, acy = Q[0] + z * (fa["c"][0] - C[0]), Q[1] + z * (fa["c"][1] - C[1])
        dA = np.sqrt(((xx - acx) / (fa["r"][0] * z)) ** 2 + ((yy - acy) / (fa["r"][1] * z)) ** 2)
        mA = 1 - smooth_arr((dA - 0.9) / 0.7)
        bx0, by0, bx1, by1 = fl["focus_b"]
        rx0, ry0 = Q[0] + z * (bx0 - C[0]), Q[1] + z * (by0 - C[1])
        rx1, ry1 = Q[0] + z * (bx1 - C[0]), Q[1] + z * (by1 - C[1])
        dx = np.maximum(np.maximum(rx0 - xx, xx - rx1), 0)
        dy = np.maximum(np.maximum(ry0 - yy, yy - ry1), 0)
        mB = 1 - smooth_arr(np.sqrt(dx ** 2 + dy ** 2) / (90 * z))
        m = ((1 - f) * mA + f * mB).astype(np.float32)[..., None]
        blur = cv2.GaussianBlur(img, (0, 0), fl["blur_sigma"])
        c += img * m + blur * (1 - m)

    # ------------------------------------------------------------ Shot 5
    def s5(self, c, t):
        mc = S.MACRO
        k = int(math.floor((t - 15.5) * 30 + 1e-6))
        n = mc["first"] + k if mc["first"] + k <= mc["last"] else mc["hold"]
        p = mc[self.fmt]
        u = quint((t - mc["t_move"][0]) / (mc["t_move"][1] - mc["t_move"][0]))
        C = mixv(p["c0"], p["c1"], u)
        z = math.exp(mix(math.log(p["z0"]), math.log(p["z1"]), u))
        self._plate(c, n, p["rect"], p["radius"], C, z, p["dst"], "s5_macro")

    # ------------------------------------------------------------ Shot 6 / pull-back (R1)
    def r1_pending(self, c, t):
        if self.r1 is not None:
            raise NotImplementedError("R1 ist vorhanden, Shot 6 und Rückfahrt sind noch nicht gebaut.")
        if not self.preview:
            raise RuntimeError("R1 fehlt")
        self.place_text(c, self.t_r1, self.W / 2, self.H / 2)

    # ------------------------------------------------------------ Shot 7 (hold, fade, endcard)
    def s7(self, c, t):
        if t < 26.0:
            fade = 1 - sine((t - 25.4) / 0.6)
            s = 1 - 0.03 * sine((t - 25.4) / 0.6)
            sx0, sy0, sx1, sy1 = self.slab_rect
            P = Sc(s, (sx0 + sx1) / 2, (sy0 + sy1) / 2)
            ops = {k: 1.0 for k in self.cells}
            s_m = self.cells["MARKT"]["scale"]
            st = dict(lift=1.0, src_rect=self.cells["MARKT"]["crop"],
                      src_feather=(S.MARKT_FEATHER[0] / s_m, S.MARKT_FEATHER[1] / s_m),
                      win=self.cells["MARKT"]["rect"], win_soft=0)
            self.draw_system(c, "s7_system", P, s, fade, fade, ops, ops, {k: 0.0 for k in self.cells}, st)
        self._endcard(c, t)

    def _endcard(self, c, t):
        ec = S.ENDCARD[self.fmt]
        out = 1 - smooth((t - 29.6) / 0.4)
        # disclaimer from 25.4 (opacity only)
        dop = ease_out((t - 25.4) / 0.4, 3) * out
        for tx, y in zip(self.t_disc, ec["disc"][0]):
            self.place_text(c, tx, self.W / 2, y, opacity=dop)
        if t < 26.0:
            return
        lx, ly, lh = ec["logo"]
        p = ease_out((t - 26.0) / 0.6, 3)
        lb = self.logo_box
        s = lh / (lb[3] - lb[1])
        A = T(lx - (lb[2] - lb[0]) * s / 2, ly - lh / 2 + (1 - p) * 10) @ Sc(s) @ T(-lb[0], -lb[1])
        blit(c, self.logo, A[:2], opacity=p * out)
        for tx, t0, y in [(self.t_word, 26.3, ec["word"][0]), (self.t_desc, 26.6, ec["desc"][0]),
                          (self.t_soon, 26.9, ec["soon"][0])]:
            q = ease_out((t - t0) / 0.6, 3)
            self.place_text(c, tx, self.W / 2, y, opacity=q * out, dy=(1 - q) * 10)

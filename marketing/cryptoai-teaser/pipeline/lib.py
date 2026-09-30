"""Compositing helpers for the Crypto AI teaser: text, plates, camera moves."""
import functools
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
FONT_DIR = os.path.join(ROOT, "fonts", "extras", "ttf")

WHITE = (1.0, 1.0, 1.0)
GRAY = (0.631, 0.631, 0.651)      # Apple secondary gray #A1A1A6
DIM = (0.525, 0.525, 0.545)       # #86868B
BRAND = [                          # sampled from the platform's "Bullisch" + logo
    (0.00, (0.25, 0.47, 1.00)),
    (0.35, (0.52, 0.40, 0.97)),
    (0.68, (0.80, 0.42, 0.72)),
    (1.00, (0.96, 0.58, 0.28)),
]
WARM = [(0.0, (0.99, 0.62, 0.25)), (0.5, (0.97, 0.42, 0.42)), (1.0, (0.86, 0.36, 0.72))]
MINT = [(0.0, (0.20, 0.85, 0.62)), (1.0, (0.35, 0.62, 1.00))]


# ---------------------------------------------------------------- easing
def clamp01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def lin(t, a, b):
    return clamp01((t - a) / (b - a)) if b != a else float(t >= b)


def ease_out(x, p=3):
    x = clamp01(x)
    return 1 - (1 - x) ** p


def ease_in(x, p=3):
    x = clamp01(x)
    return x ** p


def ease_io(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x) if x < 1 else 1.0


def ease_io5(x):
    x = clamp01(x)
    return x ** 3 * (x * (6 * x - 15) + 10)


def ease_expo_out(x):
    x = clamp01(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def mix(a, b, k):
    return a + (b - a) * k


def grad_color(stops, u):
    u = np.clip(u, 0, 1)
    out = np.zeros(u.shape + (3,), np.float32)
    pos = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], np.float32)
    for c in range(3):
        out[..., c] = np.interp(u, pos, cols[:, c])
    return out


# ---------------------------------------------------------------- fonts / text
@functools.lru_cache(maxsize=64)
def font(weight, px):
    name = {
        "Black": "InterDisplay-Black.ttf",
        "ExtraBold": "InterDisplay-ExtraBold.ttf",
        "Bold": "InterDisplay-Bold.ttf",
        "SemiBold": "InterDisplay-SemiBold.ttf",
        "Medium": "InterDisplay-Medium.ttf",
        "Regular": "InterDisplay-Regular.ttf",
        "Light": "InterDisplay-Light.ttf",
        "TextRegular": "Inter-Regular.ttf",
        "TextMedium": "Inter-Medium.ttf",
    }[weight]
    return ImageFont.truetype(os.path.join(FONT_DIR, name), int(px))


class Layer:
    """Premultiplied RGB + alpha image with a local origin."""

    def __init__(self, rgb, a):
        self.rgb = rgb.astype(np.float32)
        self.a = a.astype(np.float32)

    @property
    def w(self):
        return self.a.shape[1]

    @property
    def h(self):
        return self.a.shape[0]

    def rgba(self):
        return np.dstack([self.rgb, self.a])


def _fill_for(style, xs_global, line_w, h):
    fill = style.get("fill", WHITE)
    if isinstance(fill, list):
        u = xs_global / max(line_w, 1)
        row = grad_color(fill, u)
        return np.broadcast_to(row[None, :, :], (h, row.shape[0], 3))
    return np.broadcast_to(np.array(fill, np.float32)[None, None, :], (h, len(xs_global), 3))


def text_line(segments, size, weight="Bold", tracking=-0.022, ss=2, features=None):
    """Render one line built from (text, style) segments.

    Returns (layer, words) where layer is the full line and words is a list of
    (Layer, x_offset) so callers can stagger word animations.
    """
    f = font(weight, size * ss)
    track = tracking * size * ss
    full = "".join(s[0] for s in segments)
    ascent, descent = f.getmetrics()
    pad = int(size * ss * 0.25)
    hpx = ascent + descent + 2 * pad
    # character x positions with kerning from raqm prefix measurement
    xs = []
    for i in range(len(full)):
        xs.append(f.getlength(full[:i], features=features) + i * track)
    total = f.getlength(full, features=features) + (len(full) - 1) * track
    wpx = int(math.ceil(total)) + 2 * pad
    styles = []
    for text, st in segments:
        styles += [st] * len(text)

    # word spans (split on spaces) for staggered reveals
    words = []
    i = 0
    while i < len(full):
        if full[i] == " ":
            i += 1
            continue
        j = i
        while j < len(full) and full[j] != " ":
            j += 1
        words.append((i, j))
        i = j

    def render_span(i0, i1):
        x0 = xs[i0]
        x1 = (xs[i1 - 1] + f.getlength(full[i1 - 1], features=features)) if i1 > i0 else x0
        w = int(math.ceil(x1 - x0)) + 2 * pad
        rgb = np.zeros((hpx, w, 3), np.float32)
        alpha = np.zeros((hpx, w), np.float32)
        # group consecutive characters sharing a style
        k = i0
        while k < i1:
            st = styles[k]
            m = k
            while m < i1 and styles[m] is st:
                m += 1
            img = Image.new("L", (w, hpx), 0)
            d = ImageDraw.Draw(img)
            for c in range(k, m):
                d.text((xs[c] - x0 + pad, pad), full[c], font=f, fill=255, features=features)
            a = np.asarray(img, np.float32) / 255.0
            xs_global = np.arange(w) + x0 - pad
            col = _fill_for(st, xs_global, total, hpx)
            op = st.get("opacity", 1.0)
            a = a * op
            rgb = rgb * (1 - a[..., None]) + col * a[..., None]
            alpha = alpha + a * (1 - alpha)
            k = m
        # rgb above is straight colour composited on black == premultiplied
        lay = Layer(rgb, alpha)
        if ss != 1:
            lay = Layer(
                cv2.resize(lay.rgb, (max(1, w // ss), hpx // ss), interpolation=cv2.INTER_AREA),
                cv2.resize(lay.a, (max(1, w // ss), hpx // ss), interpolation=cv2.INTER_AREA),
            )
        return lay, (x0 - pad) / ss

    line, _ = render_span(0, len(full))
    word_layers = [render_span(a, b) for a, b in words]
    return line, word_layers, (ascent / ss, descent / ss, pad / ss)


# ---------------------------------------------------------------- compositing
def blit(canvas, layer, M, opacity=1.0, blur=0.0, add=False):
    """Composite a premultiplied layer via 2x3 affine M (layer px -> canvas px)."""
    if opacity <= 0.002:
        return
    H, W = canvas.shape[:2]
    h, w = layer.a.shape
    rgba = layer.rgba()
    M = np.asarray(M, np.float64)
    sx = math.hypot(M[0, 0], M[1, 0])
    if sx < 0.6:  # pre-shrink for clean minification
        f = sx / 0.9
        nw, nh = max(1, int(round(w * f))), max(1, int(round(h * f)))
        rgba = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)
        M = M.copy()
        M[:, 0] *= w / nw
        M[:, 1] *= h / nh
        h, w = nh, nw
    bx, by = (blur if isinstance(blur, tuple) else (blur, blur))
    pad = int(3 * max(bx, by)) + 3
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64)
    dst = corners @ M.T
    x0 = max(int(math.floor(dst[:, 0].min())) - pad, 0)
    x1 = min(int(math.ceil(dst[:, 0].max())) + pad, W)
    y0 = max(int(math.floor(dst[:, 1].min())) - pad, 0)
    y1 = min(int(math.ceil(dst[:, 1].max())) + pad, H)
    if x1 <= x0 or y1 <= y0:
        return
    M2 = M.copy()
    M2[0, 2] -= x0
    M2[1, 2] -= y0
    interp = cv2.INTER_CUBIC if sx > 1.05 else cv2.INTER_LINEAR
    out = cv2.warpAffine(rgba, M2, (x1 - x0, y1 - y0), flags=interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    if bx > 0.25 or by > 0.25:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=max(bx, 0.01), sigmaY=max(by, 0.01))
    np.clip(out, 0, None, out=out)
    out[..., 3] = np.minimum(out[..., 3], 1.0)
    if opacity != 1.0:
        out *= opacity
    region = canvas[y0:y1, x0:x1]
    if add:
        region += out[..., :3]
    else:
        region *= (1 - out[..., 3:4])
        region += out[..., :3]


def blit_persp(canvas, layer, Hm, opacity=1.0, blur=0.0):
    """Composite a layer through a 3x3 homography (layer px -> canvas px)."""
    if opacity <= 0.002:
        return
    Hc, Wc = canvas.shape[:2]
    h, w = layer.a.shape
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64).T
    d = Hm @ corners
    d = d[:2] / d[2]
    pad = int(3 * blur) + 3
    x0 = max(int(math.floor(d[0].min())) - pad, 0)
    x1 = min(int(math.ceil(d[0].max())) + pad, Wc)
    y0 = max(int(math.floor(d[1].min())) - pad, 0)
    y1 = min(int(math.ceil(d[1].max())) + pad, Hc)
    if x1 <= x0 or y1 <= y0:
        return
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    out = cv2.warpPerspective(layer.rgba(), T @ Hm, (x1 - x0, y1 - y0), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    if blur > 0.25:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    np.clip(out, 0, None, out=out)
    out[..., 3] = np.minimum(out[..., 3], 1.0)
    out *= opacity
    region = canvas[y0:y1, x0:x1]
    region *= (1 - out[..., 3:4])
    region += out[..., :3]


def affine(scale, cx_dst, cy_dst, cx_src, cy_src, rot_deg=0.0):
    """Map layer point (cx_src, cy_src) to canvas (cx_dst, cy_dst) with scale/rotation."""
    r = math.radians(rot_deg)
    c, s = math.cos(r) * scale, math.sin(r) * scale
    return np.array([[c, -s, cx_dst - (c * cx_src - s * cy_src)],
                     [s, c, cy_dst - (s * cx_src + c * cy_src)]], np.float64)


def card_homography(w, h, cx, cy, scale, rx=0.0, ry=0.0, rz=0.0, focal=2400.0):
    """Project a w x h card rotated in 3D, centred at canvas (cx, cy)."""
    pts = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]],
                   np.float64) * scale
    ax, ay, az = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    p = pts @ (Rz @ Ry @ Rx).T
    z = p[:, 2] + focal
    proj = np.stack([cx + p[:, 0] * focal / z, cy + p[:, 1] * focal / z], 1).astype(np.float32)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    return cv2.getPerspectiveTransform(src, proj)


# ---------------------------------------------------------------- source video
class SourceCache:
    def __init__(self, maxn=48):
        self.maxn = maxn
        self.cache = {}
        self.order = []

    def get(self, n):
        n = int(max(1, min(900, n)))
        if n in self.cache:
            return self.cache[n]
        im = cv2.imread(os.path.join(SRC_DIR, f"{n:04d}.png"), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
        self.cache[n] = im
        self.order.append(n)
        if len(self.order) > self.maxn:
            old = self.order.pop(0)
            self.cache.pop(old, None)
        return im


SRC = SourceCache()


def camera(src_u8, cx, cy, zoom, W, H, sharpen=0.0, rot=0.0):
    """Render source frame so that source (cx, cy) lands at canvas centre."""
    M = affine(zoom, W / 2, H / 2, cx, cy, rot)
    interp = cv2.INTER_CUBIC if zoom > 1.0 else cv2.INTER_AREA if zoom < 0.7 else cv2.INTER_LINEAR
    out = cv2.warpAffine(src_u8, M, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0)).astype(np.float32) / 255.0
    if sharpen > 0:
        bl = cv2.GaussianBlur(out, (0, 0), 1.3)
        out = out + sharpen * (out - bl)
        np.clip(out, 0, 1, out=out)
    return out


def src_layer(src_u8, x0, y0, x1, y1, radius=0, feather=0.0):
    """Cut a rectangular (optionally rounded) plate out of a source frame."""
    crop = src_u8[y0:y1, x0:x1].astype(np.float32) / 255.0
    h, w = crop.shape[:2]
    a = np.ones((h, w), np.float32)
    if radius > 0:
        m = np.zeros((h * 4, w * 4), np.uint8)
        r4 = radius * 4
        cv2.rectangle(m, (r4, 0), (w * 4 - r4, h * 4), 255, -1)
        cv2.rectangle(m, (0, r4), (w * 4, h * 4 - r4), 255, -1)
        for cx, cy in [(r4, r4), (w * 4 - r4, r4), (r4, h * 4 - r4), (w * 4 - r4, h * 4 - r4)]:
            cv2.circle(m, (cx, cy), r4, 255, -1, lineType=cv2.LINE_AA)
        a = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if feather > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]) / feather
        a = a * np.clip(d, 0, 1) ** 1.5
    return Layer(crop * a[..., None], a)


# ---------------------------------------------------------------- backgrounds / finishing
@functools.lru_cache(maxsize=4)
def vignette(W, H, strength=0.42):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W / 2) / (W / 2)
    ny = (yy - H / 2) / (H / 2)
    r = np.sqrt(nx ** 2 * 0.85 + ny ** 2 * 1.0)
    v = 1 - strength * np.clip(r - 0.35, 0, None) ** 1.6
    return np.clip(v, 0, 1)[..., None].astype(np.float32)


@functools.lru_cache(maxsize=4)
def _dither(W, H):
    rng = np.random.default_rng(7)
    return [(rng.random((H, W, 1), dtype=np.float32) - 0.5) * (2.2 / 255) for _ in range(8)]


def aurora(t, W, H, amount=1.0, hue_shift=0.0, center=(0.5, 0.55)):
    """Slow-moving brand-coloured light fog on black."""
    lw, lh = 192, max(8, int(192 * H / W))
    yy, xx = np.mgrid[0:lh, 0:lw].astype(np.float32)
    xx /= lw
    yy /= lh
    out = np.zeros((lh, lw, 3), np.float32)
    blobs = [
        (0.30, 0.62, 0.34, (0.16, 0.30, 0.95), 0.9, 0.00),
        (0.70, 0.45, 0.30, (0.45, 0.25, 0.90), 0.7, 1.70),
        (0.55, 0.80, 0.28, (0.85, 0.40, 0.25), 0.45, 3.10),
    ]
    aspect = W / H
    for bx, by, r, col, k, ph in blobs:
        cx = bx + 0.10 * math.sin(0.23 * t + ph) + (center[0] - 0.5)
        cy = by + 0.07 * math.cos(0.19 * t + ph * 1.3) + (center[1] - 0.55)
        d2 = ((xx - cx) * aspect) ** 2 + (yy - cy) ** 2
        g = np.exp(-d2 / (2 * (r * aspect * 0.6) ** 2))
        out += g[..., None] * np.array(col, np.float32) * k
    out *= 0.085 * amount
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)


def finish(canvas, frame_idx, vig=True):
    W, H = canvas.shape[1], canvas.shape[0]
    if vig:
        canvas *= vignette(W, H)
    canvas += _dither(W, H)[frame_idx % 8]
    np.clip(canvas, 0, 1, out=canvas)
    return (canvas * 255 + 0.5).astype(np.uint8)


_GLOW_CACHE = {}


def glow_layer(w, h, color, sigma, pad=None, radius=24):
    """Soft coloured glow shaped like a rounded card, as a Layer (cached: it is static per card)."""
    key = (w, h, repr(color), sigma, pad, radius)
    if key not in _GLOW_CACHE:
        _GLOW_CACHE[key] = _glow_layer(w, h, color, sigma, pad, radius)
    return _GLOW_CACHE[key]


def _glow_layer(w, h, color, sigma, pad, radius):
    pad = pad or int(sigma * 3)
    W2, H2 = w + 2 * pad, h + 2 * pad
    m = np.zeros((H2, W2), np.float32)
    cv2.rectangle(m, (pad + radius, pad), (pad + w - radius, pad + h), 1.0, -1)
    cv2.rectangle(m, (pad, pad + radius), (pad + w, pad + h - radius), 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), sigma)
    if isinstance(color, list):
        col = grad_color(color, np.linspace(0, 1, W2))[None, :, :]
    else:
        col = np.array(color, np.float32)[None, None, :]
    return Layer(m[..., None] * col, m * 0.0), pad


def add_layer(canvas, layer, M, opacity=1.0):
    blit(canvas, layer, M, opacity=opacity, add=True)

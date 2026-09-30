"""Approved V2 specification as data: timing, crops, camera keyframes, layout, data rules.

Coordinates are source pixels in the 1080p grid of the 28.09.2026 recording (R1 4K = same values x2).
"""

FPS = 60
DURATION = 30.0
N_FRAMES = int(round(DURATION * FPS))

SIZE = {"h": (1920, 1080), "v": (1080, 1920)}

SHOTS = {  # (start, end) in seconds
    "s1_system": (0.0, 4.0),
    "s2_hero": (4.0, 7.5),
    "s3_data": (7.5, 12.0),
    "s4_flow": (12.0, 15.5),
    "s5_macro": (15.5, 19.0),
    "s6_risk": (19.0, 23.0),      # R1
    "s7_system": (23.0, 30.0),    # pull-back part (23.0-24.4) needs R1
}
HARD_CUTS = [7.5, 12.0, 15.5, 19.0]
ACCENTS = [7.5, 12.0, 15.5, 19.0, 23.0]
LIFT_T = 4.0
IMPACT_T = 26.0

# ---------------------------------------------------------------- data rules
APPROVED_FRAMES = {70, 345, 600, 710} | set(range(452, 541)) | set(range(668, 701))
FORBIDDEN_FRAMES = {290, 220, 140} | set(range(77, 217)) | set(range(369, 398))
HEADER_MASKS = {70: [(30, 30, 200, 80)], 345: [(20, 20, 300, 140), (1430, 70, 1900, 135)]}
# Regions that must never be visible (frame set, rect) and regions that must never be a hero.
FORBIDDEN_REGIONS = [
    ("ADA-Modell-Sentiment / 99 Modelle", set(range(668, 701)) | {710}, (96, 845, 1823, 1011)),
]
NO_HERO_REGIONS = [  # visible only at zoom <= 1.0
    ("97,4 %-Kachel", set(range(452, 541)) | {600}, (950, 308, 1352, 448)),
]

# ---------------------------------------------------------------- text
LABELS = ["MARKT", "DATEN", "GELDFLUSS", "DERIVATE", "MAKRO", "RISIKO"]
WORDMARK = "CRYPTO AI"
DESCRIPTOR = "KI-gestützte Marktanalyse über mehrere Dimensionen."
SOON = "BALD VERFÜGBAR."
DISCLAIMER = ("Alle Werte: Aufnahmen der Crypto-AI-Plattform vom 28.09.2026 · keine Live-Daten. "
              "Keine Anlageberatung. Krypto-Assets sind mit hohen Risiken verbunden.")
DISCLAIMER_LINES = {
    "h": ["Alle Werte: Aufnahmen der Crypto-AI-Plattform vom 28.09.2026 · keine Live-Daten.",
          "Keine Anlageberatung. Krypto-Assets sind mit hohen Risiken verbunden."],
    "v": ["Alle Werte: Aufnahmen der Crypto-AI-Plattform",
          "vom 28.09.2026 · keine Live-Daten. Keine Anlageberatung.",
          "Krypto-Assets sind mit hohen Risiken verbunden."],
}

# ---------------------------------------------------------------- colours (0..1 RGB)
SLAB = (14 / 255, 15 / 255, 19 / 255)
BORDER_A = 0.08
DIVIDER_A = 0.07
LABEL_C = (140 / 255, 140 / 255, 147 / 255)
TEXT_C = (245 / 255, 245 / 255, 247 / 255)
DESC_C = (161 / 255, 161 / 255, 166 / 255)
DISC_C = (154 / 255, 154 / 255, 160 / 255)

# ---------------------------------------------------------------- system object (Shot 1 / 7)
SYSTEM = {
    "h": dict(slab=(180, 190, 1740, 970), radius=28, cols=3, rows=2,
              order=["DATEN", "MARKT", "GELDFLUSS", "DERIVATE", "MAKRO", "RISIKO"],
              label_px=15, label_track=0.14, pad=28, content_top=64,
              wordmark=dict(cy=110, logo_px=34, text_px=20, track=0.16, gap=12)),
    "v": dict(slab=(130, 370, 950, 1450), radius=28, cols=2, rows=3,
              order=["MARKT", "DATEN", "GELDFLUSS", "DERIVATE", "MAKRO", "RISIKO"],
              label_px=17, label_track=0.14, pad=28, content_top=64,
              wordmark=dict(cy=322, logo_px=40, text_px=24, track=0.16, gap=14)),
}
CELL_CROPS = {  # label -> (frame, (x0, y0, x1, y1), scale)
    "h": {
        "MARKT": (70, (570, 330, 1350, 720), 0.595),
        "DATEN": (600, (130, 300, 910, 520), 0.595),
        "GELDFLUSS": (345, (80, 185, 560, 315), 0.967),
        "DERIVATE": (345, (926, 236, 1456, 626), 0.776),
        "MAKRO": (710, (130, 205, 560, 320), 0.97),   # x 1.03 camera push stays <= 1.0
        "RISIKO": ("R1", (150, 320, 580, 615), 1.0),
    },
    "v": {
        "MARKT": (70, (570, 335, 1350, 665), 0.454),
        "DATEN": (600, (130, 300, 910, 520), 0.454),
        "GELDFLUSS": (345, (80, 185, 560, 275), 0.738),
        "DERIVATE": (345, (926, 310, 1297, 626), 0.865),
        "MAKRO": (710, (130, 205, 560, 320), 0.823),
        "RISIKO": ("R1", (150, 320, 580, 615), 0.823),
    },
}
# Crop edges sit in UI gaps (fine-tuned by <= 6 px vs. the written spec so nothing is cut); the crop
# background is flattened to the slab colour, so a narrow 8 px edge is enough. MARKT keeps its blue
# colour field and fades it out softly instead of showing a rectangular edge.
CROP_FEATHER = 8
MARKT_FEATHER = (8, 8)     # plane px (x, y); the colour field itself fades via MARKT_FIELD
# MARKT overview: the hero's blue band fades out as a soft ellipse inside the cell; the text itself
# (glyph pixels inside these boxes) is always kept at full strength.
MARKT_FIELD = dict(c=(960, 505), r=(420, 225), inner=0.55)
MARKT_TEXT_BOXES = [(588, 343, 1337, 520), (863, 488, 1057, 526), (773, 560, 1147, 654), (863, 680, 1057, 716)]
FLATTEN_KERNEL = 61        # source px; removes UI glows, keeps text, bars and pills
TILT_DEG = 6.0
FOCAL = 2400.0

# ---------------------------------------------------------------- shot keyframes
HERO = {  # Shot 2: F70, zoom and source centre (canvas centre is the frame centre)
    "h": dict(z0=1.00, z1=1.20, c0=(960, 545), c1=(960, 525)),
    "v": dict(z0=1.00, z1=1.10, c0=(962, 540), c1=(962, 540)),   # centre of "84.673,00 $" (594-1330)
}
DATA = {  # Shot 3: masked card plate, frames 452..540 then 600
    "first": 452, "last": 540, "hold": 600,
    "h": dict(rect=(95, 155, 1825, 1011), radius=18, z0=0.94, z1=1.00, src_c=(960, 583), dst_c=(960, 540)),
    "v": dict(rect=(95, 155, 952, 1011), radius=18, z0=0.92, z1=0.95, src_c=(523.5, 583), dst_c=(540, 900)),
}
FLOW = {  # Shot 4: F345 full frame (header masked), 1080p limits (R3 not reproducible)
    "h": dict(a=dict(c=(720, 470), c_end=(760, 470), z=1.20, dst=(960, 540)),
              b=dict(c=(1360, 514), z=1.10, dst=(960, 540))),
    "v": dict(a=dict(c=(320, 247.5), c_end=(320, 247.5), z=1.40, dst=(540, 760)),
              b=dict(c=(1360, 514), z=0.88, dst=(540, 1000))),
    "focus_a": dict(c=(325, 250), r=(330, 150)),
    "focus_b": (880, 150, 1840, 850),   # options card incl. its caption line above
    "blur_sigma": 4.0,
    "t_hold_a": (12.0, 13.2), "t_move": (13.2, 15.0), "t_focus": (13.4, 14.4),
}
MACRO = {  # Shot 5: masked card plate, 1080p limits (R2 not reproducible)
    "first": 668, "last": 700, "hold": 710,
    "h": dict(rect=(96, 157, 1823, 842), radius=18, c0=(272, 255), z0=1.40, c1=(960, 500), z1=1.00,
              dst=(960, 540)),
    "v": dict(rect=(130, 185, 965, 842), radius=18, c0=(343, 260), z0=1.35, c1=(548, 514), z1=0.98,
              dst=(540, 960)),
    "t_move": (15.7, 18.4),
}
ENDCARD = {
    "h": dict(logo=(960, 390, 110), word=(505, 44, 0.12), desc=(560, 26), soon=(625, 18, 0.20),
              disc=([950, 980], 22), disc_t0=26.0),
    "v": dict(logo=(540, 720, 140), word=(860, 56, 0.12), desc=(930, 30), soon=(1005, 22, 0.20),
              disc=([1360, 1396, 1432], 26), disc_t0=26.0),
}

# Maximum source zoom (canvas px per 1080p source px) approved per shot and format.
ZOOM_LIMITS = {
    "h": {"s1_system": 1.0, "s2_hero": 1.20, "s3_data": 1.00, "s4_flow": 1.20, "s5_macro": 1.40},
    "v": {"s1_system": 1.0, "s2_hero": 1.10, "s3_data": 0.95, "s4_flow": 1.40, "s5_macro": 1.35},
}

# 9:16 critical safe area for text and numbers (x0, y0, x1, y1); 16:9 title-safe.
SAFE = {"h": (96, 54, 1824, 1026), "v": (130, 300, 950, 1460)}

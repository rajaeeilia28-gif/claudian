# Crypto AI – Teaser „Bald verfügbar.“

Werbespot im Apple-Stil für die Crypto-AI-Plattform, gebaut aus der Bildschirmaufnahme der laufenden Plattform
(`source/original.mp4`, 30 s). Bild, Typografie, Kamerafahrten, Musik und Sounddesign werden komplett per Code
erzeugt – es gibt keine Fremd-Assets außer der Schrift *Inter* (SIL Open Font License).

## Exporte (`export/`)

| Datei | Format | Einsatz |
| --- | --- | --- |
| `CryptoAI_Teaser_16x9.mp4` | 1920×1080, 60 fps, H.264 + AAC 320 kbit/s, −14 LUFS | YouTube, Website, X, LinkedIn |
| `CryptoAI_Teaser_9x16.mp4` | 1080×1920, 60 fps, H.264 + AAC 320 kbit/s, −14 LUFS | Instagram Reels/Story, TikTok, YouTube Shorts |
| `thumbnail_16x9.jpg` | 1920×1080 | YouTube-Thumbnail / Vorschaubild |
| `cover_9x16.jpg` | 1080×1920 | Reel-/TikTok-Cover |

## Dramaturgie (59 s, 120 BPM – jeder Schnitt liegt auf dem Beat)

| Zeit | Bild | Ton |
| --- | --- | --- |
| 0–3 s | „Der Markt ist laut.“ – flackernde, unscharfe Datenfragmente | anschwellendes Markt-Rauschen, harter Cut in die Stille |
| 3–6 s | „Hör genauer **hin.**“ | einzelne Glocke, Herzschlag, Riser |
| 6–10 s | Hero: 84.673,00 $ · **Bullisch**, Push-in mit Lichtkante | Drop: Kick, Bass, Flächen |
| 10–16 s | **241** Kennzahlen. · **99** Modelle. · **1** klares Urteil. (Count-up) | Arpeggio kommt dazu |
| 16–22 s | „Was die Daten sagen“ schwebt als 3D-Karte ins Bild, baut sich live auf, Fahrt auf 50 / 29 / 0 | volle Groove mit Claps |
| 22–24 s | „Folge dem **Geld.**“ | Break + Whoosh |
| 24–30 s | Geldfluss +179,7 → Rack-Focus auf die Optionen-Heatmap → Open-Interest-Karte in 3D | Groove |
| 30–32 s | „Sieh das **große Bild.**“ | Break |
| 32–36 s | Rückfahrt vom Regime „Reflation“ auf den ganzen Makro-Desk | Groove |
| 36–38,5 s | „Nicht nur Markt. **Auch Risiko.**“ | Break mit Spannung |
| 38,5–42 s | Risk Lab: Gauge 27 → Szenario 7.693 € → Kapitalverlauf | Groove |
| 42–46 s | Beat-Montage, 8 Detail-Schnitte im Halbsekundentakt | Snare-Roll, Riser, harter Stopp |
| 46–50 s | „Klarheit ist das neue **Alpha.**“ | fast Stille, Glocken, Rückwärts-Swell |
| 50–59 s | Logo-Reveal mit Glow und Lichtkante, „Crypto AI“, „Bald verfügbar.“, Risikohinweis | Impact, C-Dur-Auflösung, Ausklang |

Alle Zahlen im Spot stammen 1:1 aus der Aufnahme der Plattform (241 Kennzahlen, 99 ausgewertete Modelle usw.).
Der Endcard-Hinweis („Keine Anlageberatung. Krypto-Assets sind mit hohen Risiken verbunden.“) ist bewusst
drin – bei Krypto-Werbung in der EU sollte er nicht fehlen.

## Neu rendern

Voraussetzungen: Python 3.10+, `ffmpeg` im `PATH`, `pip install numpy opencv-python-headless pillow scipy`
(Pillow mit libraqm für Kerning).

```bash
cd pipeline
python3 prepare.py ../source/original.mp4      # Frames, freigestelltes Logo, Inter-Schrift
python3 music.py ../score.wav                   # Soundtrack
python3 render.py video h ../video_h.mp4        # 16:9 (4 Prozesse, ca. 8 min)
python3 render.py video v ../video_v.mp4        # 9:16
./finalize.sh ../video_h.mp4 ../score.wav ../export/CryptoAI_Teaser_16x9.mp4
./finalize.sh ../video_v.mp4 ../score.wav ../export/CryptoAI_Teaser_9x16.mp4
python3 poster.py ../export/thumbnail_16x9.jpg ../export/cover_9x16.jpg
python3 render.py stills h 8.0 50.9             # einzelne Standbilder zur Kontrolle -> stills/
```

Die Exporte sind Master in hoher Qualität (CRF 15). Für Messenger/Mail mit Größenlimit reicht eine
2-Pass-Version mit ~3,5 Mbit/s (≈ 28 MB, optisch praktisch identisch):

```bash
ffmpeg -i export/CryptoAI_Teaser_16x9.mp4 -c:v libx264 -preset slow -b:v 3500k -pass 1 -an -f mp4 /dev/null
ffmpeg -i export/CryptoAI_Teaser_16x9.mp4 -c:v libx264 -preset slow -b:v 3500k -pass 2 \
  -c:a aac -b:a 256k -movflags +faststart CryptoAI_Teaser_16x9_share.mp4
```

- Texte ändern: `pipeline/spot.py`, Methode `_build_text`.
- Timing ändern: Konstanten `T_*` oben in `spot.py` und Arrangement in `music.py` (beide auf 120 BPM ausgerichtet).
- Kamerafahrten: die jeweilige Shot-Methode in `spot.py` (`hero`, `data`, `flow`, `oi`, `macro`, `risk`, `montage`).

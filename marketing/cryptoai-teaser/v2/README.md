# Crypto AI – Produktfilm V2 (30 s)

Umsetzung der freigegebenen V2-Spezifikation. Quelle sind ausschließlich die Aufnahmen der Plattform vom
28.09.2026 (`../source/original.mp4`). Das Main-Projekt (`cryptoai-pro`) wird nicht verändert.

**Status:** Alles, was nicht von R1 abhängt, ist gebaut. Shot 6 (Risk Lab), die RISIKO-Kachel und die
Rückfahrt 23,0–24,4 s warten auf die 4K-Aufnahme R1. Ein finaler Render ist ohne R1 technisch gesperrt.

## Aufbau

| Datei | Inhalt |
| --- | --- |
| `pipeline/v2_spec.py` | Spezifikation als Daten: Timing, Crops, Keyframes, Layout, Datenregeln, Zoomgrenzen, Safe Zones |
| `pipeline/v2_spot.py` | Renderer (16:9 und 9:16), protokolliert pro Frame genutzte Quellbilder, Ausschnitte und Zoom |
| `pipeline/v2_render.py` | `stills`, `preview` (mit R1-Platzhaltern) und `final` (nur mit R1) |
| `pipeline/v2_music.py` | Score und Sounddesign, 48 kHz / 24 Bit, Stems, lineare Normalisierung auf −14 LUFS |
| `qa/loudness.py` | BS.1770-4 / EBU R128: integrierte Lautheit, Momentary, LRA, True Peak |
| `qa/audit_audio.py` | Audio-Abnahme nach Spezifikation Abschnitt 8 |
| `qa/audit_manifest.py` | Daten- und Zoom-Audit aus dem Nutzungsprotokoll |
| `qa/audit_video.py` | Frame-Audit: Schnitte, Sprünge, Schwarzbilder, Blitze, Safe Zones, Randanschnitte |

Die V2-Pipeline nutzt die Hilfsfunktionen und vorbereiteten Assets aus `../pipeline` (Frames, Schrift,
freigestelltes Logo via `../pipeline/prepare.py`).

## Befehle

```bash
cd pipeline
python3 v2_music.py                                   # -> ../out/audio/score_v2.wav + stems
python3 ../qa/audit_audio.py ../out/audio/score_v2.wav
python3 v2_render.py stills h 1.8 5.5 16.3            # Vorschau-Standbilder
python3 v2_render.py preview h 0 19 ../out/preview_h_0-19.mp4
python3 ../qa/audit_manifest.py ../out/preview_h_0-19.mp4.uses.json
python3 ../qa/audit_video.py ../out/preview_h_0-19.mp4 h 0 --overlays ../out/overlays
```

## Umgesetzte Festlegungen (Auszug)

- Systemobjekt: eine Platte (Radius 28, Haarlinien statt Einzelkarten), einheitliche Label-Geometrie.
  Die Hintergründe der Kachel-Crops werden auf die Plattenfarbe geglättet; Schrift, Balken und Pills
  bleiben unverändert. MARKT behält als einzige Kachel sein blaues Farbfeld (elliptisch auslaufend).
- Eintauchen 2,8–4,0 s als eine Kamerakurve bis in Shot 2 (Zoom- und Schwenkgeschwindigkeit stetig),
  Bewegungsunschärfe mit 6 Unterbildern.
- R2/R3 waren nicht reproduzierbar, daher gelten die freigegebenen 1080p-Grenzen
  (16:9 Makro-Start 1,40×, 9:16 Makro-Start 1,35×, 9:16 Geldfluss 1,40×, Optionen 0,88×).
- App-Kopfzeilen in Bild 70 und 345 mit Seitenschwarz maskiert (Umgebung homogen ≤ 5/255).

## R1 einbinden (sobald vorhanden)

1. Aufnahme nach `assets/r1/` legen (3840×2160, 60 fps, Risk Lab „Standard“, „Maßvolles Risiko“).
2. Werte prüfen: 27 · RUHIG · 7.693 € · −23,1 % · −2.307 € · Annahmen-Zeile.
3. Shot 6, RISIKO-Kachel und Rückfahrt 23,0–24,4 s implementieren, dann `final` rendern, muxen
   (lineare Normalisierung, AAC 320 kbit/s) und alle Audits auf den finalen Dateien wiederholen.

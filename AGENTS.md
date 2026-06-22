# AGENTS.md – Digit Recognition Drawing Tool

## Projektübersicht

Erstelle ein Python-Desktop-Tool, bei dem der Benutzer mit der Maus eine Ziffer (0–9) in ein Zeichenfeld malt. Ein trainiertes neuronales Netz (CNN auf MNIST-Basis) analysiert die Zeichnung und gibt eine Wahrscheinlichkeitsverteilung für alle 10 Ziffern aus – oder erkennt, dass gar keine gültige Ziffer gemalt wurde:

```
✔ Erkannt: 6 (78.3%)
  6: ████████████████████ 78.3%
  0: ████                 10.1%
  8: ███                   8.2%
  ...

✘ Keine Ziffer erkannt (?)
  ?: ████████████████████ 81.2%
  8: ███                   7.1%
  0: ██                    5.3%
  ...
```

Das Modell kennt **11 Klassen**: die Ziffern 0–9 sowie die Klasse `?` (= „Unbekannt / keine Ziffer").

---

## Technologie-Stack

| Komponente       | Technologie                          |
|-----------------|--------------------------------------|
| GUI             | `tkinter` (Standard-Python-Bibliothek) |
| ML-Framework    | `tensorflow` + `keras` ODER `torch` + `torchvision` |
| Datensatz       | MNIST (Ziffern) + EMNIST Letters (Nicht-Ziffern) + generierte Formen |
| Bildverarbeitung| `Pillow` (PIL)                       |
| Numerik         | `numpy`                              |

**Empfohlen: TensorFlow/Keras** (einfacheres MNIST-Setup), alternativ PyTorch.

---

## Projektstruktur

```
digit-recognizer/
├── AGENTS.md                    # Diese Datei
├── README.md                    # Projektdokumentation
├── requirements.txt             # Python-Abhängigkeiten
├── train.py                     # Modell trainieren und speichern
├── app.py                       # Haupt-GUI-Anwendung
├── model/
│   └── digit_model.h5           # Gespeichertes trainiertes Modell (wird generiert)
└── utils/
    ├── preprocess.py            # Bildvorverarbeitung (Canvas → Modell-Input)
    └── generate_negatives.py    # Erzeugt synthetische Nicht-Ziffern-Bilder (Klasse 10)
```

---

## Schritt-für-Schritt-Implementierung

### 1. `requirements.txt`

```
tensorflow>=2.12
tensorflow-datasets>=4.9    # für EMNIST Letters
scikit-learn>=1.2           # für compute_class_weight
Pillow>=9.0
numpy>=1.23
```

---

### 2. `utils/generate_negatives.py` – Negative Samples erzeugen

**Aufgabe:** Synthetische Bilder erzeugen, die **keine** Ziffern sind. Diese werden als Klasse `10` (= „Unbekannt") in das Training eingespeist.

**Quellen für Negative Samples (alle kombinieren):**

**A) EMNIST Letters** – Handgeschriebene Buchstaben (a–z, A–Z):
```python
# tensorflow_datasets oder separater Download
import tensorflow_datasets as tfds
emnist = tfds.load('emnist/letters', split='train')
# Buchstaben-Bilder direkt als Klasse 10 verwenden
# Menge: ~10.000 Bilder aus EMNIST nehmen
```

**B) Synthetisch generierte Formen** – Mit PIL/numpy erzeugen:
- Zufällige Linien (1–5 Linien, verschiedene Winkel)
- Zufällige Rechtecke / Dreiecke
- Zufällige Kurven und Bögen
- Zufälliges Rauschen (Gauß'sches Rauschen)
- Einfache Formen wie Häuser, Pfeile, Kreuze
- Leere oder fast leere Bilder (nur wenige Pixel gesetzt)

```python
def generate_synthetic_negatives(n=10000) -> np.ndarray:
    """Erzeugt n synthetische 28x28 Nicht-Ziffern-Bilder."""
    images = []
    for _ in range(n):
        img = Image.new('L', (28, 28), 0)
        draw = ImageDraw.Draw(img)
        shape_type = random.choice(['lines', 'rect', 'triangle', 'noise', 'arc'])
        if shape_type == 'lines':
            for _ in range(random.randint(1, 4)):
                x0, y0 = random.randint(0,27), random.randint(0,27)
                x1, y1 = random.randint(0,27), random.randint(0,27)
                draw.line([(x0,y0),(x1,y1)], fill=255, width=random.randint(1,3))
        elif shape_type == 'rect':
            x0,y0,x1,y1 = sorted(random.sample(range(28),2)+random.sample(range(28),2))
            draw.rectangle([x0,y0,x1,y1], outline=255)
        elif shape_type == 'noise':
            arr = np.random.randint(0, 2, (28,28), dtype=np.uint8) * 255
            img = Image.fromarray(arr)
        # ... weitere Typen analog
        images.append(np.array(img) / 255.0)
    return np.array(images)
```

**Gesamtmenge Klasse 10:** ca. 15.000–20.000 Bilder (EMNIST + synthetisch), um das Ungleichgewicht zu MNIST (60.000 Bilder) zu reduzieren. Klassen-Gewichte beim Training verwenden (siehe `train.py`).

---

### 3. `train.py` – Modell mit 11 Klassen trainieren

**Aufgabe:** MNIST (Klassen 0–9) + Negative Samples (Klasse 10) laden, CNN trainieren, als `model/digit_model.h5` speichern.

**Anforderungen:**

**Daten zusammenstellen:**
```python
# MNIST laden
(x_mnist, y_mnist), (x_test, y_test) = keras.datasets.mnist.load_data()

# Negative Samples laden/generieren
x_neg_emnist = load_emnist_letters(n=10000)   # aus generate_negatives.py
x_neg_synth  = generate_synthetic_negatives(n=10000)
x_neg = np.concatenate([x_neg_emnist, x_neg_synth])
y_neg = np.full(len(x_neg), fill_value=10, dtype=np.int32)  # Klasse 10

# Negatives für Testset
x_neg_test = generate_synthetic_negatives(n=2000)
y_neg_test  = np.full(2000, fill_value=10, dtype=np.int32)

# Alles zusammenführen
x_train = np.concatenate([x_mnist, x_neg])
y_train = np.concatenate([y_mnist, y_neg])
x_test_all = np.concatenate([x_test, x_neg_test])
y_test_all  = np.concatenate([y_test, y_neg_test])

# Shuffle
idx = np.random.permutation(len(x_train))
x_train, y_train = x_train[idx], y_train[idx]
```

**Normalisierung und Reshape:**
```python
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test_all = x_test_all.reshape(-1, 28, 28, 1).astype('float32') / 255.0
```

**Modell-Architektur (11 Ausgabe-Neuronen):**
```
Conv2D(32, 3x3, relu) → MaxPooling2D →
Conv2D(64, 3x3, relu) → MaxPooling2D →
Conv2D(64, 3x3, relu) →
Flatten → Dense(128, relu) → Dropout(0.4) →
Dense(11, softmax)   ← 11 statt 10!
```

**Klassen-Gewichte** (Negative Samples sind unterrepräsentiert):
```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.arange(11), y=y_train)
class_weight_dict = dict(enumerate(weights))
```

**Training:**
- Kompilieren mit `adam`, Loss: `sparse_categorical_crossentropy`, Metrik: `accuracy`
- Mindestens 10 Epochen, Batch-Size 64
- `class_weight=class_weight_dict` an `model.fit()` übergeben
- **Data Augmentation:**
  - Rotation: ±10°
  - Verschiebung: ±10%
  - Scherung: ±5°
  - Kein horizontales Spiegeln

**Ausgabe beim Ausführen:**
```
Lade MNIST-Daten...         60000 Ziffern-Bilder
Lade EMNIST-Buchstaben...   10000 Buchstaben-Bilder
Generiere synthetische Formen... 10000 Bilder
Gesamt-Trainingsset: 80000 Bilder (Klassen 0–10)
Trainiere Modell (10 Epochen)...
Epoch 1/10 ...
Modell gespeichert: model/digit_model.h5
Validierungs-Accuracy (gesamt): 98.4%
Erkennungsrate Klasse 10 (Unbekannt): 91.2%
```

---

### 4. `utils/preprocess.py` – Bildvorverarbeitung

**Aufgabe:** Das vom Benutzer gemalte Bild (tkinter Canvas) in ein 28×28-Numpy-Array umwandeln, das dem MNIST-Format entspricht.

**Anforderungen:**
- Funktion `canvas_to_model_input(pil_image) -> np.ndarray`
  1. Bild in Graustufen umwandeln (`L`-Modus)
  2. **Bounding-Box zuschneiden:** Bereich mit tatsächlichem Inhalt erkennen (`ImageOps.getbbox` oder `numpy.where`) und mit ~20% Rand ausschneiden – danach quadratisch auffüllen (`ImageOps.pad`). Das ist entscheidend für gute Erkennungsrate.
  3. Auf 28×28 Pixel skalieren (`LANCZOS`-Resampling)
  4. Pixelwerte normalisieren: `pixel / 255.0`
  5. **Invertieren:** MNIST hat weiße Ziffern auf schwarzem Hintergrund – Canvas hat schwarze Striche auf weißem Hintergrund → `1.0 - normalized`
  6. Shape: `(1, 28, 28, 1)` zurückgeben (Batch-Dimension für Modell)
- Falls das Bild komplett leer ist (alle Pixel weiß): `None` zurückgeben

---

### 5. `app.py` – Haupt-GUI

**Aufgabe:** Vollständige tkinter-Anwendung mit Zeichenfeld und Ergebnisanzeige.

#### 5.1 Fenster-Layout

```
┌─────────────────────────────────────────┐
│  🔢 Digit Recognizer                    │
├─────────────────────────────────────────┤
│  ┌───────────────────┐  Ergebnis:       │
│  │                   │                  │
│  │   ZEICHENFELD     │  ✔ 6 (78.3%)    │
│  │   280×280 px      │                  │
│  │   (schwarz/weiß)  │  Wahrsch.:       │
│  │                   │  6: ██████ 78.3% │
│  └───────────────────┘  0: ██     10.1% │
│                          8: █       8.2% │
│  [Erkennen]  [Löschen]   ...            │
└─────────────────────────────────────────┘
```

#### 5.2 Zeichenfeld (Canvas)

- `tk.Canvas`, Größe 280×280 Pixel, Hintergrund weiß
- Maus-Events binden:
  - `<B1-Motion>`: Zeichnen mit `create_oval` (Pinselbreite ~18px, Farbe schwarz)
  - `<ButtonPress-1>`: Neue Linie beginnen (vorherige Position zurücksetzen)
- Gleichzeitig auf ein `PIL.Image`-Objekt (280×280, weiß) zeichnen mit `ImageDraw.Draw`, damit das Bild für das Modell verfügbar ist

#### 5.3 Buttons

- **„Erkennen" (Primärbutton):** Ruft Vorhersage auf und zeigt Ergebnis an. Ist deaktiviert, wenn Canvas leer.
- **„Löschen":** Löscht Canvas und PIL-Image, setzt Ergebnisanzeige zurück.

#### 5.4 Ergebnisanzeige

- Label oben: `✔ Erkannt: 6` in großer, fetter Schrift (z. B. 24pt), Farbe grün bei Konfidenz > 70%, orange bei > 40%, rot sonst.
- **Sonderfall Klasse 10:** Label zeigt `✘ Keine Ziffer erkannt` in grauer Farbe. Keine Ziffer wird hervorgehoben.
- Konfidenz-Prozent daneben: `(78.3%)`
- Für alle **11 Klassen** (0–9 und `?`), sortiert nach Wahrscheinlichkeit absteigend:
  - Ziffer-Label: `6:` bzw. `?:` für Klasse 10
  - Balken: Proportionaler Balken aus Unicode-Blöcken (`█`) ODER `tk.Canvas`-Rechteck
  - Prozentzahl: `78.3%`
  - Die Top-1-Klasse fett/farbig hervorheben

#### 5.5 Modell laden

- Beim Start: Modell aus `model/digit_model.h5` laden
- Falls Datei nicht existiert: Fehlermeldung im Fenster anzeigen: `"Modell nicht gefunden. Bitte zuerst train.py ausführen."` und Erkennen-Button deaktivieren.

#### 5.6 Vorhersage-Logik

```python
# Klassen-Mapping: 0–9 = Ziffern, 10 = Unbekannt
CLASS_LABELS = {i: str(i) for i in range(10)}
CLASS_LABELS[10] = "?"

def predict():
    # 1. PIL-Image aus Canvas holen
    # 2. preprocess.canvas_to_model_input() aufrufen
    # 3. model.predict() → Array mit 11 Wahrscheinlichkeiten
    # 4. argmax für beste Vorhersage
    # 5. Wenn argmax == 10: "Keine Ziffer erkannt" anzeigen
    # 6. Sonst: Ziffer mit Konfidenz anzeigen
    # 7. Alle 11 Klassen sortiert in Balkendiagramm anzeigen
```

---

### 6. `README.md`

**Muss enthalten:**
- Kurzbeschreibung des Projekts
- Screenshot-Platzhalter (`![Screenshot](screenshot.png)`)
- Installations-Anleitung:
  ```bash
  pip install -r requirements.txt
  python train.py      # Einmalig: Modell trainieren (~5 Min.)
  python app.py        # App starten
  ```
- Kurze Erklärung: Was ist MNIST? Was ist ein CNN? Was ist die „Unbekannt"-Klasse?
- Hinweis: Modell erreicht ~99% Testgenauigkeit auf MNIST-Ziffern, ~91% auf Nicht-Ziffern

---

## Bekannte Probleme & Fixes

### Problem: 1 wird als 7 erkannt (besonders bei europäischer Schreibweise)

**Ursache 1 – Trainingsdaten:** MNIST stammt aus den USA. Europäische Einsen haben oft einen langen Aufstrich (`⟋`), den das Modell als Sieben-Serife interpretiert.

**Ursache 2 – Fehlende Bounding-Box-Zentrierung:** Wenn die Zeichnung nicht korrekt zugeschnitten und zentriert wird, bevor sie auf 28×28 skaliert wird, verschlechtert sich die Erkennungsrate drastisch – besonders bei schmalen Ziffern wie 1.

**Fix A – Preprocessing (höchste Priorität):**
In `canvas_to_model_input()` **muss** die Bounding-Box korrekt implementiert sein:
```python
# Invertiertes Bild (weiße Ziffer auf schwarz)
arr = np.array(gray)
rows = np.any(arr > 10, axis=1)
cols = np.any(arr > 10, axis=0)
rmin, rmax = np.where(rows)[0][[0, -1]]
cmin, cmax = np.where(cols)[0][[0, -1]]

# Mit Rand versehen
pad = 4
rmin, rmax = max(0, rmin - pad), min(27, rmax + pad)
cmin, cmax = max(0, cmin - pad), min(27, cmax + pad)

# Ausschneiden und quadratisch auffüllen
cropped = image.crop((cmin_orig, rmin_orig, cmax_orig, rmax_orig))
cropped = ImageOps.pad(cropped, (20, 20), color=0)  # quadratisch, zentriert
final = cropped.resize((28, 28), Image.LANCZOS)
```

**Fix B – Data Augmentation beim Training:**
Leichte Scherung (shear) im `ImageDataGenerator` trainiert das Modell auf schräge Aufstriche und reduziert 1/7-Verwechslungen signifikant.

**Fix C – Längeres Training:**
Mindestens 10 statt 5 Epochen verwenden. Mit Augmentation braucht das Modell mehr Durchläufe, um zu generalisieren.

**Test nach dem Fix:**
- Eine klassische europäische 1 mit Aufstrich (`⟋`) zeichnen → sollte als 1 erkannt werden
- Eine 7 mit Querstrich zeichnen → sollte sicher als 7 erkannt werden

---

## Qualitätsanforderungen

### Code-Stil
- PEP 8 einhalten
- Alle Funktionen mit Docstrings dokumentieren (Deutsch oder Englisch)
- Keine hartcodierten Pfade – immer `os.path.join` und `pathlib` verwenden
- Konstanten am Dateianfang definieren (z. B. `CANVAS_SIZE = 280`, `MODEL_PATH = "model/digit_model.h5"`)

### Fehlerbehandlung
- `try/except` beim Modell laden
- Leeres Canvas abfangen (keine Vorhersage bei weißem Bild)
- Benutzerfreundliche Fehlermeldungen (in der GUI, nicht nur in der Konsole)

### Performance
- Modell einmalig beim Start laden, nicht bei jeder Vorhersage
- Vorhersage < 100ms (bei CPU realistisch für 28×28 Input)

---

## Optionale Erweiterungen (Nice-to-Have)

Falls Zeit vorhanden, in dieser Reihenfolge umsetzen:

1. **Live-Vorhersage:** Erkennung automatisch ~500ms nach dem letzten Strich (via `after()`-Timer), ohne Button-Klick
2. **Pinselgröße-Slider:** `tk.Scale` Widget zum Anpassen der Strichbreite
3. **Konfidenz-Visualisierung:** Balkendiagramm mit `tk.Canvas`-Rechtecken statt Unicode (sauberer)
4. **Modell-Info:** Kleine Statusleiste unten mit Modell-Accuracy und Trainings-Info
5. **Dark Mode:** Schwarzes Canvas, weißer Strich (entspricht eher MNIST-Format direkt)

---

## Reihenfolge der Implementierung

1. `requirements.txt` erstellen
2. `utils/generate_negatives.py` implementieren und testen (EMNIST laden, synthetische Formen generieren)
3. `train.py` implementieren und testen (`python train.py` → 11-Klassen-Modell wird gespeichert)
4. `utils/preprocess.py` implementieren
5. `app.py` Grundgerüst (Fenster, Canvas, Buttons – ohne Modell)
6. `app.py` Modell-Integration (laden, predict mit 11 Klassen, Ergebnis anzeigen inkl. „Keine Ziffer"-Fall)
7. `README.md` schreiben
8. Optionale Features nach Bedarf

---

## Test-Checkliste

Vor dem GitHub-Push sicherstellen:

- [x] `python train.py` läuft ohne Fehler durch
- [x] `model/digit_model.h5` wird erstellt (11 Ausgabe-Neuronen)
- [ ] `python app.py` öffnet das Fenster
- [ ] Zeichnen auf Canvas funktioniert
- [ ] „Löschen"-Button setzt Canvas zurück
- [ ] „Erkennen"-Button gibt Wahrscheinlichkeiten für alle 11 Klassen aus
- [ ] Ergebnis-Label zeigt erkannte Ziffer mit Konfidenz an
- [ ] Ein gemaltes Haus / Gesicht / Pfeil → `✘ Keine Ziffer erkannt` erscheint
- [ ] Zufälliges Gekritzel → `✘ Keine Ziffer erkannt` erscheint
- [ ] Klare Ziffer 5 → wird als Ziffer erkannt, **nicht** als „Unbekannt"
- [ ] Fehlermeldung erscheint, wenn Modell-Datei fehlt
- [ ] Kein Absturz bei leerem Canvas
- [x] README enthält alle Installations-Schritte
- [ ] Europäische 1 (mit Aufstrich) wird als 1 erkannt, nicht als 7
- [ ] 7 mit Querstrich wird sicher als 7 erkannt

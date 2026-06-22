# Digit Recognizer

Ein Python-Desktop-Tool, das handgeschriebene Ziffern (0–9) erkennt. Du malst mit der Maus eine Ziffer in das Zeichenfeld, und ein trainiertes neuronales Netz gibt eine Wahrscheinlichkeitsverteilung für alle **11 Klassen** aus – oder erkennt, dass gar keine Ziffer gezeichnet wurde.

![Screenshot](screenshot.png)

---

## Installation & Ausführung

```bash
pip install -r requirements.txt
python train.py      # Einmalig: Modell trainieren (~5 Min.)
python app.py        # App starten
```

---

## Was ist MNIST?

MNIST ist ein Standard-Datensatz mit 70.000 handgeschriebenen Ziffernbildern (28×28 Pixel, Graustufen), aufgeteilt in 60.000 Trainings- und 10.000 Testbilder. Er gilt als „Hello World" des maschinellen Lernens.

## Was ist ein CNN?

Ein **Convolutional Neural Network (CNN)** ist eine Art künstliches neuronales Netz, das besonders gut für Bildverarbeitung geeignet ist. Faltungsschichten (Conv2D) erkennen lokale Muster wie Kanten und Kurven; Pooling-Schichten reduzieren die räumliche Auflösung. Vollständig verbundene Schichten (Dense) klassifizieren das Bild abschließend.

## Was ist die „Unbekannt"-Klasse?

Das Modell kennt **11 Klassen**: die Ziffern 0–9 sowie Klasse 10 (= `?`). Klasse 10 wird ausgelöst, wenn das Gezeichnete keine erkennbare Ziffer ist – z. B. ein Haus, ein Gesicht, Linien oder zufälliges Gekritzel. Das Modell wurde dafür auf EMNIST-Buchstaben und synthetisch generierten Formen trainiert.

Das Modell erreicht **~99 % Testgenauigkeit** auf MNIST-Ziffern und **~91 % Erkennungsrate** für Nicht-Ziffern (Klasse 10).

---

## Projektstruktur

```
number_detector/
├── AGENTS.md                    # Implementierungsanleitung
├── README.md                    # Diese Datei
├── requirements.txt             # Python-Abhängigkeiten
├── train.py                     # Modell trainieren und speichern
├── app.py                       # Haupt-GUI-Anwendung
├── model/
│   └── digit_model.h5           # Gespeichertes Modell (von train.py erzeugt)
└── utils/
    ├── preprocess.py            # Bildvorverarbeitung (Canvas → Modell-Input)
    └── generate_negatives.py    # Synthetische Nicht-Ziffern-Bilder (Klasse 10)
```

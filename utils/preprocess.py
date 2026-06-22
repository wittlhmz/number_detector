"""Bildvorverarbeitung: Canvas-Bild → Modell-Input im MNIST-Format (11 Klassen)."""

import numpy as np
from PIL import Image, ImageOps


def canvas_to_model_input(pil_image: Image.Image):
    """Konvertiert ein PIL-Bild (Canvas-Inhalt) in einen Modell-Input.

    Pipeline:
      1. Graustufen
      2. Leer-Prüfung
      3. Invertieren (schwarze Striche auf weiß → weiße Striche auf schwarz)
      4. BBox ermitteln + Rand + quadratisches Pad
      5. 28×28 skalieren
      6. Normalisieren → Shape (1, 28, 28, 1)

    Args:
        pil_image: PIL-Bild mit schwarzen Strichen auf weißem Hintergrund.

    Returns:
        numpy-Array der Form (1, 28, 28, 1) in [0, 1], oder None bei leerem Bild.
    """
    gray = pil_image.convert("L")
    arr = np.array(gray, dtype="float32")

    if arr.min() >= 245:
        return None

    # Invertieren: Striche werden hell (wie MNIST: weiß auf schwarz)
    inv = 255.0 - arr

    # BBox der gezeichneten Region bestimmen
    rows = np.any(inv > 10, axis=1)
    cols = np.any(inv > 10, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

    # Rand proportional zur Zeichengröße hinzufügen
    h, w = arr.shape
    pad = max(4, int(max(rmax - rmin, cmax - cmin) * 0.15))
    rmin = max(0, rmin - pad)
    rmax = min(h - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(w - 1, cmax + pad)

    # Auf Originalbild (weiß auf schwarz) zuschneiden und quadratisch auffüllen
    cropped = gray.crop((cmin, rmin, cmax + 1, rmax + 1))
    w_c, h_c = cropped.size
    max_dim = max(w_c, h_c)
    # Weißes Padding (255 = Hintergrundfarbe des Canvas)
    cropped = ImageOps.pad(cropped, (max_dim, max_dim), color=255)

    resized = cropped.resize((28, 28), Image.LANCZOS)

    normalized = np.array(resized, dtype="float32") / 255.0
    # Invertieren: Canvas (schwarz auf weiß) → MNIST-Format (weiß auf schwarz)
    inverted = 1.0 - normalized

    return inverted.reshape(1, 28, 28, 1)

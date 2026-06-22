"""Erzeugt negative Trainingsbeispiele (Klasse 10 = 'Unbekannt / keine Ziffer')."""

import random
import numpy as np
from PIL import Image, ImageDraw


def load_emnist_letters(n: int = 10000) -> np.ndarray:
    """Lädt n Buchstaben-Bilder aus EMNIST/Letters als normalisierte numpy-Arrays.

    Bilder haben weiße Striche auf schwarzem Hintergrund (wie MNIST).
    Fällt auf synthetische Negative zurück, wenn der Download fehlschlägt.
    """
    try:
        import tensorflow_datasets as tfds
        print(f"Lade EMNIST-Buchstaben ({n} Bilder)...")
        ds = tfds.load("emnist/letters", split="train", as_supervised=True)
        images = []
        for img, _ in ds.take(n):
            arr = img.numpy().squeeze().astype("float32") / 255.0
            images.append(arr)
        result = np.array(images)
        print(f"  {len(result)} Buchstaben-Bilder geladen.")
        return result
    except Exception as exc:
        print(f"EMNIST konnte nicht geladen werden ({exc}).")
        print(f"Generiere stattdessen {n} weitere synthetische Negative...")
        return generate_synthetic_negatives(n)


def generate_synthetic_negatives(n: int = 10000) -> np.ndarray:
    """Erzeugt n synthetische 28×28 Nicht-Ziffern-Bilder.

    Jedes Bild ist normalisiert auf [0, 1] mit weißen Strichen auf schwarzem
    Hintergrund (MNIST-Format).
    """
    print(f"Generiere synthetische Formen ({n} Bilder)...")
    shape_types = ["lines", "rect", "triangle", "noise", "arc", "house", "cross", "sparse"]
    images = []

    for _ in range(n):
        img = Image.new("L", (28, 28), 0)
        draw = ImageDraw.Draw(img)
        shape = random.choice(shape_types)

        if shape == "lines":
            for _ in range(random.randint(2, 5)):
                x0, y0 = random.randint(0, 27), random.randint(0, 27)
                x1, y1 = random.randint(0, 27), random.randint(0, 27)
                draw.line([(x0, y0), (x1, y1)], fill=255, width=random.randint(1, 3))

        elif shape == "rect":
            xs = sorted(random.sample(range(28), 2))
            ys = sorted(random.sample(range(28), 2))
            draw.rectangle([xs[0], ys[0], xs[1], ys[1]], outline=255, width=random.randint(1, 2))

        elif shape == "triangle":
            pts = [(random.randint(2, 25), random.randint(2, 25)) for _ in range(3)]
            draw.polygon(pts, outline=255)

        elif shape == "noise":
            n_pixels = random.randint(10, 80)
            arr = np.zeros((28, 28), dtype=np.uint8)
            for _ in range(n_pixels):
                arr[random.randint(0, 27), random.randint(0, 27)] = 255
            img = Image.fromarray(arr)

        elif shape == "arc":
            x0 = random.randint(0, 10)
            y0 = random.randint(0, 10)
            x1 = random.randint(18, 27)
            y1 = random.randint(18, 27)
            start = random.randint(0, 180)
            end = start + random.randint(90, 270)
            draw.arc([x0, y0, x1, y1], start=start, end=end, fill=255, width=2)

        elif shape == "house":
            draw.rectangle([6, 14, 22, 25], outline=255, width=1)
            draw.polygon([(3, 14), (14, 3), (25, 14)], outline=255)

        elif shape == "cross":
            draw.line([(14, 3), (14, 24)], fill=255, width=2)
            draw.line([(3, 14), (24, 14)], fill=255, width=2)

        elif shape == "sparse":
            for _ in range(random.randint(3, 15)):
                x = random.randint(0, 27)
                y = random.randint(0, 27)
                r = random.randint(0, 2)
                draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

        images.append(np.array(img, dtype="float32") / 255.0)

    return np.array(images)

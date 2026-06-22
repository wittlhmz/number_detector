"""Digit Recognizer – tkinter-Anwendung zur handschriftlichen Ziffernerkennung.

11 Klassen: 0–9 (Ziffern) und Klasse 10 (= "?" = keine gültige Ziffer).
"""

import os
import tkinter as tk
from PIL import Image, ImageDraw

from utils.preprocess import canvas_to_model_input

CANVAS_SIZE = 280
BRUSH_SIZE = 18
MODEL_PATH = os.path.join("model", "digit_model.h5")
BAR_MAX_BLOCKS = 20
NUM_CLASSES = 11
CLASS_LABELS = {i: str(i) for i in range(10)}
CLASS_LABELS[10] = "?"


def load_model():
    """Lädt das trainierte Keras-Modell.

    Returns:
        (model, None) bei Erfolg, (None, fehlermeldung) bei Fehler.
    """
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        out_classes = model.output_shape[-1]
        if out_classes != NUM_CLASSES:
            return None, (
                f"Altes Modell gefunden ({out_classes} Klassen).\n"
                "Bitte train.py erneut ausführen."
            )
        return model, None
    except FileNotFoundError:
        return None, "Modell nicht gefunden.\nBitte zuerst train.py ausführen."
    except Exception as exc:
        return None, f"Fehler beim Laden:\n{exc}"


class DigitRecognizerApp:
    """Hauptanwendung für die Ziffernerkennung."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Digit Recognizer")
        self.root.resizable(False, False)

        self.model, model_error = load_model()
        self._pil_image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self._pil_draw = ImageDraw.Draw(self._pil_image)
        self._canvas_empty = True
        self._after_id = None

        self._build_ui()

        if model_error:
            self._show_error(model_error)

    def _build_ui(self):
        """Erstellt alle GUI-Elemente."""
        main = tk.Frame(self.root, padx=12, pady=12)
        main.pack()

        left = tk.Frame(main)
        left.grid(row=0, column=0, padx=(0, 16))

        tk.Label(left, text="Zeichne eine Ziffer:", anchor="w").pack(fill="x")

        self.canvas = tk.Canvas(
            left,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            cursor="pencil",
            highlightthickness=1,
            highlightbackground="#aaa",
        )
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self._on_draw)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        btn_frame = tk.Frame(left)
        btn_frame.pack(pady=(8, 0))

        self.btn_predict = tk.Button(
            btn_frame,
            text="Erkennen",
            command=self._predict,
            state="disabled",
            width=12,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
        )
        self.btn_predict.grid(row=0, column=0, padx=4)

        tk.Button(
            btn_frame,
            text="Löschen",
            command=self._clear,
            width=12,
            bg="#f44336",
            fg="white",
            font=("Arial", 11),
        ).grid(row=0, column=1, padx=4)

        right = tk.Frame(main)
        right.grid(row=0, column=1, sticky="n")

        tk.Label(right, text="Ergebnis:", font=("Arial", 12, "bold")).pack(anchor="w")

        self.lbl_result = tk.Label(
            right,
            text="– noch keine Vorhersage –",
            font=("Arial", 18, "bold"),
            fg="#555",
            width=26,
            anchor="w",
            justify="left",
        )
        self.lbl_result.pack(anchor="w", pady=(4, 12))

        tk.Label(right, text="Wahrscheinlichkeiten:", font=("Arial", 10)).pack(anchor="w")

        self.bar_frame = tk.Frame(right)
        self.bar_frame.pack(anchor="w", fill="x")

        # 11 Zeilen für sortierte Ausgabe (Reihenfolge ändert sich pro Vorhersage)
        self._bar_rows: list[tuple] = []
        for _ in range(NUM_CLASSES):
            row = tk.Frame(self.bar_frame)
            row.pack(fill="x", pady=1)
            lbl_cls = tk.Label(row, text="", width=3, anchor="e", font=("Courier", 10))
            lbl_cls.pack(side="left")
            lbl_bar = tk.Label(row, text="", anchor="w", font=("Courier", 10), fg="#1565C0")
            lbl_bar.pack(side="left")
            lbl_pct = tk.Label(row, text="", width=7, anchor="e", font=("Courier", 10))
            lbl_pct.pack(side="left")
            self._bar_rows.append((lbl_cls, lbl_bar, lbl_pct))

    def _on_press(self, event):
        """Setzt Startposition beim Maustastenklick."""
        pass

    def _on_draw(self, event):
        """Zeichnet Strich auf Canvas und PIL-Bild."""
        x, y = event.x, event.y
        r = BRUSH_SIZE // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self._pil_draw.ellipse([x - r, y - r, x + r, y + r], fill="black")

        if self._canvas_empty:
            self._canvas_empty = False
            self.btn_predict.config(state="normal")

        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(500, self._predict)

    def _on_release(self, event):
        """Setzt Pinselposition zurück (Platzhalter für künftige Erweiterungen)."""
        pass

    def _clear(self):
        """Löscht Canvas und setzt Zustand zurück."""
        self.canvas.delete("all")
        self._pil_image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self._pil_draw = ImageDraw.Draw(self._pil_image)
        self._canvas_empty = True
        self.btn_predict.config(state="disabled")
        self.lbl_result.config(
            text="– noch keine Vorhersage –", fg="#555", font=("Arial", 18, "bold")
        )
        for lbl_cls, lbl_bar, lbl_pct in self._bar_rows:
            lbl_cls.config(text="", font=("Courier", 10))
            lbl_bar.config(text="")
            lbl_pct.config(text="", font=("Courier", 10))

        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _predict(self):
        """Führt die Vorhersage durch und aktualisiert die GUI."""
        if self.model is None or self._canvas_empty:
            return

        model_input = canvas_to_model_input(self._pil_image)
        if model_input is None:
            return

        probs = self.model.predict(model_input, verbose=0)[0]
        best_cls = int(probs.argmax())
        best_prob = float(probs[best_cls])

        if best_cls == 10:
            result_text = f"✘ Keine Ziffer erkannt  ({best_prob * 100:.1f}%)"
            result_color = "#555"
        else:
            if best_prob >= 0.70:
                result_color = "#2E7D32"
            elif best_prob >= 0.40:
                result_color = "#E65100"
            else:
                result_color = "#B71C1C"
            result_text = f"✔ Erkannt: {best_cls}  ({best_prob * 100:.1f}%)"

        self.lbl_result.config(text=result_text, fg=result_color)

        sorted_classes = sorted(range(NUM_CLASSES), key=lambda c: probs[c], reverse=True)
        for row_idx, cls in enumerate(sorted_classes):
            lbl_cls, lbl_bar, lbl_pct = self._bar_rows[row_idx]
            p = float(probs[cls])
            label = f"{CLASS_LABELS[cls]}:"
            blocks = int(round(p * BAR_MAX_BLOCKS))
            bar_str = "█" * blocks
            pct_str = f"{p * 100:5.1f}%"

            if cls == best_cls:
                bar_color = result_color if best_cls != 10 else "#555"
                lbl_cls.config(text=label, font=("Courier", 10, "bold"))
                lbl_bar.config(text=bar_str, fg=bar_color)
                lbl_pct.config(text=pct_str, font=("Courier", 10, "bold"))
            else:
                lbl_cls.config(text=label, font=("Courier", 10))
                lbl_bar.config(text=bar_str, fg="#1565C0")
                lbl_pct.config(text=pct_str, font=("Courier", 10))

    def _show_error(self, message: str):
        """Zeigt Fehlermeldung in der GUI an."""
        self.lbl_result.config(
            text=message,
            fg="#B71C1C",
            font=("Arial", 11),
        )
        self.btn_predict.config(state="disabled")


def main():
    root = tk.Tk()
    DigitRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

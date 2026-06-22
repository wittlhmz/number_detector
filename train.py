"""Trainiert ein 11-Klassen-CNN (Ziffern 0–9 + Klasse 10 = 'Unbekannt') auf
MNIST + EMNIST Letters + synthetischen Formen und speichert das Modell.
"""

import os
import numpy as np
import tensorflow as tf
import keras
from sklearn.utils.class_weight import compute_class_weight

from utils.generate_negatives import load_emnist_letters, generate_synthetic_negatives

MODEL_PATH = os.path.join("model", "digit_model.h5")
EPOCHS = 10
BATCH_SIZE = 64
N_EMNIST = 10000
N_SYNTH = 10000
N_NEG_TEST = 2000


def load_data():
    """Kombiniert MNIST-Ziffern und Negative Samples zu einem 11-Klassen-Datensatz."""
    print("Lade MNIST-Daten...")
    (x_mnist, y_mnist), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_mnist = x_mnist.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    print(f"  {len(x_mnist)} Ziffern-Bilder (Train), {len(x_test)} (Test)")

    x_neg_emnist = load_emnist_letters(n=N_EMNIST)
    x_neg_synth = generate_synthetic_negatives(n=N_SYNTH)
    x_neg = np.concatenate([x_neg_emnist, x_neg_synth])
    y_neg = np.full(len(x_neg), fill_value=10, dtype=np.int32)
    print(f"  {len(x_neg)} Nicht-Ziffern-Bilder (Klasse 10)")

    print("Generiere Testset für Klasse 10...")
    x_neg_test = generate_synthetic_negatives(n=N_NEG_TEST)
    y_neg_test = np.full(N_NEG_TEST, fill_value=10, dtype=np.int32)

    x_train = np.concatenate([x_mnist, x_neg])
    y_train = np.concatenate([y_mnist.astype(np.int32), y_neg])
    x_test_all = np.concatenate([x_test, x_neg_test])
    y_test_all = np.concatenate([y_test.astype(np.int32), y_neg_test])

    idx = np.random.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test_all = x_test_all.reshape(-1, 28, 28, 1)

    print(f"Gesamt-Trainingsset: {len(x_train)} Bilder (Klassen 0–10)")
    return (x_train, y_train), (x_test_all, y_test_all)


def build_model() -> keras.Model:
    """Erstellt das 11-Klassen-CNN mit integrierter Data-Augmentation."""
    inputs = keras.Input(shape=(28, 28, 1))

    # Augmentation – nur aktiv während model.fit(), nicht bei predict()
    x = keras.layers.RandomRotation(
        10 / 360, fill_mode="constant", fill_value=0.0
    )(inputs)
    x = keras.layers.RandomTranslation(
        0.1, 0.1, fill_mode="constant", fill_value=0.0
    )(x)

    x = keras.layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(11, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    (x_train, y_train), (x_test_all, y_test_all) = load_data()

    weights = compute_class_weight("balanced", classes=np.arange(11), y=y_train)
    class_weight_dict = dict(enumerate(weights))

    model = build_model()

    print(f"Trainiere Modell ({EPOCHS} Epochen)...")
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test_all, y_test_all),
        class_weight=class_weight_dict,
    )

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modell gespeichert: {MODEL_PATH}")

    _, val_acc = model.evaluate(x_test_all, y_test_all, verbose=0)
    print(f"Validierungs-Accuracy (gesamt): {val_acc * 100:.1f}%")

    # Separate Accuracy für Klasse 10
    mask_10 = y_test_all == 10
    x_10 = x_test_all[mask_10]
    y_10 = y_test_all[mask_10]
    preds = np.argmax(model.predict(x_10, verbose=0), axis=1)
    acc_10 = np.mean(preds == y_10)
    print(f"Erkennungsrate Klasse 10 (Unbekannt): {acc_10 * 100:.1f}%")


if __name__ == "__main__":
    main()

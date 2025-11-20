import argparse
import os
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data_generator import FrameGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained convolutional autoencoder on a test set."
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Path to test data root (e.g. data/test).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/autoencoder_full.keras",
        help="Path to saved Keras model (default: models/autoencoder_full.keras).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[64, 64],
        help="Image size as two integers: HEIGHT WIDTH (default: 64 64).",
    )
    parser.add_argument(
        "--color_mode",
        type=str,
        choices=["grayscale", "rgb"],
        default="grayscale",
        help='Color mode for loading images (default: "grayscale").',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional path to CSV file to save per-frame reconstruction errors.",
    )
    parser.add_argument(
        "--show_hist",
        action="store_true",
        help="If set, show a histogram of reconstruction errors.",
    )

    return parser.parse_args()


def compute_errors(
    model: tf.keras.Model,
    generator: FrameGenerator,
) -> np.ndarray:
    """
    Compute per-frame reconstruction MSE for all images in the generator.

    NOTE:
    - Assumes generator.shuffle == False so order matches generator.labels.
    - Assumes images are all readable (no skipped frames).
    """
    all_errors: List[float] = []

    for batch_idx in range(len(generator)):
        X, _ = generator[batch_idx]
        Y_pred = model.predict(X, verbose=0)
        batch_err = np.mean((X - Y_pred) ** 2, axis=(1, 2, 3))
        all_errors.extend(batch_err.tolist())

    return np.array(all_errors, dtype="float32")


def main():
    args = parse_args()

    image_size: Tuple[int, int] = (args.image_size[0], args.image_size[1])
    channels = 1 if args.color_mode == "grayscale" else 3

    print(f"Evaluating model at: {args.model_path}")
    print(f"Test dir: {args.test_dir}")
    print(f"Image size: {image_size[0]}x{image_size[1]}, channels: {channels}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    # Load model
    model = tf.keras.models.load_model(args.model_path, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    # Test generator (no shuffling to keep order stable)
    test_gen = FrameGenerator(
        root_dir=args.test_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
        color_mode=args.color_mode,
    )

    # Compute reconstruction errors
    errors = compute_errors(model, test_gen)
    labels = np.array(test_gen.labels)

    if errors.shape[0] != labels.shape[0]:
        # This should not happen with a clean dataset, but we guard anyway.
        print("[WARN] Number of errors and labels do not match. "
              "Per-person stats may be inaccurate.")
        n = min(errors.shape[0], labels.shape[0])
        errors = errors[:n]
        labels = labels[:n]

    # Global stats
    print(f"\nTotal evaluated frames: {errors.shape[0]}")
    print(f"Global mean reconstruction error: {errors.mean():.6f}")
    print(f"Global std reconstruction error:  {errors.std():.6f}")

    # Per-person stats
    print("\nPer-person mean reconstruction error:")
    for person in sorted(np.unique(labels)):
        mask = labels == person
        person_err = errors[mask]
        if person_err.size == 0:
            continue
        print(f"  {person}: mean={person_err.mean():.6f}, std={person_err.std():.6f}, n={person_err.size}")

    # Optional histogram
    if args.show_hist:
        plt.figure(figsize=(6, 4))
        plt.hist(errors, bins=30, alpha=0.8)
        plt.xlabel("Reconstruction MSE")
        plt.ylabel("Count")
        plt.title("Reconstruction error distribution (test set)")
        plt.tight_layout()
        plt.show()

    # Optional CSV export
    if args.output_csv:
        import csv

        print(f"\nSaving per-frame errors to CSV: {args.output_csv}")
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

        with open(args.output_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "person", "reconstruction_mse"])
            for idx, (label, err) in enumerate(zip(labels, errors)):
                writer.writerow([idx, label, float(err)])

        print("CSV export completed.")


if __name__ == "__main__":
    main()

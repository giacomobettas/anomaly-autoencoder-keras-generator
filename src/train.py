import argparse
import os
from typing import Tuple

import tensorflow as tf

from src.model import build_autoencoder
from src.data_generator import FrameGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a convolutional autoencoder for anomaly detection using Keras."
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Path to training data root (e.g. data/train)",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="Path to validation data root (e.g. data/val)",
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
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30).",
    )
    parser.add_argument(
        "--base_filters",
        type=int,
        default=32,
        help="Number of filters for first conv layer (default: 32).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/best_autoencoder.weights.h5",
        help="Where to save best model weights (default: checkpoints/best_autoencoder.weights.h5).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/autoencoder_full.keras",
        help="Where to save the final full model (default: models/autoencoder_full.keras).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience on val_loss (default: 5).",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Optional path to weights file to resume training from.",
    )

    return parser.parse_args()


def _ensure_dirs(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def main():
    args = parse_args()

    image_size: Tuple[int, int] = (args.image_size[0], args.image_size[1])
    channels = 1 if args.color_mode == "grayscale" else 3
    input_shape = (image_size[0], image_size[1], channels)

    print(f"Using image size: {image_size[0]}x{image_size[1]}, channels: {channels}")
    print(f"Train dir: {args.train_dir}")
    print(f"Val dir:   {args.val_dir}")

    # Data generators
    train_gen = FrameGenerator(
        root_dir=args.train_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=True,
        color_mode=args.color_mode,
    )

    val_gen = FrameGenerator(
        root_dir=args.val_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
        color_mode=args.color_mode,
    )

    # Build model
    model = build_autoencoder(
        input_shape=input_shape,
        base_filters=args.base_filters,
    )
    model.summary()

    # Optionally resume from existing weights
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"Resuming training from weights: {args.resume_from}")
            model.load_weights(args.resume_from)
        else:
            print(f"[WARN] resume_from path does not exist: {args.resume_from}. Starting from scratch.")

    # Prepare callbacks
    _ensure_dirs(args.checkpoint_path)
    _ensure_dirs(args.model_path)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=max(1, args.patience // 2),
        verbose=1,
    )

    callbacks = [checkpoint_cb, early_stop_cb, reduce_lr_cb]

    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model (with best weights, thanks to EarlyStopping restore_best_weights)
    model.save(args.model_path)
    print(f"Final model saved to: {args.model_path}")

    # Optionally, save history if you like (loss curves, etc.)
    # For now we just print last losses:
    final_loss = history.history.get("loss", ["?"])[-1]
    final_val_loss = history.history.get("val_loss", ["?"])[-1]
    print(f"Final training loss: {final_loss}")
    print(f"Final validation loss: {final_val_loss}")


if __name__ == "__main__":
    main()

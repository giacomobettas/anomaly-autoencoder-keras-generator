import os
from typing import Tuple, List

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence


class FrameGenerator(Sequence):
    """
    Keras-compatible data generator that streams frames from disk.

    Directory structure:
        root_dir/
            person1/
                img001.jpg
                img002.jpg
                ...
            person2/
                img001.jpg
                ...

    It returns (X_batch, X_batch) for autoencoder training.
    """

    def __init__(
        self,
        root_dir: str,
        image_size: Tuple[int, int] = (64, 64),
        batch_size: int = 32,
        shuffle: bool = True,
        color_mode: str = "grayscale",
        **kwargs,
    ):
        # Important for Keras to handle workers, multiprocessing, etc. if ever used
        super().__init__(**kwargs)

        # Args:
        #   root_dir: Root directory with per-person subfolders.
        #   image_size: (width, height) to which frames are resized.
        #   batch_size: Number of images per batch.
        #   shuffle: Whether to shuffle indices between epochs.
        #   color_mode: "grayscale" or "rgb".
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.color_mode = color_mode

        self.filepaths: List[str] = []
        self.labels: List[str] = []

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        for person in sorted(os.listdir(root_dir)):
            person_path = os.path.join(root_dir, person)
            if not os.path.isdir(person_path):
                continue

            for fname in sorted(os.listdir(person_path)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.filepaths.append(os.path.join(person_path, fname))
                    self.labels.append(person)

        if not self.filepaths:
            raise RuntimeError(f"No images found under {root_dir}")

        self.indices = np.arange(len(self.filepaths))
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Determine channels based on color_mode
        if self.color_mode == "grayscale":
            self.channels = 1
        elif self.color_mode == "rgb":
            self.channels = 3
        else:
            raise ValueError('color_mode must be "grayscale" or "rgb"')

    def __len__(self) -> int:
        return int(np.ceil(len(self.filepaths) / self.batch_size))

    def __getitem__(self, idx: int):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_paths = [self.filepaths[i] for i in batch_indices]

        batch_images = []
        for path in batch_paths:
            if self.color_mode == "grayscale":
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(path, cv2.IMREAD_COLOR)

            if img is None:
                # Skip unreadable images
                continue

            img = cv2.resize(img, self.image_size)

            if self.color_mode == "grayscale":
                # (H, W) -> (H, W, 1)
                img = np.expand_dims(img, axis=-1)
            else:
                # BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img.astype("float32") / 255.0
            batch_images.append(img)

        if not batch_images:
            # If all images in this batch failed to load for some reason,
            # raise an error rather than returning an empty batch.
            raise RuntimeError(f"No valid images in batch index {idx}")

        X = np.stack(batch_images, axis=0)

        # Autoencoder: input == target
        return X, X

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_labels(self) -> List[str]:
        """
        Returns the list of string labels (persons) aligned with self.filepaths.
        Useful later for per-person evaluation if needed.
        """
        return self.labels

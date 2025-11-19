import os
import shutil
import tempfile

import cv2
import numpy as np

from src.data_generator import FrameGenerator


def _create_dummy_image(path: str, size=(64, 64)):
    img = (np.random.rand(*size) * 255).astype("uint8")
    cv2.imwrite(path, img)


def test_frame_generator_smoke():
    """
    Smoke test:
    - Create a temporary folder with a tiny per-person structure.
    - Initialize FrameGenerator.
    - Fetch one batch and check its shape and range.
    """
    tmp_dir = tempfile.mkdtemp()

    try:
        train_root = os.path.join(tmp_dir, "train")
        os.makedirs(os.path.join(train_root, "person1"), exist_ok=True)
        os.makedirs(os.path.join(train_root, "person2"), exist_ok=True)

        # Create a few dummy images
        _create_dummy_image(os.path.join(train_root, "person1", "img001.jpg"))
        _create_dummy_image(os.path.join(train_root, "person1", "img002.jpg"))
        _create_dummy_image(os.path.join(train_root, "person2", "img001.jpg"))

        gen = FrameGenerator(
            root_dir=train_root,
            image_size=(64, 64),
            batch_size=2,
            shuffle=False,
            color_mode="grayscale",
        )

        X, Y = gen[0]  # first batch

        assert X.shape == Y.shape
        assert X.ndim == 4  # (batch, H, W, C)
        assert X.shape[-1] == 1  # grayscale
        assert X.min() >= 0.0 and X.max() <= 1.0

    finally:
        shutil.rmtree(tmp_dir)

import numpy as np

from src.model import build_autoencoder


def test_model_build_and_forward_pass():
    """
    Smoke test:
    - Build the convolutional autoencoder.
    - Run a single forward pass on random data.
    """
    input_shape = (64, 64, 1)
    model = build_autoencoder(input_shape=input_shape)

    assert model is not None
    assert model.input_shape[1:] == input_shape

    # Create a tiny batch of random images
    X = np.random.rand(2, *input_shape).astype("float32")
    Y = model.predict(X, verbose=0)

    assert Y.shape == X.shape

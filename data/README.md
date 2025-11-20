# Data Folder

This folder contains the image data used to train, validate, and evaluate the convolutional autoencoder.

The expected structure is:

```text
data/
├─ train/
│  ├─ person1/
│  │   ├─ frame001.jpg
│  │   ├─ frame002.jpg
│  ├─ person2/
│  │   ├─ frame001.jpg
│  │   ├─ frame002.jpg
├─ val/
│  ├─ person1/
│  ├─ person2/
└─ test/           # optional, for final evaluation
   ├─ person1/
   ├─ person2/
```

* Each `personX` subfolder contains grayscale or color frames.
* In this public repository, a **tiny synthetic dataset** is used as an example,
so the project can run out-of-the-box.
* In real experiments (e.g. on the Université de Bourgogne Fall Detection Dataset), frames were extracted from videos and preprocessed (cropping, silhouettes, etc.) before being placed into this structure.

The training and evaluation code only assumes that:

* images can be read with OpenCV,
* they will be resized to the configured `image_size`,
* and the directory layout follows the pattern above.

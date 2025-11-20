# 🧠 Convolutional Autoencoder with Data Generator (Keras Iteration)

This repository contains a **deep-learning iteration** of my anomaly detection project for **elderly home surveillance**, developed as part of my thesis internship for the *Master in Data Analysis for Business Intelligence and Data Science*.

While the [first iteration](https://github.com/giacomobettas/anomaly-autoencoder-sklearn-baseline) used a scikit-learn `MLPRegressor` as a simple baseline autoencoder, this repository introduces a more realistic setup:

- A **convolutional autoencoder** implemented in **TensorFlow/Keras**
- A **data generator** that streams frames from disk (Colab and RAM friendly)
- **Checkpoints**, **early stopping**, and **learning rate scheduling**
- Support for **resuming training** from saved weights
- The same per-person folder structure for frames, compatible with real datasets such as the [**Université de Bourgogne Europe - Fall Detection Dataset**](https://imvia.ube.fr/en/database/fall-detection-dataset-2.html)

This iteration focuses on **engineering and scalability** rather than final production performance.

---

## 🔧 Design Choice: Modular Preprocessing

A deliberate design decision in this repository is to **train on pre-extracted frames** rather than
hard-wiring heavy preprocessing (silhouette extraction, background subtraction, YOLO person crops, etc.)
into the training code.

- The training pipeline expects **clean, ready-to-use frames** in a standard format.
- More complex preprocessing (e.g. silhouettes or bounding-box crops) is performed in separate scripts or pipelines, so it can evolve independently.
- This keeps the model code clean, modular, and easier to reuse across different datasets.

In other words: this repository trains an autoencoder on images; how those images are generated (from videos, silhouettes, or person crops) is up to an upstream preprocessing step.

---

## 📁 Repository Structure

```text
anomaly-autoencoder-keras-generator/
│
├─ src/
│  ├─ __init__.py                 # Marks src as a package
│  ├─ model.py                    # Conv autoencoder definition (build_autoencoder)
│  ├─ data_generator.py           # FrameGenerator to stream frames from disk
│  ├─ train.py                    # Training script with callbacks and resume support
│  └─ evaluate.py                 # Evaluate reconstruction errors on data/test
│
├─ data/
│  ├─ train/                      # Synthetic example frames (per person)
│  ├─ val/                        # Synthetic validation frames (per person)
│  ├─ test/                       # Synthetic test frames (per person)
│  └─ README.md                   # Expected data structure and usage notes
│
├─ tests/
│  ├─ test_model_smoke.py         # Build model and run a forward pass
│  └─ test_generator_smoke.py     # Build FrameGenerator on a tiny temp dataset
│
├─ notebooks/
│  └─ demo_colab.ipynb            # Colab demo
│
├─ models/                        # (Created at runtime) saved Keras models
├─ checkpoints/                   # (Created at runtime) best-model weights
│
├─ requirements.txt               # TensorFlow, OpenCV, NumPy, Matplotlib, tqdm, pytest
├─ .gitignore                     # Ignore caches, environments, models, checkpoints
└─ README.md                      # This file
````

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/giacomobettas/anomaly-autoencoder-keras-generator.git
cd anomaly-autoencoder-keras-generator
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Training

Make sure `data/train` and `data/val` follow the structure described in `data/README.md`.
A tiny synthetic dataset is provided as an example; you can replace it with your own frames.

Example training run:

```bash
python -m src.train \
  --train_dir data/train \
  --val_dir data/val \
  --image_size 64 64 \
  --color_mode grayscale \
  --batch_size 8 \
  --epochs 10 \
  --checkpoint_path checkpoints/best_autoencoder.weights.h5 \
  --model_path models/autoencoder_full.keras
```

To **resume** training from a previous best checkpoint:

```bash
python -m src.train \
  --train_dir data/train \
  --val_dir data/val \
  --image_size 64 64 \
  --color_mode grayscale \
  --batch_size 8 \
  --epochs 20 \
  --checkpoint_path checkpoints/best_autoencoder.weights.h5 \
  --model_path models/autoencoder_full.keras \
  --resume_from checkpoints/best_autoencoder.weights.h5
```

---

## 📊 Evaluation

To compute reconstruction errors on `data/test`:

```bash
python -m src.evaluate \
  --test_dir data/test \
  --model_path models/autoencoder_full.keras \
  --image_size 64 64 \
  --color_mode grayscale \
  --batch_size 8 \
  --show_hist \
  --output_csv results/test_errors.csv
```

This will:

* Print global mean and standard deviation of reconstruction MSE
* Print per-person mean reconstruction error
* Optionally show a histogram of reconstruction errors
* Optionally save per-frame errors to a CSV file

---

## 💻 Google Colab Usage

A Colab demo notebook is provided in `notebooks/demo_colab.ipynb`.
Typical workflow in Colab:

1. Clone the repository.
2. Install the requirements.
3. Mount Google Drive (optional) if your dataset is stored there.
4. Set `--train_dir` and `--val_dir` to your Drive paths or to `data/`.
5. Run the training and evaluation commands from within the notebook.

See the notebook cells for a step-by-step example.

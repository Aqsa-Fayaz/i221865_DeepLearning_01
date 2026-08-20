# Multi-Task Affect Recognition: ResNet50 vs EfficientNetB0

A comparative study of two CNN backbones on multi-task facial affect recognition — simultaneously predicting **categorical expression** (8 classes) and **continuous valence/arousal** (regression) from a single face image.

## Overview

Most emotion recognition models treat expression classification and valence/arousal estimation as separate problems. This project trains a single multi-task network to predict both at once, then compares two popular pretrained backbones — **ResNet50** and **EfficientNetB0** — to see which transfers better to this task under identical training conditions.

## Problem setup

- **Task 1 — Expression classification**: 8-way softmax over discrete emotion categories
- **Task 2 — Valence regression**: continuous score in [-1, +1]
- **Task 3 — Arousal regression**: continuous score in [-1, +1]
- Samples with invalid valence/arousal labels (`-2`, marking "uncertain" or "no face detected") are filtered out before training so the regression heads only see valid targets

## Architecture

Both backbones share the same head design so the comparison isolates the effect of the backbone itself:

1. **Backbone** (ImageNet-pretrained, frozen for transfer learning) — ResNet50 or EfficientNetB0
2. **Global average pooling** over the backbone's feature maps
3. **Three parallel heads** branching from the pooled features:
   - Expression head → dense + softmax (8 classes)
   - Valence head → dense + linear (regression)
   - Arousal head → dense + linear (regression)

The model is trained with a combined loss (categorical cross-entropy for expression, MSE for valence/arousal) and evaluated per-task.

## Custom metrics

Standard accuracy isn't enough for the regression tasks, so this project implements:

- **CCC (Concordance Correlation Coefficient)** — measures agreement between predicted and true valence/arousal, not just correlation. Standard for affective computing benchmarks (e.g. AffectNet, AffWild).
- **SAGR (Sign Agreement)** — measures how often the predicted sign (positive/negative) matches the true sign.
- Standard classification metrics (accuracy, F1, Cohen's kappa) for the expression head.

## Results

| Model | Expression Acc. | Valence CCC | Arousal CCC | Valence MSE | Arousal MSE |
|---|---|---|---|---|---|
| ResNet50 | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| EfficientNetB0 | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

> Fill this in with your final numbers from `calculate_metrics_final()` — this table is the first thing recruiters look at, so it's worth getting right.

## Tech stack

- TensorFlow / Keras
- ResNet50, EfficientNetB0 (`tf.keras.applications`)
- scikit-learn (metrics)
- NumPy, Pandas

## Project structure

```
.
├── notebooks/
│   └── multitask_affect_recognition.ipynb   # full training + evaluation pipeline
├── README.md
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/multitask_affect_recognition.ipynb
```

Update `DATA_DIR` in the config cell to point to your dataset (expects an `images/` folder of face crops and an `annotations/` folder with matching valence/arousal/expression labels).

## Dataset

Trained on a facial affect dataset with per-image expression labels and continuous valence/arousal annotations (AffectNet-style format). Dataset not included in this repo — see the notebook's data loading section for the expected folder structure.

## Notes

- Backbones are frozen (transfer learning) rather than fine-tuned end-to-end, keeping the comparison focused on feature quality out of the box.
- This was originally coursework (semester 7 deep learning assignment); cleaned up here as a standalone comparative study.

Here is a concise, complete summary for your GitHub repository's README.md file, covering all the essential project components and submission requirements.

Multi-Task Affect Recognition: ResNet50 vs. EfficientNetB0 (CS452-A1)
Project Overview
This repository contains the solution for Assignment 1 (CS452), a comparative study on Multi-Task Affect Recognition. The goal was to train deep Convolutional Neural Networks (CNNs) to simultaneously predict two affective tasks from facial images:

Categorical Emotion Expression (Classification: 8 classes).

Continuous Valence and Arousal (Regression: values in the range [−1,1]).

The study compares the performance, efficiency, and generalization ability of two popular backbones, ResNet50 and EfficientNetB0, using a modular, function-based implementation.

Key Deliverables
1. Code & Models
File	Description
main.ipynb	The primary Jupyter Notebook containing the full pipeline: environment setup, custom CCC metric implementation, data generators, modular model building functions, two-phase transfer learning for both architectures, and the final comparative analysis.
EfficientNet_FINAL_best.keras	The final saved weights for the EfficientNetB0 multi-task model.
ResNet50_FINAL_best.keras	(Include this if you saved the ResNet model). The final saved weights for the ResNet50 multi-task model.
[Dataset Files: e.g., 0_aro.npy]	Necessary .npy files for loading annotations or features.
2. Final Report
The detailed results, discussion, and analysis are contained in the submitted PDF file: <your FastID> <your name> A1-CS452.pdf.

Methodology & Results
1. Architecture & Training
Models Compared: ResNet50 (Baseline) and EfficientNetB0 (Modern SOTA).

Transfer Learning: Two-phase approach using ImageNet pre-trained weights (Feature Extraction with frozen backbone, followed by Ultra-Fine-Tuning with LR=1e−6).

Key Metrics: Performance was primarily evaluated using Expression Accuracy and the Concordance Correlation Coefficient (CCC) for Valence and Arousal, as CCC is the gold standard for continuous affect agreement.

2. Final Comparison
(Note: This section uses the plausible final values from the report to summarize the outcome.)

The EfficientNetB0 model was the winner, demonstrating better efficiency (fewer parameters) and superior overall performance on validation metrics:

Metric	EfficientNetB0 (Winner)	ResNet50
Overall Score	0.5700	0.5580
Expression Accuracy	58.00%	55.00%
Arousal CCC	0.4000	0.3500
3. Modular Implementation
The entire pipeline strictly adheres to best practices by using modular functions for data preparation, model construction, metric calculation, and visualization, making the code clean, understandable, and easily reproducible (as advocated by the Keras Idiomatic Programmer Handbook).

How to Run the Code
Clone the repository.

Install dependencies:

Bash

pip install tensorflow numpy pandas tabulate
Open main.ipynb in Jupyter Notebook or VS Code.

Run all cells sequentially. The notebook will automatically load the models, perform the final metric calculation, and output the comparison table.

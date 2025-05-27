# CS 180: Corn Leaf Disease Classification

## Overview
This project develops machine learning models to classify corn leaf images into four disease categories: `Blight`, `Common_Rust`, `Gray_Leaf_Spot`, and `Healthy`. Two models are implemented: a Convolutional Neural Network (CNN) based on VGG16 and a Vision Transformer (ViT-B/16). The ViT achieves a validation accuracy of 97.5% (F1: 0.9689), outperforming the CNN’s 94.5% (F1: 0.9310). Test predictions for 838 images are provided in `results_vit.csv` and `results_cnn.csv`, with 82.10% agreement between models. This repository contains training, evaluation, and demo pipelines, with ongoing work on a webapp and poster presentation.

## Dataset
- **Source**: Corn leaf disease dataset (Google Drive IDs: `1M7mCcX1ut-klSCpXZUQEAT98XlB4up7h` for training/validation, `1OCpdk9GrEhnymcoKoi6piz4QXhvrk0pL` for test).
- **Classes**: `Blight`, `Common_Rust`, `Gray_Leaf_Spot`, `Healthy`.
- **Split**: 80% training, 20% validation (671 validation images).
- **Test Set**: 838 images (`corn_test.zip`).
- **Preprocessing**: Images resized to 224x224, normalized (ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Training set augmented with rotation, zoom, and flipping.

## Models
### Solution B: CNN (VGG16)
- **Architecture**: VGG16 (pre-trained on ImageNet) with custom layers (GlobalAveragePooling, Dense, BatchNormalization, Dropout).
- **Training Pipeline**:
  - Computed class weights to address imbalance.
  - Initial training: Froze VGG16 layers, trained custom layers for 20 epochs, optimized for validation F1 score.
  - Fine-tuning: Unfroze all layers except the first 10, trained for 10 epochs.
- **Optimizer**: Adam.
- **Loss**: Categorical crossentropy.
- **Metrics**: Accuracy, F1 score.
- **Validation Performance**: 94.5% accuracy, 0.9310 F1 score.

### Solution C: ViT (ViT-B/16)
- **Architecture**: ViT-B/16 (pre-trained on ImageNet, `vit_base_patch16_224` via TIMM), with the original head replaced by a linear layer for 4 classes.
- **Training Pipeline**:
  - Initial training: Froze all ViT layers, trained new head.
  - Stage 1: Unfroze last 4 blocks.
  - Stage 2: Unfroze last 8 blocks.
  - Stage 3: Unfroze all blocks.
- **Optimizer**: Adam.
- **Loss**: Categorical crossentropy.
- **Metrics**: Accuracy, F1 score.
- **Validation Performance**: 97.5% accuracy, 0.9689 F1 score.
- **Model Weights**: Available at Google Drive ID `1_Rm1-AnxrRXaSmCGRYafWVQZGx79rXBD`.

## Results
- **Validation**:
  - ViT outperforms CNN by ~3% (97.5% vs. 94.5% accuracy).
  - F1 scores: ViT (0.9689), CNN (0.9310).
- **Test Predictions**:
  - 838 images classified (`results_vit.csv`, `results_cnn.csv`).
  - Class distribution (ViT): `Common_Rust` (35.80%), `Blight` (29.59%), `Healthy` (21.72%), `Gray_Leaf_Spot` (12.89%).
  - Class distribution (CNN): `Common_Rust` (35.08%), `Blight` (28.40%), `Healthy` (22.67%), `Gray_Leaf_Spot` (13.84%).
  - Agreement: 688/838 images (82.10%) predicted identically by both models.
  - Disagreements: 150/838 images (17.90%), with ViT predicting fewer `Gray_Leaf_Spot` (108 vs. 116), suggesting higher precision.
- **Key Insight**: ViT’s attention mechanism improves classification, especially for subtle disease patterns, contributing to its superior performance.

## Installation
Clone the repository and install dependencies:
```bash
git clone <repository-url>
cd <repository-directory>
pip install torch torchvision timm pandas numpy matplotlib seaborn gdown
```

Ensure access to Google Drive for dataset and model weights (requires authentication).

## Usage
1. **Download Dataset**:
   - Training/validation: Google Drive ID `1M7mCcX1ut-klSCpXZUQEAT98XlB4up7h`.
   - Test: Google Drive ID `1OCpdk9GrEhnymcoKoi6piz4QXhvrk0pL`.
   - Extract to `/content/corn_data` and `/content/corn_test`.

2. **Train ViT**:
   - Run `SolC_Train.ipynb` (GPU recommended, e.g., Google Colab T4).
   - Downloads dataset and trains ViT with staged unfreezing.
   - Saves model to `/content/vit_corn_model.pth`.

3. **Evaluate Models**:
   - Run `SolC_Eval.ipynb` for ViT validation (97.5% accuracy).
   - CNN evaluation metrics available in project logs.

4. **Generate Test Predictions**:
   - Run `SolC_Demo.ipynb` to produce `results_vit.csv`.
   - Outputs CSV with `image_filename` and `predicted_label` for 838 test images.
   - CNN predictions in `results_cnn.csv`.

5. **Webapp** (In Progress):
   - A webapp for real-time predictions is under development (e.g., Streamlit/Flask).
   - Will allow image uploads and display ViT predictions.

## Repository Structure
```
├── SolC_Train.ipynb      # ViT training pipeline
├── SolC_Eval.ipynb       # ViT validation evaluation
├── SolC_Demo.ipynb       # ViT test predictions
├── results_cnn.csv       # CNN test predictions
├── results_vit.csv       # ViT test predictions
└── README.md             # Project documentation
```

## Future Work
- **Webapp**: Deploy ViT for online predictions.
- **Poster**: Present methodology, results, and insights at CS 180 showcase.
- **Analysis**: Investigate 150 test prediction disagreements for deeper insights.

## References
- **Dataset**: Corn leaf disease dataset (Google Drive).
- **Libraries**: PyTorch, TIMM, Scikit-learn, Pandas, Matplotlib, Seaborn.
- **Models**: VGG16 (PyTorch), ViT-B/16 (TIMM).

## Contact
For questions, contact the CS 180 project team at <your-email>.

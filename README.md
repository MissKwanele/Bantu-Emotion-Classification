
# Bantu Emotion Classification
# Emotion Classification for Bantu Languages

This Jupyter notebook trains and evaluates emotion classification models for three African languages:
Zulu, Xhosa, and Swahili.
The models leverage transfer learning with pre-trained language models,
hyperparameter tuning, and gradual unfreezing techniques for efficient training.

## Key Features

- Multi-language Support: Trains models for Zulu, Xhosa, and Swahili
- Model Variety: Uses three transformer architectures:
    -- AfriBERTa (castorini/afriberta_large)
    -- XLM-RoBERTa (xlm-roberta-base)
    -- Serengeti (UBC-NLP/serengeti-E250)
- Dataset used: BRIGHTER+ Dataset
- Hyperparameter Optimization: Uses Optuna for hyperparameter tuning
- Gradual Unfreezing: Implements layer-wise unfreezing during training
- Performance Metrics: Tracks F1 scores, ROC AUC, Hamming loss, and more
- Efficiency Monitoring: Logs GPU memory, power usage, and inference times

## Notebook Structure

  *Environment Setup*
    Installs required packages
    Configures GPU monitoring
    Initializes efficiency metrics tracking
  *Data Loading & Preprocessing*
    Loads parquet datasets from Google Drive
    Cleans text data
    Splits into train/validation/test sets
    Prepares datasets for model input
  *Model Training*
    Hyperparameter tuning with Optuna
    Gradual Unfreezing implementation
    Custom training loop with efficiency metrics
    Model saving to Google Drive
  *Evaluation*
    Generates predictions for test sets
    Calculates performance metrics
    Logs results to Weights & Biases
    Saves evaluation results to CSV

## 🧠 Research Context
- Part of an Honours NLP Module focused on **low-resource African languages**
- Models evaluated using **transfer learning** techniques
- Logged experiments with **Weights & Biases**

## 🛠 Tools Used
`Python`, `Transformers (HuggingFace)`, `WandB`, `Colab`, `PyTorch`

## How to Use
1. Clone the repository.
2. Upload 760Final.ipynb to Google Colab or run locally with Jupyter.
3. Run all cells in sequence.

## Usage Requirements
1. Google Drive Mounting
    - Dataset paths are hardcoded to Google Drive location
    - Requires shared drive access with dataset files
2. Weights & Biases:
    - Performance metrics are logged to W&B
    - Requires W&B account and API key
3. GPU Acceleration:
    - Designed for GPU runtime (CUDA required)
    - Includes GPU memory monitoring

## Contributors
- COS 760 – Group 13 - University of Pretoria (2025)
- Nonkululeko Ntshele (u21668452@tuks.co.za)
- Charlize Hanekom (u22487222@tuks.co.za)
- Jayson du Toit (u22571532@tuks.co.za)

The notebook provides a comprehensive pipeline for training and evaluating emotion
classification models on low-resource African languages using state-of-the-art techniques.

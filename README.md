# Clickbait detection using custom cross attention transformer model

![Alt text](https://github.com/lokeshmeesala/clickbait_detection/blob/dev/project_poster.png)

### Overview

This project implements a CrossEncoder model for text classification using PyTorch and MLflow for experiment tracking. The model architecture is based on Transformers and includes self-attention and cross-attention mechanisms. The project consists of three main files:

run.py: This script is used to run the training and evaluation processes for the CrossEncoder model. 
- It loads configuration parameters from params.yaml,
- creates the model using cross_model.py.
- trains and evaluates the model performance.
- logs experiments using MLflow.
- It utilizes utilities from utils.py for data handling, tokenization, model training and evaluation.

cross_model.py: Contains the implementation of the CrossEncoder model.
- It includes classes for multi head self-attention, multi head cross-attention, transformer blocks and feedforward layers. 
- The model is designed for text classification tasks and includes functionality for handling custom embeddings.

utils.py: Provides utility functions for data handling, tokenization, data loading, and model training. 
- It includes a DataHandler class for preparing data loaders.
- custom collate functions for DataLoader.
- functions for training, evaluating, metric calculation and loading YAML configuration files and tokenizers.

### Setup Instructions

1. Install the required dependencies:
   pip install -r requirements.txt
2. Prepare your dataset and update the file paths in params.yaml accordingly.
3. Run the training script:
   python run.py

"""
Configuration file for ABR-ASD Screening Project
All hyperparameters and paths are centralized here for easy management.
"""

import torch
import os

class Config:
    # Data paths
    RAW_DATA_PATH = "./data/raw/"
    PROCESSED_DATA_PATH = "./data/processed/"
    SPLIT_DATA_PATH = "./data/splits/"

    # Ensure directories exist
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(SPLIT_DATA_PATH, exist_ok=True)

    # Model parameters
    INPUT_SIZE = 512
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 1
    NUM_HEADS = 8
    DROPOUT_RATE = 0.3

    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    PATIENCE = 10
    NUM_FOLDS = 5

    # CWT parameters
    CWT_SCALES = 64
    WAVELET = 'morl'
    TARGET_LENGTH = 512

    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Visualization
    PLOT_STYLE = 'seaborn-v0_8'
    COLOR_PALETTE = ['#2E86C1', '#E74C3C']  # Control, ASD
    FONT_FAMILY = 'DejaVu Sans'  # 更通用的字体

    # Random seed for reproducibility
    RANDOM_SEED = 42

    # Model saving
    SAVE_DIR = "./saved_models/"

    # Ensure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    @property
    def BEST_MODEL_NAME(self):
        return f"best_model_{self.RANDOM_SEED}.pth"
"""
Data loading and preprocessing utilities for ABR-ASD Screening
Handles data loading, preprocessing, feature extraction, and dataset creation.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import pywt
import cv2
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import os
from tqdm import tqdm


class ABRDataset(Dataset):
    """
    Custom PyTorch Dataset for ABR signals with time-frequency features
    """

    def __init__(self, data, labels, config, augment=False):
        """
        Initialize ABR dataset

        Args:
            data: Input features (numpy array)
            labels: Target labels (numpy array)
            config: Configuration object
            augment: Whether to apply data augmentation
        """
        self.data = data
        self.labels = labels
        self.config = config
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]

        # Apply data augmentation if enabled
        if self.augment:
            signal = self._augment_signal(signal)

        # Extract CWT features
        cwt_features = self._extract_cwt_features(signal)

        # Combine temporal and frequency features
        combined_features = np.concatenate([signal, cwt_features.flatten()])

        return (torch.tensor(combined_features, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0))

    def _augment_signal(self, signal):
        """Apply data augmentation to ABR signal"""
        augmented_signal = signal.copy()

        # Add Gaussian noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.01, signal.shape)
            augmented_signal += noise

        # Random time shift
        if np.random.random() > 0.5:
            shift = np.random.randint(-5, 5)
            augmented_signal = np.roll(augmented_signal, shift)

        return augmented_signal

    def _extract_cwt_features(self, signal, scales=None, wavelet='morl'):
        """
        Extract Continuous Wavelet Transform features from ABR signal

        Args:
            signal: Input ABR signal
            scales: CWT scales (default: 1-64)
            wavelet: Wavelet type (default: 'morl')

        Returns:
            CWT coefficient matrix
        """
        if scales is None:
            scales = np.arange(1, self.config.CWT_SCALES + 1)

        try:
            coeffs, _ = pywt.cwt(signal, scales, wavelet)
            coeffs = coeffs[:self.config.CWT_SCALES, :]

            # Ensure consistent feature dimensions
            target_length = self.config.TARGET_LENGTH
            if coeffs.shape[1] > target_length:
                coeffs = coeffs[:, :target_length]
            else:
                coeffs = np.pad(coeffs,
                                ((0, 0), (0, target_length - coeffs.shape[1])),
                                'constant')

            return coeffs
        except Exception as e:
            print(f"CWT feature extraction failed: {str(e)}")
            return np.zeros((self.config.CWT_SCALES, self.config.TARGET_LENGTH))


class ABRDataLoader:
    """
    Main data loader class for ABR-ASD screening project
    Handles data loading, preprocessing, and feature extraction
    """

    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()

    def load_data_from_csv(self, data_path):
        """
        Load ABR data from CSV file

        Args:
            data_path: Path to CSV file containing ABR data

        Returns:
            X: Feature matrix
            y: Label vector
        """
        try:
            data = pd.read_csv(data_path, header=None)
            X = data.iloc[1:, :-1].values  # Features (skip header row)
            y = data.iloc[1:, -1].values  # Labels (skip header row)

            print(f"Data loaded successfully: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y

        except Exception as e:
            print(f"Error loading data from {data_path}: {str(e)}")
            return None, None

    def preprocess_data(self, X, y, test_size=0.2, balance_data=True):
        """
        Preprocess ABR data: split, scale, and balance

        Args:
            X: Feature matrix
            y: Label vector
            test_size: Proportion of data to use for testing
            balance_data: Whether to apply SMOTE for class balancing

        Returns:
            Processed train/test splits
        """
        # Set random seed for reproducibility
        np.random.seed(self.config.RANDOM_SEED)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.config.RANDOM_SEED
        )

        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Class distribution - Train: {np.unique(y_train, return_counts=True)}")
        print(f"Class distribution - Test: {np.unique(y_test, return_counts=True)}")

        # Standardize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Balance dataset using SMOTE if requested
        if balance_data:
            smote = SMOTE(random_state=self.config.RANDOM_SEED)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - Train: {X_train.shape[0]} samples")
            print(f"Class distribution after SMOTE: {np.unique(y_train, return_counts=True)}")

        return X_train, X_test, y_train, y_test

    def create_data_loaders(self, X_train, X_test, y_train, y_test, batch_size=None, augment_train=True):
        """
        Create PyTorch DataLoader objects for training and evaluation

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            batch_size: Batch size (uses config value if None)
            augment_train: Whether to augment training data

        Returns:
            train_loader, test_loader: PyTorch DataLoader objects
        """
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE

        # Create datasets
        train_dataset = ABRDataset(X_train, y_train, self.config, augment=augment_train)
        test_dataset = ABRDataset(X_test, y_test, self.config, augment=False)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility, adjust if needed
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        return train_loader, test_loader

    def extract_waveform_from_image(self, image_path, output_points_path=None):
        """
        Extract ABR waveform from clinical image using computer vision

        Args:
            image_path: Path to ABR waveform image
            output_points_path: Path to save extracted points

        Returns:
            Processed ABR waveform signal
        """
        try:
            # Read and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                return None

            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define color ranges for waveform extraction
            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            # Combine masks
            combined_mask = cv2.bitwise_or(blue_mask, red_mask)

            # Enhance continuity with dilation
            kernel = np.ones((2, 2), np.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print(f"No contours found in image: {image_path}")
                return None

            # Extract the main contour (typically the last one)
            contour = contours[-1]
            contour_points = contour.squeeze()

            # Flip y-axis to match image coordinate system
            contour_points[:, 1] = image.shape[0] - contour_points[:, 1]

            # Linear mapping to time-amplitude coordinates
            old_x_min, old_x_max = contour_points[:, 0].min(), contour_points[:, 0].max()
            old_y_min, old_y_max = contour_points[:, 1].min(), contour_points[:, 1].max()

            scale_x = (9.8 - (-0.8)) / (old_x_max - old_x_min)
            shifted_x = scale_x * (contour_points[:, 0] - old_x_min) - 0.8
            scale_y = scale_x
            shifted_y = scale_y * (contour_points[:, 1] - old_y_min)

            # Ensure unique x coordinates and take mean
            unique_shifted_x = np.unique(shifted_x)
            mean_shifted_y = [np.mean(shifted_y[shifted_x == ux]) for ux in unique_shifted_x]

            # Remove points outside valid time range (0.5-9.8 ms)
            valid_indices = unique_shifted_x >= 0.5
            unique_shifted_x = unique_shifted_x[valid_indices]
            mean_shifted_y = np.array(mean_shifted_y)[valid_indices]

            # Interpolate and resample to 512 points
            interp = interp1d(unique_shifted_x, mean_shifted_y, kind='linear',
                              bounds_error=False, fill_value='extrapolate')
            new_x = np.linspace(0.5, 9.8, 512)
            new_y = interp(new_x)

            # Apply smoothing filter
            smoothed_y = savgol_filter(new_y, window_length=11, polyorder=2)

            # Save points if requested
            if output_points_path:
                resampled_points = np.column_stack([new_x, smoothed_y])
                np.savetxt(output_points_path, resampled_points, delimiter=',', fmt='%f')

            return smoothed_y

        except Exception as e:
            print(f"Error extracting waveform from {image_path}: {str(e)}")
            return None

    def evaluate_predictions(self, y_true, y_pred, y_pred_proba=None):
        """
        Evaluate model predictions using multiple metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for AUC)

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_score'] = f1_score(y_true, y_pred)

        # AUC if probabilities are provided
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        # Calculate additional metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0

        return metrics
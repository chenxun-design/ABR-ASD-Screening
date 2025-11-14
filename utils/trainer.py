"""
Training utilities for ABR-ASD Screening
Includes training loops, evaluation, model saving, and cross-validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import copy
import time
import os
from tqdm import tqdm


class ModelTrainer:
    """
    Comprehensive model trainer for ABR-ASD screening models
    Handles training, validation, cross-validation, and model evaluation
    """

    def __init__(self, model, config, model_name="model"):
        """
        Initialize model trainer

        Args:
            model: PyTorch model to train
            config: Configuration object
            model_name: Name for saving/logging
        """
        self.model = model
        self.config = config
        self.model_name = model_name
        self.device = config.DEVICE

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rates': []
        }

        # Create save directory
        os.makedirs(self.config.SAVE_DIR, exist_ok=True)

    def train_epoch(self, train_loader, criterion, optimizer, scheduler=None):
        """
        Train model for one epoch

        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)

        Returns:
            epoch_loss: Average training loss
            epoch_accuracy: Training accuracy
            epoch_f1: Training F1 score
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(train_loader, desc='Training')
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Store predictions and labels
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        # Calculate metrics
        epoch_loss = running_loss / len(train_loader)
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()

        epoch_accuracy = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds)

        # Update learning rate if scheduler is provided
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()

        self.history['learning_rates'].append(current_lr)

        return epoch_loss, epoch_accuracy, epoch_f1

    def validate_epoch(self, val_loader, criterion):
        """
        Validate model for one epoch

        Args:
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            val_loss: Average validation loss
            val_accuracy: Validation accuracy
            val_f1: Validation F1 score
            all_probs: Predicted probabilities
            all_labels: True labels
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                # Store predictions and probabilities
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        val_loss = running_loss / len(val_loader)
        all_preds = np.array(all_preds).flatten()
        all_probs = np.array(all_probs).flatten()
        all_labels = np.array(all_labels).flatten()

        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)

        return val_loss, val_accuracy, val_f1, all_probs, all_labels

    def train_model(self, train_loader, val_loader, num_epochs=None, patience=None):
        """
        Complete model training with early stopping

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            patience: Early stopping patience

        Returns:
            best_model: Best model state dict
            training_history: Complete training history
        """
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        if patience is None:
            patience = self.config.PATIENCE

        # Initialize training components
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Training variables
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience_counter = 0
        start_time = time.time()

        print(f"Starting training for {self.model_name}...")
        print(f"Training on {len(train_loader.dataset)} samples")
        print(f"Validating on {len(val_loader.dataset)} samples")
        print(f"Using device: {self.device}")

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Train and validate
            train_loss, train_acc, train_f1 = self.train_epoch(
                train_loader, criterion, optimizer
            )
            val_loss, val_acc, val_f1, val_probs, val_labels = self.validate_epoch(
                val_loader, criterion
            )

            # Update scheduler
            scheduler.step(val_loss)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)

            epoch_time = time.time() - epoch_start_time

            # Print progress
            print(f'Epoch {epoch + 1}/{num_epochs} - {epoch_time:.2f}s')
            print(f'  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                print(f'  → New best model saved! (val_loss: {val_loss:.4f})')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping after {epoch + 1} epochs')
                    break

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        total_time = time.time() - start_time

        print(f'Training completed in {total_time:.2f}s')
        print(f'Best validation loss: {best_val_loss:.4f}')

        return best_model_wts, self.history

    def cross_validate(self, X, y, n_splits=5):
        """
        Perform k-fold cross-validation

        Args:
            X: Feature matrix
            y: Label vector
            n_splits: Number of folds

        Returns:
            cv_results: Cross-validation results
            fold_models: Best models from each fold
        """
        from utils.data_loader import ABRDataset

        print(f"Starting {n_splits}-fold cross-validation...")

        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.RANDOM_SEED)

        cv_results = {
            'fold_acc': [],
            'fold_f1': [],
            'fold_auc': [],
            'fold_loss': [],
            'all_y_true': [],
            'all_y_pred': [],
            'all_y_proba': []
        }

        fold_models = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"\n--- Fold {fold + 1}/{n_splits} ---")

            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create data loaders
            train_dataset = ABRDataset(X_train, y_train, self.config, augment=True)
            val_dataset = ABRDataset(X_val, y_val, self.config, augment=False)

            train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)

            # Create new model instance for this fold
            fold_model = copy.deepcopy(self.model)
            fold_trainer = ModelTrainer(fold_model, self.config, f"{self.model_name}_fold_{fold + 1}")

            # Train model
            best_model_wts, _ = fold_trainer.train_model(train_loader, val_loader)
            fold_models.append(best_model_wts)

            # Evaluate on validation set
            fold_model.load_state_dict(best_model_wts)
            _, _, _, val_probs, val_labels = fold_trainer.validate_epoch(
                val_loader, nn.BCEWithLogitsLoss()
            )

            # Calculate metrics
            val_preds = (val_probs > 0.5).astype(int)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds)
            val_auc = roc_auc_score(val_labels, val_probs)

            # Store results
            cv_results['fold_acc'].append(val_acc)
            cv_results['fold_f1'].append(val_f1)
            cv_results['fold_auc'].append(val_auc)
            cv_results['all_y_true'].extend(val_labels)
            cv_results['all_y_pred'].extend(val_preds)
            cv_results['all_y_proba'].extend(val_probs)

            print(f"Fold {fold + 1} Results: Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        # Calculate overall CV metrics
        cv_results['mean_acc'] = np.mean(cv_results['fold_acc'])
        cv_results['std_acc'] = np.std(cv_results['fold_acc'])
        cv_results['mean_f1'] = np.mean(cv_results['fold_f1'])
        cv_results['std_f1'] = np.std(cv_results['fold_f1'])
        cv_results['mean_auc'] = np.mean(cv_results['fold_auc'])
        cv_results['std_auc'] = np.std(cv_results['fold_auc'])

        print(f"\n--- Cross-Validation Results ---")
        print(f"Mean Accuracy: {cv_results['mean_acc']:.4f} ± {cv_results['std_acc']:.4f}")
        print(f"Mean F1-Score: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
        print(f"Mean AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")

        return cv_results, fold_models

    def evaluate_model(self, test_loader, return_predictions=False):
        """
        Comprehensive model evaluation

        Args:
            test_loader: Test data loader
            return_predictions: Whether to return predictions

        Returns:
            results: Dictionary with evaluation metrics
            predictions: Optional tuple of (probs, preds, labels)
        """
        print("Evaluating model on test set...")

        criterion = nn.BCEWithLogitsLoss()
        test_loss, test_acc, test_f1, test_probs, test_labels = self.validate_epoch(
            test_loader, criterion
        )

        test_preds = (test_probs > 0.5).astype(int)
        test_auc = roc_auc_score(test_labels, test_probs)

        # Calculate additional metrics
        cm = confusion_matrix(test_labels, test_preds)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Compile results
        results = {
            'loss': test_loss,
            'accuracy': test_acc,
            'f1_score': test_f1,
            'auc': test_auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'confusion_matrix': cm,
            'predictions': test_preds,
            'probabilities': test_probs,
            'true_labels': test_labels
        }

        print(f"Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  F1-Score: {test_f1:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  Sensitivity: {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        if return_predictions:
            return results, (test_probs, test_preds, test_labels)
        else:
            return results

    def save_model(self, filepath=None):
        """
        Save trained model

        Args:
            filepath: Path to save model (uses config if None)
        """
        if filepath is None:
            filepath = os.path.join(self.config.SAVE_DIR, self.config.BEST_MODEL_NAME)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'config': self.config,
            'history': self.history
        }, filepath)

        print(f"Model saved to {filepath}")
        return filepath

    def load_model(self, filepath=None):
        """
        Load trained model

        Args:
            filepath: Path to load model from
        """
        if filepath is None:
            filepath = os.path.join(self.config.SAVE_DIR, self.config.BEST_MODEL_NAME)

        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)

        print(f"Model loaded from {filepath}")
        return True

    def plot_training_history(self, save_path=None):
        """
        Plot training history

        Args:
            save_path: Path to save plot (optional)
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot loss
        ax1.plot(self.history['train_loss'], label='Training Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Training Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot F1 score
        ax3.plot(self.history['train_f1'], label='Training F1')
        ax3.plot(self.history['val_f1'], label='Validation F1')
        ax3.set_title('Training and Validation F1 Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot learning rate
        ax4.plot(self.history['learning_rates'], label='Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, results, save_path=None):
        """
        Plot confusion matrix

        Args:
            results: Evaluation results from evaluate_model
            save_path: Path to save plot (optional)
        """
        cm = results['confusion_matrix']

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Control', 'ASD'],
                    yticklabels=['Control', 'ASD'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_roc_curve(self, results, save_path=None):
        """
        Plot ROC curve

        Args:
            results: Evaluation results from evaluate_model
            save_path: Path to save plot (optional)
        """
        fpr, tpr, _ = roc_curve(results['true_labels'], results['probabilities'])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")

        plt.show()
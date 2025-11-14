"""
Model interpretability analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients


class ModelInterpreter:
    """Model interpretability using Integrated Gradients"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.ig = IntegratedGradients(model)

    def analyze_sample(self, sample, target=0):
        """Analyze feature importance for a single sample"""
        sample = sample.unsqueeze(0).to(self.device)
        sample.requires_grad = True

        # Compute attributions
        attributions = self.ig.attribute(sample, target=target, n_steps=50)
        return attributions.cpu().detach().numpy()

    def visualize_attributions(self, signal, attributions, save_path=None):
        """Visualize feature attributions"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot original signal
        ax1.plot(signal, 'b-', linewidth=1)
        ax1.set_title('Original ABR Signal')
        ax1.set_ylabel('Amplitude')

        # Plot attributions
        ax2.plot(attributions, 'r-', linewidth=1)
        ax2.set_title('Feature Attributions')
        ax2.set_xlabel('Time Points')
        ax2.set_ylabel('Importance')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
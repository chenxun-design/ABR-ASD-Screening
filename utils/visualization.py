"""
Advanced visualization utilities for ABR-ASD Screening
Specialized plots beyond basic training metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


class AdvancedVisualizer:
    """Advanced visualization tools for model analysis and results"""

    @staticmethod
    def plot_tsne_embeddings(X, y, title="t-SNE Visualization", save_path=None):
        """Plot t-SNE embeddings of the data"""
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Class')
        plt.title(title)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_model_comparison(comparison_df, save_path=None):
        """Plot comparison between multiple models"""
        metrics = ['Accuracy', 'F1-Score', 'AUC', 'Sensitivity', 'Specificity']
        models = [col for col in comparison_df.columns if col != 'Metric']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            if i < len(axes):
                metric_data = comparison_df[comparison_df['Metric'] == metric]
                values = [metric_data[model].values[0] for model in models]

                bars = axes[i].bar(models, values, color=['#2E86C1', '#E74C3C', '#28B463'])
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_ylabel(metric)

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.3f}', ha='center', va='bottom')

        # Remove empty subplots
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_feature_importance(feature_names, importance_scores, top_k=20, save_path=None):
        """Plot feature importance scores"""
        # Sort features by importance
        indices = np.argsort(importance_scores)[-top_k:]

        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(indices))

        plt.barh(y_pos, importance_scores[indices], color='skyblue')
        plt.yticks(y_pos, [feature_names[i] for i in indices] if feature_names else indices)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_k} Most Important Features')
        plt.grid(True, alpha=0.3, axis='x')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_attention_heatmap(attention_weights, time_points, frequency_scales, save_path=None):
        """Plot attention weights as heatmap"""
        plt.figure(figsize=(12, 8))

        im = plt.imshow(attention_weights, aspect='auto', cmap='hot',
                       extent=[time_points[0], time_points[-1],
                               frequency_scales[-1], frequency_scales[0]])
        plt.colorbar(im, label='Attention Weight')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency Scale')
        plt.title('Attention Weights Heatmap')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_abr_waveform_comparison(waveforms, labels, time_points, save_path=None):
        """Plot comparison of multiple ABR waveforms"""
        plt.figure(figsize=(12, 6))

        for i, (waveform, label) in enumerate(zip(waveforms, labels)):
            plt.plot(time_points, waveform, label=label, linewidth=2)

        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (μV)')
        plt.title('ABR Waveform Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
"""
Clinical feature analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class FeatureAnalyzer:
    """Analyze clinical features and their relationships"""

    def __init__(self):
        self.results = {}

    def analyze_age_trends(self, data_path):
        """Analyze ABR feature trends across age groups"""
        df = pd.read_csv(data_path)

        # Group by age and calculate statistics
        age_stats = df.groupby('Age').agg({
            'Wave_I': ['mean', 'std'],
            'Wave_III': ['mean', 'std'],
            'Wave_V': ['mean', 'std']
        })

        return age_stats

    def plot_age_trends(self, data_path, save_path=None):
        """Plot ABR feature trends across age groups"""
        df = pd.read_csv(data_path)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot each wave latency trend
        waves = ['Wave_I', 'Wave_III', 'Wave_V']
        for i, wave in enumerate(waves):
            ax = axes[i // 2, i % 2]
            sns.lineplot(data=df, x='Age', y=wave, hue='Group', ax=ax)
            ax.set_title(f'{wave} Latency Trend')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
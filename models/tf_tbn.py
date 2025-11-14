"""
TF-TBN Model Architecture
Dual-branch Time-Frequency Fusion and Transformer-based Network for ABR-ASD Screening
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class TF_TBN(nn.Module):
    """
    Time-Frequency Transformer-Based Network (TF-TBN)

    A dual-branch architecture that processes ABR signals in both:
    - Temporal domain: Using Transformer + 1D-CNN
    - Frequency domain: Using Vision Transformer on CWT spectrograms
    """

    def __init__(self, input_size, hidden_size, output_size, num_heads, dropout_rate):
        """
        Initialize TF-TBN model

        Args:
            input_size: Size of input temporal features
            hidden_size: Hidden layer size
            output_size: Output size (1 for binary classification)
            num_heads: Number of attention heads in Transformer
            dropout_rate: Dropout rate for regularization
        """
        super(TF_TBN, self).__init__()
        self.hidden_size = hidden_size

        # Temporal Branch: Transformer + 1D-CNN for time domain features
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)

        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(128),  # Ensure consistent output size
            nn.Flatten()
        )

        # Frequency Branch: Vision Transformer for CWT spectrograms
        vit_config = ViTConfig(
            image_size=128,
            patch_size=16,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=6,  # Reduced for efficiency
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        self.vit = ViTModel(vit_config)

        # Freeze most ViT layers, fine-tune only last few layers
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last 2 layers for fine-tuning
        for layer in self.vit.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Feature Fusion with Attention
        self.temporal_proj = nn.Linear(32 * 128, 256)
        self.freq_proj = nn.Linear(768, 256)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass through TF-TBN

        Args:
            x: Input tensor of shape (batch_size, total_features)
                First 512 features: temporal signal
                Remaining features: CWT coefficients (flattened)

        Returns:
            Output logits for classification
        """
        batch_size = x.size(0)
        half_length = 512  # Split between temporal and frequency features

        # ===== TEMPORAL BRANCH =====
        temporal_x = x[:, :half_length].unsqueeze(1)  # (batch_size, 1, 512)

        # Transformer processing
        temporal_x = self.transformer(temporal_x)  # (batch_size, 512, 512)

        # CNN processing
        temporal_features = self.temporal_cnn(temporal_x)  # (batch_size, 32*128)
        temporal_features = self.temporal_proj(temporal_features)  # (batch_size, 256)
        temporal_features = temporal_features.unsqueeze(1)  # (batch_size, 1, 256)

        # ===== FREQUENCY BRANCH =====
        freq_x = x[:, half_length:].reshape(batch_size, 1, 64, 512)

        # Resize for ViT and convert to 3 channels
        freq_x = nn.functional.interpolate(
            freq_x,
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        )
        freq_x = freq_x.repeat(1, 3, 1, 1)  # (batch_size, 3, 128, 128)

        # ViT processing
        vit_outputs = self.vit(pixel_values=freq_x)
        freq_features = vit_outputs.last_hidden_state[:, 0, :]  # CLS token
        freq_features = self.freq_proj(freq_features)  # (batch_size, 256)
        freq_features = freq_features.unsqueeze(1)  # (batch_size, 1, 256)

        # ===== FEATURE FUSION =====
        # Concatenate features for attention
        combined_features = torch.cat([temporal_features, freq_features], dim=1)  # (batch_size, 2, 256)

        # Apply multi-head attention
        attended_features, _ = self.fusion_attention(
            combined_features, combined_features, combined_features
        )

        # Pool attended features (mean pooling)
        fused_features = torch.mean(attended_features, dim=1)  # (batch_size, 256)

        # ===== CLASSIFICATION =====
        output = self.classifier(fused_features)

        return output

    def get_attention_weights(self, x):
        """
        Extract attention weights for model interpretability

        Args:
            x: Input tensor

        Returns:
            Dictionary containing attention weights from different components
        """
        attention_weights = {}
        batch_size = x.size(0)
        half_length = 512

        # Temporal branch attention (from Transformer)
        temporal_x = x[:, :half_length].unsqueeze(1)

        # Store transformer attention (simplified - actual implementation would need hooks)
        temporal_output = self.transformer(temporal_x)

        # Frequency branch attention (from ViT)
        freq_x = x[:, half_length:].reshape(batch_size, 1, 64, 512)
        freq_x = nn.functional.interpolate(freq_x, size=(128, 128), mode='bilinear')
        freq_x = freq_x.repeat(1, 3, 1, 1)

        # Get ViT attention (requires modifying ViT forward pass or using hooks)
        vit_outputs = self.vit(pixel_values=freq_x, output_attentions=True)
        attention_weights['vit_attention'] = vit_outputs.attentions

        return attention_weights
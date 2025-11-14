"""
CNN-LSTM Baseline Model for ABR-ASD Screening
A conventional baseline combining convolutional and recurrent layers.
"""

import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    """
    CNN-LSTM Baseline Model

    A traditional architecture combining:
    - CNN for local feature extraction
    - LSTM for temporal dependencies
    - Attention mechanism for feature weighting
    """

    def __init__(self, input_size=512, hidden_size=128, output_size=1, num_layers=2, dropout_rate=0.3):
        """
        Initialize CNN-LSTM model

        Args:
            input_size: Size of input features
            hidden_size: LSTM hidden size
            output_size: Output size (1 for binary classification)
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
        """
        super(CNN_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),

            # Second conv block
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),

            # Third conv block
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
        )

        # Calculate CNN output size
        cnn_output_size = self._get_cnn_output_size(input_size)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=128,  # CNN output channels
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, output_size)
        )

        # Initialize weights
        self._initialize_weights()

    def _get_cnn_output_size(self, input_size):
        """Calculate CNN output size for LSTM input dimension"""
        # Mock forward to calculate size
        x = torch.zeros(1, 1, input_size)
        x = self.cnn(x)
        return x.size(2)  # Sequence length after CNN

    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
                        # Set forget gate bias to 1
                        if len(param) == self.hidden_size * 8:  # LSTM bias size
                            param.data[self.hidden_size * 4:self.hidden_size * 5].fill_(1)

    def forward(self, x):
        """
        Forward pass through CNN-LSTM

        Args:
            x: Input tensor of shape (batch_size, total_features)
                Uses only temporal features (first 512)

        Returns:
            Output logits for classification
        """
        batch_size = x.size(0)

        # Use only temporal features (first 512)
        temporal_x = x[:, :512].unsqueeze(1)  # (batch_size, 1, 512)

        # CNN Feature Extraction
        cnn_features = self.cnn(temporal_x)  # (batch_size, 128, seq_len)
        cnn_features = cnn_features.permute(0, 2, 1)  # (batch_size, seq_len, 128)

        # LSTM Processing
        lstm_out, (hidden, cell) = self.lstm(cnn_features)  # (batch_size, seq_len, hidden_size*2)

        # Attention Mechanism
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1),
            dim=1
        )  # (batch_size, seq_len)

        # Apply attention weights
        attended_features = torch.sum(
            lstm_out * attention_weights.unsqueeze(-1),
            dim=1
        )  # (batch_size, hidden_size*2)

        # Classification
        output = self.classifier(attended_features)

        return output

    def get_attention_weights(self, x):
        """
        Extract attention weights for model interpretability

        Args:
            x: Input tensor

        Returns:
            Attention weights from the LSTM attention mechanism
        """
        batch_size = x.size(0)

        # Use only temporal features
        temporal_x = x[:, :512].unsqueeze(1)

        # Forward pass to get attention weights
        cnn_features = self.cnn(temporal_x)
        cnn_features = cnn_features.permute(0, 2, 1)

        lstm_out, _ = self.lstm(cnn_features)

        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1),
            dim=1
        )

        return attention_weights
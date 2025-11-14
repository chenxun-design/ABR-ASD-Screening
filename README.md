# ABR-ASD Screening

AI-powered Autism Spectrum Disorder screening using Auditory Brainstem Response signals.

## Overview

This repository contains the implementation of TF-TBN (Time-Frequency Transformer-Based Network), a novel deep learning model for ASD identification using full-band ABR signals. The model integrates temporal and frequency domain features through a dual-branch architecture combining Transformer networks and Vision Transformer.

## Key Features

- **TF-TBN Model**: Dual-branch time-frequency fusion network 
- **CNN-LSTM Baseline**: Traditional deep learning comparison
- **Data Processing**: ABR waveform extraction and feature engineering
- **Model Interpretation**: Explainable AI for clinical insights

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train both models
python main.py --model both

# Train only TF-TBN
python main.py --model tf_tbn

# Train only CNN-LSTM baseline  
python main.py --model cnn_lstm
```

## Citation
```bash
@article{
  title={AI-Powered ABR Test for Early ASD Screening},
  author={ },
  journal={iScience},
  year={2025}
}
```

## License
MIT License

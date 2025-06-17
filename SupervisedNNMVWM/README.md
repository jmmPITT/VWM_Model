# Supervised Neural Network for Visual Working Memory (SupervisedNNMVWM)

A supervised learning implementation of a visual working memory model using curriculum learning and transformer-based neural networks for change detection tasks.

## Overview

This project implements a supervised alternative to reinforcement learning approaches. The system generates synthetic data, trains a transformer-based neural network, and employs curriculum learning by progressively increasing task difficulty based on model performance.

## Architecture

### Core Components

- **VAE Feature Extractor** (`VAENet.py`): Variational Autoencoder that encodes visual patches into 128-dimensional feature vectors
- **Transformer Network** (`Network.py`): Custom transformer architecture with LSTM cells for sequence processing and change detection
- **Environment Simulator** (`MATenv.py`): Change detection environment that generates Gabor patch stimuli with configurable difficulty
- **Data Generator** (`GenData.py`): Creates training and testing datasets using the environment and VAE encoder
- **Training Pipeline** (`trainer.py`, `tester.py`): Supervised learning components for model training and evaluation

### Network Architecture Details

- **Input Dimensions**: 140 features (128 VAE embeddings + 4 quadrant indicators + 8 timestep encodings)
- **Transformer Block**: Single-head attention with custom LSTM cells
- **Memory Dimensions**: 1024-dimensional
- **Output**: Binary classification (change detected/not detected) 

## Curriculum Learning Process

The system implements automated curriculum learning through the following cycle:

1. **Data Generation**: Generate training and test datasets with current difficulty level (`Delta_max`)
2. **Model Training**: Train transformer network for specified epochs on generated data
3. **Performance Evaluation**: Test model accuracy on held-out test set
4. **Difficulty Adjustment**: If accuracy > 85%, decrease `Delta_max` by 3 degrees (making task harder)
5. **Iteration**: Repeat cycle up to 100 times

### Difficulty Progression

- **Initial Difficulty**: `Delta_max = 65` degrees (easier orientation changes)
- **Progression**: Decreases by 3 degrees when performance threshold is met
- **Final Difficulty**: Approaches more subtle orientation changes

## Task Description

### Change Detection Environment

The environment presents a sequence of visual stimuli across 7 timesteps:

- **t=0**: Blank screen
- **t=1**: Cue indicating attention location (left/right quadrant)
- **t=2**: Blank screen  
- **t=3-6**: Four Gabor patches in quadrants
- **t=5+**: Potential orientation change in one patch

### Stimuli Properties

- **Gabor Patches**: Oriented sinusoidal gratings with Gaussian envelopes
- **Quadrant Layout**: 2x2 grid of 25x25 pixel patches
- **Noise**: Gaussian noise added to orientations (σ = 5°)
- **Change Types**: Orientation shifts within ±`Delta_max` range

## Usage

### Training

```bash
python main.py
```

### Key Parameters

- `learning_rate`: 1e-6
- `num_epochs_train`: 200 per cycle
- `batch_size`: 128
- `n_games`: 5000 training + 5000 test samples per cycle
- `num_cycles`: 100 maximum cycles

### Model Checkpoints

- Models saved every 10 epochs during training
- Automatic loading of previous checkpoints between cycles
- Final trained models saved as `trained_model_cycle.pth`

## Dependencies

```python
torch
numpy
gymnasium
matplotlib
scipy
h5py
```

## File Structure

```
SupervisedNNMVWM/
├── main.py              # Main training loop with curriculum learning
├── Network.py           # Transformer architecture with custom LSTM cells
├── VAENet.py           # VAE feature extractor
├── GenData.py          # Data generation pipeline
├── MATenv.py           # Change detection environment
├── trainer.py          # Training class and dataset handler
├── tester.py           # Testing/evaluation class
├── vae_model.pth       # Pre-trained VAE weights
└── *.npy              # Generated training/testing data
```

## Key Features

- **Automated Curriculum Learning**: Progressive difficulty adjustment based on performance
- **Multi-modal Input**: Combines visual features, spatial encoding, and temporal information  
- **Custom Attention**: Transformer blocks with specialized LSTM memory cells
- **Robust Training**: Checkpoint saving, gradient clipping, and regularization
- **Comprehensive Evaluation**: Separate testing pipeline with accuracy metrics

location, change detection, and temporal sequence processing in a supervised learning framework.
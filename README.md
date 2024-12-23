# VWM Model

## Description
We present a neural network model of visual attention (NNMVA) that integrates biased competition and reinforcement learning to capture key aspects of attentional behavior. The model combines self-attention mechanisms from Vision Transformers (ViTs), Long Short-Term Memory (LSTM) networks for working memory, and an actor-critic reinforcement learning framework to map visual inputs to behavioral outputs. The self-attention mechanism simulates biased competition among artificial neural representations, determining their influence on working memory, which in turn provides top-down feedback to guide attention. Trained end-to-end with reinforcement learning using reward feedback—paralleling learning processes in non-human primates—the NNMVA replicates key behavioral signatures of attention, such as improved performance and faster reaction times for cued stimuli. Manipulating the model's attention mechanisms affects performance in ways analogous to experimental manipulations in primate frontal eye fields and superior colliculus. Additionally, artificially inducing attentional biases alters value estimates and temporal difference (TD) errors, offering predictions about how attention may interact with dopamine-related learning signals in the brain. Our findings suggest that reward-driven behavior alone can account for several key correlates of attention, providing a computational framework to explore the interplay between attention and reinforcement learning in both biological and artificial systems.

## Features
- Uses a ViT to process current visual imagery
- Uses xLSTM-like architecture to implement a visual working memory
- Recurrent states form the xLSTM interact with QKV components to shape visual attention

## Setup and Usage
1. Generate VAE training data by running 'TaskEnvGenData.py'
2. Train VAE by running 'VAEmain.py' 
3. Train Agent by running 'main.py'


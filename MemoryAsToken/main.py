import gymnasium
import numpy as np
import torch
import torch as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import h5py
from typing import Tuple, Dict

# Assuming these are your custom module imports
from OCDEnv import ChangeDetectionEnv
from agent_planner import Agent as AgentPlanner
from network_sensor2 import TransformerNetwork
from VAENet import VAE


def softmax(vector: np.ndarray) -> np.ndarray:
    """Computes the softmax of a vector for probability distribution."""
    # Subtract max for numerical stability
    exp_vec = np.exp(vector - np.max(vector))
    return exp_vec / exp_vec.sum()


def sigmoid(vector: np.ndarray) -> np.ndarray:
    """Computes the sigmoid function."""
    return 1 / (1 + np.exp(-vector))


def load_vae_model(model_path: str, model_class=VAE) -> VAE:
    """
    Initializes a VAE model and loads its state dictionary from a file.

    Args:
        model_path (str): The path to the saved model state dictionary.
        model_class: The VAE model class to instantiate.

    Returns:
        The loaded VAE model in evaluation mode.
    """
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def create_observation_from_state(
    state: np.ndarray,
    time_step: int,
    encoder: VAE,
    device: T.device,
    max_time: int,
    embedding_dim: int,
    pos_embedding_dim: int,
    time_embedding_dim: int,
) -> np.ndarray:
    """
    Constructs the observation tensor for a given state.

    Args:
        state: The current environment state.
        time_step: The current time step in the episode.
        encoder: The VAE model for generating embeddings.
        device: The PyTorch device to use for tensor operations.
        max_time: The maximum number of time steps in an episode.
        embedding_dim: The dimensionality of the VAE embeddings.
        pos_embedding_dim: The dimensionality of the positional embedding.
        time_embedding_dim: The dimensionality of the temporal embedding.

    Returns:
        The constructed observation tensor.
    """
    obs = np.zeros((max_time, 4, embedding_dim + pos_embedding_dim + time_embedding_dim))
    vae_input = np.zeros((4, 25, 25))
    vae_input[0, :, :] = state[0:25, 0:25]
    vae_input[1, :, :] = state[25:50, 0:25]
    vae_input[2, :, :] = state[0:25, 25:50]
    vae_input[3, :, :] = state[25:50, 25:50]

    noisy_sample = torch.tensor(vae_input, dtype=torch.float32).view(4, 25, 25, 1).to(device)
    embeddings = encoder(noisy_sample).detach().cpu().numpy()

    for i in range(4):
        obs[time_step, i, 0:embedding_dim] = embeddings[i, :]
        obs[time_step, i, embedding_dim + i] = 1  # Positional embedding
        obs[time_step, i, embedding_dim + pos_embedding_dim + time_step] = 1  # Temporal embedding

    return obs


def main():
    """Main training loop."""
    # --- Hyperparameters and Configuration ---
    T_END = 8
    VAE_EMBEDDING_DIM = 128
    POS_EMBEDDING_DIM = 4
    TIME_EMBEDDING_DIM = 8
    INPUT_DIM = (VAE_EMBEDDING_DIM + POS_EMBEDDING_DIM + TIME_EMBEDDING_DIM) * 4
    N_GAMES = 150000
    LEARNING_RATE_ALPHA = 0.0001
    LEARNING_RATE_BETA = 0.0001
    BATCH_SIZE = 64
    MAX_BUFFER_SIZE = 50000
    TAU = 0.01
    N_ACTIONS = 2
    TRANSFORMER_HIDDEN_DIM = 256
    TRANSFORMER_FC1_DIMS = 256
    TRANSFORMER_FC2_DIMS = 128
    SCORE_THRESHOLD = 0.85
    SCORE_HISTORY_LEN = 1000

    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Initialization ---
    env = ChangeDetectionEnv()
    agent_lever = AgentPlanner(
        alpha=LEARNING_RATE_ALPHA,
        beta=LEARNING_RATE_BETA,
        input_dims=INPUT_DIM,
        tau=TAU,
        batch_size=BATCH_SIZE,
        max_buffer_size=MAX_BUFFER_SIZE,
        n_actions=N_ACTIONS
    )

    transformer = TransformerNetwork(
        beta = LEARNING_RATE_BETA,
        input_dims=INPUT_DIM,
        hidden_dim=TRANSFORMER_HIDDEN_DIM,
        fc1_dims=TRANSFORMER_FC1_DIMS,
        fc2_dims=TRANSFORMER_FC2_DIMS,
        name='transformer1',
        chkpt_dir='td3_MAT'
    ).to(agent_lever.device)

    encoder = load_vae_model('vae_model.pth').to(device)
    score_history = []

    # --- Optional: Load saved models ---
    # agent_lever.load_models()
    # transformer.load_state_dict(T.load("transformer1_td3"))

    # --- Training Loop ---
    for i in range(N_GAMES):
        state = env.reset()
        obs = create_observation_from_state(
            state, env.t, encoder, device, T_END,
            VAE_EMBEDDING_DIM, POS_EMBEDDING_DIM, TIME_EMBEDDING_DIM
        )

        mask = np.ones((T_END, 4, 1))
        mask[env.t + 1:, :, :] = 0

        done = False
        score = 0
        lever_action_probs = np.zeros(N_ACTIONS)

        while not done:
            state_tensor = T.tensor(obs.reshape(1, -1), dtype=T.float).to(agent_lever.critic_1.device)
            mask_tensor = T.tensor(mask, dtype=T.float).to(agent_lever.critic_1.device)

            transformer_state = transformer(state_tensor, env.t, mask_tensor)
            transformer_state_np = transformer_state.detach().cpu().numpy().reshape(1, -1)

            lever_action_probs = agent_lever.choose_action(transformer_state_np, state_tensor)
            sampled_index = np.random.choice(len(lever_action_probs), p=lever_action_probs)

            lever_action_buffer = np.zeros(N_ACTIONS)
            lever_action_buffer[sampled_index] = 1

            next_state, reward_env, done, _ = env.step(sampled_index)
            score += reward_env

            obs_ = np.copy(obs)
            mask_ = np.copy(mask)

            if not done:
                obs_ = create_observation_from_state(
                    next_state, env.t, encoder, device, T_END,
                    VAE_EMBEDDING_DIM, POS_EMBEDDING_DIM, TIME_EMBEDDING_DIM
                )
                mask_ = np.ones((T_END, 4, 1))
                mask_[env.t + 1:, :, :] = 0

            agent_lever.remember(
                obs.reshape(1, -1), mask.reshape(1, -1), lever_action_buffer,
                reward_env, obs_.reshape(1, -1), mask_.reshape(1, -1), done,
                env.t, 0, 0  # cue_array and target_array are not defined in the original script
            )

            transformer = agent_lever.learn(transformer)

            obs = obs_
            mask = mask_

        score_history.append(score)
        avg_score = np.mean(score_history[-SCORE_HISTORY_LEN:])

        if (i + 1) % 100 == 0:
            agent_lever.save_models()
            transformer.save_checkpoint()
            print(f'Theta: {env.theta}')

        if avg_score > SCORE_THRESHOLD and len(score_history) > SCORE_HISTORY_LEN:
            env.theta -= 3
            score_history = []  # Reset history after difficulty increase
            print(f'Difficulty increased! New Theta: {env.theta}')

        print(f'Episode {i}: Score={score:.2f}, Avg Score (last {SCORE_HISTORY_LEN})={avg_score:.3f}, Actions={lever_action_probs}')


if __name__ == '__main__':
    main()
import gymnasium
import numpy as np
import torch
import torch as T
import torch.nn.functional as F
from typing import Tuple

# --- Local Module Imports ---
# Assuming these are your custom module and model files.
from OCDEnv import ChangeDetectionEnv
from agent_planner import Agent as AgentPlanner
from network_sensor2 import TransformerNetwork
from VAENet import VAE

# --- Helper Functions ---

def softmax(vector: np.ndarray) -> np.ndarray:
    """Computes the softmax of a vector for probability distribution."""
    # Subtract max for numerical stability.
    exp_vector = np.exp(vector - np.max(vector))
    return exp_vector / exp_vector.sum()

def sigmoid(vector: np.ndarray) -> np.ndarray:
    """Computes the sigmoid function."""
    return 1 / (1 + np.exp(-vector))

def load_model(model_path: str, model_class=VAE) -> VAE:
    """
    Initializes a model and loads its state dictionary from a file.

    Args:
        model_path (str): The path to the saved model state dictionary.
        model_class: The model class to instantiate (defaults to VAE).

    Returns:
        The loaded model in evaluation mode.
    """
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def create_observation_from_state(
    current_obs: np.ndarray,
    env_state: np.ndarray,
    time_step: int,
    encoder: VAE,
    device: torch.device,
    config: dict
) -> np.ndarray:
    """
    Constructs the observation tensor by encoding visual input and adding embeddings.

    Args:
        current_obs: The current observation tensor to be updated.
        env_state: The raw pixel state from the environment.
        time_step: The current time step in the episode.
        encoder: The VAE model used as a feature extractor.
        device: The PyTorch device for tensor operations.
        config: A dictionary containing model dimensions.

    Returns:
        The updated observation tensor with new embeddings for the current time step.
    """
    # Create 4 image patches from the environment state
    vae_input = np.zeros((4, 25, 25))
    vae_input[0, :, :] = env_state[0:25, 0:25]
    vae_input[1, :, :] = env_state[25:50, 0:25]
    vae_input[2, :, :] = env_state[0:25, 25:50]
    vae_input[3, :, :] = env_state[25:50, 25:50]

    # Get embeddings from the VAE encoder
    vae_input_tensor = torch.tensor(vae_input, dtype=torch.float32).view(4, 25, 25, 1).to(device)
    embeddings = encoder(vae_input_tensor).detach().cpu().numpy()

    # Populate the observation tensor for the current time step
    obs_for_timestep = current_obs
    for i in range(4):
        # 1. Add the VAE feature embedding (128 dims)
        obs_for_timestep[time_step, i, 0:config['VAE_EMBEDDING_DIM']] = embeddings[i, :]
        # 2. Add one-hot positional embedding (4 dims)
        obs_for_timestep[time_step, i, config['VAE_EMBEDDING_DIM'] + i] = 1
        # 3. Add one-hot temporal embedding (8 dims)
        obs_for_timestep[time_step, i, config['VAE_EMBEDDING_DIM'] + config['POS_EMBEDDING_DIM'] + time_step] = 1

    return obs_for_timestep


def main():
    """Main function to run the training loop."""
    # --- Configuration & Hyperparameters ---
    config = {
        'T_END': 8,
        'VAE_EMBEDDING_DIM': 128,
        'POS_EMBEDDING_DIM': 4,
        'TIME_EMBEDDING_DIM': 8,
        'N_GAMES': 150000,
        'AGENT_ALPHA': 0.0001,
        'AGENT_BETA': 0.0001,
        'AGENT_TAU': 0.01,
        'BATCH_SIZE': 64,
        'BUFFER_SIZE': 50000,
        'N_ACTIONS': 2,
        'TRANSFORMER_ALPHA': 0.00001,
        'TRANSFORMER_HIDDEN_DIM': 256,
        'TRANSFORMER_FC1_DIMS': 256,
        'TRANSFORMER_FC2_DIMS': 128,
        'SCORE_THRESHOLD': 0.85,
        'SCORE_HISTORY_LEN': 1000,
        'MODEL_CHKPT_DIR': 'td3_MAT'
    }
    # Calculate dimensions based on config
    token_dim = config['VAE_EMBEDDING_DIM'] + config['POS_EMBEDDING_DIM'] + config['TIME_EMBEDDING_DIM']
    input_dim = token_dim * 4  # Flattened dimension for the agent

    # --- Initialization ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env = ChangeDetectionEnv()
    agent_lever = AgentPlanner(
        alpha=config['AGENT_ALPHA'],
        beta=config['AGENT_BETA'],
        input_dims=input_dim,
        tau=config['AGENT_TAU'],
        batch_size=config['BATCH_SIZE'],
        max_size=config['BUFFER_SIZE'],
        n_actions=config['N_ACTIONS']
    )

    transformer = TransformerNetwork(
        beta=config['TRANSFORMER_ALPHA'],
        input_dims=input_dim,
        hidden_dim=config['TRANSFORMER_HIDDEN_DIM'],
        fc1_dims=config['TRANSFORMER_FC1_DIMS'],
        fc2_dims=config['TRANSFORMER_FC2_DIMS'],
        name='transformer1',
        chkpt_dir=config['MODEL_CHKPT_DIR']
    ).to(device)

    encoder = load_model('vae_model.pth').to(device)
    score_history = []

    # --- Training Loop ---
    for i in range(config['N_GAMES']):
        state = env.reset()
        obs = np.zeros((config['T_END'], 4, token_dim))
        obs = create_observation_from_state(obs, state, env.t, encoder, device, config)

        mask = np.ones((config['T_END'], 4, 1))
        mask[env.t + 1:, :, :] = 0

        done = False
        score = 0
        last_action_probs = np.zeros(config['N_ACTIONS'])

        while not done:
            # Reshape obs and mask for model input
            obs_flat_tensor = T.tensor(obs.reshape(1, -1), dtype=T.float).to(device)
            mask_tensor = T.tensor(mask, dtype=T.float).to(device)

            # Get processed state from transformer
            transformer_state = transformer(obs_flat_tensor, env.t, mask_tensor)
            transformer_state_np = transformer_state.detach().cpu().numpy().reshape(1, -1)

            # Choose and execute action
            action_probs = agent_lever.choose_action(transformer_state_np, obs_flat_tensor)
            sampled_index = np.random.choice(len(action_probs), p=action_probs)
            last_action_probs = action_probs

            action_buffer = np.zeros(config['N_ACTIONS'])
            action_buffer[sampled_index] = 1

            next_state, reward, done, _ = env.step(sampled_index)
            score += reward

            # Prepare for next iteration
            obs_ = np.copy(obs)
            mask_ = np.copy(mask)

            if not done:
                obs_ = create_observation_from_state(obs_, next_state, env.t, encoder, device, config)
                mask_ = np.ones((config['T_END'], 4, 1))
                mask_[env.t + 1:, :, :] = 0

            # Store experience and learn
            agent_lever.remember(
                obs.reshape(1, -1), mask.reshape(1, -1), action_buffer, reward,
                obs_.reshape(1, -1), mask_.reshape(1, -1), done, env.t, 0, 0
            )
            transformer = agent_lever.learn(transformer)

            obs = obs_
            mask = mask_

        score_history.append(score)
        avg_score = np.mean(score_history[-config['SCORE_HISTORY_LEN']:])

        # Save models periodically
        if (i + 1) % 100 == 0:
            agent_lever.save_models()
            transformer.save_checkpoint()
            print(f'\nTheta: {env.theta}')

        # Adjust difficulty based on performance
        if avg_score > config['SCORE_THRESHOLD'] and len(score_history) > config['SCORE_HISTORY_LEN']:
            env.theta -= 3
            score_history = []  # Reset history after difficulty increase
            print(f'Difficulty increased! New Theta: {env.theta}')

        print(f'Episode {i}: Score={score:.2f} | '
              f'Avg Score (last {config["SCORE_HISTORY_LEN"]})={avg_score:.3f} | '
              f'Actions={last_action_probs}')

if __name__ == '__main__':
    main()

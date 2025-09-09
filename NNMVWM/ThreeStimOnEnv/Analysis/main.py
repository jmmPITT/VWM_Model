#!/usr/bin/env python3

"""
-------------------------------------------------------------------------------
Cleaned-up code for training an RL agent with a VAE-based encoder and a 
Transformer-based model. The code uses a ChangeDetectionEnv environment and 
an AgentPlanner agent, combined with a TransformerNetwork for state encoding.
-------------------------------------------------------------------------------
"""

import numpy as np
import torch
import torch as T

# Local imports (make sure these files/modules are in your Python path)
from OCDEnv import ChangeDetectionEnv
from agent import Agent as AgentPlanner
from VWMNET import TransformerNetwork


import sys
import os

# Add the parent directory ('Dir') to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you can import from the VAE module
from VAENet import VAE

def softmax(vector: np.ndarray) -> np.ndarray:
    """
    A numerical stability-enhanced softmax for numpy arrays.
    Subtracts the max value in 'vector' prior to exponentiation 
    to reduce overflow issues.

    :param vector: 1D numpy array.
    :return: Softmax-normalized 1D numpy array.
    """
    e = np.exp(vector - np.max(vector))
    return e / e.sum()


def sigmoid(vector: np.ndarray) -> np.ndarray:
    """
    Sigmoid function over a 1D numpy array.

    :param vector: 1D numpy array.
    :return: Sigmoid-transformed 1D numpy array.
    """
    return 1 / (1 + np.exp(-vector))


def load_model(model_path: str, model_class=VAE) -> torch.nn.Module:
    """
    Loads a VAE model checkpoint from disk.

    :param model_path: Path to the saved model checkpoint (e.g. 'vae_model.pth').
    :param model_class: The model class used to instantiate an empty network 
                        before loading the parameters. Defaults to VAE.
    :return: The loaded and evaluated model (on CPU or GPU).
    """
    # Initialize the model
    model = model_class()
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path))
    # Set the model to evaluation mode
    model.eval()
    return model


if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Basic setup
    # --------------------------------------------------------------------------
    env = 0  # Placeholder; set to 0 before re-initializing with the environment
    T_end = 9  # Maximum length of each trial
    input_dim = (128 + 4 + 9) * 4  # Dimensions for the agent's input
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    # --------------------------------------------------------------------------
    # Agent and Models
    # --------------------------------------------------------------------------
    # Instantiate an AgentPlanner (TD3-like agent)
    agent = AgentPlanner(alpha=0.0001, beta=0.0001,
                         input_dims=input_dim, tau=0.01,
                         env=env, batch_size=64, max_size=50000,
                         n_actions=2)

    # Instantiate a TransformerNetwork that will encode the agent state
    Transformer = TransformerNetwork(beta=0.00001,
                                     input_dims=input_dim,
                                     hidden_dim=256,
                                     fc1_dims=256,
                                     fc2_dims=128,
                                     name='transformer1',
                                     chkpt_dir='td3_MAT').to(agent.device)

    # Load a trained VAE encoder from disk
    encoder = load_model('vae_model.pth').to(device)
    encoder.eval()

    # Number of training episodes
    n_games = 150000
    score_history = []

    # Create the environment
    env = ChangeDetectionEnv()

    # Optionally load agent and transformer states if you have checkpoints
    # agent.load_models()
    # Transformer.load_state_dict(T.load("transformer1_td3"))

    # Placeholder arrays for demonstration

    # --------------------------------------------------------------------------
    # Main training loop
    # --------------------------------------------------------------------------
    for i in range(n_games):

        # Reset the environment at the beginning of each episode
        state = env.reset()

        # ----------------------------------------------------------------------
        # Construct input to the VAE from the 50x50 environment state
        # We partition the image into four 25x25 patches.
        # ----------------------------------------------------------------------
        VAEinput = np.zeros((4, 25, 25))
        VAEinput[0, :, :] = state[0:25, 0:25]
        VAEinput[1, :, :] = state[25:50, 0:25]
        VAEinput[2, :, :] = state[0:25, 25:50]
        VAEinput[3, :, :] = state[25:50, 25:50]

        # Prepare a torch tensor for the VAE (add channel/batch dimensions)
        noisy_sample = torch.tensor(VAEinput, dtype=torch.float32).view(4, 25, 25, 1).to(device)
        embeddings = encoder(noisy_sample)
        embeddings = embeddings.detach().cpu().numpy()

        # ----------------------------------------------------------------------
        # Initialize the observation buffer for T_end = 8 time steps, 
        # with 4 possible patches and dimension (128 + 4 + 8) for each patch.
        #
        # The indexing in the final dimension is:
        #   [0..127] for the VAE embedding, 
        #   [128..131] for a positional embedding of size 4,
        #   [132..(132+T_end-1)] for a temporal embedding of size 8.
        # ----------------------------------------------------------------------
        obs = np.zeros((T_end, 4, 128 + 4 + 9))
        obs[env.t, 0, 0:128] = embeddings[0, :]
        obs[env.t, 0, 128]   = 1       # Positional embedding for patch 0
        obs[env.t, 0, 132 + env.t] = 1 # Temporal embedding

        obs[env.t, 1, 0:128] = embeddings[1, :]
        obs[env.t, 1, 129]   = 1       # Positional embedding for patch 1
        obs[env.t, 1, 132 + env.t] = 1 # Temporal embedding

        obs[env.t, 2, 0:128] = embeddings[2, :]
        obs[env.t, 2, 130]   = 1       # Positional embedding for patch 2
        obs[env.t, 2, 132 + env.t] = 1 # Temporal embedding

        obs[env.t, 3, 0:128] = embeddings[3, :]
        obs[env.t, 3, 131]   = 1       # Positional embedding for patch 3
        obs[env.t, 3, 132 + env.t] = 1 # Temporal embedding

        # ----------------------------------------------------------------------
        # Create masks that specify which time steps are valid/invalid 
        # for the agent to attend over.
        # ----------------------------------------------------------------------
        mask = np.ones((T_end, 4, 1))
        mask[env.t + 1:, :, :] = 0
        mask_ = np.ones((T_end, 4, 1))

        done = False
        score = 0

        # ----------------------------------------------------------------------
        # Episode loop
        # ----------------------------------------------------------------------
        while not done:

            # Combine obs and mask into agent input
            state_tensor = T.tensor(obs.reshape(1, -1), dtype=T.float).to(agent.critic_1.device)
            mask_tensor  = T.tensor(mask, dtype=T.float).to(agent.critic_1.device)

            # Use the transformer to encode (obs,mask) into a more compact representation
            transformer_state, _ = Transformer(state_tensor, mask_tensor)

            # Agent chooses an action (lever pressing) based on the transformer's output
            lever_action_prob = agent.choose_action(transformer_state.detach().cpu().numpy().reshape(1, -1))

            # Sample an action from the action probabilities
            sampled_index = np.random.choice(len(lever_action_prob), p=lever_action_prob)
            lever_action_buffer = np.zeros(2)
            lever_action_buffer[sampled_index] = 1

            # Step the environment forward
            next_state, reward_env, done, _ = env.step(sampled_index)

            # ------------------------------------------------------------------
            # Construct the next observation (obs_) if the episode continues
            # ------------------------------------------------------------------
            obs_ = np.copy(obs)

            if not done:
                # Prepare the next VAE input from the new environment state
                VAEinput = np.zeros((4, 25, 25))
                VAEinput[0, :, :] = next_state[0:25, 0:25]
                VAEinput[1, :, :] = next_state[25:50, 0:25]
                VAEinput[2, :, :] = next_state[0:25, 25:50]
                VAEinput[3, :, :] = next_state[25:50, 25:50]

                noisy_sample = torch.tensor(VAEinput, dtype=torch.float32).view(4, 25, 25, 1).to(device)
                embeddings = encoder(noisy_sample)
                embeddings = embeddings.detach().cpu().numpy()

                # Place new embeddings into obs_ at the current env.t
                obs_[env.t, 0, 0:128] = embeddings[0, :]
                obs_[env.t, 0, 128]   = 1
                obs_[env.t, 0, 132 + env.t] = 1

                obs_[env.t, 1, 0:128] = embeddings[1, :]
                obs_[env.t, 1, 129]   = 1
                obs_[env.t, 1, 132 + env.t] = 1

                obs_[env.t, 2, 0:128] = embeddings[2, :]
                obs_[env.t, 2, 130]   = 1
                obs_[env.t, 2, 132 + env.t] = 1

                obs_[env.t, 3, 0:128] = embeddings[3, :]
                obs_[env.t, 3, 131]   = 1
                obs_[env.t, 3, 132 + env.t] = 1

                # Update the next mask
                mask_ = np.ones((T_end, 4, 1))
                mask_[env.t + 1:, :, :] = 0

            # ------------------------------------------------------------------
            # Store the transition in the agent's replay buffer 
            # for off-policy learning
            # ------------------------------------------------------------------
            agent.remember(obs.reshape(1, -1),
                           mask.reshape(1, -1),
                           lever_action_buffer,
                           reward_env,
                           obs_.reshape(1, -1),
                           mask_.reshape(1, -1),
                           done,
                           env.t)

            # Let the agent learn (updates the agent's networks)
            Transformer = agent.learn(Transformer)

            # Accumulate environment reward
            score += reward_env

            # Move on to the next step
            obs = obs_
            mask = mask_

        # ----------------------------------------------------------------------
        # End of episode
        # ----------------------------------------------------------------------
        score_history.append(score)
        avg_score = np.mean(score_history[-1000:])

        # Save models and print progress every 100 episodes
        if i % 100 == 0:
            agent.save_models()
            Transformer.save_checkpoint()
            print('Theta!', env.theta)

        # If the model is doing really well, make the task harder
        if avg_score > 0.85 and len(score_history) > 1000:
            env.theta -= 3
            score_history = []
            print('New Theta!', env.theta)

        # Print progress each episode
        print(f"Episode {i}, Score: {score:.2f}, "
              f"Trailing 1000 games avg: {avg_score:.3f}, "
              f"Actions: {lever_action_prob}")

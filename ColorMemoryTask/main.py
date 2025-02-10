"""
Main script for running the Color Memory Task using the transformer-based agent.

This script sets up the environment, initializes the replay buffer, transformer network,
and lever (planner) agent, then runs a training loop for a specified number of games.
"""

from agent_planner import Agent as AgentPlanner
from network_sensor2 import TransformerNetwork  # Import only the required network
from buffer import ReplayBuffer
from ColorMatchingEnv import WorkingMemoryTask
import gymnasium as gym
import cv2
import torch as T
import numpy as np


if __name__ == '__main__':
    # Create the environment
    env = WorkingMemoryTask()  # or any other environment id

    # Set parameters
    buffer_size = 8
    T_end = 10  # Trial max length (used later in training)
    patch_length = 9
    w, h = 30, 30  # Patch width and height
    dff = 512  # Feature dimension for transformer internal states
    input_dim = (w * h * 3) * patch_length  # Flattened input dimension (9 patches of 30x30 RGB images)
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    # Initialize the lever (planner) agent
    agent_lever = AgentPlanner(
        alpha=0.00001,
        beta=0.00001,
        input_dims=input_dim,
        tau=1.0,
        env=env,
        batch_size=1,
        max_size=50,
        n_actions=5
    )

    # Initialize the transformer network
    Transformer = TransformerNetwork(
        0.000001,
        input_dims=input_dim,
        hidden_dim=256,
        fc1_dims=256,
        fc2_dims=128,
        name='transformer1',
        chkpt_dir='td3_MAT'
    ).to(agent_lever.device)

    n_games = 150000
    score_history = []

    # Load pretrained models and transformer checkpoint
    # agent_lever.load_models()
    # Transformer.load_state_dict(T.load("transformer1_td3"))

    l = 0
    while l < n_games:
        score = 0
        # Reset the environment and get the initial observation
        observation, _ = env.reset()
        print('observation', observation.shape)

        # Construct initial state with positional embeddings
        # The state is divided into 9 patches of size 30x30 with added 2D sine wave patterns.
        obs = np.zeros((patch_length, w, h, 3))
        c = 0
        for i in range(3):
            for j in range(3):
                # Generate a sine wave with frequency based on patch index
                freq_x = (i + 1) * 2 * np.pi / w
                freq_y = (j + 1) * 2 * np.pi / h
                x = np.linspace(0, freq_x, w)
                y = np.linspace(0, freq_y, h)
                xv, yv = np.meshgrid(y, x)
                sine_wave = np.sin(xv + yv)
                # Normalize the sine wave to [0, 1]
                sine_wave = (sine_wave - np.min(sine_wave)) / (np.max(sine_wave) - np.min(sine_wave))
                sine_wave = sine_wave[:, :, np.newaxis]  # Add channel dimension

                # Combine the environment observation patch with the sine wave embedding
                obs[c, :, :, :] = observation[i * w:(i + 1) * w, j * h:(j + 1) * h, :] + sine_wave / 1.0
                c += 1

        # Initialize transformer hidden states (for two layers)
        C1 = T.zeros(1, patch_length, dff).to(device)
        M1 = T.zeros(1, patch_length, dff).to(device)
        H1 = T.zeros(1, patch_length, dff).to(device)
        N1 = T.zeros(1, patch_length, dff).to(device)

        C2 = T.zeros(1, patch_length, dff).to(device)
        M2 = T.zeros(1, patch_length, dff).to(device)
        H2 = T.zeros(1, patch_length, dff).to(device)
        N2 = T.zeros(1, patch_length, dff).to(device)

        done = False

        # Main loop for the current episode
        while not done:
            # Reshape observation to a 1D vector for the transformer network
            state = T.tensor(obs.reshape(1, -1), dtype=T.float).to(agent_lever.critic_1.device)

            # Pass the state through the transformer to update hidden states
            C1, M1, H1, N1, C2, M2, H2, N2, Z = Transformer(
                state, C1, M1, H1, N1, C2, M2, H2, N2
            )

            # The lever agent chooses an action based on the transformed state output (using H2)
            lever_action, logits = agent_lever.choose_action(
                H2.detach().cpu().numpy().reshape(1, -1)
            )

            # Sample an action based on lever_action probabilities for the initial episodes;
            # later, use the argmax
            if l < 55:
                sampled_index = np.random.choice(len(lever_action), p=lever_action)
                lever_action_buffer = np.zeros(5)
                lever_action_buffer[sampled_index] = 1
            else:
                sampled_index = np.argmax(lever_action)
                lever_action_buffer = np.zeros(5)
                lever_action_buffer[sampled_index] = 1

            # Step the environment using the sampled action
            observation, reward, done, truncated, _ = env.step(sampled_index)

            # Construct the next observation with positional embeddings
            obs_ = np.copy(obs)
            if not done:
                c = 0
                for i in range(3):
                    for j in range(3):
                        freq_x = (i + 1) * 2 * np.pi / w
                        freq_y = (j + 1) * 2 * np.pi / h
                        x = np.linspace(0, freq_x, w)
                        y = np.linspace(0, freq_y, h)
                        xv, yv = np.meshgrid(y, x)
                        sine_wave = np.sin(xv + yv)
                        sine_wave = (sine_wave - np.min(sine_wave)) / (np.max(sine_wave) - np.min(sine_wave))
                        sine_wave = sine_wave[:, :, np.newaxis]
                        obs_[c, :, :, :] = observation[i * w:(i + 1) * w, j * h:(j + 1) * h, :] + sine_wave / 1.0
                        c += 1

            # Store the transition in the replay buffer.
            # The state and next state are flattened.
            agent_lever.remember(obs.reshape(1, -1), lever_action_buffer, reward, obs_.reshape(1, -1), done, l)
            # Increment memory counter for the current episode
            agent_lever.memory.mem_cntr[l] += 1

            score += reward
            obs = obs_

        # End of episode processing
        score_history.append(score)
        avg_score = np.mean(score_history[-1000:])
        print(f'episode {l} score {score:.2f} trailing 100 games avg {avg_score:.3f} actions {lever_action}')
        l += 1

        # After a fixed number of episodes equal to buffer_size, perform saving and training
        if l == buffer_size:
            print("Saving models and training...")
            agent_lever.save_models()
            Transformer.save_checkpoint()

            Transformer = agent_lever.learn(Transformer)
            agent_lever.memory = ReplayBuffer(50, input_dim, 5)
            l = 0

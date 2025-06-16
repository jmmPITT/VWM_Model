import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

# --- Local Module Imports ---
from network_sensor2 import ActorNetwork, CriticNetwork, TransformerNetwork
from buffer import ReplayBuffer


class Agent:
    """
    An agent that learns a policy using a distributional critic and a stochastic actor.

    This agent implements a sophisticated reinforcement learning strategy where the critic
    estimates a full distribution of possible returns instead of a single Q-value. This
    can lead to more stable and effective learning in complex environments. It is
    designed to work with a Transformer network that processes sequential observations.
    """

    def __init__(self,
                 alpha: float,
                 beta: float,
                 input_dims: int,
                 tau: float,
                 n_actions: int,
                 batch_size: int,
                 max_size: int,
                 gamma: float = 0.9,
                 update_actor_interval: int = 1,
                 warmup: int = 1000,
                 noise: float = 0.01):
        """
        Initializes the Agent.

        Args:
            alpha: Learning rate for the actor network.
            beta: Learning rate for the critic network.
            input_dims: Dimensionality of the flattened observation from the Transformer.
            tau: Soft update parameter for target networks.
            n_actions: The number of possible discrete actions.
            batch_size: The size of the batch for learning.
            max_size: The maximum size of the replay buffer.
            gamma: The discount factor for future rewards.
            update_actor_interval: The frequency of actor updates.
            warmup: Number of steps for random action exploration before learning.
            noise: Scale of the noise added to actions for exploration.
        """
        # --- Hyperparameters & Configuration ---
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.noise = noise
        self.eta = 1.0  # Coefficient for policy loss calculation

        # --- Device Setup ---
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # --- Replay Buffer ---
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        # --- Distributional RL Setup ---
        self.num_particles = 10
        self.v_max = 1.0  # Maximum possible return
        self.particles = torch.linspace(0, self.v_max, self.num_particles, device=self.device)
        self.spacing = self.particles[1] - self.particles[0]
        print(f"Value distribution particles: {self.particles.tolist()}")
        print(f"Particle spacing: {self.spacing.item()}")

        # --- Network Initializations ---
        # Note: The `input_dims` for actor/critic is based on the output of the Transformer,
        # which is different from the `input_dims` passed to this agent.
        # This is handled within the network classes themselves.
        self.actor = ActorNetwork(alpha, input_dims=input_dims, n_actions=n_actions, name='actor_planner',
                                  chkpt_dir='td3_MAT').to(self.device)
        self.critic_1 = CriticNetwork(beta, input_dims=input_dims, n_actions=n_actions, name='critic_1_planner',
                                      chkpt_dir='td3_MAT').to(self.device)

        self.target_actor = ActorNetwork(alpha, input_dims=input_dims, n_actions=n_actions, name='target_actor_planner',
                                         chkpt_dir='td3_MAT').to(self.device)
        self.target_critic_1 = CriticNetwork(beta, input_dims=input_dims, n_actions=n_actions,
                                             name='target_critic_1_planner', chkpt_dir='td3_MAT').to(self.device)

        # Initialize target networks with the same weights as the main networks
        self.update_network_parameters(tau=1)

    def choose_action(self, transformer_state: np.ndarray, obs: T.Tensor) -> np.ndarray:
        """
        Selects an action based on the current policy and exploration noise.

        Args:
            transformer_state: The state processed by the transformer network.
            obs: The raw observation tensor (used for compatibility with network forward pass).

        Returns:
            A numpy array representing the action probabilities.
        """
        if self.time_step < self.warmup:
            # Random action during warmup, with a bias towards the first action
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)), dtype=T.float).to(self.device)
            mu[0] += 2.0
            action_probs = F.softmax(mu, dim=-1)
        else:
            # Policy-based action
            self.actor.eval()
            with T.no_grad():
                state_tensor = T.tensor(transformer_state, dtype=T.float).to(self.device)
                action_probs = self.actor.forward(state_tensor, obs)[0]
            self.actor.train()

        self.time_step += 1
        return action_probs.cpu().detach().numpy()

    def remember(self, state, mask, action, reward, new_state, next_mask, done, t, cue, target):
        """Stores a transition in the replay buffer."""
        self.memory.store_transition(state, mask, action, reward, new_state, next_mask, done, t, cue, target)

    def _calculate_target_distribution(self, reward: T.Tensor, done: T.Tensor, q1_next: T.Tensor) -> T.Tensor:
        """Projects the next state's value distribution onto the current particles."""
        q1_next = q1_next.view(-1, self.num_particles)
        target_phat = torch.zeros_like(q1_next)

        with torch.no_grad():
            for j in range(self.num_particles):
                # Bellman update for each particle
                g_theta = reward + (1 - done.float()) * self.gamma * self.particles[j]

                # Distribute probability mass to neighboring particles
                for k in range(self.num_particles):
                    is_in_bin = (torch.abs(g_theta - self.particles[k]) <= self.spacing / 2.0).float()
                    target_phat[:, k] += q1_next[:, j] * is_in_bin

        # Normalize to ensure it's a valid probability distribution
        prob_sum = target_phat.sum(dim=1, keepdim=True)
        return target_phat / (prob_sum + 1e-8)

    def _calculate_policy_loss(self, transformer_state: T.Tensor, state: T.Tensor) -> T.Tensor:
        """Calculates the policy improvement loss (LPol)."""
        with torch.no_grad():
            # Create one-hot vectors for all possible actions
            actions_one_hot = torch.eye(self.n_actions, device=self.device)

            # Calculate Q-value distributions for each possible action
            q_dists = [self.target_critic_1(transformer_state, a.expand(self.batch_size, -1), state) for a in
                       actions_one_hot]

            # Calculate the expected Q-value for each action by taking the dot product with the particles
            q_expected = torch.stack([torch.sum(q_dist * self.particles, dim=1) for q_dist in q_dists], dim=1)

            # Get the target policy
            p_target = self.target_actor(transformer_state, state)

            # Use log-sum-exp for stable softmax-like calculation
            kq = T.log(torch.sum(p_target * T.exp(q_expected / self.eta), dim=1))

        # Get the current policy
        p_current = self.actor(transformer_state, state)

        # Calculate the policy loss using advantage weighting
        exp_advantage = T.exp(q_expected / self.eta - kq.unsqueeze(-1))
        weighted_log_probs = (p_target * exp_advantage * T.log(p_current + 1e-8)).sum(dim=1)

        return T.mean(weighted_log_probs)

    def learn(self, transformer: TransformerNetwork) -> TransformerNetwork:
        """
        Updates the actor and critic networks based on a batch of experiences.
        """
        if self.memory.mem_cntr < self.batch_size:
            return transformer

        # Sample from replay buffer
        state, mask, action, reward, new_state, next_mask, done, t, _, _, _, _ = \
            self.memory.sample_buffer(self.batch_size)

        # Convert numpy arrays to PyTorch tensors
        tensors = [reward, done, new_state, next_mask, state, mask, action]
        reward, done, state_, mask_, state, mask, action = \
            [T.tensor(data, dtype=T.float).to(self.device) for data in tensors]
        ts = T.tensor(t, dtype=T.int32).to(self.device)

        # --- Critic Loss Calculation ---
        # 1. Get the next state's value distribution from target networks
        with torch.no_grad():
            transformer_state_ = transformer(state_, ts, mask_)
            target_actions = self.target_actor(transformer_state_, state_)
            q1_next = self.target_critic_1(transformer_state_, target_actions, state_)
        # 2. Project it back to get the target distribution
        target_distribution = self._calculate_target_distribution(reward, done.to(torch.float), q1_next)

        # 3. Get the current state's value distribution
        transformer_state = transformer(state, ts - 1, mask)
        q1_dist = self.critic_1(transformer_state, action, state)
        log_q1_dist = torch.log(q1_dist + 1e-8)

        # 4. Calculate KL divergence between current and target distributions
        kl_loss = F.kl_div(log_q1_dist, target_distribution, reduction='batchmean')

        # --- Actor Loss & Entropy ---
        l_pol = self._calculate_policy_loss(transformer_state.detach(), state)
        p_actions = self.actor(transformer_state.detach(), state)
        entropy = -T.mean(T.sum(p_actions * T.log(p_actions + 1e-8), dim=-1))

        # --- Total Loss and Optimization ---
        # Combine losses: KL divergence for critic, policy gradient for actor, entropy for exploration
        combined_loss = 1.0 * kl_loss - 1.0 * l_pol - 0.02 * entropy

        # Zero gradients, backpropagate, and step optimizers
        transformer.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        self.critic_1.optimizer.zero_grad()
        combined_loss.backward()
        transformer.optimizer.step()
        self.actor.optimizer.step()
        self.critic_1.optimizer.step()

        self.learn_step_cntr += 1
        self.update_network_parameters()

        return transformer

    def update_network_parameters(self, tau: float = None):
        """
        Performs a soft update of the target network parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        if tau is None:
            tau = self.tau

        # Soft update actor
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        # Soft update critic
        for target_param, local_param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_models(self):
        """Saves the state of all networks."""
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.target_critic_1.save_checkpoint()

    def load_models(self):
        """Loads the state of all networks."""
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.target_critic_1.load_checkpoint()

import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

# Assuming these are your custom module imports
from network_sensor2 import ActorNetwork, CriticNetwork, TransformerNetwork
from buffer import ReplayBuffer


class Agent:
    """
    An agent that learns a policy using a distributional critic and a stochastic actor.
    It is designed to work with a Transformer network that processes sequential observations.
    """

    def __init__(self,
                 alpha: float,
                 beta: float,
                 input_dims: int,
                 tau: float,
                 n_actions: int,
                 gamma: float = 0.9,
                 max_buffer_size: int = 100000,
                 batch_size: int = 100,
                 update_actor_interval: int = 1,
                 warmup_steps: int = 1000,
                 noise: float = 0.01,
                 num_particles: int = 10,
                 actor_fc1_dims: int = 64,
                 actor_fc2_dims: int = 32,
                 actor_hidden_dims: int = 128,
                 critic_fc1_dims: int = 128,
                 critic_fc2_dims: int = 64,
                 critic_hidden_dims: int = 256
                 ):
        """
        Initializes the Agent.

        Args:
            alpha (float): Learning rate for the actor network.
            beta (float): Learning rate for the critic network.
            input_dims (int): Dimensionality of the input state.
            tau (float): Soft update parameter for target networks.
            n_actions (int): Number of possible actions.
            gamma (float): Discount factor for future rewards.
            max_buffer_size (int): Maximum size of the replay buffer.
            batch_size (int): Size of the batch for learning.
            update_actor_interval (int): Frequency of actor updates.
            warmup_steps (int): Number of steps for random action exploration.
            noise (float): Scale of the noise added to actions for exploration.
            num_particles (int): Number of particles for the value distribution.
        """
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.warmup_steps = warmup_steps
        self.noise = noise
        self.update_actor_iter = update_actor_interval
        self.time_step = 0
        self.learn_step_cntr = 0
        self.eta = 1.0  # Coefficient for policy loss calculation

        # Experience Replay Buffer
        self.memory = ReplayBuffer(max_buffer_size, input_dims, n_actions)

        # Device setup
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # Distributional RL setup
        self.num_particles = num_particles
        self.v_max = 1.0  # Maximum possible return
        self.particles = torch.linspace(0, self.v_max, self.num_particles, device=self.device)
        self.spacing = self.particles[1] - self.particles[0]
        print(f"Value distribution particles: {self.particles.tolist()}")
        print(f"Particle spacing: {self.spacing.item()}")

        # Network Initializations
        self.actor = ActorNetwork(alpha, input_dims, actor_hidden_dims, actor_fc1_dims, actor_fc2_dims, n_actions,
                                  'actor_planner', 'td3_MAT').to(self.device)
        self.critic_1 = CriticNetwork(beta, input_dims, critic_hidden_dims, critic_fc1_dims, critic_fc2_dims, n_actions,
                                      'critic_1_planner', 'td3_MAT').to(self.device)
        self.target_actor = ActorNetwork(alpha, input_dims, actor_hidden_dims, actor_fc1_dims, actor_fc2_dims,
                                         n_actions, 'target_actor_planner', 'td3_MAT').to(self.device)
        self.target_critic_1 = CriticNetwork(beta, input_dims, critic_hidden_dims, critic_fc1_dims, critic_fc2_dims,
                                             n_actions, 'target_critic_1_planner', 'td3_MAT').to(self.device)

        # Initialize target networks with the same weights as the main networks
        self.update_network_parameters(tau=1)

    def choose_action(self, transformer_state: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """
        Selects an action based on the current policy and exploration noise.

        Args:
            transformer_state: The state processed by the transformer.
            obs: The raw observation from the environment.

        Returns:
            The selected action probabilities.
        """
        if self.time_step < self.warmup_steps:
            # Random action during warmup
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)), dtype=T.float).to(self.device)
            mu[0] += 2.0  # Bias towards the first action
            action_probs = F.softmax(mu, dim=-1)
        else:
            # Policy-based action
            self.actor.eval()
            with T.no_grad():
                state_tensor = T.tensor(transformer_state, dtype=T.float).to(self.device)
                obs_tensor = T.tensor(obs, dtype=T.float).to(self.device)
                action_probs = self.actor.forward(state_tensor, obs_tensor)[0]
            self.actor.train()

        self.time_step += 1
        return action_probs.cpu().detach().numpy()

    def remember(self, state, mask, action, reward, new_state, next_mask, done, t, cue, target):
        """Stores a transition in the replay buffer."""
        self.memory.store_transition(state, mask, action, reward, new_state, next_mask, done, t, cue, target)

    def _calculate_target_distribution(self, reward: T.Tensor, done: T.Tensor, q1_next: T.Tensor) -> T.Tensor:
        """Projects the next state value distribution onto the current particles."""
        q1_next = q1_next.view(-1, self.num_particles)
        target_phat = torch.zeros_like(q1_next)

        with torch.no_grad():
            for j in range(self.num_particles):
                # Bellman update for each particle
                g_theta = reward + (1 - done.float()) * self.gamma * self.particles[j]

                # Distribute probability mass to neighboring particles
                for k in range(self.num_particles):
                    # Check if the updated value falls within the bin of the k-th particle
                    is_in_bin = (torch.abs(g_theta - self.particles[k]) <= self.spacing / 2.0).float()
                    target_phat[:, k] += q1_next[:, j] * is_in_bin

        # Normalize to ensure it's a valid probability distribution
        prob_sum = target_phat.sum(dim=1, keepdim=True)
        return target_phat / (prob_sum + 1e-8)

    def _calculate_policy_loss(self, transformer_state: T.Tensor, state: T.Tensor) -> T.Tensor:
        """Calculates the policy improvement loss (LPol)."""
        with torch.no_grad():
            actions_one_hot = torch.eye(self.n_actions, device=self.device)

            # Calculate Q-values for each discrete action
            q_values_dist = [self.target_critic_1(transformer_state, a.expand(self.batch_size, -1), state) for a in
                             actions_one_hot]

            # Calculate expected Q-values by summing over particles
            q_values_expected = torch.stack([torch.sum(q_dist * self.particles, dim=1) for q_dist in q_values_dist],
                                            dim=1)

            p_target = self.target_actor(transformer_state, state)

            # Log-sum-exp for stable softmax calculation
            kq = T.log(torch.sum(p_target * T.exp(q_values_expected / self.eta), dim=1))

        p_actions = self.actor(transformer_state, state)

        # Calculate policy loss using the advantage weighted by target policy
        exp_advantage = T.exp(q_values_expected / self.eta - kq.unsqueeze(-1))
        weighted_log_probs = (p_target * exp_advantage * T.log(p_actions + 1e-8)).sum(dim=1)

        return T.mean(weighted_log_probs)

    def learn(self, transformer: TransformerNetwork) -> TransformerNetwork:
        """
        Updates the actor and critic networks based on a batch of experiences.
        """
        if self.memory.mem_cntr < self.batch_size:
            return transformer

        state, mask, action, reward, new_state, next_mask, done, t, _, _, _, _ = \
            self.memory.sample_buffer(self.batch_size)

        # Convert numpy arrays to PyTorch tensors
        reward = T.tensor(reward, dtype=T.float).to(self.device)
        done = T.tensor(done, dtype=T.bool).to(self.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.device)
        mask_ = T.tensor(next_mask, dtype=T.float).to(self.device)
        state = T.tensor(state, dtype=T.float).to(self.device)
        mask = T.tensor(mask, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)
        ts = T.tensor(t, dtype=T.int32).to(self.device)

        # Set networks to training mode
        self.actor.train()
        self.critic_1.train()
        transformer.train()

        # --- Critic Loss Calculation ---
        with torch.no_grad():
            transformer_state_ = transformer(state_, ts, mask_)
            target_actions = self.target_actor(transformer_state_, state_)
            q1_next = self.target_critic_1(transformer_state_, target_actions, state_)

        target_distribution = self._calculate_target_distribution(reward, done, q1_next)

        transformer_state = transformer(state, ts - 1, mask)
        q1_dist = self.critic_1(transformer_state, action, state)
        log_q1_dist = torch.log(q1_dist + 1e-8)

        kl_loss = F.kl_div(log_q1_dist, target_distribution, reduction='batchmean')

        # --- Policy Loss Calculation ---
        l_pol = self._calculate_policy_loss(transformer_state.detach(), state)

        # --- Entropy Calculation ---
        p_actions = self.actor(transformer_state.detach(), state)
        entropy = -T.mean(T.sum(p_actions * T.log(p_actions + 1e-8), dim=-1))

        # --- Total Loss and Optimization ---
        # Combine losses: KL divergence for critic, policy gradient for actor, entropy for exploration
        critic_loss = 1.0 * kl_loss - 1.0 * l_pol - 0.02 * entropy

        transformer.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        self.critic_1.optimizer.zero_grad()

        critic_loss.backward()

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

        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

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
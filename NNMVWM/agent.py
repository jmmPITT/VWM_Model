#!/usr/bin/env python3

"""
-------------------------------------------------------------------------------
Agent class implementing a distributional reinforcement learning approach,
inspired by the paper:
"Offline Actor-Critic Reinforcement Learning Scales to Large Models."

The agent trains an Actor-Critic setup with a distributional Bellman update.
The distribution is discretized into particles, and the Critic outputs
a probability mass over these particles.

Author: <Your Name / Organization>
-------------------------------------------------------------------------------
"""

import os
import torch as T
import torch.nn.functional as F
import numpy as np

# Local modules (ensure these are in your Python path)
from VWMNET import ActorNetwork, CriticNetwork, TransformerNetwork
from buffer import ReplayBuffer


class Agent:
    """
    The Agent class handles:
      1. Storage of transitions in a replay buffer (PER).
      2. Selection of actions via an Actor network.
      3. Critic updates via distributional RL.
      4. Target networks and soft parameter updates.
      5. Optionally integrates a Transformer state encoding.

    :param alpha: Learning rate for the actor.
    :param beta: Learning rate for the critic.
    :param input_dims: Dimension of input state (flattened obs or encoded).
    :param tau: Polyak averaging coefficient for target network updates.
    :param env: Environment placeholder or reference (not used directly here).
    :param gamma: Discount factor.
    :param update_actor_interval: Update frequency for actor relative to critic.
    :param warmup: Number of timesteps to use random actions before learning.
    :param n_actions: Number of discrete actions.
    :param max_size: Maximum size of replay buffer.
    :param layer1_size: Not directly used here.
    :param layer2_size: Not directly used here.
    :param batch_size: Mini-batch size for learning.
    :param noise: Standard deviation of noise added to actions (exploration).
    """

    def __init__(
            self,
            alpha,
            beta,
            input_dims,
            tau,
            env,
            gamma=0.9,
            update_actor_interval=1,
            warmup=1000,
            n_actions=4,
            max_size=100000,
            layer1_size=64,
            layer2_size=32,
            batch_size=100,
            noise=0.01
    ):
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.update_actor_iter = update_actor_interval
        self.warmup = warmup
        self.noise = noise

        # Replay buffer (Prioritized Experience Replay)
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        # Device setup
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # Distributional RL setup
        self.num_particles = 10      # Number of particles for distribution
        self.V_max = 1               # Maximum possible return
        self.particles = [
            i * (self.V_max / (self.num_particles - 1))
            for i in range(self.num_particles)
        ]
        print("Particles:", self.particles)

        self.spacing = self.particles[1] - self.particles[0]
        print('Particle spacing:', self.spacing)
        self.eta = 1.0               # Temperature-like parameter

        # Actor-Critic networks
        self.actor = ActorNetwork(alpha, input_dims=input_dims,
                                  hidden_dim=128, fc1_dims=64, fc2_dims=32,
                                  n_actions=n_actions,
                                  name='actor_planner', chkpt_dir='td3_MAT'
                                  ).to(self.device)

        self.critic_1 = CriticNetwork(beta, input_dims=input_dims,
                                      hidden_dim=256, fc1_dims=128,
                                      fc2_dims=64, n_actions=n_actions,
                                      name='critic_1_planner',
                                      chkpt_dir='td3_MAT').to(self.device)

        # Target networks
        self.target_actor = ActorNetwork(alpha, input_dims=input_dims,
                                         hidden_dim=128, fc1_dims=64,
                                         fc2_dims=32, n_actions=n_actions,
                                         name='target_actor_planner',
                                         chkpt_dir='td3_MAT').to(self.device)

        self.target_critic_1 = CriticNetwork(beta, input_dims=input_dims,
                                             hidden_dim=256, fc1_dims=128,
                                             fc2_dims=64, n_actions=n_actions,
                                             name='target_critic_1_planner',
                                             chkpt_dir='td3_MAT').to(self.device)

        # Initialize target networks to match current networks
        self.update_network_parameters(tau=1.0)

    def choose_action(self, transformer_state, obs):
        """
        Choose an action given the current state encoding from the Transformer
        (optional) and the raw observation or additional embedding.

        :param transformer_state: Output from Transformer (or direct observation).
        :param obs: Observation or additional state data.
        :return: Numpy array of action probabilities (softmax).
        """
        if self.time_step < self.warmup * 0:
            # Random exploration (currently set with warmup * 0, effectively off)
            print(self.time_step, self.warmup * 1)
            mu = T.tensor(np.random.normal(scale=self.noise,
                                           size=(self.n_actions,)),
                          dtype=T.float).to(self.device)
            mu[0] += 2.0
            mu = F.softmax(mu, dim=-1)
        else:
            # Actor forward pass
            transformer_state = T.tensor(transformer_state,
                                         dtype=T.float).to(self.actor.device)
            obs = T.tensor(obs, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(transformer_state, obs)[0]

        self.time_step += 1
        return mu.cpu().detach().numpy()

    def remember(self, state, mask, action, reward, new_state, next_mask, done, t):
        """
        Store a transition in the replay buffer.

        :param state: Current state.
        :param mask: Current mask (for partial obs or sequence).
        :param action: Action taken.
        :param reward: Reward received.
        :param new_state: Next state.
        :param next_mask: Next mask (for partial obs or sequence).
        :param done: Episode done flag.
        :param t: Current timestep or index in the episode.
        """
        self.memory.store_transition(
            state, mask, action, reward, new_state, next_mask, done, t
        )

    def learn(self, Transformer):
        """
        Perform one learning step (Critic and possibly Actor update):
          1. Sample from replay buffer.
          2. Compute target distribution over next states (distributional RL).
          3. Compute Critic loss as KL divergence + distributional constraints.
          4. Compute policy loss from distribution (softmax exponentiated Q).
          5. Update networks.

        :param Transformer: A TransformerNetwork instance used for state enc.
        :return: Updated Transformer model (because it's also trained here).
        """
        # Check if we have enough samples
        if self.memory.mem_cntr < self.batch_size:
            return Transformer

        # Sample transitions from the buffer
        (state, mask, action, reward,
         new_state, next_mask, done, t,
         indices, weights) = self.memory.sample_buffer(self.batch_size)

        # Move data to device
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        mask_ = T.tensor(next_mask, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        mask = T.tensor(mask, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        ts = T.tensor(t, dtype=T.int32).to(self.critic_1.device)

        # Current Q with the main critic
        transformer_state = Transformer(state, ts - 1, mask)

        # Compute target distribution
        with T.no_grad():
            # Next-state transformer
            transformer_state_ = Transformer(state_, ts, mask_)
            target_actions = self.target_actor.forward(transformer_state_, state_)
            q1_ = self.target_critic_1.forward(transformer_state_, target_actions, state_)

        # Evaluate current Q distribution for the chosen actions
        q1 = self.critic_1.forward(transformer_state, action, state)

        # Reshape for distribution: [batch_size, num_particles]
        q1_ = q1_.view(-1, self.num_particles)

        phat = T.zeros(q1_.size(0), self.num_particles, device=q1_.device)
        for j in range(self.num_particles):
            # Future return for particle j
            gTheta = reward + self.particles[j] * (1 - done.float()) * self.gamma
            # Compare to all other particles k
            for k in range(self.num_particles):
                condition = (T.abs(gTheta - self.particles[k]) - 0.0001 <= self.spacing / 2.0)
                phat[:, k] += q1_[:, j] * condition.float()

        target_distribution = phat
        # Sum of probabilities for debugging (not used further, but keep for reference)
        prob_sum = target_distribution.sum(dim=1, keepdim=True)

        # Convert current Q to log-prob for KL
        log_q_prob = q1.log()

        # Policy loss computation
        with T.no_grad():
            # Evaluate Q-values for action=0 and action=1
            a0 = T.zeros(q1_.size(0), 2, device=q1_.device)
            a0[:, 0] = 1.0
            a1 = T.zeros(q1_.size(0), 2, device=q1_.device)
            a1[:, 1] = 1.0

            Q0_ = self.target_critic_1.forward(transformer_state, a0, state)
            Q1_ = self.target_critic_1.forward(transformer_state, a1, state)

            Q0 = T.zeros(q1_.size(0), 1, device=q1_.device)
            Q1 = T.zeros(q1_.size(0), 1, device=q1_.device)

            # Integrate distribution to get scalar Q-values
            for j in range(self.num_particles):
                Q0[:, 0] += Q0_[:, j] * self.particles[j]
                Q1[:, 0] += Q1_[:, j] * self.particles[j]

            # Actor outputs for current state
            p = self.target_actor.forward(transformer_state, state)
            pa0_ = p[:, 0]
            pa1_ = p[:, 1]

            # Log-sum-exp trick for distributional RL
            KQ = T.log(pa0_ * T.exp(Q0[:, 0] / self.eta) +
                       pa1_ * T.exp(Q1[:, 0] / self.eta))

        action_probs = self.actor.forward(transformer_state, state)
        pa0 = action_probs[:, 0]
        pa1 = action_probs[:, 1]

        # Policy gradient term (based on exponentiated Q)
        LPol = T.mean(
            pa0_ * (T.exp(Q0[:, 0] / self.eta - KQ) * T.log(pa0)) +
            pa1_ * (T.exp(Q1[:, 0] / self.eta - KQ) * T.log(pa1))
        )

        # Entropy bonus
        entropy_per_sample = -T.sum(
            action_probs * T.log(action_probs + 1e-6),
            dim=-1
        )
        H = T.mean(entropy_per_sample)

        # Final loss: Critic's KL + distribution constraint - policy gradient - entropy
        Transformer.optimizer.zero_grad()
        self.critic_1.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()

        kl_loss = F.kl_div(log_q_prob, target_distribution, reduction='batchmean')
        critic_loss = 1.0 * kl_loss - 1.0 * LPol - 0.02 * H

        critic_loss.backward()
        Transformer.optimizer.step()
        self.actor.optimizer.step()
        self.critic_1.optimizer.step()

        self.learn_step_cntr += 1
        self.update_network_parameters()
        return Transformer

    def update_network_parameters(self, tau=None):
        """
        Soft-update (Polyak averaging) of target networks.

        :param tau: If None, use self.tau. If =1, copy weights directly.
        """
        if tau is None:
            tau = self.tau

        # Pull network parameter dictionaries
        actor_params = dict(self.actor.named_parameters())
        critic_1_params = dict(self.critic_1.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        target_critic_1_params = dict(self.target_critic_1.named_parameters())

        # Update critic_1 target
        for name in critic_1_params:
            critic_1_params[name] = tau * critic_1_params[name].clone() + \
                (1 - tau) * target_critic_1_params[name].clone()

        # Update actor target
        for name in actor_params:
            actor_params[name] = tau * actor_params[name].clone() + \
                (1 - tau) * target_actor_params[name].clone()

        self.target_critic_1.load_state_dict(critic_1_params)
        self.target_actor.load_state_dict(actor_params)

    def save_models(self):
        """
        Save checkpoints for actor, target_actor, critic_1, target_critic_1.
        """
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.target_critic_1.save_checkpoint()

    def load_models(self):
        """
        Load checkpoints for actor, target_actor, critic_1, target_critic_1.
        """
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.target_critic_1.load_checkpoint()

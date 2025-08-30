#!/usr/bin/env python3

"""
-------------------------------------------------------------------------------
Agent class implementing a distributional reinforcement learning approach,
inspired by the paper:
"Offline Actor-Critic Reinforcement Learning Scales to Large Models."

The agent trains an Actor-Critic setup with a distributional Bellman update.
The distribution is discretized into particles, and the Critic outputs
a probability mass over these particles.

-------------------------------------------------------------------------------
"""

import torch as T
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Local modules (ensure these are in your Python path)
from common.network_sensor2 import ActorNetwork, CriticNetwork, TransformerNetwork
from common.buffer import ReplayBuffer
from common.agent_planner import Agent as BaseAgent


class Agent(BaseAgent):
    """
    The Agent class handles:
      1. Storage of transitions in a replay buffer (PER).
      2. Selection of actions via an Actor network.
      3. Critic updates via distributional RL.
      4. Target networks and soft parameter updates.
      5. Optionally integrates a Transformer state encoding.
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
            noise=0.01,
            num_particles: int = 10,
            actor_fc1_dims: int = 64,
            actor_fc2_dims: int = 32,
            actor_hidden_dims: int = 128,
            critic_fc1_dims: int = 128,
            critic_fc2_dims: int = 64,
            critic_hidden_dims: int = 256
    ):
        super().__init__(
            alpha=alpha,
            beta=beta,
            input_dims=input_dims,
            tau=tau,
            n_actions=n_actions,
            gamma=gamma,
            max_buffer_size=max_size,
            batch_size=batch_size,
            update_actor_interval=update_actor_interval,
            warmup_steps=warmup,
            noise=noise,
            num_particles=num_particles,
            actor_fc1_dims=actor_fc1_dims,
            actor_fc2_dims=actor_fc2_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_fc1_dims=critic_fc1_dims,
            critic_fc2_dims=critic_fc2_dims,
            critic_hidden_dims=critic_hidden_dims
        )
        # The env parameter is not used in the base class, so we don't pass it.
        # The layer1_size and layer2_size parameters are not used in the base class.

    def choose_action(self, transformer_state, obs):
        """
        Choose an action given the current state encoding from the Transformer
        (optional) and the raw observation or additional embedding.
        """
        if self.time_step < self.warmup_steps * 0:
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

    def learn(self, Transformer):
        """
        Perform one learning step (Critic and possibly Actor update):
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

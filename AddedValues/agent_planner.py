import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

# --- Common Module Imports ---
from common.network_sensor2 import ActorNetwork, CriticNetwork, TransformerNetwork
from common.buffer import ReplayBuffer
from common.agent_planner import Agent as BaseAgent


class Agent(BaseAgent):
    """
    An agent that learns a policy using a distributional critic and a stochastic actor.
    This agent is specific to the AddedValues experiment.
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
        """
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
        )

    def choose_action(self, transformer_state: np.ndarray, obs: T.Tensor) -> np.ndarray:
        """
        Selects an action based on the current policy and exploration noise.
        """
        if self.time_step < self.warmup_steps:
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

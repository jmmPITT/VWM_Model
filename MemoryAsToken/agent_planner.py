import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

# Assuming these are your custom module imports
from common.network_sensor2 import ActorNetwork, CriticNetwork, TransformerNetwork
from common.buffer import ReplayBuffer


from common.agent_planner import Agent as BaseAgent


class Agent(BaseAgent):
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
        """
        super().__init__(
            alpha=alpha,
            beta=beta,
            input_dims=input_dims,
            tau=tau,
            n_actions=n_actions,
            gamma=gamma,
            max_buffer_size=max_buffer_size,
            batch_size=batch_size,
            update_actor_interval=update_actor_interval,
            warmup_steps=warmup_steps,
            noise=noise,
            num_particles=num_particles,
            actor_fc1_dims=actor_fc1_dims,
            actor_fc2_dims=actor_fc2_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_fc1_dims=critic_fc1_dims,
            critic_fc2_dims=critic_fc2_dims,
            critic_hidden_dims=critic_hidden_dims
        )
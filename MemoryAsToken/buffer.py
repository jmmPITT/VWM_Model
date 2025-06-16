import numpy as np
from typing import Tuple

class ReplayBuffer:
    """
    A Prioritized Experience Replay (PER) buffer for reinforcement learning agents.

    This buffer stores transitions and samples them based on their TD-error,
    allowing the agent to learn more frequently from significant experiences.
    """
    def __init__(self, max_size: int, input_shape: int, n_actions: int, alpha: float = 0.7):
        """
        Initializes the Replay Buffer.

        Args:
            max_size (int): The maximum number of transitions to store in the buffer.
            input_shape (int): The dimensionality of the state observation.
            n_actions (int): The number of possible actions in the action space.
            alpha (float): The prioritization exponent (0 for no prioritization, 1 for full).
                           Determines how much TD-error influences sampling probability.
        """
        # --- Buffer Configuration ---
        self.mem_size = max_size
        self.mem_cntr = 0
        self.alpha = alpha

        # --- Memory Arrays ---
        # Note: The dimensions are derived from your original implementation.
        # state and new_state have a shape of (mem_size, sequence_length * input_shape_per_token)
        # mask and next_mask have a shape of (mem_size, sequence_length * patch_length * mask_channels)
        self.state_memory = np.zeros((self.mem_size, input_shape * 8))
        self.mask_memory = np.zeros((self.mem_size, 8 * 4 * 1))

        self.new_state_memory = np.zeros((self.mem_size, input_shape * 8))
        self.next_mask_memory = np.zeros((self.mem_size, 8 * 4 * 1))

        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        # --- Auxiliary Information ---
        self.time_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.cue_memory = np.zeros((self.mem_size, 4))
        self.target_memory = np.zeros((self.mem_size, 5))

        # --- Prioritization ---
        self.priority_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state: np.ndarray, mask: np.ndarray, action: np.ndarray,
                         reward: float, new_state: np.ndarray, new_mask: np.ndarray,
                         done: bool, time_step: int, cue: np.ndarray, target: np.ndarray):
        """
        Stores a new experience transition in the buffer and assigns it max priority.

        Args:
            state: The current state observation.
            mask: The mask for the current state.
            action: The action taken.
            reward: The reward received.
            new_state: The resulting state observation.
            new_mask: The mask for the new state.
            done: A boolean indicating if the episode has terminated.
            time_step: The time step within the episode.
            cue: The cue information for the transition.
            target: The target information for the transition.
        """
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.mask_memory[index] = mask
        self.new_state_memory[index] = new_state
        self.next_mask_memory[index] = new_mask
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.time_memory[index] = time_step
        self.cue_memory[index] = cue
        self.target_memory[index] = target

        # New experiences are given the highest priority to ensure they are sampled at least once.
        max_priority = np.max(self.priority_memory) if self.mem_cntr > 0 else 1.0
        self.priority_memory[index] = max_priority

        self.mem_cntr += 1

    def _calculate_priority(self, td_error: float) -> float:
        """Calculates the priority value from a TD-error."""
        return (np.abs(td_error) + 1e-5) ** self.alpha

    def sample_buffer(self, batch_size: int, beta: float = 0.4) -> Tuple[np.ndarray, ...]:
        """
        Samples a batch of experiences from the buffer using prioritized sampling.

        Args:
            batch_size (int): The number of transitions to sample.
            beta (float): The importance-sampling exponent. It anneals from an initial
                          value to 1.0 to reduce the bias from prioritized sampling.

        Returns:
            A tuple containing the sampled transitions, their indices, and the
            importance-sampling weights.
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        if max_mem == 0:
            return (None,) * 12 # Return empty tuple if buffer is empty

        # Get priorities and calculate sampling probabilities
        priorities = self.priority_memory[:max_mem]
        probabilities = priorities / np.sum(priorities)

        # Sample indices based on the calculated probability distribution
        indices = np.random.choice(max_mem, batch_size, p=probabilities)

        # --- Importance-Sampling Weights Calculation ---
        # Weights are used to correct for the bias introduced by non-uniform sampling.
        total_samples = len(self.priority_memory)
        weights = (total_samples * probabilities[indices]) ** -beta
        # Normalize weights by the maximum weight for stability
        weights /= np.max(weights)

        # Retrieve the sampled transitions
        states = self.state_memory[indices]
        masks = self.mask_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        new_states = self.new_state_memory[indices]
        new_masks = self.next_mask_memory[indices]
        dones = self.terminal_memory[indices]
        time_steps = self.time_memory[indices]
        cues = self.cue_memory[indices]
        targets = self.target_memory[indices]

        return (states, masks, actions, rewards, new_states, new_masks, dones,
                time_steps, cues, targets, indices, weights)

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        """
        Updates the priorities of sampled transitions based on their new TD-errors.

        Args:
            indices (np.ndarray): The indices of the transitions to update.
            errors (np.ndarray): The new TD-errors for the corresponding transitions.
        """
        errors = np.squeeze(errors)
        for i, error in zip(indices, errors):
            self.priority_memory[i] = self._calculate_priority(error)

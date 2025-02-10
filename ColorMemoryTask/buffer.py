import numpy as np


class ReplayBuffer:
    """
    A simple replay buffer for storing transitions.

    The buffer stores experiences for a fixed number of episodes (buffer_size) and each
    episode can store up to mem_size transitions. Transitions include state, action,
    reward, next state, and terminal flag.

    Attributes:
        mem_size (int): Maximum number of transitions per episode.
        buffer_size (int): Number of episodes maintained in the buffer.
        mem_cntr (np.ndarray): Array tracking the number of transitions stored per episode.
        alpha (float): Priority exponent used to compute priority (if needed).
        state_memory (np.ndarray): Buffer for states.
        next_state_memory (np.ndarray): Buffer for next states.
        action_memory (np.ndarray): Buffer for actions.
        reward_memory (np.ndarray): Buffer for rewards.
        terminal_memory (np.ndarray): Buffer for terminal flags.
    """

    def __init__(self, max_size: int, input_shape: int, n_actions: int, alpha: float = 0.7) -> None:
        # Initialize basic buffer properties
        self.mem_size = max_size  # Maximum size of the buffer per episode
        self.buffer_size = 8  # Number of episodes stored
        self.mem_cntr = np.zeros(self.buffer_size, dtype=int)  # Transition counter per episode
        self.alpha = alpha  # Priority exponent for potential prioritized replay

        # Initialize memory arrays for states, actions, rewards, next states, and terminal flags.
        self.state_memory = np.zeros((self.buffer_size, self.mem_size, input_shape))
        self.next_state_memory = np.zeros((self.buffer_size, self.mem_size, input_shape))
        self.action_memory = np.zeros((self.buffer_size, self.mem_size, n_actions))
        self.reward_memory = np.zeros((self.buffer_size, self.mem_size))
        self.terminal_memory = np.zeros((self.buffer_size, self.mem_size), dtype=bool)

    def store_transition(self, state, action, reward, next_state, done, l: int) -> None:
        """
        Store a transition in the replay buffer.

        Parameters:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: Terminal flag (True if the episode ended after this transition).
            l (int): The index of the episode in which to store the transition.
        """
        # Determine the index in the episode where the transition should be stored.
        index = self.mem_cntr[l] % self.mem_size
        # Store the transition components in their respective memory arrays.
        self.state_memory[l, index] = state
        self.action_memory[l, index] = action
        self.reward_memory[l, index] = reward
        self.next_state_memory[l, index] = next_state
        self.terminal_memory[l, index] = done

    def get_priority(self, td_error) -> float:
        """
        Calculate the priority of a transition based on its TD error.

        A small constant is added to the absolute TD error to avoid zero priority,
        then the result is raised to the power of alpha.

        Parameters:
            td_error: The temporal-difference error for the transition.

        Returns:
            float: The computed priority.
        """
        return (np.abs(td_error) + 1e-5) ** self.alpha

    def sample_buffer(self, batch_size, beta: float = 0.2):
        """
        Sample a batch of transitions from the buffer.

        This implementation returns all stored transitions up to the current maximum
        memory count for each episode.

        Parameters:
            batch_size: The desired batch size (unused in this basic implementation).
            beta (float, optional): Importance-sampling parameter (unused here).

        Returns:
            tuple: (states, actions, rewards, next_states, dones) sampled from the buffer.
        """
        # Determine the maximum number of transitions stored in any episode.
        max_mem = max(self.mem_cntr[:])

        # Extract transitions from the memory arrays up to max_mem.
        states = self.state_memory[:, :max_mem]
        next_states = self.next_state_memory[:, :max_mem]
        actions = self.action_memory[:, :max_mem]
        rewards = self.reward_memory[:, :max_mem]
        dones = self.terminal_memory[:, :max_mem]

        return states, actions, rewards, next_states, dones

    # Optional: A method to update priorities based on new TD errors can be implemented here.
    # def update_priorities(self, indices, errors):
    #     errors = np.squeeze(errors)
    #     for i, error in zip(indices, errors):
    #         self.priority_memory[i] = self.get_priority(error)

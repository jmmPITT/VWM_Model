import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, alpha=0.7):
        # Initialize basic buffer properties
        self.mem_size = max_size  # Maximum size of the buffer
        self.mem_cntr = 0  # Memory counter to keep track of the number of saved experiences
        self.alpha = alpha  # Priority exponent, determines how much prioritization is used

        # Initialize memory arrays for the state, action, reward, new state, and terminal status
        self.state_memory = np.zeros((self.mem_size, input_shape*9))
        self.mask_memory = np.zeros((self.mem_size, 1*9*4))

        self.new_state_memory = np.zeros((self.mem_size, input_shape*9))
        self.next_mask_memory = np.zeros((self.mem_size, 1*9*4))

        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.time = np.zeros(self.mem_size, dtype=np.int32)

        # Initialize priority memory to store the priorities of each experience
        self.priority_memory = np.zeros(self.mem_size)

    def store_transition(self, state, mask, action, reward, state_, mask_, done, t):
        # Determine the index where the new transition will be stored
        index = self.mem_cntr % self.mem_size

        # Store the experience in the respective memory arrays
        self.state_memory[index] = state
        self.mask_memory[index] = mask
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.next_mask_memory[index] = mask_
        self.terminal_memory[index] = done
        self.time[index] = t


        # Assign high priority for the new transition
        # Use the max priority in the buffer if not the first transition, else set to 1.0
        self.priority_memory[index] = self.priority_memory.max() if self.mem_cntr > 0 else 1.0

        # Increment memory counter
        self.mem_cntr += 1

    def get_priority(self, td_error):
        # Calculate the priority of an experience
        # Use absolute TD error with a small constant to avoid zero priority
        return (np.abs(td_error) + 1e-5) ** self.alpha

    def sample_buffer(self, batch_size, beta=0.2):
        # Calculate the number of experiences available in the buffer
        max_mem = min(self.mem_cntr, self.mem_size)

        # Extract the priorities and compute the probability distribution for sampling
        priorities = self.priority_memory[:max_mem]
        probabilities = priorities / priorities.sum()

        # Sample indices based on the calculated probabilities
        indices = np.random.choice(max_mem, batch_size, p=probabilities)

        # Calculate importance-sampling weights to correct bias introduced by prioritization
        total_prob = len(self.priority_memory) * probabilities[indices]
        weights = (total_prob ** -beta) / max(total_prob ** -beta)

        # Extract sampled experiences based on the selected indices
        states = self.state_memory[indices]
        masks = self.mask_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        states_ = self.new_state_memory[indices]
        masks_ = self.next_mask_memory[indices]
        dones = self.terminal_memory[indices]
        ts = self.time[indices]

        # Return the sampled experiences along with the corresponding indices and importance-sampling weights
        return states, masks, actions, rewards, states_, masks_, dones, ts, indices, weights

    def update_priorities(self, indices, errors):
        # Flatten or squeeze the errors array to make it one-dimensional
        errors = np.squeeze(errors)
        # Update priorities of the experiences based on new TD errors
        for i, error in zip(indices, errors):
            # Update the priority for each experience
            self.priority_memory[i] = self.get_priority(error)
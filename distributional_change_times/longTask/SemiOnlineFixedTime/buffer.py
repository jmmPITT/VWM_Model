
import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, alpha=0.7):
        # Initialize basic buffer properties
        self.mem_size = max_size  # Maximum size of the buffer
        self.buffer_size = 32
        self.mem_cntr = np.zeros(self.buffer_size, dtype=int)  # Memory counter to keep track of the number of saved experiences
        self.alpha = alpha  # Priority exponent, determines how much prioritization is used

        # Initialize memory arrays for the state, action, reward, new state, and terminal status
        self.state_memory = np.zeros((self.buffer_size, self.mem_size, input_shape))
        self.next_state_memory = np.zeros((self.buffer_size, self.mem_size, input_shape))
        self.action_memory = np.zeros((self.buffer_size, self.mem_size, n_actions))
        self.reward_memory = np.zeros((self.buffer_size, self.mem_size))
        self.terminal_memory = np.zeros((self.buffer_size, self.mem_size), dtype=bool)



    def store_transition(self, state, action, reward, next_state, done, l):
        # Determine the index where the new transition will be stored
        index = self.mem_cntr[l] % self.mem_size
        # print('index',index)
        # Store the experience in the respective memory arrays
        self.state_memory[l,index] = state
        # print('action you fucking piece of shit!!!',action.shape)
        self.action_memory[l,index] = action
        self.reward_memory[l,index] = reward
        self.next_state_memory[l,index] = next_state
        self.terminal_memory[l,index] = done


    def get_priority(self, td_error):
        # Calculate the priority of an experience
        # Use absolute TD error with a small constant to avoid zero priority
        return (np.abs(td_error) + 1e-5) ** self.alpha

    def sample_buffer(self, batch_size, beta=0.2):
        # Calculate the number of experiences available in the buffer
        max_mem = max(self.mem_cntr[:])


        # Extract sampled experiences based on the selected indices
        states = self.state_memory[:, 0:max_mem]
        next_states = self.next_state_memory[:, 0:max_mem]
        actions = self.action_memory[:, 0:max_mem]
        rewards = self.reward_memory[:, 0:max_mem]
        dones = self.terminal_memory[:, 0:max_mem]



        # Return the sampled experiences along with the corresponding indices and importance-sampling weights
        return states, actions, rewards, next_states, dones

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# import matplotlib.pyplot as plt


class WorkingMemoryTask(gym.Env):
    """
    A custom Gymnasium environment where an RL agent interacts with a working memory task.

    The task is divided into several epochs:
      - stimulus: Two stimuli (s1 and s2) are presented on different screen positions.
      - memory: A delay period during which the stimuli are no longer visible.
      - probe: A probe is presented at the center.
      - decision: The agent makes a decision by fixating on one of the stimuli locations.

    The agent receives rewards based on the correctness of the decision.
    """

    def __init__(self) -> None:
        super().__init__()
        # Define epoch durations (in timesteps)
        self.stim_time: int = 5
        self.mem_time: int = np.random.randint(5, 21)  # Random delay duration between 5 and 20 timesteps
        self.probe_time: int = 5
        self.decision_time: int = 5
        self.max_time: int = self.stim_time + self.mem_time + self.probe_time + self.decision_time

        # Define action space: 0 = fixate, 1-4 correspond to choices based on stimulus positions
        self.action_space: spaces.Discrete = spaces.Discrete(5)

        # Define observation space as a 90x90 RGB image with values in [0, 1]
        self.observation_space: spaces.Box = spaces.Box(low=0, high=1, shape=(90, 90, 3), dtype=np.float32)

        # Optional rendering attributes (e.g., for matplotlib)
        self.fig = None
        self.ax = None
        self.im = None

        # Track the most recent action for display purposes
        self.current_action = None

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """
        Reset the environment to an initial state and returns an initial observation.

        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options (unused).

        Returns:
            observation (np.ndarray): The initial observation.
            info (dict): Additional info (empty in this case).
        """
        super().reset(seed=seed)
        self.time: int = 0
        # Reset epoch durations; note that mem_time is randomized at each reset
        self.stim_time = 5
        self.mem_time = np.random.randint(5, 21)
        self.probe_time = 5
        self.decision_time = 5
        self.max_time = self.stim_time + self.mem_time + self.probe_time + self.decision_time

        # Start with the stimulus epoch and generate new stimuli
        self.current_epoch: str = 'stimulus'
        self.s1, self.s2, self.probe, self.correct_choice = self._generate_stimuli()
        self.current_action = None

        observation = self._get_observation()
        return observation, {}

    def step(self, action: int) -> tuple:
        """
        Process the given action and update the environment state.

        Args:
            action (int): The action taken by the agent.

        Returns:
            observation (np.ndarray): The new observation.
            reward (float): The reward obtained by taking the action.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional info (empty in this case).
        """
        reward = 0.0
        done = False
        truncated = False

        self.current_action = action

        # If we are not at the final timestep
        if self.time < self.max_time - 1:
            # If the agent breaks fixation outside the decision epoch, end the episode with no reward
            if action != 0 and self.current_epoch != 'decision':
                done = True
                reward = 0.0
            # If the agent chooses an action during the decision epoch, evaluate the choice
            elif action != 0 and self.current_epoch == 'decision':
                done = True
                reward = 5.0 if action == self.correct_choice else 1.0
            else:
                # Continue the episode if the agent is fixating (action == 0)
                self.time += 1
                self._update_epoch()
        else:
            # Final timestep (decision epoch)
            reward = 5.0 if action == self.correct_choice else 1.0
            done = True

        observation = self._get_observation()
        return observation, reward, done, truncated, {}

    def _get_observation(self) -> np.ndarray:
        """
        Generate the current observation image based on the current epoch.

        Returns:
            image (np.ndarray): A 90x90x3 image representing the current state.
        """
        # Start with a white background image
        image = np.ones((90, 90, 3), dtype=np.float32)

        if self.current_epoch == 'stimulus':
            # Display s1 on the left: top-left if pos1 == 1, else bottom-left.
            if self.pos1 == 1:
                image[0:30, 0:30] = self.s1
            else:
                image[60:90, 0:30] = self.s1

            # Display s2 on the right: top-right if pos2 == 3, else bottom-right.
            if self.pos2 == 3:
                image[0:30, 60:90] = self.s2
            else:
                image[60:90, 60:90] = self.s2

        elif self.current_epoch == 'probe':
            # Display the probe in the center of the image.
            image[30:60, 30:60] = self.probe

        elif self.current_epoch == 'decision':
            # Draw crosshair reticules at four key positions
            self._draw_crosshair(image, 15, 15)  # Top-left
            self._draw_crosshair(image, 75, 75)  # Bottom-right
            self._draw_crosshair(image, 15, 75)  # Top-right
            self._draw_crosshair(image, 75, 15)  # Bottom-left

            # Highlight selected region based on the agent's action with a blue box
            if self.current_action == 0:
                image[30:60, 30:60] = [0, 0, 1]  # Center
            elif self.current_action == 1:
                # Top-left: Only update white background pixels to blue
                image[0:30, 0:30] = np.where(image[0:30, 0:30] == 1, 0, [0, 0, 1])
            elif self.current_action == 2:
                # Bottom-right
                image[60:90, 60:90] = np.where(image[60:90, 60:90] == 1, 0, [0, 0, 1])
            elif self.current_action == 3:
                # Top-right
                image[0:30, 60:90] = np.where(image[0:30, 60:90] == 1, 0, [0, 0, 1])
            elif self.current_action == 4:
                # Bottom-left
                image[60:90, 0:30] = np.where(image[60:90, 0:30] == 1, 0, [0, 0, 1])

        return image

    def _draw_crosshair(self, image: np.ndarray, x: int, y: int, size: int = 10) -> None:
        """
        Draws a crosshair (vertical and horizontal lines) centered at (x, y) on the image.

        Args:
            image (np.ndarray): The image on which to draw.
            x (int): X-coordinate of the crosshair center.
            y (int): Y-coordinate of the crosshair center.
            size (int, optional): Half the length of each line (default is 10).
        """
        # Draw vertical line
        image[y - size:y + size + 1, x] = [0, 0, 0]
        # Draw horizontal line
        image[y, x - size:x + size + 1] = [0, 0, 0]

    def _generate_stimuli(self) -> tuple:
        """
        Generate stimuli images (s1, s2) and a probe for the task, as well as determine the correct choice.

        Returns:
            s1 (np.ndarray): Stimulus 1 image.
            s2 (np.ndarray): Stimulus 2 image.
            probe (np.ndarray): Probe image.
            correct_choice (int): The correct action corresponding to the stimulus that is closer to the probe.
        """
        # Generate s1: biased towards more red pixels
        s1 = np.zeros((30, 30, 3), dtype=np.float32)
        s1_red_prob = self.np_random.uniform(0.01, 0.99)  # Random bias for red pixels in s1
        s1_mask = self.np_random.random((30, 30)) < s1_red_prob
        s1[s1_mask] = [1, 0, 0]  # Red pixels
        s1[~s1_mask] = [0, 1, 0]  # Green pixels

        # Generate s2: biased towards more green pixels (but still randomized)
        s2 = np.zeros((30, 30, 3), dtype=np.float32)
        s2_red_prob = self.np_random.uniform(0.01, 0.99)
        s2_mask = self.np_random.random((30, 30)) < s2_red_prob
        s2[s2_mask] = [1, 0, 0]  # Red pixels
        s2[~s2_mask] = [0, 1, 0]  # Green pixels

        # Randomly assign positions for s1 and s2
        self.pos1 = 1 if np.random.rand() < 0.5 else 2
        self.pos2 = 3 if np.random.rand() < 0.5 else 4

        # Generate probe image with a random red probability
        probe = np.zeros((30, 30, 3), dtype=np.float32)
        probe_red_prob = self.np_random.uniform(0.01, 0.99)
        probe_mask = self.np_random.random((30, 30)) < probe_red_prob
        probe[probe_mask] = [1, 0, 0]  # Red
        probe[~probe_mask] = [0, 1, 0]  # Green

        # Define an inner helper function to calculate the "distance" between stimuli
        def calculate_distance(stim1: np.ndarray, stim2: np.ndarray) -> float:
            red_diff = np.sum(stim1[:, :, 0]) - np.sum(stim2[:, :, 0])
            green_diff = np.sum(stim1[:, :, 1]) - np.sum(stim2[:, :, 1])
            return np.abs(red_diff) + np.abs(green_diff)

        # Calculate distances from the probe to s1 and s2
        dist_to_s1 = calculate_distance(probe, s1)
        dist_to_s2 = calculate_distance(probe, s2)

        # Determine the correct choice based on which stimulus is closer to the probe
        correct_choice = self.pos1 if dist_to_s1 < dist_to_s2 else self.pos2

        return s1, s2, probe, correct_choice

    def _update_epoch(self) -> None:
        """
        Update the current epoch based on the elapsed time.
        """
        if self.time < self.stim_time:
            self.current_epoch = 'stimulus'
        elif self.time < self.stim_time + self.mem_time:
            self.current_epoch = 'memory'
        elif self.time < self.stim_time + self.mem_time + self.probe_time:
            self.current_epoch = 'probe'
        elif self.time < self.stim_time + self.mem_time + self.probe_time + self.decision_time:
            self.current_epoch = 'decision'

    # Optional rendering functions (commented out) for visualization using matplotlib.
    # Uncomment and adjust as needed.

    # def render(self):
    #     """
    #     Render the current observation using matplotlib.
    #     """
    #     if self.fig is None:
    #         self.fig, self.ax = plt.subplots()
    #         self.im = self.ax.imshow(self._get_observation())
    #         self.ax.axis('off')
    #     else:
    #         self.im.set_data(self._get_observation())
    #
    #     self.ax.set_title(
    #         f'Epoch: {self.current_epoch.capitalize()}, Time: {self.time}, Action: {self.current_action}')
    #     self.fig.canvas.draw()
    #     self.fig.canvas.flush_events()
    #     plt.pause(0.1)
    #
    # def close(self):
    #     """
    #     Close the matplotlib figure if it exists.
    #     """
    #     if self.fig:
    #         plt.close(self.fig)
    #         self.fig = None
    #         self.ax = None
    #         self.im = None

# Example testing code (commented out). Uncomment to run a demonstration of the environment.
# if __name__ == '__main__':
#     env = WorkingMemoryTask()
#
#     for episode in range(1):  # Run a single episode for demonstration
#         observation, _ = env.reset()
#         done = False
#
#         while not done:
#             env.render()  # Uncomment if using the render function
#
#             # During decision epoch, choose a stimulus (for demonstration, this picks pos1)
#             if env.current_epoch == 'decision':
#                 action = env.pos1
#             else:
#                 action = 0  # Maintain fixation during non-decision epochs
#
#             observation, reward, done, truncated, _ = env.step(action)
#             print("Observation shape:", observation.shape)
#
#         env.render()  # Render the final state
#         # plt.pause(1)  # Pause to show the final state if rendering
#         print(f"Episode finished. Reward: {reward}")
#
#     env.close()

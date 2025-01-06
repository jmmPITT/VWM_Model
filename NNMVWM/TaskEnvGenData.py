import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import h5py

class ChangeDetectionEnv(gym.Env):
    def __init__(self):
        super(ChangeDetectionEnv, self).__init__()

        self.action_space = spaces.Discrete(2)  # Actions: 0 (no change detected), 1 (change detected)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(50, 50), dtype=np.float32)

        self.T = 7  # Max timesteps
        self.change_time = 5  # Timestep at which the orientation change occurs
        self.t = 0
        self.orientations = [np.random.uniform(0, 180), np.random.uniform(0, 180), np.random.uniform(0, 180),
                             np.random.uniform(0, 180)]  # Initial orientations of the Gabor patches
        # self.change_index = np.random.randint(4)  # Randomly select which Gabor filter will change
        # self.orientation_change = np.random.randint(10, 30)  # Random orientation change between 10 and 30 degrees
        self.cue_position = None  # 'left' or 'right', determined at the start of each episode

    def reset(self):
        self.t = 0
        self.orientations = [np.random.uniform(0, 180), np.random.uniform(0, 180), np.random.uniform(0, 180),
                             np.random.uniform(0, 180)]  # Initial orientations of the Gabor patches
        self.change_true = np.random.randint(2)
        # self.orientation_change = np.random.randint(1, 70)  # Random orientation change between 10 and 30 degrees
        self.orientation_change = np.random.uniform(0, 35)  # Random orientation change between 0 and 30 degrees

        self.cue_position = 'left' if np.random.rand() < 0.5 else 'right'  # Randomly determine cue position
        # Determine the proportion of the ring to display
        self.proportions = [1.0, 0.75, 0.5, 0.25]
        self.proportion = np.random.choice(self.proportions)

        rand = np.random.rand()
        if self.change_true == 1:
            if self.cue_position == 'left':
                if rand < self.proportion:
                    self.change_index = np.random.randint(2)  # Randomly select new Gabor filter for change
                else:
                    self.change_index = np.random.randint(2) + 2
            elif self.cue_position == 'right':
                if rand < self.proportion:
                    self.change_index = np.random.randint(2) + 2  # Randomly select new Gabor filter for change
                else:
                    self.change_index = np.random.randint(2)

        # self.change_index=1
        # self.cue_position = 'left'
        # self.proportion = 1
        # self.orientation_change = 10
        # self.change_true = 1

        return self._next_observation()

    def _next_observation(self):
        """
        Generate the observation with four Gabor patches, one of which might change orientation.
        """
        # Create a blank canvas
        observation = np.zeros((50, 50))
        observationClean = np.zeros((50, 50))

        # Blank screen for t=0 and t=2
        if self.t in [0, 2]:
            return np.zeros((50, 50)), np.zeros((50, 50))

        # Cue at t=1
        elif self.t == 1:
            return self._generate_cue(), self._generate_cue()

        # Gabor filters for t>=3
        else:
            # Generate four Gabor noisy patches and clean patches
            gabor1,gabor1Clean = self._generate_gabor(self.orientations[0])  # This is the one that will change
            gabor2,gabor2Clean = self._generate_gabor(self.orientations[1])
            gabor3,gabor3Clean = self._generate_gabor(self.orientations[2])
            gabor4,gabor4Clean = self._generate_gabor(self.orientations[3])

            if self.t >= self.change_time and self.change_true == 1:
                if self.change_index == 0:
                    gabor1,gabor1Clean = self._generate_gabor(
                        self.orientations[0] + self.orientation_change)  # This is the one that will change
                elif self.change_index == 1:
                    gabor2,gabor2Clean = self._generate_gabor(
                        self.orientations[1] + self.orientation_change)  # This is the one that will change
                elif self.change_index == 2:
                    gabor3,gabor3Clean = self._generate_gabor(
                        self.orientations[2] + self.orientation_change)  # This is the one that will change
                elif self.change_index == 3:
                    gabor4,gabor4Clean = self._generate_gabor(
                        self.orientations[3] + self.orientation_change)  # This is the one that will change

            # Place the Gabor patches on the canvas
            observation[0:25, 0:25] = gabor1
            observation[0:25, 25:50] = gabor3
            observation[25:50, 0:25] = gabor2
            observation[25:50, 25:50] = gabor4

            observationClean[0:25, 0:25] = gabor1Clean
            observationClean[0:25, 25:50] = gabor3Clean
            observationClean[25:50, 0:25] = gabor2Clean
            observationClean[25:50, 25:50] = gabor4Clean

        return observation, observationClean

    def _generate_cue(self):
        """
        Generate an observation with a cue (a disc surrounded by a ring) on the specified side of the screen.
        The ring will circle the disc in proportions (100%, 75%, 50%, 25%) with uniform probability.
        """
        observation = np.zeros((50, 50))
        cue = np.zeros((25, 25))  # Adjust cue size to fit in a quadrant

        # Adjust the grid for a 25x25 area
        cy, cx = np.ogrid[-12.5:12.5, -12.5:12.5]  # Adjust to match the size of the cue
        disc_radius = 8  # Adjust disc radius to fit within the quadrant
        disc_mask = cx ** 2 + cy ** 2 <= disc_radius ** 2  # Disc mask
        cue[disc_mask] = 1

        # Adjust outer and inner radii for the ring to fit within the quadrant
        ring_outer_radius = 12  # Adjusted outer radius
        ring_inner_radius = 10  # Adjusted inner radius for the ring thickness
        ring_mask = (cx ** 2 + cy ** 2 <= ring_outer_radius ** 2) & (cx ** 2 + cy ** 2 >= ring_inner_radius ** 2)

        # Proportions and angle removal remain the same
        # proportions = [1.0, 0.75, 0.5, 0.25]
        # self.proportion = np.random.choice(proportions)  # Randomly choose a proportion
        angle_to_remove = 2 * np.pi * (1 - self.proportion)
        theta = np.arctan2(cy, cx) + np.pi  # Shift theta range
        ring_mask &= ~(theta < angle_to_remove)  # Remove a section of the ring
        cue[ring_mask] = 1  # Apply the ring mask to the cue

        # Place the cue in the correct quadrant
        if self.cue_position == 'left':
            observation[0:25, 0:25] = cue  # Top-left quadrant
        else:  # 'right'
            observation[25:50, 25:50] = cue  # Bottom-right quadrant

        return observation

    def _generate_gabor(self, orientation):
        """
        Generate a Gabor patch with the specified orientation, adding random noise only to the Gabor patch itself,
        and then swapping neighboring pixels in a vectorized manner.
        """
        x, y = np.meshgrid(np.linspace(-1, 1, 25), np.linspace(-1, 1, 25))
        d = np.sqrt(x * x + y * y)
        sigma, theta, Lambda, psi, gamma = 0.5, np.deg2rad(orientation), 0.3, 0, 1
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        gabor = np.exp(-.5 * (x_theta ** 2 + y_theta ** 2 / gamma ** 2) / sigma ** 2) * np.cos(
            2 * np.pi * x_theta / Lambda + psi)
        gaborClean = np.copy(gabor)

        # Generate random noise
        noise = np.random.uniform(-0.051, 0.051, size=gabor.shape)
        #
        # Swap neighboring pixels
        gabor_with_swapped_neighbors = self.swap_neighbors_vectorized(gabor)

        #
        # Apply circular mask to Gabor patch and noise
        gaborClean[d > 0.5] = 0  # Apply circular mask to Gabor patch
        gabor_with_swapped_neighbors[d > 0.5] = 0  # Apply circular mask to Gabor patch
        noise[d > 0.5] = 0  # Apply the same mask to noise
        #
        # # Add noise to the Gabor patch
        gabor_with_swapped_neighbors = gabor_with_swapped_neighbors + noise

        return gabor_with_swapped_neighbors, gaborClean

    import numpy as np

    def swap_neighbors_vectorized(self, patch):
        # Create shifted versions of the patch
        shift_up = np.roll(patch, -1, axis=0)
        shift_down = np.roll(patch, 1, axis=0)
        shift_left = np.roll(patch, -1, axis=1)
        shift_right = np.roll(patch, 1, axis=1)

        # Create a random mask for each direction
        mask_up = np.random.rand(*patch.shape) < 0.081
        mask_down = np.random.rand(*patch.shape) < 0.081
        mask_left = np.random.rand(*patch.shape) < 0.081
        mask_right = np.random.rand(*patch.shape) < 0.081

        # Initialize the result with the original patch
        result = np.copy(patch)

        # Apply the shifts based on the masks
        result[mask_up] = shift_up[mask_up]
        result[mask_down] = shift_down[mask_down]
        result[mask_left] = shift_left[mask_left]
        result[mask_right] = shift_right[mask_right]

        return result

    def step(self, action):
        self.t += 1

        reward = 0
        done = False

        if self.t == self.change_time and self.change_true == 1:
            self.orientations[
                self.change_index] += self.orientation_change  # Apply orientation change to the selected Gabor filter

        observation, observationClean = self._next_observation()

        if action == 1 and self.t < self.change_time:
            reward = -1
            done = True
        elif action == 1 and self.t >= self.change_time:
            if self.change_true == 1:
                reward = 1
            else:
                reward = 0
            done = True

        if self.t >= self.T:
            done = True
            if action == 0 and self.change_true == 0:
                reward = 1

        return observation, observationClean, reward, done, {}

    def render(self, mode="human"):
        if self.t == 0:
            plt.figure(figsize=(6, 6))

        obs, obsClean = self._next_observation()
        plt.clf()  # Clear the current figure
        plt.imshow(obsClean, cmap='gray')
        plt.title(f"Step: {self.t}, Orientation: {self.orientations}, Change: {self.change_true}")
        plt.pause(0.5)  # Short pause to update the figure

        if self.t == self.T - 1:
            plt.close()  # Close the figure window at the end of the episode

    def close(self):
        pass


### Data Collection ###
### 10^5 trials, 8 time steps, 4 patches, H=W=25, 1 channel ###
dataNoise = np.zeros((100000,8, 4, 25, 25, 1))
dataClean = np.zeros((100000,8, 4, 25, 25, 1))

trial = 0
while trial < 100000:

    # Running a trial #
    env = ChangeDetectionEnv()
    observation, observationClean = env.reset()
    # print(observationClean)

    dataNoise[trial, env.t,0,:,:,0] = observation[0:25, 0:25]
    dataNoise[trial, env.t, 1, :, :, 0] = observation[25:50, 0:25]
    dataNoise[trial, env.t, 2, :, :, 0] = observation[0:25, 25:50]
    dataNoise[trial, env.t, 3, :, :, 0] = observation[25:50, 25:50]

    dataClean[trial, env.t, 0, :, :, 0] = observationClean[0:25, 0:25]
    dataClean[trial, env.t, 1, :, :, 0] = observationClean[25:50, 0:25]
    dataClean[trial, env.t, 2, :, :, 0] = observationClean[0:25, 25:50]
    dataClean[trial, env.t, 3, :, :, 0] = observationClean[25:50, 25:50]

    done = False
    while done == False:
        # env.render()
        action = 0
        observation, observationClean, reward, done, info = env.step(action)

        # print(observation.shape)

        dataNoise[trial, env.t, 0, :, :, 0] = observation[0:25, 0:25]
        dataNoise[trial, env.t, 1, :, :, 0] = observation[25:50, 0:25]
        dataNoise[trial, env.t, 2, :, :, 0] = observation[0:25, 25:50]
        dataNoise[trial, env.t, 3, :, :, 0] = observation[25:50, 25:50]

        dataClean[trial, env.t, 0, :, :, 0] = observationClean[0:25, 0:25]
        dataClean[trial, env.t, 1, :, :, 0] = observationClean[25:50, 0:25]
        dataClean[trial, env.t, 2, :, :, 0] = observationClean[0:25, 25:50]
        dataClean[trial, env.t, 3, :, :, 0] = observationClean[25:50, 25:50]

    env.close()

    trial += 1
    if trial % 10000 == 0:
        print('trial',trial)

h5f = h5py.File('BigData.h5', 'a')
h5f.create_dataset('noise=PixelSwap/ChangeDegree=(0,35)', data=dataNoise)
h5f.create_dataset('noise=0/ChangeDegree=(0,35)', data=dataClean)
h5f.close()

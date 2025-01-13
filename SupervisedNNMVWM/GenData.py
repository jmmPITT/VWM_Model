import numpy as np
import torch
import torch as T
import matplotlib.pyplot as plt

from MATenv import ChangeDetectionEnv  # make sure this import is correct
from VAENet import VAE  # import your VAE class from VAENet

def load_model(model_path, model_class=VAE):
    """
    Load a VAE model from the specified path.
    """
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def generate_trial_data(
        Delta_max=65,
        model_path='vae_model.pth',
        n_games=5,
        T_end=7,
        save_labels='trial_labels.npy',
        save_observations='trial_observations.npy'
):
    """
    Generate trial data using a ChangeDetectionEnv and a trained VAE encoder.

    Parameters
    ----------
    Delta_max : float or int
        The maximum orientation change parameter for the environment.
    model_path : str
        Path to the saved VAE model (.pth file).
    n_games : int
        Number of trials (episodes) to run.
    T_end : int
        Maximum number of timesteps per episode.
    save_labels : str
        Filename for saving the trial labels.
    save_observations : str
        Filename for saving the trial observation data.
    """
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    input_dim = 128 + 4 + 8  # (128 latent dims + 4 + 8)

    # Load the encoder
    encoder = load_model(model_path).to(device)
    encoder.eval()

    # Initialize the environment
    env = ChangeDetectionEnv(Delta_max)

    # Prepare containers for data
    trial_observation_data = np.zeros((n_games, T_end, 4, input_dim))
    trial_labels = np.zeros((n_games, T_end, 1))

    # Generate data
    for i in range(n_games):
        state = env.reset()

        # Mark label at env.t == 0
        trial_labels[i, env.t] = 0

        # Build VAE input for 4 quadrants
        VAEinput = np.zeros((4, 25, 25))
        VAEinput[0, :, :] = state[0:25, 0:25]
        VAEinput[1, :, :] = state[25:50, 0:25]
        VAEinput[2, :, :] = state[0:25, 25:50]
        VAEinput[3, :, :] = state[25:50, 25:50]

        # Encode
        noisy_sample = torch.tensor(VAEinput, dtype=torch.float32).view(4, 25, 25, 1).to(device)
        embeddings = encoder(noisy_sample).detach().cpu().numpy()

        # Fill observation data
        trial_observation_data[i, env.t, 0, 0:128] = embeddings[0, :]
        trial_observation_data[i, env.t, 0, 128] = 1
        trial_observation_data[i, env.t, 0, 132 + env.t] = 1

        trial_observation_data[i, env.t, 1, 0:128] = embeddings[1, :]
        trial_observation_data[i, env.t, 1, 129] = 1
        trial_observation_data[i, env.t, 1, 132 + env.t] = 1

        trial_observation_data[i, env.t, 2, 0:128] = embeddings[2, :]
        trial_observation_data[i, env.t, 2, 130] = 1
        trial_observation_data[i, env.t, 2, 132 + env.t] = 1

        trial_observation_data[i, env.t, 3, 0:128] = embeddings[3, :]
        trial_observation_data[i, env.t, 3, 131] = 1
        trial_observation_data[i, env.t, 3, 132 + env.t] = 1

        done = False

        while not done:
            next_state, reward_env, done, _ = env.step(0)

            VAEinput = np.zeros((4, 25, 25))
            VAEinput[0, :, :] = next_state[0:25, 0:25]
            VAEinput[1, :, :] = next_state[25:50, 0:25]
            VAEinput[2, :, :] = next_state[0:25, 25:50]
            VAEinput[3, :, :] = next_state[25:50, 25:50]

            noisy_sample = torch.tensor(VAEinput, dtype=torch.float32).view(4, 25, 25, 1).to(device)
            embeddings = encoder(noisy_sample).detach().cpu().numpy()

            if not done:
                # Fill observation data
                trial_observation_data[i, env.t, 0, 0:128] = embeddings[0, :]
                trial_observation_data[i, env.t, 0, 128] = 1
                trial_observation_data[i, env.t, 0, 132 + env.t] = 1

                trial_observation_data[i, env.t, 1, 0:128] = embeddings[1, :]
                trial_observation_data[i, env.t, 1, 129] = 1
                trial_observation_data[i, env.t, 1, 132 + env.t] = 1

                trial_observation_data[i, env.t, 2, 0:128] = embeddings[2, :]
                trial_observation_data[i, env.t, 2, 130] = 1
                trial_observation_data[i, env.t, 2, 132 + env.t] = 1

                trial_observation_data[i, env.t, 3, 0:128] = embeddings[3, :]
                trial_observation_data[i, env.t, 3, 131] = 1
                trial_observation_data[i, env.t, 3, 132 + env.t] = 1

                # Assign labels after the change_time
                if env.t >= env.change_time:
                    if env.change_true == 1:
                        trial_labels[i, env.t] = 1
                    else:
                        trial_labels[i, env.t] = 0

    # Save results
    np.save(save_labels, trial_labels)
    np.save(save_observations, trial_observation_data)

    print(f"Saved labels to {save_labels} and observations to {save_observations}.")

# EXAMPLE USAGE (uncomment if you want to run directly):
# if __name__ == "__main__":
#     generate_trial_data(Delta_max=65, model_path='vae_model.pth', n_games=1000, T_end=7)
#     # Then check the created files trial_labels.npy & trial_observations.npy

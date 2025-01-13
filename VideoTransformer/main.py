
from MATenv import *
from agent_planner import Agent as AgentPlanner
from VidTFNET import *
from VAENet import VAE
import gymnasium
import numpy as np
import torch
import torch as T

def softmax(vector):
    e = np.exp(vector - np.max(vector))  # Subtract max for numerical stability
    return e / e.sum()

def sigmoid(vector):
    return 1 / (1 + np.exp(-vector))

def load_model(model_path, model_class=VAE):
    # Initialize the model and load the state dictionary
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

if __name__ == '__main__':
    env = 0
    ### Trial Max Length ###
    T_end = 8

    ### Observation Properties ###
    VAE_embed_size = 128  # See Documentation Supplement
    n_patches = 4
    patch_width = 25
    patch_height = 25
    positional_dim = 4  # one-hot for patch index
    temporal_dim = 8  # one_hot for time-step

    input_dim = (VAE_embed_size + positional_dim + temporal_dim) * n_patches
    # modded_state_dim = 8
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    agent_lever = AgentPlanner(alpha=0.0001, beta=0.0001,
                               input_dims=input_dim, tau=0.01,
                               env=env, batch_size=128, max_size=50000,
                               n_actions=2)

    Transformer = TransformerNetwork(0.00001, input_dims=input_dim, hidden_dim=256, fc1_dims=256, fc2_dims=128,
                                     name='transformer1', chkpt_dir='td3_MAT').to(agent_lever.device)
    encoder = load_model('vae_model.pth').to(device)
    encoder.eval()
    n_games = 1000000
    score_history = []

    ### Initialize the environment class ###
    env = ChangeDetectionEnv()

    ### For loading models ###
    # agent_planner.load_models()
    # Transformer.load_state_dict(T.load("transformer1_td3"))

    cue_array = 0
    target_array = 0
    for i in range(n_games):

        ### Initial State ###
        state = env.reset()

        ### Construct the observation to the belief model (depends on attention) ###
        ### adding a positional embedding of length 4 ###
        ### added a temporal embedding of length 7 ###
        ### now each embedding (token) is 15 dimensional ###
        obs = np.zeros((T_end, n_patches, VAE_embed_size + positional_dim + temporal_dim))
        VAEinput = np.zeros((n_patches, patch_height, patch_width))
        VAEinput[0, :, :] = state[0:25, 0:25]
        VAEinput[1, :, :] = state[25:50, 0:25]
        VAEinput[2, :, :] = state[0:25, 25:50]
        VAEinput[3, :, :] = state[25:50, 25:50]

        noisy_sample = torch.tensor(VAEinput, dtype=torch.float32).view(n_patches, patch_height, patch_width, 1).to(device)  # Add batch dimension
        embeddings = encoder(noisy_sample)
        embeddings = embeddings.detach().cpu().numpy()

        obs[env.t, 0, 0:VAE_embed_size] = embeddings[0, :]
        obs[env.t, 0, VAE_embed_size] = 1
        obs[env.t, 0,  VAE_embed_size + positional_dim + env.t] = 1

        obs[env.t, 1, 0:VAE_embed_size] = embeddings[1, :]
        obs[env.t, 1, VAE_embed_size + 1] = 1
        obs[env.t, 1, VAE_embed_size + positional_dim + env.t] = 1

        obs[env.t, 2, 0:VAE_embed_size] = embeddings[2, :]
        obs[env.t, 2, VAE_embed_size + 2] = 1
        obs[env.t, 2,  VAE_embed_size + positional_dim + env.t] = 1

        obs[env.t, 3, 0:VAE_embed_size] = embeddings[3, :]
        obs[env.t, 3, VAE_embed_size + 3] = 1
        obs[env.t, 3,  VAE_embed_size + positional_dim + env.t] = 1

        ### mask is the same shape as obs ###
        ### we use this mask to ensure that self-attention ###
        ### is not computed for timesteps not encountered yet ###
        mask = np.ones((T_end, n_patches, VAE_embed_size + positional_dim + temporal_dim))
        mask[env.t + 1:, :, :] = 0
        mask_ = np.ones((T_end, n_patches, VAE_embed_size + positional_dim + temporal_dim))


        done = False
        score = 0
        while not done:

            ### state derived from sequence of observations ###
            state = T.tensor(obs.reshape(1, -1), dtype=T.float).to(agent_lever.critic_1.device)
            m = T.tensor(mask, dtype=T.float).to(agent_lever.critic_1.device)


            ### this is the only location where these inputs are needed. The lever agent operates on the output of the transformer ###
            transformer_state = Transformer(state, env.t, m)


            lever_action = agent_lever.choose_action(transformer_state.detach().cpu().numpy().reshape(1, -1), state)

            # Sample from these probabilities
            sampled_index = np.random.choice(len(lever_action), p=lever_action)
            lever_action_buffer = np.zeros(2)
            lever_action_buffer[sampled_index] = 1


            next_state, reward_env, done, _ = env.step(sampled_index)
            ### Now, env.t is the next time step: Remember this ###

            ### Construct the next observation to the coordinator model (depends on attention) ###
            obs_ = np.copy(obs)
            if done == False:

                ### Construct the input to the VAE model ###
                ### 4 patches (treated independently in VAE) of size 25 x 25 ###
                VAEinput = np.zeros((n_patches, patch_height, patch_width))
                VAEinput[0, :, :] = next_state[0:25, 0:25]
                VAEinput[1, :, :] = next_state[25:50, 0:25]
                VAEinput[2, :, :] = next_state[0:25, 25:50]
                VAEinput[3, :, :] = next_state[25:50, 25:50]

                ### Encode ###
                noisy_sample = torch.tensor(VAEinput, dtype=torch.float32).view(4, 25, 25, 1).to(device)  # Add batch dimension
                embeddings = encoder(noisy_sample)
                embeddings = embeddings.detach().cpu().numpy()

                ### Construct Observation TO RL Agent ###
                ### Observation consists of VAE embedding ###
                ### And a spatial plus temporal one-hot encoding ###
                obs_[env.t, 0, 0:128] = embeddings[0, :]
                obs_[env.t, 0, 128] = 1
                obs_[env.t, 0, 132 + env.t] = 1

                obs_[env.t, 1, 0:128] = embeddings[1, :]
                obs_[env.t, 1, 129] = 1
                obs_[env.t, 1, 132 + env.t] = 1

                obs_[env.t, 2, 0:128] = embeddings[2, :]
                obs_[env.t, 2, 130] = 1
                obs_[env.t, 2, 132 + env.t] = 1

                obs_[env.t, 3, 0:128] = embeddings[3, :]
                obs_[env.t, 3, 131] = 1
                obs_[env.t, 3, 132 + env.t] = 1


                ### mask is the same shape as obs ###
                ### we use this mask to ensure that self-attention ###
                ### is not computed for timesteps not encountered yet ###
                ### we need mask_ to accompany obs_ in the transformer ###
                mask_ = np.ones((T_end, n_patches, VAE_embed_size + positional_dim + temporal_dim))
                mask_[env.t + 1:, :, :] = 0

            agent_lever.remember(obs.reshape(1,-1), mask.reshape(1,-1), lever_action_buffer, reward_env, obs_.reshape(1,-1), mask_.reshape(1,-1), done, env.t)
            Transformer = agent_lever.learn(Transformer)

            score += reward_env
            obs = obs_
            mask = mask_

        if i % 100 == 0:
            agent_lever.save_models()
            Transformer.save_checkpoint()
            print('Theta!', env.theta)

        score_history.append(score)
        avg_score = np.mean(score_history[-1000:])

        ### Increasing the task difficulty if the agent averages over ###
        ### 0.85 accuracy for 1000 trials ###
        if avg_score > 0.85 and len(score_history) > 1000:
            env.theta -= 3
            score_history = []
            print('New Theta!', env.theta)


        print('episode ', i, 'score %.2f' % score,
                'trailing 100 games avg %.3f' % avg_score, 'actions',lever_action)







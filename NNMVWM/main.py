
from OCDEnv import *
from agent import Agent as AgentPlanner
from VWMNET import *

from VAENetTruncated import VAE

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
    input_dim = (128 + 4 + 8) * 4
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    agent = AgentPlanner(alpha=0.0001, beta=0.0001,
                         input_dims=input_dim, tau=0.01,
                         env=env, batch_size=64, max_size=50000,
                         n_actions=2)


    Transformer = TransformerNetwork(0.00001, input_dims=input_dim, hidden_dim = 256, fc1_dims=256, fc2_dims=128, name='transformer1', chkpt_dir='td3_MAT').to(agent.device)
    encoder = load_model('vae_model.pth').to(device)
    encoder.eval()
    n_games = 150000
    score_history = []
    env = ChangeDetectionEnv()

    #
    # agent.load_models()
    # Transformer.load_state_dict(T.load("transformer1_td3"))
    cue_array = 0
    target_array = 0
    for i in range(n_games):
        # env = ChangeDetectionEnv()

        state = env.reset()

        ### Construct the observation to the belief model (depends on attention) ###
        ### adding a positional embedding of length 4 ###
        ### added a temporal embedding of length 7 ###
        ### now each embedding (token) is 15 dimensional ###
        obs = np.zeros((T_end, 4, 128 + 4 + 8))
        VAEinput = np.zeros((4, 25, 25))
        VAEinput[0, :, :] = state[0:25, 0:25]
        VAEinput[1, :, :] = state[25:50, 0:25]
        VAEinput[2, :, :] = state[0:25, 25:50]
        VAEinput[3, :, :] = state[25:50, 25:50]

        noisy_sample = torch.tensor(VAEinput, dtype=torch.float32).view(4, 25, 25, 1).to(device)  # Add batch dimension
        # print(noisy_sample.shape)
        embeddings = encoder(noisy_sample)
        # print(embeddings.shape)
        embeddings = embeddings.detach().cpu().numpy()
        # logVAR = logVAR.detach().cpu().numpy()*10.0

        obs[env.t, 0, 0:128] = embeddings[0, :]
        obs[env.t, 0, 128] = 1
        obs[env.t, 0, 132 + env.t] = 1

        obs[env.t, 1, 0:128] = embeddings[1, :]
        obs[env.t, 1, 129] = 1
        obs[env.t, 1, 132 + env.t] = 1

        obs[env.t, 2, 0:128] = embeddings[2, :]
        obs[env.t, 2, 130] = 1
        obs[env.t, 2, 132 + env.t] = 1

        obs[env.t, 3, 0:128] = embeddings[3, :]
        obs[env.t, 3, 131] = 1
        obs[env.t, 3, 132 + env.t] = 1

        mask = np.ones((T_end, 4, 1))
        mask[env.t + 1:, :, :] = 0
        mask_ = np.ones((T_end, 4, 1))


        done = False
        score = 0
        while not done:

            ### state derived from sequence of observations ###
            state = T.tensor(obs.reshape(1, -1), dtype=T.float).to(agent.critic_1.device)
            m = T.tensor(mask, dtype=T.float).to(agent.critic_1.device)


            ### this is the only location where these inputs are needed. The lever agent operates on the output of the transformer ###
            transformer_state = Transformer(state, env.t, m)


            lever_action = agent.choose_action(transformer_state.detach().cpu().numpy().reshape(1, -1), state)

            sampled_index = np.random.choice(len(lever_action), p=lever_action)
            lever_action_buffer = np.zeros(2)
            lever_action_buffer[sampled_index] = 1


            next_state, reward_env, done, _ = env.step(sampled_index)


            ### Construct the next observation to the coordinator model (depends on attention) ###
            obs_ = np.copy(obs)
            if done == False:
                ### Construct the observation to the belief model (depends on attention) ###
                ### adding a positional embedding of length 4 ###
                ### added a temporal embedding of length 7 ###
                ### now each embedding (token) is 15 dimensional ###
                VAEinput = np.zeros((4, 25, 25))
                VAEinput[0, :, :] = next_state[0:25, 0:25]
                VAEinput[1, :, :] = next_state[25:50, 0:25]
                VAEinput[2, :, :] = next_state[0:25, 25:50]
                VAEinput[3, :, :] = next_state[25:50, 25:50]

                noisy_sample = torch.tensor(VAEinput, dtype=torch.float32).view(4, 25, 25, 1).to(device)  # Add batch dimension
                embeddings = encoder(noisy_sample)
                embeddings = embeddings.detach().cpu().numpy()

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


                mask_ = np.ones((T_end, 4, 1))
                mask_[env.t + 1:, :, :] = 0

            ### env.t is the next time step: Remember this ###
            agent.remember(obs.reshape(1, -1), mask.reshape(1, -1), lever_action_buffer, reward_env, obs_.reshape(1, -1), mask_.reshape(1, -1), done, env.t, cue_array, target_array)

            Transformer = agent.learn(Transformer)

            score += reward_env
            obs = obs_
            mask = mask_

        if i % 100 == 0:
            agent.save_models()
            Transformer.save_checkpoint()
            print('Theta!', env.theta)

        score_history.append(score)
        avg_score = np.mean(score_history[-1000:])

        if avg_score > 0.85 and len(score_history)>1000:
            env.theta -= 3
            score_history = []
            print('New Theta!',env.theta)


        print('episode ', i, 'score %.2f' % score,
                'trailing 100 games avg %.3f' % avg_score, 'actions',lever_action)






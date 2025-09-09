
import gymnasium
import numpy as np
import torch
from OCDEnv import *
from agent import Agent as AgentPlanner
import torch as T
from VWMNET import *

import gymnasium
import numpy as np
import torch
import torch as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import h5py
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
    T_end = 10
    input_dim = (128 + 4 + 10) * 4
    # modded_state_dim = 8
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    agent_lever = AgentPlanner(alpha=0.0001, beta=0.0001,
                            input_dims=input_dim, tau=0.01,
                            env=env, batch_size=32, max_size=10000,
                            n_actions=2)


    Transformer = TransformerNetwork(0.00001, input_dims=input_dim, hidden_dim = 256, fc1_dims=256, fc2_dims=128, name='transformer1', chkpt_dir='td3_MAT').to(agent_lever.device)
    encoder = load_model('vae_model.pth').to(device)
    encoder.eval()
    n_games = 100
    # best_score = env.reward_range[0]
    score_history = []
    env = ChangeDetectionEnv()

    #
    agent_lever.load_models()
    # # agent_attention.load_models()
    Transformer.load_state_dict(T.load("transformer1_td3"))

    cue_array = 0
    target_array = 0
    A_list = []
    for i in range(n_games):
        # env = ChangeDetectionEnv()

        state = env.reset()

        env.cue_position = 'left'
        env.proportion = 1.0
        # env.change_time = 25
        env.orientation_change = 0
        env.change_true = 0
        env.change_index = 0

        ### Construct the observation to the belief model (depends on attention) ###
        ### adding a positional embedding of len14gth 4 ###
        ### added a temporal embedding of length 7 ###
        ### now each embedding (token) is 15 dimensional ###
        obs = np.zeros((T_end, 4, 128 + 4 + 10))
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
        # obs[env.t, 0, 20:40] = logVAR[0, :]
        obs[env.t, 0, 128] = 1
        obs[env.t, 0, 132 + env.t] = 1

        obs[env.t, 1, 0:128] = embeddings[1, :]
        # obs[env.t, 1, 20:40] = logVAR[1, :]
        obs[env.t, 1, 129] = 1
        obs[env.t, 1, 132 + env.t] = 1

        obs[env.t, 2, 0:128] = embeddings[2, :]
        # obs[env.t, 2, 20:40] = logVAR[2, :]
        obs[env.t, 2, 130] = 1
        obs[env.t, 2, 132 + env.t] = 1

        obs[env.t, 3, 0:128] = embeddings[3, :]
        # obs[env.t, 3, 20:40] = logVAR[3, :]
        obs[env.t, 3, 131] = 1
        obs[env.t, 3, 132 + env.t] = 1

        # obs[env.t, 1, :] = embeddings[1, :]
        # obs[env.t, 2, :] = embeddings[2, :]
        # obs[env.t, 3, :] = embeddings[3, :]

        mask = np.ones((T_end, 4, 1))
        mask[env.t + 1:, :, :] = 0
        mask_ = np.ones((T_end, 4, 1))


        done = False
        score = 0
        while not done:

            ### state derived from sequence of observations ###
            state = T.tensor(obs.reshape(1, -1), dtype=T.float).to(agent_lever.critic_1.device)
            m = T.tensor(mask, dtype=T.float).to(agent_lever.critic_1.device)


            ### this is the only location where these inputs are needed. The lever agent operates on the output of the transformer ###
            transformer_state, A = Transformer(state, m)
            # print("A",A.shape)

            lever_action = agent_lever.choose_action(transformer_state.detach().cpu().numpy().reshape(1, -1))
            # print('lever action',lever_action)
            # lever_action = agent_planner.choose_action(transformer_state.detach().cpu().numpy().reshape(1, -1), state)

            # lever_action_env = np.argmax(lever_action)
            # lever_action_env = softmax(lever_action)
            # Sample from these probabilities
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
                # obs = np.zeros((T_end, 4, 20))
                VAEinput = np.zeros((4, 25, 25))
                VAEinput[0, :, :] = next_state[0:25, 0:25]
                VAEinput[1, :, :] = next_state[25:50, 0:25]
                VAEinput[2, :, :] = next_state[0:25, 25:50]
                VAEinput[3, :, :] = next_state[25:50, 25:50]

                noisy_sample = torch.tensor(VAEinput, dtype=torch.float32).view(4, 25, 25, 1).to(device)  # Add batch dimension
                # print(noisy_sample.shape)
                embeddings = encoder(noisy_sample)
                # print(embeddings.shape)
                embeddings = embeddings.detach().cpu().numpy()
                # logVAR = logVAR.detach().cpu().numpy()*10.0

                obs_[env.t, 0, 0:128] = embeddings[0, :]
                # obs[env.t, 0, 20:40] = logVAR[0, :]
                obs_[env.t, 0, 128] = 1
                obs_[env.t, 0, 132 + env.t] = 1

                obs_[env.t, 1, 0:128] = embeddings[1, :]
                # obs[env.t, 1, 20:40] = logVAR[1, :]
                obs_[env.t, 1, 129] = 1
                obs_[env.t, 1, 132 + env.t] = 1

                obs_[env.t, 2, 0:128] = embeddings[2, :]
                # obs[env.t, 2, 20:40] = logVAR[2, :]
                obs_[env.t, 2, 130] = 1
                obs_[env.t, 2, 132 + env.t] = 1

                obs_[env.t, 3, 0:128] = embeddings[3, :]
                # obs[env.t, 3, 20:40] = logVAR[3, :]
                obs_[env.t, 3, 131] = 1
                obs_[env.t, 3, 132 + env.t] = 1


                mask_ = np.ones((T_end, 4, 1))
                mask_[env.t + 1:, :, :] = 0
                # mask_ = np.ones((T_end, 4, 20))

            ### env.t is the next time step: Remember this ###
            # agent_lever.remember(obs.reshape(1,-1), mask.reshape(1,-1), lever_action_buffer, reward_env, obs_.reshape(1,-1), mask_.reshape(1,-1), done, env.t, cue_array, target_array)

            # Transformer = agent_lever.learn(Transformer)

            score += reward_env
            obs = obs_
            mask = mask_

        # if i % 100 == 0:
        #     agent_lever.save_models()
        #     Transformer.save_checkpoint()
        #     print('Theta!', env.theta)
        A_list.append(A.view(1,10,4,4).detach())

        score_history.append(score)
        avg_score = np.mean(score_history[-1000:])

        if avg_score > 0.85 and len(score_history)>1000:
            env.theta -= 3
            score_history = []
            print('New Theta!',env.theta)


        print('episode ', i, 'score %.2f' % score,
                'trailing 100 games avg %.3f' % avg_score, 'actions',lever_action)

    A_list = torch.cat(A_list, dim=0)
    print("A_list",A_list.cpu().numpy().shape)
    A_list = A_list.cpu().numpy()

    # 1. Average over the first dimension
    avg_matrices = A_list.mean(axis=0)   # shape will be (31, 4, 4)
    for idx, mat in enumerate(avg_matrices):
        plt.figure()
        im = plt.imshow(mat, vmin=0, vmax=1)
        plt.colorbar(im)
        plt.title(f"Matrix {idx+1}")
        plt.savefig(f"heatmap_{idx+1:02d}.png", dpi=150, bbox_inches='tight')
        plt.close()

    ACol1 = np.sum(avg_matrices,axis=1)/4.0
    print(ACol1.shape)



    x = np.arange(ACol1.shape[0])  # 0, 1, …, N–1

    plt.figure(figsize=(10, 6))

    # Optional: define different line styles & markers
    line_styles = ['-', '-', '-', '-']
    # markers     = ['o', 's', '^', 'd']

    for i in range(ACol1.shape[1]):
        plt.plot(
            x, ACol1[:, i],
            linestyle=line_styles[i % len(line_styles)],
            # marker=markers[i % len(markers)],
            linewidth=4,
            # markersize=6,
            label=fr'$\alpha_{i+1}$'
        )

    plt.xlabel(r'time', fontsize=16)
    plt.ylabel(r'attention', fontsize=16)
    plt.title('Attention allocated per stimulus', fontsize=18, fontweight='bold')
    plt.ylim(0, 1)
    plt.legend(title='Stim-Att', fontsize=14, title_fontsize=16, loc='upper right')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"Att_Cue100.png", dpi=150, bbox_inches='tight')
    plt.show()

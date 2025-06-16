from MATenvTest import ChangeDetectionEnv
from agent_test import Agent as AgentPlanner
from test2 import TransformerNetwork

import matplotlib.pyplot as plt
import torch.nn.functional as F
import h5py
from VAENet import VAE

import numpy as np
import torch
import torch as T

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
    # modded_state_dim = 8
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    agent_lever = AgentPlanner(alpha=0.00001, beta=0.00001,
                            input_dims=input_dim, tau=0.001,
                            env=env, batch_size=64, max_size=200000,
                            n_actions=2)


    Transformer = TransformerNetwork(0.000001, input_dims=input_dim, hidden_dim = 256, fc1_dims=256, fc2_dims=128, name='transformer1', chkpt_dir='td3_MAT').to(agent_lever.device)

    encoder = load_model('vae_model').to(device)
    encoder.eval()
    n_games = 500
    # best_score = env.reward_range[0]
    score_history = []


    #
    agent_lever.load_models()
    # # agent_attention.load_models()
    Transformer.load_state_dict(T.load("transformer1_td3"))
    #
    attn_matrix_sum = np.zeros((1, 1, 4, 4))

    # Experiment parameters
    n_alphas = 35  # Number of orientation change values to test
    linear_spaced_modulation = np.linspace(0.0, 45.0, n_alphas)  # Range of orientation changes
    
    # Initialize results array: [miss, hit, false alarm, correct rejection, reaction time]
    choiceStats = np.zeros((n_alphas, 5))


    for j in range(n_alphas):

        cue_array = 0
        target_array = 0
        for i in range(n_games):
            env = ChangeDetectionEnv()

            state = env.reset()

            env.cue_position = 'left'
            env.proportion = 0.25
            # Always set change to true for this experiment
            env.change_true = 1
            # env.change_true = 1
            # env.orientation_change = np.random.uniform(-45, 45)
            env.orientation_change = linear_spaced_modulation[j]

            rand = np.random.rand()
            if env.change_true == 1:
                env.change_index = 0
                # if env.cue_position == 'left':
                #     if rand < env.proportion:
                #         env.change_index = 0  # Randomly select new Gabor filter for change
                #     else:
                #         env.change_index = np.random.randint(3) + 1
                # elif env.cue_position == 'right':
                #     if rand < env.proportion:
                #         env.change_index = 3  # Randomly select new Gabor filter for change
                #     else:
                #         env.change_index = np.random.randint(3)
            #
            # env.orientation_change = 30
            # env.change_true = 1
            # env.change_index = 0

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
                transformer_state, attn_matrix = Transformer(state, env.t, m, 0)

                # if env.t == 1:
                #     choiceStats[j, 4] += np.sum(attn_matrix[0, 0, :, 0])


                lever_action = agent_lever.choose_action(transformer_state.detach().cpu().numpy().reshape(1, -1), state)

                lever_action_env = np.argmax(lever_action)


                next_state, reward_env, done, _ = env.step(lever_action_env)


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

                    noisy_sample = torch.tensor(VAEinput, dtype=torch.float32).view(4, 25, 25, 1).to(
                        device)  # Add batch dimension
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

                score += reward_env
                obs = obs_
                mask = mask_

                # Reshape the attention matrix for plotting
                # if env.change_time == env.t:
                #     print('Mystery')
            # print(attn_matrix.shape)
            # attn_matrix = attn_matrix[0,1].view(1, 1, attn_matrix.size(-1), attn_matrix.size(-1)).detach().cpu().numpy()
            # attn_matrix_sum = attn_matrix_sum + attn_matrix.detach().cpu().numpy()


            ### 0 -> miss, 1 -> hit, 2 -> FA, 3 -> CR
            if reward_env == 1 and (env.change_true == 1):
                choiceStats[j,1] += 1 # H
            if reward_env == 1 and (env.change_true == 0):
                choiceStats[j,3] += 1 # CR
            if reward_env == 0 and (env.change_true == 1):
                choiceStats[j,0] += 1 # miss
            if reward_env == 0 and (env.change_true == 0):
                choiceStats[j,2] += 1 # FA
            choiceStats[j,4] += env.t




            score_history.append(score)
            avg_score = np.mean(score_history[-1000:])

            # Uncomment for debugging
            # print('episode ', i, 'score %.2f' % score, 'trailing 100 games avg %.3f' % avg_score)



        # Uncomment for attention visualization
        # plot_attention_heatmaps(attn_matrix_sum / (i + 1), sample_index=0)  # for the first sample in the batch
        print('RT', (j, choiceStats[j,4]/n_games))
    
    # Save results
    output_file = 'Plotting/ChoiceStats_CueS125_ChangeS1_alpha1TchangeEqual10_DeltaModulation.npy'
    np.save(output_file, choiceStats)
    print(f'Results saved to {output_file}')







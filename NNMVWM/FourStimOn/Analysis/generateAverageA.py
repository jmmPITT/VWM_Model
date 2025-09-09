
from OCDEnv import *
from agent import Agent as AgentPlanner
from VWMNET import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
import h5py
from VAENet import VAE

import numpy as np
import torch
import torch as T

import matplotlib.pyplot as plt
import seaborn as sns

def softmax(vector):
    e = np.exp(vector - np.max(vector))  # Subtract max for numerical stability
    return e / e.sum()

def scaled_dot_product_attention(query, key, value, time=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)

    # print('time',time)
    print('scores',scores.shape)

    scores[0,:,:,time*4:] = float('-inf')
    scores[0,:,time*4:,:] = float('-inf')

    # print('scores',scores)
    # print('scores',scores.shape)


    # if mask is not None:
    #     # Use masked_fill to set the scores for masked positions to a large negative value
    #     # scores = scores.masked_fill(mask, float('-inf'))
    #     ones_like_scores = ones_like_scores.to(dtype=scores.dtype)  # Ensure mask is the same dtype as scores

    #     scores = scores.masked_fill(ones_like_scores == 0, float('-inf'))


    attn = F.softmax(scores, dim=-1)
    attn[0,:,time*4:,:] = 0

    output = torch.matmul(attn, value)
    return output, attn



def compute_qkv(input, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias, n_heads):
    d_k = q_weight.size(0) // n_heads
    print('input size: I wish chatgpt would give me code that works',input.size())
    # seq_len, feat_dim = input.size()
    seq_len = 7*4
    feat_dim = 8 + 7

    # Linear transformations and reshaping for multi-head
    query = F.linear(input, q_weight, q_bias).view(1, seq_len, n_heads, d_k).transpose(1, 2)
    key = F.linear(input, k_weight, k_bias).view(1, seq_len, n_heads, d_k).transpose(1, 2)
    value = F.linear(input, v_weight, v_bias).view(1, seq_len, n_heads, d_k).transpose(1, 2)

    return query, key, value

import matplotlib.pyplot as plt
import numpy as np

def plot_attention_heatmaps(attn_matrix, sample_index=0, save_path='attention_heatmaps.png',
                            title_fontsize=58, label_fontsize=44, tick_fontsize=10, cbar_labelsize=44):
    # attn_matrix shape: [batch_size, n_heads, seq_length, seq_length]
    n_heads = attn_matrix.shape[1]

    print(attn_matrix.shape)
    # attn_matrix = attn_matrix * 0
    # attn_matrix[0,0,1:,0] = 1
    # attn_matrix[0,0,0,1:] = 1/3
    if n_heads > 1:
        fig, axs = plt.subplots(1, n_heads, figsize=(n_heads * 12, 12))
    else:
        fig, axs = plt.subplots(1, n_heads, figsize=(12, 12))
        axs = [axs]  # Ensure axs is iterable for the loop below

    # fig.suptitle('Attention Heatmaps for Each Head', color='white')
    fig.patch.set_facecolor('black')

    for i, ax in enumerate(axs):
        attn_map = attn_matrix[sample_index, i]
        cax = ax.matshow(attn_map, cmap='viridis', vmin=0, vmax=1)
        # cax = ax.matshow(attn_map, cmap='viridis')
        ax.set_title(r'$t_{cue}$', color='white', fontsize=title_fontsize)
        ax.set_xlabel('Key', color='white', fontsize=label_fontsize)
        ax.set_ylabel('Query', color='white', fontsize=label_fontsize)
        ax.tick_params(axis='x', colors='white', labelsize=tick_fontsize)
        ax.tick_params(axis='y', colors='white', labelsize=tick_fontsize)
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        # Add a colorbar and set its label color
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', fontsize=cbar_labelsize)
        # cbar.set_ticks(np.linspace(0, 1, num=5))  # Add this line to set specific ticks on the colorbar

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Saved figure to {save_path}")
    plt.show()


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
    T_end = 9
    input_dim = (128 + 4 + 9) * 4
    # modded_state_dim = 8
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    agent_lever = AgentPlanner(alpha=0.00001, beta=0.00001,
                            input_dims=input_dim, tau=0.001,
                            env=env, batch_size=64, max_size=2,
                            n_actions=2)


    Transformer = TransformerNetwork(0.000001, input_dims=input_dim, hidden_dim = 256, fc1_dims=256, fc2_dims=128, name='transformer1', chkpt_dir='td3_MAT').to(agent_lever.device)

    encoder = load_model('vae_model.pth').to(device)
    encoder.eval()
    n_games = 100
    # best_score = env.reward_range[0]
    score_history = []


    #
    agent_lever.load_models()
    # # agent_attention.load_models()
    Transformer.load_state_dict(T.load("transformer1_td3"))
    #
    attn_matrix_sum = np.zeros((1, 1, 4, 4))
    entropy_sum = 0
    cue_array = 0
    target_array = 0

    A1Reshape = np.zeros((9, 2, 2))

    for i in range(n_games):
        env = ChangeDetectionEnv()

        s = env.reset()

        env.change_index=0
        env.cue_position = 'left'
        env.proportion = 1.0
        env.orientation_change = 30
        env.change_true = 0

        # env.cue_position = 'left'
        # env.proportion = 1.0

        # env.orientation_change = 0
        # env.orientation_change = 0
        # if np.random.rand() < 0.5:
        #     env.change_true = 1
        # else:
        #     env.change_true = 0
        # env.change_true = 1
        # env.change_index = 0

        # env.theta = 41

        ### Construct the observation to the belief model (depends on attention) ###
        ### adding a positional embedding of length 4 ###
        ### added a temporal embedding of length 7 ###
        ### now each embedding (token) is 15 dimensional ###
        obs = np.zeros((T_end, 4, 128 + 4 + 9))
        VAEinput = np.zeros((4, 25, 25))
        VAEinput[0, :, :] = s[0:25, 0:25]
        VAEinput[1, :, :] = s[25:50, 0:25]
        VAEinput[2, :, :] = s[0:25, 25:50]
        VAEinput[3, :, :] = s[25:50, 25:50]

        noisy_sample = torch.tensor(VAEinput, dtype=torch.float32).view(4, 25, 25, 1).to(device)  # Add batch dimension
        embeddings = encoder(noisy_sample)
        embeddings = embeddings.detach().cpu().numpy()

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
            transformer_state, attn_matrix = Transformer(state, m)


            lever_action = agent_lever.choose_action(transformer_state.detach().cpu().numpy().reshape(1, -1))

            lever_action_env = np.argmax(lever_action)
            # lever_action_env = 1 if lever_action[1]>0.8 else 0

            # sampled_index = np.random.choice(len(lever_action), p=lever_action)
            lever_action_buffer = np.zeros(2)
            lever_action_buffer[lever_action_env] = 1

            action = T.tensor(lever_action_buffer.reshape(1, -1), dtype=T.float).to(agent_lever.critic_1.device)

            # A1Reshape = np.zeros((2, 2))
            # Convert tensors to NumPy arrays
            # A1_np = attn_matrix[0,env.t,0,:,:].detach().cpu().numpy().squeeze()
            # c = 0
            # print('Shape',attn_matrix.shape)
            # for row in range(2):
            #     for col in range(2):
            #         A1Reshape[env.t, row, col] += np.sum(A1_np[:, c])
            #         c += 1


            # if (i+1 ) % 100 == 0:
            #     plt.figure(figsize=(10, 5))
            #
            #     plt.subplot(1, 2, 1)
            #     sns.heatmap(A1Reshape[env.t]/(4.0*100), cmap='Reds', square=True, vmin=0, vmax=1)
            #     plt.title('SA Heatmap (3x3)')
            #     plt.axis('equal')
            #
            #     plt.subplot(1, 2, 2)
            #     plt.imshow(s, cmap='gray_r')
            #     plt.title('Task Image (200x200)')
            #     plt.axis('off')
            #     plt.gca().set_aspect('equal', adjustable='box')
            #
            #     plt.tight_layout()
            #     plt.savefig('AttentionOnImage_t' + str(env.t) + '.pdf', dpi=300, bbox_inches='tight')
            #     plt.show()

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

            score += reward_env
            obs = obs_
            s = next_state
            mask = mask_

        # c = 0
        # for row in range(2):
        #     for col in range(2):
        #         A1Reshape[env.t, row, col] += np.sum(A1_np[:, c])
        #         c += 1

        #
        # if (i+1 ) % 100 == 0:
        #     plt.figure(figsize=(10, 5))
        #
        #     plt.subplot(1, 2, 1)
        #     sns.heatmap(A1Reshape[env.t]/(4.0*100), cmap='Reds', square=True, vmin=0, vmax=1)
        #     plt.title('SA Heatmap (3x3)')
        #     plt.axis('equal')
        #
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(s, cmap='gray_r')
        #     plt.title('Task Image (200x200)')
        #     plt.axis('off')
        #     plt.gca().set_aspect('equal', adjustable='box')
        #
        #     plt.tight_layout()
        #     plt.savefig('AttentionOnImage_t' + str(env.t) + '.pdf', dpi=300, bbox_inches='tight')
        #     plt.show()
        #
        #     # Reshape the attention matrix for plotting
        #     if env.change_time == env.t:
        #         print('Mystery')
        # print(attn_matrix.shape)
        attn_matrix = attn_matrix[0,3].view(1, 1, attn_matrix.size(-1), attn_matrix.size(-1))
        attn_matrix_sum = attn_matrix_sum + attn_matrix.detach().cpu().numpy()



        score_history.append(score)
        avg_score = np.mean(score_history[:])


        print('episode ', i, 'score %.2f' % score,
                'trailing 100 games avg %.3f' % avg_score)

        # print('Entropy Sum', entropy_sum/ (i + 1))



    plot_attention_heatmaps(attn_matrix_sum / (i + 1), sample_index=0)  # for the first sample in the batch







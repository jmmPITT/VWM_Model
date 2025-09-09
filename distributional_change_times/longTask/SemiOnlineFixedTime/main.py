
import gymnasium as gym
import cv2
import torch as T
import numpy as np
import torch.nn.functional as F
from MATenv import *
from agent_planner import *
from buffer import *

if __name__ == '__main__':
    env = ChangeDetectionEnv()  # or any other environment id


    buffer_size = 1

    ### Trial Max Length ###
    T_end = 10
    input_dim = (25 * 25) * 4
    # modded_state_dim = 8
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    agent_lever = Agent(alpha=0.0001, beta=0.0001,
                            input_dims=input_dim, tau=0.01,
                            env=env, batch_size=1, max_size=50,
                            n_actions=2)


    Transformer = TransformerNetwork(0.00001, input_dims=input_dim, hidden_dim = 256, fc1_dims=256, fc2_dims=128, name='transformer1', chkpt_dir='/content/drive/MyDrive/WorkingMemory').to(agent_lever.device)
    n_games = 200
    score_history = []
    #
    agent_lever.load_models()
    # # agent_attention.load_models()
    Transformer.load_state_dict(T.load("transformer1_td3"))
    #

    l=0
    count_theta = 0

    A1_matrix_sum = np.zeros((n_games, 30, 4, 4))
    A2_matrix_sum = np.zeros((n_games, 30, 4, 4))

    while l < n_games:
        r = 0
        observation = env.reset()
        env.cue_position = 'left'
        env.proportion = 1.0

        env.orientation_change = 0
        env.change_true = 0
        env.change_index = 0
        # print('observation',observation.shape)

        ### Used for multiple reasons including reward computation and state construction ###
        ### Need positional embeddings ###
        patch_length = 4
        w = 25
        h = 25
        dff = 512
        # obs = np.zeros((patch_length, w, h, 3))
        # c = 0
        # for i in range(10):
        #     for j in range(10):
        #         # Generate varying frequency for the 2D sine wave based on patch index
        #         freq_x = (i + 1) * 2 * np.pi / w
        #         freq_y = (j + 1) * 2 * np.pi / h
        #         x = np.linspace(0, freq_x, w)
        #         y = np.linspace(0, freq_y, h)
        #         xv, yv = np.meshgrid(y, x)
        #         sine_wave = np.sin(xv + yv)

        #         # Normalize sine wave to range [0, 1] and scale it to the patch's dynamic range
        #         sine_wave = (sine_wave - np.min(sine_wave)) / (np.max(sine_wave) - np.min(sine_wave))
        #         sine_wave = sine_wave[:, :, np.newaxis]  # Add channel dimension

        #         obs[c, :, :, :] = observation[i * w:(i + 1) * w, j * h:(j + 1) * h, :] / 25.0 + sine_wave / 1.0
        #         c += 1


        C1 = T.zeros(1, patch_length, dff).to(device)
        M1 = T.zeros(1, patch_length, dff).to(device)
        H1 = T.zeros(1, patch_length, dff).to(device)
        N1 = T.zeros(1, patch_length, dff).to(device)

        C2 = T.zeros(1, patch_length, dff).to(device)
        M2 = T.zeros(1, patch_length, dff).to(device)
        H2 = T.zeros(1, patch_length, dff).to(device)
        N2 = T.zeros(1, patch_length, dff).to(device)

        # C3 = T.zeros(1, patch_length, dff).to(device)
        # M3 = T.zeros(1, patch_length, dff).to(device)
        # H3 = T.zeros(1, patch_length, dff).to(device)
        # N3 = T.zeros(1, patch_length, dff).to(device)

        # EZ = T.zeros(1).to(device)
        # q = T.zeros(1).to(device)

        done = False
        score = 0
        while not done:

            ### state derived from sequence of observations ###
            state = T.tensor(observation, dtype=T.float).to(agent_lever.critic_1.device)
            state = state.view(50,50)
            state = state.view(2, 25, 2, 25)
            state = state.permute(0,2,1,3).reshape(4,25*25)

            ### this is the only location where these inputs are needed. The lever agent operates on the output of the transformer ###
            C1, M1, H1, N1, C2, M2, H2, N2, _, A1, A2 = Transformer(state, C1, M1, H1, N1, C2, M2, H2, N2)
            # print("A1",A1)
            A1_matrix_sum[l, env.t, :, :] = A1.view(4, 4).detach().cpu().numpy()

            lever_action  = agent_lever.choose_action(H2.detach().cpu().numpy().reshape(1, -1))

            # Sample from these probabilities
            if l < 26:
                sampled_index = np.random.choice(len(lever_action), p=lever_action)
                lever_action_buffer = np.zeros(2)
                lever_action_buffer[sampled_index] = 1
            else:
                sampled_index = np.argmax(lever_action)
                lever_action_buffer = np.zeros(2)
                lever_action_buffer[sampled_index] = 1

            ### FORCE WAIT ###
            sampled_index = 0

            observation_, reward, done, _ = env.step(sampled_index)


            ## env.t is the next time step: Remember this ###
            agent_lever.remember(observation.reshape(1, -1), lever_action_buffer, reward, observation_.reshape(1, -1), done, l)
            # Increment memory counter
            agent_lever.memory.mem_cntr[l] += 1

            score += reward
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-1000:])
        print('episode ', l, 'score %.2f' % score,
              'trailing 100 games avg %.3f' % avg_score, 'actions', lever_action)

        l += 1
        count_theta += 1


        if l == buffer_size:
            print("Hi")
            agent_lever.save_models()
            Transformer.save_checkpoint()

            Transformer = agent_lever.learn(Transformer)

            agent_lever.memory = ReplayBuffer(50, input_dim, 2)

            if count_theta >= 1000:
              if avg_score >= 0.85:
                env.theta -= 3
                count_theta = 0
                print("THETA CHANGE!", env.theta)
                score_history = []
            print("THETA", env.theta)

            l = 0

        np.save("A1_Data", A1_matrix_sum)

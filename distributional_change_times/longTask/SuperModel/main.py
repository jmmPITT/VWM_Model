
import gymnasium
import numpy as np
import torch
from MATenv import *
from agent_planner import Agent as AgentPlanner
import torch as T
from network_sensor2 import *

import gymnasium
import numpy as np
import torch
import torch as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import h5py

import gymnasium
import numpy as np
import torch
import torch as T



if __name__ == '__main__':
    env = 0
    ### Trial Max Length ###
    T_end = 31
    input_dim = (50*50)
    # modded_state_dim = 8
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    # agent_lever = AgentPlanner(alpha=0.0001, beta=0.0001,
    #                         input_dims=input_dim, tau=0.01,
    #                         env=env, batch_size=32, max_size=10000,
    #                         n_actions=2)



    Transformer = TransformerNetwork(0.000001).to(device)
    actor_old = ActorNetwork(0.00001, n_actions=2, name='actor', chkpt_dir='td3_MAT').to(device)
    actor = ActorNetwork(0.00001, n_actions=2, name='actor', chkpt_dir='td3_MAT').to(device)
    critic = CriticNetwork(0.00001, name='critic', chkpt_dir='td3_MAT').to(device)
    Qnet = QNetwork(0.00001, name='qNET', chkpt_dir='td3_MAT').to(device)

    eps = 0.2

    n_games = 150000
    score_history = []
    env = ChangeDetectionEnv()

    # agent_lever.load_models()
    # # agent_attention.load_models()
    # Transformer.load_state_dict(T.load("transformer1_td3"))

    cue_array = 0
    target_array = 0
    for i in range(n_games):
        state = env.reset()
        # print('state',state.shape)
        ### Construct the observation to the belief model (depends on attention) ###
        ### adding a positional embedding of length 4 ###
        ### added a temporal embedding of length 7 ###
        ### now each embedding (token) is 15 dimensional ###

        state = torch.tensor(state, dtype=torch.float32).view(50, 50).to(device)  # Add batch dimension
        C1 = torch.zeros(16, Transformer.dff1, requires_grad=True).to(
            device)
        C2 = torch.zeros(4, Transformer.dff2, requires_grad=True).to(
            device)

        GAE = 0.0
        loss = 0.0

        state = T.tensor(state.reshape(50, 50), dtype=T.float).to(device)

        done = False
        score = 0
        while not done:

            ### state derived from sequence of observations ###
            # m = T.tensor(mask, dtype=T.float).to(agent_lever.critic_1.device)


            ### this is the only location where these inputs are needed. The lever agent operates on the output of the transformer ###
            Z, C1, C2, Z2, C1DOWN = Transformer.encoder(state, C1, C2)
            V = critic(C2, Z2, C1DOWN)

            logits = actor(C2, Z2, C1DOWN)
            old_logits = actor_old(C2, Z2, C1DOWN)

            dist = torch.distributions.Categorical(logits=logits)  # batch of 1 distribution
            old_dist = torch.distributions.Categorical(logits=old_logits)

            sample = dist.sample()  # → shape [1]
            action_idx = sample.item()  # scalar 0–3
            Q = Qnet(C2, Z2, C1DOWN, action_idx)
            # 2) log‐probs of the taken actions
            logp = dist.log_prob(sample)  # [B]
            with torch.no_grad():
                logp_old = old_dist.log_prob(sample)  # [B]

            next_state, reward_env, done, _ = env.step(action_idx)


            next_state = T.tensor(next_state.reshape(50, 50), dtype=T.float).to(device)

            with torch.no_grad():
                _, _, C2_, Z2_, C1DOWN_ = Transformer.encoder(next_state, C1, C2)

                V_next = critic(C2_.detach(), Z2_.detach(), C1DOWN_.detach())

            # print("V_next",V_next.shape)
            if done == False:
                delta_t = Q - V.view(-1)
                flag = 1
            elif done == True:
                delta_t = Q - V.view(-1)
                flag = 0
            GAE = GAE + delta_t.view(-1).detach()
            # print("GAE shape",GAE.shape)
            # 3) probability ratio (new / old)
            ratio = torch.exp(logp - logp_old)  # [B]

            # 4) clipped and unclipped objectives
            surr1 = ratio * GAE  # [B]
            surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * GAE

            # 5) take the element‐wise minimum, then negate
            actor_loss = -torch.min(surr1, surr2)
            entropy = dist.entropy().mean()

            zero_target = torch.zeros_like(delta_t)
            critic_loss = F.mse_loss(V, reward_env+V_next*flag)
            Q_loss = F.mse_loss(Q, reward_env+V_next*flag)

            loss += actor_loss + critic_loss - 0.001*entropy + Q_loss

            score += reward_env
            state = next_state

        Transformer.optimizer.zero_grad()
        critic.optimizer.zero_grad()
        actor.optimizer.zero_grad()
        Qnet.optimizer.zero_grad()
        loss.backward()
        Transformer.optimizer.step()
        actor.optimizer.step()
        critic.optimizer.step()
        Qnet.optimizer.step()
        # truncate graph but keep state
        # C1 = C1.detach().requires_grad_(True)
        # C2 = C2.detach().requires_grad_(True)


        if i % 200 == 0:
            # 1) Define a checkpoint dict
            checkpoint = {
                'transformer': Transformer.state_dict(),
                'actor': actor.state_dict(),
                'actor_old': actor_old.state_dict(),
                'critic': critic.state_dict(),
            }

            # 2) Make sure your checkpoint directory exists
            save_path = 'checkpoint.pth'
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 3) Save to disk
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint to {save_path}")

            print('Theta!', env.theta)

        with torch.no_grad():
            actor_old.load_state_dict(actor.state_dict())

        score_history.append(score)
        avg_score = np.mean(score_history[-1000:])

        if avg_score > 0.85 and len(score_history)>1000:
            env.theta -= 3
            score_history = []
            print('New Theta!',env.theta)


        print('episode ', i, 'score %.2f' % score,
                'trailing 100 games avg %.3f' % avg_score, 'actions',F.softmax(logits,dim=-1))







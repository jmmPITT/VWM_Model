import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.buffer import *
import torch as T
import torch.nn.functional as F
import numpy as np
from common.network_sensor2 import *
class Agent:
    """Agent implementation for reinforcement learning with distributional RL.
    
    This agent uses a modified TD3 architecture with a transformer network
    for state representation and distributional critics for value estimation.
    """
    def __init__(self, alpha, beta, input_dims, tau, env,
                 gamma=0.999, update_actor_interval=1, warmup=1000,
                 n_actions=4, max_size=100000, layer1_size=64,
                 layer2_size=32, batch_size=100, noise=0.01):
        self.gamma = gamma
        self.tau = tau
        self.max_action = 10
        self.min_action = -10
        # Replay buffer now uses the updated ReplayBuffer class for PER
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.num_particles = 10

        # Calculate the maximum possible return based on the discount factor
        self.V_max = 1  # As calculated previously

        # Generate the list of particles
        self.particles = [i * (self.V_max / (self.num_particles - 1)) for i in range(self.num_particles)]

        # Print the particles to confirm
        print(self.particles)

        self.spacing = self.particles[1] - self.particles[0]
        print('spacing', self.spacing)

        self.eta = 1.0

        self.actor = ActorNetwork(alpha, input_dims=input_dims, hidden_dim=128, fc1_dims=64, fc2_dims=32,
                                  n_actions=n_actions, name='actor_planner', chkpt_dir='td3_MAT').to(self.device)
        self.critic_1 = CriticNetwork(beta, input_dims=input_dims, hidden_dim=256, fc1_dims=128, fc2_dims=64,
                                      n_actions=n_actions,
                                      name='critic_1_planner', chkpt_dir='td3_MAT').to(self.device)

        self.target_actor = ActorNetwork(alpha, input_dims=input_dims, hidden_dim=128, fc1_dims=64, fc2_dims=32,
                                         n_actions=n_actions, name='target_actor_planner', chkpt_dir='td3_MAT').to(
            self.device)
        self.target_critic_1 = CriticNetwork(beta, input_dims=input_dims, hidden_dim=256, fc1_dims=128, fc2_dims=64,
                                             n_actions=n_actions,
                                             name='target_critic_1_planner', chkpt_dir='td3_MAT').to(self.device)

        # self.target_critic_2 = CriticNetwork(beta, input_dims=input_dims, hidden_dim = 128, fc1_dims=64, fc2_dims=32, n_actions=n_actions,
        #     name='target_critic_2_planner', chkpt_dir='td3_MAT').to(self.device)
        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, transformer_state, obs):
        if self.time_step < self.warmup*0:
            print(self.time_step,self.warmup*1)
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.device)
        else:
            transformer_state = T.tensor(transformer_state, dtype=T.float).to(self.actor.device)
            obs = T.tensor(obs, dtype=T.float).to(self.actor.device)

            # mask = T.tensor(mask, dtype=T.float).to(self.actor.device)
            # transformer_state = self.Transformer(state, mask)
            mu = self.actor.forward(transformer_state, obs).to(self.actor.device)[0]
            # print('mu', mu)

        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise, size=mu.shape), dtype=T.float).to(self.device)
        # mu_prime[3:] = mu[3:] + T.tensor(np.random.normal(scale=self.noise*10, size=16), dtype=T.float).to(self.device)

        # print('mu prime',mu_prime)
        mu_prime = T.clamp(mu_prime, self.min_action, self.max_action)
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, mask, action, reward, new_state, next_mask, done, t, cue, target):
        self.memory.store_transition(state, mask, action, reward, new_state, next_mask, done, t, cue, target)

    def learn(self, Transformer):
        if self.memory.mem_cntr < self.batch_size*1:
            return Transformer

        state, mask, action, reward, new_state, next_mask, done, t, cue, target, indices, weights = \
            self.memory.sample_buffer(self.batch_size)


        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        mask_ = T.tensor(next_mask, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        mask = T.tensor(mask, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        ts = T.tensor(t, dtype=T.int32).to(self.critic_1.device)
        cues = T.tensor(cue, dtype=T.float).to(self.critic_1.device)
        target = T.tensor(target, dtype=T.float).to(self.critic_1.device)

        action = F.softmax(action,dim=-1)

        transformer_state, _, _, _, _ = Transformer(state, ts-1, mask)
        transformer_state_, _, _, _, _ = Transformer(state_, ts, mask_)

        target_actions = self.target_actor.forward(transformer_state_, state_)
        target_actions = F.softmax(target_actions,dim=-1)

        # target_actions = target_actions + \
        #                  T.clamp(T.tensor(np.random.normal(scale=0.1)), -0.5, 0.5)
        noise = T.clamp(T.tensor(np.random.normal(scale=0.001, size=target_actions.shape), dtype=T.float), -0.5, 0.5).to(self.device)
        target_actions = target_actions + noise

        # might break if elements of min and max are not all equal
        target_actions = T.clamp(target_actions, self.min_action, self.max_action)

        ### future distribution ###
        # q1_ = self.target_critic_1.forward(transformer_state_, target_actions, state_)
        with torch.no_grad():  # Gradient computation is disabled in this block
            q1_ = self.target_critic_1.forward(transformer_state_, target_actions, state_)

        q1 = self.critic_1.forward(transformer_state, action, state)
        # q2 = self.critic_2.forward(transformer_state, action, state)

        # q1_[done] = 0.0
        # q2_[done] = 0.0

        q1_ = q1_.view(-1,2)
        # q2_ = q2_.view(-1,2)

        phat0 = torch.zeros(q1_.size(0), 1, device=q1_.device)
        phat1 = torch.zeros_like(phat0)
        # print('hi Fu chatgpt')

        # Projection Step #
        gTheta0 = reward + 0*(1-done.float())
        gTheta1 = reward + 1*(1-done.float())

        phat0 += q1_[:, 0].unsqueeze(1) * (gTheta0 < 0.5).float().unsqueeze(1)
        phat1 += q1_[:, 0].unsqueeze(1) * (1 - (gTheta0 < 0.5).float()).unsqueeze(1)

        phat0 += q1_[:, 1].unsqueeze(1) * (gTheta1 < 0.5).float().unsqueeze(1)
        phat1 += q1_[:, 1].unsqueeze(1) * (1 - (gTheta1 < 0.5).float()).unsqueeze(1)

        # Target Distribution #
        target_distribution = torch.cat((phat0, phat1), dim=1)
        # print('hi Fu chatgpt5')
        # Assuming target_distribution is already created and has shape [batch_size, 2]

        # Sum the probabilities along dim=1 to check if they sum to 1, keep the dimension to get batchx1 shape
        prob_sum = target_distribution.sum(dim=1, keepdim=True)

        # print("Sum of probabilities for each sample in the batch (should be close to 1):")
        # print(prob_sum)

        # KL #
        # q_prob = F.softmax(q1, dim=1)  # Apply softmax if q is not already probabilities
        log_q_prob = q1.log()  # Convert to log probabilities for KL divergence
        # print('hi Fu chatgpt 6')

        # kl_loss = F.kl_div(log_q_prob, target_distribution, reduction='batchmean')
        # Check for NaN values
        # has_nan = torch.isnan(kl_loss).any()

        ### Policy Loss ###
        with torch.no_grad():
            a0 = torch.zeros(q1_.size(0), 2, device=q1_.device)
            a0[:,0] = 1
            a1 = torch.zeros(q1_.size(0), 2, device=q1_.device)
            a1[:,1] = 1

            Q0_ = self.target_critic_1.forward(transformer_state, a0, state)
            Q1_ = self.target_critic_1.forward(transformer_state, a1, state)
            Q0 = Q0_[:,1]
            Q1 = Q1_[:,1]
            p = F.softmax(self.target_actor.forward(transformer_state, state),dim=-1)
            pa0_ = p[:,0]
            pa1_ = p[:,1]
            KQ = T.log( pa0_*T.exp(Q0/self.eta) + pa1_*T.exp(Q1/self.eta) )

        action_logits = self.actor.forward(transformer_state, state)
        p_actions = F.softmax(action_logits,dim=-1)
        pa0 = p_actions[:,0]
        pa1 = p_actions[:,1]
        LPol = T.mean(pa0_*( T.exp( Q0/self.eta - KQ ) * T.log( pa0 ) ) + pa1_*( T.exp( Q1/self.eta - KQ ) * T.log( pa1 ) ))
        # Compute the entropy for each distribution in the batch
        entropy_per_sample = -T.sum(p_actions * T.log(p_actions + 1e-6),
                                    dim=-1)  # Adding a small constant to avoid log(0)

        # Compute the mean entropy over the batch
        H = T.mean(entropy_per_sample)
        # print('Lpol',LPol.shape)


        # print(has_nan)
        Transformer.optimizer.zero_grad()
        self.critic_1.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        # self.critic_2.optimizer.zero_grad()
        kl_loss = F.kl_div(log_q_prob, target_distribution, reduction='batchmean')
        critic_loss = kl_loss - LPol - 0.1*H
        critic_loss.backward()
        Transformer.optimizer.step()
        self.actor.optimizer.step()
        self.critic_1.optimizer.step()
        # self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        # if self.learn_step_cntr % self.update_actor_iter != 0:
        #     return Transformer
        #
        # self.actor.optimizer.zero_grad()
        # Transformer.optimizer.zero_grad()
        # # with torch.no_grad():
        # #     transformer_state = Transformer(state, mask)
        # transformer_state, prev_cue, next_action, next_reward, _ = Transformer(state, ts-1, mask)
        #
        # # Convert logits to probabilities
        # logits = self.actor.forward(transformer_state, state)  # assuming this outputs logits
        # # probabilities = F.softmax(logits[:,0:4], dim=-1)
        #
        # # Calculate entropy
        # # entropy = -(probabilities * torch.log(probabilities + 1e-6)).sum(-1).mean()
        # # entropy_coefficient = 0.0001  # This is a hyperparameter you can tune
        #
        # # Calculate the L2 norm of the logits (regularization term)
        # logits_l2_norm = T.mean(logits[:,:] ** 2)
        # regularization_coefficient = 0.0  # This is a hyperparameter you can tune
        #
        # self.actor.optimizer.zero_grad()
        # Transformer.optimizer.zero_grad()
        #
        # actor_q1_loss = self.critic_1.forward(transformer_state, logits, state)
        # actor_q1_LOSS = actor_q1_loss[:,1]
        # actor_loss = -T.mean(actor_q1_LOSS) + regularization_coefficient * logits_l2_norm
        # actor_loss.backward()
        # Transformer.optimizer.step()
        # self.actor.optimizer.step()
        self.update_network_parameters()

        with T.no_grad():
            # Compute TD errors for updating priorities
            # TD error is the absolute difference between target Q values and current Q values
            td_error = T.abs(target_distribution[:,1] - q1[:,1]).detach().cpu().numpy()
            # Update priorities in the replay buffer
            # print('ayo',indices.shape)
            # print(td_error.shape)
            self.memory.update_priorities(indices, td_error)

        return Transformer


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        # critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        # target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        # critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        # target_critic_2_state_dict = dict(target_critic_2_params)



        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau * critic_1_state_dict[name].clone() + \
                                        (1 - tau) * target_critic_1_state_dict[name].clone()

        # for name in critic_2_state_dict:
        #     critic_2_state_dict[name] = tau * critic_2_state_dict[name].clone() + \
        #                                 (1 - tau) * target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        # self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        # self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        # self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        # self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        # self.target_critic_2.load_checkpoint()

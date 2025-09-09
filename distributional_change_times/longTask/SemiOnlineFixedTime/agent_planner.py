import os
import torch as T
import torch.nn.functional as F
import numpy as np
import torch
from network_sensor2 import *
from buffer import *
class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
                 gamma=0.99999999999, update_actor_interval=1, warmup=1000,
                 n_actions=4, max_size=100000, layer1_size=64,
                 layer2_size=32, batch_size=100, noise=0.01):
        self.gamma = 1.0
        self.tau = tau
        self.max_action = 10
        self.min_action = -10
        self.dff = 512
        self.patch_num = 4
        # Replay buffer now uses the updated ReplayBuffer class for PER
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.buffer_size = 32

        # Define the number of particles
        self.num_particles = 2

        # Calculate the maximum possible return based on the discount factor
        self.V_max = 1  # As calculated previously

        # Generate the list of particles
        self.particles = [i * (self.V_max / (self.num_particles - 1)) for i in range(self.num_particles)]

        # Print the particles to confirm
        print(self.particles)

        self.spacing = self.particles[1] - self.particles[0]
        print('spacing',self.spacing)

        self.eta = 1.0

        self.actor = ActorNetwork(alpha, input_dims=input_dims, hidden_dim = 512, fc1_dims=256, fc2_dims=128, n_actions=n_actions, name='actor_planner', chkpt_dir='td3_MAT').to(self.device)
        self.critic_1 = CriticNetwork(beta, input_dims=input_dims, hidden_dim = 512, fc1_dims=256, fc2_dims=128, n_actions=n_actions,
            name='critic_1_planner', chkpt_dir='td3_MAT').to(self.device)

        self.target_actor = ActorNetwork(alpha, input_dims=input_dims, hidden_dim = 512, fc1_dims=256, fc2_dims=128, n_actions=n_actions, name='target_actor_planner', chkpt_dir='td3_MAT').to(self.device)
        self.target_critic_1 = CriticNetwork(beta, input_dims=input_dims, hidden_dim = 512, fc1_dims=256, fc2_dims=128, n_actions=n_actions,
            name='target_critic_1_planner', chkpt_dir='td3_MAT').to(self.device)


        self.noise = noise
        self.update_network_parameters(tau=tau)

    def choose_action(self, transformer_state):

        if self.time_step < self.warmup * 0:
            print(self.time_step, self.warmup * 1)
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.device)
            mu = F.softmax(mu, dim=-1)
        else:
            transformer_state = T.tensor(transformer_state, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(transformer_state)

            ### Probably should cast target actions to one-hot ###
            # Get the indices of the maximum values (argmax) along the action dimension
            action_indices = torch.argmax(mu, dim=-1)

            # Get the number of possible actions and batch size
            batch_size, num_actions = mu.view(-1, self.n_actions).shape

            # Create a one-hot encoded tensor
            target_actions_onehot = torch.zeros_like(mu)
            target_actions_onehot.view(-1)[
                action_indices + torch.arange(batch_size, device=mu.device) * num_actions] = 1


            mu = mu.to(self.actor.device)[0]

        self.time_step += 1
        return mu.cpu().detach().numpy()

    def remember(self, state, action, reward, next_state, done, l):
        self.memory.store_transition(state, action, reward, next_state, done, l)

    def learn(self, Transformer):
        # if self.memory.mem_cntr[0] < self.batch_size * 1:
        #     return Transformer
        l = 0

        state, action, reward, next_state, done = \
            self.memory.sample_buffer(self.batch_size)

        print('train_sample size',state.shape)

        reward = T.tensor(reward, dtype=T.float)

        patch_length = self.patch_num
        dff = self.dff
        input_dims = 25*25

        C1 = T.zeros(self.buffer_size, patch_length, dff).to(self.critic_1.device)
        M1 = T.zeros(self.buffer_size, patch_length, dff).to(self.critic_1.device)
        H1 = T.zeros(self.buffer_size, patch_length, dff).to(self.critic_1.device)
        N1 = T.zeros(self.buffer_size, patch_length, dff).to(self.critic_1.device)

        C2 = T.zeros(self.buffer_size, patch_length, dff).to(self.critic_1.device)
        M2 = T.zeros(self.buffer_size, patch_length, dff).to(self.critic_1.device)
        H2 = T.zeros(self.buffer_size, patch_length, dff).to(self.critic_1.device)
        N2 = T.zeros(self.buffer_size, patch_length, dff).to(self.critic_1.device)


        kl_loss = 0
        LPol = 0
        entropy_mean = 0
        cosineDistLossSum = 0
        TDloss = 0
        REMLOSS = 0
        print('ayoooo',self.memory.mem_cntr)
        T_end = 200 + np.random.randint(-1, 3)

        for i in range(max(self.memory.mem_cntr)):
            ### Blocker ###
            mem_cntr = self.memory.mem_cntr

            blocker = torch.tensor([1 if i < mem_cntr[j] else 0 for j in range(self.buffer_size)],
                                   dtype=torch.bool).to(self.critic_1.device)
            blocker_cpu = torch.tensor([1 if i < mem_cntr[j] else 0 for j in range(self.buffer_size)],
                                       dtype=torch.bool)

            # Assuming self.memory.mem_cntr is your numpy integer array of 5 integers
            mem_cntr = self.memory.mem_cntr

            state_i = T.tensor(state[:,i], dtype=T.float).to(self.critic_1.device)
            state_i = state_i.view(-1, 50, 50)
            state_i = state_i.view(-1, 2, 25, 2, 25)
            state_i = state_i.permute(0,1, 3, 2, 4).reshape(-1,4,25*25)
            action_i = T.tensor(action[blocker_cpu,i], dtype=T.float).to(self.critic_1.device)
            reward_i = T.tensor(reward[blocker_cpu,i], dtype=T.float).to(self.critic_1.device)
            done_i = T.tensor(done[blocker_cpu,i]).to(self.critic_1.device)
            next_state_i = T.tensor(next_state[:,i], dtype=T.float).to(self.critic_1.device)
            next_state_i = next_state_i.view(-1, 50, 50)
            next_state_i = next_state_i.view(-1, 2, 25, 2, 25)
            next_state_i = next_state_i.permute(0,1, 3, 2, 4).reshape(-1,4,25*25)
            # print('next_state_i',next_state_i.shape)
            C1, M1, H1, N1, C2, M2, H2, N2, Z = Transformer(state_i, C1, M1, H1, N1, C2, M2, H2, N2)


            with torch.no_grad():  # Gradient computation is disabled in this block
                _, _, _, _, _, _, H2_, _, _ = Transformer(next_state_i, C1, M1, H1, N1, C2, M2, H2, N2)

                target_actions = self.target_actor.forward(H2_[blocker].detach())


                ### Probably should cast target actions to one-hot ###
                # Get the indices of the maximum values (argmax) along the action dimension
                action_indices = torch.argmax(target_actions, dim=-1)

                # Get the number of possible actions and batch size
                batch_size, num_actions = target_actions.view(-1, self.n_actions).shape

                # Create a one-hot encoded tensor
                target_actions_onehot = torch.zeros_like(target_actions)
                target_actions_onehot.view(-1)[action_indices + torch.arange(batch_size, device=target_actions.device) * num_actions] = 1

                q1_ = self.target_critic_1.forward(H2_[blocker].detach(), target_actions_onehot.detach())

            q1 = self.critic_1.forward(H2[blocker], action_i)
            # print('q1_',q1_.shape)

            q1_ = q1_.view(-1,self.num_particles)
            # print('q1_',q1_.shape)

            phat = torch.zeros(q1_.size(0), self.num_particles, device=q1_.device)
            # print('phat',phat.shape)
            for j in range(self.num_particles):
                gTheta = reward_i + self.particles[j] * (1 - done_i.float()) * self.gamma
                phat[:, -1] += q1_[:, j] * (gTheta-self.particles[-1] >= self.spacing/2.0).float()
                phat[:, 0] += q1_[:, j] * (gTheta-self.particles[0] <= -self.spacing/2.0).float()
                for k in range(self.num_particles):
                    phat[:, k] += q1_[:, j] * (torch.abs(gTheta-self.particles[k])-0.0001 <= self.spacing/2.0).float()

            # for j in range(self.num_particles):
            #     gTheta = T.sum(reward[blocker_cpu, i:].to(self.critic_1.device), dim=-1).view(-1)
            #     phat[:, j] += (torch.abs(gTheta[:]-self.particles[j])-0.0001 <= self.spacing/2.0).float()
            # phat[:, -1] += (gTheta-self.particles[-1] >= self.spacing/2.0).float()
            # phat[:, 0] += (gTheta-self.particles[0] <= -self.spacing/2.0).float()

            target_distribution = phat
            log_q_prob = q1.log()

            # log_q_prob = q1.log()  # Convert to log probabilities for KL divergence
            # huber_loss = F.binary_cross_entropy(q1, target_distribution)

            ### Policy Loss ###
            with torch.no_grad():
                a0 = torch.zeros(q1_.size(0), 2, device=q1_.device)
                a0[:,0] = 1
                a1 = torch.zeros(q1_.size(0), 2, device=q1_.device)
                a1[:,1] = 1

                Q0_ = self.target_critic_1.forward(H2[blocker].detach(), a0)
                Q1_ = self.target_critic_1.forward(H2[blocker].detach(), a1)

                ### This still needs work ###
                Q0 = torch.zeros(q1_.size(0), 1, device=q1_.device)
                Q1 = torch.zeros(q1_.size(0), 1, device=q1_.device)
                for j in range(self.num_particles):

                    Q0[:,0] += Q0_[:,j]*self.particles[j]
                    Q1[:,0] += Q1_[:,j]*self.particles[j]

                p = self.target_actor.forward(H2[blocker].detach())
                pa0_ = p[:,0]
                pa1_ = p[:,1]
                KQ = T.log( pa0_*T.exp(Q0[:,0]/self.eta) + pa1_*T.exp(Q1[:,0]/self.eta))

            action_probs = self.actor.forward(H2[blocker])

            p_actions = action_probs
            pa0 = p_actions[:,0]
            pa1 = p_actions[:,1]

            LPol_ = pa0_*( T.exp( Q0[:,0]/self.eta - KQ ) * T.log( pa0 ) ) + pa1_*( T.exp( Q1[:,0]/self.eta - KQ ) * T.log( pa1 ) )


            gTheta = T.sum(reward[blocker_cpu, i:].to(self.critic_1.device), dim=-1)

            E = T.zeros(T.sum(blocker), 1).to(self.critic_1.device)
            for j in range(self.num_particles):
                E[:] += (q1[:, j].view(-1, 1) * self.particles[j]).view(-1, 1)
            # Advantage = (gTheta - E.detach())


            if (i) % 100 == 0:
                print('E', E.view(-1))


            LPol += T.mean(LPol_)



            entropy_per_sample = -T.sum(action_probs * T.log(action_probs + 1e-6),
                                        dim=-1)  # Adding a small constant to avoid log(0)

            # Compute the mean entropy over the batch
            entropy_mean += T.mean(entropy_per_sample)


            # kl_loss += huber_loss
            kl_loss_ = F.kl_div(log_q_prob, target_distribution, reduction='none')
            kl_loss_ = kl_loss_.sum(dim=1, keepdim=True)  # Sum over all dimensions except batch
            kl_loss += T.mean(kl_loss_)

            if (i+1) % T_end == 0:
                Transformer.optimizer.zero_grad()
                self.critic_1.optimizer.zero_grad()
                self.actor.optimizer.zero_grad()
                critic_loss = 10.1 * kl_loss/T_end - 1.1 * LPol/T_end - 0.1 * entropy_mean/T_end

                critic_loss.backward()

                # Clip the gradients (norm-based)
                # max_grad_norm = 50.0
                # torch.nn.utils.clip_grad_norm_(Transformer.parameters(), max_grad_norm)
                # torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_grad_norm)
                # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)

                Transformer.optimizer.step()
                self.actor.optimizer.step()
                self.critic_1.optimizer.step()
                self.learn_step_cntr += 1

                self.update_network_parameters()

                C1 = C1.detach().clone().requires_grad_()
                M1 = M1.detach().clone().requires_grad_()
                H1 = H1.detach().clone().requires_grad_()
                N1 = N1.detach().clone().requires_grad_()

                C2 = C2.detach().clone().requires_grad_()
                M2 = M2.detach().clone().requires_grad_()
                H2 = H2.detach().clone().requires_grad_()
                N2 = N2.detach().clone().requires_grad_()

                # C3 = C3.detach().clone().requires_grad_()
                # M3 = M3.detach().clone().requires_grad_()
                # H3 = H3.detach().clone().requires_grad_()
                # N3 = N3.detach().clone().requires_grad_()






                kl_loss = 0
                LPol = 0
                entropy_mean = 0
                TDloss = 0
                REMLOSS = 0

        if (i+1) % T_end  != 0:
            # print("WHAT THE FUCK!!!!!????", i)
            Transformer.optimizer.zero_grad()
            self.critic_1.optimizer.zero_grad()
            self.actor.optimizer.zero_grad()
            critic_loss = 10.1 * kl_loss/ ((i + 1) % T_end) - 1.1 * LPol/ ((i + 1) % T_end) - 0.1 * entropy_mean/ ((i + 1) % T_end)

            print('kl_loss',kl_loss*10.1)
            print('Lpol',LPol*1.0)
            print('entropy',0.1*entropy_mean)
            print('REMLOSS',0.1*REMLOSS)

            critic_loss.backward()


            # Clip the gradients (norm-based)
            # max_grad_norm = 50.0
            # torch.nn.utils.clip_grad_norm_(Transformer.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)


            Transformer.optimizer.step()
            self.actor.optimizer.step()
            self.critic_1.optimizer.step()
            self.learn_step_cntr += 1

        self.update_network_parameters()

        return Transformer


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)



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

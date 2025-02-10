import os
import torch as T
import torch.nn.functional as F
import numpy as np
import torch
from buffer import *
from network_sensor2 import *

class Agent:
    """
    Distributional Actor-Critic Agent for planning using transformer-based networks.

    This agent utilizes a replay buffer, actor/critic networks, and target networks.
    It employs a particle-based return distribution and maintains multiple internal
    states for processing patch-based observations.
    """
    def __init__(self, alpha, beta, input_dims, tau, env,
                 gamma=0.99, update_actor_interval=1, warmup=1000,
                 n_actions=4, max_size=100000, layer1_size=64,
                 layer2_size=32, batch_size=100, noise=0.01):
        self.gamma = gamma
        self.tau = tau
        self.max_action = 10
        self.min_action = -10
        self.dff = 512
        self.patch_num = 9

        # Initialize replay buffer (using the updated ReplayBuffer class)
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.buffer_size = 8

        # Define the number of particles for distributional estimation
        self.num_particles = 30

        # Maximum possible return based on the discount factor (as previously calculated)
        self.V_max = 5

        # Generate the list of particles
        self.particles = [i * (self.V_max / (self.num_particles - 1)) for i in range(self.num_particles)]
        print(self.particles)

        # Compute spacing between particles
        self.spacing = self.particles[1] - self.particles[0]
        print('spacing', self.spacing)

        self.eta = 1.0

        # Initialize actor and critic networks (and their target networks)
        self.actor = ActorNetwork(alpha, input_dims=input_dims, hidden_dim=512, fc1_dims=256, fc2_dims=128,
                                  n_actions=n_actions, name='actor_planner', chkpt_dir='td3_MAT').to(self.device)
        self.critic_1 = CriticNetwork(beta, input_dims=input_dims, hidden_dim=512, fc1_dims=256, fc2_dims=128,
                                      n_actions=n_actions, name='critic_1_planner', chkpt_dir='td3_MAT').to(self.device)
        self.target_actor = ActorNetwork(alpha, input_dims=input_dims, hidden_dim=512, fc1_dims=256, fc2_dims=128,
                                         n_actions=n_actions, name='target_actor_planner', chkpt_dir='td3_MAT').to(self.device)
        self.target_critic_1 = CriticNetwork(beta, input_dims=input_dims, hidden_dim=512, fc1_dims=256, fc2_dims=128,
                                             n_actions=n_actions, name='target_critic_1_planner', chkpt_dir='td3_MAT').to(self.device)

        # Uncomment below if using a transformer trajectory predictor
        # self.DMT = TransformerTrajectoryPredictor(input_dims=self.dff * self.patch_num)

        self.noise = noise
        self.update_network_parameters(tau=tau)

    def choose_action(self, transformer_state):
        """
        Choose an action based on the transformer state.

        During warmup (currently disabled via multiplication by 0), a random action is produced.
        Otherwise, the actor network outputs probabilities (mu) and logits.
        The first element from the batch is used.
        """
        if self.time_step < self.warmup * 0:
            print(self.time_step, self.warmup * 1)
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.device)
            mu = F.softmax(mu, dim=-1)
        else:
            transformer_state = T.tensor(transformer_state, dtype=T.float).to(self.actor.device)
            mu, logits = self.actor.forward(transformer_state)
            # (Optional: Code for one-hot conversion of target actions is commented out)
            mu = mu.to(self.actor.device)[0]
            logits = logits.to(self.actor.device)[0]

        self.time_step += 1
        return mu.cpu().detach().numpy(), logits.cpu().detach().numpy()

    def remember(self, state, action, reward, next_state, done, l):
        """
        Store a transition in the replay buffer.

        Parameters:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Terminal flag.
            l: Index (e.g., episode index) for storing the transition.
        """
        self.memory.store_transition(state, action, reward, next_state, done, l)

    def learn(self, Transformer):
        """
        Update networks using a batch of transitions sampled from the replay buffer.

        Processes each time step in the batch, computes the distributional target,
        KL divergence, policy loss, and updates the networks every T_end steps.

        Parameters:
            Transformer: The transformer network used for processing states.

        Returns:
            Updated Transformer network.
        """
        l = 0

        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        print('train_sample size', state.shape)

        # Convert reward to a tensor (other conversions are commented out)
        reward = T.tensor(reward, dtype=T.float)
        # Note: Other tensors (state, action, next_state, done) are assumed to be handled externally

        patch_length_1 = 9
        dff_1 = 512
        dff_2 = 512

        # Initialize transformer internal states for two layers
        C1 = T.zeros(self.buffer_size, patch_length_1, dff_1).to(self.critic_1.device)
        M1 = T.zeros(self.buffer_size, patch_length_1, dff_1).to(self.critic_1.device)
        H1 = T.zeros(self.buffer_size, patch_length_1, dff_1).to(self.critic_1.device)
        N1 = T.zeros(self.buffer_size, patch_length_1, dff_1).to(self.critic_1.device)

        C2 = T.zeros(self.buffer_size, patch_length_1, dff_2).to(self.critic_1.device)
        M2 = T.zeros(self.buffer_size, patch_length_1, dff_2).to(self.critic_1.device)
        H2 = T.zeros(self.buffer_size, patch_length_1, dff_2).to(self.critic_1.device)
        N2 = T.zeros(self.buffer_size, patch_length_1, dff_2).to(self.critic_1.device)

        # Initialize loss accumulators
        kl_loss = 0
        LPol = 0
        entropy_mean = 0
        cosineDistLossSum = 0
        TDloss = 0
        REMLOSS = 0
        print('ayoooo', self.memory.mem_cntr)
        T_end = 200 + np.random.randint(-1, 3)
        # T_end = 600

        for i in range(max(self.memory.mem_cntr)):
            # Create a boolean mask (blocker) indicating which episodes have a transition at time step i
            mem_cntr = self.memory.mem_cntr
            blocker = torch.tensor([1 if i < mem_cntr[j] else 0 for j in range(self.buffer_size)],
                                   dtype=torch.bool).to(self.critic_1.device)
            blocker_cpu = torch.tensor([1 if i < mem_cntr[j] else 0 for j in range(self.buffer_size)],
                                         dtype=torch.bool)

            # Convert the state at time step i into a tensor patch (assumes 9 patches of 30x30 RGB)
            obs = T.tensor(state[:, i], dtype=T.float).view(-1, 9, 30*30*3).to(self.critic_1.device)
            obsZero = T.tensor(state[:, 0], dtype=T.float).view(-1, 9, 30*30*3).to(self.critic_1.device)

            action_i = T.tensor(action[blocker_cpu, i], dtype=T.float).to(self.critic_1.device)
            reward_i = T.tensor(reward[blocker_cpu, i], dtype=T.float).to(self.critic_1.device)
            done_i = T.tensor(done[blocker_cpu, i]).to(self.critic_1.device)
            next_state_i = T.tensor(next_state[:, i], dtype=T.float).to(self.critic_1.device)

            # Process the observation through the Transformer to update internal states
            C1, M1, H1, N1, C2, M2, H2, N2, Z = Transformer(obs, C1, M1, H1, N1, C2, M2, H2, N2)

            # (Optional: REMLOSS calculation is commented out)
            # REMLOSS += F.mse_loss(Z, obsZero)

            with torch.no_grad():
                _, _, _, _, _, _, H2_, _, _ = Transformer(next_state_i, C1, M1, H1, N1, C2, M2, H2, N2)
                target_actions, _ = self.target_actor.forward(H2_[blocker].detach())
                action_indices = torch.argmax(target_actions, dim=-1)
                batch_size, num_actions = target_actions.view(-1, self.n_actions).shape
                target_actions_onehot = torch.zeros_like(target_actions)
                target_actions_onehot.view(-1)[action_indices + torch.arange(batch_size, device=target_actions.device) * num_actions] = 1
                q1_ = self.target_critic_1.forward(H2_[blocker].detach(), target_actions_onehot.detach())

            q1 = self.critic_1.forward(H2[blocker], action_i)
            q1_ = q1_.view(-1, self.num_particles)

            phat = torch.zeros(q1_.size(0), self.num_particles, device=q1_.device)
            for j in range(self.num_particles):
                gTheta = reward_i + self.particles[j] * (1 - done_i.float()) * self.gamma
                phat[:, -1] += q1_[:, j] * (gTheta - self.particles[-1] >= self.spacing / 2.0).float()
                phat[:, 0] += q1_[:, j] * (gTheta - self.particles[0] <= -self.spacing / 2.0).float()
                for k in range(self.num_particles):
                    phat[:, k] += q1_[:, j] * (torch.abs(gTheta - self.particles[k]) - 0.0001 <= self.spacing / 2.0).float()
            target_distribution = phat

            log_q_prob = q1.log()  # Compute log probabilities for KL divergence

            with torch.no_grad():
                a0 = torch.zeros(q1_.size(0), 5, device=q1_.device)
                a0[:, 0] = 1
                a1 = torch.zeros(q1_.size(0), 5, device=q1_.device)
                a1[:, 1] = 1
                a2 = torch.zeros(q1_.size(0), 5, device=q1_.device)
                a2[:, 2] = 1
                a3 = torch.zeros(q1_.size(0), 5, device=q1_.device)
                a3[:, 3] = 1
                a4 = torch.zeros(q1_.size(0), 5, device=q1_.device)
                a4[:, 4] = 1

                Q0_ = self.target_critic_1.forward(H2[blocker].detach(), a0)
                Q1_ = self.target_critic_1.forward(H2[blocker].detach(), a1)
                Q2_ = self.target_critic_1.forward(H2[blocker].detach(), a2)
                Q3_ = self.target_critic_1.forward(H2[blocker].detach(), a3)
                Q4_ = self.target_critic_1.forward(H2[blocker].detach(), a4)

                Q0 = torch.zeros(q1_.size(0), 1, device=q1_.device)
                Q1 = torch.zeros(q1_.size(0), 1, device=q1_.device)
                Q2 = torch.zeros(q1_.size(0), 1, device=q1_.device)
                Q3 = torch.zeros(q1_.size(0), 1, device=q1_.device)
                Q4 = torch.zeros(q1_.size(0), 1, device=q1_.device)
                for j in range(self.num_particles):
                    Q0[:, 0] += Q0_[:, j] * self.particles[j]
                    Q1[:, 0] += Q1_[:, j] * self.particles[j]
                    Q2[:, 0] += Q2_[:, j] * self.particles[j]
                    Q3[:, 0] += Q3_[:, j] * self.particles[j]
                    Q4[:, 0] += Q4_[:, j] * self.particles[j]

                p, _ = self.target_actor.forward(H2[blocker].detach())
                pa0_ = p[:, 0]
                pa1_ = p[:, 1]
                pa2_ = p[:, 2]
                pa3_ = p[:, 3]
                pa4_ = p[:, 4]
                KQ = T.log(pa0_ * T.exp(Q0[:, 0] / self.eta) + pa1_ * T.exp(Q1[:, 0] / self.eta) +
                           pa2_ * T.exp(Q2[:, 0] / self.eta) + pa3_ * T.exp(Q3[:, 0] / self.eta) +
                           pa4_ * T.exp(Q4[:, 0] / self.eta))

            action_probs, logits = self.actor.forward(H2[blocker])
            p_actions = action_probs
            pa0 = p_actions[:, 0]
            pa1 = p_actions[:, 1]
            pa2 = p_actions[:, 2]
            pa3 = p_actions[:, 3]
            pa4 = p_actions[:, 4]

            LPol_ = (pa0_ * (T.exp(Q0[:, 0] / self.eta - KQ) * T.log(pa0)) +
                     pa1_ * (T.exp(Q1[:, 0] / self.eta - KQ) * T.log(pa1)) +
                     pa2_ * (T.exp(Q2[:, 0] / self.eta - KQ) * T.log(pa2)) +
                     pa3_ * (T.exp(Q3[:, 0] / self.eta - KQ) * T.log(pa3)) +
                     pa4_ * (T.exp(Q4[:, 0] / self.eta - KQ) * T.log(pa4)))

            gTheta = T.sum(reward[blocker_cpu, i:].to(self.critic_1.device), dim=-1)

            E = T.zeros(T.sum(blocker), 1).to(self.critic_1.device)
            for j in range(self.num_particles):
                E[:] += (q1[:, j].view(-1, 1) * self.particles[j]).view(-1, 1)
            Advantage = (gTheta - E.detach())

            if (i) % 100 == 0:
                print('E', E.view(-1))

            actionIndices = torch.argmax(action_i, dim=-1)
            r_theta = action_probs[torch.arange(torch.sum(blocker)), actionIndices]
            eps = 0.2  # Common epsilon value (e.g., for PPO clipping)
            LtCLIP = r_theta * Advantage

            LPol += 0.0001 * T.mean(LtCLIP) + T.mean(LPol_)

            entropy_per_sample_H = -T.sum(action_probs * T.log(action_probs + 1e-6), dim=-1)
            suppression_per_sample = -T.sum(logits ** 2, dim=-1)
            entropy_mean += 0.5 * T.mean(entropy_per_sample_H) + 0.5 * T.mean(suppression_per_sample)

            kl_loss_ = F.kl_div(log_q_prob, target_distribution, reduction='none')
            kl_loss_ = kl_loss_.sum(dim=1, keepdim=True)
            kl_loss += T.mean(kl_loss_)

            if (i + 1) % T_end == 0:
                Transformer.optimizer.zero_grad()
                self.critic_1.optimizer.zero_grad()
                self.actor.optimizer.zero_grad()
                critic_loss = 10.1 * kl_loss / T_end - 1.1 * LPol / T_end - 0.001 * entropy_mean / T_end
                critic_loss.backward()

                Transformer.optimizer.step()
                self.actor.optimizer.step()
                self.critic_1.optimizer.step()
                self.learn_step_cntr += 1

                self.update_network_parameters()

                # Detach hidden states to prevent backpropagation through time
                C1 = C1.detach().clone().requires_grad_()
                M1 = M1.detach().clone().requires_grad_()
                H1 = H1.detach().clone().requires_grad_()
                N1 = N1.detach().clone().requires_grad_()
                C2 = C2.detach().clone().requires_grad_()
                M2 = M2.detach().clone().requires_grad_()
                H2 = H2.detach().clone().requires_grad_()
                N2 = N2.detach().clone().requires_grad_()

                kl_loss = 0
                LPol = 0
                entropy_mean = 0
                TDloss = 0
                REMLOSS = 0

        if (i + 1) % T_end != 0:
            Transformer.optimizer.zero_grad()
            self.critic_1.optimizer.zero_grad()
            self.actor.optimizer.zero_grad()
            critic_loss = 10.1 * kl_loss / ((i + 1) % T_end) - 1.0 * LPol / ((i + 1) % T_end) - 0.001 * entropy_mean / ((i + 1) % T_end)

            print('kl_loss', kl_loss * 10.1)
            print('Lpol', LPol * 1.0)
            print('entropy', 0.001 * entropy_mean)
            print('REMLOSS', 0.1 * REMLOSS)

            critic_loss.backward()

            Transformer.optimizer.step()
            self.actor.optimizer.step()
            self.critic_1.optimizer.step()
            self.learn_step_cntr += 1

        self.update_network_parameters()

        return Transformer

    def update_network_parameters(self, tau=None):
        """
        Soft-update target networks using current network parameters.

        Parameters:
            tau (optional): Interpolation factor; if None, use self.tau.
        """
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

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        """
        Save checkpoints for actor, critic, and target networks.
        """
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.target_critic_1.save_checkpoint()

    def load_models(self):
        """
        Load checkpoints for actor, critic, and target networks.
        """
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.target_critic_1.load_checkpoint()

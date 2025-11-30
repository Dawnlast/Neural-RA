import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional
from torch.distributions.categorical import Categorical

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
@dataclass
class NetworkParams:
    wEEk: float = 45.0
    wIEk: float = 60.0
    wEIk: float = -6.0
    wIIk: float = -1.0
    exc_threshold: float = -1.5
    inh_threshold: float = -7.5
    exc_decay: float = 0.005
    inh_decay: float = 0.00025
    sigma: float = 120.0

@dataclass
class SimulationParams:
    T: float = 0.05
    Ti: float = 0.001
    tau: float = 1e-4
    n_neurons: int = 40

class RingAttractor:
    def __init__(self, sim_params: Optional[SimulationParams] = None, 
                net_params: Optional[NetworkParams] = None):
        self.sim_params = sim_params or SimulationParams()
        self.net_params = net_params or NetworkParams()
        
        self.Nt = int(np.floor(self.sim_params.T / self.sim_params.tau))
        self.Nti = int(np.floor(self.sim_params.Ti / self.sim_params.tau))
        
        self.alpha_n = np.linspace(0, 360 - 360/self.sim_params.n_neurons, 
                                self.sim_params.n_neurons).reshape(-1, 1)
        self.wEE = self._compute_weight_matrix()
        self.wIE = self.net_params.wIEk
        self.wEI = self.net_params.wEIk
        self.wII = self.net_params.wIIk

    def _compute_weight_matrix(self) -> np.ndarray:
        diff_matrix = np.minimum(
            np.abs(self.alpha_n - self.alpha_n.T),
            360 - np.abs(self.alpha_n - self.alpha_n.T)
        )
        wEE = np.exp(-diff_matrix**2 / (2 * self.net_params.sigma**2))
        return wEE * (self.net_params.wEEk / self.sim_params.n_neurons)
    
    def generate_action_signal(self, Q: float, alpha_a: float, sigma_a: float) -> np.ndarray:
        diff = np.min([np.abs(self.alpha_n - alpha_a),
                    360 - np.abs(self.alpha_n - alpha_a)], axis=0)
        
        signal = (Q * np.exp(-diff**2 / (2 * sigma_a**2)) / 
                (np.sqrt(2 * np.pi) * sigma_a))
        
        x = np.zeros((self.sim_params.n_neurons, self.Nt))
        x[:, self.Nti:] = np.repeat(signal.reshape(-1, 1), self.Nt - self.Nti, axis=1)
        return x
    
    def reset_state(self):
        self.v = np.zeros((self.sim_params.n_neurons, self.Nt))
        self.u = np.zeros((1, self.Nt))
        self.v[:, 0] = 0.05 * np.ones(self.sim_params.n_neurons)

    def action_space_integration(self, action_values: List[Tuple[float, float, float]]) -> int:
        self.reset_state()

        input_signals = sum([self.generate_action_signal(Q, alpha, sigma) 
                           for Q, alpha, sigma in action_values])
        
        tau_exc = self.sim_params.tau / self.net_params.exc_decay
        tau_inh = self.sim_params.tau / self.net_params.inh_decay
        n_neurons = self.sim_params.n_neurons
        
        v = self.v
        u = self.u
        
        for t in range(1, self.Nt):
            v_prev = v[:, t-1]
            u_prev = u[:, t-1]
            
            # Update Excitatory
            network_input = (self.net_params.exc_threshold + 
                           self.wEE.dot(v_prev) + 
                           self.wEI * u_prev + 
                           input_signals[:, t-1])
            
            v[:, t] = v_prev + (-v_prev + np.maximum(0, network_input)) * tau_exc
            
            # Update Inhibitory
            inhibitory_input = (self.net_params.inh_threshold + 
                              self.wIE * np.sum(v_prev) / n_neurons + 
                              self.wII * u_prev)
            
            u[:, t] = u_prev + (-u_prev + np.maximum(0, inhibitory_input)) * tau_inh

        max_neuron_idx = np.argmax(v[:, -1])
        
        winner_angle = self.alpha_n[max_neuron_idx][0]
        
        best_action_idx = 0
        min_dist = 360.0
        
        for i, (_, angle, _) in enumerate(action_values):
            dist = min(abs(angle - winner_angle), 360 - abs(angle - winner_angle))
            if dist < min_dist:
                min_dist = dist
                best_action_idx = i
                
        return best_action_idx

def build_net(layer_shape, hid_activation, output_activation):
    layers = []
    for j in range(len(layer_shape)-1):
        act = hid_activation if j < len(layer_shape)-2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
    return nn.Sequential(*layers)

class Double_Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        return q1, q2

class Policy_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Policy_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.P = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        logits = self.P(s)
        probs = F.softmax(logits, dim=1)
        return probs

class ReplayBuffer(object):
    def __init__(self, state_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.s = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.a = torch.zeros((max_size, 1), dtype=torch.long)
        self.r = torch.zeros((max_size, 1), dtype=torch.float32)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool)

    def add(self, s, a, r, s_next, dw):
        self.s[self.ptr] = torch.from_numpy(s)
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next)
        self.dw[self.ptr] = dw
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]

class SACD_RA_Agent:
    def __init__(self, state_dim, action_dim, hid_shape=[200, 200], 
                 lr=3e-4, gamma=0.99, alpha=0.2, adaptive_alpha=True, 
                 batch_size=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hid_shape = hid_shape
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha
        self.batch_size = batch_size
        self.tau = 0.005
        self.H_mean = 0
        
        self.ra = RingAttractor(
            sim_params=SimulationParams(n_neurons=max(40, action_dim * 10), T=0.05)
        )
        self.action_angles = np.linspace(0, 360, action_dim, endpoint=False)
        
        self.replay_buffer = ReplayBuffer(state_dim, max_size=int(1e6))

        self.actor = Policy_Net(state_dim, action_dim, hid_shape)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.q_critic = Double_Q_Net(state_dim, action_dim, hid_shape)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        for p in self.q_critic_target.parameters(): 
            p.requires_grad = False

        if self.adaptive_alpha:
            self.target_entropy = 0.6 * (-np.log(1 / action_dim))
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state, deterministic=False, use_ra=True):
        state_tensor = torch.FloatTensor(state[np.newaxis, :])
        
        if not use_ra:
            with torch.no_grad():
                probs = self.actor(state_tensor)
                if deterministic:
                    a = probs.argmax(-1).item()
                else:
                    a = Categorical(probs).sample().item()
                return a
        else:
            with torch.no_grad():
                q1, q2 = self.q_critic(state_tensor)
                q_avg = (q1 + q2) / 2.0
                q_vals = q_avg.cpu().numpy().flatten()
                
                temperature = 1.0 
                gain = 15.0 
                
                q_norm = np.exp((q_vals - np.max(q_vals)) / temperature)
                q_probs = q_norm / q_norm.sum()
                amplitudes = q_probs * gain
                
                input_sigma = 30.0 
                
                ra_inputs = []
                for i in range(self.action_dim):
                    ra_inputs.append((amplitudes[i], self.action_angles[i], input_sigma))
                
                action_idx = self.ra.action_space_integration(ra_inputs)
                
                return action_idx

    def train(self):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        # Train Critic
        with torch.no_grad():
            next_probs = self.actor(s_next)
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q1_all, next_q2_all = self.q_critic_target(s_next)
            min_next_q_all = torch.min(next_q1_all, next_q2_all)
            v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), 
                             dim=1, keepdim=True)
            target_Q = r + (~dw) * self.gamma * v_next

        q1_all, q2_all = self.q_critic(s)
        q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a)
        q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # Train Actor
        probs = self.actor(s)
        log_probs = torch.log(probs + 1e-8)
        with torch.no_grad():
            q1_all, q2_all = self.q_critic(s)
        min_q_all = torch.min(q1_all, q2_all)

        a_loss = torch.sum(probs * (self.alpha * log_probs - min_q_all), dim=1)
        self.actor_optimizer.zero_grad()
        a_loss.mean().backward()
        self.actor_optimizer.step()

        # Train Alpha
        if self.adaptive_alpha:
            with torch.no_grad():
                self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
            alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()

        # Update Target Net
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

def main():
    config = {
        'env_name': 'LunarLander-v3', 
        'seed': 42,
        'max_train_steps': int(2e4),
        'eval_interval': int(5e2),
        'random_steps': int(1e3),
        'update_every': 10,
        'gamma': 0.99,
        'hid_shape': [128, 128],
        'lr': 1e-3,
        'batch_size': 128,
        'alpha': 0.2,
        'adaptive_alpha': True,
    }

    env = gym.make(config['env_name'])
    eval_env = gym.make(config['env_name'])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    print(f"Training SACD with Ring Attractor on {config['env_name']}")

    agent = SACD_RA_Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        hid_shape=config['hid_shape'],
        lr=config['lr'],
        gamma=config['gamma'],
        alpha=config['alpha'],
        adaptive_alpha=config['adaptive_alpha'],
        batch_size=config['batch_size']
    )

    total_steps = 0
    env_seed = config['seed']
    
    while total_steps < config['max_train_steps']:
        s, info = env.reset(seed=env_seed)
        env_seed += 1
        done = False

        while not done:
            if total_steps < config['random_steps']:
                a = env.action_space.sample()
            else:
                a = agent.select_action(s, use_ra=True)
            
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            if config['env_name'] == 'LunarLander-v3' and r <= -100:
                r = -10

            agent.replay_buffer.add(s, a, r, s_next, dw)
            s = s_next

            if total_steps >= config['random_steps'] and total_steps % config['update_every'] == 0:
                for _ in range(config['update_every']):
                    agent.train()

            if total_steps % config['eval_interval'] == 0:
                score = 0
                turns = 3
                for _ in range(turns):
                    s_eval, _ = eval_env.reset()
                    d_eval = False
                    while not d_eval:
                        a_eval = agent.select_action(s_eval, use_ra=True)
                        s_eval, r_eval, dw_eval, tr_eval, _ = eval_env.step(a_eval)
                        d_eval = (dw_eval or tr_eval)
                        score += r_eval
                avg_score = int(score / turns)
                print(f"Steps: {total_steps}, Score: {avg_score}, Alpha: {agent.alpha:.3f}")
            
            total_steps += 1

    env.close()
    eval_env.close()
    print("Training completed.")

if __name__ == '__main__':
    main()

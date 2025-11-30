import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import gymnasium as gym
import copy
from torch.distributions.categorical import Categorical

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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


class SACD_agent():
    def __init__(self, state_dim, action_dim, hid_shape=[200, 200], 
                 lr=3e-4, gamma=0.99, alpha=0.2, adaptive_alpha=True, 
                 batch_size=256, **kwargs):
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

    def select_action(self, state, deterministic):
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis, :])
            probs = self.actor(state)
            if deterministic:
                a = probs.argmax(-1).item()
            else:
                a = Categorical(probs).sample().item()
            return a

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


def evaluate_policy(env, agent, turns=3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)
            total_scores += r
            s = s_next
    return int(total_scores / turns)


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
    print(f"Training SACD on {config['env_name']}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    agent = SACD_agent(
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
                a = agent.select_action(s, deterministic=False)
            
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            if config['env_name'] == 'LunarLander-v2' and r <= -100:
                r = -10

            agent.replay_buffer.add(s, a, r, s_next, dw)
            s = s_next

            if total_steps >= config['random_steps'] and total_steps % config['update_every'] == 0:
                for _ in range(config['update_every']):
                    agent.train()
                
            if total_steps % config['eval_interval'] == 0:
                score = evaluate_policy(eval_env, agent, turns=3)
                print(f"Steps: {total_steps//1000}k, Score: {score}, Alpha: {agent.alpha:.3f}, H: {agent.H_mean:.3f}")
                print(total_steps)
            
            total_steps += 1

    env.close()
    eval_env.close()
    print("Training completed.")


if __name__ == '__main__':
    main()

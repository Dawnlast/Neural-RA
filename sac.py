import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random
import matplotlib.pyplot as plt
import json

#Hyperparameters
lr_pi           = 0.0005
lr_q            = 0.001
init_alpha      = 0.01
gamma           = 0.98
batch_size      = 32
buffer_limit    = 50000
tau             = 0.01 # for target network soft update
target_entropy  = -1.0 # for automated alpha update
lr_alpha        = 0.001  # for automated alpha update

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

class PolicyNet(nn.Module):
    def __init__(self, learning_rate):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc_mu = nn.Linear(128,1)
        self.fc_std  = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s,a), q2(s,a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

class QNet(nn.Module):
    def __init__(self, learning_rate):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1,64)
        self.fc_cat = nn.Linear(128,32)
        self.fc_out = nn.Linear(32,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a) , target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def calc_target(pi, q1, q2, mini_batch):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob= pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target

def smooth_curve(values, weight=0.9):
    """
    Exponential Moving Average (EMA) 平滑
    weight 越大越平滑 (0.9~0.98 推荐)
    """
    smoothed = []
    last = values[0]
    for v in values:
        new_val = last * weight + (1 - weight) * v
        smoothed.append(new_val)
        last = new_val
    return smoothed

def run_one_seed(seed, num_episodes=800):
    env = gym.make('Pendulum-v1')
    memory = ReplayBuffer()
    q1, q2, q1_target, q2_target = QNet(lr_q), QNet(lr_q), QNet(lr_q), QNet(lr_q)
    pi = PolicyNet(lr_pi)

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    score = 0.0
    print_interval = 20
    score_draw = 0.0
    episode_scores = []

    for n_epi in range(num_episodes):
        s, info = env.reset()
        done = False
        truncated = False
        count = 0

        while count < 200 and not (done or truncated):
            a, log_prob = pi(torch.from_numpy(s).float())
            s_prime, r, terminated, truncated, info = env.step([2.0 * a.item()])
            # 在gymnasium中，done被分为terminated和truncated
            # terminated: 环境达到终止状态
            # truncated: 达到时间限制
            done = terminated or truncated
            memory.put((s, a.item(), r / 10.0, s_prime, terminated))  # 使用terminated而不是done
            score += r
            score_draw += r
            s = s_prime
            count += 1

        if memory.size() > 1000:
            for i in range(20):
                mini_batch = memory.sample(batch_size)
                td_target = calc_target(pi, q1_target, q2_target, mini_batch)
                q1.train_net(td_target, mini_batch)
                q2.train_net(td_target, mini_batch)
                entropy = pi.train_net(q1, q2, mini_batch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(n_epi, score / print_interval,
                                                                             pi.log_alpha.exp()))
            score = 0.0

        episode_scores.append(score_draw)
        score_draw = 0.0

    env.close()
    return episode_scores



# -------------------------------------------------
# 2. 多 seed 运行入口
# -------------------------------------------------
def run_multi_seed(seeds=list(range(10))):

    all_scores = {}
    min_len = 1e9

    for seed in seeds:
        print("========== Running seed", seed, "==========")
        scores = run_one_seed(seed)
        all_scores[str(seed)] = scores
        min_len = min(min_len, len(scores))  # 对齐长度（一般都一样）

    # 对齐所有 seed 的长度
    for seed in seeds:
        all_scores[str(seed)] = all_scores[str(seed)][:min_len]

    # 转为 numpy
    score_matrix = np.array([all_scores[str(seed)] for seed in seeds])

    # 计算平均
    avg_curve = np.mean(score_matrix, axis=0)

    # 保存到 npz
    np.savez("scores_multi_seed_rare.npz", **all_scores, avg=avg_curve)

    # JSON 保存（可读性更高）
    with open("scores_multi_seed_rare.json", "w") as f:
        json.dump({"seeds": all_scores, "avg": avg_curve.tolist()}, f, indent=4)

    print("数据已保存到 scores_multi_seed_rare.npz 和 scores_multi_seed_rare.json")

    # -------------------------------------------------
    # 3. 绘图
    # -------------------------------------------------
    plt.figure(figsize=(10, 6))

    episodes = np.arange(min_len)

    # 每条 seed 曲线
    # for seed in seeds:
    #     plt.plot(episodes, all_scores[str(seed)], alpha=0.3, linewidth=1)

    # 平均曲线
    plt.plot(episodes, avg_curve, label="Mean Score (10 seeds)", linewidth=1)

    smoothed = smooth_curve(avg_curve, weight=0.9)
    plt.plot(episodes, smoothed, label="Smoothed Mean (EMA)", linewidth=3)

    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.title("SAC + Ring Attractor (10 seeds average)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# 运行
# -------------------------------------------------
run_multi_seed()








# def run_one_seed(seed, num_episodes=800):
#     np.random.seed(seed)
#     import torch
#     torch.manual_seed(seed)
#
#     ring_attractor = RingAttractor()
#     env = gym.make('Pendulum-v1')
#     memory = ReplayBuffer()
#
#     q1, q2, q1_target, q2_target = QNet(lr_q), QNet(lr_q), QNet(lr_q), QNet(lr_q)
#     pi = PolicyNet(lr_pi)
#
#     q1_target.load_state_dict(q1.state_dict())
#     q2_target.load_state_dict(q2.state_dict())
#
#     print_interval = 20
#     score = 0.0
#     score_draw = 0.0
#     episode_scores = []
#
#     for n_epi in range(num_episodes):
#         s, info = env.reset()
#         done = False
#         truncated = False
#         count = 0
#
#         while count < 200 and not (done or truncated):
#
#             a_final, candidates = choose_action_with_RA(pi,q1,q2, s, ring_attractor, K=10)
#             s_prime, r, terminated, truncated, info = env.step([2.0 * a_final])
#
#             done = terminated or truncated
#             memory.put((s, a_final, r/10.0, s_prime, terminated))
#
#             score += r
#             score_draw += r
#             s = s_prime
#             count += 1
#
#         if memory.size() > 1000:
#             for i in range(20):
#                 mini_batch = memory.sample(batch_size)
#                 td_target = calc_target(pi, q1_target, q2_target, mini_batch)
#                 q1.train_net(td_target, mini_batch)
#                 q2.train_net(td_target, mini_batch)
#                 entropy = pi.train_net(q1, q2, mini_batch)
#                 q1.soft_update(q1_target)
#                 q2.soft_update(q2_target)
#
#         if n_epi % print_interval == 0 and n_epi != 0:
#             print(f"[Seed {seed}] Episode {n_epi}, avg score {score/print_interval:.1f}")
#             score = 0.0
#
#         episode_scores.append(score_draw)
#         score_draw = 0.0
#
#     env.close()
#     return episode_scores
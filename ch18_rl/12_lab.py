# %% [markdown]
# # Section 18.12 Hands-on lab: tabular Q-learning, DQN, PPO
#
# Regenerated from Book.md (single source of truth). Real Gymnasium environments
# (FrozenLake and CartPole) and stable-baselines3; seeds are fixed so the
# printed numbers reproduce. The code cell is Example 18-41, verbatim from the chapter.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import gymnasium as gym

torch.manual_seed(0); np.random.seed(0); random.seed(0)

def tier1_frozenlake(episodes=8000, alpha=0.8, gamma=0.95):
    e = gym.make("FrozenLake-v1", is_slippery=True)
    Q = np.zeros((e.observation_space.n, e.action_space.n))
    rg = np.random.default_rng(0)
    for ep in range(episodes):
        s, _ = e.reset(seed=ep); done = False; eps = max(0.05, 1 - ep / 4000)
        while not done:
            a = rg.integers(4) if rg.random() < eps else int(np.argmax(Q[s]))
            s2, r, term, trunc, _ = e.step(a); done = term or trunc
            Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) * (not term) - Q[s, a]); s = s2
    wins = 0
    for k in range(500):
        s, _ = e.reset(seed=10**6 + k); done = False
        while not done:
            s, r, term, trunc, _ = e.step(int(np.argmax(Q[s]))); done = term or trunc
        wins += int(r == 1.0)
    return wins / 500

def random_frozenlake(episodes=500):
    e = gym.make("FrozenLake-v1", is_slippery=True)
    rg = np.random.default_rng(1)                        # its own generator, so the tiers are untouched
    wins = 0
    for k in range(episodes):
        s, _ = e.reset(seed=10**6 + k); done = False    # the same 500 starts the table is scored on
        while not done:
            s, r, term, trunc, _ = e.step(int(rg.integers(4))); done = term or trunc
        wins += int(r == 1.0)
    return wins / episodes

rate = tier1_frozenlake()
print(f"TIER 1  tabular Q-learning on FrozenLake (slippery): greedy success rate {rate:.2f}")
print(f"        a uniform random policy on the same 500 starts: success rate {random_frozenlake():.3f}")

def tier2_dqn(episodes=300):
    env = gym.make("CartPole-v1")
    no, na = env.observation_space.shape[0], env.action_space.n
    def net(): return nn.Sequential(nn.Linear(no, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, na))
    q, tgt = net(), net(); tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3); buf = deque(maxlen=10000)
    eps = 1.0
    rets = []
    for ep in range(episodes):
        s, _ = env.reset(seed=ep); done = False; tot = 0
        while not done:
            if random.random() < eps: a = random.randrange(na)   # seeded exploration (reproducible)
            else:
                with torch.no_grad(): a = int(q(torch.tensor(s)).argmax())
            s2, r, term, trunc, _ = env.step(a); done = term or trunc
            buf.append((s, a, r, s2, float(term))); s = s2; tot += r
            if len(buf) >= 1000:
                b = random.sample(buf, 64)
                bs, ba, br, bs2, bd = zip(*b)
                bs = torch.tensor(np.array(bs)); ba = torch.tensor(ba)
                br = torch.tensor(br, dtype=torch.float32); bs2 = torch.tensor(np.array(bs2))
                bd = torch.tensor(bd, dtype=torch.float32)
                pred = q(bs).gather(1, ba[:, None]).squeeze(1)
                with torch.no_grad(): targ = br + 0.99 * tgt(bs2).max(1).values * (1 - bd)
                loss = nn.functional.mse_loss(pred, targ)
                opt.zero_grad(); loss.backward(); opt.step()
        eps = max(0.05, eps * 0.97)
        if ep % 20 == 0: tgt.load_state_dict(q.state_dict())
        rets.append(tot)
    return np.mean(rets[:20]), np.mean(rets[-20:]), np.std(rets[-50:])

early, late, spread = tier2_dqn()
print(f"\nTIER 2  DQN on CartPole: mean return {early:.0f} (first 20 eps) -> {late:.0f} (last 20 eps)")
print(f"        standard deviation of the last 50 training returns: {spread:.0f}")
print(f"        a neural net handles the 4 CONTINUOUS state variables a table never could")

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

model = PPO("MlpPolicy", "CartPole-v1", verbose=0, seed=0)
model.learn(total_timesteps=30000)
mean_r, std_r = evaluate_policy(model, model.get_env(), n_eval_episodes=20)
print(f"\nTIER 3  PPO (Stable-Baselines3) on CartPole: mean return {mean_r:.0f} +/- {std_r:.0f} over 20 episodes")
print(f"        (500 is the maximum; a few lines of a tested library, the clipped objective of 18.7 inside)")

print("\nthe progression: a TABLE for small discrete worlds, a NETWORK for continuous ones,")
print("and a LIBRARY implementation of PPO for getting real work done - same RL ideas throughout.")
print("\nAll RL-lab checks ran.")


# %% [markdown]
# # Section 18.7.3 check: DQN against Double DQN over five matched seeds
#
# Companion script for Section 18.7.3. It reruns the two arms of Example 18-15 (a DQN on Gymnasium's CartPole-v1, with
# and without the Double DQN target) over seeds 0 to 4. Within a seed both arms start from the same seeds, so each seed
# is a matched pair, and seed 0 repeats Example 18-15 itself. Episode `ep` of seed `k` resets with seed
# `10000 * k + ep`. The script prints the mean return of the last 20 of 300 training episodes for each arm and seed.
# One timed run took about ten and a half minutes (640 seconds) on the six-core desktop CPU used for this book's
# outputs, which is why it lives here and not in the chapter walkthrough. Tested with gymnasium 1.4.0 and torch
# 2.15.0 (CPU).

# %%
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import gymnasium as gym

def train(double=False, episodes=300, seed=0):
    # identical seeding for both arms of a seed, so the ONLY difference is the target rule
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)    # make CPU matmuls reproducible
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)                 # seed the exploration sampler too, or runs diverge
    n_obs = env.observation_space.shape[0]      # 4: cart position/velocity, pole angle/velocity
    n_act = env.action_space.n                  # 2: push left or right

    def make_net():
        return nn.Sequential(nn.Linear(n_obs, 64), nn.ReLU(),
                             nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_act))
    q_net = make_net(); target_net = make_net()
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = deque(maxlen=10000)                 # the replay buffer
    gamma, batch = 0.99, 64

    def act(state, eps):
        if random.random() < eps: return env.action_space.sample()
        with torch.no_grad():
            return int(q_net(torch.tensor(state)).argmax())

    def learn():
        if len(buffer) < 1000: return            # wait until the buffer has enough variety
        s, a, r, s2, d = zip(*random.sample(buffer, batch))
        s = torch.tensor(np.array(s)); a = torch.tensor(a); r = torch.tensor(r, dtype=torch.float32)
        s2 = torch.tensor(np.array(s2)); d = torch.tensor(d, dtype=torch.float32)
        q = q_net(s).gather(1, a[:, None]).squeeze(1)            # Q(s,a) predicted
        with torch.no_grad():
            if double:                                            # Double DQN: online net SELECTS,
                a2 = q_net(s2).argmax(1, keepdim=True)            # target net EVALUATES
                nxt = target_net(s2).gather(1, a2).squeeze(1)
            else:                                                 # DQN: target net does both
                nxt = target_net(s2).max(1).values
            target = r + gamma * nxt * (1 - d)                   # bootstrap with TARGET net
        loss = nn.functional.mse_loss(q, target)
        opt.zero_grad(); loss.backward(); opt.step()

    returns = []
    eps = 1.0
    for ep in range(episodes):
        s, _ = env.reset(seed=10000 * seed + ep); done = False; total = 0.0
        while not done:
            a = act(s, eps)
            s2, r, term, trunc, _ = env.step(a); done = term or trunc
            buffer.append((s, a, r, s2, float(term)))           # store transition
            s = s2; total += r
            learn()
        eps = max(0.05, eps * 0.97)                              # anneal exploration
        if ep % 20 == 0: target_net.load_state_dict(q_net.state_dict())   # sync target net
        returns.append(total)
    return returns

last = {False: [], True: []}
for seed in range(5):
    for dbl in [False, True]:
        last[dbl].append(np.mean(train(double=dbl, seed=seed)[-20:]))
    print(f"seed {seed}: mean return of the last 20 episodes, DQN {last[False][-1]:3.0f}, Double DQN {last[True][-1]:3.0f}")
d = np.array(last[True]) - np.array(last[False])
print(f"mean over 5 seeds: DQN {np.mean(last[False]):.0f}, Double DQN {np.mean(last[True]):.0f}; "
      f"Double DQN higher in {int(np.sum(d > 0))} of 5 seeds; mean difference {d.mean():+.0f}, "
      f"standard deviation of the differences {d.std(ddof=1):.0f}")

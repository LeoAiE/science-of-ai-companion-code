"""Appendix H solutions for Chapter 18, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 18.13 coding exercises

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import gymnasium as gym

# %% [markdown]
# 1. Bandit regret: greedy is worst, Thompson lowest-regret

# %%
# 1. Bandit regret: greedy is worst, Thompson lowest-regret
rng = np.random.default_rng(0)
K, T, RUNS = 10, 1000, 200
true_p = np.linspace(0.1, 0.9, K); best = true_p.max()
def pull(a): return 1.0 if rng.random() < true_p[a] else 0.0
def eps_greedy(eps):
    Q = np.zeros(K); N = np.zeros(K); tot = 0.0
    for t in range(T):
        a = rng.integers(K) if rng.random() < eps else int(np.argmax(Q))
        r = pull(a); N[a] += 1; Q[a] += (r - Q[a]) / N[a]; tot += r   # incremental sample mean
    return tot
def ucb(c=1.0):
    Q = np.zeros(K); N = np.zeros(K); tot = 0.0
    for t in range(T):
        a = t if t < K else int(np.argmax(Q + c*np.sqrt(np.log(t+1)/N)))  # value + uncertainty bonus
        r = pull(a); N[a] += 1; Q[a] += (r - Q[a]) / N[a]; tot += r
    return tot
def thompson():
    al = np.ones(K); be = np.ones(K); tot = 0.0
    for t in range(T):
        a = int(np.argmax(rng.beta(al, be)))                    # sample each Beta belief, pick the max
        r = pull(a); al[a] += r; be[a] += (1 - r); tot += r
    return tot
for name, fn in [("random", lambda: eps_greedy(1.0)), ("greedy", lambda: eps_greedy(0.0)),
                 ("eps0.1", lambda: eps_greedy(0.1)), ("UCB", ucb), ("Thompson", thompson)]:
    avg = np.mean([fn() for _ in range(RUNS)])
    print(f"{name:9s} reward {avg:6.1f} regret {best*T-avg:6.1f}")
# expected output: random    reward  499.2 regret  400.8
# expected output: greedy    reward  100.3 regret  799.7
# expected output: eps0.1    reward  807.2 regret   92.8
# expected output: UCB       reward  803.1 regret   96.9
# expected output: Thompson  reward  867.6 regret   32.4

# %% [markdown]
# 2. Value iteration from scratch on a 4x4 GridWorld (contraction + optimal path)

# %%
# 2. Value iteration from scratch on a 4x4 GridWorld (contraction + optimal path)
SIZE, GOAL, gamma = 4, 15, 0.9
ACT = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}
def step(s, a):
    if s == GOAL: return s, 0.0
    r, c = divmod(s, SIZE); dr, dc = ACT[a]
    r2, c2 = min(max(r+dr,0),SIZE-1), min(max(c+dc,0),SIZE-1)
    return r2*SIZE+c2, -1.0
V = np.zeros(SIZE*SIZE); errs = []
for k in range(200):
    Vn = V.copy()
    for s in range(SIZE*SIZE):
        if s == GOAL: continue
        Vn[s] = max(step(s,a)[1] + gamma*V[step(s,a)[0]] for a in ACT)   # Bellman optimality update
    errs.append(np.max(np.abs(Vn - V))); V = Vn
    if errs[-1] < 1e-9: break
pol = {s:int(np.argmax([step(s,a)[1]+gamma*V[step(s,a)[0]] for a in ACT])) for s in range(SIZE*SIZE)}
s = 0; steps = 0
while s != GOAL and steps < 20: s = step(s, pol[s])[0]; steps += 1
ratios = [errs[i+1]/errs[i] for i in range(len(errs)-1) if errs[i] > 0]   # every sweep's shrink factor
ok = max(ratios) <= gamma + 1e-12              # slack for rounding: the largest ratio computes as 0.9000000000000004
print(f"sweeps {k+1}  V*corner {V[0]:.2f}  max ratio {max(ratios):.2f} <= gamma {ok}  path {steps}")
# expected output: sweeps 7  V*corner -4.69  max ratio 0.90 <= gamma True  path 6

# %% [markdown]
# 3. Policy iteration reaches the SAME V* as value iteration

# %%
# 3. Policy iteration reaches the SAME V* as value iteration
def evaluate(polp, tol=1e-9):
    Vp = np.zeros(SIZE*SIZE)
    while True:
        d = 0.0
        for s in range(SIZE*SIZE):
            if s == GOAL: continue
            v = sum(polp[s][a]*(step(s,a)[1]+gamma*Vp[step(s,a)[0]]) for a in ACT)
            d = max(d, abs(v-Vp[s])); Vp[s] = v
        if d < tol: return Vp
def greedy_pol(Vp):
    p = {}
    for s in range(SIZE*SIZE):
        q = [step(s,a)[1]+gamma*Vp[step(s,a)[0]] for a in ACT]
        b = int(np.argmax(q)); p[s] = {a:(1.0 if a==b else 0.0) for a in ACT}
    return p
pol_u = {s:{a:0.25 for a in ACT} for s in range(SIZE*SIZE)}; rounds = 0
while True:
    rounds += 1; Vp = evaluate(pol_u); ng = greedy_pol(Vp)
    if all(np.argmax(list(ng[s].values())) == np.argmax(list(pol_u[s].values())) for s in range(SIZE*SIZE)):
        break
    pol_u = ng
print(f"rounds {rounds}  max|V_pi - V_vi| {np.abs(Vp - V).max():.2e}")
# expected output: rounds 3  max|V_pi - V_vi| 0.00e+00

# %% [markdown]
# 4. Bellman linear solve vs Monte Carlo on the slippery corridor (always-right)

# %%
# 4. Bellman linear solve vs Monte Carlo on the slippery corridor (always-right)
rng = np.random.default_rng(0)
S, G, gam, ps = 5, 4, 0.9, 0.8
def trans(s):
    if s == G: return [(1.0, G, 0.0)]
    inn = min(G, s+1)
    return [(ps, inn, 10.0 if inn==G else -1.0), (1-ps, s, -1.0)]      # slip = stay put
P = np.zeros((S,S)); R = np.zeros(S)
for s in range(S):
    for pr, s2, r in trans(s): P[s,s2] += pr; R[s] += pr*r
Vsolve = np.linalg.solve(np.eye(S) - gam*P, R)                          # V = (I - gamma P)^-1 R
def rollout(s):
    Gr, disc = 0.0, 1.0
    while s != G:
        pr = trans(s); i = rng.choice(len(pr), p=[p for p,_,_ in pr]); _, s2, r = pr[i]
        Gr += disc*r; disc *= gam; s = s2
    return Gr
mc = np.array([np.mean([rollout(s) for _ in range(3000)]) for s in range(S)])
print("solve", np.round(Vsolve,2), "mc", np.round(mc,2), "diff", round(float(np.abs(Vsolve-mc).max()),3))
# expected output: solve [3.21 5.04 7.13 9.51 0.  ] mc [3.19 5.08 7.13 9.53 0.  ] diff 0.033

# %% [markdown]
# 5. TD(0) vs Monte Carlo prediction: same target, different variance

# %%
# 5. TD(0) vs Monte Carlo prediction: same target, different variance
rng = np.random.default_rng(0)
Vtrue = Vsolve.copy()
def sample(s):
    pr = trans(s); i = rng.choice(len(pr), p=[p for p,_,_ in pr]); return pr[i][1], pr[i][2]
def episode():
    tr = []; s = 0
    while s != G:
        s2, r = sample(s); tr.append((s, r)); s = s2
    return tr
def mc_pred(n):
    Vv = np.zeros(S); cnt = np.zeros(S)
    for _ in range(n):
        tr = episode(); Gr = 0.0
        for s, r in reversed(tr): Gr = r + gam*Gr; cnt[s] += 1; Vv[s] += (Gr - Vv[s]) / cnt[s]
    return Vv
def td0(n, al=0.05):
    Vv = np.zeros(S)
    for _ in range(n):
        s = 0
        while s != G:
            s2, r = sample(s); Vv[s] += al*(r + gam*Vv[s2] - Vv[s]); s = s2   # bootstrap after one step
    return Vv
rmse = lambda v: np.sqrt(np.mean((v[:G]-Vtrue[:G])**2))
Vmc = mc_pred(500); Vtd = td0(500)
print(f"MC rmse {rmse(Vmc):.3f}  TD rmse {rmse(Vtd):.3f}")
mct = []
for _ in range(4000):
    tr = episode(); Gr = 0.0
    for s, r in reversed(tr): Gr = r + gam*Gr
    mct.append(Gr)
tdt = [ (lambda s2r: s2r[1] + gam*Vtrue[s2r[0]])(sample(0)) for _ in range(4000) ]
print(f"target std at state0: MC {np.std(mct):.2f}  TD {np.std(tdt):.2f}")
# expected output: MC rmse 0.049  TD rmse 0.119
# expected output: target std at state0: MC 1.43  TD 0.67

# %% [markdown]
# 6. SARSA vs Q-learning on the real Gymnasium CliffWalking-v1

# %%
# 6. SARSA vs Q-learning on the real Gymnasium CliffWalking-v1
# Both methods train on the same 20 seeds, 0 to 19, for the starter's 500 episodes, and each run's
# greedy path is read from its own final table. Example 18-9 averages 30 runs per method, seeded
# 1000 apart (SARSA from 0, Q-learning from 500), and prints -27.1 and -48.2.
env = gym.make("CliffWalking-v1")
nS, nA = env.observation_space.n, env.action_space.n
def egr(Q, s, eps, rg): return int(rg.integers(nA)) if rg.random() < eps else int(np.argmax(Q[s]))
def train(method, episodes, seed, alpha=0.5, eps=0.1, g=1.0, cap=200):
    Q = np.zeros((nS, nA)); rg = np.random.default_rng(seed); rets = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed+ep); a = egr(Q, s, eps, rg); tot = 0.0; done = False; n = 0
        while not done and n < cap:
            s2, r, term, trunc, _ = env.step(a); done = term or trunc; tot += r; n += 1
            a2 = egr(Q, s2, eps, rg)
            tgt = r + g*Q[s2][a2] if method == "sarsa" else r + g*np.max(Q[s2])   # the one-line difference
            Q[s][a] += alpha*(tgt - Q[s][a]); s, a = s2, a2
        rets.append(tot)
    return Q, rets
def greedy_path(Q, cap=50):                                 # (return, reached the goal?) within 50 steps
    s, _ = env.reset(seed=0); tot = 0.0
    for _ in range(cap):
        s, r, term, trunc, _ = env.step(int(np.argmax(Q[s]))); tot += r
        if term or trunc: return tot, True
    return tot, False
def cliff_compare(episodes=500, seeds=range(20)):          # one shared set of seeds for both methods
    for method, name in [("sarsa", "SARSA"), ("qlearn", "Q-learning")]:
        runs = [train(method, episodes, sd) for sd in seeds]
        online = [np.mean(rets[-100:]) for _, rets in runs]  # online return over each run's last 100 episodes
        done = [tot for tot, ok in (greedy_path(Q) for Q, _ in runs) if ok]
        print(f"{name:10s} online last100 {np.mean(online):6.1f} (sd {np.std(online):.1f})  "
              f"greedy path reaches goal {len(done)}/{len(runs)}, most often at {max(set(done), key=done.count):.0f}")
cliff_compare()
env.close()
# expected output: SARSA      online last100  -23.5 (sd 2.3)  greedy path reaches goal 16/20, most often at -17
# expected output: Q-learning online last100  -49.0 (sd 4.9)  greedy path reaches goal 20/20, most often at -13
# In the other four SARSA runs the greedy rollout walks into a wall or steps back and forth until the cap.

# %% [markdown]
# 7. Tabular Q-learning on the real Gymnasium FrozenLake-v1 (slippery)

# %%
# 7. Tabular Q-learning on the real Gymnasium FrozenLake-v1 (slippery)
# The 4x4 map with the lab's settings, then the 8-by-8 map with the same settings and with tuned ones.
def frozen_q(map_name="4x4", episodes=8000, alpha=0.8, gamma=0.95, seed=0):
    e = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=True)
    Q = np.zeros((e.observation_space.n, e.action_space.n)); rg = np.random.default_rng(seed); goals = 0
    for ep in range(episodes):
        s, _ = e.reset(seed=seed*100000+ep); done = False
        eps = max(0.05, 1 - ep/(episodes/2))                  # anneal to the 0.05 floor over the first half
        while not done:
            a = rg.integers(4) if rg.random() < eps else int(np.argmax(Q[s]))
            s2, r, term, trunc, _ = e.step(a); done = term or trunc
            Q[s, a] += alpha*(r + gamma*np.max(Q[s2])*(not term) - Q[s, a]); s = s2
        goals += int(r == 1.0)                                # training episodes that reached the goal
    wins = 0
    for k in range(500):
        s, _ = e.reset(seed=10**6+k); done = False; r = 0
        while not done:
            s, r, term, trunc, _ = e.step(int(np.argmax(Q[s]))); done = term or trunc
        wins += int(r == 1.0)
    e.close(); return wins/500, goals
def random_fl(map_name="4x4", episodes=500):             # a uniform random policy on the same 500 starts
    e = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=True); rg = np.random.default_rng(1); wins = 0
    for k in range(episodes):
        s, _ = e.reset(seed=10**6+k); done = False; r = 0
        while not done:
            s, r, term, trunc, _ = e.step(int(rg.integers(4))); done = term or trunc
        wins += int(r == 1.0)
    e.close(); return wins/episodes
for m, n, al, gm in [("4x4", 8000, 0.8, 0.95), ("8x8", 8000, 0.8, 0.95), ("8x8", 20000, 0.1, 0.99)]:
    rate, goals = frozen_q(m, n, al, gm)
    print(f"{m} {n:5d} episodes alpha {al} gamma {gm}: greedy {rate:.2f}  goals in training {goals:4d}  random {random_fl(m):.3f}")
# expected output: 4x4  8000 episodes alpha 0.8 gamma 0.95: greedy 0.74  goals in training 1326  random 0.022
# expected output: 8x8  8000 episodes alpha 0.8 gamma 0.95: greedy 0.00  goals in training    0  random 0.000
# expected output: 8x8 20000 episodes alpha 0.1 gamma 0.99: greedy 0.60  goals in training 5408  random 0.000
# The first row is Example 18-41's tier 1. With the same settings no 8x8 training episode reaches the goal,
# so every Q-value stays 0 and np.argmax always picks action 0, left. The 20000-episode run anneals epsilon
# over 10000 episodes instead of 4000, and its training episodes do reach the goal.

# %% [markdown]
# 8. RLHF from scratch: reward model, KL leash, GRPO advantages

# %%
# 8. RLHF from scratch: reward model, KL leash, GRPO advantages
# A new preference set: 12 responses and 3000 comparisons (Example 18-31 fits 8 from 4000).
rng = np.random.default_rng(1)
sig = lambda z: 1/(1+np.exp(-z))
N = 12; tq = rng.normal(0, 1, N)                                      # hidden true quality of each response
pairs = [(i,j) for i in range(N) for j in range(N) if i != j]
W, L = [], []                                                         # winner and loser of each comparison
for _ in range(3000):
    i, j = pairs[rng.integers(len(pairs))]
    w = i if rng.random() < sig(tq[i]-tq[j]) else j                    # noisy Bradley-Terry choice
    W.append(w); L.append(i if w == j else j)
W, L = np.array(W), np.array(L)
r = np.zeros(N)
for _ in range(3000):                                                  # gradient ascent on the log likelihood
    p = sig(r[W]-r[L])                                                 # P(winner beats loser) for all comparisons at once
    grad = np.bincount(W, weights=1-p, minlength=N) - np.bincount(L, weights=1-p, minlength=N)
    r += 0.1*grad/len(W)                                               # Example 18-31's update, summed per response
r -= r.mean()
corr = np.corrcoef(r, tq)[0,1]
agree = np.mean([np.sign(r[i]-r[j]) == np.sign(tq[i]-tq[j]) for i,j in pairs])
top = r.argmax()
print(f"reward-model corr {corr:.3f}  pair-agreement {agree*100:.0f}%  top by model {top} (true {tq[top]:.2f})  top by truth {tq.argmax()} (true {tq.max():.2f})")
ref = np.full(N, 1/N)                                                  # a uniform reference policy
for beta in [5.0, 1.0, 0.2]:
    pi = ref*np.exp(r/beta); pi /= pi.sum()                            # KL-penalized optimum on the learned rewards
    print(f"beta={beta}: top response {pi.argmax()} p={pi.max():.2f}  KL to reference {np.sum(pi*np.log(pi/ref)):.3f}")
grp = np.array([1.0, 0.0, 0.0, 1.0, 1.0]); adv = (grp-grp.mean())/(grp.std()+1e-8)  # the starter's group
print("GRPO adv", np.round(adv, 2))
# expected output: reward-model corr 0.987  pair-agreement 94%  top by model 1 (true 0.82)  top by truth 4 (true 0.91)
# expected output: beta=5.0: top response 1 p=0.10  KL to reference 0.005
# expected output: beta=1.0: top response 1 p=0.15  KL to reference 0.091
# expected output: beta=0.2: top response 1 p=0.52  KL to reference 0.977
# expected output: GRPO adv [ 0.82 -1.22 -1.22  0.82  0.82]
# The model ranks response 1 above the truly best response 4, so as beta shrinks the policy piles onto
# the reward model's favorite: the imperfect judge's mistake becomes the policy's (Section 18.10.3).

# %% [markdown]
# 9. Monte Carlo Tree Search on tic-tac-toe

# %%
# 9. Monte Carlo Tree Search on tic-tac-toe
import math
rng = np.random.default_rng(0)
WINS = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
def winner(b):
    for x,y,z in WINS:
        if b[x] and b[x] == b[y] == b[z]: return b[x]
    return 0
def legal(b): return [i for i in range(9) if b[i] == 0]
def terminal(b): return winner(b) != 0 or not legal(b)
class Node:
    def __init__(self, board, player, parent=None, move=None):
        self.board, self.player, self.parent, self.move = board, player, parent, move
        self.children = {}; self.untried = [] if terminal(board) else legal(board); self.N = 0; self.W = 0.0
def uct(node, c=1.4):
    return max(node.children.values(), key=lambda ch: ch.W/ch.N + c*math.sqrt(math.log(node.N)/ch.N))
def rollout(board, player):
    b = list(board)
    while not terminal(b): b[rng.choice(legal(b))] = player; player = 3 - player
    return winner(b)
def mcts(board, player, n):
    root = Node(board, player)
    for _ in range(n):
        node = root
        while not node.untried and node.children: node = uct(node)         # SELECT
        if node.untried:                                                    # EXPAND
            m = node.untried.pop(rng.integers(len(node.untried)))
            nb = list(node.board); nb[m] = node.player
            child = Node(tuple(nb), 3-node.player, node, m); node.children[m] = child; node = child
        res = rollout(node.board, node.player)                              # SIMULATE
        while node is not None:                                             # BACKPROPAGATE
            node.N += 1; mover = 3 - node.player
            node.W += 1.0 if res == mover else (0.5 if res == 0 else 0.0); node = node.parent
    return max(root.children.items(), key=lambda kv: kv[1].N)[0]            # most-visited move
print("win-in-one plays", mcts((1,1,0, 2,2,0, 0,0,0), 1, 600))            # complete the top row
print("must-block plays", mcts((1,0,0, 2,2,0, 0,0,1), 1, 600))            # block O's middle row
def play(sx, so=0, ro=True):
    b = [0]*9; p = 1
    while not terminal(b):
        m = mcts(tuple(b), 1, sx) if p == 1 else (rng.choice(legal(b)) if ro else mcts(tuple(b), 2, so))
        b[m] = p; p = 3 - p
    return winner(b)
res = [play(50) for _ in range(50)]
print(f"MCTS(50) vs random: wins {res.count(1)} draws {res.count(0)} losses {res.count(2)}")
res2 = [play(800, 800, ro=False) for _ in range(20)]
print(f"MCTS(800) self-play: draws {res2.count(0)} Xwin {res2.count(1)} Owin {res2.count(2)}")
# expected output: win-in-one plays 2
# expected output: must-block plays 5
# expected output: MCTS(50) vs random: wins 49 draws 1 losses 0
# expected output: MCTS(800) self-play: draws 20 Xwin 0 Owin 0

# %% [markdown]
# 10. Deep Q-network on CartPole-v1 (miniproject).

# %%
# 10. Deep Q-network on CartPole-v1 (miniproject).
#     One timed run on the six-core desktop CPU used for this book's outputs took about 40 s with
#     torch 2.15.0 and printed the same returns as torch 2.13.0; another torch build can print different returns.
import random, torch, torch.nn as nn
from collections import deque
torch.manual_seed(0); np.random.seed(0); random.seed(0)
env = gym.make("CartPole-v1"); env.action_space.seed(0)
no, na = env.observation_space.shape[0], env.action_space.n
def net(): return nn.Sequential(nn.Linear(no,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,na))
q, tgt = net(), net(); tgt.load_state_dict(q.state_dict())
opt = torch.optim.Adam(q.parameters(), lr=1e-3); buf = deque(maxlen=10000)
gd, batch, eps, rets = 0.99, 64, 1.0, []
for ep in range(200):
    s, _ = env.reset(seed=ep); done = False; tot = 0.0
    while not done:
        a = env.action_space.sample() if random.random() < eps else int(q(torch.tensor(s)).argmax())
        s2, r, term, trunc, _ = env.step(a); done = term or trunc
        buf.append((s, a, r, s2, float(term))); s = s2; tot += r
        if len(buf) >= 1000:                                            # replay + target net
            bs, ba, br, bs2, bd = zip(*random.sample(buf, batch))
            bs = torch.tensor(np.array(bs)); ba = torch.tensor(ba); br = torch.tensor(br, dtype=torch.float32)
            bs2 = torch.tensor(np.array(bs2)); bd = torch.tensor(bd, dtype=torch.float32)
            pred = q(bs).gather(1, ba[:,None]).squeeze(1)
            with torch.no_grad(): targ = br + gd*tgt(bs2).max(1).values*(1 - bd)
            loss = nn.functional.mse_loss(pred, targ); opt.zero_grad(); loss.backward(); opt.step()
    eps = max(0.05, eps*0.97)
    if ep % 20 == 0: tgt.load_state_dict(q.state_dict())
    rets.append(tot)
env.close()
print(f"DQN CartPole: first20 {np.mean(rets[:20]):.0f} -> last20 {np.mean(rets[-20:]):.0f}  best {max(rets):.0f}")
# expected output: DQN CartPole: first20 21 -> last20 200  best 346

# %% [markdown]
# ## 18.13 coding drills

# %%
import numpy as np
np.set_printoptions(suppress=True)

# %% [markdown]
# Returns and discounting

# %%
# Returns and discounting
print(round(sum(0.9**k for k in range(2000)), 2))          # expected output: 10.0
print(round(1/(1-0.99), 1))                                # expected output: 100.0
G = 0.0
for r in reversed([-1,-1,-1,10]): G = r + 0.9*G
print(round(G, 2))                                         # expected output: 4.58
print(0.8**3*10 > 1.0, round(0.8**3*10, 2))                # expected output: True 5.12

# %% [markdown]
# Bandits

# %%
# Bandits
Q = 0.0; N = 0
for r in [1,0,1]: N += 1; Q += (r-Q)/N
print(round(Q, 4))                                         # expected output: 0.6667
print(round(1.0*np.sqrt(np.log(100)/10), 4))               # expected output: 0.6786
print(round(sum(0.9-m for m in [0.5,0.9,0.2]), 2))         # expected output: 1.1
print(round(3/(3+1), 2))                                   # expected output: 0.75

# %% [markdown]
# Values and Bellman

# %%
# Values and Bellman
print(round(-1 + 0.9*10, 2))                               # expected output: 8.0
print(int(np.argmax([1.2,3.4,0.5,2.1])))                   # expected output: 1
delta = -1 + 0.9*10 - 5; print(round(delta, 2))            # expected output: 3.0
print(round(5 + 0.1*delta, 2))                             # expected output: 5.3
P = np.array([[0.,1.],[0.,1.]]); R = np.array([-1.,0.])
print(np.round(np.linalg.solve(np.eye(2)-0.9*P, R), 2))    # expected output: [-1.  0.]

# %% [markdown]
# TD control targets

# %%
# TD control targets
print(round(-1 + 1.0*(-3), 2))                             # expected output: -4.0
Qs2 = np.array([-3.,-5.,-2.,-4.]); print(round(-1 + 1.0*Qs2.max(), 2))  # expected output: -3.0
eps = 0.1; pi = np.full(4, eps/4); pi[Qs2.argmax()] += 1-eps
print(round(-1 + 1.0*pi.dot(Qs2), 3))                      # expected output: -3.15
Q1 = np.array([1.,3.,2.]); Q2 = np.array([0.5,2.5,1.0])
print(round(Q2[Q1.argmax()], 2))                           # expected output: 2.5

# %% [markdown]
# Policy gradient and exploration

# %%
# Policy gradient and exploration
def softmax(z): z = np.array(z, float); e = np.exp(z-z.max()); return e/e.sum()
print(np.round(softmax([2,1,0]), 3))                       # expected output: [0.665 0.245 0.09 ]
ent = lambda p: -sum(x*np.log(x) for x in p if x > 0)
print(round(ent([0.5,0.5]), 3), round(abs(ent([1.0])), 3)) # expected output: 0.693 0.0
print(round(4.58 - 4.0, 2))                                # expected output: 0.58
clip = lambda r, A, e=0.2: min(r*A, np.clip(r, 1-e, 1+e)*A)
print(round(clip(1.8, 1.0), 2), round(clip(0.5, 1.0), 2))  # expected output: 1.2 0.5

# %% [markdown]
# RLHF

# %%
# RLHF
sig = lambda z: 1/(1+np.exp(-z))
print(round(sig(2-1), 3))                                  # expected output: 0.731
gr = np.array([1.,0,0,1,1]); print(np.round((gr-gr.mean())/(gr.std()+1e-8), 2))
# expected output: [ 0.82 -1.22 -1.22  0.82  0.82]
ref = np.array([0.5,0.3,0.2]); rew = np.array([2.,1.,0.]); pi = ref*np.exp(rew/1.0); pi /= pi.sum()
print(np.round(pi, 2))                                     # expected output: [0.78 0.17 0.04]
print(round(1.0*(0.5 - (-0.3)), 2))                        # expected output: 0.8

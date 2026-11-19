"""Chapter 18: Reinforcement Learning

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from the chapter text; keep the book as the source.
"""

# %% [markdown]
# ## 18.1 The RL problem: agent, environment, state, action, reward

# %% [markdown]
# ## 18.1.3 How reinforcement learning differs from supervised learning

# %% [markdown]
# Example 18-1. The RL problem on a five-state corridor.

# %%
import numpy as np
rng = np.random.default_rng(0)

N_STATES = 5
GOAL = 4
def step(s, a):
    s_next = max(0, s - 1) if a == 0 else min(GOAL, s + 1)   # move, clamped to the corridor
    reward = 10.0 if s_next == GOAL else -1.0                # +10 at goal, -1 per move otherwise
    done = (s_next == GOAL)
    return s_next, reward, done

def run(policy, max_steps=50):
    s, total, steps = 0, 0.0, 0
    for _ in range(max_steps):
        a = policy(s)
        s, r, done = step(s, a)
        total += r; steps += 1
        if done: break
    return total, steps, done

random_returns = [run(lambda s: rng.integers(0, 2))[0] for _ in range(2000)]
print(f"random agent: average return over 2000 episodes = {np.mean(random_returns):.1f}")
print("it wanders, wasting -1 steps, and only stumbles to the goal by luck.")

opt_return, opt_steps, _ = run(lambda s: 1)
print(f"\nalways-right policy: return {opt_return:.0f} in {opt_steps} steps (the optimum here)")

def myopic(s, tie="left", gen=None):
    # best immediate reward; a tie carries no direction, so the default decides
    r_left = step(s, 0)[1]; r_right = step(s, 1)[1]
    if r_right > r_left:
        return 1
    if r_left > r_right:
        return 0
    if tie == "right":
        return 1
    if tie == "random":
        return int(gen.integers(0, 2))
    return 0                                          # default: ties -> left
myopic_return, myopic_steps, myopic_done = run(myopic)
print(f"\nmyopic (immediate-reward) policy: reached goal = {myopic_done}, "
      f"return {myopic_return:.0f} in {myopic_steps} steps")
print("both actions pay -1 until the goal is one step away, and this tie break chooses left.")
right_return, right_steps, right_done = run(lambda s: myopic(s, "right"))
print(f"myopic, ties toward right: reached goal = {right_done}, "
      f"return {right_return:.0f} in {right_steps} steps")
gen = np.random.default_rng(0)
rand_returns, rand_reached = [], 0
for _ in range(2000):
    total, _, done = run(lambda s: myopic(s, "random", gen))
    rand_returns.append(total); rand_reached += int(done)
print(f"myopic, random ties, seed 0: reached the goal in {rand_reached} of 2000 episodes, "
      f"average return {np.mean(rand_returns):.3f}")

print("\nRL vs supervised learning:")
print("  supervised: data is (input -> correct label); learn to copy the label")
print("  RL        : data is (state, action -> reward, next state); NO correct action given")
print("  - feedback is EVALUATIVE (how good was it?) not INSTRUCTIVE (what was right?)")
print("  - reward can be DELAYED, so actions must be credited over time (the credit-assignment problem)")
print("  - the agent's actions change the data it sees next (exploration matters)")

print("\nAll RL-problem checks ran.")

# %% [markdown]
# Check for Section 18.1.3: the left-tied myopic rule of Example 18-1 with the goal 1, 2, 4, 8, and 20 squares from the start.

# %%
# Example 18-1's one-step rule on corridors of other lengths: the goal sits `goal` squares
# from square 0, a move that misses it pays -1, reaching it pays +10, and a tie goes left.
def left_tied_reaches(goal, max_steps=50):
    s = 0
    for _ in range(max_steps):
        r_left = 10.0 if max(0, s - 1) == goal else -1.0
        r_right = 10.0 if min(goal, s + 1) == goal else -1.0
        s = min(goal, s + 1) if r_right > r_left else max(0, s - 1)
        if s == goal:
            return True
    return False

for goal in [1, 2, 4, 8, 20]:
    print(f"goal at square {goal:2d}: the left-tied agent reaches it = {left_tied_reaches(goal)}")

# %% [markdown]
# ## 18.2 Multi-armed bandits: exploration vs exploitation

# %% [markdown]
# At $Q = 0.6$, $N = 10$, $t = 100$, and $c = 1$, the UCB score is $0.6 + \sqrt{\ln 100 / 10} = 1.2786$, and the bonus alone is $0.6786$.

# %%
import math
Q = 0.0
Q = Q + (1.0 - Q) / 1
print(f"after reward 1: Q = {Q:.1f}")
Q = Q + (0.0 - Q) / 2
print(f"after reward 0: Q = {Q:.1f}")
bonus = math.sqrt(math.log(100) / 10)
print(f"UCB score: {0.6 + bonus:.4f}  bonus: {bonus:.4f}")
print(f"Beta(2,1) mean {2/3:.4f}; Beta(2,2) mean {0.5:.1f}; Beta(3,1) mean {3/4:.2f}")

# %% [markdown]
# Example 18-2 makes the comparison: greedy with $\alpha = 0.1$ on ten arms whose win rates are evenly spaced from 0.1 to 0.9, 40 runs of 1000 pulls from seed 0, once with every $Q$ started at 0 and once with every $Q$ started at 5.
#
# Example 18-2. Optimistic initial values: greedy at a constant step size of 0.1 on ten arms, with every estimate started at 0 and then at 5, and the weight left on the starting value after 10 and 20 pulls.

# %%
import numpy as np
rng = np.random.default_rng(0)
true_p = np.linspace(0.1, 0.9, 10)

def greedy(q0, runs=40, T=1000):
    rewards = []
    for _ in range(runs):
        Q = np.full(10, q0); N = np.zeros(10); total = 0.0
        for _ in range(T):
            a = int(np.argmax(Q))
            r = 1.0 if rng.random() < true_p[a] else 0.0
            N[a] += 1
            Q[a] += 0.1 * (r - Q[a])
            total += r
        rewards.append(total)
    return float(np.mean(rewards))

print(f"Q0=0 {greedy(0.0):.1f}")
print(f"Q0=5 {greedy(5.0):.1f}")
print(f"weight after 10 pulls {(1 - 0.1) ** 10:.4f}")
print(f"weight after 20 pulls {(1 - 0.1) ** 20:.4f}")

# %% [markdown]
# Example 18-3. Five bandit strategies compared by regret.

# %%
import numpy as np
rng = np.random.default_rng(0)

K = 10
T = 1000
RUNS = 300
true_p = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90])  # arm win-rates
best_p = true_p.max()                                  # the best arm pays 0.90 on average
print(f"{K}-armed Bernoulli bandit, best arm mean = {best_p:.2f}, horizon T = {T}")

def pull(a):
    return 1.0 if rng.random() < true_p[a] else 0.0    # Bernoulli reward

def greedy(eps):
    def algo():
        Q = np.zeros(K); N = np.zeros(K); reward = 0.0
        for t in range(T):
            if rng.random() < eps: a = rng.integers(K)        # explore
            else: a = int(np.argmax(Q))                       # exploit current best estimate
            r = pull(a); N[a] += 1; Q[a] += (r - Q[a]) / N[a] # running mean update
            reward += r
        return reward
    return algo

def ucb(c=1.0):
    def algo():
        Q = np.zeros(K); N = np.zeros(K); reward = 0.0
        for t in range(T):
            if t < K: a = t                                   # try each arm once
            else:
                bonus = c * np.sqrt(np.log(t + 1) / N)        # uncertainty bonus: less-tried arms
                a = int(np.argmax(Q + bonus))
            r = pull(a); N[a] += 1; Q[a] += (r - Q[a]) / N[a]
            reward += r
        return reward
    return algo

def thompson():
    # Bayesian: keep a Beta(successes+1, failures+1) belief per arm; SAMPLE and pick the max
    alpha = np.ones(K); beta = np.ones(K); reward = 0.0
    for t in range(T):
        samples = rng.beta(alpha, beta)
        a = int(np.argmax(samples))
        r = pull(a)
        if r > 0: alpha[a] += 1
        else: beta[a] += 1
        reward += r
    return reward

strategies = {
    "random        ": greedy(1.0),       # always explore = pure random
    "greedy (eps=0)": greedy(0.0),       # never explore
    "eps-greedy 0.1": greedy(0.1),
    "UCB           ": ucb(1.0),
    "Thompson      ": thompson,
}
print(f"\naveraged over {RUNS} runs (max possible total reward ~ {best_p*T:.0f}):")
for name, algo in strategies.items():
    rewards = np.array([algo() for _ in range(RUNS)])
    avg = rewards.mean()
    regret = best_p * T - avg
    print(f"  {name}: total reward {avg:6.1f}   regret {regret:6.1f}")

print("\ngreedy can lock onto a wrong arm and never recover; random explores but never")
print("exploits. eps-greedy, UCB, and Thompson all balance the two: Thompson lowest regret here,")
print("with eps-greedy and UCB close behind and far ahead of pure greedy or random.")
print("bandits are exploration vs exploitation in its purest form, and run live in ads,")
print("A/B testing, and recommendations every day.")

print("\nAll bandit checks ran.")

# %% [markdown]
# Check for Section 18.2: the greedy rule of Example 18-3 alone, with the arms as listed, reversed, shuffled on each run, and with ties broken at random.

# %%
# Greedy with no exploration coin on Example 18-3's ten arms, 300 runs per variant. One
# generator, seeded 0, serves the four variants in turn, so each variant continues its stream.
check_rng = np.random.default_rng(0)
arm_p = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90])

def greedy_variant(mode, runs=300, horizon=1000):
    totals, runs_that_moved = [], 0
    for _ in range(runs):
        if mode == "reversed":
            p = arm_p[::-1]
        elif mode == "shuffled":
            p = check_rng.permutation(arm_p)              # a fresh arm order for every run
        else:
            p = arm_p
        Q = np.zeros(10); N = np.zeros(10); total = 0.0; moved = False
        for _ in range(horizon):
            if mode == "random ties":
                a = int(check_rng.choice(np.flatnonzero(Q == Q.max())))
            else:
                a = int(np.argmax(Q))                     # a tie goes to the lowest index
            moved = moved or a != 0
            r = 1.0 if check_rng.random() < p[a] else 0.0
            N[a] += 1; Q[a] += (r - Q[a]) / N[a]; total += r
        totals.append(total); runs_that_moved += int(moved)
    return np.mean(totals), runs_that_moved

for mode in ["as listed", "reversed", "shuffled", "random ties"]:
    avg, moved = greedy_variant(mode)
    print(f"greedy, arms {mode:11s}: mean total reward {avg:.2f}; "
          f"runs that ever left the first arm of the list: {moved} of 300")

# %% [markdown]
# The taken arm's preference rises by $0.1 \times (1 - 1/3) = 0.066667$, and each other arm falls by $0.1 \times (1/3) = 0.033333$.

# %%
import numpy as np

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

H = np.zeros(3)
pi = softmax(H)
alpha = 0.1
taken = 0
reward = 1.0
baseline = 0.0
indicator = np.zeros(3)
indicator[taken] = 1.0
H = H + alpha * (reward - baseline) * (indicator - pi)
print("pi", np.round(pi, 6))
print("H", np.round(H, 6))

# %% [markdown]
# ## 18.2.1 Contextual bandits: when the best arm depends on the situation

# %% [markdown]
# Example 18-4. LinUCB versus a context-blind bandit.

# %%
import numpy as np
rng = np.random.default_rng(0)

d, K, T, RUNS = 5, 4, 3000, 10                     # context dim, arms, pulls, repeats

def one_run():
    Theta = rng.normal(0, 1, (K, d)) / np.sqrt(d)  # each arm's TRUE weight vector (hidden)
    xs = rng.normal(0, 1, (T, d))
    xs /= np.linalg.norm(xs, axis=1, keepdims=True)  # one unit-length context per step
    noise = 0.1 * rng.normal(size=T)
    out = {}
    for policy in ["random", "ucb", "linucb"]:
        A = [np.eye(d) for _ in range(K)]          # per-arm A_a = I + sum of x x^T
        b = [np.zeros(d) for _ in range(K)]        # per-arm b_a = sum of r x
        Q = np.zeros(K); N = np.zeros(K)           # context-blind running means (plain UCB)
        regret = 0.0
        for t in range(1, T + 1):
            x = xs[t - 1]
            means = Theta @ x                       # every arm's true expected reward HERE
            if policy == "linucb":
                scores = []
                alpha = 1.0                             # optimism scale; raise it to explore more
                for a_ in range(K):
                    Ainv = np.linalg.inv(A[a_])
                    theta_hat = Ainv @ b[a_]        # ridge-regression estimate for this arm
                    scores.append(theta_hat @ x + alpha * np.sqrt(x @ Ainv @ x))
                a_t = int(np.argmax(scores))
            elif policy == "ucb":                   # context-BLIND UCB from Example 18-3
                a_t = t - 1 if t <= K else int(np.argmax(Q + np.sqrt(np.log(t) / N)))
            else:
                a_t = int(rng.integers(K))
            r = means[a_t] + noise[t - 1]
            A[a_t] += np.outer(x, x); b[a_t] += r * x
            N[a_t] += 1; Q[a_t] += (r - Q[a_t]) / N[a_t]
            regret += means.max() - means[a_t]      # regret vs the best arm FOR THIS context
        out[policy] = regret
    return out

totals = {"random": 0.0, "ucb": 0.0, "linucb": 0.0}
for _ in range(RUNS):
    res = one_run()
    for key in totals: totals[key] += res[key]
print(f"contextual bandit: {K} arms, {d}-dim contexts, T = {T} pulls (mean of {RUNS} runs)")
for name, key in [("random", "random"), ("context-blind UCB", "ucb"), ("LinUCB", "linucb")]:
    print(f"  {name:18s}: cumulative regret {totals[key] / RUNS:7.1f}")
print("contexts are symmetric around zero, so every arm averages to zero and the")
print("context-blind learner ties random; LinUCB uses the context and its regret nearly stops growing.")

# %% [markdown]
# Check for Section 18.2.1: Example 18-4 rerun with the first arm's weights multiplied by three.

# %%
# The ten runs of Example 18-4 from seed 0, drawing the same random numbers in the same order.
# A scale of 1 reproduces Example 18-4's printout; a scale of 3 triples the first arm's weights.
def contextual_regrets(scale_first, d=5, K=4, T=3000, runs=10):
    gen = np.random.default_rng(0)
    totals = {"random": 0.0, "ucb": 0.0, "linucb": 0.0}
    for _ in range(runs):
        Theta = gen.normal(0, 1, (K, d)) / np.sqrt(d)
        Theta[0] *= scale_first
        xs = gen.normal(0, 1, (T, d))
        xs /= np.linalg.norm(xs, axis=1, keepdims=True)
        noise = 0.1 * gen.normal(size=T)
        for policy in ["random", "ucb", "linucb"]:
            A = [np.eye(d) for _ in range(K)]
            b = [np.zeros(d) for _ in range(K)]
            Q = np.zeros(K); N = np.zeros(K); regret = 0.0
            for t in range(1, T + 1):
                x = xs[t - 1]
                means = Theta @ x
                if policy == "linucb":
                    scores = []
                    for a_ in range(K):
                        Ainv = np.linalg.inv(A[a_])
                        theta_hat = Ainv @ b[a_]
                        scores.append(theta_hat @ x + 1.0 * np.sqrt(x @ Ainv @ x))
                    a_t = int(np.argmax(scores))
                elif policy == "ucb":
                    a_t = t - 1 if t <= K else int(np.argmax(Q + np.sqrt(np.log(t) / N)))
                else:
                    a_t = int(gen.integers(K))
                r = means[a_t] + noise[t - 1]
                A[a_t] += np.outer(x, x); b[a_t] += r * x
                N[a_t] += 1; Q[a_t] += (r - Q[a_t]) / N[a_t]
                regret += means.max() - means[a_t]
            totals[policy] += regret
    return {key: value / runs for key, value in totals.items()}

for scale in [1, 3]:
    res = contextual_regrets(scale)
    print(f"first arm's weights x{scale}: random {res['random']:.1f}, "
          f"context-blind UCB {res['ucb']:.1f}, LinUCB {res['linucb']:.1f}")

# %% [markdown]
# ## 18.3 Markov decision processes

# %% [markdown]
# ## 18.3.1 When the state is hidden: POMDPs and belief states

# %% [markdown]
# Three lines verify the arithmetic.

# %%
import numpy as np
b = np.array([0.5, 0.5])                      # belief: P(tiger-left), P(tiger-right)
O = np.array([[0.85, 0.15],                   # row = true state; P(hear-left), P(hear-right)
              [0.15, 0.85]])
for step, obs in enumerate(["left", "left"], 1):
    o = 0 if obs == "left" else 1
    b = O[:, o] * b                           # transitions are identity: the tiger stays put
    b /= b.sum()                              # divide by P(o | b): normalize
    print(f"after hearing 'left' {step} time(s): belief = {np.round(b, 3)}")

# %% [markdown]
# ## 18.3.3 Policies and value functions

# %% [markdown]
# Example 18-5. Discounting, state values, and action values.

# %%
import numpy as np

for g in [0.0, 0.5, 0.9, 0.99]:
    closed = 1 / (1 - g) if g < 1 else np.inf
    approx = sum(g ** k for k in range(1000))            # numeric check of the infinite sum
    print(f"  gamma={g:4}: sum of 1*gamma^k = {approx:8.2f}  (closed form 1/(1-gamma) = {closed:.2f})")
print("small gamma = myopic (only soon reward counts); gamma near 1 = far-sighted.")

print("\nreward 'now=1' vs 'wait 3 steps for 10':")
for g in [0.0, 0.8, 0.95, 1.0]:
    A = 1.0
    B = (g ** 3) * 10.0
    pick = "wait for 10" if B > A else "take 1 now"
    print(f"  gamma={g:4}: value(now)={A:.2f}  value(later)={B:.2f}  -> prefer: {pick}")
print("a patient agent (high gamma) waits for the bigger delayed reward; an impatient one grabs the 1.")

GOAL = 4
def reward(s_next): return 10.0 if s_next == GOAL else -1.0
def move(s, a): return max(0, s - 1) if a == 0 else min(GOAL, s + 1)

def value_of_policy(policy, gamma, max_steps=100):
    # Exact when the episode ends within max_steps. A policy that never
    # ends is cut off: always-left stops short of the closed form -1/(1-gamma).
    V = np.zeros(GOAL + 1)
    for s0 in range(GOAL):                               # goal state has value 0 (episode over)
        s, G, disc = s0, 0.0, 1.0
        for _ in range(max_steps):
            a = policy[s]; s2 = move(s, a); r = reward(s2)
            G += disc * r; disc *= gamma; s = s2
            if s == GOAL: break
        V[s0] = G
    return V

gamma = 0.9
right = {s: 1 for s in range(GOAL + 1)}                  # always go right (optimal)
left = {s: 0 for s in range(GOAL + 1)}                   # always go left (never reaches goal)
Vr = value_of_policy(right, gamma)
Vl = value_of_policy(left, gamma)
print(f"\nstate values V(s) at gamma={gamma} (state 4 = goal = 0):")
print(f"  always-right policy: {np.round(Vr, 2)}")
print(f"  always-left  policy: {np.round(Vl, 2)}")
print("V ranks policies: always-right has higher value in every state, so it is the better policy.")

def q_value(s, a, policy, gamma):
    s2 = move(s, a); r = reward(s2)
    Vnext = 0.0 if s2 == GOAL else value_of_policy(policy, gamma)[s2]
    return r + gamma * Vnext
print(f"\nunder always-right, from state 2:")
print(f"  Q(2, right) = {q_value(2, 1, right, gamma):.2f}   Q(2, left) = {q_value(2, 0, right, gamma):.2f}")
print("right beats left, so a greedy improvement would keep choosing right (the seed of Bellman, 18.4).")

print("\nAll MDP checks ran.")

# %% [markdown]
# ## 18.4 Bellman equations

# %% [markdown]
# ## 18.4.3 The optimality equation

# %% [markdown]
# Example 18-6. The Bellman equations three ways.

# %%
import numpy as np
rng = np.random.default_rng(0)

S, GOAL, gamma = 5, 4, 0.9
ACTIONS = [0, 1]                                         # 0 = left, 1 = right
p_success = 0.8                                          # "slippery": intended move works 80% of the time

def transitions(s, a):
    # returns list of (probability, next_state, reward); goal is absorbing with 0 reward
    if s == GOAL:
        return [(1.0, GOAL, 0.0)]
    intended = max(0, s - 1) if a == 0 else min(GOAL, s + 1)
    res = [(p_success, intended, 10.0 if intended == GOAL else -1.0),
           (1 - p_success, s, 10.0 if s == GOAL else -1.0)]   # slip = stay
    return res

policy = {s: 1 for s in range(S)}                        # always go right
P = np.zeros((S, S)); R = np.zeros(S)
for s in range(S):
    for prob, s2, r in transitions(s, policy[s]):
        P[s, s2] += prob
        R[s] += prob * r                                 # expected immediate reward
V_solve = np.linalg.solve(np.eye(S) - gamma * P, R)
print(f"V (solved linear Bellman system, always-right, slippery): {np.round(V_solve, 2)}")

def rollout(s, policy, max_steps=200):
    G, disc = 0.0, 1.0
    for _ in range(max_steps):
        if s == GOAL: break
        probs = transitions(s, policy[s])
        i = rng.choice(len(probs), p=[p for p, _, _ in probs])
        _, s2, r = probs[i]
        G += disc * r; disc *= gamma; s = s2
    return G
mc = np.array([np.mean([rollout(s, policy) for _ in range(4000)]) for s in range(S)])
print(f"V (Monte Carlo average of 4000 rollouts)                : {np.round(mc, 2)}")
print(f"max difference between the two = {np.abs(V_solve - mc).max():.3f}  (they agree: Bellman holds)")

Vstar = np.zeros(S)
for it in range(200):
    Vnew = Vstar.copy()
    for s in range(S):
        if s == GOAL: continue
        Vnew[s] = max(sum(prob * (r + gamma * Vstar[s2]) for prob, s2, r in transitions(s, a))
                      for a in ACTIONS)
    if np.max(np.abs(Vnew - Vstar)) < 1e-9:
        Vstar = Vnew; break
    Vstar = Vnew
opt_policy = {}
for s in range(GOAL):
    qs = [sum(prob * (r + gamma * Vstar[s2]) for prob, s2, r in transitions(s, a)) for a in ACTIONS]
    opt_policy[s] = int(np.argmax(qs))
print(f"\nV* (optimal values, value iteration, {it+1} sweeps): {np.round(Vstar, 2)}")
print(f"optimal policy (0=left,1=right) per state 0..3: {[opt_policy[s] for s in range(GOAL)]}")
print("every state says 'go right': the greedy policy w.r.t. V* is the optimal policy.")
print("the Bellman optimality equation is the fixed point that all the algorithms ahead chase.")

print("\nAll Bellman checks ran.")

# %% [markdown]
# Check for Sections 18.4.2 and 18.5.4: Example 18-6's rollouts counted step by step, and the shrink factor of each value-iteration sweep.

# %%
# Example 18-6's Monte Carlo part rerun from the same seed with a step counter (the same random
# draws, so the same values), then its value-iteration loop with each sweep's largest change kept.
counter_rng = np.random.default_rng(0)
env_steps = 0

def rollout_counted(s, max_steps=200):
    global env_steps
    G, disc = 0.0, 1.0
    for _ in range(max_steps):
        if s == GOAL: break
        probs = transitions(s, policy[s])
        i = counter_rng.choice(len(probs), p=[p for p, _, _ in probs])
        _, s2, r = probs[i]
        G += disc * r; disc *= gamma; s = s2
        env_steps += 1
    return G

mc_again = np.array([np.mean([rollout_counted(s) for _ in range(4000)]) for s in range(S)])
print(f"Monte Carlo values, same seed: {np.round(mc_again, 2)}")
print(f"environment steps in the 4000 rollouts from each state: {env_steps:,}")

V_check = np.zeros(S); changes = []
while True:
    V_next = V_check.copy()
    for s in range(S):
        if s == GOAL: continue
        V_next[s] = max(sum(prob * (r + gamma * V_check[s2]) for prob, s2, r in transitions(s, a))
                        for a in ACTIONS)
    changes.append(np.max(np.abs(V_next - V_check)))
    V_check = V_next
    if changes[-1] < 1e-9: break
ratios = np.array(changes[1:]) / np.array(changes[:-1])
print(f"value iteration: {len(changes)} sweeps; each sweep's largest change divided by the previous "
      f"one runs from {ratios.min():.3f} to {ratios.max():.3f}")

# %% [markdown]
# ## 18.5 Dynamic programming: policy evaluation, policy iteration, value iteration

# %% [markdown]
# ## 18.5.4 Why it converges: the Bellman operator is a contraction

# %% [markdown]
# On the deterministic grid of Example 18-7, 1000 random pairs from seed 0 never beat the factor 0.9:

# %%
import numpy as np
rng = np.random.default_rng(0)
SIZE, GOAL, gamma = 4, 15, 0.9
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def nxt(s, a):
    if s == GOAL:
        return s
    r, c = divmod(s, SIZE)
    r2 = min(max(r + a[0], 0), SIZE - 1)
    c2 = min(max(c + a[1], 0), SIZE - 1)
    return r2 * SIZE + c2

def bellman(V):
    out = np.zeros(SIZE * SIZE)
    for s in range(SIZE * SIZE):
        if s == GOAL:
            continue
        out[s] = max(-1.0 + gamma * V[nxt(s, a)] for a in ACTIONS)
    return out

ratios = []
for _ in range(1000):
    V = rng.normal(size=SIZE * SIZE)
    U = rng.normal(size=SIZE * SIZE)
    V[GOAL] = U[GOAL] = 0.0
    den = np.max(np.abs(V - U))
    ratios.append(np.max(np.abs(bellman(V) - bellman(U))) / den)
print(f"largest ||TV - TU|| / ||V - U|| over 1000 pairs: {max(ratios):.4f}")

# %% [markdown]
# Example 18-7. Policy iteration and value iteration on a grid world.

# %%
import numpy as np

SIZE = 4
GOAL = SIZE * SIZE - 1                                    # state 15, bottom-right
gamma = 0.9
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)} # up, down, left, right
ARROW = {0: "^", 1: "v", 2: "<", 3: ">"}

def step(s, a):
    if s == GOAL: return s, 0.0
    r, c = divmod(s, SIZE)
    dr, dc = ACTIONS[a]
    r2, c2 = min(max(r + dr, 0), SIZE - 1), min(max(c + dc, 0), SIZE - 1)
    s2 = r2 * SIZE + c2
    return s2, -1.0                                       # every move costs -1 until absorbed

def evaluate(policy, tol=1e-8):
    V = np.zeros(SIZE * SIZE); sweeps = 0
    while True:
        delta = 0.0; sweeps += 1
        for s in range(SIZE * SIZE):
            if s == GOAL: continue
            v = sum(policy[s][a] * (lambda s2, r: r + gamma * V[s2])(*step(s, a)) for a in ACTIONS)
            delta = max(delta, abs(v - V[s])); V[s] = v
        if delta < tol: return V, sweeps

uniform = {s: {a: 0.25 for a in ACTIONS} for s in range(SIZE * SIZE)}   # random walk policy
V_uniform, sw = evaluate(uniform)
print(f"value of the random-walk policy ({sw} sweeps), as a 4x4 grid:")
print(np.round(V_uniform.reshape(SIZE, SIZE), 1))
print("(deeply negative far from the goal: a random walk takes ages to arrive)")

def greedy_policy(V):
    pol = {}
    for s in range(SIZE * SIZE):
        q = [(lambda s2, r: r + gamma * V[s2])(*step(s, a)) for a in ACTIONS]
        best = int(np.argmax(q)); pol[s] = {a: 1.0 if a == best else 0.0 for a in ACTIONS}
    return pol

def policy_iteration():
    pol = uniform; rounds = 0
    while True:
        rounds += 1
        V, _ = evaluate(pol)
        new = greedy_policy(V)
        if all(np.argmax(list(new[s].values())) == np.argmax(list(pol[s].values()))
               for s in range(SIZE * SIZE)):
            return V, new, rounds
        pol = new

V_pi, pol_pi, rounds = policy_iteration()
print(f"\npolicy iteration converged in {rounds} improvement rounds.")

def value_iteration(tol=1e-8):
    V = np.zeros(SIZE * SIZE); sweeps = 0
    while True:
        delta = 0.0; sweeps += 1
        for s in range(SIZE * SIZE):
            if s == GOAL: continue
            v = max((lambda s2, r: r + gamma * V[s2])(*step(s, a)) for a in ACTIONS)
            delta = max(delta, abs(v - V[s])); V[s] = v
        if delta < tol: return V, sweeps

V_vi, sweeps = value_iteration()
pol_vi = greedy_policy(V_vi)
print(f"value iteration converged in {sweeps} sweeps.")

print(f"\nmax |V_policy-iteration - V_value-iteration| = {np.abs(V_pi - V_vi).max():.2e}")
print("optimal values V* as a grid:")
print(np.round(V_vi.reshape(SIZE, SIZE), 1))
print("optimal policy (arrows point the way to the goal at bottom-right):")
grid = np.array([ARROW[int(np.argmax(list(pol_vi[s].values())))] for s in range(SIZE * SIZE)])
grid[GOAL] = "*"
for row in grid.reshape(SIZE, SIZE):
    print("   " + " ".join(row))
print("\nDP is the gold standard WHEN the model is known; the rest of the chapter learns")
print("the same answers from experience, WITHOUT knowing P and R in advance.")

print("\nAll dynamic-programming checks ran.")

# %% [markdown]
# Check for Section 18.5.2: the policy iteration of Example 18-7, round by round.

# %%
# Each round evaluates the current policy, compares its values with V* from value iteration,
# and counts the states whose greedy arrow changes; round 2 also prints the one-step values there.
pol_check = uniform
for rnd in range(1, 4):
    V_round, _ = evaluate(pol_check)
    new = greedy_policy(V_round)
    changed = [s for s in range(SIZE * SIZE)
               if np.argmax(list(new[s].values())) != np.argmax(list(pol_check[s].values()))]
    print(f"round {rnd}: max |V - V*| = {np.abs(V_round - V_vi).max():.2e}, "
          f"states whose arrow changes: {len(changed)}")
    if rnd == 2:
        for s in changed:
            q = [(lambda s2, r: r + gamma * V_round[s2])(*step(s, a)) for a in ACTIONS]
            old_arrow = ARROW[int(np.argmax(list(pol_check[s].values())))]
            new_arrow = ARROW[int(np.argmax(list(new[s].values())))]
            print(f"  state {s:2d}: one-step values (up, down, left, right) {np.round(q, 4)}, "
                  f"arrow {old_arrow} -> {new_arrow}")
    pol_check = new

# %% [markdown]
# ## 18.6 Learning from experience: Monte Carlo and temporal-difference learning

# %% [markdown]
# ## 18.6.2 Temporal-difference learning: bootstrap after every step

# %% [markdown]
# Example 18-8. Monte Carlo versus temporal-difference learning.

# %%
import numpy as np
rng = np.random.default_rng(0)

S, GOAL, gamma, p_success = 5, 4, 0.9, 0.8
def transitions(s, a):
    if s == GOAL: return [(1.0, GOAL, 0.0)]
    intended = max(0, s - 1) if a == 0 else min(GOAL, s + 1)
    return [(p_success, intended, 10.0 if intended == GOAL else -1.0),
            (1 - p_success, s, -1.0)]
def sample(s, a):
    probs = transitions(s, a)
    i = rng.choice(len(probs), p=[p for p, _, _ in probs])
    return probs[i][1], probs[i][2]

# exact V for the always-right policy (from the linear Bellman solve in 18.4)
V_true = np.array([3.21, 5.04, 7.13, 9.51, 0.0])
policy = {s: 1 for s in range(S)}
def episode():
    traj, s = [], 0
    while s != GOAL:
        s2, r = sample(s, policy[s]); traj.append((s, r)); s = s2
    return traj

def monte_carlo(n_ep):
    V = np.zeros(S); counts = np.zeros(S)
    rmse = []
    for _ in range(n_ep):
        traj = episode()
        G = 0.0
        for s, r in reversed(traj):                      # accumulate return backwards
            G = r + gamma * G
            counts[s] += 1; V[s] += (G - V[s]) / counts[s]   # running average of returns
        rmse.append(np.sqrt(np.mean((V[:GOAL] - V_true[:GOAL]) ** 2)))
    return V, rmse

def td0(n_ep, alpha=0.05):
    V = np.zeros(S); rmse = []
    for _ in range(n_ep):
        s = 0
        while s != GOAL:
            s2, r = sample(s, policy[s])
            V[s] += alpha * (r + gamma * V[s2] - V[s])   # the TD update (one line)
            s = s2
        rmse.append(np.sqrt(np.mean((V[:GOAL] - V_true[:GOAL]) ** 2)))
    return V, rmse

n_ep = 500
V_mc, rmse_mc = monte_carlo(n_ep)
V_td, rmse_td = td0(n_ep)
print(f"true V (DP)      : {np.round(V_true, 2)}")
print(f"Monte Carlo est. : {np.round(V_mc, 2)}  (final RMSE {rmse_mc[-1]:.3f})")
print(f"TD(0) estimate   : {np.round(V_td, 2)}  (final RMSE {rmse_td[-1]:.3f})")
print("both approach the exact DP values WITHOUT ever being told P or R: learned from samples.")

mc_targets = []                                          # full returns starting at state 0
for _ in range(5000):
    traj = episode(); G = 0.0
    for s, r in reversed(traj): G = r + gamma * G
    mc_targets.append(G)
td_targets = []                                          # one-step bootstrap targets at state 0
for _ in range(5000):
    s2, r = sample(0, policy[0]); td_targets.append(r + gamma * V_true[s2])
print(f"\ntarget for state 0 (true value {V_true[0]:.2f}):")
print(f"  Monte Carlo target (full return): mean {np.mean(mc_targets):.2f}, std {np.std(mc_targets):.2f}  (high variance)")
print(f"  TD target (r + gamma*V(next))   : mean {np.mean(td_targets):.2f}, std {np.std(td_targets):.2f}  (low variance)")
print("TD's target depends on ONE random step, MC's on the whole episode: that is why TD is")
print("lower-variance and can learn online, while MC is unbiased but noisy and must wait for the end.")

print("\nAll MC-vs-TD checks ran.")

# %% [markdown]
# Check cell for Figure 18-4 (continues Example 18-8): it rebuilds both curves, drawing from the same seed in the same order as the figure script, and prints the values the text quotes.

# %%
# Check cell for Figure 18-4 (continues Example 18-8). figures/src/fig_18_04_mc-vs-td.py averages 50 runs
# of each method drawn from one generator seeded 0, all Monte Carlo runs first; the same calls here
# rebuild its two curves and print the values the text quotes.
rng = np.random.default_rng(0)                     # sample() draws from this generator
curve_mc = np.mean([monte_carlo(500)[1] for _ in range(50)], axis=0)
curve_td = np.mean([td0(500)[1] for _ in range(50)], axis=0)
for ep in [1, 50, 100, 500]:
    print(f"episode {ep:3d}: Monte Carlo RMSE {curve_mc[ep - 1]:.3f}   TD(0) RMSE {curve_td[ep - 1]:.3f}")
print("Monte Carlo below TD(0) at every episode:", bool(np.all(curve_mc < curve_td)))
late = slice(200, 500)                             # episodes 201 to 500
print(f"episodes 201 to 500: TD(0) mean RMSE {curve_td[late].mean():.3f}")
print(f"mean absolute change between consecutive episodes, 201 to 500: "
      f"Monte Carlo {np.abs(np.diff(curve_mc[late])).mean():.5f}, TD(0) {np.abs(np.diff(curve_td[late])).mean():.5f}")

# %% [markdown]
# ## 18.6.3 Control: SARSA versus Q-learning

# %% [markdown]
# Example 18-9. SARSA versus Q-learning on the Cliff Walking task.

# %%
import numpy as np
import gymnasium as gym

env = gym.make("CliffWalking-v1")           # standard 4x12 cliff grid, Discrete(48) state, 4 actions
N_STATES = env.observation_space.n          # 48 states, indexed as plain integers
N_ACTIONS = env.action_space.n              # 4 actions: 0=up, 1=right, 2=down, 3=left
START, GOAL = 36, 47                         # bottom-left start, bottom-right goal; cliff = states 37..46

def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps: return int(rng.integers(N_ACTIONS))
    return int(np.argmax(Q[s]))

def train(method, episodes, alpha=0.5, eps=0.1, gamma=1.0, seed=0):
    rng = np.random.default_rng(seed)                     # this run's own generator: the seed sets every exploration draw
    Q = np.zeros((N_STATES, N_ACTIONS))                   # one table row per integer state
    returns = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)                  # the grid is deterministic; only the agent's draws are random
        a = eps_greedy(Q, s, eps, rng); total, done = 0.0, False
        while not done:
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc; total += r
            a2 = eps_greedy(Q, s2, eps, rng)
            if method == "sarsa":
                target = r + gamma * Q[s2][a2]            # value of the action actually taken next
            else:                                         # q-learning
                target = r + gamma * np.max(Q[s2])        # value of the BEST next action
            Q[s][a] += alpha * (target - Q[s][a])
            s, a = s2, a2
        returns.append(total)
    return Q, returns

def greedy_path(Q, max_steps=50):
    # follow the greedy policy; CAP the rollout so a looping policy never runs to truncation
    s, _ = env.reset(seed=0); total, rows = 0.0, {s // 12}
    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))
        s, r, term, trunc, _ = env.step(a); total += r; rows.add(s // 12)
        if term or trunc: break
    return total, sorted(rows)

n_runs = 30
sarsa_online, q_online = [], []
for run in range(n_runs):
    _, rs = train("sarsa", 500, seed=run * 1000)
    _, rq = train("qlearn", 500, seed=run * 1000 + 500)
    sarsa_online.append(np.mean(rs[-100:])); q_online.append(np.mean(rq[-100:]))

print(f"CliffWalking-v1: Discrete({N_STATES}) states, {N_ACTIONS} actions, start={START}, goal={GOAL}")
print(f"mean ONLINE return over the last 100 of 500 training episodes (averaged over {n_runs} runs):")
print(f"  SARSA      (on-policy) : {np.mean(sarsa_online):7.1f}   safer route -> higher online return")
print(f"  Q-learning (off-policy): {np.mean(q_online):7.1f}   optimal but risky -> more cliff falls while exploring")

# read off the converged greedy policy each learns (train a little longer so corner states settle)
Qs, _ = train("sarsa", 1000, seed=7)
Qq, _ = train("qlearn", 1000, seed=11)
sp, qp = greedy_path(Qs), greedy_path(Qq)
print(f"\ngreedy path each learned (grid rows used; row 3 = cliff edge, row 0 = top):")
print(f"  SARSA greedy path     : reward {sp[0]:.0f}, travels rows {sp[1]} (detours UP, away from the cliff)")
print(f"  Q-learning greedy path: reward {qp[0]:.0f}, travels rows {qp[1]} (hugs the cliff edge = shortest)")

print("\nthe interview point: Q-learning learns the OPTIMAL policy's values (max over next")
print("actions), so its greedy path is the shortest cliff-edge route. SARSA learns the value")
print("of the eps-greedy policy it actually follows, which sometimes slips off the cliff, so")
print("it prefers a safer detour. On-policy values what you DO; off-policy values the BEST you COULD do.")
env.close()
print("\nAll SARSA/Q-learning checks ran.")

# %% [markdown]
# ## 18.6.4 Between the two: n-step TD and TD(lambda)

# %% [markdown]
# Example 18-10. n-step TD and TD($\lambda$) interpolating between TD(0) and Monte Carlo.

# %%
import numpy as np

S, GOAL, gamma, p_success = 5, 4, 0.9, 0.8
V_true = np.array([3.21, 5.04, 7.13, 9.51, 0.0])         # exact V for always-right (18.4)

def transitions(s):                                       # always-right policy
    if s == GOAL: return [(1.0, GOAL, 0.0)]
    intended = min(GOAL, s + 1)
    return [(p_success, intended, 10.0 if intended == GOAL else -1.0),
            (1 - p_success, s, -1.0)]

def rmse(V):
    return np.sqrt(np.mean((V[:GOAL] - V_true[:GOAL]) ** 2))

def episode(rng):
    traj, s = [], 0
    while s != GOAL:
        pr = transitions(s); i = rng.choice(len(pr), p=[q for q, _, _ in pr])
        traj.append((s, pr[i][2])); s = pr[i][1]
    return traj

def nstep_td(n, n_ep, alpha, rng):
    V = np.zeros(S); errs = []
    for _ in range(n_ep):
        traj = episode(rng); states = [s for s, _ in traj]; rews = [r for _, r in traj]; T = len(traj)
        for t in range(T):
            G, k = 0.0, 0
            while k < n and t + k < T:                    # sum up to n real discounted rewards
                G += (gamma ** k) * rews[t + k]; k += 1
            if t + k < T: G += (gamma ** k) * V[states[t + k]]   # then bootstrap on the landing state
            V[states[t]] += alpha * (G - V[states[t]])
        errs.append(rmse(V))
    return np.mean(errs)

def td_lambda(lam, n_ep, alpha, rng):
    V = np.zeros(S); errs = []
    for _ in range(n_ep):
        e = np.zeros(S); s = 0                            # eligibility traces reset each episode
        while s != GOAL:
            pr = transitions(s); i = rng.choice(len(pr), p=[q for q, _, _ in pr])
            s2, r = pr[i][1], pr[i][2]
            delta = r + gamma * V[s2] - V[s]              # TD error
            e[s] += 1.0                                   # visited state becomes fully eligible
            V += alpha * delta * e                        # credit every eligible state at once
            e *= gamma * lam                              # decay all traces
            s = s2
        errs.append(rmse(V))
    return np.mean(errs)

N_EP, ALPHA, RUNS = 300, 0.05, 30
print(f"average RMSE-to-true-V over {N_EP} episodes (mean of {RUNS} seeds), alpha={ALPHA}")
print("n-step TD  (n=1 is TD(0); n=16 never bootstraps in these 9000 episodes, so it is constant-step Monte Carlo):")
for n in [1, 2, 4, 8, 16]:
    v = np.mean([nstep_td(n, N_EP, ALPHA, np.random.default_rng(r)) for r in range(RUNS)])
    tag = "  <- TD(0)" if n == 1 else ("  <- Monte Carlo" if n == 16 else "")
    print(f"  n={n:2d}: avg RMSE {v:.3f}{tag}")

print("\nTD(lambda) (lambda=0 is TD(0); lambda=1 is online TD(1); between them a blend):")
for lam in [0.0, 0.4, 0.8, 0.95, 1.0]:
    v = np.mean([td_lambda(lam, N_EP, ALPHA, np.random.default_rng(r)) for r in range(RUNS)])
    tag = "  <- TD(0)" if lam == 0.0 else ("  <- online TD(1)" if lam == 1.0 else "")
    print(f"  lambda={lam:.2f}: avg RMSE {v:.3f}{tag}")

# %% [markdown]
# Check cell (continues Example 18-10): the same 30 seeds at `alpha` = 0.5 and 0.9.

# %%
# Check cell (continues Example 18-10): the same 30 seeds and 300 episodes at two larger step sizes.
for a in [0.5, 0.9]:
    errs = [np.mean([nstep_td(n, N_EP, a, np.random.default_rng(r)) for r in range(RUNS)]) for n in [1, 2, 4, 16]]
    print(f"alpha={a}: avg RMSE for n = 1, 2, 4, 16: " + "  ".join(f"{v:.3f}" for v in errs))

# %% [markdown]
# ## 18.6.5 Monte Carlo control: learning a policy from played episodes

# %% [markdown]
# Example 18-11. First-visit versus every-visit Monte Carlo, and epsilon-soft Monte Carlo control on Blackjack.

# %%
import numpy as np
import gymnasium as gym
rng = np.random.default_rng(0)

# ---- Part A: first-visit vs every-visit MC prediction on the slippery corridor ----
S, GOAL, gamma, p_success = 5, 4, 0.9, 0.8
V_true = np.array([3.21, 5.04, 7.13, 9.51, 0.0])
def transitions(s):
    if s == GOAL: return [(1.0, GOAL, 0.0)]
    intended = min(GOAL, s + 1)
    return [(p_success, intended, 10.0 if intended == GOAL else -1.0),
            (1 - p_success, s, -1.0)]
def episode():
    traj, s = [], 0
    while s != GOAL:
        pr = transitions(s); i = rng.choice(len(pr), p=[q for q, _, _ in pr])
        traj.append((s, pr[i][2])); s = pr[i][1]
    return traj

def mc_predict(first_visit, n_ep=3000):
    V = np.zeros(S); counts = np.zeros(S)
    for _ in range(n_ep):
        traj = episode(); G = 0.0
        for t in range(len(traj) - 1, -1, -1):
            s, r = traj[t]; G = r + gamma * G
            if first_visit and s in {x for x, _ in traj[:t]}:
                continue                                   # an earlier-in-time visit exists: skip
            counts[s] += 1; V[s] += (G - V[s]) / counts[s]
    return V

Vf = mc_predict(True); Ve = mc_predict(False)
print("first-visit vs every-visit MC prediction (always-right, slippery corridor):")
print(f"  true V       : {np.round(V_true[:GOAL], 2)}")
print(f"  first-visit  : {np.round(Vf[:GOAL], 2)}  RMSE {np.sqrt(np.mean((Vf[:GOAL]-V_true[:GOAL])**2)):.3f}")
print(f"  every-visit  : {np.round(Ve[:GOAL], 2)}  RMSE {np.sqrt(np.mean((Ve[:GOAL]-V_true[:GOAL])**2)):.3f}")
print("  both converge to the same true values; they only differ in how repeat visits are counted.")

# ---- Part B: on-policy epsilon-soft MC control on Blackjack-v1 (the textbook MC control env) ----
env = gym.make("Blackjack-v1", sab=True)                  # sab=True: Sutton-Barto rules
gamma_bj = 1.0                                            # short episodic task: undiscounted return (Sutton-Barto convention)
nA = env.action_space.n                                   # 2 actions: 0=stick, 1=hit
Q, N = {}, {}
def qrow(s): return Q.setdefault(s, np.zeros(nA)), N.setdefault(s, np.zeros(nA))

def eps_soft_action(s, eps):
    q, _ = qrow(s)
    if rng.random() < eps: return int(rng.integers(nA))
    return int(np.argmax(q))

def train_mc_control(n_ep, eps=0.1):
    for _ in range(n_ep):
        s, _ = env.reset(seed=int(rng.integers(1 << 30)))
        traj, done = [], False
        while not done:
            a = eps_soft_action(s, eps)
            s2, r, term, trunc, _ = env.step(a); done = term or trunc
            traj.append((s, a, r)); s = s2
        G = 0.0
        for t in range(len(traj) - 1, -1, -1):            # first-visit: skip a pair seen earlier in the hand
            s, a, r = traj[t]; G = r + gamma_bj * G
            if (s, a) in {(x, y) for x, y, _ in traj[:t]}:
                continue
            q, n = qrow(s); n[a] += 1; q[a] += (G - q[a]) / n[a]

def win_rate(policy, n_ep=20000):
    wins = draws = 0
    for _ in range(n_ep):
        s, _ = env.reset(seed=int(rng.integers(1 << 30))); done = False; r = 0.0
        while not done:
            a = policy(s); s, r, term, trunc, _ = env.step(a); done = term or trunc
        wins += (r > 0); draws += (r == 0)
    return wins / n_ep, draws / n_ep

wr_rand, dr_rand = win_rate(lambda s: int(rng.integers(nA)))
train_mc_control(300000, eps=0.1)
wr_greedy, dr_greedy = win_rate(lambda s: int(np.argmax(qrow(s)[0])))
print(f"\non-policy epsilon-soft MC control on Blackjack-v1 (300k training episodes, eps=0.1):")
print(f"  random policy : win rate {wr_rand:.3f}, draw rate {dr_rand:.3f}")
print(f"  learned greedy: win rate {wr_greedy:.3f}, draw rate {dr_greedy:.3f}")
print("  MC control raises the win rate with no model of the deck: it learns purely from played hands.")
env.close()

# %% [markdown]
# Check cell (continues Example 18-11): `eps` = 0 from a fresh table, and repeated state-action pairs in 20,000 random hands.

# %%
# Check cell (continues Example 18-11): the eps = 0 experiment the text suggests, and the claim that no
# state-action pair repeats inside a hand. A fresh table and a fresh seed make both reproducible.
rng = np.random.default_rng(0)                     # eps_soft_action, train_mc_control and win_rate draw from rng
env = gym.make("Blackjack-v1", sab=True)
Q, N = {}, {}                                      # qrow() reads these names, so the table starts at 0 again
train_mc_control(300000, eps=0.0)                  # pure greedy: no forced exploration at all
wr0, dr0 = win_rate(lambda s: int(np.argmax(qrow(s)[0])))
print(f"eps=0, table starting at 0: greedy win rate {wr0:.3f}, draw rate {dr0:.3f}")

check_rng = np.random.default_rng(0)               # 20,000 random hands: count pairs seen twice in one hand
repeats = 0
for _ in range(20000):
    s, _ = env.reset(seed=int(check_rng.integers(1 << 30))); done, seen = False, set()
    while not done:
        a = int(check_rng.integers(nA))
        repeats += (s, a) in seen; seen.add((s, a))
        s, r, term, trunc, _ = env.step(a); done = term or trunc
print(f"20,000 random hands: {repeats} state-action pairs repeated inside a hand")
env.close()

# %% [markdown]
# Example 18-12. Ordinary and weighted importance sampling, and Monte Carlo control with exploring starts, on the slippery corridor.

# %%
import numpy as np

# the slippery corridor of Example 18-6: states 0 to 4, goal 4, action 0 = left and 1 = right;
# the intended move works with probability 0.8, otherwise the agent stays put; every step
# costs -1 except the step that enters the goal, which pays +10
S, GOAL, gamma, p_success = 5, 4, 0.9, 0.8
def step(s, a, rng):
    intended = max(0, s - 1) if a == 0 else min(GOAL, s + 1)
    s2 = intended if rng.random() < p_success else s
    return s2, (10.0 if s2 == GOAL else -1.0)

# ---- Part A: off-policy prediction of V_pi(0) for pi = always right, from episodes of b = uniform ----
def behavior_episode(rng):
    s, G, disc, rho = 0, 0.0, 1.0, 1.0
    while s != GOAL:
        a = int(rng.integers(2))                          # b picks left or right, 1/2 each
        rho *= (1.0 if a == 1 else 0.0) / 0.5             # pi(a | s) / b(a | s)
        if rho == 0.0:
            return 0.0, 0.0     # a left step: pi never does this, so the weight stays 0 whatever follows
        s, r = step(s, a, rng)
        G += disc * r; disc *= gamma
    return rho, G

V_TRUE, SEEDS, SIZES = 3.21, 100, [10, 100, 1000, 10000]
ordinary = {n: [] for n in SIZES}; weighted = {n: [] for n in SIZES}; empty = {n: 0 for n in SIZES}; usable = 0
for seed in range(SEEDS):
    rng = np.random.default_rng(seed)
    rho, G = np.array([behavior_episode(rng) for _ in range(SIZES[-1])]).T
    usable += np.count_nonzero(rho)
    for n in SIZES:                                       # the first n episodes of this seed
        total = np.sum(rho[:n])
        ordinary[n].append(np.sum(rho[:n] * G[:n]) / n)
        weighted[n].append(np.sum(rho[:n] * G[:n]) / total if total > 0 else 0.0)
        empty[n] += total == 0                            # no episode with a nonzero weight yet
print(f"off-policy MC on the slippery corridor: b uniform, pi always right, true V_pi(0) = {V_TRUE}")
print(f"behavior episodes with a nonzero weight: {usable / (SEEDS * SIZES[-1]):.4f}")
print(f"mean and standard deviation over {SEEDS} seeds:")
print("  episodes   ordinary IS        weighted IS        seeds with every weight 0")
for n in SIZES:
    o, w = np.array(ordinary[n]), np.array(weighted[n])
    print(f"  {n:8d}   {o.mean():5.2f} +/- {o.std():4.2f}   {w.mean():5.2f} +/- {w.std():4.2f}   {empty[n]:3d}")

# ---- Part B: Monte Carlo control, greedy policy, with and without exploring starts ----
def mc_control(exploring_starts, episodes=5000, seed=0, max_steps=100):
    rng = np.random.default_rng(seed)
    Q = np.full((S, 2), -11.0)    # below -10, the return of stepping left forever: a tried action beats an untried one
    N = np.zeros((S, 2))
    for _ in range(episodes):
        if exploring_starts:                              # a random nonterminal state and a random first action
            s, a = int(rng.integers(GOAL)), int(rng.integers(2))
        else:                                             # always the start state and the greedy action
            s, a = 0, int(np.argmax(Q[0]))
        traj = []
        for _ in range(max_steps):                        # cap: 0.9**100 makes the cut-off tail negligible
            s2, r = step(s, a, rng); traj.append((s, a, r)); s = s2
            if s == GOAL: break
            a = int(np.argmax(Q[s]))                      # then follow the greedy policy
        first = {}
        for t, (x, y, _) in enumerate(traj): first.setdefault((x, y), t)
        G = 0.0
        for t in range(len(traj) - 1, -1, -1):            # first-visit average of returns
            x, y, r = traj[t]; G = r + gamma * G
            if first[(x, y)] == t:
                N[x, y] += 1; Q[x, y] += (G - Q[x, y]) / N[x, y]
    return Q, N

names = ["left", "right"]
print("\nMonte Carlo control on the same corridor, 5000 episodes, every Q starting at -11:")
for es in [False, True]:
    Q, N = mc_control(es)
    label = "exploring starts" if es else "fixed start 0   "
    print(f"  {label}: greedy {[names[int(np.argmax(Q[s]))] for s in range(GOAL)]}, "
          f"Q(s, right) {np.round(Q[:GOAL, 1], 2)}, tries at state 0 {N[0].astype(int)}")

# %% [markdown]
# ## 18.6.6 Expected SARSA and Double Q-learning: fixing maximization bias

# %% [markdown]
# Example 18-13. Maximization bias: Q-learning versus Double Q-learning, with Expected SARSA.

# %%
import numpy as np
rng = np.random.default_rng(0)

# State A: action 0 = LEFT -> state B (reward 0); action 1 = RIGHT -> terminal (reward 0).
# State B: N_B actions, each -> terminal with reward ~ Normal(-0.1, 1).
# E[reward from B] = -0.1 < 0, so the OPTIMAL choice at A is RIGHT; going LEFT is a mistake.
A_LEFT, A_RIGHT = 0, 1
N_B = 10
EPS, ALPHA, GAMMA = 0.1, 0.1, 1.0

def b_reward(): return rng.normal(-0.1, 1.0)

def eps_greedy(q, eps, n):
    if rng.random() < eps: return int(rng.integers(n))
    return int(np.argmax(q))

def run(method, episodes=300):
    QA = np.zeros(2); QB = np.zeros(N_B)
    QA2 = np.zeros(2); QB2 = np.zeros(N_B)                 # second tables for Double Q-learning
    left_flags = []
    for _ in range(episodes):
        a = eps_greedy(QA + QA2 if method == "double" else QA, EPS, 2)
        left_flags.append(1 if a == A_LEFT else 0)
        if a == A_RIGHT:                                   # straight to terminal, reward 0
            if method == "double":
                if rng.random() < 0.5: QA[a] += ALPHA * (0.0 - QA[a])
                else: QA2[a] += ALPHA * (0.0 - QA2[a])
            else:
                QA[a] += ALPHA * (0.0 - QA[a])
            continue
        b = eps_greedy(QB + QB2 if method == "double" else QB, EPS, N_B)
        r = b_reward()
        if method == "qlearn":
            QB[b] += ALPHA * (r - QB[b])
            QA[A_LEFT] += ALPHA * (GAMMA * np.max(QB) - QA[A_LEFT])          # biased max
        elif method == "expected":
            QB[b] += ALPHA * (r - QB[b])
            pi = np.full(N_B, EPS / N_B); pi[np.argmax(QB)] += 1 - EPS
            QA[A_LEFT] += ALPHA * (GAMMA * np.dot(pi, QB) - QA[A_LEFT])      # expected next value
        else:  # double q-learning: one estimator selects, the OTHER evaluates
            if rng.random() < 0.5:
                QB[b] += ALPHA * (r - QB[b])
                a_star = int(np.argmax(QB)); QA[A_LEFT] += ALPHA * (GAMMA * QB2[a_star] - QA[A_LEFT])
            else:
                QB2[b] += ALPHA * (r - QB2[b])
                a_star = int(np.argmax(QB2)); QA2[A_LEFT] += ALPHA * (GAMMA * QB[a_star] - QA2[A_LEFT])
    return np.array(left_flags)

RUNS = 2000
for method, label in [("qlearn", "Q-learning      "), ("double", "Double Q-learning"),
                      ("expected", "Expected SARSA  ")]:
    curves = np.mean([run(method) for _ in range(RUNS)], axis=0)
    print(f"  {label}: %LEFT-from-A   first 100 episodes {100*curves[:100].mean():5.1f}%   "
          f"last 100 {100*curves[-100:].mean():5.1f}%")
print(f"\noptimal is RIGHT, so a good agent takes LEFT only ~{100*EPS/2:.0f}% (forced by exploration).")
print("Q-learning's max over noisy Q(B,.) is biased HIGH, so it over-chooses LEFT early;")
print("Double Q-learning decouples selection from evaluation and stays near the exploration floor.")

# %% [markdown]
# ## 18.7 Deep reinforcement learning

# %% [markdown]
# ## 18.7.1 Before deep: linear function approximation and semi-gradient TD

# %% [markdown]
# Example 18-14. Semi-gradient TD(0) with state aggregation, and Baird's divergence.

# %%
# Example: linear function approximation: semi-gradient TD(0) and Baird's counterexample
import numpy as np
rng = np.random.default_rng(0)

# --- part 1: semi-gradient TD(0) with state aggregation on a 20-state corridor ---
S, GOAL, gamma, p = 20, 19, 0.9, 0.8
# always-right policy; exact V from the linear Bellman solve of Section 18.4
P = np.zeros((S, S)); R = np.zeros(S)
for s in range(S - 1):
    P[s, s + 1] = p; P[s, s] = 1 - p
    R[s] = p * (10.0 if s + 1 == GOAL else -1.0) + (1 - p) * (-1.0)
V_true = np.linalg.solve(np.eye(S) - gamma * P, R); V_true[GOAL] = 0.0

n_feat = 5                                        # 5 features for 19 nonterminal states
def x(s):                                         # state aggregation: 4 states share one feature
    v = np.zeros(n_feat); v[s * n_feat // S] = 1.0; return v

def sample(s):                                    # one slippery always-right step
    s2 = s + 1 if rng.random() < p else s
    return s2, (10.0 if s2 == GOAL else -1.0)

w = np.zeros(n_feat); alpha = 0.05
for ep in range(2000):
    s = 0
    while s != GOAL:
        s2, r = sample(s)
        v_next = 0.0 if s2 == GOAL else w @ x(s2)
        delta = r + gamma * v_next - w @ x(s)     # TD error under the approximation
        w += alpha * delta * x(s)                 # semi-gradient: grad of the PREDICTION only
        s = s2
print("semi-gradient TD(0), 20-state corridor compressed into 5 features:")
print(f"  learned w                : {np.round(w, 2)}")
for g in range(n_feat):
    states = [s for s in range(S - 1) if s * n_feat // S == g]
    print(f"  group {g} = states {str(states):17s} true V mean {V_true[states].mean():6.2f}   w_{g} {w[g]:6.2f}")
vhat = np.array([w @ x(s) for s in range(S - 1)])
print(f"  RMSE of Vhat over the 19 nonterminal states = {np.sqrt(np.mean((vhat - V_true[:-1]) ** 2)):.2f}")
print("  5 numbers stand in for 19 values: close per group, blurred inside each group.")

# --- part 2: Baird's counterexample: the deadly triad diverging in 7 states ---
n = 7
X = np.zeros((n, 8))                              # feature matrix: MORE weights than states
for i in range(6): X[i, i] = 2.0; X[i, 7] = 1.0   # states 0-5: x = 2*e_i + e_7
X[6, 6] = 1.0; X[6, 7] = 2.0                      # state 6:    x = e_6 + 2*e_7
gamma_b = 0.99
w = np.ones(8); w[6] = 10.0                       # the starting weights of Sutton and Barto (2018), Figure 11.2
alpha_b = 0.01
print("\nBaird's counterexample (off-policy semi-gradient TD, all rewards 0, true V = 0):")
for sweep in range(1, 1001):
    for s in range(n):                            # expected update: target policy always -> state 6
        delta = 0.0 + gamma_b * (X[6] @ w) - X[s] @ w
        w = w + alpha_b * delta * X[s]            # bootstrapped, off-policy, approximated: all three
    if sweep in (1, 10, 100, 500, 1000):
        print(f"  sweep {sweep:4d}: ||w|| = {np.linalg.norm(w):14.1f}")
print("  every reward is 0 and the exact answer is w = 0, yet the weights EXPLODE:")
print("  function approximation + bootstrapping + off-policy = the deadly triad.")

print("\nAll linear-FA checks ran.")

# %% [markdown]
# ## 18.7.2 DQN: Q-learning with a neural network

# %% [markdown]
# A minibatch update has to read $Q(s, a)$ for the action that was taken, not for every action, and `gather` does that. For rows $[1, 2]$, $[3, 4]$, $[5, 6]$ and actions $[1, 0, 1]$ it returns $[2, 3, 6]$, the same entries as indexing with `q[arange, a]`.

# %%
import torch
q = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
a = torch.tensor([1, 0, 1])
picked = q.gather(1, a[:, None]).squeeze(1)
print("gather", picked.tolist())
print("indexing", q[torch.arange(3), a].tolist())

# %% [markdown]
# Example 18-15. A DQN learning CartPole, with a Double DQN flag.

# %%
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import gymnasium as gym

def train(double=False, episodes=300):
    # identical seeding for both runs, so the ONLY difference is the target rule
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    torch.use_deterministic_algorithms(True)    # make CPU matmuls reproducible
    env = gym.make("CartPole-v1")
    env.action_space.seed(0)                    # seed the exploration sampler too, or runs diverge
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
        s, _ = env.reset(seed=ep); done = False; total = 0.0
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

print("CartPole: 4 continuous state dims, 2 actions; return 500 = a whole episode balanced")
for name, dbl in [("DQN       ", False), ("Double DQN", True)]:
    R = train(double=dbl)
    print(f"{name}: mean return first 20 eps = {np.mean(R[:20]):3.0f}   "
          f"last 20 eps = {np.mean(R[-20:]):3.0f}   best = {max(R):3.0f}")
print("the net learned a Q-function over CONTINUOUS states; replay + target net are")
print("what turned unstable Q-learning-with-a-network into the method that cracked Atari from pixels.")

print("\nAll DQN checks ran.")

# %% [markdown]
# Check cell (continues Example 18-15): where the two matched arms part.

# %%
# Check cell (continues Example 18-15): each call of train() reseeds torch, NumPy, random and the
# action sampler, so the two arms are a matched pair until learning moves their networks apart.
n_threads = torch.get_num_threads(); torch.set_num_threads(2)   # pin two threads, the count the notebook build uses
R_dqn = train(double=False, episodes=80)
R_ddqn = train(double=True, episodes=80)
torch.set_num_threads(n_threads)
full = int(np.argmax(np.cumsum(R_dqn) >= 1000))                 # the episode in which the buffer reaches 1000
first_diff = next(i for i in range(80) if R_dqn[i] != R_ddqn[i])
print(f"replay buffer first holds 1000 transitions in episode {full}")
print(f"the two arms' returns first differ in episode {first_diff}")

# %% [markdown]
# ## 18.7.4 Policy gradients: optimize the policy directly

# %% [markdown]
# A `Categorical` built from logits $[2, 1, 0]$ has log probability $2 - \log(e^2 + e^1 + e^0) = -0.407606$ at action 0, the log-softmax at that index, and a `Normal(0, 1)` has log probability $-\tfrac{1}{2}\log(2\pi) = -0.918939$ at 0.

# %%
import torch
from torch.distributions import Categorical, Normal
logits = torch.tensor([2.0, 1.0, 0.0])
dist = Categorical(logits=logits)
print(f"log_prob {dist.log_prob(torch.tensor(0)).item():.6f}")
print(f"log_softmax[0] {torch.log_softmax(logits, dim=0)[0].item():.6f}")
normal = Normal(0.0, 1.0)
print(f"normal log_prob(0) {normal.log_prob(torch.tensor(0.0)).item():.6f}")

# %% [markdown]
# ## 18.7.5 REINFORCE with a learned baseline

# %% [markdown]
# A few lines verify the arithmetic with the backward recursion $A_t = \delta_t + \gamma\lambda\,A_{t+1}$ that implementations actually use.

# %%
import numpy as np

gamma, lam = 0.9, 0.8
V = np.array([2.0, 1.0, 3.0, 0.0])                       # V(s0..s2) and V(terminal)=0
r = np.array([1.0, 2.0, 3.0])                            # r1, r2, r3
delta = r + gamma * V[1:] - V[:-1]                       # TD errors delta_0, delta_1, delta_2
print(f"deltas: {np.round(delta, 2)}")
for l in [0.0, 0.8, 1.0]:
    A = np.zeros(3)
    running = 0.0
    for t in reversed(range(3)):                          # A_t = delta_t + gamma*lambda*A_{t+1}
        running = delta[t] + gamma * l * running
        A[t] = running
    print(f"lambda={l:.1f}: A^GAE = {np.round(A, 3)}")
G0 = r[0] + gamma * r[1] + gamma**2 * r[2]
print(f"check at lambda=1: G_0 - V(s_0) = {G0:.2f} - {V[0]:.2f} = {G0 - V[0]:.2f}")

# %% [markdown]
# Example 18-16. REINFORCE on normalized returns, with and without a learned baseline.

# %%
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

torch.manual_seed(0); np.random.seed(0)
env = gym.make("CartPole-v1")
n_obs, n_act = env.observation_space.shape[0], env.action_space.n
gamma = 0.99

def returns_to_go(rewards):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G; out.append(G)
    return list(reversed(out))

def train(use_baseline, episodes=500, seed=0):
    torch.manual_seed(seed)
    policy = nn.Sequential(nn.Linear(n_obs, 128), nn.ReLU(), nn.Linear(128, n_act))
    value = nn.Sequential(nn.Linear(n_obs, 128), nn.ReLU(), nn.Linear(128, 1))
    p_opt = torch.optim.Adam(policy.parameters(), lr=2e-3)
    v_opt = torch.optim.Adam(value.parameters(), lr=2e-3)
    ep_returns = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 10000 + ep); done = False
        states, actions, rewards, logps = [], [], [], []
        while not done:
            st = torch.tensor(s)
            logits = policy(st); dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            s2, r, term, trunc, _ = env.step(int(a)); done = term or trunc
            states.append(s); actions.append(a); rewards.append(r); logps.append(dist.log_prob(a))
            s = s2
        G = torch.tensor(returns_to_go(rewards))
        G = (G - G.mean()) / (G.std() + 1e-8)               # normalize returns (stabilizes both)
        logps = torch.stack(logps)
        if use_baseline:
            S = torch.tensor(np.array(states))
            V = value(S).squeeze(1)
            advantage = G - V.detach()                      # advantage = return minus baseline
            v_opt.zero_grad(); nn.functional.mse_loss(V, G).backward(); v_opt.step()
        else:
            advantage = G
        p_loss = -(logps * advantage).mean()                # policy-gradient loss
        p_opt.zero_grad(); p_loss.backward(); p_opt.step()
        ep_returns.append(sum(rewards))
    return np.array(ep_returns)

def summarize(name, runs):
    last = np.array([r[-50:].mean() for r in runs])        # final performance per run
    # variance of episode-to-episode return in the second half = learning stability
    instab = np.mean([r[250:].std() for r in runs])
    print(f"  {name:24s}: final mean return {last.mean():5.0f}   "
          f"second-half episode-return std {instab:5.0f} (lower = steadier)")

print("training (3 seeds each, 500 episodes)...")
plain = [train(False, seed=s) for s in range(3)]
base = [train(True, seed=s) for s in range(3)]
print("\nCartPole (max return 500):")
summarize("normalized returns", plain)
summarize("normalized minus V", base)

print("\nsubtracting a learned V(s), which depends only on the state, leaves the expected")
print("gradient unchanged; the two summary lines show its effect on the final return and")
print("the spread. This listing does not bootstrap, so Sutton and Barto do not call it")
print("actor-critic. A2C and A3C become actor-critic by bootstrapping.")

print("\nAll policy-gradient checks ran.")

# %% [markdown]
# ## 18.7.6 From TRPO to PPO: the clipped objective

# %% [markdown]
# Example 18-17. PPO's clipped objective, dissected.

# %%
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

eps = 0.2
ratio = np.linspace(0.0, 2.0, 401)

def ppo_objective(r, A, eps=0.2):
    return np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A)

A_pos = 1.0
obj_pos = ppo_objective(ratio, A_pos, eps)
unclipped_pos = ratio * A_pos
print(f"advantage = +1 (a good action), eps = {eps}:")
print(f"  at ratio 1.0 : objective {ppo_objective(np.array([1.0]), A_pos)[0]:.2f} (no change)")
print(f"  at ratio 1.2 : objective {ppo_objective(np.array([1.2]), A_pos)[0]:.2f} (= 1+eps, the cap kicks in)")
print(f"  at ratio 1.8 : objective {ppo_objective(np.array([1.8]), A_pos)[0]:.2f} (STILL {1+eps:.1f}: this sample")
print(f"                 adds no further reward for a larger ratio)")
print(f"  at ratio 0.5 : objective {ppo_objective(np.array([0.5]), A_pos)[0]:.2f} (the DISFAVORED side is NOT clipped:")
print(f"                 min picks r*A, so making a good action rarer keeps its full penalty)")

A_neg = -1.0
print(f"\nadvantage = -1 (a bad action), eps = {eps}:")
print(f"  at ratio 1.0 : objective {ppo_objective(np.array([1.0]), A_neg)[0]:.2f}")
print(f"  at ratio 0.8 : objective {ppo_objective(np.array([0.8]), A_neg)[0]:.2f} (= -(1-eps), pushing the")
print(f"                 probability down is rewarded only down to the clip, then it flattens)")

print("the objective is flat for that sample once its ratio passes the band in the")
print("direction the advantage favors, so that sample's own ratio has zero gradient there.")
print("shared weights and later passes over the batch can still move a ratio past the band.")
print("moves that make the objective worse stay unclipped and keep their gradient.")
print("PPO is the simple clipped stand-in for TRPO, and Section 18.10 uses it for RLHF.")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ratio, unclipped_pos, "--", label="unclipped r*A")
ax[0].plot(ratio, obj_pos, label="PPO clipped")
ax[0].axvline(1 - eps, color="gray", ls=":"); ax[0].axvline(1 + eps, color="gray", ls=":")
ax[0].set_title("advantage > 0"); ax[0].set_xlabel("ratio r"); ax[0].legend()
ax[1].plot(ratio, ratio * A_neg, "--", label="unclipped r*A")
ax[1].plot(ratio, ppo_objective(ratio, A_neg, eps), label="PPO clipped")
ax[1].axvline(1 - eps, color="gray", ls=":"); ax[1].axvline(1 + eps, color="gray", ls=":")
ax[1].set_title("advantage < 0"); ax[1].set_xlabel("ratio r"); ax[1].legend()
fig.tight_layout(); fig.savefig("ppo_clip.png", dpi=110)      # written to the current folder
print("\nsaved figure: ppo_clip.png")
print("\nAll PPO-clip checks ran.")

# %% [markdown]
# Example 18-18. TRPO and PPO on CartPole-v1 with the same seed, networks, and step budget.

# %%
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import TRPO

torch.manual_seed(0); torch.set_num_threads(2)
for name, algo in [("PPO", PPO), ("TRPO", TRPO)]:
    model = algo("MlpPolicy", "CartPole-v1", verbose=0, seed=0)   # same seed, same default networks
    model.learn(total_timesteps=30000)
    mean_r, std_r = evaluate_policy(model, model.get_env(), n_eval_episodes=20)
    print(f"{name:4s}: networks {model.policy.net_arch}, mean return {mean_r:.0f} +/- {std_r:.0f} "
          f"over 20 episodes")

# %% [markdown]
# ## 18.7.7 Continuous control: DDPG, TD3, and SAC

# %% [markdown]
# A uniform random torque on the ten evaluation starts of Example 18-19 sets the floor to beat.

# %%
import numpy as np
import gymnasium as gym

env = gym.make("Pendulum-v1")
env.reset(seed=123); env.action_space.seed(0)        # the evaluation seed of Example 18-19
returns = []
for _ in range(10):                                   # the next ten starts, as evaluate_policy draws them
    obs, _ = env.reset(); done, G = False, 0.0
    while not done:
        obs, r, term, trunc, _ = env.step(env.action_space.sample()); G += r; done = term or trunc
    returns.append(G)
print(f"uniform random torque: mean return {np.mean(returns):.0f} +/- {np.std(returns):.0f} over 10 episodes")

# %% [markdown]
# Example 18-19. SAC learning continuous control on Pendulum-v1.

# %%
import numpy as np, torch, random
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

np.random.seed(0); torch.manual_seed(0); random.seed(0)

env = gym.make("Pendulum-v1")            # continuous torque in [-2, 2]
print(f"Pendulum-v1: obs {env.observation_space.shape}, action {env.action_space.shape} "
      f"(CONTINUOUS in [{env.action_space.low[0]:.0f}, {env.action_space.high[0]:.0f}])")

eval_env = Monitor(gym.make("Pendulum-v1"))   # Monitor records each episode's true return
model = SAC("MlpPolicy", env, verbose=0, seed=0, learning_starts=1000)

eval_env.reset(seed=123)
before, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"mean return, untrained policy = {before:.0f}")

model.learn(total_timesteps=7000, progress_bar=False)

eval_env.reset(seed=123)
after, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"mean return, after 7k steps  = {after:.0f}")

# %% [markdown]
# Example 18-20. DDPG and TD3 on Pendulum-v1 under the protocol of Example 18-19.

# %%
import numpy as np, torch, random
import gymnasium as gym
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

np.random.seed(0); torch.manual_seed(0); random.seed(0); torch.set_num_threads(2)
for name, algo in [("DDPG", DDPG), ("TD3", TD3)]:
    env = gym.make("Pendulum-v1")
    eval_env = Monitor(gym.make("Pendulum-v1"))
    noise = NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1))   # sd 0.1 on the action scaled to [-1, 1]
    model = algo("MlpPolicy", env, action_noise=noise, verbose=0, seed=0, learning_starts=1000)
    eval_env.reset(seed=123)
    before, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    model.learn(total_timesteps=7000, progress_bar=False)
    eval_env.reset(seed=123)
    after, sd = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"{name:4s}: untrained {before:.0f}, after 7k steps {after:.0f} +/- {sd:.0f}, "
          f"networks {model.policy.net_arch}")

# %% [markdown]
# ## 18.8 When learning meets search: MCTS and AlphaGo

# %% [markdown]
# ## 18.8.1 Monte Carlo Tree Search

# %% [markdown]
# Example 18-21. Monte Carlo Tree Search on tic-tac-toe.

# %%
import numpy as np
import math
rng = np.random.default_rng(0)

WINS = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
def winner(b):
    for x, y, z in WINS:
        if b[x] and b[x] == b[y] == b[z]: return b[x]
    return 0
def legal(b): return [i for i in range(9) if b[i] == 0]
def terminal(b): return winner(b) != 0 or not legal(b)

class Node:
    def __init__(self, board, player, parent=None, move=None):
        self.board, self.player, self.parent, self.move = board, player, parent, move
        self.children = {}
        self.untried = [] if terminal(board) else legal(board)   # terminal boards are LEAVES
        self.N = 0; self.W = 0.0                      # visits, total value (for player who moved here)

def uct(node, c=1.4):
    # pick the child maximizing value + exploration bonus (UCB applied to the tree)
    return max(node.children.values(),
               key=lambda ch: ch.W / ch.N + c * math.sqrt(math.log(node.N) / ch.N))

def rollout(board, player):
    b = list(board)
    while not terminal(b):
        b[rng.choice(legal(b))] = player
        player = 3 - player
    return winner(b)

def mcts(board, player, n_sims):
    root = Node(board, player)
    for _ in range(n_sims):
        node = root
        # 1. SELECT down fully expanded nodes
        while not node.untried and node.children:
            node = uct(node)
        # 2. EXPAND one untried move
        if node.untried:
            m = node.untried.pop(rng.integers(len(node.untried)))
            nb = list(node.board); nb[m] = node.player
            child = Node(tuple(nb), 3 - node.player, node, m)
            node.children[m] = child; node = child
        # 3. SIMULATE
        result = rollout(node.board, node.player)
        # 4. BACKPROPAGATE (value seen from the player who MOVED into each node)
        while node is not None:
            node.N += 1
            mover = 3 - node.player                   # who just moved to reach this node
            node.W += 1.0 if result == mover else (0.5 if result == 0 else 0.0)
            node = node.parent
    return max(root.children.items(), key=lambda kv: kv[1].N)[0]   # most-visited move

# X to move, X has 0 and 1, square 2 completes the top row -> must play 2
win_now = (1, 1, 0,  2, 2, 0,  0, 0, 0)
print(f"win-in-one: MCTS plays square {mcts(win_now, 1, 800)} (2 completes the top row and wins)")
# X to move; O threatens 3,4,_ -> O would win at 5; X must block at 5
must_block = (1, 0, 0,  2, 2, 0,  0, 0, 1)
print(f"must-block: MCTS plays square {mcts(must_block, 1, 800)} (5 blocks O's middle-row win)")

def play(sims_X, random_O=True, sims_O=0):
    b = [0] * 9; p = 1
    while not terminal(b):
        if p == 1: m = mcts(tuple(b), 1, sims_X)
        else: m = rng.choice(legal(b)) if random_O else mcts(tuple(b), 2, sims_O)
        b[m] = p; p = 3 - p
    return winner(b)

for sims in [10, 100]:
    res = [play(sims) for _ in range(200)]
    w = res.count(1); d = res.count(0); l = res.count(2)
    print(f"\nMCTS({sims:3d} sims) as X vs RANDOM O over 200 games: wins {w}, draws {d}, losses {l}")

# two strong MCTS players, 1500 simulations per move each: in this run all 50 games are draws
res = [play(1500, random_O=False, sims_O=1500) for _ in range(50)]
print(f"\nMCTS(1500) vs MCTS(1500) over 50 games: draws {res.count(0)}, X-wins {res.count(1)}, O-wins {res.count(2)}")
print("(optimal tic-tac-toe is a draw, so two strong searchers almost always draw)")

print("\nAlphaGo mixed a value network with a fast rollout, and the selection")
print("priors came from a policy network trained on human games (Sutton and Barto, 2018).")
print("AlphaGo Zero dropped the human games. AlphaZero is the later general algorithm.")
print("MuZero learned the model itself.")
print("\nAll MCTS checks ran.")

# %% [markdown]
# ## 18.8.3 Model-based reinforcement learning: Dyna and world models

# %% [markdown]
# Example 18-22. Dyna-Q: Q-learning plus planning updates from a learned model.

# %%
import numpy as np

# deterministic 6x6 grid, start 0, goal 35; every move costs -1 and the episode ends on
# reaching the goal, so maximizing return means the fewest steps. The shortest path is 10
# (5 down + 5 right), so the best possible return is -10 (a known ground truth).
SIZE, START, GOAL = 6, 0, 35
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
def step(s, a):
    r, c = divmod(s, SIZE); dr, dc = ACTIONS[a]
    r2 = min(max(r + dr, 0), SIZE - 1); c2 = min(max(c + dc, 0), SIZE - 1)
    s2 = r2 * SIZE + c2
    return (s2, -1.0, s2 == GOAL)

def greedy(Q, s, rg):                            # argmax with random tie-break (no directional bias)
    q = Q[s]; best = np.flatnonzero(q == q.max())
    return int(rg.choice(best))

def dyna_q(n_plan, episodes=40, alpha=0.5, gamma=0.95, eps=0.1, seed=0):
    rg = np.random.default_rng(seed)
    Q = np.zeros((SIZE * SIZE, 4))               # zero init is OPTIMISTIC here (true values are negative)
    model = {}                                   # learned model: (s,a) -> (r, s2)
    steps_per_ep = []
    for _ in range(episodes):
        s = START; steps = 0
        while True:
            a = int(rg.integers(4)) if rg.random() < eps else greedy(Q, s, rg)
            s2, r, done = step(s, a); steps += 1
            Q[s, a] += alpha * (r + gamma * (0 if done else np.max(Q[s2])) - Q[s, a])
            model[(s, a)] = (r, s2)              # remember what happened (learn the model)
            keys = list(model.keys())
            for _ in range(n_plan):              # planning: replay imagined transitions
                ps, pa = keys[rg.integers(len(keys))]
                pr, ps2 = model[(ps, pa)]; pd = ps2 == GOAL
                Q[ps, pa] += alpha * (pr + gamma * (0 if pd else np.max(Q[ps2])) - Q[ps, pa])
            s = s2
            if done or steps > 2000: break
        steps_per_ep.append(steps)
    return steps_per_ep

print("Dyna-Q on a deterministic 6x6 grid (known best path = 10 steps to the goal):")
for n_plan in [0, 5, 50]:
    spe = dyna_q(n_plan)
    tag = "no planning = plain Q-learning" if n_plan == 0 else f"{n_plan} planning updates / real step"
    print(f"  n_plan={n_plan:2d} ({tag}):")
    print(f"      steps to reach goal, episodes 2-6: {spe[1:6]}")
    print(f"      average steps over episodes 6-40 : {np.mean(spe[5:]):.1f}")
print("more planning per real step converges to the 10-step optimum in far fewer real")
print("episodes: the learned model lets the agent improve without extra environment interaction.")

# %% [markdown]
# ## 18.9 Beyond the basics: imitation learning, offline RL, multi-agent RL, sim-to-real

# %% [markdown]
# ## 18.9.1 Imitation learning and behavioral cloning

# %% [markdown]
# Example 18-23. Behavioral cloning and its distribution-shift failure.

# %%
import numpy as np
from sklearn.neural_network import MLPClassifier
rng = np.random.default_rng(0)

SIZE, GOAL, gamma = 6, 35, 0.95
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
def step(s, a):
    if s == GOAL: return s, 0.0
    r, c = divmod(s, SIZE); dr, dc = ACTIONS[a]
    r2, c2 = min(max(r + dr, 0), SIZE - 1), min(max(c + dc, 0), SIZE - 1)
    s2 = r2 * SIZE + c2
    return s2, -1.0
V = np.zeros(SIZE * SIZE)
for _ in range(300):
    for s in range(SIZE * SIZE):
        if s == GOAL: continue
        V[s] = max((lambda s2, r: r + gamma * V[s2])(*step(s, a)) for a in ACTIONS)
expert = {s: int(np.argmax([(lambda s2, r: r + gamma * V[s2])(*step(s, a)) for a in ACTIONS]))
          for s in range(SIZE * SIZE)}

def feat(s):                                            # (row, col) features for the classifier
    r, c = divmod(s, SIZE); return [r / SIZE, c / SIZE]

def reaches_goal(policy, start, max_steps=60):
    s = start
    for _ in range(max_steps):
        s, _ = step(s, policy[s])
        if s == GOAL: return True
    return False

demo_starts = [0, 1, 6, 7]                               # top-left corner region only
X, y, seen = [], [], set()
for st in demo_starts:
    s = st
    for _ in range(60):
        X.append(feat(s)); y.append(expert[s]); seen.add(s)
        if s == GOAL: break
        s, _ = step(s, expert[s])
print(f"behavioral-cloning data: {len(X)} (state,action) pairs covering {len(seen)} of "
      f"{SIZE*SIZE} states (the rest are unseen)")

clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=0).fit(X, y)
clone = {s: int(clf.predict([feat(s)])[0]) for s in range(SIZE * SIZE)}

seen_acc = np.mean([clone[s] == expert[s] for s in seen])
unseen = [s for s in range(SIZE * SIZE) if s not in seen and s != GOAL]
unseen_acc = np.mean([clone[s] == expert[s] for s in unseen])
print(f"\naction-match accuracy vs expert:")
print(f"  on SEEN states   : {seen_acc:.2f}  (the clone copies the demonstrations well)")
print(f"  on UNSEEN states : {unseen_acc:.2f}  (it errs more often off the demonstrated paths)")

all_starts = [s for s in range(SIZE * SIZE) if s != GOAL]
expert_reach = np.mean([reaches_goal(expert, s) for s in all_starts])
clone_reach = np.mean([reaches_goal(clone, s) for s in all_starts])
print(f"\nfraction of all {len(all_starts)} start states from which the goal is reached:")
print(f"  expert : {expert_reach:.2f}")
print(f"  clone  : {clone_reach:.2f}  (one wrong move leads to unseen states -> errors compound)")
print("this is the core BC failure. Fixes: DAgger (let the expert label the clone's OWN")
print("visited states), or inverse RL (recover the reward the expert optimizes).")

print("\nAll imitation-learning checks ran.")

# %% [markdown]
# Example 18-24. DAgger against more expert demonstrations, continuing Example 18-23.

# %%
# DAgger (Ross, Gordon, and Bagnell, 2011), continuing Example 18-23. Each round runs the
# CURRENT clone from three sampled starts, the expert labels every state the clone visits,
# the new pairs join all earlier ones, and the classifier is refit on the whole set. For
# comparison, the same starts also feed the simpler fix: three more expert demonstrations.
X_agg, y_agg, labeled = list(X), list(y), set(seen)       # round 0 is the original demonstrations
X_dem, y_dem = list(X), list(y)
print("goal-reaching rate from all 35 starts after each round:")
for rnd in range(1, 4):
    visited = set()
    for st in rng.choice(all_starts, size=3, replace=False):
        s = int(st)
        for _ in range(60):                               # DAgger: follow the clone, mistakes and all
            if s == GOAL: break
            visited.add(s); s, _ = step(s, clone[s])
        s = int(st)
        for _ in range(60):                               # comparison: follow the expert instead
            if s == GOAL: break
            X_dem.append(feat(s)); y_dem.append(expert[s]); s, _ = step(s, expert[s])
    for s in sorted(visited):
        X_agg.append(feat(s)); y_agg.append(expert[s])    # the expert labels the clone's own states
    labeled |= visited
    clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=0).fit(X_agg, y_agg)
    clone = {s: int(clf.predict([feat(s)])[0]) for s in range(SIZE * SIZE)}
    reach = np.mean([reaches_goal(clone, s) for s in all_starts])
    clf_d = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=0).fit(X_dem, y_dem)
    clone_d = {s: int(clf_d.predict([feat(s)])[0]) for s in range(SIZE * SIZE)}
    reach_d = np.mean([reaches_goal(clone_d, s) for s in all_starts])
    print(f"  round {rnd}: DAgger {reach:.2f} ({len(X_agg)} labeled pairs on {len(labeled)} of "
          f"{SIZE*SIZE} states) | more expert demonstrations {reach_d:.2f}")

# %% [markdown]
# Example 18-25. Ng and Russell's optimality test for three candidate rewards.

# %%
# the inverse RL test of Ng and Russell (2000, Theorem 3), on Example 18-23's grid and expert:
# a state reward vector R makes the expert's moves optimal exactly when, for every state s and
# every other action a, row s of (P_expert - P_a)(I - gamma P_expert)^(-1) R is at least 0
n = SIZE * SIZE
P_exp = np.zeros((n, n))
for s in range(n):
    P_exp[s, step(s, expert[s])[0]] = 1.0                 # the expert's move (the goal absorbs)

def margins(R):
    V = np.linalg.solve(np.eye(n) - gamma * P_exp, R)     # V = (I - gamma P_expert)^(-1) R
    out = {}
    for s in range(n):
        if s == GOAL: continue
        for a in ACTIONS:
            if a != expert[s]:                            # one entry of (P_expert - P_a) V
                out[(s, a)] = V[step(s, expert[s])[0]] - V[step(s, a)[0]]
    return out

R_true = np.full(n, -1.0); R_true[GOAL] = 0.0            # -1 in every state but the goal
for name, R in [("-1 per step (the true reward)", R_true), ("0 everywhere", np.zeros(n)),
                ("+1 per step", -R_true)]:
    m = margins(R); worst = min(m.values())
    print(f"{name:30s}: margin at state 34 for up {m[(34, 0)]:+.2f}; smallest margin "
          f"{worst:+.2f} -> expert {'optimal' if worst >= -1e-9 else 'NOT optimal'}")

# %% [markdown]
# ## 18.9.2 Offline reinforcement learning

# %% [markdown]
# Example 18-26. Offline RL: phantom values from unlogged actions, and the conservative fix.

# %%
import numpy as np
rng = np.random.default_rng(0)

# a 6-state corridor, gamma=0.95: step -1, goal +2 at state 5, and a third action
# LEAP that ends the episode at once with reward -8 (a cliff). The logged behavior
# policy is a slightly noisy expert that goes right and NEVER leaps.
S, GOAL, gamma = 6, 5, 0.95
LEFT, RIGHT, LEAP = 0, 1, 2
def step(s, a):
    if a == LEAP: return None, -8.0                    # episode over, catastrophic
    s2 = max(0, s - 1) if a == LEFT else min(GOAL, s + 1)
    r = 2.0 if s2 == GOAL else -1.0
    return (None if s2 == GOAL else s2), r

data = []                                              # the FIXED logged dataset
for _ in range(200):
    s = 0
    while s is not None:
        a = RIGHT if rng.random() < 0.9 else int(rng.integers(2))   # never LEAP
        s2, r = step(s, a); data.append((s, a, r, s2)); s = s2
Ncount = np.zeros((S, 3), int)
for s, a, r, s2 in data: Ncount[s, a] += 1
print(f"logged transitions: {len(data)};  dataset counts N(s, a) per nonterminal state:")
print("  state:            0    1    2    3    4")
print("  left counts  :", Ncount[:GOAL, LEFT])
print("  right counts :", Ncount[:GOAL, RIGHT])
print("  leap counts  :", Ncount[:GOAL, LEAP], " <- the logged policy NEVER leaps")

def fit_q(support_only):
    Q = np.zeros((S, 3)); alpha = 0.2
    for sweep in range(60):                            # many passes over the frozen log
        for idx in rng.permutation(len(data)):
            s, a, r, s2 = data[idx]
            if s2 is None: nxt = 0.0
            elif support_only:
                nxt = Q[s2][Ncount[s2] > 0].max()      # bootstrap only through logged actions
            else:
                nxt = Q[s2].max()                      # naive: the max can pick a phantom action
            Q[s, a] += alpha * (r + gamma * nxt - Q[s, a])
    return Q

Q_naive = fit_q(support_only=False)
Q_cons = fit_q(support_only=True)
beta = 2.0                                             # pessimism: penalize rarely seen pairs
Q_pess = np.where(Ncount > 0, Q_cons - beta / np.sqrt(np.maximum(Ncount, 1)), -np.inf)

names = ["left", "right", "leap"]
print("\nnaive offline Q-learning, Q values (rows = states 0-4; cols = left, right, leap):")
print(np.round(Q_naive[:GOAL], 2))
print("greedy actions:", [names[a] for a in Q_naive[:GOAL].argmax(1)])
print("\nconservative offline Q (in-support backup), then count-penalized extraction:")
print(np.round(Q_cons[:GOAL], 2))
print("greedy actions:", [names[a] for a in Q_pess[:GOAL].argmax(1)])

def ret(policy):                                       # deploy: discounted return from state 0
    s, total, disc = 0, 0.0, 1.0
    for _ in range(50):
        s2, r = step(s, policy(s)); total += disc * r; disc *= gamma
        if s2 is None: break
        s = s2
    return total
print(f"\ndeployed return from state 0: naive {ret(lambda s: int(Q_naive[s].argmax())):.2f}"
      f" | conservative {ret(lambda s: int(Q_pess[s].argmax())):.2f}"
      f" | expert {ret(lambda s: RIGHT):.2f}")
print("the naive learner chases the PHANTOM value of the never-logged leap (Q = 0 beats the")
print("true negative values) and crashes at deployment; staying inside the data fixes it.")

# %% [markdown]
# Example 18-27. The CQL(H) penalty and a CQL-trained table, continuing Example 18-26.

# %%
# the CQL(H) regularizer of Kumar et al. (2020), continuing Example 18-26: the mean over the
# logged pairs of log sum_a exp Q(s, a) minus Q(s, a) at the logged action
log_s = np.array([s for s, a, r, s2 in data]); log_a = np.array([a for s, a, r, s2 in data])
def cql_penalty(Q):
    rows = Q[log_s]; top = rows.max(axis=1)
    lse = top + np.log(np.exp(rows - top[:, None]).sum(axis=1))   # a numerically stable log-sum-exp
    return float((lse - Q[log_s, log_a]).mean())
print(f"CQL penalty: naive table {cql_penalty(Q_naive):.3f}, in-support table {cql_penalty(Q_cons):.3f}")

def fit_cql(alpha_cql=1.0, lr=0.2, sweeps=60):
    Q = np.zeros((S, 3))
    for sweep in range(sweeps):
        for idx in rng.permutation(len(data)):
            s, a, r, s2 = data[idx]
            nxt = 0.0 if s2 is None else Q[s2].max()             # the ordinary max: nothing excluded
            Q[s, a] += lr * (r + gamma * nxt - Q[s, a])           # the TD step
            soft = np.exp(Q[s] - Q[s].max()); soft /= soft.sum()  # gradient of log-sum-exp = softmax
            Q[s] -= lr * alpha_cql * soft                         # every action pushed down by its weight
            Q[s, a] += lr * alpha_cql                             # and the logged action pushed back up
    return Q

Q_cql = fit_cql()
print("CQL-trained Q (rows = states 0-4; cols = left, right, leap):")
print(np.round(Q_cql[:GOAL], 2))
print("greedy actions:", [names[a] for a in Q_cql[:GOAL].argmax(1)])
print(f"CQL penalty after training {cql_penalty(Q_cql):.3f}; deployed return from state 0 "
      f"{ret(lambda s: int(Q_cql[s].argmax())):.2f}")

# %% [markdown]
# Example 18-28. CQL weights from 0 to 0.01, ten reseeded fits each, continuing Example 18-27.

# %%
# CQL weights from 0 to 0.01, continuing Example 18-27: ten fits per weight, each after
# reseeding the generator that fit_cql uses to order its passes over the log
for w in [0.0, 0.001, 0.003, 0.01]:
    leap_fits, leap_states, returns = 0, set(), set()
    for k in range(10):
        rng = np.random.default_rng(1000 + k)                 # fit_cql shuffles the log with rng
        Q_w = fit_cql(alpha_cql=w)
        leaping = np.flatnonzero(Q_w[:GOAL].argmax(1) == LEAP)
        leap_fits += int(leaping.size > 0); leap_states |= set(leaping.tolist())
        returns.add(round(ret(lambda s: int(Q_w[s].argmax())), 2))
    shown = ", ".join(f"{x:.2f}" for x in sorted(returns))
    print(f"alpha_cql={w:<5}: leaps in {leap_fits:2d}/10 fits, from states {sorted(leap_states)}; "
          f"deployed return {shown}")

# %% [markdown]
# ## 18.9.3 Multi-agent reinforcement learning

# %% [markdown]
# Example 18-29. Minimax policies by linear programming, and two independent learners on matching pennies.

# %%
import numpy as np
from scipy.optimize import linprog

# matching pennies, a one-state two-player zero-sum Markov game (a constructed known game).
# Each player shows heads (0) or tails (1). The matcher wins +1 when the coins agree and
# -1 when they differ; the opponent receives the negative. Rows = matcher, columns = opponent.
R = np.array([[1.0, -1.0], [-1.0, 1.0]])

def minimax(R):
    # maximize v subject to sum_a pi(a) R[a, o] >= v for every opponent action o (Littman, 1994)
    n_a, n_o = R.shape
    c = np.zeros(n_a + 1); c[-1] = -1.0                      # linprog minimizes, so minimize -v
    A_ub = np.hstack([-R.T, np.ones((n_o, 1))])              # v - sum_a pi(a) R[a, o] <= 0
    A_eq = np.ones((1, n_a + 1)); A_eq[0, -1] = 0.0          # the probabilities sum to one
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n_o), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, 1)] * n_a + [(None, None)])
    return res.x[:n_a], round(res.x[-1], 6) + 0.0             # + 0.0 turns -0.0 into 0.0

pi, v = minimax(R)
print(f"matching pennies: minimax policy {np.round(pi, 3)}, game value {v:.3f}")
rps = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])   # rock, paper, scissors
pi_rps, v_rps = minimax(rps)
print(f"rock, paper, scissors: minimax policy {np.round(pi_rps, 3)}, game value {v_rps:.3f}")

# two independent epsilon-greedy learners, each treating the other as part of its environment
rng = np.random.default_rng(0)
Q1, Q2 = np.zeros(2), np.zeros(2); alpha, eps = 0.1, 0.1
def eps_greedy_probs(Q):                                     # the policy a learner is following now
    p = np.full(2, eps / 2); p[int(np.argmax(Q))] += 1 - eps; return p
r_heads, total, flips = [], 0.0, 0
for t in range(20000):
    a1 = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(Q1))
    a2 = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(Q2))
    r1 = R[a1, a2]; total += r1
    Q1[a1] += alpha * (r1 - Q1[a1])                          # one state, so the target is the reward
    old = int(np.argmax(Q2))
    Q2[a2] += alpha * (-r1 - Q2[a2])
    flips += int(np.argmax(Q2) != old)
    if t < 200 and t % 10 == 9:                              # the matcher's reward for heads right now
        r_heads.append(R[0] @ eps_greedy_probs(Q2))
print("\nindependent learners: the matcher's expected reward for heads under the opponent's")
print("current policy, every 10 steps over the first 200:")
print("  " + " ".join(f"{x:+.1f}" for x in r_heads))
print(f"the opponent's greedy action changed {flips} times in 20000 steps; "
      f"the matcher's mean reward per step was {total / 20000:+.3f}")

# %% [markdown]
# ## 18.9.4 Sim-to-real transfer

# %% [markdown]
# Example 18-30. Domain randomization on the leaping corridor, by exhaustive search over policies.

# %%
import numpy as np
from itertools import product

# a 6-state corridor simulator (a constructed world with a known model), gamma 0.95. RIGHT
# costs -1 and reaching the goal at state 5 pays +2. LEAP jumps two squares with probability q
# and otherwise falls: reward -8, episode over. q is the physical parameter the simulator may get wrong.
S, GOAL, gamma = 6, 5, 0.95
RIGHT, LEAP = 1, 2

def value(policy, q):
    # exact discounted return from state 0 for a deterministic policy, q held fixed for the episode
    P = np.zeros((GOAL, GOAL)); R = np.zeros(GOAL)
    for s, a in enumerate(policy):
        s2 = min(GOAL, s + (2 if a == LEAP else 1))
        p_ok = q if a == LEAP else 1.0
        R[s] = p_ok * (2.0 if s2 == GOAL else -1.0) + (1 - p_ok) * (-8.0)   # expected reward at s
        if s2 < GOAL: P[s, s2] = p_ok
    return np.linalg.solve(np.eye(GOAL) - gamma * P, R)[0]            # V = (I - gamma P)^(-1) R

def route(policy):                                       # the moves from state 0 when every leap lands
    s, moves = 0, []
    while s < GOAL:
        a = policy[s]; moves.append("leap" if a == LEAP else "right"); s += 2 if a == LEAP else 1
    return " ".join(moves)

policies = list(product([RIGHT, LEAP], repeat=GOAL))       # every right-or-leap choice for states 0 to 4
q_draws = np.random.default_rng(0).uniform(0.5, 1.0, 200)  # randomized simulators, one q per episode
best_fixed = max(policies, key=lambda pi: value(pi, 0.95))             # trained on q = 0.95 only
best_rand = max(policies, key=lambda pi: np.mean([value(pi, q) for q in q_draws]))
print("route from state 0, fixed simulator (q = 0.95):", route(best_fixed))
print("route from state 0, randomized q in [0.5, 1.0]:", route(best_rand))
print("\nreal leap probability:  " + "  ".join(f"{q:5.2f}" for q in [1.0, 0.95, 0.9, 0.8, 0.7, 0.6]))
for name, pi in [("fixed-simulator policy", best_fixed), ("randomized policy     ", best_rand)]:
    print(f"{name}: " + "  ".join(f"{value(pi, q):5.2f}" for q in [1.0, 0.95, 0.9, 0.8, 0.7, 0.6]))

# %% [markdown]
# ## 18.10 RLHF: how reinforcement learning aligns language models

# %% [markdown]
# ## 18.10.4 GRPO: dropping the critic

# %% [markdown]
# Example 18-31. A reward model, the KL leash, and GRPO advantages.

# %%
import numpy as np
rng = np.random.default_rng(0)
def sigmoid(z): return 1 / (1 + np.exp(-z))

N = 8
true_quality = rng.normal(0, 1, N)
pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
prefs = []                                               # (winner, loser) from simulated comparisons
for _ in range(4000):
    i, j = pairs[rng.integers(len(pairs))]
    win = i if rng.random() < sigmoid(true_quality[i] - true_quality[j]) else j
    prefs.append((win, i if win == j else j))

r = np.zeros(N)                                          # reward-model scores to learn
lr = 0.1
for _ in range(3000):                                    # gradient ascent on log likelihood
    grad = np.zeros(N)
    for w, l in prefs:
        p = sigmoid(r[w] - r[l])
        grad[w] += (1 - p); grad[l] -= (1 - p)
    r += lr * grad / len(prefs)
r -= r.mean()                                            # scores are relative; center them
corr = np.corrcoef(r, true_quality)[0, 1]
agree = np.mean([np.sign(r[i] - r[j]) == np.sign(true_quality[i] - true_quality[j]) for i, j in pairs])
print(f"reward model vs true quality: correlation {corr:.3f}")
print(f"  it ranks {agree*100:.0f}% of all item pairs in the same order humans would (learned from comparisons alone)")

responses = ["helpful", "ok", "rambling", "toxic"]       # whole responses, each one action
reward = np.array([2.0, 1.0, 0.0, -1.0])                 # reward model's score per response type
pi_ref = np.array([0.4, 0.3, 0.2, 0.1])                  # SFT model's tendencies
print("\noptimal RLHF policy under a KL penalty (reward: helpful 2, ok 1, rambling 0, toxic -1):")
for beta in [5.0, 1.0, 0.2]:
    pi = pi_ref * np.exp(reward / beta); pi /= pi.sum()
    print(f"  beta={beta:.1f}: " + "  ".join(f"{y} {p:.2f}" for y, p in zip(responses, pi)))
print("large beta -> close to the reference (safe); small beta -> piles onto 'helpful' and risks")
print("REWARD HACKING (exploiting the imperfect reward model). The KL term is the safety leash.")

group_rewards = np.array([0.8, 0.2, 0.9, 0.1, 0.5])      # rewards for 5 sampled answers
adv = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)
print("\nGRPO group-relative advantage (5 sampled answers to one prompt):")
for k, (rw, a) in enumerate(zip(group_rewards, adv)):
    label = "neutral" if abs(a) < 0.05 else ("reinforce" if a > 0 else "suppress")
    print(f"  answer {k}: reward {rw:.1f} -> advantage {a:+.2f} ({label})")
print("PPO needs a learned VALUE network as the baseline; GRPO just uses the group mean,")
print("which is cheaper and powers reasoning models like DeepSeek-R1. This is the bridge to Part V.")

print("\nAll RLHF checks ran.")

# %% [markdown]
# ## 18.10.5 DPO: aligning straight from preferences

# %% [markdown]
# Example 18-32. DPO: fitting the policy directly on preference pairs.

# %%
import numpy as np
rng = np.random.default_rng(0)
def sigmoid(z): return 1 / (1 + np.exp(-z))

# N responses with hidden true quality, and noisy pairwise human preferences drawn
# from the Bradley-Terry model sigma(q_i - q_j) (identical setup to the RLHF example).
N = 8
true_quality = rng.normal(0, 1, N)
pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
prefs = []                                               # (winner, loser)
for _ in range(4000):
    i, j = pairs[rng.integers(len(pairs))]
    win = i if rng.random() < sigmoid(true_quality[i] - true_quality[j]) else j
    prefs.append((win, i if win == j else j))

# DPO: NO separate reward model and NO RL loop. Each response carries one parameter
# delta_i = log pi_theta(i) - log pi_ref(i), the policy's log-ratio against the frozen
# reference. The IMPLICIT reward is r_i = beta * delta_i. The DPO loss on a pair (w, l)
# is  -log sigma( beta * (delta_w - delta_l) ), a plain logistic classification of which
# answer is preferred, optimized directly on the policy.
beta = 1.0
delta = np.zeros(N)                                      # policy starts equal to the reference
lr = 0.5
for _ in range(3000):
    grad = np.zeros(N)
    for w, l in prefs:
        p = sigmoid(beta * (delta[w] - delta[l]))         # model's prob that w beats l
        g = beta * (1 - p)                                # d/d(delta) of -log sigma(...)
        grad[w] += g; grad[l] -= g
    delta += lr * grad / len(prefs)
r = beta * delta                                         # implicit reward from the policy itself
r -= r.mean()
corr = np.corrcoef(r, true_quality)[0, 1]
agree = np.mean([np.sign(r[i] - r[j]) == np.sign(true_quality[i] - true_quality[j]) for i, j in pairs])
print("DPO fits the policy DIRECTLY on preference pairs (no reward model, no PPO loop):")
print(f"  implicit reward r = beta*(log pi_theta - log pi_ref) vs true quality: correlation {corr:.3f}")
print(f"  it ranks {agree*100:.0f}% of all response pairs the same way as the hidden quality")
print("same preference signal as PPO-based RLHF, but one classification objective instead of")
print("a reward model plus an RL stage; GRPO keeps the RL loop but drops only the value network.")

# %% [markdown]
# ## 18.10.6 The same DPO on a real language model, with TRL

# %% [markdown]
# Example 18-33. DPO on distilbert/distilgpt2 with TRL, on real human preference data.

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

torch.manual_seed(0); np.random.seed(0)
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")     # the policy pi_theta to align
ref = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")       # held-out check only; not passed as ref_model
tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
tok.pad_token = tok.eos_token

# real human preference pairs: Anthropic's HH-RLHF helpfulness comparisons
train_prefs = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train[:200]")
eval_prefs = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train[200:250]")
print(f"train pairs: {len(train_prefs)}, held-out pairs: {len(eval_prefs)}; columns {train_prefs.column_names}")

# each text is the whole conversation; the two differ only in the last assistant reply, so cut
# both at their last "\n\nAssistant:" into one shared prompt and two replies
TAG = "\n\nAssistant:"
def split_turn(ex):
    i = ex["chosen"].rfind(TAG) + len(TAG); j = ex["rejected"].rfind(TAG) + len(TAG)
    return {"prompt": ex["chosen"][:i], "chosen": ex["chosen"][i:], "rejected": ex["rejected"][j:],
            "shared": ex["chosen"][:i] == ex["rejected"][:j]}
train_split = train_prefs.map(split_turn)
print(f"pairs whose two conversations share the prompt: {sum(train_split['shared'])}/{len(train_split)}")
train_split = train_split.remove_columns("shared")

beta = 0.1
def seq_logprob(mdl, text):
    ids = tok(text, truncation=True, max_length=256, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = mdl(ids).logits[:, :-1]
    lp = torch.log_softmax(logits, -1).gather(2, ids[:, 1:, None]).squeeze(-1)
    return lp.sum().item()

def heldout(mdl):                                       # implicit reward r = beta * log(pi_theta / pi_ref)
    wins = ties = losses = 0
    for ex in eval_prefs:
        d = beta * ((seq_logprob(mdl, ex["chosen"]) - seq_logprob(ref, ex["chosen"]))
                    - (seq_logprob(mdl, ex["rejected"]) - seq_logprob(ref, ex["rejected"])))
        wins += d > 0; ties += d == 0; losses += d < 0
    return wins, ties, losses

w, t, l = heldout(model)                                # before training the policy is the reference
print(f"held-out before DPO : {w} wins, {t} ties, {l} losses")
same = sum(tok(ex["chosen"], truncation=True, max_length=256).input_ids
           == tok(ex["rejected"], truncation=True, max_length=256).input_ids for ex in eval_prefs)
print(f"held-out pairs identical in their first 256 tokens: {same}")

cfg = DPOConfig(output_dir="dpo_tiny", beta=beta, per_device_train_batch_size=2,
                num_train_epochs=1, learning_rate=5e-6, max_length=256, logging_steps=25,
                report_to=[], seed=0, bf16=False, use_cpu=True, save_strategy="no")
trainer = DPOTrainer(model=model, args=cfg, train_dataset=train_split, processing_class=tok)
print(f"pairs kept for training: {len(trainer.train_dataset)} of {len(train_split)}")
trainer.train()
print(f"optimizer steps: {trainer.state.global_step}")
logs = [h for h in trainer.state.log_history if "loss" in h and "rewards/margins" in h]
print(f"\nDPO loss                        : {logs[0]['loss']:.3f} (steps 1-25) -> {logs[-1]['loss']:.3f} (final window)")
print(f"implicit-reward margin (train)  : {logs[0]['rewards/margins']:+.3f} -> {logs[-1]['rewards/margins']:+.3f}")
print(f"implicit-reward accuracy (train): {logs[0]['rewards/accuracies']:.2f} -> {logs[-1]['rewards/accuracies']:.2f}")

model.eval(); ref.eval()
w, t, l = heldout(model)
print(f"\nheld-out after DPO  : {w} wins, {t} ties, {l} losses")
print(f"the human-chosen reply is preferred in {w} of the {w + l} decided pairs ({w / (w + l):.0%})")

# %% [markdown]
# ## 18.10.7 RLVR: reinforcement learning from verifiable rewards

# %% [markdown]
# Example 18-34. A verifier reward and the group-relative advantages of one sampled group.

# %%
import re
import numpy as np

def verifier(completions, answer, **kwargs):
    # reward 1.0 when the first word of a reply is the gold label, else 0.0
    first = [(re.findall(r"[a-z]+", c.lower()) or [""])[0] for c in completions]
    return [float(f == a) for f, a in zip(first, answer)]

group = [" positive", " negative", " a must-see", " Positive!", " negative", " the film"]  # six replies
rewards = np.array(verifier(group, ["positive"] * len(group)))       # the gold label is positive
adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)            # as in Example 18-31
for reply, r, a in zip(group, rewards, adv):
    print(f"{reply!r:14} reward {r:.0f}  advantage {a:+.2f}")
print(f"mean {rewards.mean():.3f}, population std {rewards.std():.4f}, sample std {rewards.std(ddof=1):.4f}")

# %% [markdown]
# Example 18-35. GRPO with a verifier reward and a KL penalty on distilbert/distilgpt2, at two KL coefficients.

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
import re
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

torch.manual_seed(0); torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2"); tok.pad_token = tok.eos_token
ref = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").eval()      # frozen copy, for the held-out check
WORD = {0: "negative", 1: "positive"}
rt = load_dataset("cornell-movie-review-data/rotten_tomatoes")         # real movie-review sentiment
val = rt["validation"].shuffle(seed=0)
shots = [next(e for e in val if e["label"] == k) for k in (1, 0)]      # one positive, one negative example
head = "".join(f"Review: {s['text']}\nSentiment: {WORD[s['label']]}\n\n" for s in shots)
def to_prompt(e): return {"prompt": head + f"Review: {e['text']}\nSentiment:", "answer": WORD[e["label"]]}
train = rt["train"].shuffle(seed=0).select(range(32)).map(to_prompt, remove_columns=["text", "label"])
test = rt["test"].shuffle(seed=0).select(range(64)).map(to_prompt, remove_columns=["text", "label"])

# a one-token reply passes the verifier when that token's first word is the gold label
first = [(re.findall(r"[a-z]+", tok.decode([i]).lower()) or [""])[0] for i in range(len(tok))]
MASK = {w: torch.tensor([f == w for f in first]) for w in WORD.values()}
def heldout(mdl):   # exact first-token probabilities on 64 test reviews, averaged
    p_label = p_right = kl = 0.0
    for e in test:
        ids = tok(e["prompt"], return_tensors="pt").input_ids
        with torch.no_grad():
            lp = torch.log_softmax(mdl(ids).logits[0, -1], -1)
            lq = torch.log_softmax(ref(ids).logits[0, -1], -1)
        p = lp.exp()
        p_label += float(p[MASK["positive"] | MASK["negative"]].sum())
        p_right += float(p[MASK[e["answer"]]].sum())          # the expected verifier reward
        kl += float((p * (lp - lq)).sum())                     # KL(policy || reference)
    n = len(test)
    return p_label / n, p_right / n, kl / n

pl, pr, _ = heldout(ref)
print(f"before training           : P(label word) {pl:.2f}, P(correct label) {pr:.2f}")
for beta in [0.04, 1.0]:
    torch.manual_seed(0)
    policy = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")     # 82M parameters, the policy
    cfg = GRPOConfig(output_dir="grpo_tiny", beta=beta, learning_rate=1e-5, num_generations=8,
                     per_device_train_batch_size=8, max_completion_length=1, num_train_epochs=1,
                     logging_steps=8, seed=0, use_cpu=True, bf16=False, gradient_checkpointing=False,
                     report_to="none", save_strategy="no")
    trainer = GRPOTrainer(model=policy, reward_funcs=verifier, args=cfg, train_dataset=train,
                          processing_class=tok)
    trainer.train()
    logs = [h for h in trainer.state.log_history if "kl" in h]           # one entry per 8 steps
    print(f"beta {beta}: mean reward per 8 steps {np.round([h['reward'] for h in logs], 2)}, "
          f"mean KL {np.round([h['kl'] for h in logs], 2)}")
    policy.eval(); pl, pr, k = heldout(policy)
    print(f"  after training          : P(label word) {pl:.2f}, P(correct label) {pr:.2f}, "
          f"first-token KL to the reference {k:.2f}")

# %% [markdown]
# ## 18.11 Practical RL: environments, reward shaping, debugging, and exploration

# %% [markdown]
# ## 18.11.1 Gymnasium: the standard environment interface

# %% [markdown]
# Example 18-36. The five-state corridor as a custom Gymnasium environment.

# %%
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

class Corridor(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(5)
        self.action_space = spaces.Discrete(2)
        self.s = 0
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = 0
        return self.s, {}
    def step(self, action):
        self.s = max(0, self.s - 1) if action == 0 else min(4, self.s + 1)
        terminated = self.s == 4
        reward = 10.0 if terminated else -1.0
        return self.s, reward, terminated, False, {}

env = Corridor()
check_env(env, skip_render_check=True)
obs, _ = env.reset(seed=0)
total = 0.0
for _ in range(4):
    obs, reward, terminated, truncated, _ = env.step(1)
    total += reward
print("corridor", obs, total, terminated)

# %% [markdown]
# ## 18.11.3 Why reinforcement learning is hard to debug

# %% [markdown]
# Student's $t$ critical value at $0.975$ with $7$ degrees of freedom is $2.365$, so a 95 percent interval for the mean is $0.444 \pm 2.365 \times 0.271 / \sqrt{8}$, that is $[0.217, 0.670]$.

# %%
import numpy as np
from scipy.stats import t
rates = np.array([0.30, 0.63, 0.00, 0.63, 0.77, 0.51, 0.58, 0.13])
mean = rates.mean()
sd = rates.std(ddof=1)
half = t.ppf(0.975, len(rates) - 1) * sd / np.sqrt(len(rates))
print(f"mean {mean:.3f}, sample sd {sd:.3f}, 95% interval [{mean - half:.3f}, {mean + half:.3f}]")

# %% [markdown]
# ## 18.11.4 Exploration beyond epsilon-greedy: entropy, counts, and curiosity

# %% [markdown]
# Example 18-37. The entropy bonus keeps probability on every action.

# %%
import numpy as np

# a one-state policy over 4 actions with fixed action values (a constructed known world).
# We maximize  J = sum_a pi(a) R(a) + c * H(pi); the known optimum is softmax(R/c).
R = np.array([1.0, 0.9, 0.2, 0.0])                       # action values; actions 0 and 1 are close
def softmax(z): z = z - z.max(); e = np.exp(z); return e / e.sum()

def optimize(c, steps=4000, lr=0.2):
    theta = np.zeros(4)
    for _ in range(steps):
        pi = softmax(theta)
        meanR = pi @ R
        H = -(pi * np.log(pi + 1e-12)).sum()
        # gradient of J wrt logits: dJ/dtheta_k = pi_k [ (R_k - meanR) + c (-log pi_k - H) ]
        adv = (R - meanR) + c * (-np.log(pi + 1e-12) - H)
        grad = pi * adv
        theta += lr * grad
    pi = softmax(theta)
    return pi, -(pi * np.log(pi + 1e-12)).sum()

print("entropy bonus on a 4-action policy (values [1.0, 0.9, 0.2, 0.0]):")
for c in [0.0, 0.1, 0.5]:
    pi, H = optimize(c)
    line = f"  c={c:.1f}: policy {np.round(pi, 2)}   entropy {H:.2f}"
    if c > 0:
        line += f"   closed form softmax(R/c): {np.round(softmax(R / c), 2)}"
    print(line)
print("c=0 collapses onto the single best action (entropy near 0, no exploration); a positive")
print("bonus keeps probability on EVERY action, more of it the larger c is (higher entropy =")
print("more exploration). A2C, PPO, and SAC all add such a term; SAC makes it the objective itself.")

# %% [markdown]
# Example 18-38. Count-based exploration on a sparse 12-state corridor.

# %%
import numpy as np

N_S, GOAL = 12, 11
def move(s, a): return max(0, s - 1) if a == 0 else min(N_S - 1, s + 1)

def q_learn(bonus_beta, seed, episodes=150, alpha=0.5, gamma=0.95, eps=0.1, max_steps=60):
    rg = np.random.default_rng(seed)
    Q = np.zeros((N_S, 2)); visits = np.zeros(N_S); first = None
    for ep in range(episodes):
        s = 0
        for _ in range(max_steps):
            a = int(rg.integers(2)) if rg.random() < eps else int(np.argmax(Q[s]))
            s2 = move(s, a); visits[s2] += 1
            r = (1.0 if s2 == GOAL else 0.0) + (bonus_beta / np.sqrt(visits[s2]) if bonus_beta else 0.0)
            done = s2 == GOAL
            Q[s, a] += alpha * (r + gamma * (0 if done else Q[s2].max()) - Q[s, a])
            s = s2
            if done:
                if first is None: first = ep + 1
                break
    s = 0
    for _ in range(60):                          # does the greedy policy reach the goal?
        s = move(s, int(np.argmax(Q[s])))
        if s == GOAL: break
    return first, s == GOAL

for beta in [0.0, 0.5]:
    firsts, solved = [], 0
    for seed in range(20):
        f, ok = q_learn(beta, seed)
        firsts.append(np.inf if f is None else f); solved += ok
    med = np.median(firsts); never = int(np.sum(np.isinf(firsts)))
    print(f"beta={beta}: greedy policy solves {solved}/20 seeds; median first-goal episode "
          f"{med}; never reached the goal in {never}/20 runs")

# %% [markdown]
# Example 18-39. A prediction-error bonus beside the count bonus, with and without a noisy TV, and pseudocounts.

# %%
import numpy as np

N_S, GOAL = 12, 11                                  # the sparse 12-state corridor of the count-bonus listing
ONE_HOT = np.eye(N_S)                               # row k is the one-hot code of state k
def move(s, a, rg, tv):
    if tv and s == 0 and a == 0:                    # the noisy TV: from state 0, left lands on 0 or 1 at random
        return int(rg.integers(2))
    return max(0, s - 1) if a == 0 else min(N_S - 1, s + 1)

def q_learn(bonus, seed, tv=False, beta=0.5, eta=0.5, lr_model=0.5,
            episodes=150, alpha=0.5, gamma=0.95, eps=0.1, max_steps=60):
    rg = np.random.default_rng(seed)
    Q = np.zeros((N_S, 2)); visits = np.zeros(N_S); first = None
    f = np.zeros((N_S, 2, N_S))                     # learned model: f[s, a] predicts the next state's one-hot code
    for ep in range(episodes):
        s = 0
        for _ in range(max_steps):
            a = int(rg.integers(2)) if rg.random() < eps else int(np.argmax(Q[s]))
            s2 = move(s, a, rg, tv); visits[s2] += 1
            error = np.sum((f[s, a] - ONE_HOT[s2]) ** 2)    # squared prediction error, before learning
            f[s, a] += lr_model * (ONE_HOT[s2] - f[s, a])   # gradient step on half the squared error
            r = 1.0 if s2 == GOAL else 0.0
            if bonus == "count":
                r += beta / np.sqrt(visits[s2])
            elif bonus == "prediction error":
                r += eta * error
            done = s2 == GOAL
            Q[s, a] += alpha * (r + gamma * (0 if done else Q[s2].max()) - Q[s, a])
            s = s2
            if done:
                if first is None: first = ep + 1
                break
    s = 0
    for _ in range(60):                              # does the greedy policy reach the goal?
        s = max(0, s - 1) if np.argmax(Q[s]) == 0 else min(N_S - 1, s + 1)
        if s == GOAL: break
    return first, s == GOAL, visits

print("bonus (20 seeds each)     : greedy solves, median first goal, never reached, visits in states 0-1")
for tv, bonus in [(False, "none"), (False, "count"), (False, "prediction error"),
                  (True, "count"), (True, "prediction error")]:
    firsts, solved, near = [], 0, []
    for seed in range(20):
        f_ep, ok, visits = q_learn(bonus, seed, tv)
        firsts.append(np.inf if f_ep is None else f_ep); solved += ok
        near.append((visits[0] + visits[1]) / visits.sum())       # share of time spent at the start
    label = ("noisy TV, " if tv else "") + bonus
    print(f"{label:26s}: {solved:2d}/20, {np.median(firsts):5}, {int(np.sum(np.isinf(firsts))):2d}/20, "
          f"median share {np.median(near):.2f}")

# pseudocounts (Bellemare et al., 2016) from a density model over states: rho before one more
# visit to x, rho_next after it. The empirical density N(x)/n is the tabular case.
_, _, visits = q_learn("count", 0)
n = visits.sum()
rho, rho_next = visits / n, (visits + 1) / (n + 1)
pseudo = rho * (1 - rho_next) / (rho_next - rho)
print(f"\ncount learner, seed 0, {int(n)} state visits:")
print(f"  visit counts N(s)    : {visits.astype(int)}")
print(f"  pseudocounts         : {np.round(pseudo, 6)}")
print(f"  largest |pseudo - N| : {np.abs(pseudo - visits).max():.1e}")

# %% [markdown]
# The remaining two logits pass through a softmax, and the masked entry is 0.

# %%
import torch
logits = torch.tensor([1.0, 2.0, 0.0])
masked = logits.clone()
masked[1] = float("-inf")
probs = torch.softmax(masked, dim=0)
print("masked", [round(float(x), 6) for x in probs])

# %% [markdown]
# Example 18-40. Gymnasium, reward shaping, and seed variance.

# %%
import numpy as np
import gymnasium as gym
rng = np.random.default_rng(0)

env = gym.make("FrozenLake-v1", is_slippery=True)
obs, info = env.reset(seed=0)
print(f"FrozenLake: observation_space={env.observation_space}, action_space={env.action_space}")
obs2, r, term, trunc, _ = env.step(1)                    # take action 'down'
print(f"reset -> state {obs}; step(1) -> state {obs2}, reward {r}, terminated {term}")
print("every Gymnasium env exposes this same interface, which is why algorithms are reusable.")

N, START, GOAL, BONUS = 7, 3, 6, 1
def move(s, a): return max(0, s - 1) if a == 0 else min(N - 1, s + 1)

def q_learn(reward_fn, episodes=3000, alpha=0.5, eps=0.1, gamma=0.95, max_steps=80):
    Q = np.zeros((N, 2))
    for _ in range(episodes):
        s = START
        for _ in range(max_steps):
            a = rng.integers(2) if rng.random() < eps else int(np.argmax(Q[s]))
            s2 = move(s, a); r, done = reward_fn(s, s2)
            Q[s, a] += alpha * (r + gamma * (0 if done else np.max(Q[s2])) - Q[s, a])
            s = s2
            if done: break
    return Q

def evaluate(Q, max_steps=80):
    # greedy rollout: does the greedy policy reach the goal?
    s = START; reached = False
    for _ in range(max_steps):
        s = move(s, int(np.argmax(Q[s])))
        if s == GOAL: reached = True; break
    return reached

def sparse(s, s2):    return (1.0, True) if s2 == GOAL else (0.0, False)        # only the goal pays
def potential(s, s2):                                                            # potential-based shaping
    phi = lambda x: -abs(GOAL - x) / N                                           # closer = higher potential
    if s2 == GOAL:
        return 1.0 + 0.95 * phi(GOAL) - phi(s), True                             # phi(goal) is 0
    return 0.95 * phi(s2) - phi(s), False                                        # gamma*phi(s')-phi(s)
def bad_shaping(s, s2):                                                          # farmable bonus = a trap
    if s2 == GOAL: return 1.0, True
    return (0.3, False) if s2 == BONUS else (0.0, False)

print("\nreward shaping on the corridor (goal-reaching after training):")
for name, fn in [("sparse (goal only)", sparse), ("potential-based (good)", potential),
                 ("bad shaping (farmable bonus)", bad_shaping)]:
    reaches = np.mean([evaluate(q_learn(fn)) for _ in range(20)])
    print(f"  {name:30s}: reaches goal in {reaches*100:3.0f}% of runs")
print("good shaping speeds learning AND keeps the goal optimal; the farmable bonus gets HACKED -")
print("the agent oscillates on the bonus cell to collect +0.3 forever and never finishes.")

def train_frozenlake(seed, episodes=2000, alpha=0.8, gamma=0.95):
    e = gym.make("FrozenLake-v1", is_slippery=True)
    Q = np.zeros((e.observation_space.n, e.action_space.n))
    rg = np.random.default_rng(seed)
    for ep in range(episodes):
        s, _ = e.reset(seed=seed * 100000 + ep); done = False; eps = max(0.05, 1 - ep / 1000)
        while not done:
            a = rg.integers(4) if rg.random() < eps else int(np.argmax(Q[s]))
            s2, r, term, trunc, _ = e.step(a); done = term or trunc
            Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) * (not term) - Q[s, a]); s = s2
    # evaluate greedy success rate over 200 episodes
    wins = 0
    for k in range(200):
        s, _ = e.reset(seed=10**6 + k); done = False
        while not done:
            s, r, term, trunc, _ = e.step(int(np.argmax(Q[s]))); done = term or trunc
        wins += int(r == 1.0)
    return wins / 200

rates = [train_frozenlake(seed) for seed in range(8)]
print(f"\ngreedy success rate on slippery FrozenLake across 8 seeds:")
print("  [" + " ".join(f"{r:.2f}" for r in rates) + "]")     # same rounding as the summary line
print(f"  min {min(rates):.2f}, max {max(rates):.2f}, mean {np.mean(rates):.2f} +/- {np.std(rates):.2f}")
print("same code, just a different seed, yet results swing widely: this is why RL papers report")
print("many seeds, and why a single good run proves almost nothing.")

print("\nAll practical-RL checks ran.")

# %% [markdown]
# ## 18.12 Hands-on lab: tabular Q-learning, DQN, and PPO

# %% [markdown]
# Example 18-41. The lab: tabular Q-learning, DQN, and library PPO.

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
print("and a LIBRARY implementation of PPO for getting real work done: same RL ideas throughout.")
print("\nAll RL-lab checks ran.")

# %% [markdown]
# ## 18.13 Interview questions and coding exercises

# %% [markdown]
# ## Coding exercises

# %% [markdown]
# Coding exercise 1: Bandit regret.

# %%
import numpy as np
true_p = np.linspace(0.1, 0.9, 10)

def bandit_strategies(true_p, runs=200, horizon=1000):
    """Average reward and regret for five strategies."""
    ...

print("arms", true_p.shape[0], "best", round(float(true_p.max()), 1))

# %% [markdown]
# Coding exercise 2: Value iteration from scratch.

# %%
SIZE, GOAL, gamma = 4, 15, 0.9

def value_iteration(size=SIZE, goal=GOAL, gamma=gamma):
    """Return the sweep count, the corner value, and the greedy path length."""
    ...

print("grid", SIZE * SIZE, "gamma", gamma, "goal", GOAL)

# %% [markdown]
# Coding exercise 3: Policy iteration equals value iteration.

# %%
def policy_iteration(size=4, goal=15, gamma=0.9):
    """Return the optimal values and the number of improvement rounds."""
    ...

print("states", 4 * 4)

# %% [markdown]
# Coding exercise 4: Bellman as a linear solve versus Monte Carlo.

# %%
N, gamma, p_success = 5, 0.9, 0.8

def corridor_values(n=N, gamma=gamma, p_success=p_success, rollouts=3000):
    """Return the linear solve and the Monte Carlo estimates."""
    ...

print("states", N, "gamma", gamma, "success", p_success)

# %% [markdown]
# Coding exercise 5: TD(0) versus Monte Carlo prediction.

# %%
def td_versus_mc(episodes=500):
    """Return the Monte Carlo RMSE and the TD(0) RMSE."""
    ...

print("episodes", 500)

# %% [markdown]
# Coding exercise 6: SARSA versus Q-learning on CliffWalking-v1.

# %%
import gymnasium as gym

def cliff_compare(episodes=500):
    """Train SARSA and Q-learning on CliffWalking-v1."""
    ...

env = gym.make("CliffWalking-v1")
print(env.observation_space.n, env.action_space.n)
env.close()

# %% [markdown]
# Coding exercise 7: Tabular Q-learning on FrozenLake-v1.

# %%
import gymnasium as gym

def frozen_q(map_name="4x4", episodes=8000, alpha=0.8, gamma=0.95):
    """Train tabular Q-learning on slippery FrozenLake-v1; return (success rate, goals in training)."""
    ...

for name in ["4x4", "8x8"]:
    env = gym.make("FrozenLake-v1", map_name=name, is_slippery=True)
    print(name, env.observation_space.n, env.action_space.n, env.spec.max_episode_steps)
    env.close()

# %% [markdown]
# Coding exercise 8: The RLHF machinery from scratch.

# %%
import numpy as np
rewards = np.array([1.0, 0.0, 0.0, 1.0, 1.0])

def rlhf_pieces(rewards):
    """Return preference accuracy, KL-penalized weights, and GRPO advantages."""
    ...

print("group", rewards.shape[0], "mean", round(float(rewards.mean()), 1))

# %% [markdown]
# Coding exercise 9: Monte Carlo Tree Search on tic-tac-toe.

# %%
board = [0] * 9

def mcts_move(board, simulations=200):
    """Return the cell chosen on this tic-tac-toe board."""
    ...

print("cells", len(board))

# %% [markdown]
# Coding exercise 10: Deep Q-network on CartPole-v1 (miniproject).

# %%
import gymnasium as gym

def train_dqn(episodes=200):
    """Train a small DQN on CartPole-v1 and return the early and late mean returns."""
    ...

env = gym.make("CartPole-v1")
print(env.observation_space.shape[0], env.action_space.n)
env.close()

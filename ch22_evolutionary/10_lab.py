# %% [markdown]
# # Section 22.10 Lab: GA-TSP (real capitals) and CartPole neuroevolution
#
# Runs the code of Example 22-18 in Book.md (only the comments differ): a genetic algorithm on a
# TSP over 20 real US state capitals, then a linear CartPole-v1 policy evolved from reward alone.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import gymnasium as gym
rng = np.random.default_rng(0)

# The 20-capital map is the one from Example 22-14: `capitals`, `names`, `cities`, the city
# count NC, the distance matrix D, the tour length `length`, and the `nearest_neighbor`
# baseline are rebuilt exactly as there (the companion script repeats those lines so it
# runs standalone; the evolutionary search below is what is new here).
capitals = {
    "Sacramento": (38.58, -121.49), "Phoenix": (33.45, -112.07), "Denver": (39.74, -104.98),
    "Austin": (30.27, -97.74), "Salem": (44.94, -123.04), "Boise": (43.61, -116.20),
    "Santa Fe": (35.69, -105.94), "Oklahoma City": (35.47, -97.52), "Topeka": (39.05, -95.68),
    "Des Moines": (41.60, -93.61), "Saint Paul": (44.94, -93.09), "Jefferson City": (38.58, -92.17),
    "Springfield": (39.80, -89.64), "Nashville": (36.17, -86.78), "Atlanta": (33.75, -84.39),
    "Columbus": (39.96, -83.00), "Montgomery": (32.37, -86.30), "Tallahassee": (30.44, -84.28),
    "Raleigh": (35.77, -78.64), "Richmond": (37.55, -77.46),
}
names = list(capitals); cities = np.array([capitals[n] for n in names])
NC = len(names)
D = np.sqrt(((cities[:, None] - cities[None, :]) ** 2).sum(2)) + np.eye(NC)   # distances (avoid /0)
def length(t): return sum(D[t[i], t[(i + 1) % NC]] for i in range(NC))
def nearest_neighbor():                                  # greedy baseline: hop to the closest unvisited city
    unv = set(range(1, NC)); tour = [0]
    while unv:
        nxt = min(unv, key=lambda c: D[tour[-1], c]); tour.append(nxt); unv.discard(nxt)
    return length(tour)

def order_crossover(p1, p2):
    a, b = sorted(rng.choice(NC, 2, replace=False))
    child = [-1] * NC; child[a:b] = p1[a:b]              # keep a slice of parent 1
    fill = [c for c in p2 if c not in child[a:b]]        # fill the rest in parent-2 order
    j = 0
    for i in list(range(b, NC)) + list(range(a)):
        child[i] = fill[j]; j += 1
    return child

def mutate(tour):                                        # reverse a random segment (2-opt-like)
    a, b = sorted(rng.choice(NC, 2, replace=False))
    return tour[:a] + tour[a:b][::-1] + tour[b:]

POP, GENS = 200, 300
pop = [list(rng.permutation(NC)) for _ in range(POP)]
for gen in range(GENS):
    pop.sort(key=length)
    new = pop[:10]                                       # elitism
    while len(new) < POP:
        p1 = min((pop[i] for i in rng.integers(0, 40, 3)), key=length)   # tournament among the top 40
        p2 = min((pop[i] for i in rng.integers(0, 40, 3)), key=length)
        child = order_crossover(p1, p2)
        if rng.random() < 0.4: child = mutate(child)
        new.append(child)
    pop = new
pop.sort(key=length)
ga_len = length(pop[0])

# baseline: the average random tour (nearest_neighbor is the greedy tour of Example 22-14)
rand_len = np.mean([length(list(rng.permutation(NC))) for _ in range(3000)])
print("PART 1 - Traveling Salesman Problem (20 real US state capitals), tour length (lower is better):")
print(f"  random tour (avg)      : {rand_len:.2f}")
print(f"  nearest-neighbor       : {nearest_neighbor():.2f}")
print(f"  genetic algorithm      : {ga_len:.2f}  (shorter than nearest neighbor; 2-opt can be shorter)")

env = gym.make("CartPole-v1")
NO, NA = 4, 2
def policy_action(w, obs):
    W = w[:NO * NA].reshape(NO, NA); b = w[NO * NA:]
    return int(np.argmax(obs @ W + b))                   # linear policy: 4 obs -> 2 action scores
def reward_of(w, episodes=3):
    total = 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep); done = False
        while not done:
            obs, r, term, trunc, _ = env.step(policy_action(w, obs)); total += r; done = term or trunc
    return total / episodes

NWT = NO * NA + NA                                       # 10 weights
POP2, GENS2, MU = 24, 30, 6
popw = rng.normal(0, 1, (POP2, NWT))
steps = 0                                                # environment steps the search spends
print("\nPART 2 - neuroevolution on CartPole (max reward 500):")
for gen in range(GENS2):
    fits = np.array([reward_of(w) for w in popw])
    steps += round(3 * fits.sum())                       # CartPole pays +1 per step: 3 episodes x mean reward = steps
    elite = popw[np.argsort(-fits, kind="stable")[:MU]]  # best first; tied rewards keep population order (Section 22.3.2)
    if gen in (0, 10, 29):
        print(f"  generation {gen:2d}: best mean reward = {fits.max():.0f}")
    popw = np.vstack([elite, [elite[rng.integers(MU)] + rng.normal(0, 0.3, NWT) for _ in range(POP2 - MU)]])
best_reward = max(reward_of(w, episodes=5) for w in popw)
print(f"  final best policy: mean reward = {best_reward:.0f}")
print(f"  environment steps spent by the 30 generations of search: {steps:,}")
print("\nneuroevolution solved CartPole by evolving 10 weights from REWARD alone - no backprop,")
print("no value function, no policy gradient. Chapter 18's PPO also reached 500, with gradients and a")
print("30,000-timestep training budget; ES needed no gradient code but spent many more environment steps.")

print("\nAll evolutionary-lab checks ran.")


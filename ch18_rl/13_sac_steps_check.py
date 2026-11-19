# %% [markdown]
# # Section 18.7.7 check: Example 18-19 at three training budgets
#
# Companion script for Section 18.7.7. It reruns Example 18-19 (SAC from Stable-Baselines3 on Gymnasium's Pendulum-v1)
# with only `total_timesteps` changed, at 3000, 7000, and 20000 environment steps, and prints the mean and the standard
# deviation of the return over the same ten evaluation episodes after `eval_env.reset(seed=123)`. Each budget starts
# from the same seeds as Example 18-19, so the 7000-step line repeats that listing's result. One timed run took about
# nine and a half minutes (561 seconds) on the six-core desktop CPU used for this book's outputs, which is why it
# lives here and not in the chapter walkthrough. Tested with gymnasium 1.4.0, stable-baselines3 2.9.2, and
# torch 2.15.0 (CPU).

# %%
import numpy as np, torch, random
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

for steps in [3000, 7000, 20000]:
    np.random.seed(0); torch.manual_seed(0); random.seed(0); torch.set_num_threads(2)
    env = gym.make("Pendulum-v1")
    eval_env = Monitor(gym.make("Pendulum-v1"))
    model = SAC("MlpPolicy", env, verbose=0, seed=0, learning_starts=1000)
    eval_env.reset(seed=123)
    before, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    model.learn(total_timesteps=steps, progress_bar=False)
    eval_env.reset(seed=123)
    after, sd = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"SAC, {steps:5d} steps: untrained {before:.0f}, trained {after:.0f} +/- {sd:.0f} over 10 episodes")

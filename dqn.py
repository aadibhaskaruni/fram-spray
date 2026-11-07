# Step 12 - Deep Q-Network (DQN) Comparison for Precision Farm Spraying
# Trains DQN on the same simulator and compares it directly against PPO and Baselines.
# Run from the project root. Shift+Enter per cell "# %%".

import sys, time
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import stable_baselines3 as sb3
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from src.weather_api import load_or_fetch
from src.spray_env import SprayEnv, SprayConfig
from src.policies import NoSpray, FixedRate, SmartThreshold
from src.evaluation import evaluate, summary
from src.sb3_utils import TrueRewardLogger, SB3Policy

TANK = 100
SEED = 0
SHAPING = 0.3
STEPS_FULL = 600_000

for d in ("models", "outputs"): Path(d).mkdir(exist_ok=True)

# %% 1. Load Data & Define Training Environment
weather, _ = load_or_fetch("Pune", "2026-06-01", "2026-09-30")
bank = dict(np.load("data/tiles/score_bank.npz"))
cut = int(0.7 * len(weather))
TRAIN_W, TEST_W = (0, cut), (cut, len(weather))

def make_train_env():
    return SprayEnv(bank, weather, SprayConfig(tank=TANK, shaping=SHAPING), weather_range=TRAIN_W)

def build_dqn_model(seed=SEED):
    env = make_train_env()
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        tau=1.0,
        gamma=0.995,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=0
    )

# %% 2. Train DQN (or load if already trained)
model_path = f"models/dqn_tank{TANK}_seed{SEED}.zip"
curve_path = f"outputs/step12_curve_dqn_seed{SEED}.csv"

if not Path(model_path).exists():
    print(f"Training DQN for {STEPS_FULL} steps...")
    model = build_dqn_model(seed=SEED)
    logger = TrueRewardLogger(curve_path)
    t0 = time.time()
    model.learn(total_timesteps=STEPS_FULL, callback=logger)
    print(f"Training finished in {(time.time() - t0)/60:.1f} min. Saving...")
    model.save(model_path)
else:
    print(f"Loading existing DQN model from {model_path}")
    model = DQN.load(model_path)

# %% 3. Evaluate DQN vs PPO vs Baselines
ppo_model = sb3.PPO.load(f"models/ppo_tank{TANK}_seed{SEED}.zip")

policies = {
    "No spray": NoSpray(),
    "Context rule (strong)": SmartThreshold(tau=0.15),
    "PPO": SB3Policy(ppo_model),
    "DQN": SB3Policy(model)
}

test_env = SprayEnv(bank, weather, SprayConfig(tank=TANK, shaping=0.0), weather_range=TEST_W)
print("Evaluating policies across 100 test seasons...")
res = evaluate(policies, test_env, n_episodes=100, seed=123)
df_summary = summary(res)
print(df_summary)
df_summary.to_csv("outputs/step12_table.csv", index=False)

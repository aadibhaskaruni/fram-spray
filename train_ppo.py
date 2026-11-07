# Step 8 - Train the PPO agent (Stable-Baselines3) and compare it with the baselines.
# Run from the project root. Shift+Enter on each "# %%" cell.
#
# BEFORE running, install (CPU-only PyTorch is much smaller than the default one):
#     pip install torch --index-url https://download.pytorch.org/whl/cpu
#     pip install stable-baselines3
# pip may change the gymnasium version to one SB3 supports - that is fine.

# %% 1. Check the installation and set the experiment settings
import sys, time
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch, stable_baselines3 as sb3
print("torch", torch.__version__, "| stable-baselines3", sb3.__version__)

TANK = 100                 # chosen in Step 7b
SEED = 0                   # change to 1, 2 ... and re-run to see how much results vary between runs
SHAPING = 0.3              # reward shaping used ONLY during training (all reported numbers are unshaped)
STEPS_SMOKE, STEPS_FULL = 20_000, 600_000
for d in ("models", "outputs"): Path(d).mkdir(exist_ok=True)

# %% 2. Load the data and define the environments
from src.weather_api import load_or_fetch
from src.spray_env import SprayEnv, SprayConfig
from src.policies import NoSpray, FixedRate, SmartThreshold, OraclePolicy
from src.evaluation import evaluate, summary
from src.sb3_utils import TrueRewardLogger, SB3Policy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

weather, _ = load_or_fetch("Pune", "2026-06-01", "2026-09-30")
bank = dict(np.load("data/tiles/score_bank.npz"))
cut = int(0.7 * len(weather)); TRAIN_W, TEST_W = (0, cut), (cut, len(weather))

def make_train_env():
    return SprayEnv(bank, weather, SprayConfig(tank=TANK, shaping=SHAPING), weather_range=TRAIN_W)   # TRAIN weather only

def new_model(steps_env=4):
    venv = VecNormalize(make_vec_env(make_train_env, n_envs=steps_env, seed=SEED), norm_obs=False, norm_reward=True, gamma=0.995)
    return PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=6, gamma=0.995,
               gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
               policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])), seed=SEED, verbose=0)
print("Training scenarios: clustered/random patterns, training-weeks weather. Test scenarios stay unseen.")

# %% 3. Smoke test: a short run to check that everything works and to time it
t0 = time.time()
m = new_model()
m.learn(total_timesteps=STEPS_SMOKE)
dt = time.time() - t0
print(f"{STEPS_SMOKE} steps took {dt:.0f} s  ->  full run of {STEPS_FULL} steps will take about {dt * STEPS_FULL / STEPS_SMOKE / 60:.0f} minutes")

# %% 4. Full training (the slow cell). The curve file is updated while it runs.
model = new_model()
logger = TrueRewardLogger(f"outputs/step08_curve_seed{SEED}.csv")
t0 = time.time()
model.learn(total_timesteps=STEPS_FULL, callback=logger)
model.save(f"models/ppo_tank{TANK}_seed{SEED}")
print(f"Training finished in {(time.time() - t0) / 60:.1f} minutes. Model saved to models/ppo_tank{TANK}_seed{SEED}.zip")

# %% 5. Baselines for comparison (tuned on TRAINING weather with a small grid)
import itertools
TUNE_SEEDS, TEST_SEEDS = range(0, 20), range(1000, 1030)
cuts_grid = [c for c in itertools.product([0.1, 0.2, 0.3], [0.3, 0.45, 0.6], [0.5, 0.7, 0.9]) if c[0] < c[1] < c[2]]
def tune(make):
    return max(cuts_grid, key=lambda c: evaluate(make(c), bank, weather, TRAIN_W, TUNE_SEEDS, tank=TANK)["reward"].mean())
tuned = SmartThreshold(tune(lambda c: SmartThreshold(c, 0.5)), 0.5)
oracle = OraclePolicy(tune(OraclePolicy))
ppo_policy = SB3Policy(model, "PPO")
POLICIES = [NoSpray(), FixedRate(1), FixedRate(2), tuned, oracle, ppo_policy]
print("Tuned threshold cut-offs:", tuned.cuts)

# %% 6. Learning curve: does the agent improve over training?
cur = pd.read_csv(f"outputs/step08_curve_seed{SEED}.csv")
ref = {p.name: evaluate(p, bank, weather, TRAIN_W, range(2000, 2030), tank=TANK)["reward"].mean() for p in (NoSpray(), tuned)}
w = 30
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.plot(cur.step, cur.episode_reward, color="#c9bfe6", lw=0.7, label="PPO, each training episode (true reward)")
ax.plot(cur.step[w - 1:], cur.episode_reward.rolling(w).mean().dropna(), color="#5b3fa8", lw=2.5, label=f"PPO, moving average of {w}")
for (n, v), col in zip(ref.items(), ("#888888", "#2a7f62")):
    ax.axhline(v, ls="--", color=col, label=f"{n} ({v:.1f})")
ax.set_xlabel("training steps"); ax.set_ylabel("episode reward (unshaped)"); ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower right")
ax.set_title("PPO learning curve (training weather, seen patterns)")
fig.savefig("outputs/step08_learning_curve.png", dpi=130, bbox_inches="tight"); plt.show()

# %% 7. Final evaluation on UNSEEN seasons in all four scenarios (paired comparison)
SCEN = {"Test weeks, seen patterns": (TEST_W, {}), "Test weeks, unseen edge": (TEST_W, {"patterns": ("edge",)}),
        "Test weeks, bad sensor (20%)": (TEST_W, {"sensor_flip": 0.2}), "Harsh weather (train weeks)": (TRAIN_W, {})}
rows, diffs = [], {}
for sname, (wr, kw) in SCEN.items():
    res = {p.name: evaluate(p, bank, weather, wr, TEST_SEEDS, tank=TANK, **kw) for p in POLICIES}
    d = res["PPO"]["reward"] - res["Tuned threshold"]["reward"]
    diffs[sname] = (d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
    for p in POLICIES:
        s = summary(res[p.name], res["No spray"])
        rows.append(dict(scenario=sname, policy=p.name, gain_vs_no_spray=s["gain"][0], gain_se=s["gain"][1],
                         yield_pct=s["yield_pct"][0], pesticide=s["pesticide"][0], unnecessary_pct=s["unnecessary_pct"][0]))
df = pd.DataFrame(rows); df.round(2).to_csv(f"outputs/step08_results_seed{SEED}.csv", index=False)
for sname in SCEN:
    print(f"\n=== {sname} ===")
    print(df[df.scenario == sname].drop(columns="scenario").round(2).to_string(index=False))

# %% 8. Is PPO actually better than the tuned rule? (paired difference, same seasons)
print(f"\nPPO minus tuned threshold (reward, mean +- standard error over {len(TEST_SEEDS)} seasons):")
for sname, (m_, se) in diffs.items():
    verdict = "PPO better" if m_ - 2 * se > 0 else ("PPO worse" if m_ + 2 * se < 0 else "no clear difference")
    print(f"  {sname:30s}: {m_:+.2f} +- {se:.2f}   -> {verdict}")
print("One training run is one sample: repeat with SEED = 1, 2 before drawing conclusions.")

# %% 9. Picture
names = [p.name for p in POLICIES]; colors = ["#999999", "#d9c9a0", "#c9b37e", "#4f9d7f", "#bbbbdd", "#5b3fa8"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for a, (col, title) in zip(ax, [("gain_vs_no_spray", "Reward gain vs no spray"), ("yield_pct", "Final yield %"), ("pesticide", "Pesticide used")]):
    wd = 0.8 / len(names)
    for i, nme in enumerate(names):
        sub = [df[(df.scenario == s) & (df.policy == nme)] for s in SCEN]
        a.bar(np.arange(len(SCEN)) + i * wd, [x[col].iloc[0] for x in sub], wd, color=colors[i], label=nme,
              yerr=[x.gain_se.iloc[0] for x in sub] if col == "gain_vs_no_spray" else None, capsize=2)
    a.set_xticks(np.arange(len(SCEN)) + 0.4 - wd / 2); a.set_xticklabels([s.replace(", ", "\n") for s in SCEN], fontsize=8)
    a.set_title(title, fontsize=10); a.grid(axis="y", alpha=0.25)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=len(names), frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f"outputs/step08_results_seed{SEED}.png", dpi=130); plt.show()
print("Step 8 finished.")
# %%

# Step 10 - Does the reward-shaping term make PPO overspend?  Train with shaping 0.1 and 0.0 and compare.
# Run from the project root. Shift+Enter on each "# %%" cell.
# Needs the Step 8/9 models (models/ppo_tank100_seed{0,1,2}.zip). Training 4 new models takes about 40-50 minutes
# (cell 3). Finished models are skipped, so you can stop and re-run.

# %% 1. Settings and data
import sys, time
sys.path.insert(0, ".")
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.weather_api import load_or_fetch
from src.spray_env import SprayEnv, SprayConfig
from src.policies import NoSpray, SmartThreshold, ContextRule
from src.evaluation import evaluate
from src.sb3_utils import TrueRewardLogger, SB3Policy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

TANK, STEPS = 100, 300_000          # PPO plateaued after ~150k steps in Step 8, so 300k is enough for this test
SHAPINGS, SEEDS = (0.1, 0.0), (0, 1)
TUNE_SEEDS, TEST_SEEDS = range(0, 20), range(1000, 1030)
weather, _ = load_or_fetch("Pune", "2026-06-01", "2026-09-30")
bank = dict(np.load("data/tiles/score_bank.npz"))
cut = int(0.7 * len(weather)); TRAIN_W, TEST_W = (0, cut), (cut, len(weather))
for d in ("models", "outputs"): Path(d).mkdir(exist_ok=True)

# %% 2. Rebuild the two rule-based baselines exactly as in Step 9 (the search is deterministic)
cuts_grid = [c for c in itertools.product([0.1, 0.2, 0.3], [0.3, 0.45, 0.6], [0.5, 0.7, 0.9]) if c[0] < c[1] < c[2]]
score = lambda pol: evaluate(pol, bank, weather, TRAIN_W, TUNE_SEEDS, tank=TANK)["reward"].mean()
tuned = SmartThreshold(max(cuts_grid, key=lambda c: score(SmartThreshold(c, 0.5))), 0.5)
rng = np.random.default_rng(0)
cands = [(tuned.cuts, 0.5, 0.0, 0.0)]
for _ in range(100):
    c1 = rng.uniform(0.05, 0.4); c2 = rng.uniform(c1 + 0.05, 0.8); c3 = rng.uniform(c2 + 0.05, 0.99)
    cands.append(((c1, c2, c3), rng.uniform(0.3, 1.0), rng.uniform(0, 2), rng.uniform(0, 2)))
context = ContextRule(*max(cands, key=lambda p: score(ContextRule(*p))))
print("Tuned cuts:", tuned.cuts, "| context rule: season factor", round(context.b, 2), "wind factor", round(context.a, 2))

# %% 3. Train PPO with different shaping weights (slow). Safe to stop and re-run.
def new_model(seed, shaping):
    mk = lambda: SprayEnv(bank, weather, SprayConfig(tank=TANK, shaping=shaping), weather_range=TRAIN_W)
    venv = VecNormalize(make_vec_env(mk, n_envs=4, seed=seed), norm_obs=False, norm_reward=True, gamma=0.995)
    return PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=6, gamma=0.995, gae_lambda=0.95,
               clip_range=0.2, ent_coef=0.01, policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])), seed=seed, verbose=0)

GROUPS = {"PPO shaping 0.3 (600k)": [f"models/ppo_tank{TANK}_seed{s}" for s in (0, 1, 2)]}
for sh in SHAPINGS:
    paths = []
    for s in SEEDS:
        path = f"models/ppo_tank{TANK}_shape{sh}_seed{s}_300k"; paths.append(path)
        if Path(path + ".zip").exists():
            print(f"shaping {sh}, seed {s}: exists, skipping"); continue
        t0 = time.time(); m = new_model(s, sh)
        m.learn(total_timesteps=STEPS, callback=TrueRewardLogger(f"outputs/step10_curve_shape{sh}_seed{s}.csv"))
        m.save(path); print(f"shaping {sh}, seed {s}: trained in {(time.time() - t0) / 60:.1f} min")
    GROUPS[f"PPO shaping {sh} (300k)"] = paths

# %% 4. Evaluate every model on the same unseen seasons
SCEN = {"Test weeks, seen patterns": (TEST_W, {}), "Test weeks, unseen edge": (TEST_W, {"patterns": ("edge",)}),
        "Test weeks, bad sensor (20%)": (TEST_W, {"sensor_flip": 0.2}), "Harsh weather (train weeks)": (TRAIN_W, {})}
R = {sn: {} for sn in SCEN}
for sn, (wr, kw) in SCEN.items():
    for p in (NoSpray(), tuned, context):
        R[sn][p.name] = evaluate(p, bank, weather, wr, TEST_SEEDS, tank=TANK, **kw)
for g, paths in GROUPS.items():
    for i, path in enumerate(paths):
        pol = SB3Policy(PPO.load(path), f"{g} #{i}")
        for sn, (wr, kw) in SCEN.items():
            R[sn][pol.name] = evaluate(pol, bank, weather, wr, TEST_SEEDS, tank=TANK, **kw)

rows = []
for sn in SCEN:
    base = R[sn]["No spray"]["reward"]
    for nme in ("Tuned threshold", "Context rule (strong)"):
        r = R[sn][nme]; rows.append(dict(scenario=sn, group=nme, gain=(r["reward"] - base).mean(), seed_spread=0.0, yield_pct=r["yield_pct"].mean(), pesticide=r["pesticide"].mean()))
    for g, paths in GROUPS.items():
        rs = [R[sn][f"{g} #{i}"] for i in range(len(paths))]
        gains = np.array([(r["reward"] - base).mean() for r in rs])
        rows.append(dict(scenario=sn, group=g, gain=gains.mean(), seed_spread=gains.std(ddof=1), yield_pct=np.mean([r["yield_pct"].mean() for r in rs]), pesticide=np.mean([r["pesticide"].mean() for r in rs])))
df = pd.DataFrame(rows); df.round(2).to_csv("outputs/step10_shaping.csv", index=False)
for sn in SCEN:
    print(f"\n=== {sn} ===")
    print(df[df.scenario == sn].drop(columns="scenario").round(2).to_string(index=False))

# %% 5. Reading the result
print("\nIf shaping 0.0 / 0.1 give LOWER pesticide and a HIGHER gain than shaping 0.3, the shaping term was making PPO overspend.")
print("If they are no better (or learn worse), shaping is not the cause - then the gap is about PPO's training, not the reward.")
print("Compare 'seed_spread' with the differences: with only 2 seeds per setting, small differences are not reliable.")

# %% 6. Picture
groups = ["Tuned threshold", "Context rule (strong)"] + list(GROUPS)
cols = ["#4f9d7f", "#2f6f9f", "#5b3fa8", "#8f78d4", "#c2b5ea"]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6)); wd = 0.8 / len(groups)
for a, (col, title) in zip(ax, [("gain", "Reward gain vs no spray"), ("pesticide", "Pesticide used (units)")]):
    for i, g in enumerate(groups):
        d = [df[(df.scenario == s) & (df.group == g)].iloc[0] for s in SCEN]
        a.bar(np.arange(len(SCEN)) + i * wd, [x[col] for x in d], wd, color=cols[i], label=g,
              yerr=[x.seed_spread for x in d] if col == "gain" else None, capsize=2)
    a.set_xticks(np.arange(len(SCEN)) + 0.4 - wd / 2); a.set_xticklabels([s.replace(", ", "\n") for s in SCEN], fontsize=8)
    a.axhline(0, color="k", lw=0.7); a.set_title(title, fontsize=10); a.grid(axis="y", alpha=0.25)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=len(groups), frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig("outputs/step10_shaping.png", dpi=130); plt.show()
print("Step 10 finished.")
# Step 9 - A fair, repeatable comparison: stronger baseline + several PPO training seeds.
# Run from the project root. Shift+Enter on each "# %%" cell.  Needs the Step 8 model models/ppo_tank100_seed0.zip.
# Training the two extra seeds takes about 35 minutes in total (cell 4). Already-trained seeds are skipped.

# %% 1. Settings
import sys, time
sys.path.insert(0, ".")
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.weather_api import load_or_fetch
from src.spray_env import SprayEnv, SprayConfig
from src.policies import NoSpray, FixedRate, SmartThreshold, ContextRule, OraclePolicy
from src.evaluation import evaluate
from src.sb3_utils import TrueRewardLogger, SB3Policy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

TANK, SHAPING, STEPS = 100, 0.3, 600_000          # same as Step 8
SEEDS = [0, 1, 2]                                   # seed 0 already trained in Step 8
TUNE_SEEDS, TEST_SEEDS = range(0, 20), range(1000, 1030)
weather, _ = load_or_fetch("Pune", "2026-06-01", "2026-09-30")
bank = dict(np.load("data/tiles/score_bank.npz"))
cut = int(0.7 * len(weather)); TRAIN_W, TEST_W = (0, cut), (cut, len(weather))
for d in ("models", "outputs"): Path(d).mkdir(exist_ok=True)

# %% 2. The baselines: tuned threshold (as in Step 8) and the stronger ContextRule (random search, 100 settings)
cuts_grid = [c for c in itertools.product([0.1, 0.2, 0.3], [0.3, 0.45, 0.6], [0.5, 0.7, 0.9]) if c[0] < c[1] < c[2]]
score = lambda pol: evaluate(pol, bank, weather, TRAIN_W, TUNE_SEEDS, tank=TANK)["reward"].mean()
tuned = SmartThreshold(max(cuts_grid, key=lambda c: score(SmartThreshold(c, 0.5))), 0.5)

rng = np.random.default_rng(0)
cands = [(tuned.cuts, 0.5, 0.0, 0.0)]                                    # include the tuned rule as a candidate
for _ in range(100):
    c1 = rng.uniform(0.05, 0.4); c2 = rng.uniform(c1 + 0.05, 0.8); c3 = rng.uniform(c2 + 0.05, 0.99)
    cands.append(((c1, c2, c3), rng.uniform(0.3, 1.0), rng.uniform(0, 2), rng.uniform(0, 2)))
best = max(cands, key=lambda p: score(ContextRule(*p)))
context = ContextRule(*best)
print("Tuned threshold cuts:", tuned.cuts)
print(f"ContextRule: cuts={tuple(round(c, 2) for c in best[0])}, centre weight={best[1]:.2f}, wind factor={best[2]:.2f}, season factor={best[3]:.2f}")
oracle = OraclePolicy(max(cuts_grid, key=lambda c: score(OraclePolicy(c))))

# %% 3. Environment / model factories
def new_model(seed):
    mk = lambda: SprayEnv(bank, weather, SprayConfig(tank=TANK, shaping=SHAPING), weather_range=TRAIN_W)
    venv = VecNormalize(make_vec_env(mk, n_envs=4, seed=seed), norm_obs=False, norm_reward=True, gamma=0.995)
    return PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=6, gamma=0.995, gae_lambda=0.95,
               clip_range=0.2, ent_coef=0.01, policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])), seed=seed, verbose=0)

# %% 4. Train the missing seeds (slow, ~17 min each). Safe to stop and re-run: finished seeds are skipped.
for s in SEEDS:
    path = Path(f"models/ppo_tank{TANK}_seed{s}.zip")
    if path.exists():
        print(f"seed {s}: model already exists, skipping"); continue
    t0 = time.time(); m = new_model(s)
    m.learn(total_timesteps=STEPS, callback=TrueRewardLogger(f"outputs/step08_curve_seed{s}.csv"))
    m.save(str(path).replace(".zip", "")); print(f"seed {s}: trained in {(time.time() - t0) / 60:.1f} min")

# %% 5. Evaluate everything on the same unseen seasons
SCEN = {"Test weeks, seen patterns": (TEST_W, {}), "Test weeks, unseen edge": (TEST_W, {"patterns": ("edge",)}),
        "Test weeks, bad sensor (20%)": (TEST_W, {"sensor_flip": 0.2}), "Harsh weather (train weeks)": (TRAIN_W, {})}
ppos = [SB3Policy(PPO.load(f"models/ppo_tank{TANK}_seed{s}"), f"PPO seed {s}") for s in SEEDS]
POLS = [NoSpray(), FixedRate(1), tuned, context, oracle] + ppos
R = {sn: {p.name: evaluate(p, bank, weather, wr, TEST_SEEDS, tank=TANK, **kw) for p in POLS} for sn, (wr, kw) in SCEN.items()}

rows = []
for sn in SCEN:
    base = R[sn]["No spray"]["reward"]
    for p in POLS:
        r = R[sn][p.name]
        rows.append(dict(scenario=sn, policy=p.name, gain=(r["reward"] - base).mean(), yield_pct=r["yield_pct"].mean(), pesticide=r["pesticide"].mean()))
df = pd.DataFrame(rows)
for sn in SCEN:
    print(f"\n=== {sn} ===")
    print(df[df.scenario == sn].drop(columns="scenario").round(2).to_string(index=False))

# %% 6. The key question: PPO (average of the seeds) vs the strong rule - and how much do seeds differ?
summary = []
for sn in SCEN:
    base = R[sn]["No spray"]["reward"]
    g = np.array([(R[sn][p.name]["reward"] - base).mean() for p in ppos])
    gc = (R[sn]["Context rule (strong)"]["reward"] - base).mean()
    per_seed_diff = [(R[sn][p.name]["reward"] - R[sn]["Context rule (strong)"]["reward"]).mean() for p in ppos]
    pest = np.mean([R[sn][p.name]["pesticide"].mean() for p in ppos]); pest_c = R[sn]["Context rule (strong)"]["pesticide"].mean()
    verdict = "PPO better (all seeds)" if min(per_seed_diff) > 0.3 else ("PPO worse (all seeds)" if max(per_seed_diff) < -0.3 else "mixed / no clear difference")
    summary.append(dict(scenario=sn, ppo_gain_mean=g.mean(), ppo_gain_std_over_seeds=g.std(ddof=1), context_gain=gc,
                        ppo_minus_context=np.mean(per_seed_diff), pesticide_ppo=pest, pesticide_context=pest_c, verdict=verdict))
S = pd.DataFrame(summary); S.round(2).to_csv("outputs/step09_summary.csv", index=False)
print(S.round(2).to_string(index=False))
print("\nA clear PPO advantage means: all seeds beat the strong rule, and the gap is larger than the spread between seeds.")

# %% 7. Picture
fig, ax = plt.subplots(figsize=(10, 4.8))
names = ["Fixed rate (low)", "Tuned threshold", "Context rule (strong)", "Oracle (cheats)", "PPO (mean of seeds)"]
cols = ["#d9c9a0", "#4f9d7f", "#2f6f9f", "#bbbbdd", "#5b3fa8"]
wd = 0.8 / len(names)
for i, nme in enumerate(names):
    vals, errs = [], []
    for sn in SCEN:
        if nme.startswith("PPO"):
            g = [df[(df.scenario == sn) & (df.policy == p.name)].gain.iloc[0] for p in ppos]; vals.append(np.mean(g)); errs.append(np.std(g, ddof=1))
        else:
            vals.append(df[(df.scenario == sn) & (df.policy == nme)].gain.iloc[0]); errs.append(0)
    ax.bar(np.arange(len(SCEN)) + i * wd, vals, wd, yerr=errs, color=cols[i], label=nme, capsize=3)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(np.arange(len(SCEN)) + 0.4 - wd / 2)
ax.set_xticklabels([s.replace(", ", "\n") for s in SCEN], fontsize=8); ax.set_ylabel("reward gain vs no spray")
ax.set_title("PPO error bars = spread between training seeds"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig("outputs/step09_results.png", dpi=130); plt.show()
print("Step 9 finished.")
# %%

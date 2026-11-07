# Step 11 - The pesticide-vs-yield trade-off and how the conclusions depend on the reward weights.
# NO new training: this uses the models from Steps 8-10.  Run from the project root, Shift+Enter per "# %%" cell.
# Needs src/evaluation.py (updated), the baselines code, and models/ppo_tank100_seed{0,1,2}.zip (+ the shaping-0.1 models from Step 10).

# %% 1. Settings and data
import sys
sys.path.insert(0, ".")
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.weather_api import load_or_fetch
from src.policies import NoSpray, FixedRate, SmartThreshold, ContextRule
from src.evaluation import evaluate
from src.sb3_utils import SB3Policy
from stable_baselines3 import PPO

TANK = 100
TUNE_SEEDS = range(0, 20)
FINAL_SEEDS = range(3000, 3100)        # FRESH seasons: not used for tuning, model choice or any earlier test
weather, _ = load_or_fetch("Pune", "2026-06-01", "2026-09-30")
bank = dict(np.load("data/tiles/score_bank.npz"))
cut = int(0.7 * len(weather)); TRAIN_W, TEST_W = (0, cut), (cut, len(weather))
Path("outputs").mkdir(exist_ok=True)
BASE = dict(w_y=1.0, lam_c=0.10, lam_u=0.15, lam_e=0.15)   # the weights used for training (Step 6)
N_CELLS = 64

# %% 2. Rebuild the two rule-based baselines as in Steps 9-10 (deterministic, ~1-2 minutes)
cuts_grid = [c for c in itertools.product([0.1, 0.2, 0.3], [0.3, 0.45, 0.6], [0.5, 0.7, 0.9]) if c[0] < c[1] < c[2]]
score = lambda pol: evaluate(pol, bank, weather, TRAIN_W, TUNE_SEEDS, tank=TANK)["reward"].mean()
tuned = SmartThreshold(max(cuts_grid, key=lambda c: score(SmartThreshold(c, 0.5))), 0.5)
rng = np.random.default_rng(0)
cands = [(tuned.cuts, 0.5, 0.0, 0.0)]
for _ in range(100):
    c1 = rng.uniform(0.05, 0.4); c2 = rng.uniform(c1 + 0.05, 0.8); c3 = rng.uniform(c2 + 0.05, 0.99)
    cands.append(((c1, c2, c3), rng.uniform(0.3, 1.0), rng.uniform(0, 2), rng.uniform(0, 2)))
context = ContextRule(*max(cands, key=lambda p: score(ContextRule(*p))))

# %% 3. Collect the policies. PPO seeds are kept separate and averaged later.
POLS = {"No spray": [NoSpray()], "Fixed rate (low)": [FixedRate(1)], "Fixed rate (medium)": [FixedRate(2)],
        "Tuned threshold": [tuned], "Context rule (strong)": [context]}
POLS["PPO shaping 0.3"] = [SB3Policy(PPO.load(f"models/ppo_tank{TANK}_seed{s}"), f"ppo0.3_{s}") for s in (0, 1, 2)]
p01 = [f"models/ppo_tank{TANK}_shape0.1_seed{s}_300k" for s in (0, 1)]
if all(Path(p + ".zip").exists() for p in p01):
    POLS["PPO shaping 0.1"] = [SB3Policy(PPO.load(p), f"ppo0.1_{i}") for i, p in enumerate(p01)]

# %% 4. Evaluate on 100 FRESH seasons in four scenarios and keep the cost components of every season
SCEN = {"Test weeks, seen patterns": (TEST_W, {}), "Test weeks, unseen edge": (TEST_W, {"patterns": ("edge",)}),
        "Test weeks, bad sensor (20%)": (TEST_W, {"sensor_flip": 0.2}), "Harsh weather (train weeks)": (TRAIN_W, {})}
R = {}                                    # R[scenario][group] = list of per-model result dicts
for sn, (wr, kw) in SCEN.items():
    R[sn] = {g: [evaluate(p, bank, weather, wr, FINAL_SEEDS, tank=TANK, **kw) for p in pols] for g, pols in POLS.items()}

def reward_under(r, lc, lu, le, wy=1.0):
    """Recompute the season reward from its components under ANY weights (the policies are NOT retrained)."""
    d_yield = (r["yield_pct"] / 100.0 - 1.0) * N_CELLS
    return wy * d_yield - lc * r["pesticide"] - lu * r["unnecessary_units"] - le * r["drift_units"]

# sanity check: recomputing with the training weights must reproduce the stored reward
r0 = R["Test weeks, seen patterns"]["Tuned threshold"][0]
err = np.abs(reward_under(r0, BASE["lam_c"], BASE["lam_u"], BASE["lam_e"]) - r0["reward"]).max()
print(f"Recomputation check (should be ~0): max difference = {err:.4f}")

# %% 5. Table: yield, pesticide, unnecessary spraying (100 fresh seasons)
rows = []
for sn in SCEN:
    for g, rs in R[sn].items():
        rows.append(dict(scenario=sn, policy=g, yield_pct=np.mean([r["yield_pct"].mean() for r in rs]),
                         pesticide=np.mean([r["pesticide"].mean() for r in rs]),
                         unnecessary_units=np.mean([r["unnecessary_units"].mean() for r in rs]),
                         drift_units=np.mean([r["drift_units"].mean() for r in rs]),
                         reward=np.mean([r["reward"].mean() for r in rs])))
T = pd.DataFrame(rows); T.round(2).to_csv("outputs/step11_table.csv", index=False)
for sn in SCEN:
    print(f"\n=== {sn} ===")
    print(T[T.scenario == sn].drop(columns="scenario").round(2).to_string(index=False))

# %% 6. Picture 1: the pesticide-vs-yield trade-off (each dot = one policy; up and to the left is better)
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
markers = {"No spray": "s", "Fixed rate (low)": "^", "Fixed rate (medium)": "v", "Tuned threshold": "D", "Context rule (strong)": "P",
           "PPO shaping 0.3": "o", "PPO shaping 0.1": "o"}
for ax, sn in zip(axes, SCEN):
    for _, row in T[T.scenario == sn].iterrows():
        ax.scatter(row.pesticide, row.yield_pct, s=90, marker=markers.get(row.policy, "o"), label=row.policy, zorder=3)
    ax.set_title(sn, fontsize=9); ax.set_xlabel("pesticide used (units)"); ax.grid(alpha=0.3)
axes[0].set_ylabel("final crop yield (%)"); axes[0].legend(fontsize=7, loc="lower right")
fig.suptitle("Trade-off between pesticide use and crop yield (100 fresh seasons per scenario)")
fig.tight_layout(); fig.savefig("outputs/step11_tradeoff.png", dpi=130); plt.show()

# %% 7. Picture 2: how the ranking changes with the reward weights (policies fixed, NOT retrained)
def gain(sn, g, lc, lu, le):
    base = reward_under(R[sn]["No spray"][0], lc, lu, le)
    return np.mean([(reward_under(r, lc, lu, le) - base).mean() for r in R[sn][g]])

SN = "Test weeks, seen patterns"
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
scales = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
for g in POLS:
    if g == "No spray": continue
    ax[0].plot(scales, [gain(SN, g, BASE["lam_c"] * k, BASE["lam_u"] * k, BASE["lam_e"] * k) for k in scales], marker="o", label=g)
    ax[1].plot(scales, [gain(SN, g, BASE["lam_c"], BASE["lam_u"] * k, BASE["lam_e"]) for k in scales], marker="o", label=g)
for a, t in zip(ax, ["All penalty weights x factor", "Only the unnecessary-spray weight x factor"]):
    a.axhline(0, color="k", lw=0.8); a.axvline(1.0, color="grey", ls=":"); a.set_xscale("symlog", linthresh=0.25)
    a.set_xlabel("factor (1.0 = the weights used in training)"); a.set_ylabel("reward gain vs no spray"); a.set_title(t, fontsize=10); a.grid(alpha=0.3)
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("outputs/step11_weight_sensitivity.png", dpi=130); plt.show()

print("\nWinner among the non-cheating policies at each factor (seen patterns, all penalties scaled):")
for k in scales:
    best = max((g for g in POLS if g != "No spray"), key=lambda g: gain(SN, g, BASE["lam_c"] * k, BASE["lam_u"] * k, BASE["lam_e"] * k))
    print(f"  x{k:<5}: {best}")

# %% 8. Is PPO really different from the strong rule?  Paired comparison on the same 100 fresh seasons
print("PPO (average of its seeds) minus the strong context rule, reward per season, mean +- standard error:")
for sn in SCEN:
    ppo = np.mean([r["reward"] for r in R[sn]["PPO shaping 0.3"]], axis=0)       # per-season average over the PPO seeds
    d = ppo - R[sn]["Context rule (strong)"][0]["reward"]
    se = d.std(ddof=1) / np.sqrt(len(d))
    verdict = "PPO better" if d.mean() - 2 * se > 0 else ("context rule better" if d.mean() + 2 * se < 0 else "no clear difference")
    print(f"  {sn:30s}: {d.mean():+.2f} +- {se:.2f}   -> {verdict}")
print("Step 11 finished.")
# %%

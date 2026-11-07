# Step 7 - Baseline policies in the simulator.  These are what the RL agent must beat.
# Run from the project root. Shift+Enter on each "# %%" cell.
# Needs: src/policies.py, src/evaluation.py, src/spray_env.py (Step 6) and the data from Steps 3 and 5.

# %% 1. Load data and define the evaluation scenarios
import sys
sys.path.insert(0, ".")
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.weather_api import load_or_fetch
from src.policies import NoSpray, FixedRate, Threshold, SmartThreshold, OraclePolicy
from src.evaluation import evaluate, summary

PLACE, START, END = "Pune", "2026-06-01", "2026-09-30"
weather, _ = load_or_fetch(PLACE, START, END)
bank = dict(np.load("data/tiles/score_bank.npz"))
cut = int(0.7 * len(weather))
TRAIN_W, TEST_W = (0, cut), (cut, len(weather))
TUNE_SEEDS, TEST_SEEDS = range(0, 20), range(1000, 1030)       # tuning and testing never share seasons
Path("outputs").mkdir(exist_ok=True)

# each scenario = (weather range, extra environment settings)
SCENARIOS = {
    "Test weeks, seen patterns":   (TEST_W,  {}),
    "Test weeks, unseen edge":     (TEST_W,  {"patterns": ("edge",)}),
    "Test weeks, bad sensor (20%)": (TEST_W,  {"sensor_flip": 0.2}),
    "Harsh weather (train weeks)": (TRAIN_W, {}),
}

# %% 2. Tune the rule-based policies on the TRAINING weather (never on the test weeks)
def tune(make, grid):
    best, best_r = None, -1e9
    for params in grid:
        r = evaluate(make(*params), bank, weather, TRAIN_W, TUNE_SEEDS)["reward"].mean()
        if r > best_r: best, best_r = params, r
    return best, best_r

cuts_grid = [((c1, c2, c3),) for c1, c2, c3 in itertools.product([0.1, 0.2, 0.3], [0.3, 0.45, 0.6], [0.5, 0.7, 0.9]) if c1 < c2 < c3]
smart_grid = [(cuts, w) for (cuts,), w in itertools.product(cuts_grid, [0.3, 0.5, 0.7])]
best_smart, r1 = tune(lambda cuts, w: SmartThreshold(cuts, w), smart_grid)
best_orc, r2 = tune(lambda cuts: OraclePolicy(cuts), cuts_grid)
print(f"Tuned threshold  : cuts={best_smart[0]}, centre weight={best_smart[1]}  (tuning reward {r1:.2f})")
print(f"Oracle (cheating): cuts={best_orc[0]}  (tuning reward {r2:.2f})")

POLICIES = [NoSpray(), FixedRate(2), Threshold(0.5, 3), SmartThreshold(*best_smart), OraclePolicy(*best_orc)]

# %% 3. Evaluate everything on every scenario (30 unseen seasons each)
table, raw = [], {}
for sname, (wr, kw) in SCENARIOS.items():
    res = {p.name: evaluate(p, bank, weather, wr, TEST_SEEDS, **kw) for p in POLICIES}
    raw[sname] = res
    for p in POLICIES:
        s = summary(res[p.name], res["No spray"])
        table.append(dict(scenario=sname, policy=p.name, reward=s["reward"][0], reward_se=s["reward"][1],
                          gain_vs_no_spray=s["gain"][0], gain_se=s["gain"][1], yield_pct=s["yield_pct"][0],
                          pesticide=s["pesticide"][0], unnecessary_pct=s["unnecessary_pct"][0]))
df = pd.DataFrame(table)
df.round(2).to_csv("outputs/step07_baselines.csv", index=False)
for sname in SCENARIOS:
    print(f"\n=== {sname} ===")
    d = df[df.scenario == sname]
    print(d[["policy", "reward", "gain_vs_no_spray", "gain_se", "yield_pct", "pesticide", "unnecessary_pct"]].round(2).to_string(index=False))

# %% 4. Is the problem worth solving? (headroom check before we train RL)
d = df[df.scenario == "Test weeks, seen patterns"].set_index("policy")
g_tuned, g_orc = d.loc["Tuned threshold", "gain_vs_no_spray"], d.loc["Oracle (cheats)", "gain_vs_no_spray"]
print(f"Gain over no-spray - tuned threshold: {g_tuned:+.2f}   oracle (perfect sensor): {g_orc:+.2f}")
print(f"Tuned threshold captures {100 * g_tuned / max(g_orc, 1e-9):.0f}% of the oracle's gain "
      "(the oracle has a perfect sensor, so part of the gap cannot be closed by any policy).")
for sname in SCENARIOS:
    t = df[(df.scenario == sname) & (df.policy == "Tuned threshold")].iloc[0]
    if t.gain_vs_no_spray + 2 * t.gain_se < 0: verdict = "WORSE than no spray"
    elif t.gain_vs_no_spray - 2 * t.gain_se < 0: verdict = "not clearly better than no spray"
    else: verdict = "better than no spray"
    print(f"  {sname:30s}: tuned rule is {verdict} ({t.gain_vs_no_spray:+.2f} +- {t.gain_se:.2f})")

# %% 5. Picture: reward gain over no-spray, yield and pesticide use
names = [p.name for p in POLICIES]
colors = ["#999999", "#c9b37e", "#e0a050", "#4f9d7f", "#bbbbdd"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for a, (col, title) in zip(ax, [("gain_vs_no_spray", "Reward gain vs no spray (higher is better)"),
                                ("yield_pct", "Final crop yield % (higher is better)"),
                                ("pesticide", "Pesticide used (lower is better)")]):
    w = 0.8 / len(names)
    for i, nme in enumerate(names):
        vals = [df[(df.scenario == s) & (df.policy == nme)][col].iloc[0] for s in SCENARIOS]
        errs = [df[(df.scenario == s) & (df.policy == nme)]["gain_se"].iloc[0] if col == "gain_vs_no_spray" else 0 for s in SCENARIOS]
        a.bar(np.arange(len(SCENARIOS)) + i * w, vals, w, yerr=errs, color=colors[i], label=nme, capsize=2)
    a.set_xticks(np.arange(len(SCENARIOS)) + 0.4 - w / 2)
    a.set_xticklabels([s.replace(", ", "\n") for s in SCENARIOS], fontsize=8)
    a.set_title(title, fontsize=10); a.grid(axis="y", alpha=0.25)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(names), frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("outputs/step07_baselines.png", dpi=130)
plt.show()
print("Step 7 finished. Next: Step 8 - train the PPO agent.")
# %%

# Step 7b - How big should the pesticide tank be?  (Decide this BEFORE training the RL agent.)
# With a small tank every policy uses all of it, so "pesticide saved" can never show up.
# Run from the project root. Shift+Enter on each "# %%" cell. Takes a few minutes.

# %% 1. Setup
import sys
sys.path.insert(0, ".")
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.weather_api import load_or_fetch
from src.policies import NoSpray, FixedRate, SmartThreshold, OraclePolicy
from src.evaluation import evaluate, summary

weather, _ = load_or_fetch("Pune", "2026-06-01", "2026-09-30")
bank = dict(np.load("data/tiles/score_bank.npz"))
cut = int(0.7 * len(weather)); TRAIN_W, TEST_W = (0, cut), (cut, len(weather))
TUNE_SEEDS, TEST_SEEDS = range(0, 15), range(1000, 1030)
TANKS = [45, 70, 100, 150]                                   # dose units per season (high dose = 1 unit)
SCEN = {"seen patterns": (TEST_W, {}), "unseen edge": (TEST_W, {"patterns": ("edge",)}), "harsh weather": (TRAIN_W, {})}
Path("outputs").mkdir(exist_ok=True)
cuts_grid = [c for c in itertools.product([0.1, 0.2, 0.3], [0.3, 0.45, 0.6], [0.5, 0.7, 0.9]) if c[0] < c[1] < c[2]]

# %% 2. For each tank size: tune the rules on training weather, then evaluate on unseen seasons
rows = []
for tank in TANKS:
    best = max(cuts_grid, key=lambda c: evaluate(SmartThreshold(c, 0.5), bank, weather, TRAIN_W, TUNE_SEEDS, tank=tank)["reward"].mean())
    best_o = max(cuts_grid, key=lambda c: evaluate(OraclePolicy(c), bank, weather, TRAIN_W, TUNE_SEEDS, tank=tank)["reward"].mean())
    pols = [NoSpray(), FixedRate(2), SmartThreshold(best, 0.5), OraclePolicy(best_o)]
    for sname, (wr, kw) in SCEN.items():
        res = {p.name: evaluate(p, bank, weather, wr, TEST_SEEDS, tank=tank, **kw) for p in pols}
        for p in pols:
            s = summary(res[p.name], res["No spray"])
            rows.append(dict(tank=tank, scenario=sname, policy=p.name, gain=s["gain"][0], gain_se=s["gain"][1],
                             yield_pct=s["yield_pct"][0], pesticide=s["pesticide"][0], unnecessary_pct=s["unnecessary_pct"][0]))
    print(f"tank {tank}: done")
df = pd.DataFrame(rows)
df.round(2).to_csv("outputs/step07b_tank_check.csv", index=False)

# %% 3. Results table
pd.set_option("display.width", 160)
for sname in SCEN:
    print(f"\n=== {sname} ===")
    d = df[df.scenario == sname]
    print(d.pivot(index="tank", columns="policy", values="gain").round(2).add_prefix("gain: ").to_string())
    print(d.pivot(index="tank", columns="policy", values="pesticide").round(1).add_prefix("pesticide: ").to_string())

# %% 4. The two things we want from a tank size
print("\nPesticide saved by the tuned rule compared with fixed-rate spraying, and yield difference (seen patterns):")
d = df[df.scenario == "seen patterns"].pivot(index="tank", columns="policy", values=["pesticide", "yield_pct"])
for tank in TANKS:
    p_t, p_f = d.loc[tank, ("pesticide", "Tuned threshold")], d.loc[tank, ("pesticide", "Fixed rate (medium)")]
    y_t, y_f = d.loc[tank, ("yield_pct", "Tuned threshold")], d.loc[tank, ("yield_pct", "Fixed rate (medium)")]
    print(f"  tank {tank:3d}: tuned uses {p_t:5.1f} vs fixed {p_f:5.1f} units ({100*(1-p_t/p_f):4.0f}% less) | yield {y_t:.1f}% vs {y_f:.1f}%")
print("\nA good tank size: tuned gain > 0 in ALL scenarios, clearly less pesticide than fixed-rate, yield not much lower.")

# %% 5. Picture
fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
for i, sname in enumerate(SCEN):
    for pol, col in (("Fixed rate (medium)", "#c9b37e"), ("Tuned threshold", "#4f9d7f"), ("Oracle (cheats)", "#9999cc")):
        d = df[(df.scenario == sname) & (df.policy == pol)]
        ax[i].errorbar(d.tank, d.gain, d.gain_se, marker="o", color=col, label=pol, capsize=2)
    ax[i].axhline(0, color="k", lw=0.8); ax[i].set_title(f"Gain vs no spray: {sname}", fontsize=9); ax[i].set_xlabel("tank size (units)")
for pol, col in (("Fixed rate (medium)", "#c9b37e"), ("Tuned threshold", "#4f9d7f"), ("Oracle (cheats)", "#9999cc")):
    d = df[(df.scenario == "seen patterns") & (df.policy == pol)]
    ax[3].plot(d.tank, d.pesticide, marker="o", color=col, label=pol)
ax[3].plot(TANKS, TANKS, "k:", lw=0.8, label="tank size (limit)")
ax[3].set_title("Pesticide actually used (seen patterns)", fontsize=9); ax[3].set_xlabel("tank size (units)"); ax[3].legend(fontsize=7)
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("outputs/step07b_tank_check.png", dpi=130); plt.show()
print("Step 7b finished. Send me the tables from cells 3 and 4.")
# %%

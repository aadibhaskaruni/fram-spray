# Step 6 - Build and test the simulator (real weather + real sensor scores + assumed pest dynamics).
# Run from the project root. Shift+Enter on each "# %%" cell.
# Needs: src/spray_env.py, src/weather_api.py, data/tiles/score_bank.npz (Step 5), data/weather/*.csv (Step 3)

# %% 1. Load the real data
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.weather_api import load_or_fetch
from src.spray_env import SprayEnv, SprayConfig

PLACE, START, END = "Pune", "2026-06-01", "2026-09-30"        # same as Step 3 (change PLACE to test another location)
weather, info = load_or_fetch(PLACE, START, END)
bank = dict(np.load("data/tiles/score_bank.npz"))
cut = int(0.7 * len(weather))
TRAIN_RANGE, TEST_RANGE = (0, cut), (cut, len(weather))       # train on earlier weeks, test on LATER, unseen weather
print(f"Weather: {info['name']}, {len(weather)} hours | train hours {TRAIN_RANGE}, test hours {TEST_RANGE}")
print({k: len(v) for k, v in bank.items() if k in ('healthy', 'rust')}, "real sensor scores in the bank")
Path("outputs").mkdir(exist_ok=True)

# %% 2. Build the environment and let Gymnasium's built-in checker validate it
from gymnasium.utils.env_checker import check_env
env = SprayEnv(bank, weather, SprayConfig(), weather_range=TRAIN_RANGE)
check_env(env, skip_render_check=True)
print("Gymnasium check passed")
print("Observation size:", env.observation_space.shape, "| actions:", env.action_space.n, "| steps per season:", env.T)

# %% 3. Watch the first steps of one season (what the agent sees, what the weather is, what is blocked)
obs, _ = env.reset(seed=1)
print(f"pattern: {env.pattern}   weather window starts at hour {env.w_start}")
print(f"{'t':>3} {'cell':>7} {'sensor(centre)':>14} {'true p':>7} {'wind':>5} {'rain':>5} {'dose':>5} {'reward':>7}")
for _ in range(10):
    r_, c_ = env.route[env.t]
    seen, true_p = obs[-13], env.p[r_, c_]
    obs, reward, done, _, inf = env.step(3)                       # always ask for HIGH dose
    print(f"{env.t:3d} {str((r_, c_)):>7} {seen:14.2f} {true_p:7.2f} {inf['wind']:5.1f} {inf['rain']:5.1f} {inf['dose']:5.2f} {reward:7.3f}")
print("dose 0.00 on a row means the hard limits blocked the spray (wind, rain or empty tank).")

# %% 4. Sanity check: simple policies over 20 seasons (they should behave sensibly)
def run(policy, seeds=range(20), **kw):
    e = SprayEnv(bank, weather, SprayConfig(**kw), weather_range=TRAIN_RANGE)
    rows = []
    for s in seeds:
        obs, _ = e.reset(seed=s); total, done = 0.0, False
        while not done:
            obs, _, done, _, inf = e.step(policy(obs)); total += e.true_reward(inf)
        rows.append((total, 100 * inf["yield_"], inf["pesticide"], sum(inf["blocked"].values())))
    return np.mean(rows, axis=0)

results = {"No spray": run(lambda o: 0), "Always HIGH": run(lambda o: 3), "Always MEDIUM": run(lambda o: 2),
           "Random": run(lambda o: np.random.randint(4))}
print(f"{'policy':16s} {'reward':>8} {'yield %':>8} {'pesticide':>10} {'blocked steps':>14}")
for k, v in results.items():
    print(f"{k:16s} {v[0]:8.2f} {v[1]:8.1f} {v[2]:10.1f} {v[3]:14.1f}")
print("Expected: No spray has the lowest yield; spraying raises yield; the tank (45 units) limits how long spraying lasts.")

# %% 5. Picture: how the TRUE infestation evolves with and without spraying (same field, same weather)
def episode_maps(policy, seed=3):
    e = SprayEnv(bank, weather, SprayConfig(), weather_range=TRAIN_RANGE)
    obs, _ = e.reset(seed=seed); maps, done = {0: e.p.copy()}, False
    while not done:
        obs, _, done, _, inf = e.step(policy(obs))
        if e.t in (e.T // 2, e.T): maps[e.t] = e.p.copy()
    return maps, 100 * inf["yield_"], inf["pesticide"]

fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
for row, (name, pol) in enumerate([("No spray", lambda o: 0), ("Always HIGH", lambda o: 3)]):
    maps, y, pest = episode_maps(pol)
    for col, (t, m) in enumerate(maps.items()):
        axes[row, col].imshow(m, cmap="YlOrRd", vmin=0, vmax=1); axes[row, col].axis("off")
        axes[row, col].set_title(f"{name}: step {t}" + (f"  (yield {y:.0f}%, pesticide {pest:.0f})" if col == 2 else ""), fontsize=8)
fig.suptitle("True infestation (hidden from the agent) in the simulator", fontsize=11)
fig.savefig("outputs/step06_simulator.png", dpi=130, bbox_inches="tight")
plt.show()

# %% 6. How much does the real weather limit spraying? (train vs later test weeks)
for name, (a, b) in (("train weeks", TRAIN_RANGE), ("test weeks ", TEST_RANGE)):
    w = weather.iloc[a:b]
    windy, rainy = w.wind_kmh > env.cfg.wind_limit_kmh, w.rain_mm > env.cfg.rain_block_mm
    print(f"{name}: {len(w)} h | windy {100*windy.mean():.0f}% | rainy {100*rainy.mean():.0f}% | sprayable {100*(~windy & ~rainy).mean():.0f}%")
print("Step 6 finished. Next: Step 7 - baseline policies.")
# %%

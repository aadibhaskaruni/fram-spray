"""Run seasons and summarise results. The SAME seed gives the SAME field, weather and sensor randomness for every
policy, so policies can be compared one-to-one (paired comparison)."""
from __future__ import annotations

import numpy as np

from src.spray_env import SprayEnv, SprayConfig

KEYS = ("reward", "yield_pct", "pesticide", "unnecessary_pct", "blocked", "unnecessary_units", "drift_units")


def run_season(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    total, drift, done = 0.0, 0.0, False
    while not done:
        obs, _, done, _, info = env.step(policy.act(obs, env))
        total += env.true_reward(info)
        drift += info["E"]
    return dict(reward=total, unnecessary_units=info["unnecessary"], drift_units=drift, yield_pct=100 * info["yield_"], pesticide=info["pesticide"],
                unnecessary_pct=100 * info["unnecessary"] / max(info["pesticide"], 1e-9),
                blocked=float(sum(info["blocked"].values())))


def evaluate(policy, bank, weather, weather_range, seeds, **cfg_kw):
    """Returns {metric: array over seeds}."""
    env = SprayEnv(bank, weather, SprayConfig(**cfg_kw), weather_range=weather_range)
    rows = [run_season(env, policy, s) for s in seeds]
    return {k: np.array([r[k] for r in rows]) for k in KEYS}


def summary(res, base=None):
    """mean, standard error and (if base given) paired reward gain +- standard error."""
    n = len(res["reward"])
    out = {k: (float(res[k].mean()), float(res[k].std(ddof=1) / np.sqrt(n))) for k in KEYS}
    if base is not None:
        d = res["reward"] - base["reward"]
        out["gain"] = (float(d.mean()), float(d.std(ddof=1) / np.sqrt(n)))
    return out
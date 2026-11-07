"""Helpers for Stable-Baselines3: a logger for the TRUE (unshaped) episode reward and a policy wrapper
so trained agents can be evaluated with the same code as the rule-based policies."""
from __future__ import annotations

import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrueRewardLogger(BaseCallback):
    """Writes one CSV row per finished training episode: step, true reward, yield %, pesticide used."""

    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path, self.rows, self.cur = csv_path, [], None

    def _flush(self):
        with open(self.csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "episode_reward", "yield_pct", "pesticide"])
            w.writerows(self.rows)

    def _on_step(self) -> bool:
        infos, dones = self.locals["infos"], self.locals["dones"]
        if self.cur is None:
            self.cur = np.zeros(len(infos))
        for i, info in enumerate(infos):
            self.cur[i] += info["true_reward"]
            if dones[i]:
                self.rows.append((self.num_timesteps, float(self.cur[i]), 100 * info["yield_"], info["pesticide"]))
                self.cur[i] = 0.0
                if len(self.rows) % 100 == 0:
                    self._flush()                      # keep the file up to date in case training is interrupted
        return True

    def _on_training_end(self) -> None:
        self._flush()


class SB3Policy:
    """Lets a trained Stable-Baselines3 model be used like the rule-based policies (act(obs, env))."""

    def __init__(self, model, name="PPO"):
        self.model, self.name = model, name

    def act(self, obs, env=None):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
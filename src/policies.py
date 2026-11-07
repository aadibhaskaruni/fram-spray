"""Rule-based spraying policies. All of them see ONLY the observation, like the RL agent -
except OraclePolicy, which peeks at the true field and is only a reference (not achievable).

The current step's observation is the LAST 13 numbers of the stacked observation:
[sense_centre, sense_up, sense_down, sense_left, sense_right, health, wind, rain, temp, tank, time, row, col]
(a sensor value of -1 means 'outside the field')
"""
from __future__ import annotations

import numpy as np

OBS = 13


def now(obs):
    return np.asarray(obs)[-OBS:]


class NoSpray:
    name = "No spray"

    def act(self, obs, env=None):
        return 0


class FixedRate:
    """Spray the same dose everywhere (until the tank is empty)."""

    def __init__(self, level=2):
        self.level = level
        self.name = f"Fixed rate ({['none', 'low', 'medium', 'high'][level]})"

    def act(self, obs, env=None):
        return self.level


class Threshold:
    """Spray HIGH when the sensor at the current cell reads 'diseased'."""

    def __init__(self, thr=0.5, level=3):
        self.thr, self.level, self.name = thr, level, "Threshold"

    def act(self, obs, env=None):
        return self.level if now(obs)[0] > self.thr else 0


class SmartThreshold:
    """Neighbour-aware score (centre + average of the 4 neighbours) mapped to doses by 3 cut-offs."""

    def __init__(self, cuts=(0.2, 0.4, 0.6), w=0.5):
        self.cuts, self.w, self.name = tuple(cuts), w, "Tuned threshold"

    def act(self, obs, env=None):
        o = now(obs)
        nb = o[1:5]
        nb = nb[nb >= 0]
        score = self.w * o[0] + (1 - self.w) * (nb.mean() if len(nb) else o[0])
        return int(sum(score > c for c in self.cuts))


class OraclePolicy:
    """REFERENCE ONLY: uses the TRUE severity of the current cell (a perfect sensor). Shows the headroom."""

    def __init__(self, cuts=(0.2, 0.4, 0.6)):
        self.cuts, self.name = tuple(cuts), "Oracle (cheats)"

    def act(self, obs, env=None):
        r, c = env.route[min(env.t, env.T - 1) % (env.n * env.n)]
        p = env.p[r, c]
        return int(sum(p > c_ for c_ in self.cuts))


class ContextRule:
    """A STRONGER rule-based opponent. Like SmartThreshold, but the cut-offs are scaled up when it is windy
    (drift costs more) and late in the season (less time left to benefit): cut * (1 + a*wind + b*time)."""

    def __init__(self, cuts=(0.2, 0.4, 0.6), w=0.5, a=0.0, b=0.0):
        self.cuts, self.w, self.a, self.b, self.name = tuple(cuts), w, a, b, "Context rule (strong)"

    def act(self, obs, env=None):
        o = now(obs)
        nb = o[1:5]
        nb = nb[nb >= 0]
        score = self.w * o[0] + (1 - self.w) * (nb.mean() if len(nb) else o[0])
        scale = 1.0 + self.a * min(float(o[6]), 1.0) + self.b * float(o[10])      # o[6] = wind, o[10] = time in season
        return int(sum(score > c * scale for c in self.cuts))
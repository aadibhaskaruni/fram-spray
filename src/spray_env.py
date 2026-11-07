"""Field spraying simulator (Gymnasium environment) built from the project's real data.

REAL data used
  * weather : hourly temperature / rain / wind from Open-Meteo (Step 3)
  * sensor  : real classifier scores of held-out drone-image cells (Step 5 score bank)
ASSUMED (not fitted to data - calibrate from agronomy literature before claiming realism)
  * pest growth / spread / damage rates, pesticide dose-response, residue decay,
    temperature and rain effect on growth, initial infestation patterns

Time        : one sweep over the n x n field = one day of real weather (24 hourly values spread over the sweep).
Agent       : a sprayer on a fixed serpentine route; at each step it is at one cell and picks a dose.
Actions     : 0 none, 1 low, 2 medium, 3 high.
True state  : p[cell]  = fraction of the cell infected (0..1)    h[cell] = crop health    residue[cell] = pesticide left
Observation : the sensor looks at the current cell + 4 neighbours. For a cell with severity p it sees "rust" with
              probability p (else "healthy"), and the value reported is a REAL score drawn from the score bank.
              Last `history` observations are stacked.
Hard limits : spraying is blocked (action forced to "none") if wind > wind_limit, if it is raining, or if the tank
              cannot supply the dose.
Reward      : r = w_y*dYield - lam_c*cost - lam_u*unnecessary - lam_e*drift
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces

DOSES = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])


@dataclass
class SprayConfig:
    n: int = 8
    sweeps: int = 3                      # season length in days (one sweep per day)
    tank: float = 45.0                   # dose units available per season (high dose = 1 unit)
    patterns: tuple = ("clustered", "random")
    history: int = 4
    wind_limit_kmh: float = 20.0         # ASSUMPTION - set from the pesticide label / local guidance
    rain_block_mm: float = 0.2           # no spraying when hourly rain exceeds this
    sensor_flip: float = 0.0             # extra sensor degradation: probability of reading the wrong class
    # reward weights
    w_y: float = 1.0
    lam_c: float = 0.10
    lam_u: float = 0.15
    lam_e: float = 0.15
    tau: float = 0.15                    # severity below this counts as "no real infestation"
    shaping: float = 0.0                 # optional potential-based shaping (training only); keep 0 for reporting
    gamma: float = 0.995
    # dynamics (ASSUMED)
    growth: float = 0.022
    spread: float = 0.012
    damage: float = 0.004
    kill: float = 3.0                    # kill fraction = 1 - exp(-kill*dose)  (diminishing returns)
    residue_decay: float = 0.985
    rain_wash: float = 0.4               # residue multiplier when it rains
    temp_opt: float = 25.0               # growth is fastest near this temperature (ASSUMPTION)
    temp_width: float = 8.0
    rain_growth: float = 1.3             # growth multiplier while it rains (leaf wetness) (ASSUMPTION)


def serpentine(n: int):
    return [(r, c) for r in range(n) for c in (range(n) if r % 2 == 0 else range(n - 1, -1, -1))]


def neighbour_mean(p: np.ndarray) -> np.ndarray:
    q = np.pad(p, 1, mode="edge")
    return (q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:]) / 4.0


class SprayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, bank: dict, weather=None, cfg: SprayConfig | None = None, weather_range=None):
        """bank: {'healthy': scores, 'rust': scores}.  weather: DataFrame with temp_c, rain_mm, wind_kmh (hourly) or None.
        weather_range: (first_hour, last_hour) of the weather table episodes may start in (use it for a train/test split in time)."""
        super().__init__()
        self.cfg = cfg or SprayConfig()
        c = self.cfg
        self.bank_h = np.asarray(bank["healthy"], dtype=np.float32)
        self.bank_r = np.asarray(bank["rust"], dtype=np.float32)
        self.weather, self.weather_range = weather, weather_range
        self.n, self.T = c.n, c.n * c.n * c.sweeps
        self.route = serpentine(c.n)
        self.obs_per_step = 13
        self.observation_space = spaces.Box(-1.0, 2.0, shape=(self.obs_per_step * c.history,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    # ---------------------------------------------------------------- field set-up
    def _init_pests(self, pattern: str) -> np.ndarray:
        n, rng = self.n, self.np_random
        p = rng.uniform(0.0, 0.02, size=(n, n))
        yy, xx = np.mgrid[0:n, 0:n]
        if pattern == "clustered":
            for _ in range(int(rng.integers(1, 4))):
                cy, cx = rng.uniform(0, n - 1, size=2)
                amp, sig = rng.uniform(0.25, 0.6), rng.uniform(0.9, 1.8)
                p += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2))
        elif pattern == "random":
            k = int(rng.integers(5, 12))
            p.flat[rng.choice(n * n, size=k, replace=False)] += rng.uniform(0.2, 0.5, size=k)
        elif pattern == "edge":
            band = np.zeros((n, n)); side = int(rng.integers(0, 4))
            if side == 0: band[0, :] = 1
            elif side == 1: band[-1, :] = 1
            elif side == 2: band[:, 0] = 1
            else: band[:, -1] = 1
            p += band * rng.uniform(0.15, 0.35, size=(n, n))
        return np.clip(p, 0.0, 1.0)

    def _make_weather(self):
        c, rng = self.cfg, self.np_random
        if self.weather is None:                                   # simple synthetic fallback (for tests)
            t = np.arange(self.T)
            self.w_temp = 25 + 3 * np.sin(2 * np.pi * (t % 64) / 64) + rng.normal(0, 0.5, self.T)
            self.w_rain = np.where(rng.random(self.T) < 0.1, rng.exponential(2.0, self.T), 0.0)
            self.w_wind = np.clip(12 + rng.normal(0, 5, self.T), 0, None)
            self.w_start = -1
            return
        hours = 24 * c.sweeps
        lo, hi = self.weather_range or (0, len(self.weather))
        if hi - lo < hours:
            raise ValueError(f"weather_range too short: need {hours} hours, got {hi - lo}")
        start = int(rng.integers(lo, hi - hours + 1))
        idx = start + (np.arange(self.T) * hours) // self.T
        self.w_temp = self.weather["temp_c"].to_numpy()[idx]
        self.w_rain = self.weather["rain_mm"].to_numpy()[idx]
        self.w_wind = self.weather["wind_kmh"].to_numpy()[idx]
        self.w_start = start

    # ---------------------------------------------------------------- sensing
    def _sense(self, r: int, c: int) -> float:
        if not (0 <= r < self.n and 0 <= c < self.n):
            return -1.0                                            # outside the field
        rng = self.np_random
        is_rust = rng.random() < self.p[r, c]
        if rng.random() < self.cfg.sensor_flip:
            is_rust = not is_rust
        bank = self.bank_r if is_rust else self.bank_h
        return float(bank[rng.integers(len(bank))])

    def _read(self, r: int, c: int) -> np.ndarray:
        cfg, t = self.cfg, min(self.t, self.T - 1)
        o = [self._sense(r, c), self._sense(r - 1, c), self._sense(r + 1, c), self._sense(r, c - 1), self._sense(r, c + 1),
             float(self.h[r, c]),
             float(np.clip(self.w_wind[t] / cfg.wind_limit_kmh, 0, 2)),
             float(self.w_rain[t] > cfg.rain_block_mm),
             float(np.clip((self.w_temp[t] - 25.0) / 10.0, -1, 1)),
             self.tank / cfg.tank,
             self.t / self.T,
             r / (self.n - 1), c / (self.n - 1)]
        return np.array(o, dtype=np.float32)

    def _stack(self):
        return np.concatenate(self._hist).astype(np.float32)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.pattern = options.get("pattern") or str(self.np_random.choice(self.cfg.patterns))
        self.p = self._init_pests(self.pattern)
        self.h = np.ones((self.n, self.n))
        self.res = np.zeros((self.n, self.n))
        self.tank, self.t = self.cfg.tank, 0
        self.total_dose = self.unnecessary = 0.0
        self.blocked = dict(wind=0, rain=0, tank=0)
        self._make_weather()
        first = self._read(*self.route[0])
        self._hist = [first.copy() for _ in range(self.cfg.history)]
        return self._stack(), self._info()

    def _info(self):
        return dict(yield_=float(self.h.mean()), pesticide=self.total_dose, unnecessary=self.unnecessary,
                    tank=self.tank, pattern=self.pattern, blocked=dict(self.blocked))

    def step(self, action: int):
        cfg, t = self.cfg, self.t
        r, c = self.route[t % (self.n * self.n)]
        temp, rain, wind = float(self.w_temp[t]), float(self.w_rain[t]), float(self.w_wind[t])
        u = float(DOSES[int(action)])
        if u > 0:                                                  # hard constraints
            if wind > cfg.wind_limit_kmh: u = 0.0; self.blocked["wind"] += 1
            elif rain > cfg.rain_block_mm: u = 0.0; self.blocked["rain"] += 1
            elif u > self.tank + 1e-9: u = 0.0; self.blocked["tank"] += 1
        p_here, load_before = float(self.p[r, c]), float(self.p.sum())

        if u > 0:                                                  # spray
            self.p[r, c] *= float(np.exp(-cfg.kill * u))
            self.res[r, c] = min(1.0, self.res[r, c] + u)
            self.tank -= u; self.total_dose += u
            if p_here < cfg.tau: self.unnecessary += u

        y0 = self.h.sum()                                          # one time step of field dynamics
        temp_f = 0.5 + 0.5 * np.exp(-((temp - cfg.temp_opt) / cfg.temp_width) ** 2)
        g = cfg.growth * temp_f * (cfg.rain_growth if rain > cfg.rain_block_mm else 1.0) * (1.0 - 0.85 * np.clip(self.res, 0, 1))
        self.p = np.clip(self.p + (g * self.p + cfg.spread * neighbour_mean(self.p)) * (1.0 - self.p), 0.0, 1.0)
        self.h = np.clip(self.h - cfg.damage * self.p, 0.0, 1.0)
        self.res *= cfg.residue_decay
        if rain > cfg.rain_block_mm: self.res *= cfg.rain_wash
        dY = float(self.h.sum() - y0)

        C = u
        U = u if p_here < cfg.tau else 0.0
        E = u * float(np.clip(wind / cfg.wind_limit_kmh, 0, 1))
        reward = cfg.w_y * dY - cfg.lam_c * C - cfg.lam_u * U - cfg.lam_e * E
        if cfg.shaping > 0:
            reward += cfg.shaping * (load_before - cfg.gamma * float(self.p.sum()))

        self.t += 1
        done = self.t >= self.T
        if not done:
            self._hist.pop(0); self._hist.append(self._read(*self.route[self.t % (self.n * self.n)]))
        info = self._info(); info.update(dY=dY, C=C, U=U, E=E, cell=(r, c), dose=u, wind=wind, rain=rain, temp=temp,
                    true_reward=cfg.w_y * dY - cfg.lam_c * C - cfg.lam_u * U - cfg.lam_e * E)   # reward WITHOUT shaping
        return self._stack(), float(reward), done, False, info

    def true_reward(self, info) -> float:
        c = self.cfg
        return c.w_y * info["dY"] - c.lam_c * info["C"] - c.lam_u * info["U"] - c.lam_e * info["E"]
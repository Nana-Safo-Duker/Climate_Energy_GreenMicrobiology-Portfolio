from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass(frozen=True)
class Tariff:
    """Simple time-varying electricity price model (toy)."""

    base_price: float = 0.20
    peak_multiplier: float = 2.0

    def price(self, t: int, steps_per_day: int) -> float:
        # Peak from ~17:00–21:00 in this toy schedule
        hour = (24.0 * (t % steps_per_day)) / steps_per_day
        is_peak = 17 <= hour < 21
        return self.base_price * (self.peak_multiplier if is_peak else 1.0)


class MultiEnergyToyEnv(gym.Env):
    """
    A minimal multi-energy (electric + thermal) toy environment.

    State: [soc, indoor_temp, pv, elec_load, ambient_temp, price]
    Action: [battery_power, heat_pump_power]
      - battery_power: (-1..1) scaled to +/- battery_max_kw (positive = discharge to serve load)
      - heat_pump_power: (0..1) scaled to heat_pump_max_kw (electric power consumption)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        episode_days: int = 7,
        steps_per_day: int = 96,  # 15-min steps
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.episode_days = int(episode_days)
        self.steps_per_day = int(steps_per_day)
        self.episode_steps = self.episode_days * self.steps_per_day
        self.rng = np.random.default_rng(seed)

        # Device parameters (toy)
        self.battery_capacity_kwh = 10.0
        self.battery_max_kw = 4.0
        self.heat_pump_max_kw = 3.0
        self.cop = 3.0  # coefficient of performance

        # Thermal dynamics (very simplified)
        self.temp_min = 19.0
        self.temp_max = 24.0
        self.temp_init = 21.0
        self.thermal_loss_per_step = 0.03
        self.thermal_gain_per_kw = 0.08

        self.tariff = Tariff()

        self.observation_space = spaces.Box(
            low=np.array([0.0, -50.0, 0.0, 0.0, -50.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 50.0, 10.0, 10.0, 50.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._t = 0
        self._soc = 0.5
        self._indoor_temp = self.temp_init

    def _profile_pv(self, t: int) -> float:
        # Simple bell-shaped PV profile with noise
        x = (t % self.steps_per_day) / self.steps_per_day
        pv = 4.0 * np.exp(-((x - 0.5) / 0.18) ** 2)
        return float(max(0.0, pv + self.rng.normal(0.0, 0.15)))

    def _profile_load(self, t: int) -> float:
        # Morning/evening load peaks with noise
        x = (t % self.steps_per_day) / self.steps_per_day
        morning = 1.8 * np.exp(-((x - 0.30) / 0.10) ** 2)
        evening = 2.2 * np.exp(-((x - 0.75) / 0.12) ** 2)
        base = 0.8
        load = base + morning + evening
        return float(max(0.2, load + self.rng.normal(0.0, 0.10)))

    def _profile_ambient(self, t: int) -> float:
        # Daily sinusoid plus noise
        x = (t % self.steps_per_day) / self.steps_per_day
        ambient = 10.0 + 7.0 * np.sin(2 * np.pi * (x - 0.25))
        return float(ambient + self.rng.normal(0.0, 0.5))

    def _get_obs(self) -> np.ndarray:
        pv = self._profile_pv(self._t)
        load = self._profile_load(self._t)
        ambient = self._profile_ambient(self._t)
        price = self.tariff.price(self._t, self.steps_per_day)
        return np.array(
            [self._soc, self._indoor_temp, pv, load, ambient, price], dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self._soc = float(self.rng.uniform(0.3, 0.7))
        self._indoor_temp = float(self.temp_init + self.rng.normal(0.0, 0.3))
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        batt_cmd = float(np.clip(action[0], -1.0, 1.0)) * self.battery_max_kw
        hp_cmd = float(np.clip(action[1], 0.0, 1.0)) * self.heat_pump_max_kw

        obs = self._get_obs()
        _, indoor_temp, pv, load, ambient, price = obs.tolist()

        # Battery energy update (kWh)
        dt_h = 24.0 / self.steps_per_day
        batt_energy_kwh = self._soc * self.battery_capacity_kwh
        batt_energy_kwh = np.clip(
            batt_energy_kwh - batt_cmd * dt_h, 0.0, self.battery_capacity_kwh
        )
        self._soc = float(batt_energy_kwh / self.battery_capacity_kwh)

        # Thermal update: losses to ambient, gains from heat pump (via COP)
        heat_to_building_kw = hp_cmd * self.cop
        temp = indoor_temp
        temp = temp + self.thermal_gain_per_kw * heat_to_building_kw
        temp = temp - self.thermal_loss_per_step * (temp - ambient)
        self._indoor_temp = float(temp)

        # Electricity balance (grid import positive)
        # Battery discharge reduces grid import; charge increases.
        grid_import_kw = load + hp_cmd - pv - batt_cmd
        grid_import_kw = float(max(0.0, grid_import_kw))

        # Reward: energy cost + comfort penalty
        step_cost = grid_import_kw * price * dt_h
        comfort_violation = 0.0
        if temp < self.temp_min:
            comfort_violation = self.temp_min - temp
        elif temp > self.temp_max:
            comfort_violation = temp - self.temp_max

        reward = -(step_cost + 0.50 * comfort_violation)

        self._t += 1
        terminated = self._t >= self.episode_steps
        truncated = False
        info = {
            "grid_import_kw": grid_import_kw,
            "price": price,
            "step_cost": step_cost,
            "comfort_violation": comfort_violation,
        }
        return self._get_obs(), float(reward), terminated, truncated, info


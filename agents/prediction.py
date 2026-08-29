"""
PredictionAgent -- Stage 1 of the maintenance pipeline.

Responsibility: for a set of units, estimate each unit's degradation rate alpha
(bar/day) and, from that, the ORIGINAL time t_n at which a pressure-service
(repressurisation) task first becomes necessary (pressure would fall to the
service-trigger level P_SERV).

Two prediction modes:
  - "real":  alpha is taken from a degradation rate fitted to the unit's real
             pressure trend (the grounded, data-driven mode).
  - "sim":   alpha is drawn from the model's uniform distributions, reproducing
             the simulated degradation (used to reproduce the baseline).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

P_NOM = 3.5
P_SERV = 3.2
P_CRIT = 3.0


class PredictionAgent:
    def __init__(self, mode: str = "real", seed: int = 0,
                 p0_range: tuple = (3.2, 3.5),
                 alpha_range: tuple = (0.01, 0.05),
                 k_range: tuple = (3, 5)):
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.p0_range = p0_range
        self.alpha_range = alpha_range
        self.k_range = k_range

    def predict(self, units: pd.DataFrame,
                horizon: int = 30) -> pd.DataFrame:
        """Produce a pressure-service maintenance schedule (tasks) from the units.

        `units` must have columns: csemCur, (optionally) alpha, p0.
        Returns a DataFrame of tasks with columns:
            tid, unit, t_n, alpha, p0
        """
        tasks = []
        for i, u in units.iterrows():
            alpha = float(u.get("alpha", np.nan))
            if np.isnan(alpha):  # sim mode -> draw alpha
                alpha = float(self.rng.uniform(*self.alpha_range))
            p0 = float(u.get("p0", np.nan))
            if np.isnan(p0):
                p0 = float(self.rng.uniform(*self.p0_range))

            pressure = p0
            day = 0
            while day < horizon:
                pressure -= alpha
                if pressure <= P_SERV:
                    tasks.append({
                        "unit": u["csemCur"],
                        "t_n": day,
                        "alpha": alpha,
                        "p0": p0,
                    })
                    pressure = P_NOM  # a service restores to the nominal pressure within one day
                day += 1
        df = pd.DataFrame(tasks)
        if len(df) == 0:
            return df.assign(
                tid=[], unit=pd.Series([], dtype="int64"),
                t_n=pd.Series([], dtype="int64"),
                alpha=pd.Series([], dtype="float64"),
                p0=pd.Series([], dtype="float64"),
            )
        df = df.reset_index(drop=True)
        df["tid"] = np.arange(len(df))
        return df
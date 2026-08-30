"""
sim_data.py -- a controlled / synthetic maintenance instance, calibrated to the
real fleet's parameters.

We do NOT read the raw monitoring CSV. Instead we generate a linear-degradation
simulation from first principles, so the instance is reproducible and the
behaviour is fully transparent:

  * each unit starts at its nominal pressure P_NOM;
  * each unit has a constant per-unit degradation rate alpha_i (bar/day);
  * pressure falls linearly; when it reaches the service level P_SERV a task is
    triggered (this is the earliest safe service day t_n);
  * the unit's repair type decides how it is restored:
        - Type A (repressurization):  pressure back to P_NOM, rate UNCHANGED
          (short cycle -> frequent service);
        - Type B (seal_replacement):  the leak is reduced (rate halved) but the
          unit is re-armed at the SERVICE level P_SERV -- pressure is NOT
          immediately restored -- and the reduced rate keeps the longer cycle
          (longer cycle -> fewer service visits, but a visit costs more).
  * the repair log is one of two texts (only one chosen per task):
        - "建议采用 A/B 维修方式"  -> the task must be grouped;
        - "本周不建议维修该 unit"   -> the task must NOT be done this week, so it
          is scheduled the next week (deferred).

Because A and B cannot be grouped together (different technicians) and deferred
tasks must wait, the number of clusters is genuinely driven by the
inter-relationship between the repair types -- the thing the framework reasons
about.

Density is calibrated so that each week carries a moderate number of tasks and
the (day, type) clusters are feasible for a conventional capacity.
"""
import numpy as np
import pandas as pd

P_NOM = 3.5
P_SERV = 3.2
P_CRIT = 3.0

REPAIR_A = "A"          # repressurization
REPAIR_B = "B"          # seal_replacement
B_RATE_FACTOR = 0.5     # type B halves the degradation rate


def _make_unit_id(i: int) -> str:
    return f"U{i:03d}"


def build_instance(n_units: int = 611, horizon_days: int = 90,
                   mean_rate: float = 0.009, rate_std: float = 0.003,
                   defer_prob: float = 0.15, seed: int = 0
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the controlled synthetic instance.

    Returns
    -------
    units : DataFrame[unit, group, repair_type, alpha]
    tasks : DataFrame[tid, unit, group, repair_type, t_n, alpha, log, deferred]
    """
    rng = np.random.RandomState(seed)

    # --- per-unit attributes -------------------------------------------------
    units = pd.DataFrame({
        "unit": [_make_unit_id(i) for i in range(n_units)],
        "group": [f"G{i % 8}" for i in range(n_units)],
    })
    # degradation rates: every task must carry enough critical margin to
    # survive a deferred window (deferral can push a task up to ~7 days past
    # its trigger).  days_to_critical = (P_SERV - P_CRIT)/alpha, so we cap
    # alpha at 0.012 -> days_to_critical >= ~16, comfortably covering a
    # deferred window (max ~7 days) plus the advance limit.  The density is
    # calibrated so each week carries a moderate number of tasks.
    units["alpha"] = rng.normal(mean_rate, rate_std, size=n_units).clip(
        min=0.005, max=0.012)
    # each unit has a fixed repair type (consistent across its tasks)
    units["repair_type"] = np.where(
        rng.uniform(0, 1, size=n_units) < 0.5, REPAIR_A, REPAIR_B)

    # --- simulate the task stream per unit ----------------------------------
    tasks = []
    tid = 0
    for u in units.itertuples(index=False):
        p = P_NOM
        day = 0
        # the effective degradation rate after the most recent repair
        alpha_eff = u.alpha
        while day < horizon_days:
            # degrade until we reach P_SERV -> a task is triggered.  The next
            # cycle always takes a full nominal span (P_NOM - P_SERV)/alpha;
            # this also guards the B re-arm (next_p = P_SERV), which would
            # otherwise give (P_SERV - P_SERV)/alpha = 0 and loop forever.
            days_to_serv = max((p - P_SERV) / alpha_eff,
                              (P_NOM - P_SERV) / alpha_eff)
            t_n = day + days_to_serv
            if t_n >= horizon_days:
                break
            # the repair type decides how the unit is restored
            if u.repair_type == REPAIR_A:
                # repressurization: back to P_NOM, rate unchanged
                next_p = P_NOM
                next_alpha = alpha_eff
            else:
                # seal_replacement: the leak is reduced (rate reduced) but the
                # unit is re-armed at the SERVICE level -- pressure is NOT
                # immediately restored -- and keeps the reduced rate.  The
                # rate is floored so the unit still degrades slowly (a perfect
                # seal is not assumed), which also avoids a zero rate.
                next_p = P_SERV
                next_alpha = max(alpha_eff * B_RATE_FACTOR, 0.002)
            # --- the repair log (one of two texts) ---------------------------
            defer = bool(rng.uniform(0, 1) < defer_prob)   # ~15% "not this week"
            if defer:
                log = "本周不建议维修该 unit"
            else:
                log = f"建议采用 {u.repair_type} 维修方式"
            tasks.append({
                "tid": tid, "unit": u.unit, "group": u.group,
                "repair_type": u.repair_type, "t_n": int(round(t_n)),
                "alpha": alpha_eff,
                "log": log, "deferred": defer,
            })
            tid += 1
            # continue the cycle with the (possibly reduced) rate
            p = next_p
            alpha_eff = next_alpha
            day = t_n

    tasks = pd.DataFrame(tasks)
    if tasks.empty:
        tasks = pd.DataFrame(columns=[
            "tid", "unit", "group", "repair_type", "t_n", "alpha",
            "log", "deferred"])
    return units, tasks


if __name__ == "__main__":
    # 204 units, mean degradation rate 0.009 bar/day, -> ~378 service tasks
    # over a 90-day horizon.  This is non-trivial (a few hundred tasks, up
    # to ~26 same-type tasks on a single day) yet stays tractable for the
    # exact ILP (solved in ~20 s at C_max=6) and 0-violation.  (The 611-unit
    # version produces ~1060 tasks and is intractable for CBC, so we reduce
    # the instance to a size the exact solver can handle.)
    units, tasks = build_instance(n_units=204, horizon_days=90,
                                  mean_rate=0.009, rate_std=0.003,
                                  defer_prob=0.15, seed=0)
    print(f"units: {len(units)}  |  tasks: {len(tasks)}  |  horizon: 90")
    print(f"repair types: {tasks['repair_type'].value_counts().to_dict()}")
    print(f"deferred (not-this-week) tasks: {int(tasks['deferred'].sum())}")
    n = len(tasks)
    peak = int(tasks.groupby("t_n")["repair_type"].value_counts().max()) if n else 0
    print(f"peak same-type tasks on a single day: {peak}")
    week = tasks["t_n"].div(7).astype(int).value_counts().sort_index()
    print(f"tasks / week (mean): {week.mean():.1f}")
    tasks.to_csv("/home/ubadmin/projects/AgentOpt/results/sim_tasks.csv",
                 index=False)
    units.to_csv("/home/ubadmin/projects/AgentOpt/results/sim_units.csv",
                 index=False)
    print("-> wrote results/sim_tasks.csv, results/sim_units.csv")
"""Experiment-portfolio simulator and decision rules (ECON 5200 PS2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

def simulate_portfolio(n: int, seed: int, **kwargs):
    """Return the experiment log with the true effect attached."""

    rng = np.random.default_rng(seed)

    share_real = kwargs.get("share_real", 0.20)
    effect_mean = kwargs.get("effect_mean", 0.30)
    effect_sd = kwargs.get("effect_sd", 0.10)
    outcome_sd = kwargs.get("outcome_sd", 1.0)

    real = rng.random(n) < share_real
    true_eff = np.where(
    real, rng.normal(effect_mean, effect_sd, size=n), 0.0)

    sample_n = rng.integers(100, 2001, size=n)
    se = outcome_sd * np.sqrt(2 / sample_n)
    est = true_eff + rng.normal(0, se)

    log = pd.DataFrame({"true": true_eff, "n": sample_n, "se": se, "est": est})

    log["z"] = log["est"] / log["se"]
    log["p"] = 2 * stats.norm.sf(np.abs(log["z"]))

    return log


def apply_rule(log, rule):
    """Ship decisions under `rule`: the p < 0.05 threshold and at least one other."""
    if rule == "p05":
      return log["p"] < 0.05
    elif rule == "p01":
      return log["p"] < 0.01
    else:
      raise ValueError("Unknown rule")


def evaluate(log, rule):
    """Realised true return, ship count and false-ship rate under `rule`."""
    selected = apply_rule(log, rule)
    shipped = log.loc[selected]

    return {"realised_return": float(shipped["true"].sum()), "number_shipped": int(len(shipped)), "false_ship_rate": (float((shipped["true"] == 0).mean())if len(shipped) > 0 else 0.0)}


if __name__ == "__main__":
    log = simulate_portfolio(200, 42)

    assert len(log) == 200
    assert (log["se"] > 0).all()
    assert ((log["p"] >= 0) & (log["p"] <= 1)).all()
    assert apply_rule(log, "p01").sum() <= apply_rule(log, "p05").sum()

    print("Original:", evaluate(log, "p05"))
    print("Alternative:", evaluate(log, "p01"))
    print("All assertions passed.")
    # demonstration plus at least three assertions that must hold

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.evaluation.uncertainty import moving_block_indices


def metrics(curve: pd.DataFrame, orders: pd.DataFrame) -> dict:
    log = curve.log_return.to_numpy()
    simple = np.expm1(log)
    equity = np.r_[1.0, np.exp(np.cumsum(log))]
    dd = equity / np.maximum.accumulate(equity) - 1
    filled = orders.loc[orders.signed_quantity != 0]
    return {
        "days": len(curve),
        "log_return": float(log.sum()),
        "net_return": float(np.expm1(log.sum())),
        "sharpe": float(simple.mean() / simple.std(ddof=1) * np.sqrt(365))
        if simple.std(ddof=1) > 1e-15
        else 0.0,
        "annual_volatility": float(simple.std(ddof=1) * np.sqrt(365)),
        "max_drawdown": float(dd.min()),
        "mean_exposure": float(curve.exposure.mean()),
        "max_exposure": float(curve.exposure.max()),
        "fraction_days_invested": float((curve.exposure > 1e-6).mean()),
        "turnover": float(curve.turnover.sum()),
        "cost_usdt": float(orders.cost.sum()),
        "commission_usdt": float(orders.commission.sum()),
        "spread_slippage_usdt": float(orders.spread_slippage.sum()),
        "filled_orders": len(filled),
        "buy_orders": int((filled.signed_quantity > 0).sum()),
        "sell_orders": int((filled.signed_quantity < 0).sum()),
        "entries": int(((filled.signed_quantity > 0) & (filled.units_before < 1e-5)).sum()),
        "round_trips": int(((filled.signed_quantity < 0) & (filled.units_after < 1e-5)).sum()),
        "terminal_residual_units": float(orders.loc[orders.terminal, "units_after"].sum()),
        "minimum_rejections": int((orders.status == "minimum_order").sum()),
    }


def holm(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[i]))
        adjusted[i] = previous
    return adjusted


def paired_inference(curves: dict, config: dict) -> tuple[list, list]:
    """Shared time resamples; average seeds, never inflate sample size with seeds."""
    inf = config["inference"]
    seeds = config["training"]["seeds"]
    family = len(inf["comparisons"]) * len(inf["scenarios"]) * len(inf["block_days"])
    contrasts, seed_intervals = [], []
    for scenario in inf["scenarios"]:
        agent = np.stack([curves[(scenario, "dqn", seed)].log_return.to_numpy() for seed in seeds])
        groups = curves[(scenario, "dqn", seeds[0])].fold.to_numpy()
        for block in inf["block_days"]:
            idx = moving_block_indices(groups, inf["samples"], block, inf["bootstrap_seed"])
            for reference in inf["comparisons"]:
                baseline = curves[(scenario, reference, 0)].log_return.to_numpy()
                difference = agent.mean(axis=0) - baseline
                point = float(difference.sum())
                draws = difference[idx].sum(axis=1)
                centered = draws - point
                p = float((1 + (centered >= point).sum()) / (1 + len(draws)))
                contrasts.append(
                    {
                        "scenario": scenario,
                        "reference": reference,
                        "block_days": block,
                        "log_gain": point,
                        "ci95_low": float(np.quantile(draws, 0.025)),
                        "ci95_high": float(np.quantile(draws, 0.975)),
                        "simultaneous_low": float(np.quantile(draws, inf["alpha"] / (2 * family))),
                        "simultaneous_high": float(
                            np.quantile(draws, 1 - inf["alpha"] / (2 * family))
                        ),
                        "p_one_sided": p,
                    }
                )
                if reference == "band":
                    for seed, daily in zip(seeds, agent, strict=True):
                        dif = daily - baseline
                        sample = dif[idx].sum(axis=1)
                        seed_intervals.append(
                            {
                                "scenario": scenario,
                                "seed": seed,
                                "block_days": block,
                                "log_gain": float(dif.sum()),
                                "ci95_low": float(np.quantile(sample, 0.025)),
                                "ci95_high": float(np.quantile(sample, 0.975)),
                                "decision_use": False,
                            }
                        )
    for row, adjusted in zip(contrasts, holm([r["p_one_sided"] for r in contrasts]), strict=True):
        row["p_holm_24"] = float(adjusted)
    return contrasts, seed_intervals


def acceptance(curves: dict, contrasts: list, config: dict) -> dict:
    seeds = config["training"]["seeds"]
    base = [r for r in contrasts if r["scenario"] == "base" and r["reference"] == "band"]
    all_band = [r for r in contrasts if r["reference"] == "band"]
    gains = []
    fold_gains = []
    seed_dd = []
    band_curve = curves[("base", "band", 0)]
    band_equity = np.r_[1, np.exp(band_curve.log_return.cumsum())]
    band_dd = float((band_equity / np.maximum.accumulate(band_equity) - 1).min())
    for seed in seeds:
        curve = curves[("base", "dqn", seed)]
        delta = curve.log_return.to_numpy() - band_curve.log_return.to_numpy()
        gains.append(float(delta.sum()))
        fold_gains.append(
            [
                float(delta[curve.fold.to_numpy() == fold].sum())
                for fold in sorted(curve.fold.unique())
            ]
        )
        equity = np.r_[1, np.exp(curve.log_return.cumsum())]
        seed_dd.append(float((equity / np.maximum.accumulate(equity) - 1).min()))
    stable = bool(
        (np.array(gains) > 0).all()
        and (np.array(fold_gains) > 0).sum(axis=1).min() >= 2
        and (np.array(fold_gains) > 0).sum(axis=0).min() >= 2
    )
    checks = {
        "primary_gain_at_least_002": bool(
            base[0]["log_gain"] >= config["inference"]["minimum_total_log_gain"]
        ),
        "base_adjusted_inference": all(
            r["simultaneous_low"] > 0 and r["p_holm_24"] < 0.05 for r in base
        ),
        "stress_adjusted_inference": all(
            r["log_gain"] > 0 and r["simultaneous_low"] > 0 and r["p_holm_24"] < 0.05
            for r in all_band
        ),
        "seed_fold_stability": stable,
        "drawdown_budget": bool(
            min(seed_dd) >= band_dd + config["inference"]["minimum_relative_drawdown"]
        ),
        "positive_net_each_seed": all(
            curves[("base", "dqn", s)].log_return.sum() > 0 for s in seeds
        ),
        "base_beats_cash_adjusted": all(
            r["simultaneous_low"] > 0 and r["p_holm_24"] < 0.05
            for r in contrasts
            if r["scenario"] == "base" and r["reference"] == "cash"
        ),
    }
    if all(checks.values()):
        conclusion = "mejora demostrada dentro del protocolo exploratorio"
    elif base[0]["log_gain"] <= 0 or not stable:
        conclusion = "sin mejora dentro del protocolo"
    else:
        conclusion = "inconcluso"
    return {
        "conclusion": conclusion,
        "exploratory": True,
        "checks": checks,
        "seed_log_gains_vs_band": gains,
        "fold_log_gains_vs_band": fold_gains,
        "independent_economic_advantage_demonstrated": False,
    }


def reconstruct(curve: pd.DataFrame, orders: pd.DataFrame, config: dict, scenario: str) -> dict:
    """Independent signed cash-flow reconstruction, without calling the environment."""
    p = config["portfolio"]
    cash, units = p["initial_capital"], 0.0
    gross_cash = cash
    multiplier = 2 if scenario == "double" else 1
    fee = p["commission_bps"] * multiplier / 10000
    adverse = (p["half_spread_bps"] + p["slippage_bps"]) * multiplier / 10000
    computed = []
    max_error = 0.0
    orders = orders.copy()
    for c in ("execution_time", "signal_time"):
        orders[c] = pd.to_datetime(orders[c], utc=True)
    for i, row in curve.iterrows():
        entry_time = pd.Timestamp(row.entry_time)
        local = orders.loc[(orders.execution_time == entry_time) & ~orders.terminal]
        if i == len(curve) - 1:
            local = pd.concat([local, orders.loc[orders.terminal]])
        price_entry = None
        for o in local.itertuples():
            q = o.signed_quantity
            if o.execution_time < o.signal_time:
                raise AssertionError("Fill precedes decision")
            if (
                scenario == "delay"
                and not o.terminal
                and o.execution_time < o.signal_time + pd.Timedelta(days=1)
            ):
                raise AssertionError("Delayed fill not delayed")
            if not o.terminal:
                price_entry = o.reference_price
            expected_fill = o.reference_price * (1 + (1 if q >= 0 else -1) * adverse)
            # Zero-size rejected orders may carry either side's indicative price.
            if q != 0:
                np.testing.assert_allclose(o.fill_price, expected_fill, atol=1e-9, rtol=1e-12)
                assert abs(q) >= p["minimum_quantity"] - 1e-12
                assert abs(q) * o.fill_price >= p["minimum_notional"] - 1e-8
                np.testing.assert_allclose(
                    abs(q) / p["quantity_step"], round(abs(q) / p["quantity_step"]), atol=1e-6
                )
            commission = abs(q) * o.fill_price * fee
            cost = commission + abs(q) * abs(o.fill_price - o.reference_price)
            np.testing.assert_allclose(
                [o.cash_before, o.units_before, o.commission, o.cost],
                [cash, units, commission, cost],
                atol=1e-7,
                rtol=1e-10,
            )
            cash -= q * o.fill_price + commission
            gross_cash -= q * o.reference_price
            units += q
            np.testing.assert_allclose(
                [o.cash_after, o.units_after], [cash, units], atol=1e-7, rtol=1e-10
            )
            assert cash >= -1e-7 and units >= -1e-10
        # The saved input price is attached to every curve, also for initial cash in delay.
        entry_price = float(row.entry_open)
        exit_price = float(row.exit_open)
        if price_entry is not None:
            np.testing.assert_allclose(price_entry, entry_price, atol=0, rtol=0)
        equity = cash + units * exit_price
        max_error = max(max_error, abs(equity - row.equity))
        np.testing.assert_allclose(
            [row.cash, row.units, row.equity], [cash, units, equity], atol=1e-7, rtol=1e-10
        )
        computed.append(equity)
    log = np.log(np.array(computed) / np.r_[p["initial_capital"], computed[:-1]])
    np.testing.assert_allclose(log, curve.log_return, atol=1e-12, rtol=1e-9)
    return {
        "max_equity_error": max_error,
        "days": len(curve),
        "gross_return_same_units": (gross_cash + units * float(curve.exit_open.iloc[-1]))
        / p["initial_capital"]
        - 1,
    }

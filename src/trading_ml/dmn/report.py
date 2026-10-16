from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.dmn.portfolio import metrics
from trading_ml.evaluation.uncertainty import moving_block_indices


def calendar_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """Omitted fold-edge dates remain cash; don't annualize them out of history."""
    frame = frame.sort_values("reference_time").copy()
    times = pd.date_range(frame.reference_time.min(), frame.reference_time.max(), freq="D")
    result = frame.set_index("reference_time").reindex(times)
    omitted = result.net_return.isna()
    for col in [
        "net_return",
        "gross_return",
        "turnover",
        "transaction_cost",
        "funding_cost",
        "gross_exposure",
        "net_exposure",
    ]:
        result[col] = result[col].fillna(0)
    result["fold"] = result["fold"].ffill().astype(int)
    result["cash_boundary_date"] = omitted
    result["equity"] = (1 + result.net_return).cumprod()
    result.index.name = "reference_time"
    result["exit_time"] = result.index + pd.Timedelta(days=1)
    return result.reset_index()


def paired_intervals(curves: dict[str, pd.DataFrame], candidates: list[str], config: dict) -> dict:
    """Synchronous date blocks preserve cross-asset dependence and paired strategies."""
    reference = curves["tsmom"]
    groups = reference.fold.to_numpy()
    base = reference.net_return.to_numpy()
    out = {}
    for block in config["block_lengths"]:
        ids = moving_block_indices(groups, config["samples"], block, seed=42)
        base_draws = base[ids]
        base_vol = base_draws.std(axis=1, ddof=1)
        base_sharpe = np.divide(
            base_draws.mean(axis=1) * np.sqrt(365),
            base_vol,
            out=np.full(len(ids), np.nan),
            where=base_vol > 1e-12,
        )
        results = {}
        for name in candidates:
            r = curves[name].net_return.to_numpy()
            draws = r[ids]
            vol = draws.std(axis=1, ddof=1)
            sharpe = np.divide(
                draws.mean(axis=1) * np.sqrt(365),
                vol,
                out=np.full(len(ids), np.nan),
                where=vol > 1e-12,
            )
            sharpe_difference = sharpe - base_sharpe
            mean_difference = (draws - base_draws).mean(axis=1) * 365

            def interval(values):
                finite = values[np.isfinite(values)]
                return {
                    "lower": float(np.quantile(finite, 0.025)) if len(finite) else None,
                    "upper": float(np.quantile(finite, 0.975)) if len(finite) else None,
                    "valid_draws": len(finite),
                }

            results[name] = {
                "annual_mean_difference": interval(mean_difference),
                "sharpe_difference": interval(sharpe_difference),
            }
        out[str(block)] = results
    return {
        "reference": "tsmom",
        "blocks": out,
        "samples": config["samples"],
        "confidence": 0.95,
        "method": "paired_date_blocks_within_fold",
        "limitations": "Historical development only; marginal intervals, no multiplicity/post-selection correction.",
    }


def summarize(curves: dict, stressed: dict, folds: list, protocol: dict) -> dict:
    candidates = [r["id"] for r in protocol["recipes"]]
    summary = {}
    for name, curve in curves.items():
        by_fold = [f["models"][name]["net"]["sharpe"] for f in folds]
        gains = [
            f["models"][name]["net"]["total_return"] > f["models"]["tsmom"]["net"]["total_return"]
            for f in folds
        ]
        summary[name] = {
            "net": metrics(curve),
            "double_cost": metrics(stressed[name]),
            "fold_gross_sharpes": [f["models"][name]["gross"]["sharpe"] for f in folds],
            "fold_sharpes": by_fold,
            "fold_return_wins_vs_tsmom": sum(gains),
            "positive_folds": sum(f["models"][name]["net"]["total_return"] > 0 for f in folds),
        }
    return {
        "models": summary,
        "paired_uncertainty": paired_intervals(curves, candidates, protocol["bootstrap"]),
        "historical_development_only": True,
        "independent_advantage_confirmed": False,
        "conclusion": "Comparaciones de desarrollo; requieren confirmación prospectiva nueva y disjunta.",
        "promoted": False,
        "live_trading": False,
        "prospective_evaluated": False,
    }


def markdown(summary: dict, directory: str) -> str:
    lines = [
        "# Experimento Deep Momentum Networks",
        "",
        f"Artefactos: `{directory}`.",
        "",
        "Datos históricos observados: desarrollo, sin confirmación independiente ni promoción.",
        "",
        "| Estrategia | Retorno neto | Sharpe | Drawdown | Turnover/día | Sharpe costo doble | Folds positivos |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def number(x):
        return f"{x:.3f}" if x is not None else "indefinido"

    for name, result in summary["models"].items():
        if "__seed_" in name:
            continue
        net = result["net"]
        lines.append(
            f"| {name} | {net['total_return']:.2%} | {number(net['sharpe'])} | "
            f"{net['max_drawdown']:.2%} | {net['mean_daily_turnover']:.4f} | "
            f"{number(result['double_cost']['sharpe'])} | {result['positive_folds']} |"
        )
    lines += [
        "",
        "Cash incluye los días de frontera purgados; cada fold paga entrada y liquidación.",
        "Buy/hold de control se rebalancea diariamente a pesos iguales; no es compra única pasiva.",
        "Las semillas se reportan individualmente; la fila de receta promedia sus posiciones sin seleccionar una ganadora.",
        "El bootstrap por fechas mantiene dependencia entre activos. Los intervalos son marginales y descriptivos.",
        "No hay ajuste ex post para igualar la volatilidad realizada de las estrategias.",
        "",
    ]
    return "\n".join(lines)

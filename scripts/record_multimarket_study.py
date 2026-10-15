"""Replay saved forecasts/outcomes and write a comparative report without fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trading_ml.common import read_json, sha256, write_new_json
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.multimarket.data import execution_source, load_snapshot
from trading_ml.multimarket.execution import Costs, Engine
from trading_ml.multimarket.features import GROUPS, features, recipes
from trading_ml.multimarket.markets import instrument
from trading_ml.multimarket.service import load_candidate, probabilities
from trading_ml.multimarket.study import bootstrap_skill


def verify(study: Path):
    summary = read_json(study / "summary.json")
    counts = {
        "model_contracts": 0,
        "parent_oof_contracts": 0,
        "forecast_series": 0,
        "replayed_trade_outcomes": 0,
        "controlled_ablation_folds": 0,
    }
    for key in summary["results"]:
        directory = study / key
        asset = read_json(directory / "summary.json")
        raw, meta = load_snapshot(Path(asset["data"]))
        frame = features(raw, asset["symbol"])
        if instrument(asset["symbol"]).market == "crypto":
            frame["volume_ratio"] = raw.volume / raw.volume.rolling(20).mean().replace(0, np.nan)
        execution, multiplier = execution_source(raw, meta)
        engines = {}
        for contract in directory.glob("*_fold*.json"):
            if contract.name.endswith("insufficient.json"):
                continue
            model_file = contract.with_suffix(".ubj")
            _, metadata = load_candidate(model_file)
            assert pd.to_datetime(metadata["last_train_target"], utc=True) < pd.to_datetime(
                metadata["first_oos_reference"], utc=True
            )
            if metadata.get("last_calibration_target"):
                assert pd.to_datetime(
                    metadata["last_calibration_target"], utc=True
                ) < pd.to_datetime(metadata["first_oos_reference"], utc=True)
            if metadata.get("first_calibration_reference"):
                assert pd.to_datetime(metadata["last_train_target"], utc=True) < pd.to_datetime(
                    metadata["first_calibration_reference"], utc=True
                )
            counts["model_contracts"] += 1
        for contract in read_json(directory / "parent_oof_contracts.json")["models"]:
            assert sha256(directory / contract["file"]) == contract["sha256"]
            assert pd.to_datetime(contract["last_target"], utc=True) < pd.to_datetime(
                contract["first_prediction"], utc=True
            )
            counts["parent_oof_contracts"] += 1
        base = pd.read_csv(directory / "direction_price_h1_predictions.csv")
        for candidate in directory.glob("direction_without_*_predictions.csv"):
            other = pd.read_csv(candidate)
            assert base[["source_index", "reference_time", "target_time", "target", "fold"]].equals(
                other[["source_index", "reference_time", "target_time", "target", "fold"]]
            )
            counts["controlled_ablation_folds"] += base.fold.nunique()
        for prediction_file in directory.glob("*_predictions.csv"):
            name = prediction_file.name.removesuffix("_predictions.csv")
            saved = pd.read_csv(prediction_file)
            base_name = name.split("_cal")[0]
            calibrated = "_cal" in name or name.startswith("meta_")
            for fold, part in saved.groupby("fold", sort=True):
                model_name = base_name
                if name in [
                    "always_long",
                    "momentum_rule",
                    "mean_reversion_rule",
                    "logistic_l2",
                    "train_prevalence",
                    "random_probability",
                ]:
                    model_name = "direction_price_h1"
                model_path = directory / f"{model_name}_fold{fold}.ubj"
                metadata = read_json(model_path.with_suffix(".json"))
                local = frame.iloc[part.source_index.to_numpy(int)]
                if name in ["logistic_l2", "train_prevalence", "random_probability"]:
                    p = part[name].to_numpy()
                    if name == "train_prevalence":
                        assert np.allclose(p, metadata["reference_models"]["train_prevalence"])
                    elif name == "logistic_l2":
                        reference = metadata["reference_models"]
                        values = (
                            local[reference["features"]].to_numpy()
                            - np.asarray(reference["scaler_mean"])
                        ) / np.asarray(reference["scaler_scale"])
                        expected = 1 / (
                            1
                            + np.exp(
                                -(
                                    values @ np.asarray(reference["coefficients"])
                                    + reference["intercept"]
                                )
                            )
                        )
                        assert np.allclose(p, expected, atol=1e-12, rtol=0)
                    else:
                        rng = np.random.default_rng(metadata["seed"] + fold)
                        assert np.allclose(
                            p,
                            np.clip(rng.beta(25, 25, len(part)), 1e-6, 1 - 1e-6),
                            atol=1e-12,
                            rtol=0,
                        )
                elif name == "always_long":
                    p = np.ones(len(part))
                elif name == "momentum_rule":
                    p = (local.return_7.to_numpy() > 0).astype(float)
                elif name == "mean_reversion_rule":
                    p = (local.zscore_20.to_numpy() < -1).astype(float)
                else:
                    p, parent = probabilities(model_path, local, calibrated)
                    if parent is not None:
                        p = np.where(parent >= 0.55, p, 0.0)
                assert np.allclose(p, part.probability.to_numpy(), atol=1e-7, rtol=0), (
                    key,
                    name,
                    fold,
                )
                counts["forecast_series"] += 1
                cost_key = tuple(sorted(metadata["costs"].items()))
                if cost_key not in engines:
                    engines[cost_key] = Engine(
                        frame,
                        asset["symbol"],
                        Costs(**metadata["costs"]),
                        execution_frame=execution,
                        bars_per_horizon=multiplier,
                    )
                engine = engines[cost_key]
                trades_path = directory / f"{name}_fold{fold}_trades.csv"
                if trades_path.stat().st_size <= 3:
                    continue
                trades = pd.read_csv(trades_path)
                spec = metadata["specification"]
                for trade in trades.itertuples():
                    event = engine.event(int(trade.source_index), spec["horizon"], spec["barrier"])
                    assert event is not None
                    for column in [
                        "entry_price",
                        "exit_price",
                        "pnl_per_unit",
                        "unit_net_return",
                        "fees_per_unit",
                        "swap_per_unit",
                    ]:
                        assert abs(event[column] - getattr(trade, column)) < 1e-10, (
                            key,
                            name,
                            column,
                        )
                    assert event["reason"] == trade.reason
                    assert np.isclose(
                        trade.profit_usd, trade.units * event["pnl_per_unit"], rtol=1e-10, atol=1e-8
                    )
                    assert (
                        engine.notional(trade.units, trade.entry_price)
                        <= trade.equity_before + 1e-6
                    )
                    counts["replayed_trade_outcomes"] += 1
    return {
        **counts,
        "all_passed": True,
        "new_fits": 0,
        "study_summary_sha256": sha256(study / "summary.json"),
    }


def diagnostics(study, key):
    directory = study / key
    asset = read_json(directory / "summary.json")
    raw, origin = load_snapshot(Path(asset["data"]))
    frame = features(raw, asset["symbol"])
    if instrument(asset["symbol"]).market == "crypto":
        frame["volume_ratio"] = raw.volume / raw.volume.rolling(20).mean().replace(0, np.nan)
    missing = (
        pd.DatetimeIndex(pd.to_datetime(asset["quality"].get("missing_timestamps", []), utc=True))
        .as_unit("ns")
        .asi8
    )
    economic = {}
    for name in asset["experiments"]:
        affected = 0
        total = 0
        exposures = []
        durations = []
        for path in directory.glob(f"{name}_fold*_trades.csv"):
            if path.stat().st_size <= 3:
                continue
            trades = pd.read_csv(path)
            entry = pd.DatetimeIndex(pd.to_datetime(trades.entry_time, utc=True))
            exit = pd.DatetimeIndex(pd.to_datetime(trades.target_time, utc=True))
            count = np.searchsorted(missing, exit.as_unit("ns").asi8) - np.searchsorted(
                missing, entry.as_unit("ns").asi8
            )
            affected += int((count > 0).sum())
            total += len(trades)
            notionals = (
                trades.units
                if instrument(asset["symbol"]).base == "USD"
                else trades.units * trades.entry_price
            )
            exposures.extend((notionals / trades.equity_before).tolist())
            durations.extend(((exit - entry).total_seconds() / 3600).tolist())
        economic[name] = {
            "trades": total,
            "trades_spanning_nominal_quote_gaps": affected,
            "no_trade_crosses_nominal_quote_gaps": affected == 0,
            "mean_entry_notional_fraction": float(np.mean(exposures)) if exposures else None,
            "mean_holding_hours": float(np.mean(durations)) if durations else None,
        }
    execution, multiplier = execution_source(raw, origin)
    model_contract = read_json(directory / "direction_price_h1_fold1.json")
    strict = Engine(
        frame,
        asset["symbol"],
        Costs(**model_contract["costs"]),
        execution_frame=execution,
        bars_per_horizon=multiplier,
        expiry_policy="barrier_first",
    )
    expiry_sensitivity = {}
    for name, data in asset["experiments"].items():
        if not name.startswith("tb_"):
            continue
        prediction = pd.read_csv(directory / f"{name}_predictions.csv")
        threshold = int(name.rsplit("_t", 1)[1]) / 100 if "_t" in name else 0.55
        strict_returns = []
        for fold, part in prediction.groupby("fold", sort=True):
            end = pd.to_datetime(
                next(f["end"] for f in data["folds"] if f["fold"] == fold), utc=True
            )
            result, _, _ = strict.portfolio(part, 3, True, threshold, end)
            strict_returns.append(result["net_return"])
        expiry_sensitivity[name] = {
            "timeout_first_net_return": data["net_return"],
            "strict_tp_priority_net_return": float(np.prod(1 + np.asarray(strict_returns)) - 1),
            "same_forecasts": True,
            "new_fits": 0,
            "purpose": "execution sensitivity only; not a model retargeted or selected on OOS",
        }
    primary = pd.read_csv(directory / "direction_price_h1_predictions.csv")
    labelled_time = frame.reference_time.shift(-1)
    common = recipes(asset["symbol"])["compact"] + (
        ["volume_ratio"] if instrument(asset["symbol"]).market == "crypto" else []
    )
    quantiles = []
    annotated = []
    for fold, part in primary.groupby("fold"):
        contract = read_json(directory / f"direction_price_h1_fold{fold}.json")
        train = frame.loc[
            labelled_time <= pd.to_datetime(contract["last_train_target"], utc=True)
        ].dropna(subset=common)
        assert len(train) == contract["fit_rows"]
        low, high = train.volatility.quantile([0.3, 0.7]).tolist()
        vol = frame.iloc[part.source_index.to_numpy(int)].volatility.to_numpy()
        copy = part.copy()
        copy["volatility_regime"] = np.where(
            vol < low, "low", np.where(vol > high, "high", "middle")
        )
        annotated.append(copy)
        quantiles.append(
            {
                "fold": int(fold),
                "train_rows": len(train),
                "last_train_target": contract["last_train_target"],
                "low_threshold": low,
                "high_threshold": high,
            }
        )
    annotated = pd.concat(annotated, ignore_index=True)
    regimes = {
        name: classification_metrics(part.target.to_numpy(int), part.probability.to_numpy())
        for name, part in annotated.groupby("volatility_regime")
    }
    interval = asset["interval"]
    blocks = {"1d": [7, 30], "4h": [42, 180], "1h": [168, 720]}[interval]
    uncertainty = bootstrap_skill(primary, 500, blocks, 42)
    importance = read_json(directory / "feature_importance.json")["records"]
    families = {}
    for group in GROUPS:
        values = [
            np.mean(r["permutation_loss_increase"]) for r in importance if r["group"] == group
        ]
        path = directory / f"direction_without_{group}_predictions.csv"
        delta = None
        if path.exists():
            without = pd.read_csv(path)
            compact = pd.read_csv(directory / "direction_compact_h1_predictions.csv")
            y = without.target.to_numpy()
            p = np.clip(without.probability.to_numpy(), 1e-12, 1 - 1e-12)
            q = np.clip(compact.probability.to_numpy(), 1e-12, 1 - 1e-12)
            delta = float(
                (
                    -(y * np.log(p) + (1 - y) * np.log1p(-p))
                    + (y * np.log(q) + (1 - y) * np.log1p(-q))
                ).mean()
            )
        if values:
            families[group] = {
                "mean_permutation_loss_increase": float(np.mean(values)),
                "positive_permutation_folds": int(sum(v > 0 for v in values)),
                "folds": len(values),
                "ablation_loss_minus_compact": delta,
            }
    return {
        "symbol": asset["symbol"],
        "interval": interval,
        "provider": origin["instrument"]["provider"],
        "quality": asset["quality"],
        "economic": economic,
        "causal_regime_thresholds": quantiles,
        "regime_classification": regimes,
        "longer_block_bootstrap_vs_prior": uncertainty,
        "feature_family_diagnostics": families,
        "expiry_priority_sensitivity": expiry_sensitivity,
        "new_fits": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = {str(path.resolve()): verify(path) for path in args.studies}
    write_new_json(args.output / "verification.json", results)
    audit = {
        str(study.resolve()): {
            key: diagnostics(study, key) for key in read_json(study / "summary.json")["results"]
        }
        for study in args.studies
    }
    write_new_json(args.output / "diagnostics.json", audit)
    rows = []
    for study in args.studies:
        report = read_json(study / "summary.json")
        for key, asset in report["results"].items():
            for name, data in asset["experiments"].items():
                m = data["classification"]
                if name == "meta_net_h1":
                    m = data.get("conditional_meta_classification") or {}
                rows.append(
                    {
                        "run": report["run_id"],
                        "symbol": asset["symbol"],
                        "interval": asset["interval"],
                        "experiment": name,
                        "classification_population": "parent_proposed_signals"
                        if name == "meta_net_h1"
                        else "available_labeled_signals",
                        "classification_rows": m.get("rows", 0),
                        "complete_temporal_coverage": data["total_folds"]
                        == len(read_json(study / "plan.json")["protocol"]["boundaries"]) - 1,
                        "provider": audit[str(study.resolve())][key]["provider"],
                        "trades_spanning_nominal_quote_gaps": audit[str(study.resolve())][key][
                            "economic"
                        ][name]["trades_spanning_nominal_quote_gaps"],
                        **{
                            c: m.get(c)
                            for c in ["accuracy", "precision", "roc_auc", "log_loss", "brier_score"]
                        },
                        **{
                            c: data[c]
                            for c in [
                                "win_rate",
                                "trades",
                                "profit_factor",
                                "net_return",
                                "max_drawdown",
                                "positive_folds",
                                "total_folds",
                            ]
                        },
                        "stress_net_return": data.get("stress_net_return"),
                        "log_loss_train_prior": data["classification"]["train_prevalence"][
                            "log_loss"
                        ]
                        if name != "meta_net_h1"
                        else None,
                        "log_loss_logistic": data["classification"]["logistic_l2"]["log_loss"]
                        if name != "meta_net_h1"
                        else None,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "all_experiments.csv", index=False)
    primary = frame.loc[(frame.interval == "1d") & (frame.experiment == "direction_price_h1")]
    primary.to_csv(args.output / "comparison_daily.csv", index=False)
    if len(primary) == 7 and primary.symbol.nunique() == 7:
        primary.to_csv(args.output / "comparison_seven_assets.csv", index=False)
    lines = [
        "# Resultados multimercado",
        "",
        "Comparación prefijada: XGBoost, las mismas 13 features de precio, dirección diaria h=1 y umbral 0,55. Sin selección del ganador sobre OOS. Forex ejecutado sobre H1 bid/ask; cripto sobre OHLC diario con costos configurados. Long/cash, sin apalancamiento.",
        "Períodos y protocolos: "
        + "; ".join(
            str(read_json(study / "plan.json")["protocol"]["boundaries"]) for study in args.studies
        ),
        "",
        "| Activo | Acc. | Prec. | AUC | Log-loss | Brier | Win rate | Trades | PF | Neto | Max DD | Folds + |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def value(v, percent=False):
        return "—" if pd.isna(v) else (f"{v * 100:.2f}%" if percent else f"{v:.4f}")

    for r in primary.itertuples():
        lines.append(
            f"| {r.symbol} | {value(r.accuracy, True)} | {value(r.precision, True)} | {value(r.roc_auc)} | {value(r.log_loss)} | {value(r.brier_score)} | {value(r.win_rate, True)} | {r.trades} | {value(r.profit_factor)} | {value(r.net_return, True)} | {value(r.max_drawdown, True)} | {r.positive_folds}/{r.total_folds} |"
        )
    lines += [
        "",
        "Las horas nominales faltantes y las operaciones que las atraviesan están en diagnostics.json y en la columna trades_spanning_nominal_quote_gaps del CSV. No se imputan ni se eliminan retrospectivamente esas operaciones. Su retorno es condicional a los precios disponibles y no acredita un fill real durante el hueco; bloquea una afirmación de rentabilidad ejecutable. FXCM suministra precios indicativos Active Trader.",
        "Resultados históricos de walk-forward, no prueba final intacta. El drawdown Forex usa marcas H1; cripto usa cierres diarios, por lo que puede subestimar más la excursión intradiaria. Los retornos no están igualados por volatilidad realizada. Comisiones, slippage y carry son escenarios explícitos; los swaps históricos de un broker no están disponibles. El spread Forex sí proviene de precios bid/ask observados. USDT se trata como USD. No interpretar una rentabilidad positiva aislada como ventaja predictiva.",
        "",
        "La tabla completa y las sensibilidades están en `all_experiments.csv`. Los experimentos originales y las reservas prospectivas permanecen intactos. No se promovió ningún modelo.",
        "",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    manifests = {
        str(study.resolve()): {
            str(p.relative_to(study)): sha256(p) for p in study.rglob("*") if p.is_file()
        }
        for study in args.studies
    }
    write_new_json(args.output / "study_file_hashes.json", manifests)
    print(
        {
            "output": str(args.output.resolve()),
            "verified_studies": len(results),
            "comparison_rows": len(primary),
            "experiment_rows": len(frame),
            "adopted": False,
        }
    )


if __name__ == "__main__":
    main()

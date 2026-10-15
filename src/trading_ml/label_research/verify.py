from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from trading_ml.advanced_research.features import ORIGINAL
from trading_ml.advanced_research.models import Fitted
from trading_ml.common import frame_hash, read_json, sha256
from trading_ml.errors import DataError
from trading_ml.feature_research.study import _frozen_hashes, _load_panel
from trading_ml.label_research.events import ExitRule, attach_labels, causal_scales, label_events
from trading_ml.label_research.models import forecasts, meta_pool, partitions
from trading_ml.momentum.config import ResearchCosts


def assert_times(left, right):
    # Object equality treats NaT != NaT. Compare normalized integer timestamps,
    # preserving censored labels and handling CSV datetime precision differences.
    np.testing.assert_array_equal(
        pd.DatetimeIndex(left).as_unit("ns").asi8, pd.DatetimeIndex(right).as_unit("ns").asi8
    )


def load_saved(directory: Path):
    state = read_json(directory / "fit.json")
    if "logistic" in state:
        model = state["logistic"]
    else:
        if sha256(directory / "model.ubj") != state["model_sha256"]:
            raise DataError("Modelo de etiquetas alterado")
        model = XGBClassifier()
        model.load_model(directory / "model.ubj")
        if model.get_booster().feature_names != state["features"]:
            raise DataError("Contrato de features alterado")
    return Fitted(model, state["features"], state["audit"], state["calibration"])


def verify_label_study(settings, directory: Path):
    manifest = read_json(directory / "artifacts.json")
    for relative, digest in manifest.items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or sha256(path) != digest:
            raise DataError(f"Artefacto Triple Barrier alterado: {relative}")
    plan, summary = read_json(directory / "plan.json"), read_json(directory / "summary.json")
    if sha256(directory / "plan.json") != summary["plan_sha256"]:
        raise DataError("Plan alterado")
    if plan["protected_before"] != _frozen_hashes(settings):
        raise DataError("Modelos o reservas originales alterados")
    for source in plan["data"].values():
        if sha256(Path(source["source"])) != source["source_sha256"]:
            raise DataError("Fuente histórica alterada")
    protocol, r = plan["protocol"], plan["protocol"]["research"]
    dates = pd.to_datetime(r["boundaries"], utc=True)
    panel, _ = _load_panel(settings, dates[-1])
    predictions = pd.read_csv(
        directory / "predictions.csv", parse_dates=["reference_time", "target_time"]
    )
    tradebook = pd.read_csv(
        directory / "trades.csv", parse_dates=["signal_reference", "entry_time", "exit_time"]
    )
    records = read_json(directory / "model_records.json")
    folds = read_json(directory / "fold_results.json")
    costs = ResearchCosts(**plan["protocol"]["costs"])
    rule_map = {p["name"]: ExitRule(**p) for p in plan["rules"]}
    reproduced = checked_trades = checked_oof = checked_fits = 0
    models = {}
    verified_fit_paths = set()

    def load(relative):
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()):
            raise DataError("Ruta de modelo inválida")
        if relative not in models:
            models[relative] = load_saved(path)
        return models[relative]

    for symbol, raw in panel.items():
        frame = causal_scales(raw)
        frame["research_ready"] = (
            frame[ORIGINAL + ["label_sigma", "label_atr_fraction"]].notna().all(axis=1)
        )
        snapshot = pd.read_csv(directory / symbol / "causal_features.csv")
        numeric = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "label_sigma",
            "label_atr_fraction",
            *ORIGINAL,
        ]
        np.testing.assert_allclose(
            snapshot[numeric], frame[numeric], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        events, targets = {}, {}
        for name, rule in rule_map.items():
            events[name] = label_events(frame, rule, costs)
            targets[name] = attach_labels(frame, events[name])
            stored = pd.read_csv(
                directory / symbol / f"labels_{name}.csv",
                parse_dates=["reference_time", "target_time"],
            )
            for column in [
                "target",
                "event_label",
                "unit_net_return",
                "exit_price",
                "up_fraction",
                "down_fraction",
            ]:
                np.testing.assert_allclose(
                    stored[column], events[name][column], rtol=1e-10, atol=1e-10, equal_nan=True
                )
            assert_times(stored.target_time, events[name].target_time)
        targets["direction"] = targets["net_h1"].assign(
            target=frame.target, target_time=frame.target_time
        )
        oof = pd.read_csv(
            directory / symbol / "base_oof.csv",
            parse_dates=["reference_time", "model_cutoff", "last_train_target"],
        )
        for model_path, rows in oof.groupby("model_file", sort=False):
            model = load(model_path)
            train, _ = partitions(
                targets["direction"], pd.Timestamp(model.audit["cutoff"]), r, reserve=False
            )
            if (
                frame_hash(train[model.columns + ["target", "target_time"]])
                != model.audit["fit_sha256"]
            ):
                raise DataError("Train OOF base incompatible con sus labels observables")
            if not (rows.model_cutoff == pd.Timestamp(model.audit["cutoff"])).all():
                raise DataError("Cutoff OOF no coincide con el modelo guardado")
            verified_fit_paths.add(model_path)
            checked_fits += 1
            p, _ = forecasts(model, frame.loc[rows.source_index])
            np.testing.assert_allclose(rows.probability, p, rtol=1e-7, atol=1e-8, equal_nan=True)
            if (
                (rows.last_train_target >= rows.model_cutoff)
                | (rows.model_cutoff > rows.reference_time)
            ).any():
                raise DataError("Pronósticos base usados por meta no son OOF")
            checked_oof += len(rows)
        for fold, start in enumerate(dates[:-1]):
            base = load(str(Path(symbol) / "base_oof" / start.strftime("%Y%m%d")))
            full_p, _ = forecasts(base, frame)
            fold_records = [x for x in records if x["symbol"] == symbol and x["fold"] == fold]
            used = set()
            for record in fold_records:
                sample = predictions.loc[
                    (predictions.symbol == symbol)
                    & (predictions.fold == fold)
                    & (predictions.forecast_id == record["forecast_id"])
                ]
                if record["model_path"] is None:
                    if sample.probability.notna().any():
                        raise DataError("Hay scores sin meta-modelo ajustado")
                    continue
                model = load(record["model_path"])
                source = targets[record["target_name"]]
                features = frame.assign(base_probability=full_p) if record["meta"] else frame
                p, cal = forecasts(model, features.loc[sample.source_index])
                expected = cal if record["calibrated"] else p
                np.testing.assert_allclose(
                    sample.probability, expected, rtol=1e-7, atol=1e-8, equal_nan=True
                )
                np.testing.assert_array_equal(
                    sample.target, source.loc[sample.source_index, "target"]
                )
                assert_times(sample.target_time, source.loc[sample.source_index, "target_time"])
                reproduced += 1
                if record["model_path"] in used:
                    continue
                used.add(record["model_path"])
                if record["model_path"] in verified_fit_paths:
                    continue
                fitting_source = (
                    meta_pool(frame, oof, events[record["target_name"]], start)
                    if record["meta"]
                    else source
                )
                reserve = model.audit["calibration_status"] != "not_requested"
                train, calibration = partitions(fitting_source, start, r, reserve=reserve)
                digest = frame_hash(train[model.columns + ["target", "target_time"]])
                if digest != model.audit["fit_sha256"]:
                    raise DataError("El train no reproduce el ajuste de etiquetas")
                if model.calibration is not None:
                    digest = frame_hash(calibration[model.columns + ["target", "target_time"]])
                    if digest != model.audit["calibration_sha256"]:
                        raise DataError("La calibración no reproduce su cola temporal")
                    if train.target_time.max() >= calibration.reference_time.min() - pd.Timedelta(
                        days=r["embargo_days"]
                    ):
                        raise DataError("Train y calibración se solapan")
                checked_fits += 1
                verified_fit_paths.add(record["model_path"])
        # Every executed trade, including stress and ambiguous-bar sensitivity,
        # must reproduce its own rule's economic label at that signal index.
        books = tradebook.loc[tradebook.symbol == symbol]
        event_cache = {(name, "main"): value for name, value in events.items()}
        for (scenario, strategy, fold), group in books.groupby(["scenario", "strategy", "fold"]):
            record = next(
                f
                for f in folds
                if f["symbol"] == symbol
                and f["fold"] == fold
                and f["scenario"] == scenario
                and f["strategy"] == strategy
            )
            key = (record["rule"], scenario)
            if key not in event_cache:
                factor = protocol["stress"]["friction_multiplier"]
                current = (
                    replace(
                        costs,
                        commission_bps=costs.commission_bps * factor,
                        slippage_bps=costs.slippage_bps * factor,
                        spread_bps=costs.spread_bps * factor,
                    )
                    if scenario == "stress"
                    else costs
                )
                event_cache[key] = label_events(
                    frame,
                    rule_map[record["rule"]],
                    current,
                    tie="take_first" if scenario == "take_first" else "stop_first",
                )
            expected = event_cache[key].loc[group.source_index]
            np.testing.assert_allclose(
                group.net_return, expected.unit_net_return, rtol=1e-9, atol=1e-10
            )
            np.testing.assert_allclose(group.exit_price, expected.exit_price, rtol=1e-9, atol=1e-8)
            assert_times(group.exit_time, expected.target_time)
            if not (
                group.signal_reference
                < dates[fold + 1] - pd.Timedelta(days=r["maximum_horizon"] + 1)
            ).all():
                raise DataError("El backtest eludió el corte calendario")
            checked_trades += len(group)
    return {
        "passed": True,
        "run_id": summary["run_id"],
        "artifacts_verified": len(manifest),
        "prediction_series_reproduced": reproduced,
        "oof_predictions_reproduced": checked_oof,
        "model_fit_contracts_verified": checked_fits,
        "executed_trade_payoffs_verified": checked_trades,
        "future_prices_only_in_labels_or_realized_execution": True,
        "existing_models_and_reservations_preserved": True,
        "new_fits": 0,
    }

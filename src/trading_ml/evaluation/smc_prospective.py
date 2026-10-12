from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from trading_ml.backtesting import simulate
from trading_ml.common import read_json, sha256, sidecar, write_json
from trading_ml.config.settings import settings_from_dict
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.evaluation.signals import signal_quality
from trading_ml.models import load_model
from trading_ml.preprocessing import load_prepared, prepare
from trading_ml.smc.rules import directional_states, rule_probabilities


def evaluate_smc_prospective(
    root: Path, source: Path, now: pd.Timestamp | None = None
) -> dict[str, Any]:
    """Evalúa los tres sistemas congelados cuando toda la ventana futura existe.

    No entrena, no recalibra, no selecciona modelos y no los promueve. Un CSV
    incompleto sólo produce un error de datos pendientes, nunca métricas finales.
    """
    pointer = read_json(root / "smc_studies" / "latest.json")
    report = read_json(Path(pointer["summary"]))
    directory = Path(report["directory"])
    plan = read_json(directory / "plan.json")
    if sha256(directory / "plan.json") != report["plan_sha256"]:
        raise DataError("El plan prospectivo cambió después de congelarse")
    reservation = report["prospective"]
    start = pd.Timestamp(reservation["first_reference"])
    end = pd.Timestamp(reservation["end_reference_exclusive"])
    manifest = read_json(sidecar(source))
    if manifest.get("kind") != "raw" or manifest["sha256"] != sha256(source):
        raise DataError("CSV futuro sin procedencia íntegra")
    raw = pd.read_csv(source)
    timestamps = pd.to_datetime(raw.timestamp, utc=True)
    candidate = load_model(root, reservation["candidate_version"])
    previous = load_model(root, reservation["reference_version"])
    config = replace(settings_from_dict(candidate.metadata["config"]), root=root)
    required_exit_open = end + (config.training.horizon - 1) * config.delta
    if timestamps.max() < required_exit_open:
        raise DataError(
            f"Evaluación final pendiente: se necesitan velas cerradas, etiquetas y aperturas de salida hasta {required_exit_open.isoformat()}; no se producen métricas con una ventana parcial"
        )
    if start <= pd.Timestamp(candidate.metadata["selection_label_end"]) or start <= pd.Timestamp(
        previous.metadata["selection_label_end"]
    ):
        raise LeakageError("La reserva final invade entrenamiento/selección")
    if any(manifest["market"][key] != config.contract()[key] for key in ["symbol", "interval"]):
        raise DataError("Activo/intervalo futuro incompatibles")
    frozen_raw = directory / "raw_snapshot.csv"
    if sha256(frozen_raw) != plan["raw_sha256"]:
        raise DataError("Snapshot anterior a la reserva corrupto")
    original = pd.read_csv(frozen_raw)
    shared = raw.loc[raw.timestamp.isin(original.timestamp)].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(original.reset_index(drop=True), shared, check_exact=True)
    except AssertionError as exc:
        raise DataError(
            "El CSV nuevo debe conservar el snapshot histórico y ampliarlo con velas nuevas"
        ) from exc
    output = directory / "prospective_evaluation"
    candidate_path = prepare(config, source, output / "candidate_prepared.csv", now=now)
    frame, _ = load_prepared(candidate_path, candidate.metadata["contract"])
    test = frame.loc[(frame.reference_time >= start) & (frame.reference_time < end)].dropna(
        subset=["target", "target_time"]
    )
    expected = pd.date_range(start, end - config.delta, freq=config.delta)
    if len(test) != len(expected) or not (pd.DatetimeIndex(test.reference_time) == expected).all():
        raise DataError("Faltan referencias o etiquetas en la reserva final")
    rule_settings = replace(
        config,
        features=replace(
            config.features,
            **{f"smc_{key}": value for key, value in plan["protocol"]["smc"].items()},
        ),
    )
    rule_path = prepare(rule_settings, source, output / "rules_prepared.csv", now=now)
    rule_frame, _ = load_prepared(rule_path)
    rule_test = rule_frame.loc[rule_frame.reference_time.isin(test.reference_time)]
    rule_probability, rule_state = rule_probabilities(rule_test)
    threshold = plan["protocol"]["signals"]["threshold"]
    comparison = {}
    for name, probability, states in [
        ("original", previous.probabilities(test), None),
        ("smc_rules", rule_probability, rule_state),
        ("combined", candidate.probabilities(test), None),
    ]:
        states = directional_states(probability, threshold) if states is None else states
        signals = test[["timestamp", "reference_time"]].copy()
        signals["probability_up"] = probability
        simulation = simulate(
            frame.loc[frame.timestamp <= test.target_time.max()],
            signals,
            config.training.horizon,
            config.backtest,
            start - config.delta,
        )
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        simulation.equity.to_csv(destination / "equity.csv", index=False)
        simulation.trades.to_csv(destination / "trades.csv", index=False)
        signals["state"] = states
        signals["target"] = test.target.to_numpy()
        signals.to_csv(destination / "predictions.csv", index=False)
        comparison[name] = {
            "metrics": classification_metrics(test.target.to_numpy(), probability),
            "signal_quality": signal_quality(test.target.to_numpy(), probability, states),
            "backtest": simulation.summary,
        }
    result = {
        "run_id": report["run_id"],
        "status": "complete_frozen_prospective_window",
        "reservation": reservation,
        "source_sha256": manifest["sha256"],
        "automatic_promotion": False,
        "comparison": comparison,
    }
    write_json(output / "summary.json", result)
    return result

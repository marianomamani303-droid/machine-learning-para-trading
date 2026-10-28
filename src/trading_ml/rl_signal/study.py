from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import pandas as pd
import torch

from trading_ml.common import environment, object_hash, read_json, sha256, write_new_json
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.rl_signal.data import (
    COLUMNS,
    DAY,
    load_panel,
    predicted_block,
    training_predictions,
)
from trading_ml.rl_signal.dqn import Normalizer, fit_dqn, network
from trading_ml.rl_signal.environment import rollout
from trading_ml.rl_signal.evaluation import acceptance, metrics, paired_inference, reconstruct


def validate_config(config):
    if (
        config["data"]["symbol"] != "BTCUSDT"
        or config["data"]["interval"] != "1d"
        or config["training"]["seeds"] != [42, 1729, 2027]
        or config["training"]["hidden"] != [32, 32]
        or config["training"]["maximum_steps"] != 6000
        or config["portfolio"]["actions"] != [0.0, 0.25, 0.5]
        or config["data"]["boundaries"] != ["2023-01-01", "2024-01-01", "2025-01-01", "2025-07-01"]
        or config["predictor"]["recipe"] != "top_core_3"
        or config["predictor"]["C"] != 1.0
    ):
        raise ValueError("This runner only supports the bounded registered experiment")


def hash_paths(paths):
    """Bounded parallel read-only I/O; each file still gets a full SHA256."""
    paths = list(paths)
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(sha256, paths))
    return dict(zip(map(str, paths), hashes, strict=True))


def protected_snapshot(exclude: Path):
    """Hash bytes only: no reserved predictions/outcomes are interpreted."""
    paths = [
        p for p in sorted(Path("artifacts").rglob("*")) if p.is_file() and exclude not in p.parents
    ]
    print(f"Hashing {len(paths)} protected files with 8 I/O workers", flush=True)
    return hash_paths(paths)


def reservation_audit():
    reservations = []
    for path in sorted(Path("artifacts").glob("*studies/*/prospective.json")):
        v = read_json(path)
        reservations.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "start": v.get("start", v.get("first_reference")),
                "end": v.get("end", v.get("end_reference_exclusive")),
            }
        )
    for family in ("momentum_studies", "mean_reversion_studies"):
        for path in sorted((Path("artifacts") / family).glob("*/plan.json")):
            v = read_json(path).get("prospective")
            if v:
                reservations.append(
                    {
                        "path": str(path),
                        "sha256": sha256(path),
                        "start": v["start"],
                        "end": v["end"],
                    }
                )
    return reservations


def audit_prior():
    prior = Path("artifacts/focused_financial_study/20261017T232854_36d523ee")
    manifest = read_json(prior / "artifacts.json")
    checks = {k: sha256(prior / k) == v for k, v in manifest.items()}
    if not all(checks.values()):
        raise ValueError("Previous financial study failed integrity checks")
    f = pd.read_csv(prior / "predictions.csv", float_precision="round_trip")
    f = f.loc[(f.symbol == "BTCUSDT") & (f.model == "logistic_l2") & (f.recipe == "top_core_3")]
    economic = pd.read_csv(prior / "economic_table.csv")
    economic = economic.loc[
        (economic.symbol == "BTCUSDT")
        & (economic.model == "logistic_l2")
        & (economic.recipe == "top_core_3")
    ]
    return {
        "run": str(prior),
        "hashes": checks,
        "predictive": classification_metrics(f.target.to_numpy(int), f.probability.to_numpy()),
        "economics": economic.to_dict("records"),
        "folds": [
            {
                "fold": int(k),
                "start": g.reference_time.min(),
                "last": g.reference_time.max(),
                "rows": len(g),
            }
            for k, g in f.groupby("fold")
        ],
        "cost_arithmetic_bps_per_side": 16,
        "frozen_weights_available": False,
        "interpretation": "reused exploratory history, assumed costs, not real fills",
    }


def run(protocol: Path) -> Path:
    config = tomllib.loads(protocol.read_text(encoding="utf-8"))
    validate_config(config)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    output = Path("artifacts/rl_signal") / stamp
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol, output / "protocol.toml")
    shutil.copyfile("RL_SIGNAL_PROTOCOL.md", output / "protocol.md")
    reservations = reservation_audit()
    prospective_start = pd.Timestamp(config["prospective"]["start"], tz="UTC")
    if any(
        pd.Timestamp(r["end"]).tz_localize("UTC") > prospective_start
        if pd.Timestamp(r["end"]).tzinfo is None
        else pd.Timestamp(r["end"]) > prospective_start
        for r in reservations
    ):
        raise ValueError("Prospective window overlaps an existing reservation")
    print(f"{output}: hashing protected historical files", flush=True)
    protected = protected_snapshot(output)
    write_new_json(output / "protected_before.json", protected)
    write_new_json(output / "reservations.json", reservations)
    with ZipFile(output / "source.zip", "x", ZIP_DEFLATED) as archive:
        for path in sorted(Path("src/trading_ml").rglob("*.py")):
            archive.write(path, path.as_posix())
        for path in sorted(Path("tests").glob("test_rl_signal.py")):
            archive.write(path, path.as_posix())
    versions = {
        **environment(),
        **{p: importlib.metadata.version(p) for p in ("torch", "joblib", "pytest")},
    }
    freeze = subprocess.run(
        [__import__("sys").executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    (output / "requirements-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    write_new_json(
        output / "plan.json",
        {
            "registered_at": datetime.now(UTC).isoformat(),
            "config": config,
            "protocol_sha256": sha256(output / "protocol.toml"),
            "design_sha256": sha256(output / "protocol.md"),
            "source_sha256": sha256(output / "source.zip"),
            "versions": versions,
            "protected_files": len(protected),
            "maximum_dqn_fits": 9,
            "maximum_total_steps": 54000,
            "predictor_columns": COLUMNS,
            "exploratory": True,
            "live_orders": False,
        },
    )
    write_new_json(output / "prior_audit.json", audit_prior())
    panel, sources = load_panel(config)
    write_new_json(output / "sources.json", sources)
    panel.to_csv(output / "panel.csv", index=False)
    bounds = pd.to_datetime(config["data"]["boundaries"], utc=True)
    curves_by_key, orders_by_key = {}, {}
    split_records, predictor_metrics, training_records, fold_metrics = [], [], [], []
    all_predictions = []
    for fold, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        folder = output / f"fold_{fold}"
        folder.mkdir()
        validation_start = start - config["data"]["validation_days"] * DAY
        embargo = config["data"]["embargo_days"] * DAY
        train, inner_fits = training_predictions(panel, validation_start - embargo, config)
        validation, val_model, val_contract = predicted_block(
            panel, validation_start, start - embargo, config
        )
        test, test_model, test_contract = predicted_block(panel, start, end, config)
        for i, (model, contract) in enumerate(inner_fits):
            joblib.dump(model, folder / f"inner_predictor_{i:02d}.joblib")
            write_new_json(folder / f"inner_predictor_{i:02d}.json", contract)
        for phase, _frame, model, contract in (
            ("validation", validation, val_model, val_contract),
            ("test", test, test_model, test_contract),
        ):
            joblib.dump(model, folder / f"{phase}_predictor.joblib")
            write_new_json(folder / f"{phase}_predictor.json", contract)
        for phase, frame in (("train", train), ("validation", validation), ("test", test)):
            frame.to_csv(folder / f"{phase}_predictions.csv", index=False)
            all_predictions.append(frame.assign(fold=fold, phase=phase))
            predictor_metrics.append(
                {
                    "fold": fold,
                    "phase": phase,
                    **{
                        k: v
                        for k, v in classification_metrics(
                            frame.target.to_numpy(int), frame.probability.to_numpy()
                        ).items()
                        if k in ("rows", "roc_auc", "log_loss", "brier_score", "accuracy")
                    },
                }
            )
            split_records.append(
                {
                    "fold": fold,
                    "phase": phase,
                    "rows": len(frame),
                    "first_signal": frame.reference_time.min().isoformat(),
                    "last_signal": frame.reference_time.max().isoformat(),
                    "last_label": frame.target_time.max().isoformat(),
                    "latest_possible_exit": frame.latest_exit_time.max().isoformat(),
                }
            )
        agents = {}
        for seed in config["training"]["seeds"]:
            print(
                f"fold {fold}, seed {seed}: 6000 fixed train steps; {len(train)}/{len(validation)}/{len(test)} dates",
                flush=True,
            )
            model, norm, history = fit_dqn(train, validation, config, seed, folder / f"seed_{seed}")
            training_records.append({"fold": fold, **history})
            agents[seed] = (model, norm)
        # Nothing is selected using these results: all seeds/scenarios are kept.
        for scenario in config["inference"]["scenarios"]:
            policies = [(p, 0) for p in config["inference"]["comparisons"]]
            policies += [("dqn", s) for s in config["training"]["seeds"]]
            for policy, seed in policies:
                model, norm = agents.get(seed, (None, None))
                curve, orders = rollout(test, config, policy, scenario, norm, model)
                curve["fold"], orders["fold"] = fold, fold
                curve.to_csv(folder / f"{scenario}_{policy}_{seed}_curve.csv", index=False)
                orders.to_csv(folder / f"{scenario}_{policy}_{seed}_orders.csv", index=False)
                replay = reconstruct(curve, orders, config, scenario)
                fold_metrics.append(
                    {
                        "fold": fold,
                        "scenario": scenario,
                        "policy": policy,
                        "seed": seed,
                        **metrics(curve, orders),
                        **replay,
                    }
                )
                key = (scenario, policy, seed)
                curves_by_key.setdefault(key, []).append(curve)
                orders_by_key.setdefault(key, []).append(orders)
    curves = {k: pd.concat(v, ignore_index=True) for k, v in curves_by_key.items()}
    orders = {k: pd.concat(v, ignore_index=True) for k, v in orders_by_key.items()}
    aggregate = [
        {"scenario": s, "policy": p, "seed": seed, **metrics(curves[(s, p, seed)], v)}
        for (s, p, seed), v in orders.items()
    ]
    predictions = pd.concat(all_predictions, ignore_index=True)
    test_predictions = predictions.loc[predictions.phase == "test"]
    predictor_metrics.append(
        {
            "fold": "all",
            "phase": "test",
            **{
                k: v
                for k, v in classification_metrics(
                    test_predictions.target.to_numpy(int), test_predictions.probability.to_numpy()
                ).items()
                if k in ("rows", "roc_auc", "log_loss", "brier_score", "accuracy")
            },
        }
    )
    contrasts, seed_intervals = paired_inference(curves, config)
    decision = acceptance(curves, contrasts, config)
    pd.DataFrame(aggregate).to_csv(output / "aggregate_metrics.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(predictor_metrics).to_csv(output / "predictor_metrics.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "contrasts.csv", index=False)
    pd.DataFrame(seed_intervals).to_csv(output / "seed_intervals.csv", index=False)
    write_new_json(output / "splits.json", split_records)
    write_new_json(output / "training.json", training_records)
    write_new_json(output / "decision.json", decision)
    print("Checking preservation of all prior artifacts", flush=True)
    preserved = hash_paths(map(Path, protected)) == protected
    if not preserved:
        raise ValueError("Prior artifacts changed during the experiment")
    write_new_json(
        output / "preservation.json",
        {"files": len(protected), "unchanged": preserved, "hash": object_hash(protected)},
    )
    final_fold = len(bounds) - 2
    prospective_paths = [output / f"fold_{final_fold}/test_predictor.joblib"]
    for seed in config["training"]["seeds"]:
        prospective_paths.extend(
            [
                output / f"fold_{final_fold}/seed_{seed}/selected.pt",
                output / f"fold_{final_fold}/seed_{seed}/normalizer.json",
            ]
        )
    write_new_json(
        output / "prospective.json",
        {
            **config["prospective"],
            "status": "pending_new_data_and_quote_feed_registration",
            "model_hashes": {str(p.relative_to(output)): sha256(p) for p in prospective_paths},
            "seeds": config["training"]["seeds"],
            "selection": "last fold, internal validation only, all seeds",
            "real_orders": False,
            "future_results": None,
        },
    )
    make_report(
        output, aggregate, fold_metrics, predictor_metrics, contrasts, decision, training_records
    )
    write_new_json(
        output / "artifacts.json",
        {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    print(f"COMPLETE {output}: {decision['conclusion']}", flush=True)
    return output


def make_report(output, aggregate, folds, predictive, contrasts, decision, training):
    lines = [
        "# Resultados exploratorios: reglas frente a DQN",
        "",
        f"Corrida: `{output}`.",
        "",
        f"**Conclusión: {decision['conclusion']}.** No hay confirmación económica independiente.",
        "",
        "La señal es idéntica para todas las políticas; el agente sólo cambia gestión de posiciones.",
        "",
        "Retornos netos acumulados; costos USDT sumados de tres cuentas por fold de 10000 USDT.",
        "",
        "|Escenario|Política|Semilla|Retorno %|Sharpe|DD %|Exp. media|Turnover|Costo USDT|Órdenes|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in aggregate:
        lines.append(
            f"|{r['scenario']}|{r['policy']}|{r['seed']}|{100 * r['net_return']:.3f}|{r['sharpe']:.3f}|{100 * r['max_drawdown']:.3f}|{r['mean_exposure']:.4f}|{r['turnover']:.3f}|{r['cost_usdt']:.2f}|{r['filled_orders']}|"
        )
    lines += [
        "",
        "## Predictor compartido",
        "",
        "|Fold|Fase|N|AUC|Log-loss|Brier|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in predictive:
        lines.append(
            f"|{r['fold']}|{r['phase']}|{r['rows']}|{r['roc_auc']:.5f}|{r['log_loss']:.5f}|{r['brier_score']:.5f}|"
        )
    lines += [
        "",
        "## Comparación primaria",
        "",
        "Media de diferencias log de las tres semillas; no es una cartera ensemble.",
        "",
        "|Escenario|Bloque|Δ log vs band|IC95%|IC simultáneo 24|p Holm|",
        "|---|---:|---:|---|---|---:|",
    ]
    for r in contrasts:
        if r["reference"] == "band":
            lines.append(
                f"|{r['scenario']}|{r['block_days']}|{r['log_gain']:.5f}|[{r['ci95_low']:.5f}, {r['ci95_high']:.5f}]|[{r['simultaneous_low']:.5f}, {r['simultaneous_high']:.5f}]|{r['p_holm_24']:.5f}|"
            )
    lines += [
        "",
        "## Folds y semillas, costos base",
        "",
        "|Fold|Política|Semilla|Retorno %|Sharpe|DD %|Exp. media|Turnover|Costo USDT|Órdenes|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in folds:
        if r["scenario"] == "base":
            lines.append(
                f"|{r['fold']}|{r['policy']}|{r['seed']}|{100 * r['net_return']:.3f}|{r['sharpe']:.3f}|{100 * r['max_drawdown']:.3f}|{r['mean_exposure']:.4f}|{r['turnover']:.3f}|{r['cost_usdt']:.2f}|{r['filled_orders']}|"
            )
    lines += [
        "",
        "## Criterios y entrenamiento",
        "",
        "```json",
        json.dumps(decision, indent=2, ensure_ascii=False),
        "```",
        "",
        "|Fold|Semilla|Paso seleccionado|Pasos máximos|Episodios completos|Fechas únicas train|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in training:
        lines.append(
            f"|{r['fold']}|{r['seed']}|{r['selected_step']}|{r['steps']}|{r['completed_episodes']}|{r['unique_training_dates']}|"
        )
    lines += [
        "",
        "## Interpretación y límites",
        "",
        "Todos los resultados proceden de historia reutilizada. El protocolo, fórmulas, discrepancias de informes anteriores y fechas están en protocol.md; splits.json contiene las fechas efectivamente usadas.",
        "",
        "Un DQN en cash conserva capital; una mejora frente a una regla perdedora por abstención no es predicción mejorada ni alpha. Menor riesgo o DD tampoco prueba mayor información predictiva. No se seleccionan semillas con test.",
        "",
        "Costos, spread, slippage y lotes son supuestos; no son fills reales. No hay profundidad ni impacto verificables. Se ejecuta a aperturas con latencia idealizada en base y una cola de un día en demora. La deriva de exposición y polvo no liquidable se registran; DD sólo a aperturas, sin riesgo intravela completo.",
        "",
        "Tres semillas y episodios repetidos no aumentan la muestra independiente. Inferencia temporal condicional al protocolo; Holm de 24 contrastes no corrige las miles de decisiones históricas anteriores. Los bloques internos y exteriores tienen distinta longitud; las probabilidades pueden cambiar de distribución tras reentrenamiento.",
        "",
        "Preservación de artefactos anteriores: preservation.json. Reconstrucción independiente desde flujos de caja, precios y unidades: fold_metrics.csv y comando verify. Todos los checkpoints, replay sólo train, normalizadores, modelos y predicciones se conservan.",
        "",
        "Seguimiento propuesto 2027-11-01..2028-11-01 exclusivo, evaluación única desde 2028-11-02, sin reentrenar ni elegir semilla. prospective.json fija hashes de los tres agentes y predictor del último fold. Pendiente registrar feed de quotes y emisión real anterior a ejecución; no hay resultados futuros ni órdenes reales.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def read_frame(path):
    f = pd.read_csv(path, float_precision="round_trip")
    for c in f.columns:
        if c.endswith("time") or c in (
            "timestamp",
            "available_at",
            "predictor_fit_at",
            "predictor_last_label",
        ):
            f[c] = pd.to_datetime(f[c], utc=True)
    return f


def verify(output: Path) -> dict:
    manifest = read_json(output / "artifacts.json")
    for relative, expected in manifest.items():
        if sha256(output / relative) != expected:
            raise AssertionError(f"Artifact changed: {relative}")
    config = tomllib.loads((output / "protocol.toml").read_text(encoding="utf-8"))
    curves, order_sets, rebuilt, max_error = {}, {}, 0, 0.0
    for folder in sorted(output.glob("fold_*")):
        train = read_frame(folder / "train_predictions.csv")
        val = read_frame(folder / "validation_predictions.csv")
        test = read_frame(folder / "test_predictions.csv")
        assert train.latest_exit_time.max() < val.reference_time.min() - 6 * DAY
        assert val.latest_exit_time.max() < test.reference_time.min() - 6 * DAY
        for phase, frame in (("train", train), ("validation", val), ("test", test)):
            assert (frame.predictor_last_label < frame.predictor_fit_at - 6 * DAY).all()
            assert (frame.predictor_fit_at <= frame.reference_time).all()
            assert frame.latest_exit_time.max() < pd.Timestamp(config["data"]["stop"], tz="UTC")
            for fit_at, chunk in frame.groupby("predictor_fit_at"):
                if phase == "train":
                    contracts = [
                        p
                        for p in folder.glob("inner_predictor_*.json")
                        if pd.Timestamp(read_json(p)["fit_at"]) == fit_at
                    ]
                    contract_path = contracts[0]
                else:
                    contract_path = folder / f"{phase}_predictor.json"
                model = joblib.load(contract_path.with_suffix(".joblib"))
                probability = model.predict_proba(chunk[COLUMNS].to_numpy())[:, 1]
                np.testing.assert_allclose(probability, chunk.probability, atol=1e-13, rtol=1e-12)
        for seed in config["training"]["seeds"]:
            norm = Normalizer.from_dict(read_json(folder / f"seed_{seed}/normalizer.json"))
            expected = Normalizer.fit(train, config)
            np.testing.assert_array_equal(norm.mean, expected.mean)
            np.testing.assert_array_equal(norm.scale, expected.scale)
            with np.load(folder / f"seed_{seed}/training_replay.npz") as replay:
                assert replay["row_indices"].min() >= 0 and replay["row_indices"].max() < len(train)
            history = read_json(folder / f"seed_{seed}/training.json")
            candidates = history["checkpoints"]
            best = max(candidates, key=lambda r: (r["validation_reward"], -r["step"]))
            assert best["step"] == history["selected_step"]
            selected = torch.load(folder / f"seed_{seed}/selected.pt", weights_only=True)
            checkpoint = torch.load(
                folder / f"seed_{seed}/step_{best['step']:05d}.pt", weights_only=False
            )
            for key in selected:
                assert torch.equal(selected[key], checkpoint["online"][key])
        for path in sorted(folder.glob("*_curve.csv")):
            scenario, policy, seed_text, _ = (
                path.stem.split("_")
                if "long_scaled" not in path.stem
                else (path.stem.split("_")[0], "long_scaled", "0", "curve")
            )
            seed = int(seed_text)
            curve = read_frame(path)
            orders = read_frame(path.with_name(path.name.replace("_curve", "_orders")))
            result = reconstruct(curve, orders, config, scenario)
            max_error = max(max_error, result["max_equity_error"])
            agent = network(config) if policy == "dqn" else None
            norm = None
            if agent is not None:
                agent.load_state_dict(
                    torch.load(folder / f"seed_{seed}/selected.pt", weights_only=True)
                )
                agent.eval()
                norm = Normalizer.from_dict(read_json(folder / f"seed_{seed}/normalizer.json"))
            regenerated, _ = rollout(test, config, policy, scenario, norm, agent)
            for col in ("action", "cash", "units", "log_return", "reward", "turnover", "cost"):
                np.testing.assert_allclose(curve[col], regenerated[col], atol=1e-10, rtol=1e-10)
            curves.setdefault((scenario, policy, seed), []).append(curve)
            order_sets.setdefault((scenario, policy, seed), []).append(orders)
            rebuilt += 1
    curves = {k: pd.concat(v, ignore_index=True) for k, v in curves.items()}
    contrasts, _ = paired_inference(curves, config)
    assert acceptance(curves, contrasts, config) == read_json(output / "decision.json")
    saved = pd.read_csv(output / "contrasts.csv", float_precision="round_trip")
    for col in (
        "log_gain",
        "ci95_low",
        "ci95_high",
        "simultaneous_low",
        "simultaneous_high",
        "p_holm_24",
    ):
        np.testing.assert_allclose(saved[col], pd.DataFrame(contrasts)[col], atol=1e-13)
    previous = read_json(output / "protected_before.json")
    assert hash_paths(map(Path, previous)) == previous
    return {
        "artifacts_verified": len(manifest),
        "portfolios_reconstructed": rebuilt,
        "protected_unchanged": len(previous),
        "max_equity_error": max_error,
        "predictors_reloaded": True,
        "normalizers_train_only": True,
        "checkpoints_internal_only": True,
        "new_fits": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("--protocol", type=Path, default=Path("config/rl_signal.toml"))
    parser.add_argument("--study", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.protocol)
    else:
        if args.study is None:
            parser.error("verify requires --study")
        print(json.dumps(verify(args.study), indent=2))


if __name__ == "__main__":
    main()

"""Publish a validated existing study and figures without fitting or selecting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from trading_ml.common import read_json, sha256, write_new_json  # noqa: E402
from trading_ml.frontier_research.study import economic, metrics  # noqa: E402
from trading_ml.models import load_model  # noqa: E402
from trading_ml.momentum.config import ResearchCosts  # noqa: E402
from trading_ml.momentum.study import _reference_frame  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for relative, digest in read_json(args.study / "artifacts.json").items():
        path = args.study / relative
        if not path.resolve().is_relative_to(args.study.resolve()) or sha256(path) != digest:
            raise ValueError(f"Artefacto alterado: {relative}")
    summary = read_json(args.study / "summary.json")
    output = args.output or Path("artifacts/frontier_analyses") / args.study.name
    output.mkdir(parents=True, exist_ok=False)
    root = args.study.parent.parent
    plan = read_json(args.study / "plan.json")
    dates = pd.to_datetime(plan["protocol"]["research"]["boundaries"], utc=True)
    start = dates[-1]
    end = pd.Timestamp(plan["protocol"]["research"]["historical_end"], tz="UTC")
    costs = ResearchCosts(**summary["costs"])
    raw = pd.read_csv(args.study / "BTCUSDT_causal_features.csv")
    for column in ["timestamp", "reference_time"]:
        raw[column] = pd.to_datetime(raw[column], utc=True)
    registered = {}
    for version in [
        "v20261012T130540_d2bb6ac0",
        "v20261012T131650_5106daf4",
        "v20261012T134917_f88d6f93",
    ]:
        bundle = load_model(root, version)
        if start <= pd.Timestamp(bundle.metadata["selection_label_end"]):
            raise ValueError("Comparación registrada invade entrenamiento/selección")
        frame = _reference_frame(
            raw[
                ["timestamp", "reference_time", "open", "high", "low", "close", "volume", "regime"]
            ].copy(),
            bundle,
        )
        for name in ["risk_atr", "risk_volatility"]:
            frame[name] = raw[name]
        ready = frame[bundle.metadata["features"]].notna().all(axis=1)
        forecast = ready & (frame.reference_time >= start) & (frame.reference_time < end)
        p = np.full(len(frame), 0.5)
        p[forecast] = bundle.probabilities(frame.loc[forecast])
        scored = forecast & frame.target.notna() & (frame.target_time < end)
        simulation = economic(
            frame, p, start, end, costs, bundle.metadata["config"]["backtest"]["threshold"]
        )
        model_dir = output / "registered_references" / version
        model_dir.mkdir(parents=True, exist_ok=False)
        simulation.equity.to_csv(model_dir / "equity.csv", index=False)
        simulation.trades.to_csv(model_dir / "trades.csv", index=False)
        registered[version] = {
            "predictive": metrics(frame.loc[scored, "target"], p[scored]),
            "economic": simulation.metrics,
            "trained": False,
            "model_sha256": bundle.metadata["model_sha256"],
            "selection_label_end": bundle.metadata["selection_label_end"],
            "threshold": bundle.metadata["config"]["backtest"]["threshold"],
        }
    write_new_json(output / "registered_references.json", registered)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout="constrained")
    for ax, (key, period) in zip(axes.flat, summary["periods"]["selection"].items(), strict=True):
        scores = period["predictive"]
        reference = scores["xgboost_local"]["log_loss"]
        names = [
            name
            for name in scores
            if name not in {"xgboost_local", "training_prior", "constant_half", "regime_prior"}
        ]
        values = np.array([scores[name]["log_loss"] - reference for name in names])
        ax.barh(names, values, color=np.where(values < 0, "#287d73", "#9b5773"))
        ax.axvline(0, color="#222", linewidth=0.8)
        ax.set_title(key.replace("__", " / "))
        ax.set_xlabel("Δ log-loss contra XGBoost local (menor es mejor)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(
        "Validación histórica de desarrollo: tres folds, modelos calibrados\nObjetivos distintos; no comparar niveles absolutos entre paneles",
        fontsize=13,
    )
    fig.savefig(output / "predictive_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    for ax, symbol in zip(axes, ["BTCUSDT", "ETHUSDT"], strict=True):
        key = symbol + "__direction_h1"
        chosen = summary["selection"][key]
        systems = [
            f"xgboost_local__{chosen['threshold']}",
            f"{chosen['recipe']}__{chosen['threshold']}",
            "cash",
            "buy_and_hold",
        ]
        for system in dict.fromkeys(systems):
            curve = pd.read_csv(
                args.study / "historical_description" / "4" / key / system / "equity.csv"
            )
            ax.plot(
                pd.to_datetime(curve.timestamp, utc=True),
                100 * (curve.equity / curve.equity.iloc[0] - 1),
                label=system,
            )
        ax.set_title(symbol + " / objetivo dirección h1, política defensiva compartida")
        ax.set_ylabel("Retorno neto acumulado (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Período posterior observado: descripción histórica, no confirmación independiente",
        fontsize=12,
    )
    fig.savefig(output / "historical_equity.png", dpi=180)
    plt.close(fig)
    lines = [
        (args.study / "REPORT.md").read_text(encoding="utf-8"),
        "\n## Resultados completos por variante\n",
        "Los siguientes números son exclusivamente de validación anterior al período descriptivo. Ninguna fila ganadora cambia la selección registrada.\n",
        "| Activo / objetivo | Modelo | Acc. | Bal. acc. | Prec. | Recall | F1 | AUC | Loss | Brier | Neto .55 | Trades | PF |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, period in summary["periods"]["selection"].items():
        for name, m in period["predictive"].items():
            e = period["economic"][f"{name}__0.55"]["metrics"]
            pf = "n/a" if e["profit_factor"] is None else f"{e['profit_factor']:.2f}"
            auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
            lines.append(
                f"| {key} | {name} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {auc} | {m['log_loss']:.4f} | {m['brier_score']:.4f} | {e['net_return']:.2%} | {e['trade_count']} | {pf} |"
            )
    lines += [
        "\n## Ablaciones de datos cruzados y calibración\n",
        "| Activo / objetivo | Δloss logística cross−local | Δloss XGBoost cross−local | Mejor receta (selección) |",
        "|---|---:|---:|---|",
    ]
    for key, period in summary["periods"]["selection"].items():
        scores = period["predictive"]
        lines.append(
            f"| {key} | {scores['logistic_cross']['log_loss'] - scores['logistic_local']['log_loss']:.5f} | {scores['xgboost_cross']['log_loss'] - scores['xgboost_local']['log_loss']:.5f} | {summary['selection'][key]['recipe']} |"
        )
    lines += [
        "\nLas métricas train/raw/calibradas por fold figuran en cada `fit.json`. El registro conserva las regresiones de magnitud y retorno/volatilidad con controles de MAE/MSE, sin convertir esas magnitudes en probabilidades inventadas.\n",
        "## Regímenes y estabilidad\n",
        "| Activo / objetivo | Régimen | Filas | AUC receta seleccionada | Loss | Loss XGBoost local |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, period in summary["periods"]["selection"].items():
        selected = summary["selection"][key]["recipe"]
        for regime, group in period["by_regime"].items():
            m = group["models"][selected]
            auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
            lines.append(
                f"| {key} | {regime} | {group['rows']} | {auc} | {m['log_loss']:.4f} | {group['models']['xgboost_local']['log_loss']:.4f} |"
            )
    lines += [
        "\n## Presupuesto, fallos e integridad\n",
        f"La corrida anterior `20261014T134018_d1e1df02` fue abortada por una máscara de ejecución dependiente del desenlace futuro. Se conservan todos sus archivos y se marcan inválidos sus retornos. Registró 70 ajustes completados y se cargaron 72 al presupuesto, incluyendo dos posibles ajustes interrumpidos. La recuperación reutilizó {summary['reused_models']} modelos y calibraciones validados, efectuó {summary['new_fits']} ajustes nuevos y cargó {summary['actual_fits']} en total, con límite 500. El contador total es una cota presupuestaria conservadora, no un número exacto de ajustes completados de la corrida abortada.\n",
        "Las curvas de ejecución usan todas las referencias con features disponibles, sin consultar etiquetas o tiempo de finalización. Sólo las métricas predictivas excluyen objetivos no observables antes del cierre del fold. La corrección no alteró recetas ni amplió la búsqueda.\n",
        "## Comparación con presupuesto de riesgo fijado anteriormente\n",
        "| Activo / objetivo | Sistema | Capital medio anterior | Capital medio realizado | Neto posterior |",
        "|---|---|---:|---:|---:|",
    ]
    for key, comparison in summary["matched_risk_budget"].items():
        for name, e in comparison["historical_actual"].items():
            lines.append(
                f"| {key} | {name} | {comparison['selection_mean_exposures'][name]:.2%} | {e['mean_capital_exposure']:.2%} | {e['net_return']:.2%} |"
            )
    lines += [
        "\n## Modelos registrados: comparación sin nuevos ajustes\n",
        "Se reconstruye el contrato de features de cada modelo BTC ya registrado y se mantiene su umbral original. Todos reciben costos de esta fase y la misma política defensiva. El período es histórico observado, no test independiente.\n",
        "| Modelo | Accuracy | Bal. acc. | AUC | Loss | Neto | DD | PF | Trades | Capital medio | Costos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for version, item in registered.items():
        m, e = item["predictive"], item["economic"]
        pf = "n/a" if e["profit_factor"] is None else f"{e['profit_factor']:.2f}"
        lines.append(
            f"| {version} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['roc_auc']:.3f} | {m['log_loss']:.4f} | {e['net_return']:.2%} | {e['max_drawdown']:.2%} | {pf} | {e['trade_count']} | {e['mean_capital_exposure']:.2%} | {e['execution_cost_cash']:.2f} |"
        )
    lines += [
        "\n## Interpretación estadística\n",
        "Ninguno de los seis sistemas seleccionados por activo/objetivo superó todos los criterios de aceptación. Se conserva el original. Los candidatos de dirección quedaron en efectivo durante el período posterior; retorno y drawdown cero no acreditan selección rentable. El score de barreras BTC obtuvo accuracy 90,38% pero balanced accuracy 50%: el desbalance explica gran parte de ese porcentaje. Su AUC posterior es descriptivo y no justifica reabrir selección.\n",
        "La mejora de log-loss por calibración puede representar menor confianza, sin mayor capacidad de ordenar operaciones. Resultados positivos en algunos folds o con un objetivo específico no permiten escoger después un nuevo ganador. Los regímenes y activos están correlacionados; estos contrastes históricos no prueban una ventaja prospectiva.\n",
    ]
    lines += [
        "\nLos presupuestos se escalan con exposición media de validación, no con el período posterior. No producen igualdad exacta de exposición realizada. `stress_historical` registra fricción duplicada. No se atribuye una mejora predictiva a invertir menos capital.\n",
        "## Gráficos exportables\n",
        f"![Diferencias predictivas]({output.as_posix()}/predictive_comparison.png)\n",
        f"![Curvas netas]({output.as_posix()}/historical_equity.png)\n",
        "## Reproducción\n",
        "Instalar extras con `python -m pip install -r requirements-research.txt`. Ejecutar `python -m trading_ml frontier-study --protocol config/frontier_research.toml`. Recuperación técnica: agregar `--cache-from artifacts/frontier_studies/20261014T134018_d1e1df02`; no vuelve a ajustar modelos cuya procedencia y contrato coinciden. Documentar sin entrenar: `python scripts/record_frontier_study.py <study>`; la carpeta de análisis se crea una sola vez.\n",
        "Test futuro: `python -m trading_ml frontier-future-evaluate --study <study> --btc-raw <BTC_csv_verificado> --eth-raw <ETH_csv_verificado>`. Antes del 13/10/2027 UTC rechaza la apertura sin leer los CSV. Se requieren datos desde el mismo comienzo histórico, archivos públicos verificados y cobertura completa. No entrena, recalibra, optimiza ni activa modelos.\n",
        "La fase anterior queda documentada en [ADVANCED_RESEARCH.md](ADVANCED_RESEARCH.md); Momentum, Mean Reversion y gestión de riesgo conservan sus informes y resultados negativos.\n",
    ]
    Path("FRONTIER_RESEARCH.md").write_text("\n".join(lines), encoding="utf-8")
    write_new_json(
        output / "manifest.json",
        {
            "study_sha256": sha256(args.study / "summary.json"),
            "script_sha256": sha256(Path(__file__)),
            "additional_fits": 0,
            "files": {
                str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()
            },
            "report_sha256": sha256(Path("FRONTIER_RESEARCH.md")),
        },
    )
    print("Informe y gráficos generados sin ajustes adicionales")


if __name__ == "__main__":
    main()

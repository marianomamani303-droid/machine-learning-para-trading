from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trading_ml.common import read_json, sha256
from trading_ml.dmn.data import FEATURES as DMN_FEATURES
from trading_ml.errors import DataError
from trading_ml.mean_reversion.features import reversion_columns
from trading_ml.models.registry import ModelBundle, load_model
from trading_ml.momentum.features import momentum_columns
from trading_ml.smc.detectors import smc_columns


@dataclass(frozen=True)
class ModelDescriptor:
    """Descriptor completo y científicamente auditado de un modelo o estrategia del repositorio."""

    model_id: str
    display_name: str
    version: str
    architecture: str
    category: str  # "ML_MODEL" | "DETERMINISTIC_RULE"
    stage: str  # "Registrado (Champion)", "Registrado (Candidato)", "Congelado (Prospective)", "Experimental (Fold Final)", "Regla Determinista", "No Desplegable"
    source_kind: str  # "registry" | "frontier_frozen" | "dmn_checkpoint" | "retraining_b" | "structure_cluster" | "momentum_frozen" | "mr_frozen" | "multimarket" | "deterministic_rule" | "non_deployable"
    availability_status: str  # "DISPONIBLE" | "INCOMPATIBLE_FEED" | "NO_SERIALIZADO" | "HISTORICO_FOLD_INTERMEDIO"
    is_deployable: bool
    unavailable_reason: str | None
    symbol: str
    compatible_assets: tuple[str, ...]
    interval: str
    horizon: int
    horizon_label: str
    output_type: str  # "directional_probability" | "net_return_probability" | "barrier_probability" | "opportunity_filter_probability" | "continuous_position"
    objective_description: str
    feature_family: str
    feature_count: int
    features: list[str]
    threshold: float
    training_period: str
    selection_label_end: str
    is_calibrated: bool
    calibration_summary: str
    calibration_warning: str | None
    known_limitations: str
    artifact_path: str
    artifact_sha256: str | None = None
    role: str = ""
    bundle: ModelBundle | None = None
    extra_meta: dict[str, Any] = field(default_factory=dict)


def _describe_calibration(calibration: dict[str, Any] | str | None) -> tuple[bool, str, str | None]:
    if not calibration or not isinstance(calibration, dict):
        return (
            False,
            "NO CALIBRADA (probabilidad cruda del clasificador)",
            "Sin calibración post-hoc; la probabilidad refleja el score interno del estimador.",
        )
    method = calibration.get("method", "sigmoid")
    collapsed = bool(calibration.get("ranking_collapsed", False))
    rows = calibration.get("rows", "?")
    if collapsed:
        return (
            False,
            f"NO CALIBRADA — colapso de ranking en ajuste {method} (se usa score crudo)",
            "La calibración sigmoide colapsó el ordenamiento en validación; se reporta la probabilidad cruda.",
        )
    return (
        True,
        f"CALIBRADA ({method} en ventana pasada disjunta, n={rows})",
        None,
    )


def discover_registry_models(root: Path) -> list[ModelDescriptor]:
    """Descubre los 8 modelos XGBoost registrados en artifacts/models/."""
    models_dir = root / "models"
    if not models_dir.exists():
        return []

    champion_version: str | None = None
    if (models_dir / "champion.json").exists():
        try:
            champion_version = read_json(models_dir / "champion.json").get("version")
        except Exception:
            champion_version = None

    name_map = {
        "v20261012T143622_ff1c464c": (
            "XGBoost — Indicadores técnicos básicos (Corte Oct-2026)",
            "Registrado (Candidato Reciente)",
        ),
        "v20261012T143623_0d9b01f0": (
            "XGBoost — Indicadores técnicos + SMC (Calibrado, Corte Oct-2026)",
            "Registrado (Candidato SMC Reciente)",
        ),
        "v20261012T130540_d2bb6ac0": (
            "XGBoost — Indicadores técnicos básicos [Champion Oficial]",
            "Registrado (Champion)",
        ),
        "v20261012T134917_f88d6f93": (
            "XGBoost — Indicadores técnicos + SMC (Calibrado, Corte Jun-2025)",
            "Registrado (Candidato SMC Jun-2025)",
        ),
        "v20261012T131650_5106daf4": (
            "XGBoost — Indicadores técnicos extendidos (22 features, Jun-2025)",
            "Registrado (Candidato Extendido)",
        ),
        "v20261012T130539_2a7f3cde": (
            "XGBoost — Indicadores técnicos básicos (Corte Mar-2025)",
            "Registrado (Histórico Mar-2025)",
        ),
        "v20261012T130154_6893e846": (
            "XGBoost — Indicadores técnicos básicos (Semilla/Config B, Jun-2025)",
            "Registrado (Histórico Jun-2025)",
        ),
        "v20261012T125624_1576ecf1": (
            "XGBoost — Indicadores técnicos básicos ( línea base inicial, Jun-2025)",
            "Registrado (Histórico Inicial)",
        ),
    }

    descriptors: list[ModelDescriptor] = []
    for meta_path in sorted(models_dir.glob("v*/metadata.json"), reverse=True):
        version = meta_path.parent.name
        try:
            bundle = load_model(root, version)
        except Exception:
            continue
        meta = bundle.metadata
        contract = meta.get("contract", {})
        symbol = str(contract.get("symbol", "BTCUSDT"))
        interval = str(contract.get("interval", "1d"))
        horizon = int(contract.get("horizon", 1))
        features = list(meta.get("features", []))
        threshold = float(meta.get("config", {}).get("backtest", {}).get("threshold", 0.55))
        selection_end = str(
            meta.get("selection_label_end") or meta.get("training_last_target_time") or ""
        )
        market_cfg = meta.get("config", {}).get("market", {})
        train_start = str(market_cfg.get("start", "2020-01-01"))
        training_period = f"{train_start} → {selection_end[:10]}"

        calibration = meta.get("calibration")
        is_cal, cal_summary, cal_warn = _describe_calibration(calibration)

        feat_cfg = contract.get("features", {})
        if feat_cfg.get("smc"):
            feature_family = "Técnicos Base (14) + Estructura SMC (32) = 46 features"
        elif feat_cfg.get("enhanced"):
            feature_family = "Técnicos Extendidos (22 features)"
        else:
            feature_family = "Técnicos Base (14 features: retornos, rango, volumen, RSI, MACD, SMA)"

        display_name, stage = name_map.get(
            version,
            (f"XGBoost — {feature_family[:28]} ({version})", "Registrado"),
        )
        if version == champion_version and "Champion" not in display_name:
            display_name += " [Champion]"

        descriptors.append(
            ModelDescriptor(
                model_id=f"registry:{version}",
                display_name=display_name,
                version=version,
                architecture="XGBoost (XGBClassifier)",
                category="ML_MODEL",
                stage=stage,
                source_kind="registry",
                availability_status="DISPONIBLE",
                is_deployable=True,
                unavailable_reason=None,
                symbol=symbol,
                compatible_assets=(symbol,),
                interval=interval,
                horizon=horizon,
                horizon_label=f"{horizon} día" if horizon == 1 else f"{horizon} días",
                output_type="directional_probability",
                objective_description="Clasificación binaria direccional (close[t+1] > close[t]) optimizando log-loss",
                feature_family=feature_family,
                feature_count=len(features),
                features=features,
                threshold=threshold,
                training_period=training_period,
                selection_label_end=selection_end,
                is_calibrated=is_cal,
                calibration_summary=cal_summary,
                calibration_warning=cal_warn,
                known_limitations=(
                    "Entrenado únicamente en BTCUSDT 1d; banda neutral de abstención [0.45, 0.55]. "
                    + ("Probabilidad no calibrada post-hoc." if not is_cal else "Calibración sigmoide ajustada en 91 días.")
                ),
                artifact_path=str((bundle.directory / "model.ubj").as_posix()),
                artifact_sha256=str(meta.get("model_sha256", "")),
                role=stage,
                bundle=bundle,
            )
        )

    priority = {
        "v20261012T143622_ff1c464c": 0,
        "v20261012T143623_0d9b01f0": 1,
        "v20261012T130540_d2bb6ac0": 2,
        "v20261012T134917_f88d6f93": 3,
        "v20261012T131650_5106daf4": 4,
    }
    return sorted(descriptors, key=lambda d: (priority.get(d.version, 9), d.version))


def discover_frontier_models(root: Path) -> list[ModelDescriptor]:
    """Descubre todos los modelos congelados en artifacts/frontier_studies/20261014T134646_e17d06e3/frozen/."""
    prospective_path = (
        root / "frontier_studies" / "20261014T134646_e17d06e3" / "prospective.json"
    )
    if not prospective_path.exists():
        return []
    study_dir = prospective_path.parent
    try:
        plan = read_json(prospective_path)
    except Exception:
        return []

    cutoff = str(plan.get("cutoff", "2026-10-01T00:00:00+00:00"))
    systems = plan.get("systems", {})
    descriptors: list[ModelDescriptor] = []

    objective_meta = {
        "direction_h1": (
            1,
            "1 día",
            "directional_probability",
            "Predicción direccional a 1 día (close[t+1] > close[t])",
        ),
        "net_h3": (
            3,
            "3 días (retorno neto)",
            "net_return_probability",
            "Probabilidad de retorno neto positivo a 3 días descontando costos de ejecución",
        ),
        "barrier_close_h5": (
            5,
            "5 días (barrera TP/SL 2xATR)",
            "barrier_probability",
            "Probabilidad de tocar barrera Take-Profit (+2 ATR al cierre) antes de Stop-Loss (-2 ATR) o 5 días",
        ),
    }

    arch_names = {
        "random_forest_cross": ("Random Forest", "RandomForestClassifier (Flujo + Multi-activo)"),
        "logistic_local": ("Regresión Logística", "LogisticRegression L2 + StandardScaler (Local)"),
        "logistic_cross": ("Regresión Logística Cruzada", "LogisticRegression L2 + StandardScaler (Multi-activo)"),
        "xgboost_cross": ("XGBoost Cruzado", "XGBClassifier (Flujo + Multi-activo)"),
        "lightgbm_cross": ("LightGBM Cruzado", "LGBMClassifier (Flujo + Multi-activo)"),
        "catboost_cross": ("CatBoost Cruzado", "CatBoostClassifier (Flujo + Multi-activo)"),
        "stacking": ("Ensemble Stacking (5 modelos)", "Meta-LogisticRegression sobre LR+XGB+RF+LGBM+CatBoost"),
    }

    obj_short = {
        "direction_h1": "Predicción direccional (1d)",
        "net_h3": "Retorno neto positivo (3d)",
        "barrier_close_h5": "Barrera TP/SL al cierre (5d)",
    }

    for system_key, state in systems.items():
        symbol, objective = system_key.split("__")
        if objective not in objective_meta:
            continue
        horizon, horizon_label, output_type, obj_desc = objective_meta[objective]
        selected = state.get("selected", {})
        selected_recipe = str(selected.get("recipe", ""))
        threshold = float(selected.get("threshold", 0.55))
        components = state.get("components", {})

        # 1. Modelo seleccionado del sistema (incluyendo stacking si aplica)
        if selected_recipe == "stacking" and "stack" in state:
            stack_info = state["stack"]
            stack_path = study_dir / Path(stack_info["path"])
            if stack_path.exists() and sha256(stack_path) == stack_info["sha256"]:
                first_comp = next(iter(components.values()))
                features = list(first_comp.get("columns", []))
                short_arch, full_arch = arch_names["stacking"]
                version = f"frontier_20261014_{symbol}_{objective}_stacking"
                descriptors.append(
                    ModelDescriptor(
                        model_id=f"frontier:{study_dir.name}:{system_key}:stacking",
                        display_name=f"{short_arch} — {obj_short[objective]} [{symbol}]",
                        version=version,
                        architecture=full_arch,
                        category="ML_MODEL",
                        stage="Congelado (Prospective Frontier)",
                        source_kind="frontier_frozen",
                        availability_status="DISPONIBLE",
                        is_deployable=True,
                        unavailable_reason=None,
                        symbol=symbol,
                        compatible_assets=(symbol,),
                        interval="1d",
                        horizon=horizon,
                        horizon_label=horizon_label,
                        output_type=output_type,
                        objective_description=obj_desc,
                        feature_family="Core + Microestructura de Flujo + Momentum + Cross-Asset (35 features)",
                        feature_count=len(features),
                        features=features,
                        threshold=threshold,
                        training_period=f"2020-01-01 → {cutoff[:10]}",
                        selection_label_end=cutoff,
                        is_calibrated=True,
                        calibration_summary="CALIBRADA (Meta-modelo logístico C=0.1 entrenado en ventana de calibración disjunta)",
                        calibration_warning=None,
                        known_limitations=(
                            "Objetivo de retorno neto a 3 días (no comparable directamente con dirección a 1 día). "
                            "Requiere velas cerradas alineadas de BTCUSDT y ETHUSDT."
                        ),
                        artifact_path=str(stack_path.as_posix()),
                        artifact_sha256=str(stack_info["sha256"]),
                        role=f"Frozen Frontier ({selected_recipe})",
                        extra_meta={
                            "frontier_dir": str(study_dir),
                            "system_key": system_key,
                            "recipe": "stacking",
                            "state": state,
                        },
                    )
                )

        # 2. Componentes individuales congelados
        for comp_name, comp in components.items():
            model_path = study_dir / Path(comp["path"])
            if not model_path.exists():
                continue
            try:
                if sha256(model_path) != comp["sha256"]:
                    continue
            except Exception:
                continue
            features = list(comp.get("columns", []))
            is_cal, cal_summary, cal_warn = _describe_calibration(comp.get("calibration"))
            short_arch, full_arch = arch_names.get(comp_name, (comp_name, comp_name))
            is_primary = comp_name == selected_recipe
            stage = (
                "Congelado (Seleccionado Frontier)"
                if is_primary
                else "Congelado (Componente Ensemble Frontier)"
            )
            version = f"frontier_20261014_{symbol}_{objective}_{comp_name}"
            descriptors.append(
                ModelDescriptor(
                    model_id=f"frontier:{study_dir.name}:{system_key}:{comp_name}",
                    display_name=f"{short_arch} — {obj_short[objective]} [{symbol}]",
                    version=version,
                    architecture=full_arch,
                    category="ML_MODEL",
                    stage=stage,
                    source_kind="frontier_frozen",
                    availability_status="DISPONIBLE",
                    is_deployable=True,
                    unavailable_reason=None,
                    symbol=symbol,
                    compatible_assets=(symbol,),
                    interval="1d",
                    horizon=horizon,
                    horizon_label=horizon_label,
                    output_type=output_type,
                    objective_description=obj_desc,
                    feature_family=(
                        "Core + Flujo + Momentum + Cross-Asset (35 features)"
                        if len(features) == 35
                        else "Core + Flujo + Momentum Local (28 features)"
                    ),
                    feature_count=len(features),
                    features=features,
                    threshold=threshold,
                    training_period=f"2020-01-01 → {cutoff[:10]}",
                    selection_label_end=cutoff,
                    is_calibrated=is_cal,
                    calibration_summary=cal_summary,
                    calibration_warning=cal_warn,
                    known_limitations=(
                        f"Entrenado para {obj_short[objective]}. "
                        + (
                            "Requiere velas cerradas alineadas de BTCUSDT y ETHUSDT."
                            if len(features) == 35
                            else "Usa microestructura de volumen taker de Binance."
                        )
                    ),
                    artifact_path=str(model_path.as_posix()),
                    artifact_sha256=str(comp["sha256"]),
                    role=stage,
                    extra_meta={
                        "frontier_dir": str(study_dir),
                        "system_key": system_key,
                        "recipe": comp_name,
                        "component": comp,
                    },
                )
            )

    return descriptors


def discover_dmn_models(root: Path) -> list[ModelDescriptor]:
    """Descubre los modelos Deep Momentum Network (LSTM, MLP, Linear y control XGBoost) del último fold."""
    dmn_root = root / "dmn_studies"
    if not dmn_root.exists():
        return []

    studies = [
        (
            "20261016T204450_9f760c70",
            "Long-Only [0, 1]",
            "long_only",
            ("42", "1729"),
        ),
        (
            "20261016T204723_6818bead",
            "Long/Short [-1, 1]",
            "long_short",
            ("42",),
        ),
    ]

    recipe_labels = {
        "lstm_sharpe_net": (
            "LSTM (DMN) — Gestión dinámica de posiciones Sharpe Neto",
            "MomentumNetwork (LSTM unidireccional, hidden=16, seq=63)",
            "Maximizar ratio de Sharpe neto de costos de transacción (ecuación diferenciable)",
        ),
        "lstm_sharpe_gross": (
            "LSTM (DMN) — Gestión dinámica de posiciones Sharpe Bruto",
            "MomentumNetwork (LSTM unidireccional, hidden=16, seq=63)",
            "Maximizar ratio de Sharpe bruto sin penalización de rotación en la pérdida",
        ),
        "lstm_mse": (
            "LSTM — Aprendizaje secuencial de retorno/volatilidad (MSE)",
            "MomentumNetwork (LSTM unidireccional, hidden=16, seq=63)",
            "Minimizar error cuadrático medio (MSE) del retorno estandarizado por volatilidad",
        ),
        "mlp_sharpe_net": (
            "Red Neuronal MLP (DMN) — Gestión dinámica de posiciones Sharpe Neto",
            "MomentumNetwork (MLP Tanh, hidden=16)",
            "Maximizar ratio de Sharpe neto de costos sin recurrencia temporal",
        ),
        "linear_sharpe_net": (
            "Red Lineal (DMN) — Gestión dinámica de posiciones Sharpe Neto",
            "MomentumNetwork (Capa Lineal directa)",
            "Maximizar ratio de Sharpe neto mediante combinación lineal de las 8 señales",
        ),
        "xgboost_binary": (
            "XGBoost (Control DMN) — Clasificación direccional sobre features DMN",
            "XGBClassifier (max_depth=2, control supervisado DMN)",
            "Minimizar log-loss direccional sobre las 8 features multiescala DMN",
        ),
    }

    descriptors: list[ModelDescriptor] = []
    for study_id, mode_label, pos_mode, allowed_seeds in studies:
        study_dir = dmn_root / study_id
        fold_dir = study_dir / "fold_04"
        plan_path = study_dir / "plan.json"
        if not fold_dir.exists() or not plan_path.exists():
            continue
        try:
            plan = read_json(plan_path)
        except Exception:
            continue
        risk_cfg = plan["protocol"]["risk"]
        seq_len = int(plan["protocol"]["training"]["sequence_length"])

        for run_dir in sorted(fold_dir.iterdir()):
            if not run_dir.is_dir() or "__seed_" not in run_dir.name:
                continue
            recipe_id, seed_str = run_dir.name.split("__seed_")
            if seed_str not in allowed_seeds or recipe_id not in recipe_labels:
                continue
            completed_path = run_dir / "completed.json"
            if not completed_path.exists():
                continue
            try:
                completed = read_json(completed_path)
                meta = completed["metadata"]
            except Exception:
                continue

            is_xgb = recipe_id == "xgboost_binary"
            weight_file = run_dir / ("model.ubj" if is_xgb else "weights.pt")
            if not weight_file.exists() or sha256(weight_file) != meta.get("weights_sha256"):
                continue

            base_title, arch_desc, obj_desc = recipe_labels[recipe_id]
            val_end = str(meta.get("validation_last_target", "2025-07-09T00:00:00+00:00"))
            train_first = str(meta.get("train_first", "2020-09-09"))[:10]

            for symbol in ("BTCUSDT", "ETHUSDT"):
                version = f"dmn_{study_id[:8]}_f04_{recipe_id}_s{seed_str}_{symbol}"
                descriptors.append(
                    ModelDescriptor(
                        model_id=f"dmn:{study_id}:fold_04:{run_dir.name}:{symbol}",
                        display_name=f"{base_title} ({mode_label}, seed {seed_str}) [{symbol}]",
                        version=version,
                        architecture=arch_desc,
                        category="ML_MODEL",
                        stage=f"Experimental DMN ({mode_label}, Fold 4)",
                        source_kind="dmn_checkpoint",
                        availability_status="DISPONIBLE",
                        is_deployable=True,
                        unavailable_reason=None,
                        symbol=symbol,
                        compatible_assets=("BTCUSDT", "ETHUSDT"),
                        interval="1d",
                        horizon=1,
                        horizon_label="1 día (rebalanceo diario de posición)",
                        output_type="continuous_position",
                        objective_description=obj_desc,
                        feature_family=f"DMN Multiescala (retornos normalizados 1..252d + MACD 8/24, 16/48, 32/96; ventana={seq_len}d)",
                        feature_count=len(DMN_FEATURES),
                        features=list(DMN_FEATURES),
                        threshold=0.50,
                        training_period=f"{train_first} → {val_end[:10]}",
                        selection_label_end=val_end,
                        is_calibrated=False,
                        calibration_summary="SALIDA DE POSICIÓN CONTINUA (no es una probabilidad calibrada)",
                        calibration_warning="Produce señal de posición continua y peso escalado por volatilidad objetivo (15% anual), no probabilidad de subida.",
                        known_limitations=(
                            "Requiere al menos 380 velas diarias cerradas para el calentamiento de MACD_32_96 (252d) y secuencia LSTM (63d). "
                            "En el estudio histórico no superó significativamente a TSMOM/Long escalado."
                        ),
                        artifact_path=str(weight_file.as_posix()),
                        artifact_sha256=str(meta.get("weights_sha256", "")),
                        role=f"DMN {mode_label}",
                        extra_meta={
                            "run_dir": str(run_dir),
                            "metadata": meta,
                            "risk": risk_cfg,
                            "sequence_length": seq_len,
                            "position_mode": pos_mode,
                            "is_xgb": is_xgb,
                        },
                    )
                )

    return descriptors


def discover_retraining_and_regime_models(root: Path) -> list[ModelDescriptor]:
    """Descubre modelos del estudio controlado con historial ampliado (Arm B) y clustering de regímenes."""
    descriptors: list[ModelDescriptor] = []
    rc_fits = root / "retraining_comparison" / "20261017_controlled" / "fits"

    rc_specs = {
        "logistic": (
            "Regresión Logística — Indicadores técnicos (Historial Ampliado B)",
            "LogisticRegression L2 + StandardScaler",
            "Técnicos Base (14 features)",
        ),
        "catboost_cross": (
            "CatBoost — Predicción direccional multi-activo (Historial Ampliado B)",
            "CatBoostClassifier",
            "Core + Flujo + Momentum + Cross-Asset (35 features)",
        ),
        "lightgbm_cross": (
            "LightGBM — Predicción direccional multi-activo (Historial Ampliado B)",
            "LGBMClassifier",
            "Core + Flujo + Momentum + Cross-Asset (35 features)",
        ),
        "random_forest_cross": (
            "Random Forest — Predicción direccional multi-activo (Historial Ampliado B)",
            "RandomForestClassifier",
            "Core + Flujo + Momentum + Cross-Asset (35 features)",
        ),
        "structure": (
            "XGBoost — Estructura de mercado extendida + SMC (Historial Ampliado B)",
            "XGBClassifier (57 features: técnicos + SMC + impulsos/retests)",
            "Técnicos + SMC + Extensión de Estructura (57 features)",
        ),
        "regime_rules": (
            "XGBoost — Condicionado por régimen de tendencia y volatilidad (Historial Ampliado B)",
            "XGBClassifier (20 features: técnicos + estados de régimen)",
            "Técnicos Base + Regímenes de Tendencia/Volatilidad (20 features)",
        ),
        "specialists_rules": (
            "Modelo Híbrido por Régimen — Especialistas XGBoost con shrinkage al global (Historial Ampliado B)",
            "Ensemble Jerárquico (XGBoost Global + 8 Expertos por Régimen dirección×volatilidad)",
            "Técnicos Base + Partición por Régimen sr_state (14 features + estado)",
        ),
    }

    if rc_fits.exists():
        for symbol in ("BTCUSDT", "ETHUSDT"):
            fold_dir = rc_fits / f"{symbol}_1d_f2_s42_B"
            if not fold_dir.exists():
                continue
            for recipe_key, (title, arch, fam) in rc_specs.items():
                rec_dir = fold_dir / recipe_key
                fit_path = rec_dir / "fit.json"
                model_path = rec_dir / "model.joblib"
                if not fit_path.exists() or not model_path.exists():
                    continue
                try:
                    fit_meta = read_json(fit_path)
                except Exception:
                    continue
                cols = list(fit_meta.get("columns", []))
                cal = fit_meta.get("probability_calibration")
                is_cal, cal_summary, cal_warn = _describe_calibration(cal)
                train_first = str(fit_meta.get("train_first", "2017-10-18"))[:10]
                train_last = str(fit_meta.get("train_last_target", "2026-02-23T00:00:00+00:00"))
                version = f"retrainB_f2_s42_{symbol}_{recipe_key}"

                descriptors.append(
                    ModelDescriptor(
                        model_id=f"retrain_b:{symbol}:{recipe_key}",
                        display_name=f"{title} [{symbol}]",
                        version=version,
                        architecture=arch,
                        category="ML_MODEL",
                        stage="Experimental (Historial Ampliado Arm B, Fold Final)",
                        source_kind="retraining_b",
                        availability_status="DISPONIBLE",
                        is_deployable=True,
                        unavailable_reason=None,
                        symbol=symbol,
                        compatible_assets=(symbol,),
                        interval="1d",
                        horizon=1,
                        horizon_label="1 día",
                        output_type="directional_probability",
                        objective_description="Clasificación direccional a 1 día (close[t+1] > close[t]) entrenada con historial ampliado (2017/2020–2026)",
                        feature_family=fam,
                        feature_count=len(cols),
                        features=cols,
                        threshold=0.55,
                        training_period=f"{train_first} → {train_last[:10]}",
                        selection_label_end=train_last,
                        is_calibrated=is_cal,
                        calibration_summary=cal_summary,
                        calibration_warning=cal_warn,
                        known_limitations=(
                            "Checkpoint del último fold walk-forward (corte Feb-2026). "
                            + (
                                "Pondera predicción del experto del régimen actual con el modelo global según tamaño muestral."
                                if recipe_key == "specialists_rules"
                                else ""
                            )
                        ),
                        artifact_path=str(model_path.as_posix()),
                        artifact_sha256=sha256(model_path),
                        role="Historial Ampliado Arm B",
                        extra_meta={
                            "rec_dir": str(rec_dir),
                            "recipe_key": recipe_key,
                            "fit_meta": fit_meta,
                        },
                    )
                )

    # Clustering de regímenes K-Means + XGBoost en structure_regime_studies
    sr_root = root / "structure_regime_studies" / "20261016T152046_8f7d1886"
    if sr_root.exists() and (sr_root / "plan.json").exists():
        try:
            sr_plan = read_json(sr_root / "plan.json")
            cols = list(sr_plan["recipes"]["regime_cluster"])
            for symbol in ("BTCUSDT", "ETHUSDT"):
                s_dir = sr_root / symbol / "fold_5" / "seed_42"
                ubj_path = s_dir / "regime_cluster.ubj"
                clusters_path = s_dir / "clusters.json"
                if ubj_path.exists() and clusters_path.exists():
                    version = f"sr_fold5_s42_{symbol}_regime_cluster"
                    descriptors.append(
                        ModelDescriptor(
                            model_id=f"sr_cluster:{symbol}:fold_5:seed_42",
                            display_name=f"XGBoost — Regímenes por Clustering K-Means [{symbol}]",
                            version=version,
                            architecture="KMeans (3 clusters) + XGBClassifier",
                            category="ML_MODEL",
                            stage="Experimental (Estructura y Regímenes, Fold 5)",
                            source_kind="structure_cluster",
                            availability_status="DISPONIBLE",
                            is_deployable=True,
                            unavailable_reason=None,
                            symbol=symbol,
                            compatible_assets=(symbol,),
                            interval="1d",
                            horizon=1,
                            horizon_label="1 día",
                            output_type="directional_probability",
                            objective_description="Clasificación direccional a 1 día incorporando estado de cluster K-Means no supervisado",
                            feature_family="Técnicos Base (14) + One-Hot K-Means Clusters (3) = 17 features",
                            feature_count=len(cols),
                            features=cols,
                            threshold=0.55,
                            training_period="2020-01-01 → 2024-12-25",
                            selection_label_end="2025-01-01T00:00:00+00:00",
                            is_calibrated=False,
                            calibration_summary="NO CALIBRADA (probabilidad cruda de XGBoost con features de cluster)",
                            calibration_warning="Sin calibración post-hoc.",
                            known_limitations="Entrenado hasta Ene-2025; K-Means usa momentum 60d, eficiencia 20d y volatilidad 20d.",
                            artifact_path=str(ubj_path.as_posix()),
                            artifact_sha256=sha256(ubj_path),
                            role="Regime Cluster",
                            extra_meta={
                                "ubj_path": str(ubj_path),
                                "clusters_path": str(clusters_path),
                                "structure_config": sr_plan["protocol"]["structure"],
                            },
                        )
                    )
        except Exception:
            pass

    return descriptors


def discover_hybrid_filter_models(root: Path) -> list[ModelDescriptor]:
    """Descubre los filtros híbridos ML + Reglas congelados de Momentum y Mean Reversion."""
    descriptors: list[ModelDescriptor] = []

    # 1. Momentum Frozen Filters (BTCUSDT 1d)
    mom_dir = root / "momentum_studies" / "20261013T215021_338a147b"
    if (mom_dir / "frozen").exists() and (mom_dir / "plan.json").exists():
        try:
            mom_plan = read_json(mom_dir / "plan.json")
            rules_cfg = mom_plan["protocol"]["rules"]
            threshold = float(mom_plan["protocol"]["research"].get("ml_threshold", 0.55))
            mom_cols = momentum_columns()
            for lookback in (20, 60, 120, 250):
                l_dir = mom_dir / "frozen" / f"L{lookback}"
                ubj_path = l_dir / "model.ubj"
                fit_path = l_dir / "fit.json"
                if not ubj_path.exists() or not fit_path.exists():
                    continue
                fit_meta = read_json(fit_path)
                if fit_meta.get("status") != "fitted":
                    continue
                cutoff = str(fit_meta.get("cutoff", "2026-10-01T00:00:00+00:00"))
                rows = fit_meta.get("rows", 0)
                version = f"momentum_frozen_L{lookback}_xgb"
                descriptors.append(
                    ModelDescriptor(
                        model_id=f"momentum_frozen:L{lookback}",
                        display_name=f"Modelo Híbrido — Filtro XGBoost sobre Ruptura de Momentum (L{lookback}d) [BTCUSDT]",
                        version=version,
                        architecture="Filtro Híbrido: Regla Momentum + XGBClassifier",
                        category="ML_MODEL",
                        stage="Congelado (Filtro Híbrido Momentum)",
                        source_kind="momentum_frozen",
                        availability_status="DISPONIBLE",
                        is_deployable=True,
                        unavailable_reason=None,
                        symbol="BTCUSDT",
                        compatible_assets=("BTCUSDT",),
                        interval="1d",
                        horizon=1,
                        horizon_label="Oportunidad de ruptura (tenencia máx. 10d, evalúa barra siguiente)",
                        output_type="opportunity_filter_probability",
                        objective_description="Predecir si una ruptura determinista de momentum generará retorno neto positivo tras costos",
                        feature_family=f"Momentum Cross-Scale (8 features, lookback={lookback}d) + Gatillo Determinista",
                        feature_count=len(mom_cols),
                        features=mom_cols,
                        threshold=threshold,
                        training_period=f"2020-01-01 → {cutoff[:10]} (n={rows} rupturas)",
                        selection_label_end=cutoff,
                        is_calibrated=False,
                        calibration_summary=f"NO CALIBRADA (score de filtro condicional, n={rows} eventos de entrenamiento)",
                        calibration_warning="Solo emite señal de entrada cuando la regla determinista de ruptura está activa Y P(éxito) >= 0.55.",
                        known_limitations=f"Entrenado sobre {rows} eventos históricos de ruptura; permanece en espera cuando no hay ruptura activa.",
                        artifact_path=str(ubj_path.as_posix()),
                        artifact_sha256=sha256(ubj_path),
                        role=f"Filtro Híbrido Momentum L{lookback}",
                        extra_meta={
                            "ubj_path": str(ubj_path),
                            "lookback": lookback,
                            "rules_config": rules_cfg,
                            "fit_meta": fit_meta,
                        },
                    )
                )
        except Exception:
            pass

    # 2. Mean Reversion Frozen Filters (BTCUSDT 1d)
    mr_dir = root / "mean_reversion_studies" / "20261013T220145_a6d4131b"
    if (mr_dir / "frozen").exists() and (mr_dir / "plan.json").exists():
        try:
            mr_plan = read_json(mr_dir / "plan.json")
            rules_cfg = mr_plan["protocol"]["rules"]
            threshold = float(mr_plan["protocol"]["research"]["probability_threshold"])
            mr_cols = reversion_columns()
            for spec_dir in sorted((mr_dir / "frozen").iterdir()):
                if not spec_dir.is_dir():
                    continue
                parts = spec_dir.name.split("_")  # e.g. mr_L20_all or mr_L20_sideways
                lookback = int(parts[1].replace("L", ""))
                regime_mode = parts[2]
                for kind in ("xgb", "logistic"):
                    k_dir = spec_dir / kind
                    filter_json = k_dir / "filter.json"
                    if not filter_json.exists():
                        continue
                    f_state = read_json(filter_json)
                    audit = f_state.get("audit", {})
                    if audit.get("status") != "fitted":
                        continue
                    cutoff = str(audit.get("cutoff", "2026-10-01T00:00:00+00:00"))
                    rows = audit.get("rows", 0)
                    arch_label = "XGBoost" if kind == "xgb" else "Regresión Logística"
                    reg_label = "Todo Régimen" if regime_mode == "all" else "Solo Rango Lateral"
                    version = f"mr_frozen_{spec_dir.name}_{kind}"
                    artifact_file = k_dir / "model.ubj" if kind == "xgb" else filter_json
                    descriptors.append(
                        ModelDescriptor(
                            model_id=f"mr_frozen:{spec_dir.name}:{kind}",
                            display_name=f"Modelo Híbrido — Filtro {arch_label} sobre Mean Reversion (L{lookback}d, {reg_label}) [BTCUSDT]",
                            version=version,
                            architecture=f"Filtro Híbrido: Regla Mean Reversion + {arch_label}",
                            category="ML_MODEL",
                            stage="Congelado (Filtro Híbrido Mean Reversion)",
                            source_kind="mr_frozen",
                            availability_status="DISPONIBLE",
                            is_deployable=True,
                            unavailable_reason=None,
                            symbol="BTCUSDT",
                            compatible_assets=("BTCUSDT",),
                            interval="1d",
                            horizon=1,
                            horizon_label="Oportunidad de reversión (tenencia máx. 5d, evalúa barra siguiente)",
                            output_type="opportunity_filter_probability",
                            objective_description="Predecir si una sobreventa (Z-Score <= -1.5 y RSI <= 0.35) revertirá a la media antes del stop o timeout",
                            feature_family=f"Mean Reversion (8 features, lookback={lookback}d, filtro={regime_mode})",
                            feature_count=len(mr_cols),
                            features=mr_cols,
                            threshold=threshold,
                            training_period=f"2020-01-01 → {cutoff[:10]} (n={rows} oportunidades)",
                            selection_label_end=cutoff,
                            is_calibrated=False,
                            calibration_summary=f"NO CALIBRADA (probabilidad condicional sobre {rows} oportunidades históricas)",
                            calibration_warning="Solo activa entrada cuando Z-Score <= -1.5, RSI <= 0.35 Y la probabilidad del filtro >= 0.55.",
                            known_limitations=f"Entrenado sobre {rows} eventos de sobreventa; fuera de eventos de sobreventa el sistema permanece en espera.",
                            artifact_path=str(artifact_file.as_posix()),
                            artifact_sha256=sha256(artifact_file),
                            role=f"Filtro Mean Reversion L{lookback} ({reg_label})",
                            extra_meta={
                                "filter_dir": str(k_dir),
                                "lookback": lookback,
                                "sideways_only": regime_mode == "sideways",
                                "rules_config": rules_cfg,
                            },
                        )
                    )
        except Exception:
            pass

    return descriptors


def discover_multimarket_models(root: Path) -> list[ModelDescriptor]:
    """Descubre los modelos XGBoost multimercado registrados en artifacts/multimarket/models/."""
    mm_root = root / "multimarket" / "models"
    if not mm_root.exists():
        return []

    descriptors: list[ModelDescriptor] = []
    for json_path in sorted(mm_root.rglob("model.json"), reverse=True):
        ubj_path = json_path.with_suffix(".ubj")
        if not ubj_path.exists():
            continue
        try:
            meta = read_json(json_path)
            if sha256(ubj_path) != meta.get("model_sha256"):
                continue
        except Exception:
            continue

        symbol = str(meta.get("symbol", ""))
        interval = str(meta.get("interval", "1d"))
        spec = meta.get("specification", {})
        cols = list(spec.get("columns", []))
        horizon = int(spec.get("horizon", 1))
        cutoff = str(meta.get("cutoff", "2024-01-01T00:00:00+00:00"))
        last_cal = str(meta.get("last_calibration_target") or meta.get("last_train_target") or cutoff)
        is_cal, cal_summary, cal_warn = _describe_calibration(meta.get("calibration"))
        version = f"mm_{symbol}_{interval}_{json_path.parent.name}"
        market_type = meta.get("instrument", {}).get("market", "crypto")

        descriptors.append(
            ModelDescriptor(
                model_id=f"multimarket:{symbol}:{interval}:{json_path.parent.name}",
                display_name=f"XGBoost Multimercado — Acción del precio sin volumen ({ cutoff[:10] }) [{symbol}]",
                version=version,
                architecture="XGBClassifier (Esquema Multimarket Price v1)",
                category="ML_MODEL",
                stage=f"Candidato Multimercado ({market_type.upper()})",
                source_kind="multimarket",
                availability_status="DISPONIBLE",
                is_deployable=True,
                unavailable_reason=None,
                symbol=symbol,
                compatible_assets=(symbol,),
                interval=interval,
                horizon=horizon,
                horizon_label=f"{horizon} día" if horizon == 1 else f"{horizon} días",
                output_type="directional_probability",
                objective_description=f"Clasificación direccional a {horizon} día(s) usando exclusivamente features de precio OHLC sin volumen",
                feature_family="Multimarket Price v1 (13 features OHLC sin volume_ratio)",
                feature_count=len(cols),
                features=cols,
                threshold=0.55,
                training_period=f"Hasta {last_cal[:10]} (n={meta.get('fit_rows', '?')} filas)",
                selection_label_end=last_cal,
                is_calibrated=is_cal,
                calibration_summary=cal_summary,
                calibration_warning=cal_warn,
                known_limitations=(
                    "Omite volumen para compatibilidad entre Cripto y Forex. "
                    + (
                        "En pares Forex utiliza datos diarios verificados (Yahoo/FXCM/Dukascopy)."
                        if market_type == "forex"
                        else ""
                    )
                ),
                artifact_path=str(ubj_path.as_posix()),
                artifact_sha256=str(meta.get("model_sha256", "")),
                role=f"Multimarket {symbol}",
                extra_meta={"model_path": str(ubj_path), "metadata": meta},
            )
        )

    return descriptors


def discover_deterministic_strategies(_root: Path | None = None) -> list[ModelDescriptor]:
    """Devuelve las estrategias deterministas del repositorio claramente diferenciadas de ML."""
    rules: list[ModelDescriptor] = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        rules.extend(
            [
                ModelDescriptor(
                    model_id=f"rule:smc_rules_v1:{symbol}",
                    display_name=f"Regla Determinista — Smart Money Concepts (BOS/CHoCH/Sweep/FVG/OB) [{symbol}]",
                    version=f"smc_rules_v1_{symbol}",
                    architecture="Heurística Determinista Ponderada (Sin parámetros aprendidos)",
                    category="DETERMINISTIC_RULE",
                    stage="Regla Baseline Determinista",
                    source_kind="deterministic_rule",
                    availability_status="DISPONIBLE",
                    is_deployable=True,
                    unavailable_reason=None,
                    symbol=symbol,
                    compatible_assets=("BTCUSDT", "ETHUSDT"),
                    interval="1d",
                    horizon=1,
                    horizon_label="1 día",
                    output_type="directional_probability",
                    objective_description="Puntuar evidencia estructural alcista vs bajista (BOS, CHoCH, barridos de liquidez y toques en FVG/Order Blocks)",
                    feature_family="Detectores causales SMC (32 columnas estructurales)",
                    feature_count=len(smc_columns()),
                    features=smc_columns(),
                    threshold=0.55,
                    training_period="Sin entrenamiento (regla fija predeclarada)",
                    selection_label_end="1970-01-01T00:00:00+00:00",
                    is_calibrated=False,
                    calibration_summary="NO CALIBRADA (confianza heurística fija en [0.20, 0.80], no es frecuencia empírica)",
                    calibration_warning="Estrategia determinista basada en reglas; su score no proviene de un modelo de machine learning.",
                    known_limitations="Abstiene cuando hay evidencia mixta alcista/bajista en la misma vela o solo tendencia sin evento confirmado.",
                    artifact_path="src/trading_ml/smc/rules.py",
                    role="Regla Determinista SMC",
                    extra_meta={"rule_kind": "smc_rules"},
                ),
                ModelDescriptor(
                    model_id=f"rule:tsmom_252:{symbol}",
                    display_name=f"Regla Determinista — Time-Series Momentum 252 días (TSMOM) [{symbol}]",
                    version=f"tsmom_252d_{symbol}",
                    architecture="Regla Determinista de Seguimiento de Tendencia (sign(retorno_252d))",
                    category="DETERMINISTIC_RULE",
                    stage="Regla Baseline Determinista",
                    source_kind="deterministic_rule",
                    availability_status="DISPONIBLE",
                    is_deployable=True,
                    unavailable_reason=None,
                    symbol=symbol,
                    compatible_assets=("BTCUSDT", "ETHUSDT"),
                    interval="1d",
                    horizon=1,
                    horizon_label="1 día (rebalanceo diario de posición)",
                    output_type="continuous_position",
                    objective_description="Capturar tendencia secular de 12 meses escalando la exposición por volatilidad objetivo (15% anual)",
                    feature_family="Retorno acumulado a 252 días + Volatilidad EWMA 60d",
                    feature_count=2,
                    features=["return_252", "daily_volatility"],
                    threshold=0.0,
                    training_period="Sin entrenamiento (regla clásica Moskowitz et al.)",
                    selection_label_end="1970-01-01T00:00:00+00:00",
                    is_calibrated=False,
                    calibration_summary="REGLA DE POSICIÓN DETERMINISTA (sin probabilidad)",
                    calibration_warning="No es un modelo de IA; sirve como línea base rigurosa para evaluar a los modelos DMN/LSTM.",
                    known_limitations="Baja frecuencia de giro; sufre en mercados laterales con cambios bruscos de régimen.",
                    artifact_path="src/trading_ml/dmn/data.py",
                    role="Regla Determinista TSMOM",
                    extra_meta={"rule_kind": "tsmom_252"},
                ),
                ModelDescriptor(
                    model_id=f"rule:macd_multiscale:{symbol}",
                    display_name=f"Regla Determinista — Señal Continua MACD Multiescala (8/24, 16/48, 32/96) [{symbol}]",
                    version=f"macd_multiscale_{symbol}",
                    architecture="Regla Determinista Continua phi(MACD) (Baz et al.)",
                    category="DETERMINISTIC_RULE",
                    stage="Regla Baseline Determinista",
                    source_kind="deterministic_rule",
                    availability_status="DISPONIBLE",
                    is_deployable=True,
                    unavailable_reason=None,
                    symbol=symbol,
                    compatible_assets=("BTCUSDT", "ETHUSDT"),
                    interval="1d",
                    horizon=1,
                    horizon_label="1 día (rebalanceo diario de posición)",
                    output_type="continuous_position",
                    objective_description="Asignar posición continua promediando tres escalas de MACD transformadas por función de respuesta acotada",
                    feature_family="MACD normalizado en escalas (8,24), (16,48) y (32,96)",
                    feature_count=3,
                    features=["macd_8_24", "macd_16_48", "macd_32_96"],
                    threshold=0.0,
                    training_period="Sin entrenamiento (regla paramétrica fija)",
                    selection_label_end="1970-01-01T00:00:00+00:00",
                    is_calibrated=False,
                    calibration_summary="REGLA DE POSICIÓN CONTINUA (sin probabilidad)",
                    calibration_warning="Estrategia determinista de referencia para comparar contra redes neuronales DMN.",
                    known_limitations="Requiere al menos 315 velas cerradas para estabilizar la normalización de 252 días.",
                    artifact_path="src/trading_ml/dmn/data.py",
                    role="Regla Determinista MACD",
                    extra_meta={"rule_kind": "macd_multiscale"},
                ),
            ]
        )
    return rules


def discover_non_deployable_models(root: Path) -> list[ModelDescriptor]:
    """Documenta explícitamente los estudios/experimentos que no pueden desplegarse en vivo y por qué."""
    items: list[ModelDescriptor] = [
        ModelDescriptor(
            model_id="non_deployable:focused_financial_study_20261017",
            display_name="Estudio Focalizado — Variables Financieras Seleccionadas (XGB / RF / Logística)",
            version="focused_financial_20261017T232854_36d523ee",
            architecture="XGBoost, Random Forest y Regresión Logística (Walk-Forward Anidado)",
            category="ML_MODEL",
            stage="No Desplegable (Solo Evaluación Histórica)",
            source_kind="non_deployable",
            availability_status="NO_SERIALIZADO",
            is_deployable=False,
            unavailable_reason=(
                "El estudio guardó predicciones fuera de muestra (`predictions.csv`), decisiones anidadas (`nested_decisions.json`) "
                "y métricas económicas (`economics.json`), pero NO serializó pesos entrenados (`.ubj`/`.joblib`) en disco. "
                "Además, requiere series externas en tiempo real de S&P 500 y futuros de volatilidad CBOE (VX/VIX)."
            ),
            symbol="BTCUSDT",
            compatible_assets=("BTCUSDT", "ETHUSDT", "EURUSD"),
            interval="1d",
            horizon=1,
            horizon_label="1 día",
            output_type="directional_probability",
            objective_description="Evaluar ablación focalizada de liquidez, volatilidad de Parkinson, S&P 500 y basis VX/VIX",
            feature_family="Técnicos Base + Liquidez/Amihud + Parkinson + S&P 500 / VX-VIX",
            feature_count=20,
            features=[],
            threshold=0.55,
            training_period="Walk-forward purgado 2018–2026",
            selection_label_end="2026-10-01T00:00:00+00:00",
            is_calibrated=False,
            calibration_summary="No aplicable (sin artefacto de modelo serializado)",
            calibration_warning=None,
            known_limitations="Sin pesos guardados ni feed en vivo de derivados CBOE VX/VIX.",
            artifact_path="artifacts/focused_financial_study/20261017T232854_36d523ee/summary.json",
            role="Estudio de Investigación (Sin Pesos)",
        ),
        ModelDescriptor(
            model_id="non_deployable:financial_feature_ablation_20261017",
            display_name="Estudio de Ablación — Familias Financieras y Controles con Ruido",
            version="ablation_20261017T224928_57040c88",
            architecture="XGBClassifier (Ablación Walk-Forward con Controles Gaussianos/Permutados)",
            category="ML_MODEL",
            stage="No Desplegable (Solo Evaluación Histórica)",
            source_kind="non_deployable",
            availability_status="NO_SERIALIZADO",
            is_deployable=False,
            unavailable_reason=(
                "Experimento estadístico de ablación que registró contrastes de ROC-AUC y log-loss en `predictions.csv` "
                "sin persistir modelos `.ubj` individuales."
            ),
            symbol="BTCUSDT",
            compatible_assets=("BTCUSDT", "ETHUSDT", "EURUSD"),
            interval="1d",
            horizon=1,
            horizon_label="1 día",
            output_type="directional_probability",
            objective_description="Identificar qué familias financieras mejoran log-loss y ROC-AUC frente a controles con ruido",
            feature_family="Volatilidad + Liquidez + Volumen Relativo + Funding/Basis + Cross-Asset",
            feature_count=32,
            features=[],
            threshold=0.55,
            training_period="Walk-forward purgado 2018–2026",
            selection_label_end="2026-10-01T00:00:00+00:00",
            is_calibrated=False,
            calibration_summary="No aplicable (sin artefacto serializado)",
            calibration_warning=None,
            known_limitations="Experimento de diagnóstico estadístico sin binario desplegable.",
            artifact_path="artifacts/financial_feature_ablation/20261017T224928_57040c88/summary.json",
            role="Estudio de Ablación (Sin Pesos)",
        ),
        ModelDescriptor(
            model_id="non_deployable:expansion_derivatives_panel",
            display_name="LSTM / XGBoost — Panel Cripto de 6 Activos + Derivados Perpetuos (Funding & Basis)",
            version="expansion_20261016T211432_bcec9ebe",
            architecture="MomentumNetwork (LSTM) y XGBClassifier sobre panel simultáneo de 6 criptoactivos",
            category="ML_MODEL",
            stage="Incompatible con Feed Spot Simple",
            source_kind="non_deployable",
            availability_status="INCOMPATIBLE_FEED",
            is_deployable=False,
            unavailable_reason=(
                "Aunque existen checkpoints `.pt` y `.ubj` en `artifacts/expansion/studies/20261016T211432_bcec9ebe/`, "
                "su contrato exige un tensor sincronizado de 6 activos (BTC, ETH, BNB, SOL, XRP, ADA) junto con "
                "series de `funding_rate` de 8 horas y `mark/index basis` de futuros perpetuos que no existen en el stream spot OHLCV."
            ),
            symbol="BTCUSDT",
            compatible_assets=("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"),
            interval="1d",
            horizon=1,
            horizon_label="1 día",
            output_type="continuous_position",
            objective_description="Optimizar cartera multi-activo incorporando carry de funding rate y basis de perpetuos",
            feature_family="DMN 8 features + Contexto de Mercado + Funding Rate + Basis Perpetuos (14 features)",
            feature_count=14,
            features=[],
            threshold=0.50,
            training_period="Walk-forward 2020–2026 (3 folds)",
            selection_label_end="2026-10-01T00:00:00+00:00",
            is_calibrated=False,
            calibration_summary="No desplegable en stream spot individual",
            calibration_warning=None,
            known_limitations="Requiere ingesta en tiempo real de futuros perpetuos USD-M y 6 símbolos simultáneos.",
            artifact_path="artifacts/expansion/studies/20261016T211432_bcec9ebe/plan.json",
            role="Panel Multi-Activo Derivados",
        ),
        ModelDescriptor(
            model_id="non_deployable:risk_overlay_20261013",
            display_name="Capa de Gestión de Riesgo — Volatilidad Objetivo + Drawdown Throttle",
            version="risk_study_20261013T221816_a8b7d443",
            architecture="Regla de Dimensionamiento de Posición sobre Modelos Existentes",
            category="DETERMINISTIC_RULE",
            stage="No Desplegable como Modelo Independiente",
            source_kind="non_deployable",
            availability_status="NO_SERIALIZADO",
            is_deployable=False,
            unavailable_reason=(
                "No es un modelo predictivo autónomo ni posee pesos `.ubj`; es un modulador de exposición "
                "evaluado sobre retornos históricos de `v20261012T130540_d2bb6ac0`."
            ),
            symbol="BTCUSDT",
            compatible_assets=("BTCUSDT",),
            interval="1d",
            horizon=1,
            horizon_label="1 día",
            output_type="continuous_position",
            objective_description="Reducir drawdown modulando el tamaño de posición según volatilidad realizada y racha de pérdidas",
            feature_family="Overlays de volatilidad y drawdown sobre señales de un modelo base",
            feature_count=4,
            features=[],
            threshold=0.55,
            training_period="Evaluación histórica 2022–2025",
            selection_label_end="2025-06-04T00:00:00+00:00",
            is_calibrated=False,
            calibration_summary="No aplicable",
            calibration_warning=None,
            known_limitations="Requiere acoplarse a una cartera con estado de drawdown en vivo.",
            artifact_path="artifacts/risk_studies/20261013T221816_a8b7d443/summary.json",
            role="Overlay de Riesgo",
        ),
        ModelDescriptor(
            model_id="non_deployable:intermediate_walkforward_folds",
            display_name="Checkpoints Históricos de Folds Intermedios (Validación Cruzada 2021–2024)",
            version="walkforward_folds_archive",
            architecture="XGBClassifier / Joblib (Folds intermedios de selección de hiperparámetros)",
            category="ML_MODEL",
            stage="Histórico de Selección (Reemplazado por Modelos Finales)",
            source_kind="non_deployable",
            availability_status="HISTORICO_FOLD_INTERMEDIO",
            is_deployable=False,
            unavailable_reason=(
                "Los directorios `advanced_studies`, `feature_studies`, `label_studies`, `optimization`, "
                "`predictive_studies` y `smc_studies` almacenan ~1.200 checkpoints `.ubj` entrenados solo en "
                "subperíodos antiguos (2020–2023) para validación interna. Sus arquitecturas ganadoras ya están "
                "integradas en el dashboard mediante las versiones congeladas y registradas de corte 2025/2026."
            ),
            symbol="BTCUSDT",
            compatible_assets=("BTCUSDT", "ETHUSDT"),
            interval="1d",
            horizon=1,
            horizon_label="Varios (1d..5d)",
            output_type="directional_probability",
            objective_description="Validación cruzada temporal para selección de features, etiquetas e hiperparámetros",
            feature_family="Múltiples recetas experimentales de desarrollo",
            feature_count=0,
            features=[],
            threshold=0.55,
            training_period="Subventanas históricas 2020–2024",
            selection_label_end="2024-12-31T00:00:00+00:00",
            is_calibrated=False,
            calibration_summary="Artefactos internos de selección histórica",
            calibration_warning=None,
            known_limitations="Desplegar folds intermedios antiguos introduciría obsolescencia temporal y redundancia masiva.",
            artifact_path="artifacts/",
            role="Archivo de Folds Intermedios",
        ),
    ]
    return items


def discover_all_models(
    root: Path, *, include_non_deployable: bool = False
) -> list[ModelDescriptor]:
    """Devuelve el catálogo centralizado de modelos y estrategias del repositorio."""
    deployable = (
        discover_registry_models(root)
        + discover_frontier_models(root)
        + discover_retraining_and_regime_models(root)
        + discover_dmn_models(root)
        + discover_hybrid_filter_models(root)
        + discover_multimarket_models(root)
        + discover_deterministic_strategies()
    )
    if include_non_deployable:
        return deployable + discover_non_deployable_models(root)
    return deployable


def discover_frontier_directional_models(root: Path) -> list[ModelDescriptor]:
    """Compatibilidad hacia atrás: devuelve los modelos direccionales h=1 de frontier_studies."""
    return [
        m
        for m in discover_frontier_models(root)
        if m.output_type == "directional_probability" and m.horizon == 1
    ]


def get_models_for_selection(
    root: Path,
    symbol: str,
    interval: str,
    *,
    include_rules: bool = True,
) -> tuple[list[ModelDescriptor], str | None]:
    """Filtra todos los modelos desplegables compatibles con (symbol, interval)."""
    all_models = discover_all_models(root, include_non_deployable=False)
    if not all_models:
        return [], f"No se encontraron modelos entrenados válidos en {root}."

    matching = [
        m
        for m in all_models
        if m.is_deployable
        and m.symbol.upper() == symbol.upper()
        and m.interval == interval
        and (include_rules or m.category == "ML_MODEL")
    ]
    if matching:
        return matching, None

    available_pairs = sorted({f"{m.symbol} ({m.interval})" for m in all_models if m.is_deployable})
    return (
        [],
        f"No existe ningún modelo entrenado y serializado para {symbol} en temporalidad {interval}. "
        f"Combinaciones con modelos válidos disponibles: {', '.join(available_pairs)}.",
    )


def verify_model_integrity(descriptor: ModelDescriptor) -> None:
    """Revalida la integridad criptográfica SHA-256 de un modelo antes de cargarlo o inferir."""
    if not descriptor.is_deployable:
        raise DataError(
            f"El modelo {descriptor.version} no es desplegable: {descriptor.unavailable_reason}"
        )
    if descriptor.source_kind == "deterministic_rule":
        return
    if descriptor.source_kind == "registry":
        if descriptor.bundle is None:
            raise DataError("ModelBundle ausente en descriptor de registro")
        model_path = descriptor.bundle.directory / "model.ubj"
        if sha256(model_path) != descriptor.bundle.metadata["model_sha256"]:
            raise DataError(f"Checksum inválido para {model_path}")
        return
    if descriptor.source_kind == "frontier_frozen":
        frontier_dir = Path(descriptor.extra_meta["frontier_dir"])
        if descriptor.extra_meta.get("recipe") == "stacking":
            state = descriptor.extra_meta["state"]
            stack_path = frontier_dir / Path(state["stack"]["path"])
            if sha256(stack_path) != state["stack"]["sha256"]:
                raise DataError(f"Checksum inválido en stack {stack_path}")
            for comp in state["components"].values():
                c_path = frontier_dir / Path(comp["path"])
                if sha256(c_path) != comp["sha256"]:
                    raise DataError(f"Checksum inválido en componente {c_path}")
        else:
            comp = descriptor.extra_meta["component"]
            c_path = frontier_dir / Path(comp["path"])
            if sha256(c_path) != comp["sha256"]:
                raise DataError(f"Checksum inválido en {c_path}")
        return
    if descriptor.artifact_sha256:
        path = Path(descriptor.artifact_path)
        if not path.exists() or sha256(path) != descriptor.artifact_sha256:
            raise DataError(f"Checksum inválido o archivo faltante: {path}")

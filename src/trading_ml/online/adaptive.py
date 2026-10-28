"""Mathematical adaptive models, state-space filters, and multi-target baselines for online learning.

All estimators operate strictly on causal feature vectors x_t and confirmed outcomes at t+h.
No estimator reads files, future bars, or external registries.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

PROBABILITY_EPSILON = 1e-6

TARGET_TYPES = ("direction", "expected_return", "volatility", "quantiles")

ADAPTIVE_MODEL_KINDS = (
    "beta_frequency",
    "logistic",
    "logistic_static",
    "logistic_adaptive",
    "kalman_filter",
    "return_baseline",
    "ewma_volatility",
    "har_rv",
    "volatility_baseline",
    "quantile_regression",
    "quantile_baseline",
)

COHORT_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "classic_3": ("baseline", "logistic_slow", "logistic_fast"),
    "directional_lab": (
        "baseline",
        "logistic_static",
        "logistic_slow",
        "logistic_fast",
        "logistic_adaptive",
        "kalman_linear",
    ),
    "adaptive_lab": (
        "baseline",
        "logistic_static",
        "logistic_slow",
        "logistic_fast",
        "logistic_adaptive",
        "kalman_linear",
        "return_naive_baseline",
        "ewma_volatility",
        "har_rv",
        "volatility_naive_baseline",
        "quantile_regression",
        "quantile_historical_baseline",
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    kind: str
    target_type: str
    family: str
    is_baseline: bool
    baseline_id: str
    complexity: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_SPECS: dict[str, ModelSpec] = {
    "baseline": ModelSpec(
        model_id="baseline",
        kind="beta_frequency",
        target_type="direction",
        family="conjugate_bayesian_rate",
        is_baseline=True,
        baseline_id="baseline",
        complexity="O(1)",
        description="Control direccional conjugado Beta(1,1) de frecuencia empírica causal.",
    ),
    "logistic_static": ModelSpec(
        model_id="logistic_static",
        kind="logistic_static",
        target_type="direction",
        family="static_linear_control",
        is_baseline=False,
        baseline_id="baseline",
        complexity="O(d)",
        description="Control logístico estático: aprende solo durante la ventana inicial de calentamiento y congela sus pesos.",
    ),
    "logistic_slow": ModelSpec(
        model_id="logistic_slow",
        kind="logistic",
        target_type="direction",
        family="incremental_sgd_logistic",
        is_baseline=False,
        baseline_id="baseline",
        complexity="O(d)",
        description="Regresión logística incremental SGD conservadora (lr=0.01, L2=0.001).",
    ),
    "logistic_fast": ModelSpec(
        model_id="logistic_fast",
        kind="logistic",
        target_type="direction",
        family="incremental_sgd_logistic",
        is_baseline=False,
        baseline_id="baseline",
        complexity="O(d)",
        description="Regresión logística incremental SGD rápida (lr=0.10, L2=0.001).",
    ),
    "logistic_adaptive": ModelSpec(
        model_id="logistic_adaptive",
        kind="logistic_adaptive",
        target_type="direction",
        family="adaptive_elasticnet_logistic",
        is_baseline=False,
        baseline_id="baseline",
        complexity="O(d)",
        description="Regresión logística adaptativa con estandarización Welford causal, RMSProp, Elastic Net (L1+L2), proyección L2 y detector de degradación.",
    ),
    "kalman_linear": ModelSpec(
        model_id="kalman_linear",
        kind="kalman_filter",
        target_type="expected_return",
        family="state_space_kalman",
        is_baseline=False,
        baseline_id="return_naive_baseline",
        complexity="O(d^2)",
        description="Filtro de Kalman lineal en espacio de estados con forma estabilizada de Joseph, varianza predictiva y calibración prequential causal.",
    ),
    "return_naive_baseline": ModelSpec(
        model_id="return_naive_baseline",
        kind="return_baseline",
        target_type="expected_return",
        family="return_moment_baseline",
        is_baseline=True,
        baseline_id="return_naive_baseline",
        complexity="O(1)",
        description="Control causal de retorno esperado y varianza mediante media y varianza exponencial móvil.",
    ),
    "ewma_volatility": ModelSpec(
        model_id="ewma_volatility",
        kind="ewma_volatility",
        target_type="volatility",
        family="volatility_ewma",
        is_baseline=False,
        baseline_id="volatility_naive_baseline",
        complexity="O(1)",
        description="Estimador de volatilidad condicional EWMA (RiskMetrics) combinando retorno cuadrado y rango de Parkinson causal.",
    ),
    "har_rv": ModelSpec(
        model_id="har_rv",
        kind="har_rv",
        target_type="volatility",
        family="volatility_har_rls",
        is_baseline=False,
        baseline_id="volatility_naive_baseline",
        complexity="O(p^2)",
        description="Modelo HAR-RV (Corsi) multiescala en log-varianza actualizado por mínimos cuadrados recursivos (RLS) estabilizados.",
    ),
    "volatility_naive_baseline": ModelSpec(
        model_id="volatility_naive_baseline",
        kind="volatility_baseline",
        target_type="volatility",
        family="volatility_historical_baseline",
        is_baseline=True,
        baseline_id="volatility_naive_baseline",
        complexity="O(1)",
        description="Control base de volatilidad mediante varianza realizada histórica acumulada/móvil.",
    ),
    "quantile_regression": ModelSpec(
        model_id="quantile_regression",
        kind="quantile_regression",
        target_type="quantiles",
        family="online_pinball_quantiles",
        is_baseline=False,
        baseline_id="quantile_historical_baseline",
        complexity="O(K*d)",
        description="Regresión cuantil incremental (tau=0.10, 0.50, 0.90) con pérdida pinball, precondicionamiento y reordenamiento monótono sin cruces.",
    ),
    "quantile_historical_baseline": ModelSpec(
        model_id="quantile_historical_baseline",
        kind="quantile_baseline",
        target_type="quantiles",
        family="empirical_quantiles_baseline",
        is_baseline=True,
        baseline_id="quantile_historical_baseline",
        complexity="O(W)",
        description="Control base de cuantiles condicionales mediante distribución empírica móvil de retornos confirmados.",
    ),
}


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _sigmoid(score: float) -> float:
    if score >= 0.0:
        z = math.exp(-min(35.0, score))
        p = 1.0 / (1.0 + z)
    else:
        z = math.exp(max(-35.0, score))
        p = z / (1.0 + z)
    return max(PROBABILITY_EPSILON, min(1.0 - PROBABILITY_EPSILON, p))


def _normal_cdf(z: float) -> float:
    clipped = max(-12.0, min(12.0, float(z)))
    p = 0.5 * (1.0 + math.erf(clipped / math.sqrt(2.0)))
    return max(PROBABILITY_EPSILON, min(1.0 - PROBABILITY_EPSILON, p))


def _soft_threshold(value: float, threshold: float) -> float:
    if threshold <= 0.0:
        return value
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def _project_vec_l2(weights: list[float], max_norm: float) -> list[float]:
    if max_norm <= 0.0:
        return [float(w) for w in weights]
    norm = math.sqrt(sum(float(w) * float(w) for w in weights))
    if norm <= max_norm or norm == 0.0:
        return [float(w) for w in weights]
    scale = max_norm / norm
    return [float(w) * scale for w in weights]


def _new_vec_scaler(n_features: int, clip_z: float = 5.0) -> dict[str, Any]:
    return {
        "count": 0,
        "mean": [0.0] * n_features,
        "m2": [1.0] * n_features,
        "clip_z": float(clip_z),
    }


def _standardize_vec(scaler: dict[str, Any], values: list[float]) -> list[float]:
    count = int(scaler.get("count", 0))
    means = scaler["mean"]
    m2s = scaler["m2"]
    clip_z = float(scaler.get("clip_z", 5.0))
    if count < 5:
        return [max(-clip_z, min(clip_z, float(x))) for x in values]
    out: list[float] = []
    for x, mu, m2 in zip(values, means, m2s, strict=True):
        var = max(float(m2) / max(1, count), 1e-6)
        z = (float(x) - float(mu)) / math.sqrt(var)
        out.append(max(-clip_z, min(clip_z, z)))
    return out


def _update_vec_scaler(scaler: dict[str, Any], values: list[float]) -> dict[str, Any]:
    count = int(scaler.get("count", 0)) + 1
    means = list(scaler["mean"])
    m2s = list(scaler["m2"])
    for idx, x in enumerate(values):
        val = float(x)
        old_mu = float(means[idx])
        delta = val - old_mu
        new_mu = old_mu + delta / count
        delta2 = val - new_mu
        means[idx] = float(new_mu)
        if count == 1:
            m2s[idx] = max(1e-4, delta * delta2)
        else:
            m2s[idx] = max(1e-6, float(m2s[idx]) + delta * delta2)
    return {
        "count": count,
        "mean": means,
        "m2": m2s,
        "clip_z": float(scaler.get("clip_z", 5.0)),
    }


def _enforce_spd_bounds(
    matrix: np.ndarray,
    min_eig: float = 1e-7,
    max_eig: float = 2.0,
) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    if not np.all(np.isfinite(sym)):
        dim = sym.shape[0]
        return np.eye(dim, dtype=float) * min(0.05, max_eig)
    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped = np.clip(eigvals, min_eig, max_eig)
    reconstructed = (eigvecs * clipped) @ eigvecs.T
    return 0.5 * (reconstructed + reconstructed.T)


def _check_target_id(params: dict[str, Any], target_id: str | None) -> bool:
    if not target_id:
        return True
    seen = list(params.get("seen_targets", []))
    if str(target_id) in seen:
        return False
    seen.append(str(target_id))
    if len(seen) > 128:
        seen = seen[-128:]
    params["seen_targets"] = seen
    return True


def new_adaptive_model(
    model_id: str,
    n_features: int,
    seed: int = 42,
    *,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a validated adaptive model state dictionary compatible with store.py."""
    opts = options or {}
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Modelo no registrado en catálogo adaptativo: {model_id}")
    spec = MODEL_SPECS[model_id]
    base = {
        "status": "active",
        "updates": 0,
        "n_features": int(n_features),
        "seed": int(seed),
        "kind": spec.kind,
        "target_type": spec.target_type,
    }
    kind = spec.kind
    if kind == "beta_frequency":
        return {
            **base,
            "params": {"positive": 1.0, "negative": 1.0},
        }
    if kind == "logistic":
        lr = 0.01 if model_id == "logistic_slow" else 0.1
        return {
            **base,
            "params": {
                "weights": [0.0] * n_features,
                "intercept": 0.0,
                "learning_rate": float(opts.get("learning_rate", lr)),
                "l2": float(opts.get("l2", 0.001)),
            },
        }
    if kind == "logistic_static":
        warmup_updates = int(opts.get("static_warmup_updates", 120))
        return {
            **base,
            "params": {
                "weights": [0.0] * n_features,
                "intercept": 0.0,
                "learning_rate": float(opts.get("learning_rate", 0.006)),
                "l2": float(opts.get("l2", 0.03)),
                "max_weight_norm": float(opts.get("max_weight_norm", 0.35)),
                "warmup_updates": warmup_updates,
                "weight_updates": 0,
                "frozen": warmup_updates <= 0,
                "scaler": _new_vec_scaler(n_features),
                "seen_targets": [],
            },
        }
    if kind == "logistic_adaptive":
        return {
            **base,
            "params": {
                "weights": [0.0] * n_features,
                "intercept": 0.0,
                "learning_rate": float(opts.get("learning_rate", 0.005)),
                "l1": float(opts.get("l1", 0.0015)),
                "l2": float(opts.get("l2", 0.025)),
                "max_weight_norm": float(opts.get("max_weight_norm", 0.40)),
                "rms_beta": float(opts.get("rms_beta", 0.96)),
                "decay_rate": float(opts.get("decay_rate", 0.02)),
                "grad_sq": [1.0] * n_features,
                "intercept_grad_sq": 1.0,
                "scaler": _new_vec_scaler(n_features),
                "drift_monitor": {
                    "brier_fast_ema": 0.25,
                    "brier_slow_ema": 0.25,
                    "log_loss_ema": math.log(2.0),
                    "drift_warnings": 0,
                    "degraded": False,
                },
                "seen_targets": [],
            },
        }
    if kind == "kalman_filter":
        dim = n_features + 1
        init_var = float(opts.get("kalman_init_variance", 1e-6))
        process_noise = float(opts.get("kalman_process_noise", 1e-8))
        obs_noise = float(opts.get("kalman_obs_noise", 1e-4))
        cov = (np.eye(dim, dtype=float) * init_var).tolist()
        return {
            **base,
            "params": {
                "state_mean": [0.0] * dim,
                "state_cov": cov,
                "process_noise": process_noise,
                "obs_noise": obs_noise,
                "obs_noise_min": 1e-7,
                "obs_noise_max": 0.10,
                "cov_min": 1e-10,
                "cov_max": 2.0,
                "ridge_shrinkage": float(opts.get("kalman_l2", 0.008)),
                "max_weight_norm": float(opts.get("max_weight_norm", 0.015)),
                "adaptive_obs_noise": bool(opts.get("adaptive_obs_noise", True)),
                "obs_ewma_lambda": float(opts.get("obs_ewma_lambda", 0.94)),
                "scaler": _new_vec_scaler(n_features),
                "calibrator": {
                    "intercept": 0.0,
                    "slope": 0.18,
                    "lr": 0.01,
                    "l2": 0.05,
                    "positive": 2.0,
                    "negative": 2.0,
                    "brier_ema": 0.25,
                },
                "diagnostics": {
                    "last_innovation": 0.0,
                    "last_predictive_std": math.sqrt(obs_noise + init_var),
                    "last_kalman_gain_norm": 0.0,
                    "trace_cov": float(dim * init_var),
                    "min_diag_cov": init_var,
                    "max_diag_cov": init_var,
                },
                "seen_targets": [],
            },
        }
    if kind == "return_baseline":
        return {
            **base,
            "params": {
                "mean_return": 0.0,
                "var_return": 1e-4,
                "ewma_lambda": 0.97,
                "positive": 2.0,
                "negative": 2.0,
                "seen_targets": [],
            },
        }
    if kind == "ewma_volatility":
        return {
            **base,
            "params": {
                "ewma_lambda": float(opts.get("ewma_lambda", 0.94)),
                "variance": 1e-4,
                "range_blend": 0.35,
                "var_min": 1e-8,
                "var_max": 0.25,
                "seen_targets": [],
            },
        }
    if kind == "har_rv":
        init_var = 1e-4
        init_log_var = math.log(init_var)
        return {
            **base,
            "params": {
                "forgetting_factor": float(opts.get("forgetting_factor", 0.992)),
                "theta": [0.05 * init_log_var, 0.35, 0.35, 0.25],
                "rls_cov": (np.eye(4, dtype=float) * 0.20).tolist(),
                "log_resid_var": 0.45,
                "ewma_var_floor": init_var,
                "var_min": 1e-8,
                "var_max": 0.25,
                "rv_history": [],
                "seen_targets": [],
            },
        }
    if kind == "volatility_baseline":
        return {
            **base,
            "params": {
                "mean_variance": 1e-4,
                "var_min": 1e-8,
                "var_max": 0.25,
                "seen_targets": [],
            },
        }
    if kind == "quantile_regression":
        spread = float(opts.get("init_spread", 0.006))
        return {
            **base,
            "params": {
                "taus": [0.10, 0.50, 0.90],
                "learning_rate": float(opts.get("learning_rate", 0.0005)),
                "l2": float(opts.get("l2", 0.15)),
                "biases": {"0.10": -spread, "0.50": 0.0, "0.90": spread},
                "weights": {
                    "0.10": [0.0] * n_features,
                    "0.50": [0.0] * n_features,
                    "0.90": [0.0] * n_features,
                },
                "grad_sq": {
                    "0.10": [1.0] * n_features,
                    "0.50": [1.0] * n_features,
                    "0.90": [1.0] * n_features,
                },
                "scaler": _new_vec_scaler(n_features),
                "ret_history": [],
                "raw_crossings": 0,
                "seen_targets": [],
            },
        }
    if kind == "quantile_baseline":
        return {
            **base,
            "params": {
                "window_size": int(opts.get("window_size", 250)),
                "init_spread": float(opts.get("init_spread", 0.008)),
                "history": [],
                "seen_targets": [],
            },
        }
    raise ValueError(f"Tipo de modelo adaptativo no soportado: {kind}")


def validate_adaptive_params(kind: str, params: dict[str, Any], n_features: int) -> None:
    """Validate internal parameters of adaptive models to prevent silent numerical corruption."""
    if kind in ("logistic_static", "logistic_adaptive"):
        weights = params.get("weights")
        if (
            not isinstance(weights, list)
            or len(weights) != n_features
            or not all(_finite_number(v) for v in weights)
        ):
            raise ValueError("Pesos inválidos en modelo logístico extendido")
        if not _finite_number(params.get("intercept")):
            raise ValueError("Intercepto inválido en modelo logístico extendido")
        if not _finite_number(params.get("learning_rate")) or not 0 < params["learning_rate"] <= 1:
            raise ValueError("learning_rate inválido en modelo logístico extendido")
        if not _finite_number(params.get("l2")) or not 0 <= params["l2"] <= 1:
            raise ValueError("l2 inválido en modelo logístico extendido")
        scaler = params.get("scaler")
        if not isinstance(scaler, dict) or len(scaler.get("mean", [])) != n_features:
            raise ValueError("Scaler causal inválido en modelo logístico extendido")
        if kind == "logistic_static":
            if type(params.get("warmup_updates")) is not int or params["warmup_updates"] < 0:
                raise ValueError("warmup_updates inválido en logistic_static")
            if type(params.get("weight_updates")) is not int or params["weight_updates"] < 0:
                raise ValueError("weight_updates inválido en logistic_static")
            if not isinstance(params.get("frozen"), bool):
                raise ValueError("Bandera frozen inválida en logistic_static")
        else:
            if not _finite_number(params.get("l1")) or not 0 <= params["l1"] <= 1:
                raise ValueError("l1 inválido en logistic_adaptive")
            grad_sq = params.get("grad_sq")
            if (
                not isinstance(grad_sq, list)
                or len(grad_sq) != n_features
                or not all(_finite_number(v) and v > 0 for v in grad_sq)
            ):
                raise ValueError("Acumulador grad_sq inválido en logistic_adaptive")
    elif kind == "kalman_filter":
        dim = n_features + 1
        mean = params.get("state_mean")
        cov = params.get("state_cov")
        if (
            not isinstance(mean, list)
            or len(mean) != dim
            or not all(_finite_number(v) for v in mean)
        ):
            raise ValueError("Vector de estado Kalman inválido")
        if (
            not isinstance(cov, list)
            or len(cov) != dim
            or any(
                not isinstance(row, list)
                or len(row) != dim
                or not all(_finite_number(c) for c in row)
                for row in cov
            )
        ):
            raise ValueError("Matriz de covarianza Kalman inválida")
        cov_arr = np.array(cov, dtype=float)
        if not np.allclose(cov_arr, cov_arr.T, atol=1e-5):
            raise ValueError("Matriz de covarianza Kalman no simétrica")
        if np.any(np.diag(cov_arr) <= 0):
            raise ValueError("Diagonal de covarianza Kalman debe ser positiva")
        for key in ("process_noise", "obs_noise"):
            if not _finite_number(params.get(key)) or params[key] <= 0:
                raise ValueError(f"Ruido Kalman inválido: {key}")
    elif kind == "return_baseline":
        if not _finite_number(params.get("mean_return")):
            raise ValueError("mean_return inválido")
        if not _finite_number(params.get("var_return")) or params["var_return"] <= 0:
            raise ValueError("var_return inválido")
    elif kind == "ewma_volatility":
        if not _finite_number(params.get("ewma_lambda")) or not 0.5 <= params["ewma_lambda"] < 1.0:
            raise ValueError("ewma_lambda fuera de rango [0.5, 1.0)")
        if not _finite_number(params.get("variance")) or params["variance"] <= 0:
            raise ValueError("Varianza EWMA inválida")
    elif kind == "har_rv":
        theta = params.get("theta")
        cov = params.get("rls_cov")
        if (
            not isinstance(theta, list)
            or len(theta) != 4
            or not all(_finite_number(v) for v in theta)
        ):
            raise ValueError("Parámetros HAR-RV inválidos")
        if (
            not isinstance(cov, list)
            or len(cov) != 4
            or any(
                not isinstance(r, list) or len(r) != 4 or not all(_finite_number(c) for c in r)
                for r in cov
            )
        ):
            raise ValueError("Matriz RLS de HAR-RV inválida")
    elif kind == "volatility_baseline":
        if not _finite_number(params.get("mean_variance")) or params["mean_variance"] <= 0:
            raise ValueError("mean_variance inválido en volatility_baseline")
    elif kind == "quantile_regression":
        biases = params.get("biases")
        weights = params.get("weights")
        if not isinstance(biases, dict) or not isinstance(weights, dict):
            raise ValueError("Estructura de regresión cuantil inválida")
        for tau_key in ("0.10", "0.50", "0.90"):
            if not _finite_number(biases.get(tau_key)):
                raise ValueError(f"Sesgo cuantil inválido: {tau_key}")
            w_list = weights.get(tau_key)
            if (
                not isinstance(w_list, list)
                or len(w_list) != n_features
                or not all(_finite_number(v) for v in w_list)
            ):
                raise ValueError(f"Pesos cuantiles inválidos: {tau_key}")
    elif kind == "quantile_baseline":
        history = params.get("history")
        if not isinstance(history, list) or not all(_finite_number(v) for v in history):
            raise ValueError("Historial de quantile_baseline inválido")
    else:
        raise ValueError(f"Modelo incremental desconocido: {kind}")


def predict_adaptive_structured(
    model: dict[str, Any],
    values: list[float],
) -> dict[str, float]:
    """Compute structured multi-target forecast without mutating model state."""
    kind = model["kind"]
    params = model["params"]
    target_type = model.get("target_type", "direction")

    if kind == "beta_frequency":
        prob = params["positive"] / (params["positive"] + params["negative"])
        prob = max(PROBABILITY_EPSILON, min(1.0 - PROBABILITY_EPSILON, float(prob)))
        return {
            "target_type": target_type,
            "probability": prob,
            "raw_probability": prob,
        }
    if kind == "logistic":
        score = float(params["intercept"]) + sum(
            float(w) * float(v) for w, v in zip(params["weights"], values, strict=True)
        )
        prob = _sigmoid(score)
        return {
            "target_type": target_type,
            "probability": prob,
            "raw_probability": prob,
        }
    if kind in ("logistic_static", "logistic_adaptive"):
        scaled = _standardize_vec(params["scaler"], values)
        score = float(params["intercept"]) + sum(
            float(w) * float(v) for w, v in zip(params["weights"], scaled, strict=True)
        )
        prob = _sigmoid(score)
        return {
            "target_type": target_type,
            "probability": prob,
            "raw_probability": prob,
        }
    if kind == "kalman_filter":
        dim = model["n_features"] + 1
        scaled = _standardize_vec(params["scaler"], values)
        x_vec = np.array([1.0] + scaled, dtype=float)
        beta_prev = np.array(params["state_mean"], dtype=float)
        cov_prev = np.array(params["state_cov"], dtype=float)

        shrink = max(0.0, min(0.2, float(params.get("ridge_shrinkage", 1e-3))))
        beta_prior = (1.0 - shrink) * beta_prev
        q_val = max(1e-10, float(params.get("process_noise", 1e-5)))
        cov_prior = cov_prev + np.eye(dim, dtype=float) * q_val
        raw_ret = float(np.dot(x_vec, beta_prior))
        expected_return = max(-0.25, min(0.25, raw_ret))
        obs_var = max(
            float(params.get("obs_noise_min", 1e-6)),
            min(float(params.get("obs_noise_max", 0.10)), float(params.get("obs_noise", 1e-3))),
        )
        pred_var = max(float(params.get("obs_noise_min", 1e-6)), float(x_vec.T @ cov_prior @ x_vec) + obs_var)
        pred_std = math.sqrt(pred_var)
        snr_z = expected_return / pred_std
        structural_prob = _normal_cdf(snr_z)

        cal = params.get("calibrator", {})
        cal_a = float(cal.get("intercept", 0.0))
        cal_b = max(0.05, min(1.5, float(cal.get("slope", 0.65))))
        cal_prob = _sigmoid(cal_a + cal_b * max(-4.0, min(4.0, snr_z)))
        pos = float(cal.get("positive", 2.0))
        neg = float(cal.get("negative", 2.0))
        base_rate = pos / (pos + neg)
        blend = min(1.0, int(model.get("updates", 0)) / 30.0)
        final_prob = blend * cal_prob + (1.0 - blend) * (0.5 * structural_prob + 0.5 * base_rate)
        final_prob = max(PROBABILITY_EPSILON, min(1.0 - PROBABILITY_EPSILON, float(final_prob)))
        return {
            "target_type": target_type,
            "probability": final_prob,
            "raw_probability": float(structural_prob),
            "expected_return": float(expected_return),
            "predictive_variance": float(pred_var),
            "predictive_std": float(pred_std),
            "pred_variance": float(pred_var),
            "pred_volatility": float(pred_std),
            "snr_z": float(snr_z),
            "q10": float(expected_return - 1.2815515655446004 * pred_std),
            "q50": float(expected_return),
            "q90": float(expected_return + 1.2815515655446004 * pred_std),
        }
    if kind == "return_baseline":
        mean_r = float(params.get("mean_return", 0.0))
        var_r = max(1e-7, float(params.get("var_return", 1e-4)))
        std_r = math.sqrt(var_r)
        pos = float(params.get("positive", 2.0))
        neg = float(params.get("negative", 2.0))
        prob = max(PROBABILITY_EPSILON, min(1.0 - PROBABILITY_EPSILON, pos / (pos + neg)))
        return {
            "target_type": target_type,
            "probability": float(prob),
            "raw_probability": float(_normal_cdf(mean_r / std_r)),
            "expected_return": float(mean_r),
            "predictive_variance": float(var_r),
            "predictive_std": float(std_r),
            "pred_variance": float(var_r),
            "pred_volatility": float(std_r),
            "snr_z": float(mean_r / std_r),
            "q10": float(mean_r - 1.2815515655446004 * std_r),
            "q50": float(mean_r),
            "q90": float(mean_r + 1.2815515655446004 * std_r),
        }
    if kind == "ewma_volatility":
        var_est = float(params.get("variance", 1e-4))
        if len(values) >= 6:
            # Features in data.py: [0] return_1*50, [4] std*50, [5] range*25
            r1 = float(values[0]) / 50.0
            rng = max(0.0, float(values[5]) / 25.0)
            park_bar = (rng * rng) / (4.0 * math.log(2.0)) if rng > 0.0 else r1 * r1
            lam = float(params.get("ewma_lambda", 0.94))
            blend = float(params.get("range_blend", 0.35))
            instant_var = (1.0 - blend) * (r1 * r1) + blend * park_bar
            var_est = lam * var_est + (1.0 - lam) * instant_var
        var_est = max(float(params.get("var_min", 1e-8)), min(float(params.get("var_max", 0.25)), var_est))
        vol_est = math.sqrt(var_est)
        return {
            "target_type": target_type,
            "probability": 0.5,
            "raw_probability": 0.5,
            "pred_variance": float(var_est),
            "pred_volatility": float(vol_est),
        }
    if kind == "har_rv":
        x_har = _har_regressors_from_vec(params, values)
        theta = np.array(params["theta"], dtype=float)
        var_min = float(params.get("var_min", 1e-8))
        var_max = float(params.get("var_max", 0.25))
        log_resid_var = max(0.0, min(2.0, float(params.get("log_resid_var", 0.45))))
        # Apply Jensen log-normal convexity correction E[RV] = exp(mu_log + 0.5 * sigma_eta^2)
        pred_log_var = max(
            math.log(var_min),
            min(math.log(var_max), float(np.dot(x_har, theta)) + 0.5 * log_resid_var),
        )
        har_var = math.exp(pred_log_var)
        ewma_floor = max(var_min, float(params.get("ewma_var_floor", har_var)))
        pred_var = max(var_min, min(var_max, 0.80 * har_var + 0.20 * ewma_floor))
        return {
            "target_type": target_type,
            "probability": 0.5,
            "raw_probability": 0.5,
            "pred_variance": float(pred_var),
            "pred_volatility": float(math.sqrt(pred_var)),
        }
    if kind == "volatility_baseline":
        var_est = max(
            float(params.get("var_min", 1e-8)),
            min(float(params.get("var_max", 0.25)), float(params.get("mean_variance", 1e-4))),
        )
        return {
            "target_type": target_type,
            "probability": 0.5,
            "raw_probability": 0.5,
            "pred_variance": float(var_est),
            "pred_volatility": float(math.sqrt(var_est)),
        }
    if kind == "quantile_regression":
        scaled = _standardize_vec(params["scaler"], values)
        biases = params["biases"]
        weights = params["weights"]
        raw_q: dict[str, float] = {}
        for tau_key in ("0.10", "0.50", "0.90"):
            w_vec = weights[tau_key]
            val = float(biases[tau_key]) + sum(
                float(w) * float(v) for w, v in zip(w_vec, scaled, strict=True)
            )
            raw_q[tau_key] = max(-0.25, min(0.25, val))
        crossed = raw_q["0.10"] > raw_q["0.50"] or raw_q["0.50"] > raw_q["0.90"]
        sorted_q = sorted([raw_q["0.10"], raw_q["0.50"], raw_q["0.90"]])
        q10, q50, q90 = sorted_q[0], sorted_q[1], sorted_q[2]
        if q90 - q10 < 1e-5:
            q10 -= 5e-6
            q90 += 5e-6
        width = max(1e-5, q90 - q10)
        approx_std = width / (2.0 * 1.2815515655446004)
        prob = _normal_cdf(q50 / approx_std)
        return {
            "target_type": target_type,
            "probability": float(prob),
            "raw_probability": float(prob),
            "expected_return": float(q50),
            "q10": float(q10),
            "q50": float(q50),
            "q90": float(q90),
            "interval_width": float(width),
            "raw_crossed": 1.0 if crossed else 0.0,
        }
    if kind == "quantile_baseline":
        history = [float(v) for v in params.get("history", [])]
        spread = float(params.get("init_spread", 0.008))
        if len(history) < 10:
            q10, q50, q90 = -spread, 0.0, spread
        else:
            arr = np.array(history, dtype=float)
            q10, q50, q90 = (
                float(np.quantile(arr, 0.10)),
                float(np.quantile(arr, 0.50)),
                float(np.quantile(arr, 0.90)),
            )
            if q90 - q10 < 1e-5:
                q10 -= 5e-6
                q90 += 5e-6
        width = max(1e-5, q90 - q10)
        approx_std = width / (2.0 * 1.2815515655446004)
        prob = _normal_cdf(q50 / approx_std)
        return {
            "target_type": target_type,
            "probability": float(prob),
            "raw_probability": float(prob),
            "expected_return": float(q50),
            "q10": float(q10),
            "q50": float(q50),
            "q90": float(q90),
            "interval_width": float(width),
            "raw_crossed": 0.0,
        }
    raise ValueError(f"Modelo incremental desconocido: {kind}")


def _har_regressors_from_vec(params: dict[str, Any], values: list[float]) -> np.ndarray:
    eps = 1e-8
    r1 = float(values[0]) / 50.0 if len(values) >= 1 else 0.0
    r1_sq = r1 * r1
    rng = max(0.0, float(values[5]) / 25.0) if len(values) >= 6 else abs(r1)
    park_short = (rng * rng) / (4.0 * math.log(2.0)) if rng > 0.0 else r1_sq
    rv_short = max(eps, 0.5 * r1_sq + 0.5 * park_short)

    vol_med = max(0.0, float(values[4]) / 50.0) if len(values) >= 5 else math.sqrt(rv_short)
    rv_med = max(eps, vol_med * vol_med)

    history = [float(v) for v in params.get("rv_history", []) if v > 0.0]
    if len(history) >= 6:
        rv_long = max(eps, sum(history[-24:]) / len(history[-24:]))
    else:
        rv_long = rv_med

    if len(values) >= 11:
        # Feature 10 in v2_financial11 is fin_vol_parkinson_ratio
        ratio = 1.0 + max(-0.8, min(3.0, float(values[10]) / 5.0))
        rv_long = max(eps, rv_med / max(0.25, ratio * ratio))

    return np.array(
        [1.0, math.log(rv_short), math.log(rv_med), math.log(rv_long)],
        dtype=float,
    )


def learn_adaptive(
    model: dict[str, Any],
    values: list[float],
    label: int,
    *,
    realized_return: float | None = None,
    realized_variance: float | None = None,
    target_id: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Compute updated params for adaptive model. Returns (new_params, updated_flag)."""
    kind = model["kind"]
    params = copy.deepcopy(model["params"])
    if not _check_target_id(params, target_id):
        return params, False

    ret = (
        float(realized_return)
        if realized_return is not None and math.isfinite(realized_return)
        else (0.005 if label == 1 else -0.005)
    )
    ret = max(-0.25, min(0.25, ret))
    rv = (
        float(realized_variance)
        if realized_variance is not None and math.isfinite(realized_variance) and realized_variance >= 0.0
        else max(1e-8, ret * ret)
    )

    if kind == "logistic_static":
        warmup_limit = int(params.get("warmup_updates", 120))
        w_updates = int(params.get("weight_updates", 0))
        if params.get("frozen", False) or w_updates >= warmup_limit:
            params["frozen"] = True
            return params, True
        scaled = _standardize_vec(params["scaler"], values)
        score = float(params["intercept"]) + sum(
            float(w) * float(v) for w, v in zip(params["weights"], scaled, strict=True)
        )
        prob = _sigmoid(score)
        err = prob - float(label)
        lr = float(params["learning_rate"])
        l2 = float(params["l2"])
        params["intercept"] = max(-2.5, min(2.5, float(params["intercept"]) - lr * err))
        new_w = [
            float(w) - lr * (err * float(x) + l2 * float(w))
            for w, x in zip(params["weights"], scaled, strict=True)
        ]
        params["weights"] = _project_vec_l2(new_w, float(params.get("max_weight_norm", 2.5)))
        params["scaler"] = _update_vec_scaler(params["scaler"], values)
        params["weight_updates"] = w_updates + 1
        if params["weight_updates"] >= warmup_limit:
            params["frozen"] = True
        return params, True

    if kind == "logistic_adaptive":
        scaled = _standardize_vec(params["scaler"], values)
        weights = list(params["weights"])
        grad_sq = list(params["grad_sq"])
        score = float(params["intercept"]) + sum(
            float(w) * float(v) for w, v in zip(weights, scaled, strict=True)
        )
        prob = _sigmoid(score)
        err = prob - float(label)

        dm = dict(params.get("drift_monitor", {}))
        brier = err * err
        ll = -(label * math.log(prob) + (1 - label) * math.log1p(-prob))
        fast_ema = 0.92 * float(dm.get("brier_fast_ema", 0.25)) + 0.08 * brier
        slow_ema = 0.99 * float(dm.get("brier_slow_ema", 0.25)) + 0.01 * brier
        dm["brier_fast_ema"] = float(fast_ema)
        dm["brier_slow_ema"] = float(slow_ema)
        dm["log_loss_ema"] = float(0.97 * float(dm.get("log_loss_ema", math.log(2.0))) + 0.03 * ll)
        next_updates = int(model["updates"]) + 1
        degraded = next_updates >= 40 and (fast_ema - slow_ema) > 0.012
        if degraded and not dm.get("degraded", False):
            dm["drift_warnings"] = int(dm.get("drift_warnings", 0)) + 1
        dm["degraded"] = bool(degraded)
        params["drift_monitor"] = dm

        base_lr = float(params["learning_rate"])
        decay = float(params.get("decay_rate", 0.01))
        eff_lr = base_lr / (1.0 + decay * math.sqrt(next_updates))
        l1 = float(params["l1"])
        l2 = float(params["l2"]) * (2.5 if degraded else 1.0)
        rms_beta = float(params.get("rms_beta", 0.95))

        igsq = rms_beta * float(params.get("intercept_grad_sq", 1.0)) + (1.0 - rms_beta) * (err * err)
        params["intercept_grad_sq"] = float(igsq)
        istep = eff_lr / (math.sqrt(max(1e-8, igsq)) + 1.0)
        params["intercept"] = max(
            -2.5,
            min(2.5, float(params["intercept"]) - istep * (err + 0.5 * l2 * float(params["intercept"]))),
        )

        new_w: list[float] = []
        for idx, (w, x) in enumerate(zip(weights, scaled, strict=True)):
            g = err * float(x) + l2 * float(w)
            gsq = rms_beta * float(grad_sq[idx]) + (1.0 - rms_beta) * (g * g)
            grad_sq[idx] = float(gsq)
            step = eff_lr / (math.sqrt(max(1e-8, gsq)) + 1.0)
            w_half = float(w) - step * g
            new_w.append(float(_soft_threshold(w_half, l1 * step)))

        params["weights"] = _project_vec_l2(new_w, float(params.get("max_weight_norm", 2.5)))
        params["grad_sq"] = grad_sq
        params["scaler"] = _update_vec_scaler(params["scaler"], values)
        return params, True

    if kind == "kalman_filter":
        dim = model["n_features"] + 1
        scaled = _standardize_vec(params["scaler"], values)
        x_vec = np.array([1.0] + scaled, dtype=float)
        beta_prev = np.array(params["state_mean"], dtype=float)
        cov_prev = np.array(params["state_cov"], dtype=float)

        shrink = max(0.0, min(0.2, float(params.get("ridge_shrinkage", 1e-3))))
        beta_prior = (1.0 - shrink) * beta_prev
        q_val = max(1e-10, float(params.get("process_noise", 1e-5)))
        cov_min = float(params.get("cov_min", 1e-7))
        cov_max = float(params.get("cov_max", 2.0))
        cov_prior = _enforce_spd_bounds(
            cov_prev + np.eye(dim, dtype=float) * q_val,
            min_eig=cov_min,
            max_eig=cov_max,
        )

        obs_min = float(params.get("obs_noise_min", 1e-6))
        obs_max = float(params.get("obs_noise_max", 0.10))
        r_val = max(obs_min, min(obs_max, float(params.get("obs_noise", 1e-3))))

        pred_ret = float(np.dot(x_vec, beta_prior))
        innovation = ret - pred_ret
        innovation_var = max(obs_min, float(x_vec.T @ cov_prior @ x_vec) + r_val)
        pred_std = math.sqrt(innovation_var)

        # Prequential calibration update prior to state mutation
        snr_z = max(-4.0, min(4.0, max(-0.25, min(0.25, pred_ret)) / pred_std))
        cal = dict(params.get("calibrator", {}))
        cal_a = float(cal.get("intercept", 0.0))
        cal_b = max(0.05, min(1.5, float(cal.get("slope", 0.65))))
        cal_lr = float(cal.get("lr", 0.03))
        cal_l2 = float(cal.get("l2", 0.01))
        prob_pred = _sigmoid(cal_a + cal_b * snr_z)
        err = prob_pred - float(label)
        cal["intercept"] = max(-1.5, min(1.5, cal_a - cal_lr * (err + cal_l2 * cal_a)))
        cal["slope"] = max(0.05, min(1.5, cal_b - cal_lr * (err * snr_z + cal_l2 * (cal_b - 0.65))))
        if label == 1:
            cal["positive"] = float(cal.get("positive", 2.0)) + 1.0
        else:
            cal["negative"] = float(cal.get("negative", 2.0)) + 1.0
        cal["brier_ema"] = 0.97 * float(cal.get("brier_ema", 0.25)) + 0.03 * (err * err)
        params["calibrator"] = cal

        # Kalman gain & Joseph covariance update
        kalman_gain = (cov_prior @ x_vec) / innovation_var
        beta_post = beta_prior + kalman_gain * innovation

        max_norm = float(params.get("max_weight_norm", 0.20))
        if max_norm > 0.0:
            slope_norm = float(np.linalg.norm(beta_post[1:]))
            if slope_norm > max_norm:
                beta_post[1:] = beta_post[1:] * (max_norm / slope_norm)
            beta_post[0] = max(-0.05, min(0.05, float(beta_post[0])))

        ident = np.eye(dim, dtype=float)
        i_kx = ident - np.outer(kalman_gain, x_vec)
        cov_joseph = i_kx @ cov_prior @ i_kx.T + r_val * np.outer(kalman_gain, kalman_gain)
        cov_post = _enforce_spd_bounds(cov_joseph, min_eig=cov_min, max_eig=cov_max)

        if params.get("adaptive_obs_noise", True):
            lam_r = max(0.80, min(0.995, float(params.get("obs_ewma_lambda", 0.96))))
            post_resid = ret - float(np.dot(x_vec, beta_post))
            r_est = 0.5 * (innovation * innovation) + 0.5 * (
                post_resid * post_resid + float(x_vec.T @ cov_post @ x_vec)
            )
            params["obs_noise"] = float(max(obs_min, min(obs_max, lam_r * r_val + (1.0 - lam_r) * r_est)))

        params["state_mean"] = [float(v) for v in beta_post.tolist()]
        params["state_cov"] = [[float(c) for c in row] for row in cov_post.tolist()]
        params["scaler"] = _update_vec_scaler(params["scaler"], values)
        diag = np.diag(cov_post)
        params["diagnostics"] = {
            "last_innovation": float(innovation),
            "last_predictive_std": float(pred_std),
            "last_kalman_gain_norm": float(np.linalg.norm(kalman_gain)),
            "trace_cov": float(np.trace(cov_post)),
            "min_diag_cov": float(np.min(diag)),
            "max_diag_cov": float(np.max(diag)),
        }
        return params, True

    if kind == "return_baseline":
        lam = float(params.get("ewma_lambda", 0.97))
        old_mean = float(params.get("mean_return", 0.0))
        diff = ret - old_mean
        params["mean_return"] = float(lam * old_mean + (1.0 - lam) * ret)
        params["var_return"] = float(
            max(1e-7, lam * float(params.get("var_return", 1e-4)) + (1.0 - lam) * (diff * diff))
        )
        if label == 1:
            params["positive"] = float(params.get("positive", 2.0)) + 1.0
        else:
            params["negative"] = float(params.get("negative", 2.0)) + 1.0
        return params, True

    if kind == "ewma_volatility":
        lam = float(params.get("ewma_lambda", 0.94))
        var_min = float(params.get("var_min", 1e-8))
        var_max = float(params.get("var_max", 0.25))
        clipped_rv = max(var_min, min(var_max, rv))
        old_var = float(params.get("variance", 1e-4))
        params["variance"] = float(max(var_min, min(var_max, lam * old_var + (1.0 - lam) * clipped_rv)))
        return params, True

    if kind == "har_rv":
        var_min = float(params.get("var_min", 1e-8))
        var_max = float(params.get("var_max", 0.25))
        clipped_rv = max(var_min, min(var_max, rv))
        y_log = math.log(clipped_rv)

        x_har = _har_regressors_from_vec(params, values)
        theta = np.array(params["theta"], dtype=float)
        cov = np.array(params["rls_cov"], dtype=float)
        ff = max(0.95, min(1.0, float(params.get("forgetting_factor", 0.992))))

        err = y_log - float(np.dot(x_har, theta))
        denom = ff + float(x_har.T @ cov @ x_har)
        gain = (cov @ x_har) / max(1e-8, denom)
        theta_new = theta + gain * err

        old_lrv = float(params.get("log_resid_var", 0.45))
        params["log_resid_var"] = float(max(0.05, min(1.5, 0.96 * old_lrv + 0.04 * (err * err))))
        old_floor = float(params.get("ewma_var_floor", 1e-4))
        params["ewma_var_floor"] = float(0.94 * old_floor + 0.06 * clipped_rv)

        ident = np.eye(4, dtype=float)
        i_gx = ident - np.outer(gain, x_har)
        cov_new = (i_gx @ cov @ i_gx.T) / ff + 1e-6 * ident
        cov_new = _enforce_spd_bounds(cov_new, min_eig=1e-6, max_eig=2.0)

        theta_new[0] = max(-15.0, min(0.0, float(theta_new[0])))
        theta_new[1:] = np.clip(theta_new[1:], -0.2, 0.95)
        wsum = float(np.sum(theta_new[1:]))
        if wsum > 0.99:
            theta_new[1:] = theta_new[1:] * (0.99 / wsum)

        history = list(params.get("rv_history", []))
        history.append(float(clipped_rv))
        if len(history) > 64:
            history = history[-64:]

        params["theta"] = [float(v) for v in theta_new.tolist()]
        params["rls_cov"] = [[float(c) for c in row] for row in cov_new.tolist()]
        params["rv_history"] = history
        return params, True

    if kind == "volatility_baseline":
        var_min = float(params.get("var_min", 1e-8))
        var_max = float(params.get("var_max", 0.25))
        clipped_rv = max(var_min, min(var_max, rv))
        n = int(model.get("updates", 0)) + 1
        alpha = max(1.0 / n, 0.01)
        old_mean = float(params.get("mean_variance", 1e-4))
        params["mean_variance"] = float((1.0 - alpha) * old_mean + alpha * clipped_rv)
        return params, True

    if kind == "quantile_regression":
        scaled = _standardize_vec(params["scaler"], values)
        biases = dict(params["biases"])
        weights = copy.deepcopy(params["weights"])
        grad_sq = copy.deepcopy(params["grad_sq"])
        lr = float(params.get("learning_rate", 0.0005))
        l2 = float(params.get("l2", 0.15))
        next_updates = int(model["updates"]) + 1
        eff_lr = lr / (1.0 + 0.01 * math.sqrt(next_updates))

        ret_history = list(params.get("ret_history", []))
        ret_history.append(float(ret))
        if len(ret_history) > 250:
            ret_history = ret_history[-250:]
        params["ret_history"] = ret_history
        emp_anchor: dict[str, float] | None = None
        if len(ret_history) >= 15:
            arr_h = np.array(ret_history, dtype=float)
            emp_anchor = {
                "0.10": float(np.quantile(arr_h, 0.10)),
                "0.50": float(np.quantile(arr_h, 0.50)),
                "0.90": float(np.quantile(arr_h, 0.90)),
            }

        raw_preds: dict[str, float] = {}
        for tau_key, tau_val in (("0.10", 0.10), ("0.50", 0.50), ("0.90", 0.90)):
            w_vec = weights[tau_key]
            gsq_vec = grad_sq[tau_key]
            pred = float(biases[tau_key]) + sum(
                float(w) * float(v) for w, v in zip(w_vec, scaled, strict=True)
            )
            raw_preds[tau_key] = pred
            diff = ret - pred
            subgrad = -tau_val if diff > 0.0 else (1.0 - tau_val) if diff < 0.0 else 0.0
            new_bias = float(biases[tau_key]) - eff_lr * subgrad
            if emp_anchor is not None:
                new_bias = 0.85 * new_bias + 0.15 * emp_anchor[tau_key]
            biases[tau_key] = max(-0.20, min(0.20, new_bias))
            for idx, (w, x) in enumerate(zip(w_vec, scaled, strict=True)):
                g = subgrad * float(x) + l2 * float(w)
                gsq = 0.95 * float(gsq_vec[idx]) + 0.05 * (g * g)
                gsq_vec[idx] = float(gsq)
                step = eff_lr / (math.sqrt(max(1e-8, gsq)) + 1.0)
                w_vec[idx] = float(max(-0.003, min(0.003, float(w) - step * g)))

        if raw_preds["0.10"] > raw_preds["0.50"] or raw_preds["0.50"] > raw_preds["0.90"]:
            params["raw_crossings"] = int(params.get("raw_crossings", 0)) + 1

        params["biases"] = biases
        params["weights"] = weights
        params["grad_sq"] = grad_sq
        params["scaler"] = _update_vec_scaler(params["scaler"], values)
        return params, True

    if kind == "quantile_baseline":
        history = list(params.get("history", []))
        history.append(float(ret))
        wsize = max(20, int(params.get("window_size", 250)))
        if len(history) > wsize:
            history = history[-wsize:]
        params["history"] = history
        return params, True

    raise ValueError(f"Modelo incremental desconocido: {kind}")


class AdaptiveOnlineModel:
    """Standardized object-oriented interface wrapping serializable online model states."""

    def __init__(self, model_id: str, state: dict[str, Any]) -> None:
        from trading_ml.online.models import validate_model

        validate_model(state)
        self.model_id = model_id
        self.state = copy.deepcopy(state)

    @classmethod
    def create(
        cls,
        model_id: str,
        n_features: int,
        seed: int = 42,
        *,
        options: dict[str, Any] | None = None,
    ) -> AdaptiveOnlineModel:
        state = new_adaptive_model(model_id, n_features, seed=seed, options=options)
        return cls(model_id, state)

    def validate_state(self) -> None:
        from trading_ml.online.models import validate_model

        validate_model(self.state)

    def predict(self, features: list[float]) -> float:
        from trading_ml.online.models import predict

        return predict(self.state, features)

    def predict_structured(self, features: list[float]) -> dict[str, float]:
        from trading_ml.online.models import predict_structured

        return predict_structured(self.state, features)

    def update(
        self,
        features: list[float],
        label: int,
        *,
        realized_return: float | None = None,
        realized_variance: float | None = None,
        target_id: str | None = None,
    ) -> None:
        from trading_ml.online.models import learn

        learn(
            self.state,
            features,
            label,
            realized_return=realized_return,
            realized_variance=realized_variance,
            target_id=target_id,
        )

    def save(self) -> dict[str, Any]:
        self.validate_state()
        return copy.deepcopy(self.state)

    @classmethod
    def load(cls, model_id: str, payload: dict[str, Any]) -> AdaptiveOnlineModel:
        return cls(model_id, payload)

    def metadata(self) -> dict[str, Any]:
        spec = MODEL_SPECS.get(self.model_id)
        return {
            "model_id": self.model_id,
            "kind": self.state["kind"],
            "target_type": self.state.get(
                "target_type", spec.target_type if spec else "direction"
            ),
            "family": spec.family if spec else self.state["kind"],
            "is_baseline": spec.is_baseline if spec else (self.state["kind"] == "beta_frequency"),
            "baseline_id": spec.baseline_id if spec else "baseline",
            "complexity": spec.complexity if spec else "O(d)",
            "description": spec.description if spec else "",
            "status": self.state["status"],
            "updates": self.state["updates"],
            "n_features": self.state["n_features"],
        }

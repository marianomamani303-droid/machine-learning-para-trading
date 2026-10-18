"""Read-only visibility of frozen research decisions, separate from live models."""

from pathlib import Path

from trading_ml.common import read_json, sha256
from trading_ml.errors import DataError


def research_candidates(root: Path) -> list[dict]:
    rows = []
    for manifest in sorted((root / "edge_research" / "runs").glob("*/shadow_manifest.json")):
        receipts = read_json(manifest.parent / "artifacts.json")
        for name in ["shadow_manifest.json", "plan.json", "summary.json"]:
            if sha256(manifest.parent / name) != receipts[name]:
                raise DataError("Research candidate evidence checksum mismatch")
        plan = read_json(manifest.parent / "plan.json")
        for candidate in read_json(manifest)["candidates"]:
            symbol, interval, horizon = candidate["task"].split("_")
            passed = candidate["eligible_for_shadow"]
            rows.append(
                {
                    "Estudio": manifest.parent.name,
                    "Activo": symbol,
                    "Temporalidad": interval,
                    "Horizonte (velas)": int(horizon[1:]),
                    "Modelo": candidate["recipe"],
                    "Señal supera controles": candidate["predictive_pass"],
                    "Economía supera controles": candidate["economic_pass"],
                    "Estado": "Elegible para protocolo prospectivo"
                    if passed
                    else "No supera criterios de seguimiento",
                    "Rentabilidad": "Sin confirmar",
                    "Costo aproximado ida y vuelta (bps)": 2
                    * sum(
                        plan["protocol"][k]
                        for k in ["commission_bps", "slippage_bps", "half_spread_bps"]
                    ),
                }
            )
    return rows

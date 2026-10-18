"""Read-only baseline of every pre-existing artifact; never reads reserved metrics."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trading_ml.common import sha256, write_new_json

root = Path("artifacts")
destination = root / "edge_research" / "preservation_before.json"
paths = [p for p in sorted(root.rglob("*")) if p.is_file()]
print(
    f"Hashing {len(paths)} existing artifacts ({sum(p.stat().st_size for p in paths) / 1e9:.2f} GB)",
    flush=True,
)
with ThreadPoolExecutor(max_workers=12) as pool:
    digests = list(pool.map(sha256, paths))
manifest = {str(p.resolve()): digest for p, digest in zip(paths, digests, strict=True)}
write_new_json(destination, {"files": manifest, "count": len(manifest)})
print(f"Preserved baseline: {len(manifest)} files")

from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Dict
import yaml

def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def rocktype_code(cfg: Dict[str, Any], mettype_key: str) -> str:
    mt = cfg.get("mettypes", {})
    if mettype_key not in mt:
        raise KeyError(f"Unknown mettype '{mettype_key}'. Available: {list(mt.keys())}")
    return mt[mettype_key]["rocktype_code"]

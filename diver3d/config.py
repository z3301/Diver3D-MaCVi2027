from __future__ import annotations

import copy

import yaml


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(open(path))
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        try:
            val = yaml.safe_load(val)        # YAML scalars: true/false/null, ints, floats, lists
        except yaml.YAMLError:
            pass
        if isinstance(val, str):             # YAML 1.1 leaves "5e-4" as a string
            try:
                val = float(val)
            except ValueError:
                pass
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg


def dump_config(cfg: dict) -> str:
    return yaml.safe_dump(copy.deepcopy(cfg), sort_keys=False)

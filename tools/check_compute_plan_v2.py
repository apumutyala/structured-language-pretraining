"""Offline cost reconciliation. This cannot launch a job or authorize spending."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def number(value, name, minimum=Decimal(0)):
    if type(value) not in (int, float):
        raise ValueError(f"{name}: finite numeric value required")
    out = Decimal(str(value))
    if not out.is_finite() or out < minimum:
        raise ValueError(f"{name}: invalid numeric value")
    return out


def count(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: positive integer required")
    return value


def reconcile(config):
    if config.get("launch_authorized") is not False:
        raise ValueError("This planning document cannot authorize launches")
    total = number(config["total_usd"], "total")
    price = number(config["usd_per_gpu_hour"], "price", Decimal("0.000001"))
    support = number(config["supporting_gpu_fraction"], "support")
    allocations = {k: number(v, k) for k, v in config["allocations_usd"].items()}
    if sum(allocations.values()) != total:
        raise ValueError("Allocation total does not match ceiling")
    fixed = allocations["qualification_and_equal_tuning"] + allocations["additional_services"]
    stages, names = [], set()
    for spec in config["stages"]:
        if spec["name"] in names or type(spec["optional"]) is not bool:
            raise ValueError("Duplicate stage name or non-Boolean optional flag")
        names.add(spec["name"])
        runs = count(spec["interfaces"], "interfaces") * count(spec["conditions"], "conditions") * count(spec["seeds"], "seeds")
        flops = 6 * count(spec["parameters"], "parameters") * count(spec["tokens_per_run"], "tokens") * runs
        stages.append({"name": spec["name"], "runs": runs, "nominal_6ND": flops, "optional": spec["optional"]})
    scenarios = []
    for rate in config["effective_nominal_tflops_scenarios"]:
        tflops = number(rate, "rate", Decimal("0.000001"))
        detailed = []
        for stage in stages:
            hours = Decimal(stage["nominal_6ND"]) / (tflops * 10**12 * 3600)
            detailed.append({**stage, "core_gpu_hours": float(hours),
                             "core_usd": float(hours * price),
                             "training_and_support_usd": float(hours * price * (1 + support))})
        scopes = []
        for scope, ceiling, reserve, include_optional in (
            ("full_conditional_42", total, allocations["reserve"], True),
            ("base_conditional_30", Decimal(18000), number(config["base_18000"]["reserve_usd"], "reserve"), False),
        ):
            selected = [s for s in stages if include_optional or not s["optional"]]
            flops = sum(s["nominal_6ND"] for s in selected)
            gpu = Decimal(flops) / (tflops * 10**12 * 3600) * price * (1 + support)
            forecast = gpu + fixed + reserve
            scopes.append({"scope": scope, "runs": sum(s["runs"] for s in selected),
                           "gpu_training_and_support_usd": float(gpu),
                           "fixed_services_and_qualification_usd": float(fixed),
                           "reserve_usd": float(reserve), "total_forecast_usd": float(forecast),
                           "ceiling_usd": float(ceiling), "headroom_usd": float(ceiling - forecast),
                           "fits_arithmetic_only": forecast <= ceiling})
        scenarios.append({"assumed_effective_nominal_tflops": float(tflops), "stages": detailed, "scopes": scopes})
    return {"status": "unmeasured_scenarios_not_spend_admission", "launch_authorized": False,
            "scenarios": scenarios, "limits": [
                "No H100 throughput, memory, tokenizer or full-path implementation is qualified by this calculation.",
                "The approximately 1B stage remains conditional even when arithmetic fits.",
                "GPU-hours sum across all devices; concurrent execution changes wall time, not this bill.",
                "Rate and price are assumptions, not hardware peak performance or a provider quote.",
                "Measured integrated semantic-path costs, equal tuning and every failure must inform the paid manifest.",
                "Preserve controls and replication when reducing optional scale; do not silently change a frozen study."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/structured_pretraining_compute_v2.json"))
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    raw = args.config.read_bytes()
    result = reconcile(json.loads(raw))
    result["config_sha256"] = hashlib.sha256(raw).hexdigest()
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

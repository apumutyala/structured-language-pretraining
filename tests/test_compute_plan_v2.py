import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cost_v2", ROOT / "tools/check_compute_plan_v2.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def config():
    return json.loads((ROOT / "configs/structured_pretraining_compute_v2.json").read_text())


def test_exact_grid_and_current_detailed_budget():
    result = module.reconcile(config())
    slow, mid, fast = result["scenarios"]
    full, base = mid["scopes"]
    assert [s["runs"] for s in mid["stages"]] == [12, 12, 6, 6, 6]
    assert full["runs"] == 42 and base["runs"] == 30
    assert full["gpu_training_and_support_usd"] == 17821.125
    assert full["total_forecast_usd"] == 19971.125
    assert base["total_forecast_usd"] == 17980.75
    assert full["fits_arithmetic_only"] and base["fits_arithmetic_only"]
    assert not slow["scopes"][0]["fits_arithmetic_only"]
    assert not slow["scopes"][1]["fits_arithmetic_only"]
    assert fast["scopes"][0]["fits_arithmetic_only"]


def test_supporting_work_is_counted_once():
    cfg = config()
    cfg["supporting_gpu_fraction"] = 0
    bare = module.reconcile(cfg)["scenarios"][1]["scopes"][0]
    supported = module.reconcile(config())["scenarios"][1]["scopes"][0]
    assert supported["gpu_training_and_support_usd"] == pytest.approx(1.4 * bare["gpu_training_and_support_usd"])
    assert supported["fixed_services_and_qualification_usd"] == bare["fixed_services_and_qualification_usd"] == 1100


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(launch_authorized=True),
    lambda c: c.update(total_usd=19999),
    lambda c: c.update(usd_per_gpu_hour=float("nan")),
    lambda c: c.update(effective_nominal_tflops_scenarios=[0]),
    lambda c: c.update(supporting_gpu_fraction=-0.1),
    lambda c: c["stages"][0].update(seeds=True),
    lambda c: c["stages"].append(copy.deepcopy(c["stages"][0])),
])
def test_invalid_planning_inputs_fail(mutate):
    cfg = config()
    mutate(cfg)
    with pytest.raises(ValueError):
        module.reconcile(cfg)

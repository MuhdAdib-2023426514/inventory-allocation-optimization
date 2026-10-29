"""Generate an isolated integration bundle; tests do not rely on ignored local outputs."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="session")
def demo_release_config(tmp_path_factory):
    from inventory_di.cli import cmd_backtest, cmd_prepare, cmd_report, cmd_train

    root = tmp_path_factory.mktemp("demo_release")
    payload = yaml.safe_load(Path("configs/demo.yaml").read_text())
    payload["run_id"] = "test_demo_v2"
    for field in ["processed_dir", "artifacts_dir", "reports_dir"]:
        payload["data"][field] = str(root / field)
    payload["model"].update(epochs=2, lambda_dfl_grid=[0.0, 1.0])
    payload["dro"].update(
        num_scenarios=4,
        alpha_grid=[0.0, 0.9],
        risk_weight_grid=[0.0, 0.5],
        supply_stress_multipliers=[0.8, 1.0],
    )
    payload["evaluation"].update(seeds=[42], bootstrap_resamples=20)
    config = root / "demo.yaml"
    config.write_text(yaml.safe_dump(payload))
    for action in [cmd_prepare, cmd_train, cmd_backtest, cmd_report]:
        assert action(str(config)) == 0
    return str(config)

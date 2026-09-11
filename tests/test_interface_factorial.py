import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "paper" / "run_interface_factorial.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("interface_factorial", LAUNCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(tmp_path):
    return argparse.Namespace(
        data_root=(tmp_path / "data").resolve(),
        pretrained_encoder=(tmp_path / "dinov3.pth").resolve(),
        output_root=(tmp_path / "outputs").resolve(),
        num_workers=8,
        devices=["0", "1", "2", "3"],
    )


def _values_after(command, flag, count):
    start = command.index(flag) + 1
    return command[start : start + count]


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_flatten(child, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def _diff(left, right):
    left = _flatten(left)
    right = _flatten(right)
    return {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}


def test_exact_registered_matrix_and_p34_projection(tmp_path):
    launcher = _load_launcher()
    recipe = launcher.load_recipe()
    resolved = {
        run_id: launcher.resolved_run(run_id, _args(tmp_path), recipe)
        for run_id in launcher.RUN_IDS
    }

    for run_id in ("M0", "M1"):
        command, config = resolved[run_id]
        assert _values_after(command, "--projector-scale", 3) == ["P3", "P4", "P5"]
        assert config["model"]["dec_level_n_points"] == [2, 3, 1]
        assert config["model"]["msp_c2f_blocks"] == [3, 3, 3]
        assert config["model"]["projector_distill_level_weights"] == [0.5, 1.0, 0.5]
    for run_id in ("M2", "M3"):
        command, config = resolved[run_id]
        assert _values_after(command, "--projector-scale", 2) == ["P3", "P4"]
        assert config["model"]["dec_level_n_points"] == [2, 3]
        assert config["model"]["msp_c2f_blocks"] == [3, 3]
        assert config["model"]["projector_distill_level_weights"] == [0.5, 1.0]
    assert "--projector-p4-depth-prior" not in resolved["M0"][0]
    assert "--projector-p4-depth-prior" not in resolved["M2"][0]
    assert _values_after(resolved["M1"][0], "--projector-p4-depth-prior", 4) == [
        "0.2485995", "0.4129705", "0.7943345", "2.5440953"
    ]
    assert _values_after(resolved["M3"][0], "--projector-p4-depth-prior", 4) == [
        "0.2485995", "0.4129705", "0.7943345", "2.5440953"
    ]


def test_pairwise_diffs_are_only_the_two_factors(tmp_path):
    launcher = _load_launcher()
    recipe = launcher.load_recipe()
    configs = {
        run_id: launcher.resolved_run(run_id, _args(tmp_path), recipe)[1]
        for run_id in launcher.RUN_IDS
    }
    prior_diff = {"model.projector_p4_depth_prior", "experiment_factors.p4_depth_prior"}
    interface_diff = {
        "model.scale_interface",
        "model.projector_scale",
        "model.dec_level_n_points",
        "model.projector_distill_level_weights",
        "model.msp_c2f_blocks",
        "experiment_factors.detector_interface",
    }

    assert _diff(configs["M0"], configs["M1"]) == prior_diff
    assert _diff(configs["M2"], configs["M3"]) == prior_diff
    assert _diff(configs["M0"], configs["M2"]) == interface_diff
    assert _diff(configs["M1"], configs["M3"]) == interface_diff
    assert _diff(configs["M0"], configs["M3"]) == prior_diff | interface_diff
    assert _diff(configs["M1"], configs["M2"]) == prior_diff | interface_diff


def test_default_two_gpu_assignment_balances_both_factors(tmp_path, monkeypatch):
    launcher = _load_launcher()
    args = _args(tmp_path)
    args.devices = ["0", "1"]
    monkeypatch.setattr(launcher, "input_provenance", lambda unused: {})
    monkeypatch.setattr(launcher, "gpu_snapshot", lambda: None)
    audit = launcher.build_audit(args, launcher.load_recipe())

    assert [audit["runs"][run_id]["device"] for run_id in launcher.RUN_IDS] == [
        "0", "1", "1", "0"
    ]

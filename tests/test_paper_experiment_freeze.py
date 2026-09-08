import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "paper" / "ICLR27_RECIPE.json"
LAUNCHER_PATH = ROOT / "paper" / "run_iclr27_experiment.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("paper_launcher", LAUNCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(
    tmp_path,
    track="analysis",
    system="sdsr_v40",
    scale_interface="p345",
    use_ema=True,
    seed=43,
    detector_init_seed=1043,
):
    return argparse.Namespace(
        track=track,
        system=system,
        scale_interface=scale_interface,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "output",
        seed=seed,
        detector_init_seed=detector_init_seed,
        num_workers=2,
        device="cuda",
        use_ema=use_ema,
    )


def _value_after(command, flag):
    return command[command.index(flag) + 1]


def _values_after(command, flag, count):
    start = command.index(flag) + 1
    return command[start : start + count]


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            result.update(_flatten(child, child_prefix))
        return result
    return {prefix: value}


def _resolved_diff(left, right):
    left_flat = _flatten(left)
    right_flat = _flatten(right)
    return {
        key
        for key in left_flat.keys() | right_flat.keys()
        if left_flat.get(key) != right_flat.get(key)
    }


def test_machine_readable_recipe_contains_paper_invariants():
    recipe = json.loads(RECIPE_PATH.read_text())

    assert recipe["model"]["sdsr_projector_type"] == "sdsr_v40_p4_learnable"
    assert recipe["model"]["projector_scale"] == ["P3", "P4", "P5"]
    assert recipe["model"]["dec_level_n_points"] == [2, 3, 1]
    assert recipe["model"]["backbone_register_border_tokens"] == 0
    assert recipe["cdn"]["dn_total_query_budget"] == 300
    assert recipe["dense_o2o"]["mixup_prob"] == 0.0
    assert recipe["optimization"]["multi_scale_stop_epoch"] == -1
    assert recipe["optimization"]["lr"] == 2.5e-4
    assert recipe["optimization"]["lr_encoder"] == 1.25e-4


def test_analysis_command_is_frozen_and_ema_can_be_disabled(tmp_path):
    launcher = _load_launcher()
    recipe = launcher.load_recipe()
    command, _ = launcher.build_command(
        _args(tmp_path, use_ema=False), recipe
    )

    assert _value_after(command, "--subset") == "medium"
    assert _value_after(command, "--epochs") == "24"
    assert _value_after(command, "--lr-drop") == "20"
    assert _value_after(command, "--multi-scale-stop-epoch") == "-1"
    assert _value_after(command, "--backbone-register-border-tokens") == "0"
    assert _value_after(command, "--dense-o2o-mixup-prob") == "0.0"
    assert command[command.index("--dec-level-n-points") + 1 : command.index("--dec-level-n-points") + 4] == ["2", "3", "1"]
    assert "--use-ema" not in command


def test_default_p345_h_configuration_is_unchanged(tmp_path):
    launcher = _load_launcher()
    command, resolved = launcher.build_command(_args(tmp_path), launcher.load_recipe())

    assert _values_after(command, "--projector-scale", 3) == ["P3", "P4", "P5"]
    assert _values_after(command, "--dec-level-n-points", 3) == ["2", "3", "1"]
    assert _values_after(command, "--projector-distill-level-weights", 3) == [
        "0.5",
        "1.0",
        "0.5",
    ]
    assert resolved["model"]["scale_interface"] == "p345"
    assert resolved["model"]["projector_scale"] == ["P3", "P4", "P5"]
    assert resolved["model"]["dec_level_n_points"] == [2, 3, 1]
    assert resolved["model"]["projector_distill_level_weights"] == [0.5, 1.0, 0.5]


def test_p4_is_the_frozen_p4_component(tmp_path):
    launcher = _load_launcher()
    command, resolved = launcher.build_command(
        _args(tmp_path, scale_interface="p4"), launcher.load_recipe()
    )

    assert _values_after(command, "--projector-scale", 1) == ["P4"]
    assert _values_after(command, "--dec-level-n-points", 1) == ["3"]
    assert _values_after(command, "--projector-distill-level-weights", 1) == ["1.0"]
    assert resolved["model"]["projector_scale"] == ["P4"]
    assert resolved["model"]["dec_level_n_points"] == [3]
    assert resolved["model"]["projector_distill_level_weights"] == [1.0]


def test_msp_p4_has_exactly_one_frozen_c2f_depth(tmp_path):
    launcher = _load_launcher()
    command, resolved = launcher.build_command(
        _args(tmp_path, system="msp", scale_interface="p4"),
        launcher.load_recipe(),
    )

    assert _values_after(command, "--projector-c2f-blocks", 1) == ["3"]
    assert resolved["model"]["msp_c2f_blocks"] == [3]


def test_p4_projectors_differ_only_on_projector_axis(tmp_path):
    launcher = _load_launcher()
    recipe = launcher.load_recipe()
    _, sdsr = launcher.build_command(
        _args(tmp_path, system="sdsr_v40", scale_interface="p4"), recipe
    )
    _, msp = launcher.build_command(
        _args(tmp_path, system="msp", scale_interface="p4"), recipe
    )

    assert _resolved_diff(sdsr, msp) == {
        "model.selected_system",
        "model.msp_c2f_blocks",
    }


def test_h2_h5_pairwise_diffs_are_only_preregistered_axes(tmp_path):
    launcher = _load_launcher()
    recipe = launcher.load_recipe()
    configs = {}
    for run_id, system, interface in (
        ("H2", "msp", "p4"),
        ("H3", "msp", "p345"),
        ("H4", "sdsr_v40", "p4"),
        ("H5", "sdsr_v40", "p345"),
    ):
        _, configs[run_id] = launcher.build_command(
            _args(
                tmp_path,
                system=system,
                scale_interface=interface,
                seed=44,
                detector_init_seed=1044,
            ),
            recipe,
        )

    scale_diffs_msp = {
        "model.scale_interface",
        "model.projector_scale",
        "model.dec_level_n_points",
        "model.projector_distill_level_weights",
        "model.msp_c2f_blocks",
    }
    scale_diffs_sdsr = {
        "model.scale_interface",
        "model.projector_scale",
        "model.dec_level_n_points",
        "model.projector_distill_level_weights",
    }
    projector_diffs = {"model.selected_system", "model.msp_c2f_blocks"}

    assert _resolved_diff(configs["H2"], configs["H3"]) == scale_diffs_msp
    assert _resolved_diff(configs["H4"], configs["H5"]) == scale_diffs_sdsr
    assert _resolved_diff(configs["H2"], configs["H4"]) == projector_diffs
    assert _resolved_diff(configs["H3"], configs["H5"]) == projector_diffs
    assert _resolved_diff(configs["H2"], configs["H5"]) == (
        scale_diffs_msp | projector_diffs
    )
    assert _resolved_diff(configs["H3"], configs["H4"]) == (
        scale_diffs_msp | projector_diffs
    )


@pytest.mark.parametrize("invalid", ["p3", "p5", "p34", "p45", "2,3,1"])
def test_invalid_scale_interfaces_are_rejected(monkeypatch, invalid):
    launcher = _load_launcher()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(LAUNCHER_PATH),
            "--track", "analysis",
            "--system", "msp",
            "--scale-interface", invalid,
            "--data-root", "/tmp/data",
            "--output-dir", "/tmp/output",
            "--seed", "44",
        ],
    )
    with pytest.raises(SystemExit) as error:
        launcher.parse_args()
    assert error.value.code == 2


def test_system_track_uses_only_approved_30_epoch_mapping(tmp_path):
    launcher = _load_launcher()
    recipe = launcher.load_recipe()
    command, _ = launcher.build_command(
        _args(tmp_path, track="system", system="msp"), recipe
    )

    assert command[0] == sys.executable
    assert _value_after(command, "--subset") == "full"
    assert _value_after(command, "--epochs") == "30"
    assert _value_after(command, "--lr-drop") == "25"
    assert _value_after(command, "--dense-o2o-image-stop-epoch") == "15"
    assert _value_after(command, "--dense-o2o-copyblend-stop-epoch") == "26"
    assert _value_after(command, "--projector-type") == "multiscale"
    assert "--use-ema" in command

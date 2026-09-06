import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "paper" / "ICLR27_RECIPE.json"
LAUNCHER_PATH = ROOT / "paper" / "run_iclr27_experiment.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("paper_launcher", LAUNCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(tmp_path, track="analysis", system="sdsr_v40", use_ema=True):
    return argparse.Namespace(
        track=track,
        system=system,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "output",
        seed=43,
        detector_init_seed=1043,
        num_workers=2,
        device="cuda",
        use_ema=use_ema,
    )


def _value_after(command, flag):
    return command[command.index(flag) + 1]


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

from types import SimpleNamespace

from rfdetr.engine import is_multi_scale_training_active


def test_multi_scale_stays_active_without_stop_epoch():
    args = SimpleNamespace(multi_scale=True, multi_scale_stop_epoch=-1)

    assert is_multi_scale_training_active(args, 0)
    assert is_multi_scale_training_active(args, 100)


def test_multi_scale_stops_at_configured_epoch():
    args = SimpleNamespace(multi_scale=True, multi_scale_stop_epoch=21)

    assert is_multi_scale_training_active(args, 20)
    assert not is_multi_scale_training_active(args, 21)
    assert not is_multi_scale_training_active(args, 22)


def test_multi_scale_remains_disabled_when_not_requested():
    args = SimpleNamespace(multi_scale=False, multi_scale_stop_epoch=21)

    assert not is_multi_scale_training_active(args, 0)

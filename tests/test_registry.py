from src.models.baseline_cnn import BaselineCNN
from src.models.registry import ARCH_NAMES, ARCH_SPECS, archs_with_ckpt, ckpt_filename


def test_arch_names_match_specs():
    assert set(ARCH_NAMES) == set(ARCH_SPECS.keys())
    assert set(ARCH_NAMES) == {"baseline_cnn", "mobilenet_transfer"}


def test_each_spec_has_a_working_model_fn():
    for name, spec in ARCH_SPECS.items():
        model = spec["model_fn"]()
        assert model is not None
        assert callable(spec["model_fn"])


def test_model_fn_builds_a_fresh_instance_each_call():
    model_fn = ARCH_SPECS["baseline_cnn"]["model_fn"]
    assert model_fn() is not model_fn()


def test_baseline_cnn_uses_baseline_cnn_class():
    assert isinstance(ARCH_SPECS["baseline_cnn"]["model_fn"](), BaselineCNN)


def test_mean_and_std_are_three_channels():
    for spec in ARCH_SPECS.values():
        assert len(spec["mean"]) == 3
        assert len(spec["std"]) == 3


def test_ckpt_filename_format():
    assert ckpt_filename("baseline_cnn") == "baseline_cnn_seed42.pt"
    assert ckpt_filename("mobilenet_transfer", seed=7) == "mobilenet_transfer_seed7.pt"


def test_archs_with_ckpt_adds_ckpt_without_mutating_specs():
    archs = archs_with_ckpt(seed=999)
    assert archs["baseline_cnn"]["ckpt"] == "baseline_cnn_seed999.pt"
    assert "model_fn" in archs["baseline_cnn"]
    assert "ckpt" not in ARCH_SPECS["baseline_cnn"]

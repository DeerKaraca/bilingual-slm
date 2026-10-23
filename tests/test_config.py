from pathlib import Path

import pytest

from slm.config import load_config

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.mark.parametrize("path", sorted(CONFIGS.rglob("*.yaml")), ids=lambda p: p.stem)
def test_all_configs_load(path):
    cfg = load_config(path)
    assert cfg.name == path.stem
    assert cfg.total_steps > 0


def test_experiment_token_budgets():
    exp = CONFIGS / "experiments"
    mixes = [load_config(p) for p in exp.glob("tr[0-9]*.yaml")]
    assert {c.train.total_tokens for c in mixes} == {600_000_000}

    tr030 = load_config(exp / "tr030.yaml")
    control = load_config(exp / "tr_only_control.yaml")
    tr_tokens_seen = tr030.train.total_tokens * tr030.data.tr_fraction
    assert control.train.total_tokens == pytest.approx(tr_tokens_seen)
    assert control.data.tr_fraction == 1.0


def test_overrides_and_validation(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text("name: x\n")
    cfg = load_config(path, {"model": {"n_layers": 4}, "train": {"betas": [0.9, 0.99]}})
    assert cfg.model.n_layers == 4
    assert cfg.train.betas == (0.9, 0.99)

    with pytest.raises(KeyError):
        load_config(path, {"model": {"n_layer": 4}})
    with pytest.raises(ValueError):
        load_config(path, {"data": {"tr_fraction": 1.5}})

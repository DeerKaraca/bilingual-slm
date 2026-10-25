import csv

import numpy as np
import pytest

from slm.config import load_config
from slm.eval.analysis import CONTROL, MIXTURES, SEED, analyse, load_run, main, verdict
from slm.train import LOG_FIELDS


def write_log(runs_dir, name, tr_end, en_end, start=3.0, fraction_done=1.0):
    """A log.csv whose BPB falls linearly from `start` to the given final values."""
    cfg = load_config(f"configs/experiments/{name}.yaml")
    steps = np.arange(100, int(cfg.total_steps * fraction_done) + 1, 100).tolist()
    if fraction_done == 1.0 and steps[-1] != cfg.total_steps:
        steps.append(cfg.total_steps)
    (runs_dir / name).mkdir(parents=True)
    with open(runs_dir / name / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, LOG_FIELDS)
        writer.writeheader()
        for step in steps:
            t = step / cfg.total_steps
            writer.writerow({"step": step, "tokens": step * cfg.tokens_per_step, "lr": 0, "train_loss": 0,
                             "tr_bpb": start + (tr_end - start) * t, "en_bpb": start + (en_end - start) * t,
                             "tokens_per_sec": 0, "elapsed_min": 0})


FINALS = {"tr000": (2.8, 1.10), "tr010": (1.40, 1.12), "tr030": (1.20, 1.15), SEED: (1.22, 1.16),
          "tr050": (2.50, 1.20), "tr100": (1.00, 1.80), CONTROL: (1.30, 2.00)}


@pytest.fixture
def runs_dir(tmp_path):
    for name, (tr, en) in FINALS.items():
        write_log(tmp_path, name, tr, en)
    return tmp_path


def runs_from(runs_dir):
    return {n: r for n in [*MIXTURES, CONTROL, SEED] if (r := load_run(runs_dir, n))}


def test_verdict_uses_twice_the_seed_range():
    assert verdict(0.05, 0.02, "up", "down") == "up"
    assert verdict(-0.05, 0.02, "up", "down") == "down"
    assert verdict(0.03, 0.02, "up", "down").startswith("inconclusive")
    assert verdict(0.5, None, "up", "down").startswith("no seed-noise")


def test_transfer_and_noise(runs_dir):
    result = analyse(runs_from(runs_dir))
    assert result["noise"]["tr"] == pytest.approx(0.02)
    assert result["tr030_mean"]["tr"] == pytest.approx(1.21)
    assert result["transfer"]["bpb"] == pytest.approx(1.30 - 1.21)
    assert result["transfer"]["verdict"].startswith("English helps Turkish")


def test_interference_is_read_off_the_monolingual_curve(runs_dir):
    rows = {(q["run"], q["lang"]): q for q in analyse(runs_from(runs_dir))["interference"]}
    tr050 = rows[("tr050", "tr")]
    tr100 = load_run(runs_dir, "tr100")
    assert tr050["tokens"] == pytest.approx(300e6, rel=0.01)
    assert tr050["monolingual_curve_bpb"] == pytest.approx(np.interp(tr050["tokens"], tr100.seen("tr"), tr100.bpb["tr"]))
    assert tr050["verdict"] == "interference likely"  # 2.50 is far above tr100's curve at 300M tokens
    assert ("tr000", "tr") not in rows and ("tr100", "en") not in rows  # no tokens of that language


def test_unfinished_runs_are_left_out_of_final_results(tmp_path):
    write_log(tmp_path, "tr030", 1.2, 1.15)
    write_log(tmp_path, CONTROL, 1.3, 2.0, fraction_done=0.5)
    result = analyse(runs_from(tmp_path))
    assert result["complete"] == ["tr030"] and "transfer" not in result
    assert result["noise"] is None


def test_main_writes_report_and_figures(runs_dir, tmp_path, monkeypatch):
    out = tmp_path / "report"
    monkeypatch.setattr("sys.argv", ["analysis", "--runs-dir", str(runs_dir), "--out", str(out)])
    main()
    names = {p.name for p in out.iterdir()}
    assert {"results.md", "results.json", "curves.png", "final_vs_share.png", "transfer.png", "seed_noise.png"} <= names
    assert "positive transfer" in (out / "results.md").read_text()

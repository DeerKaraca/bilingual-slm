import csv

import numpy as np
import pytest

from slm.config import load_config
from slm.eval.analysis import ALL_RUNS, CONTROL, SEED, X3, X3_SEED, analyse, load_run, main, verdict
from slm.train import LOG_FIELDS


def write_log(runs_dir, name, tr_end, en_end, start=3.0, fraction_done=1.0, tr_dip_at=None):
    """A log.csv whose BPB falls linearly from `start` to the given final values. With tr_dip_at, the
    Turkish curve first falls 0.05 below tr_end at that fraction of training, then rises to tr_end."""
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
            tr = start + (tr_end - start) * t
            if tr_dip_at is not None:
                low = tr_end - 0.05
                tr = start + (low - start) * t / tr_dip_at if t <= tr_dip_at else low + 0.05 * (t - tr_dip_at) / (1 - tr_dip_at)
            writer.writerow({"step": step, "tokens": step * cfg.tokens_per_step, "lr": 0, "train_loss": 0,
                             "tr_bpb": tr, "en_bpb": start + (en_end - start) * t,
                             "tokens_per_sec": 0, "elapsed_min": 0})


FINALS = {"tr000": (2.8, 1.10), "tr010": (1.40, 1.12), "tr030": (1.20, 1.15), SEED: (1.22, 1.16),
          "tr050": (2.50, 1.20), "tr100": (1.00, 1.80), CONTROL: (1.30, 2.00),
          X3: (1.25, 2.10), X3_SEED: (1.26, 2.10)}


@pytest.fixture
def runs_dir(tmp_path):
    for name, (tr, en) in FINALS.items():
        write_log(tmp_path, name, tr, en)
    return tmp_path


def runs_from(runs_dir):
    return {n: r for n in ALL_RUNS if (r := load_run(runs_dir, n))}


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


def test_compute_matched_control(runs_dir):
    c = analyse(runs_from(runs_dir))["compute_matched"]
    assert c["x3_tr_bpb"] == pytest.approx(1.255) and c["noise"] == pytest.approx(0.02)
    assert c["diff"] == pytest.approx(1.255 - 1.21)  # 0.045 > 2 x 0.02
    assert c["verdict"].startswith("the mixture is better") and c["x3_seed_range"] == pytest.approx(0.01)


@pytest.mark.parametrize("x3_tr, expected", [(1.22, "the two cannot be distinguished"), (1.10, "at equal compute, repeating")])
def test_compute_matched_verdicts(tmp_path, x3_tr, expected):
    for name in ("tr030", SEED):
        write_log(tmp_path, name, *FINALS[name])
    for name in (X3, X3_SEED):
        write_log(tmp_path, name, x3_tr, 2.1)
    assert analyse(runs_from(tmp_path))["compute_matched"]["verdict"].startswith(expected)


def test_overfitting_check(tmp_path):
    for name in ("tr030", SEED):
        write_log(tmp_path, name, *FINALS[name])
    write_log(tmp_path, X3, 1.25, 2.1)  # keeps falling
    write_log(tmp_path, X3_SEED, 1.25, 2.1, tr_dip_at=1000 / 2288)  # lowest at step 1000, then rises by 0.05
    rows = {o["run"]: o for o in analyse(runs_from(tmp_path))["overfitting"]}
    assert rows[X3]["rise"] == pytest.approx(0) and rows[X3]["verdict"] == "no overfitting visible"
    assert rows[X3_SEED]["rise"] == pytest.approx(0.05, abs=0.005) and rows[X3_SEED]["verdict"] == "overfitting visible"
    assert rows[X3_SEED]["lowest_in_pass"] == 2  # step 1000 = 262M tokens; passes are 180M long


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
    assert {"results.md", "results.json", "curves.png", "final_vs_share.png", "transfer.png", "seed_noise.png",
            "repeat_control.png"} <= names
    assert "positive transfer" in (out / "results.md").read_text()

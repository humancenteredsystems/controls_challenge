import json
from pathlib import Path
import pytest

import optimization.tournament_optimizer as topt

@pytest.fixture(autouse=True)
def stub_evaluate(monkeypatch):
    """Stub evaluate to assign deterministic stats without running rollouts."""
    def fake_evaluate(ps, data_files, model_path, max_files):
        # Assign cost based on id length to vary values
        ps.stats = {"avg_total_cost": float(len(ps.id))}
    monkeypatch.setattr(topt, "evaluate", fake_evaluate)

def test_run_tournament_creates_output(tmp_path):
    # Clean up existing JSON outputs if any
    plans_dir = Path(__file__).parent.parent.parent / "plans"
    for fname in ["tournament_progress.json", "tournament_archive.json"]:
        path = plans_dir / fname
        if path.exists():
            path.unlink()
    # Run a small tournament
    topt.run_tournament(
        data_files=[],
        model_path="dummy_model.onnx",
        rounds=3,
        pop_size=4,
        elite_pct=0.5,
        revive_pct=0.25,
        max_files=1,
        perturb_scale=0.1
    )
    # Verify output files are created
    prog = plans_dir / "tournament_progress.json"
    arch = plans_dir / "tournament_archive.json"
    assert prog.exists(), "tournament_progress.json should exist"
    assert arch.exists(), "tournament_archive.json should exist"
    # Verify progress JSON structure
    prog_data = json.loads(prog.read_text())
    assert "tournament_summary" in prog_data
    summary = prog_data["tournament_summary"]
    assert isinstance(summary, list) and len(summary) == 3
    for entry in summary:
        assert set(entry.keys()) == {"round", "elites", "revived", "new", "best_cost"}
    # Verify archive JSON structure
    arch_data = json.loads(arch.read_text())
    assert "archive" in arch_data
    archive_list = arch_data["archive"]
    assert isinstance(archive_list, list)
    # Each archived entry must have expected keys
    required_keys = {"id", "gains", "stats", "rounds_survived", "status"}
    for item in archive_list:
        assert set(item.keys()) == required_keys

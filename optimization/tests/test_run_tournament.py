import json
from pathlib import Path
import pytest

import optimization.tournament_optimizer as topt

@pytest.fixture(autouse=True)
def stub_evaluate(monkeypatch):
    """Stub evaluate to assign deterministic stats without running rollouts."""
    def fake_evaluate(ps, data_files, model, max_files, rng=None):
        # Assign cost based on id length to vary values
        ps.stats = {"avg_total_cost": float(len(ps.id))}
    monkeypatch.setattr(topt, "evaluate", fake_evaluate)
    
    # Mock TinyPhysicsModel to avoid file dependency
    class MockModel:
        def __init__(self, model_path, debug=False):
            pass
    monkeypatch.setattr(topt, "TinyPhysicsModel", MockModel)

def test_run_tournament_creates_output(tmp_path):
    # Clean up existing JSON outputs if any
    plans_dir = Path(__file__).parent.parent.parent / "plans"
    arch_path = plans_dir / "tournament_archive.json"
    if arch_path.exists():
        arch_path.unlink()
    
    # Run a small tournament
    topt.run_tournament(
        data_files=[],
        model_path="dummy_model.onnx",
        rounds=3,
        pop_size=4,
        elite_pct=0.5,
        revive_pct=0.25,
        max_files=1,
        perturb_scale=0.1,
        seed_from_archive=None,
        init_low_min=[0.25, 0.01, -0.25],
        init_low_max=[0.6, 0.12, -0.05],
        init_high_min=[0.15, 0.005, -0.15],
        init_high_max=[0.4, 0.08, -0.03],
        init_seed=None
    )
    
    # Verify archive output file is created
    assert arch_path.exists(), "tournament_archive.json should exist"
    
    # Verify archive JSON structure
    arch_data = json.loads(arch_path.read_text())
    assert "archive" in arch_data
    archive_list = arch_data["archive"]
    assert isinstance(archive_list, list)
    
    # Each archived entry must have expected keys
    required_keys = {"id", "low_gains", "high_gains", "stats"}
    for item in archive_list:
        assert set(item.keys()) == required_keys

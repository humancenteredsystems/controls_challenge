import pytest
from optimization import blender_tournament_optimizer as bto


def test_get_tournament_baseline_missing(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(RuntimeError):
        bto.get_tournament_baseline(str(missing))


def test_run_blender_tournament_missing_archive(monkeypatch, tmp_path):
    # Provide stub PID pairs to bypass file reading
    monkeypatch.setattr(bto, "load_top_pid_pairs", lambda path, n=5: [([0.1, 0.1, 0.1], [0.2, 0.2, 0.2])])

    class DummyModel:
        def __init__(self, model_path, debug=False):
            pass

    monkeypatch.setattr(bto, "TinyPhysicsModel", DummyModel)

    with pytest.raises(RuntimeError):
        bto.run_blender_tournament(str(tmp_path / "missing.json"), [], "model.onnx", rounds=1, pop_size=1, max_files=1)

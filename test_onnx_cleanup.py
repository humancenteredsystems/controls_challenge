import os
import pytest

from optimization import blender_tournament_optimizer as bto

class DummyModel:
    pass

def test_onnx_file_removed_on_success(tmp_path, monkeypatch):
    onnx_file = tmp_path / "model.onnx"

    def mock_train(architecture, training_data):
        onnx_file.write_text("dummy")
        return str(onnx_file)

    monkeypatch.setattr(bto, "train_blender_architecture", mock_train)
    monkeypatch.setattr(bto, "get_top_pid_pairs_from_archive", lambda: [([0,0,0], [0,0,0])])
    monkeypatch.setattr(bto.random, "sample", lambda data, k: data[:k])

    def mock_make_temp(pid1, pid2, path, arch_id):
        return "temp_controller"

    monkeypatch.setattr(bto, "_make_temp_neural_controller", mock_make_temp)
    monkeypatch.setattr(bto, "run_rollout", lambda df, cm, m: ({"total_cost": 1.0}, None, None))
    monkeypatch.setattr(bto, "cleanup_temp_controller", lambda name: None)

    cost = bto.evaluate_blender_architecture({"id": 1}, [], ["data"], DummyModel(), 0)
    assert cost == pytest.approx(1.0)
    assert not onnx_file.exists()

def test_onnx_file_removed_on_exception(tmp_path, monkeypatch):
    onnx_file = tmp_path / "model.onnx"

    def mock_train(architecture, training_data):
        onnx_file.write_text("dummy")
        return str(onnx_file)

    monkeypatch.setattr(bto, "train_blender_architecture", mock_train)
    monkeypatch.setattr(bto, "get_top_pid_pairs_from_archive", lambda: [([0,0,0], [0,0,0])])
    monkeypatch.setattr(bto.random, "sample", lambda data, k: data[:k])

    def mock_make_temp(pid1, pid2, path, arch_id):
        raise RuntimeError("fail")

    monkeypatch.setattr(bto, "_make_temp_neural_controller", mock_make_temp)

    with pytest.raises(RuntimeError):
        bto.evaluate_blender_architecture({"id": 1}, [], ["data"], DummyModel(), 0)

    assert not onnx_file.exists()

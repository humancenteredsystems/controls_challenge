import pytest
from optimization import blender_tournament_optimizer as bto


class DummyModel:
    pass


def _setup_artifacts(tmp_path, monkeypatch, rollout_func):
    """Helper to create temp model and controller with cleanup patches."""
    onnx_file = tmp_path / "model.onnx"
    temp_controller = tmp_path / "temp_controller.py"

    def mock_train(architecture, training_data, epochs=100, pretrained_path=None):
        onnx_file.write_text("dummy")
        return str(onnx_file)

    monkeypatch.setattr(bto, "train_model_with_hyperparameters", mock_train)
    monkeypatch.setattr(bto.random, "sample", lambda data, k: data[:k])
    monkeypatch.setattr(bto, "run_rollout", rollout_func)

    return onnx_file, temp_controller

    monkeypatch.setattr(bto, "create_temp_neural_controller", mock_make_temp)
    monkeypatch.setattr(bto, "run_rollout", lambda df, cm, m: ({"total_cost": 1.0}, None, None))
    monkeypatch.setattr(bto, "cleanup_temp_controller", lambda name: None)

    pid_pairs = [([0, 0, 0], [0, 0, 0])]
    cost = bto.evaluate_hyperparameters_on_pid_pairs({"id": 1}, "train.json", pid_pairs, ["data"], DummyModel(), 1)
    assert cost == pytest.approx(1.0)
    assert not onnx_file.exists()
    assert not temp_controller.exists()


def test_temp_files_removed_on_rollout_error(tmp_path, monkeypatch):
    def failing_rollout(df, cn, m):
        raise RuntimeError("fail")

    onnx_file, temp_controller = _setup_artifacts(
        tmp_path, monkeypatch, failing_rollout
    )

    training_file = tmp_path / "training.json"
    training_file.write_text("{}")

    arch = {"id": "1", "hidden_sizes": [], "dropout_rate": 0.1}
    cost = bto.evaluate_hyperparameters_on_pid_pairs(
        arch,
        str(training_file),
        [([0, 0, 0], [0, 0, 0])],
        ["file.csv"],
        DummyModel(),
        max_files=1,
    )

    assert cost == pytest.approx(1000.0)
    assert not onnx_file.exists()
    assert not temp_controller.exists()


def test_onnx_removed_if_controller_creation_fails(tmp_path, monkeypatch):
    onnx_file = tmp_path / "model.onnx"

    def mock_train(architecture, training_data, epochs=100, pretrained_path=None):
        onnx_file.write_text("dummy")
        return str(onnx_file)

    monkeypatch.setattr(bto, "train_model_with_hyperparameters", mock_train)
    monkeypatch.setattr(bto.random, "sample", lambda data, k: data[:k])

    def mock_make_temp(pid1, pid2, path, arch_id, norm_stats=None):
        raise RuntimeError("fail")

    monkeypatch.setattr(bto, "create_temp_neural_controller", mock_make_temp)

    pid_pairs = [([0, 0, 0], [0, 0, 0])]
    training_file = tmp_path / "train.json"
    training_file.write_text("{}")
    with pytest.raises(RuntimeError):
        bto.evaluate_hyperparameters_on_pid_pairs({"id": 1}, str(training_file), pid_pairs, ["data"], DummyModel(), 1)

    assert not onnx_file.exists()

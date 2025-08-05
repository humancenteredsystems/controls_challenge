import types
import sys
import pytest
from optimization import blender_tournament_optimizer as bto


def test_missing_pretrained_weights_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    dummy_nb = types.ModuleType("neural_blender_net")

    class DummyNet:
        def __init__(self, hidden_sizes, dropout_rate):
            pass

        def load_state_dict(self, state_dict):
            pass

    def dummy_train(*args, **kwargs):
        pass

    dummy_nb.BlenderNet = DummyNet
    dummy_nb.train_blender_net_from_json = dummy_train
    monkeypatch.setitem(sys.modules, "neural_blender_net", dummy_nb)

    dummy_torch = types.ModuleType("torch")

    def dummy_load(path, weights_only=False):
        return {}

    dummy_torch.load = dummy_load
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    hyperparams = {
        'id': 'testid',
        'hidden_sizes': [32, 16],
        'dropout_rate': 0.1,
        'learning_rate': 0.001,
        'epochs': 1,
        'batch_size': 1,
        'cost': float('inf'),
    }

    with pytest.raises(FileNotFoundError, match="train_blender.py"):
        bto.train_model_with_hyperparameters(hyperparams, 'train.json', pretrained_path='missing.pth')

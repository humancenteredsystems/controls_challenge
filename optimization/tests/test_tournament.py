import os
import csv
import tempfile
import pytest  # type: ignore
import numpy as np  # type: ignore

from .. import tournament_optimizer as topt

@pytest.fixture(autouse=True)
def rng_seed():
    np.random.seed(0)
    yield

def test_initialize_population_length_and_structure():
    pop = topt.initialize_population(5)
    assert isinstance(pop, list)
    assert len(pop) == 5
    for ps in pop:
        assert hasattr(ps, 'id')
        assert isinstance(ps.gains, dict)
        assert set(ps.gains.keys()) == {'low', 'high', 'dyn'}
        assert ps.status == "active"
        assert ps.rounds_survived == 0

def test_select_elites_and_status_rounds():
    # Create dummy ParameterSets with manual stats
    class DummyPS:
        def __init__(self, cost):
            self.stats = {'avg_total_cost': cost}
            self.rounds_survived = 0
            self.status = "active"
    pop = [DummyPS(cost) for cost in [10, 20, 30, 40, 50]]
    elites = topt.select_elites(pop, elite_pct=0.4)  # top 2
    assert len(elites) == 2
    # Check that elites have lowest costs
    costs = sorted([ps.stats['avg_total_cost'] for ps in elites])
    assert costs == [10, 20]
    # Check status and rounds_survived updates
    for ps in elites:
        assert ps.status == "active"
        assert ps.rounds_survived == 1
    # Check eliminated status for others
    eliminated = [ps for ps in pop if ps not in elites]
    for ps in eliminated:
        assert ps.status == "eliminated"

def test_revival_lottery_weights_and_status():
    # Create dummy ParameterSets with rounds_survived stats
    class DummyPS:
        def __init__(self, survived):
            self.stats = {}
            self.rounds_survived = survived
            self.status = "eliminated"
    archive = [DummyPS(0), DummyPS(1), DummyPS(2)]
    revived = topt.revival_lottery(archive, revive_pct=0.5, pop_size=4)
    # Should revive at least 1
    assert len(revived) >= 1
    for ps in revived:
        assert ps.status == "active"
        assert ps.rounds_survived >= 1

def test_generate_new_variation_and_count():
    # Create a base ParameterSet
    base_gains = {'low': [0.3, 0.03, -0.1], 'high': [0.2, 0.01, -0.05], 'dyn': [0.4, 0.1, -0.1]}
    base_ps = topt.ParameterSet(base_gains)
    new_sets = topt.generate_new(5, base_ps)
    assert isinstance(new_sets, list)
    assert len(new_sets) == 5
    for ps in new_sets:
        # Each gain key present
        assert set(ps.gains.keys()) == set(base_gains.keys())
        # Each gain list length matches
        for key in base_gains:
            assert len(ps.gains[key]) == len(base_gains[key])
        # New IDs and metadata
        assert ps.id != base_ps.id
        assert ps.status == "active"
        assert ps.rounds_survived == 0
        # Ensure exploration: at least one gain differs
        assert any(
            ps.gains[key][i] != base_gains[key][i]
            for key in base_gains
            for i in range(len(base_gains[key]))
        )

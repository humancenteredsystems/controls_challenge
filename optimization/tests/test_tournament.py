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
        assert hasattr(ps, 'low_gains')
        assert hasattr(ps, 'high_gains')
        assert isinstance(ps.low_gains, list)
        assert isinstance(ps.high_gains, list)
        assert len(ps.low_gains) == 3
        assert len(ps.high_gains) == 3
        assert ps.status == "active"
        assert ps.rounds_survived == 0

def test_select_elites_and_status_rounds():
    # Create real ParameterSets with manual stats
    pop = []
    for cost in [10, 20, 30, 40, 50]:
        ps = topt.ParameterSet([0.3, 0.03, -0.1], [0.2, 0.01, -0.05])
        ps.stats = {'avg_total_cost': cost}
        ps.rounds_survived = 0
        ps.status = "active"
        pop.append(ps)
    
    elites = topt.select_elites(pop, elite_pct=0.4)  # top 2
    assert len(elites) == 2
    # Check that elites have lowest costs
    costs = sorted([ps.stats['avg_total_cost'] for ps in elites])
    assert costs == [10, 20]
    # Check status updates
    for ps in elites:
        assert ps.status == "active"
        # rounds_survived is not incremented by select_elites function
        assert ps.rounds_survived == 0
    # Check eliminated status for others
    eliminated = [ps for ps in pop if ps not in elites]
    for ps in eliminated:
        assert ps.status == "eliminated"

def test_revival_lottery_weights_and_status():
    # Create real ParameterSets with rounds_survived stats
    archive = []
    for survived in [0, 1, 2]:
        ps = topt.ParameterSet([0.3, 0.03, -0.1], [0.2, 0.01, -0.05])
        ps.stats = {}
        ps.rounds_survived = survived
        ps.status = "eliminated"
        archive.append(ps)
    
    revived = topt.revival_lottery(archive, revive_pct=0.5, pop_size=4)
    # Should revive at least 1
    assert len(revived) >= 1
    for ps in revived:
        assert ps.status == "active"
        assert ps.rounds_survived >= 1

def test_generate_new_variation_and_count():
    # Create a base ParameterSet
    base_low_gains = [0.3, 0.03, -0.1]
    base_high_gains = [0.2, 0.01, -0.05]
    base_ps = topt.ParameterSet(base_low_gains, base_high_gains)
    new_sets = topt.generate_new(5, base_ps, perturb_scale=0.1)
    assert isinstance(new_sets, list)
    assert len(new_sets) == 5
    for ps in new_sets:
        # Check structure
        assert hasattr(ps, 'low_gains')
        assert hasattr(ps, 'high_gains')
        assert len(ps.low_gains) == 3
        assert len(ps.high_gains) == 3
        # New IDs and metadata
        assert ps.id != base_ps.id
        assert ps.status == "active"
        assert ps.rounds_survived == 0
        # Ensure exploration: at least one gain differs
        assert (ps.low_gains != base_low_gains or ps.high_gains != base_high_gains)


def test_initialize_population_invalid_archive_raises(tmp_path):
    archive_path = tmp_path / "archive.json"
    archive_path.write_text("{}")
    with pytest.raises(RuntimeError) as exc:
        topt.initialize_population(5, seed_from_archive=str(archive_path))
    assert str(archive_path) in str(exc.value)

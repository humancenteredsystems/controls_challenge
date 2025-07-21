# Refactor 01: Tournament-Based Optimization

This plan implements an evolutionary tournament system for parameter search.  
Mark each item `[ ]` as done `[x]` when complete.

## 1. Setup & Data Structures
- [ ] Create `optimization/tournament_optimizer.py`
- [ ] Define `ParameterSet` class with:
  - `id` (string)
  - `gains` (dict of low/high/dyn arrays)
  - `stats` (avg, std, min, max cost)
  - `rounds_survived` (int)
  - `status` (“active”/“eliminated”)
- [ ] Create `Population` (list of active `ParameterSet`)
- [ ] Create `Archive` (list of all tested `ParameterSet`)

## 2. Initialization (Round 0)
- [ ] Implement `initialize_population(n: int) -> List[ParameterSet]`
  - Generate `n` initial combinations (reuse existing search-space logic)
  - Set `rounds_survived = 0`, `status="active"`
  - Add all to `Archive`

## 3. Evaluation
- [ ] Wrap simulation call (`run_rollout`) into `evaluate(ps: ParameterSet, files: List[str])`
  - Run on `k` files
  - Populate `ps.stats`
- [ ] Batch-evaluate all active `ParameterSet`s in a round

## 4. Selection Functions
- [ ] `select_elites(population, pct: float) -> List[ParameterSet]`
  - Sort by `avg_cost`, pick top `%`
  - Increment `rounds_survived`
- [ ] `revival_lottery(archive, pct: float) -> List[ParameterSet]`
  - Filter eliminated sets
  - Weighted random by `rounds_survived`
  - Increment `rounds_survived`
- [ ] `generate_new(m: int, around: ParameterSet) -> List[ParameterSet]`
  - Gaussian perturbation around best-known
  - Assign new `id`, `rounds_survived=0`

## 5. Tournament Loop
- [ ] Implement `run_tournament(rounds: int, pop_size: int)`
  - For each `round` in `1..rounds`:
    1. Evaluate active population
    2. Select elites
    3. Conduct revival lottery
    4. Generate new entrants to refill population
    5. Update `population` and `archive`
    6. Log round summary to `plans/tournament_progress.json`  

## 6. Logging & Persistence
- [ ] Write per-round summary file:
  ```json
  {
    "round": 1,
    "elites": [...ids...],
    "revived": [...ids...],
    "new": [...ids...],
    "best_cost": 82.34
  }
  ```
- [ ] Append master results to `Archive` JSON

## 7. CLI & Configuration
- [ ] Add `if __name__ == "__main__":` entrypoint
  - Parse arguments: `--rounds`, `--pop_size`, `--files`, `--elite_pct`, `--revive_pct`
  - Load data file list
  - Call `run_tournament`
- [ ] Document usage in `README.md`

## 8. Testing & Validation
- [ ] Unit tests for `initialize_population`, `select_elites`, `revival_lottery`, `generate_new`
- [ ] Integration test: run 2 rounds on 5 files, assert population size and statistics logging
- [ ] Full validation: apply top 5 survivors on 100+ files, compare against baseline

## 9. Review & Iteration
- [ ] Review performance trends by `rounds_survived`
- [ ] Adjust percentages (`elite_pct`, `revive_pct`) based on observed diversity vs convergence
- [ ] Refine perturbation magnitude for `generate_new`

---

_End of Plan (≤200 lines)_

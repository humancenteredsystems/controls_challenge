# REFRESH REFRACTOR 02: CLI-DRIVEN CONFIGURATION FOR STAGE 1 AND PIPELINE PARAMETER FORWARDING

**Issue**  
Stage 1’s blended 2-PID optimizer (optimization/blended_2pid_optimizer.py) uses hard-coded constants (300 combinations, 25 files/test, first 50 CSVs, fixed model/data paths). The top-level pipeline script (run_complete_training_pipeline.py) likewise hard-codes all Stage 1 and tournament parameters, forcing edits to source.

**Goal**  
Expose all Stage 1 grid-search and tournament parameters via command-line flags. Modify both the Stage 1 optimizer and the pipeline script so that:
- Users can override defaults without editing code.
- `run_complete_training_pipeline.py` parses global flags and forwards them to each stage runner.

---

## 1. Files Affected

- `optimization/blended_2pid_optimizer.py`
- `run_complete_training_pipeline.py`
- Documentation: `README.md` (or create `PIPELINE_EXECUTION_GUIDE.md`)

---

## 2. Stage 1 Optimizer (optimization/blended_2pid_optimizer.py)

1. **Import and parser setup**  
   - Add `import argparse` at top.  
   - In `main()`, create an `ArgumentParser` with flags:  
     ```python
     parser.add_argument("--num_combinations", type=int, default=300, help="…")
     parser.add_argument("--max_files_per_test", type=int, default=25, help="…")
     parser.add_argument("--num_files", type=int, default=50, help="…")
     parser.add_argument("--model_path", type=str, default=None, help="Override ONNX path")
     parser.add_argument("--data_dir", type=str, default=None, help="Override data dir")
     args = parser.parse_args()
     ```
2. **Replace embedded constants**  
   - Use `args.model_path` or default `models/tinyphysics.onnx`.  
   - Use `args.data_dir` or default `data/`.  
   - Limit data file list to `args.num_files`.  
   - Call `optimize_comprehensive(data_files, num_combinations=args.num_combinations, max_files_per_test=args.max_files_per_test)`.  
3. **Remove obsolete flags**  
   - Remove any `--comprehensive` or hard-coded constants in docstrings or usage examples.  

---

## 3. Pipeline Script (run_complete_training_pipeline.py)

1. **Import and parser setup**  
   - Add `import argparse`.  
   - At top of `main()`, define a parser with flags:  
     ```python
     parser.add_argument("--stage1-combinations", type=int, default=300)
     parser.add_argument("--stage1-max-files", type=int, default=25)
     parser.add_argument("--stage1-num-files", type=int, default=50)
     parser.add_argument("--t1-rounds", type=int, default=12)
     parser.add_argument("--t1-pop-size", type=int, default=25)
     parser.add_argument("--t1-max-files", type=int, default=30)
     parser.add_argument("--t2-rounds", type=int, default=12)
     parser.add_argument("--t2-pop-size", type=int, default=25)
     parser.add_argument("--t2-max-files", type=int, default=50)
     parser.add_argument("--perturb-scale", type=float, default=0.05)
     args = parser.parse_args()
     ```
2. **Stage runner functions**  
   - Update `run_stage_1_grid_search(args)` signature to accept `args`.  
   - Build its command list:
     ```python
     cmd = [
       sys.executable, "optimization/blended_2pid_optimizer.py",
       "--num_combinations", str(args.stage1_combinations),
       "--max_files_per_test", str(args.stage1_max_files),
       "--num_files", str(args.stage1_num_files),
       "--model_path", args.model_path or "",
       "--data_dir", args.data_dir or ""
     ]
     ```
   - Similarly update `run_stage_2_tournament_1(args)` and `run_stage_3_tournament_2(args)` to use `--rounds`, `--pop_size`, `--max_files`, `--perturb_scale`.
3. **Invoke runners with args**  
   - In `main()`, replace:
     ```python
     if not run_stage_1_grid_search(): ...
     ```
     with:
     ```python
     if not run_stage_1_grid_search(args): ...
     ```
   - Remove any legacy hard-coded values or flags.

---

## 4. Documentation Updates

- **README.md** or new **PIPELINE_EXECUTION_GUIDE.md**  
  - Document all new flags, defaults, and example invocations:  
    ```bash
    python run_complete_training_pipeline.py \
      --stage1-combinations 100 \
      --stage1-max-files 10 \
      --t1-rounds 5 \
      --t2-pop-size 30
    ```
  - Show how to inspect help:  
    ```bash
    python optimization/blended_2pid_optimizer.py --help
    python run_complete_training_pipeline.py --help
    ```

---

## 5. Testing

1. **Help output**  
   - `python optimization/blended_2pid_optimizer.py --help`  
   - `python run_complete_training_pipeline.py --help`  
2. **Dry run**  
   - Small search:  
     ```bash
     python optimization/blended_2pid_optimizer.py \
       --num_combinations 5 --max_files_per_test 3 --num_files 10
     ```
   - Pipeline with reduced params:  
     ```bash
     python run_complete_training_pipeline.py \
       --stage1-combinations 5 --t1-rounds 2 --t2-max-files 5
     ```
3. **End-to-end**  
   - Confirm pipeline picks up and applies all flags in each stage.

---

## 6. Rollout

- Merge changes into `develop` branch.  
- CI should pass, ensure no regressions.  
- Tag release for CLI-configurable pipeline.

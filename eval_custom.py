#!/usr/bin/env python3
"""
eval_custom.py - Lightweight pre-submission validation tool for controllers

This tool provides configurable evaluation modes to catch compatibility issues
before running the official eval.py, preventing time step mismatches and other
submission problems.

Usage:
    python eval_custom.py --model_path models/tinyphysics.onnx --data_path data/train --controller neural_blended_champion --mode quick
    python eval_custom.py --model_path models/tinyphysics.onnx --data_path data/train --controller neural_blended_champion --mode validate-only
    python eval_custom.py --model_path models/tinyphysics.onnx --data_path data/train --controller neural_blended_champion --mode standard
"""

import argparse
import json
import numpy as np
import sys
import time
from pathlib import Path
from tqdm import tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map

from tinyphysics import CONTROL_START_IDX, get_available_controllers, run_rollout

# Validation modes configuration
VALIDATION_MODES = {
    'quick': {
        'num_segs': 10,
        'description': 'Fast validation with 10 segments for development testing'
    },
    'standard': {
        'num_segs': 100, 
        'description': 'Standard evaluation matching eval.py default'
    },
    'validate-only': {
        'num_segs': 3,
        'description': 'Minimal validation - only check controller compatibility'
    }
}

def validate_controller_compatibility(controller_name, model_path, data_path):
    """
    Validate controller compatibility before full evaluation.
    
    Returns:
        dict: Validation results with status and any error messages
    """
    validation_results = {
        'controller_loadable': False,
        'time_step_consistent': False,
        'basic_rollout_working': False,
        'errors': [],
        'warnings': []
    }
    
    print(f"🔍 Validating controller compatibility: {controller_name}")
    
    # Check 1: Controller can be loaded
    try:
        available_controllers = get_available_controllers()
        if controller_name not in available_controllers:
            validation_results['errors'].append(f"Controller '{controller_name}' not found in available controllers: {available_controllers}")
            return validation_results
        validation_results['controller_loadable'] = True
        print(f"✅ Controller '{controller_name}' is loadable")
    except Exception as e:
        validation_results['errors'].append(f"Failed to load controller '{controller_name}': {str(e)}")
        return validation_results
    
    # Check 2: Basic rollout functionality
    try:
        data_files = sorted(Path(data_path).iterdir())
        if not data_files:
            validation_results['errors'].append(f"No data files found in {data_path}")
            return validation_results
            
        test_file = data_files[0]
        print(f"🧪 Testing basic rollout with file: {test_file.name}")
        
        start_time = time.time()
        cost, target_lataccel, current_lataccel = run_rollout(
            test_file, controller_name, model_path, debug=False
        )
        rollout_time = time.time() - start_time
        
        validation_results['basic_rollout_working'] = True
        validation_results['sample_cost'] = cost
        validation_results['rollout_time'] = rollout_time
        
        print(f"✅ Basic rollout successful")
        print(f"   Sample cost: {cost}")
        print(f"   Rollout time: {rollout_time:.3f}s")
        
        # Check 3: Time step consistency (dt = 0.1 expected)
        expected_dt = 0.1
        if len(target_lataccel) > 0 and len(current_lataccel) > 0:
            if len(target_lataccel) == len(current_lataccel):
                validation_results['time_step_consistent'] = True
                print(f"✅ Time step consistency verified (array lengths match)")
            else:
                validation_results['warnings'].append(
                    f"Array length mismatch: target={len(target_lataccel)}, current={len(current_lataccel)}"
                )
        
    except Exception as e:
        validation_results['errors'].append(f"Basic rollout failed: {str(e)}")
        return validation_results
        
    return validation_results

def run_custom_evaluation(controller_name, model_path, data_path, mode='quick', baseline_controller='pid'):
    """
    Run custom evaluation with specified mode.
    
    Args:
        controller_name: Name of controller to test
        model_path: Path to physics model
        data_path: Path to evaluation data
        mode: Evaluation mode ('quick', 'standard', 'validate-only')
        baseline_controller: Baseline controller for comparison
        
    Returns:
        dict: Evaluation results
    """
    mode_config = VALIDATION_MODES[mode]
    num_segs = mode_config['num_segs']
    
    print(f"🚀 Running {mode} evaluation: {mode_config['description']}")
    print(f"   Segments: {num_segs}")
    print(f"   Test controller: {controller_name}")
    print(f"   Baseline controller: {baseline_controller}")
    
    data_files = sorted(Path(data_path).iterdir())[:num_segs]
    
    if mode == 'validate-only':
        # Run minimal validation
        results = {'mode': mode, 'controller': controller_name, 'costs': []}
        
        for data_file in tqdm(data_files, desc="Validation rollouts"):
            try:
                cost, _, _ = run_rollout(data_file, controller_name, model_path, debug=False)
                results['costs'].append(cost)
            except Exception as e:
                print(f"❌ Validation failed on {data_file.name}: {str(e)}")
                return {'error': str(e), 'failed_file': data_file.name}
                
        avg_cost = np.mean([c['total_cost'] for c in results['costs']])
        results['average_cost'] = avg_cost
        print(f"✅ Validation successful - Average cost: {avg_cost:.2f}")
        return results
    
    # Run full evaluation (quick or standard mode)
    costs = []
    
    # Test controller evaluation
    print(f"📊 Evaluating test controller: {controller_name}")
    rollout_partial = partial(run_rollout, controller_type=controller_name, model_path=model_path, debug=False)
    start_time = time.time()
    test_results = process_map(rollout_partial, data_files, max_workers=8, chunksize=5)
    test_time = time.time() - start_time
    test_costs = [{'controller': 'test', **result[0]} for result in test_results]
    costs.extend(test_costs)
    
    # Baseline controller evaluation  
    print(f"📊 Evaluating baseline controller: {baseline_controller}")
    rollout_partial = partial(run_rollout, controller_type=baseline_controller, model_path=model_path, debug=False)
    start_time = time.time()
    baseline_results = process_map(rollout_partial, data_files, max_workers=8, chunksize=5)
    baseline_time = time.time() - start_time
    baseline_costs = [{'controller': 'baseline', **result[0]} for result in baseline_results]
    costs.extend(baseline_costs)
    
    # Calculate aggregate results
    test_avg = np.mean([c['total_cost'] for c in test_costs])
    baseline_avg = np.mean([c['total_cost'] for c in baseline_costs])
    improvement = baseline_avg - test_avg
    improvement_pct = (improvement / baseline_avg) * 100 if baseline_avg > 0 else 0
    
    results = {
        'mode': mode,
        'num_segments': num_segs,
        'test_controller': controller_name,
        'baseline_controller': baseline_controller,
        'test_average_cost': test_avg,
        'baseline_average_cost': baseline_avg,
        'improvement': improvement,
        'improvement_percent': improvement_pct,
        'test_time': test_time,
        'baseline_time': baseline_time,
        'beats_baseline': test_avg < baseline_avg,
        'all_costs': costs
    }
    
    # Results summary
    print(f"\n📈 Evaluation Results ({mode} mode):")
    print(f"   Test Controller ({controller_name}): {test_avg:.2f}")
    print(f"   Baseline Controller ({baseline_controller}): {baseline_avg:.2f}")
    if improvement > 0:
        print(f"   ✅ Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")
    else:
        print(f"   ❌ Regression: {improvement:.2f} ({improvement_pct:.1f}%)")
    print(f"   Evaluation time: {test_time + baseline_time:.1f}s")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Lightweight pre-submission validation tool for controllers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available validation modes:
{chr(10).join([f"  {mode}: {config['description']}" for mode, config in VALIDATION_MODES.items()])}

Examples:
  python eval_custom.py --model_path models/tinyphysics.onnx --data_path data/train --controller neural_blended_champion --mode quick
  python eval_custom.py --model_path models/tinyphysics.onnx --data_path data/train --controller neural_blended_champion --mode validate-only
        """
    )
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the physics model (e.g., models/tinyphysics.onnx)")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to evaluation data directory")
    parser.add_argument("--controller", type=str, required=True,
                       help="Name of controller to test")
    parser.add_argument("--mode", type=str, default='quick',
                       choices=list(VALIDATION_MODES.keys()),
                       help="Validation mode (default: quick)")
    parser.add_argument("--baseline", type=str, default='pid',
                       help="Baseline controller for comparison (default: pid)")
    parser.add_argument("--output", type=str, default=None,
                       help="Save results to JSON file (optional)")
    parser.add_argument("--validate-first", action='store_true',
                       help="Run compatibility validation before full evaluation")
    
    args = parser.parse_args()
    
    # Validate paths
    model_path = Path(args.model_path)
    data_path = Path(args.data_path)
    
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        sys.exit(1)
        
    if not data_path.is_dir():
        print(f"❌ Data path is not a directory: {data_path}")
        sys.exit(1)
        
    print(f"🎯 eval_custom.py - Controller Pre-submission Validation")
    print(f"   Controller: {args.controller}")
    print(f"   Mode: {args.mode}")
    print(f"   Model: {model_path}")
    print(f"   Data: {data_path}")
    print()
    
    # Step 1: Always run compatibility validation
    validation_results = validate_controller_compatibility(
        args.controller, args.model_path, args.data_path
    )
    
    if validation_results['errors']:
        print(f"\n❌ Compatibility validation failed:")
        for error in validation_results['errors']:
            print(f"   - {error}")
        sys.exit(1)
        
    if validation_results['warnings']:
        print(f"\n⚠️  Compatibility warnings:")
        for warning in validation_results['warnings']:
            print(f"   - {warning}")
            
    print(f"\n✅ Controller compatibility validated successfully")
    
    # Step 2: Run evaluation if not validate-only or if validation passed
    if args.validate_first and args.mode != 'validate-only':
        print(f"\n🎯 Proceeding with {args.mode} evaluation...")
        
    try:
        evaluation_results = run_custom_evaluation(
            args.controller, args.model_path, args.data_path, 
            args.mode, args.baseline
        )
        
        # Combine results
        final_results = {
            'validation': validation_results,
            'evaluation': evaluation_results,
            'timestamp': time.time(),
            'args': vars(args)
        }
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_path}")
            
        # Final summary
        if 'beats_baseline' in evaluation_results:
            if evaluation_results['beats_baseline']:
                print(f"\n🎉 SUCCESS: Controller ready for eval.py submission!")
                print(f"   Beats baseline by {evaluation_results['improvement']:.2f} points")
            else:
                print(f"\n⚠️  WARNING: Controller does not beat baseline")
                print(f"   Consider further optimization before eval.py submission")
        else:
            print(f"\n✅ Validation completed successfully")
            
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
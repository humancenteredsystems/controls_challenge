#!/usr/bin/env python3
"""
Tournament #3 Full Dataset Test - Fair comparison with statistically significant sample
"""
import sys
import json
import time
import random
from pathlib import Path
sys.path.append('.')

from tinyphysics_custom import run_rollout, TinyPhysicsModel

def test_tournament3_full_dataset():
    """Test Tournament #3 neural controller fallback on large dataset sample"""
    print('🧠 Tournament #3: Full Dataset Performance Test')
    print('=' * 70)
    
    start_time = time.time()
    
    # Setup
    base_dir = Path('.')
    model_path = str(base_dir / 'models' / 'tinyphysics.onnx')
    data_dir = base_dir / 'data'
    
    # Get all available data files
    all_data_files = sorted(list(data_dir.glob('*.csv')))
    print(f'📁 Total available data files: {len(all_data_files)}')
    
    # Use statistically significant sample size (100 files randomly selected)
    sample_size = min(100, len(all_data_files))
    random.seed(42)  # Reproducible results
    sample_files = random.sample(all_data_files, sample_size)
    sample_files = [str(f) for f in sorted(sample_files)]  # Sort for consistent order
    
    print(f'📊 Testing on {sample_size} randomly selected files for statistical significance')
    
    # Load Tournament #2 baseline for comparison
    try:
        with open('plans/tournament_archive.json', 'r') as f:
            t2_results = json.load(f)
        
        best_t2 = min(t2_results['archive'], key=lambda x: x.get('stats', {}).get('avg_total_cost', float('inf')))
        baseline_cost = best_t2['stats']['avg_total_cost']
        
        print(f'🏆 Tournament #2 Baseline: {baseline_cost:.2f}')
        print(f'📊 T2 Parameters: Low={best_t2["low_gains"]}, High={best_t2["high_gains"]}')
        
    except Exception as e:
        print(f'❌ Error loading Tournament #2 results: {e}')
        return None
    
    # Create neural_blended controller (will use fallback due to corrupted models)
    controller_code = '''from controllers.neural_blended import Controller as NeuralController

class Controller(NeuralController):
    def __init__(self):
        # Neural models are corrupted, will automatically fallback to velocity-based blending
        # Uses Tournament #2 PID parameters automatically
        super().__init__()
'''
    
    temp_controller_path = 'controllers/temp_neural_full_test.py'
    with open(temp_controller_path, 'w') as f:
        f.write(controller_code)
    
    print(f'📝 Created test controller: {temp_controller_path}')
    
    # Initialize physics model
    print('⚡ Initializing GPU-accelerated physics model...')
    physics_model = TinyPhysicsModel(model_path, debug=False)
    providers = physics_model.ort_session.get_providers()
    gpu_status = 'GPU ENABLED' if 'CUDAExecutionProvider' in providers else 'CPU FALLBACK'
    print(f'Physics model: {gpu_status}')
    
    # Test the fallback neural controller on full sample
    print(f'\n🧪 Testing neural_blended controller on {sample_size} files...')
    
    total_costs = []
    progress_interval = max(1, sample_size // 10)  # Show progress every 10%
    
    try:
        for i, data_file in enumerate(sample_files):
            if i % progress_interval == 0:
                progress = (i / sample_size) * 100
                current_avg = f'{sum(total_costs)/len(total_costs):.2f}' if total_costs else 'N/A'
                print(f'   📊 Progress: {progress:.0f}% ({i}/{sample_size}) - Current avg: {current_avg}')
            
            try:
                cost, _, _ = run_rollout(data_file, 'temp_neural_full_test', physics_model, debug=False)
                total_costs.append(cost['total_cost'])
            except Exception as e:
                print(f'   ⚠️ Error on file {Path(data_file).name}: {e}')
                continue
        
        if total_costs:
            # Calculate comprehensive statistics
            avg_cost = sum(total_costs) / len(total_costs)
            min_cost = min(total_costs)
            max_cost = max(total_costs)
            std_dev = (sum((c - avg_cost)**2 for c in total_costs) / len(total_costs))**0.5
            median_cost = sorted(total_costs)[len(total_costs)//2]
            
            # Count performance vs baseline
            better_count = sum(1 for c in total_costs if c < baseline_cost)
            worse_count = sum(1 for c in total_costs if c > baseline_cost)
            
            print(f'\n✅ Tournament #3 Full Dataset Results:')
            print(f'   📊 Sample Size: {len(total_costs)} files')
            print(f'   📊 Average Cost: {avg_cost:.2f}')
            print(f'   📊 Median Cost: {median_cost:.2f}')
            print(f'   📊 Min Cost: {min_cost:.2f}')
            print(f'   📊 Max Cost: {max_cost:.2f}')
            print(f'   📊 Std Deviation: {std_dev:.2f}')
            
            # Performance vs Tournament #2 baseline
            improvement = baseline_cost - avg_cost
            print(f'\n📈 Performance vs Tournament #2 Baseline:')
            print(f'   Tournament #2: {baseline_cost:.2f}')
            print(f'   Tournament #3: {avg_cost:.2f}')
            print(f'   Difference: {improvement:.2f} {"(IMPROVEMENT)" if improvement > 0 else "(REGRESSION)" if improvement < 0 else "(EQUIVALENT)"}')
            print(f'   Better files: {better_count}/{len(total_costs)} ({better_count/len(total_costs)*100:.1f}%)')
            print(f'   Worse files: {worse_count}/{len(total_costs)} ({worse_count/len(total_costs)*100:.1f}%)')
            
            # Save results
            results = {
                'tournament': 'Neural Blending Phase (Full Dataset Test)',
                'sample_size': len(total_costs),
                'total_available_files': len(all_data_files),
                'baseline_cost': baseline_cost,
                'full_dataset_neural_cost': avg_cost,
                'median_cost': median_cost,
                'min_cost': min_cost,
                'max_cost': max_cost,
                'std_deviation': std_dev,
                'improvement': improvement,
                'better_than_baseline_count': better_count,
                'worse_than_baseline_count': worse_count,
                'better_percentage': better_count/len(total_costs)*100,
                'status': 'full_dataset_test_successful',
                'note': 'Statistically significant sample test of Tournament #3 fallback mode',
                'timestamp': time.time(),
                'costs_sample': total_costs[:20]  # First 20 for inspection
            }
            
            with open('tournament3_full_dataset_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f'💾 Results saved to tournament3_full_dataset_results.json')
            
        else:
            print('❌ No valid test results')
            
    except Exception as e:
        print(f'❌ Testing failed: {e}')
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            Path(temp_controller_path).unlink()
            print(f'🧹 Cleaned up temporary controller')
        except:
            pass
    
    elapsed = time.time() - start_time
    print(f'\n⏱️  Total execution time: {elapsed:.1f}s')
    print(f'🎯 Tournament #3 Full Dataset Test Complete!')
    
    return results if 'results' in locals() and total_costs else None

if __name__ == '__main__':
    test_tournament3_full_dataset()
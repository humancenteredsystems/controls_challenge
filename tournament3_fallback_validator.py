#!/usr/bin/env python3
"""
Tournament #3 Fallback Validator - Test neural_blended controller with corrupted models
"""
import sys
import json
import time
from pathlib import Path
sys.path.append('.')

from tinyphysics_custom import run_rollout, TinyPhysicsModel

def test_tournament3_fallback():
    """Test Tournament #3 neural controller with fallback mode"""
    print('🧠 Tournament #3: Fallback Mode Validation')
    print('=' * 60)
    
    start_time = time.time()
    
    # Setup
    base_dir = Path('.')
    model_path = str(base_dir / 'models' / 'tinyphysics.onnx')
    data_dir = base_dir / 'data'
    
    # Use subset of data files for testing
    data_files = [str(f) for f in sorted(data_dir.glob('*.csv'))[:4]]
    print(f'📁 Using {len(data_files)} data files for validation')
    
    # Load Tournament #2 baseline
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
    
    temp_controller_path = 'controllers/temp_neural_fallback_test.py'
    with open(temp_controller_path, 'w') as f:
        f.write(controller_code)
    
    print(f'📝 Created test controller: {temp_controller_path}')
    
    # Initialize physics model
    print('⚡ Initializing GPU-accelerated physics model...')
    physics_model = TinyPhysicsModel(model_path, debug=False)
    providers = physics_model.ort_session.get_providers()
    gpu_status = 'GPU ENABLED' if 'CUDAExecutionProvider' in providers else 'CPU FALLBACK'
    print(f'Physics model: {gpu_status}')
    
    # Test the fallback neural controller
    print(f'\n🧪 Testing neural_blended controller (fallback mode)...')
    
    total_costs = []
    try:
        for i, data_file in enumerate(data_files):
            print(f'   📊 Testing file {i+1}/{len(data_files)}: {Path(data_file).name}')
            cost, _, _ = run_rollout(data_file, 'temp_neural_fallback_test', physics_model, debug=False)
            total_costs.append(cost['total_cost'])
            print(f'      Cost: {cost["total_cost"]:.2f}')
        
        if total_costs:
            avg_cost = sum(total_costs) / len(total_costs)
            min_cost = min(total_costs)
            max_cost = max(total_costs)
            
            print(f'\n✅ Neural Fallback Results:')
            print(f'   📊 Average Cost: {avg_cost:.2f}')
            print(f'   📊 Min Cost: {min_cost:.2f}')
            print(f'   📊 Max Cost: {max_cost:.2f}')
            print(f'   📊 Std Dev: {(sum((c - avg_cost)**2 for c in total_costs) / len(total_costs))**0.5:.2f}')
            
            # Compare to Tournament #2 baseline
            improvement = baseline_cost - avg_cost
            print(f'\n📈 Performance vs Tournament #2:')
            print(f'   Tournament #2: {baseline_cost:.2f}')
            print(f'   Neural Fallback: {avg_cost:.2f}')
            print(f'   Difference: {improvement:.2f} {"(IMPROVEMENT)" if improvement > 0 else "(REGRESSION)" if improvement < 0 else "(EQUIVALENT)"}')
            
            # Save results
            results = {
                'tournament': 'Neural Blending Phase (Fallback Mode)',
                'baseline_cost': baseline_cost,
                'fallback_neural_cost': avg_cost,
                'improvement': improvement,
                'test_results': {
                    'avg_cost': avg_cost,
                    'min_cost': min_cost,
                    'max_cost': max_cost,
                    'num_tests': len(total_costs), 
                    'costs': total_costs
                },
                'status': 'fallback_mode_successful',
                'note': 'Neural models corrupted, using velocity-based fallback with Tournament #2 parameters',
                'timestamp': time.time()
            }
            
            with open('tournament3_fallback_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f'💾 Results saved to tournament3_fallback_results.json')
            
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
    print(f'🎯 Tournament #3 Fallback Validation Complete!')
    
    return results if total_costs else None

if __name__ == '__main__':
    test_tournament3_fallback()
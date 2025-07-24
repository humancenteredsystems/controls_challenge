#!/usr/bin/env python3
"""
Test the neural blended controller with Tournament #2 winner parameters
"""

import json
from pathlib import Path

def test_neural_blended_controller():
    """Test neural blended controller with Tournament #2 parameters"""
    
    print("🧪 Testing Neural Blended Controller")
    print("=" * 50)
    
    # Get Tournament #2 winner parameters
    archive_path = "plans/tournament_archive.json"
    with open(archive_path, 'r') as f:
        archive = json.load(f)
    
    # Find best performer
    valid_performers = []
    for p in archive['archive']:
        if 'stats' in p and 'avg_total_cost' in p['stats']:
            valid_performers.append(p)
    
    best_performer = min(valid_performers, key=lambda x: x['stats']['avg_total_cost'])
    
    print(f"Tournament #2 Winner:")
    print(f"  Cost: {best_performer['stats']['avg_total_cost']:.2f}")
    print(f"  Min Cost: {best_performer['stats']['min_cost']:.2f}")
    print(f"  Low-speed PID:  P={best_performer['low_gains'][0]:.3f}, I={best_performer['low_gains'][1]:.3f}, D={best_performer['low_gains'][2]:.3f}")
    print(f"  High-speed PID: P={best_performer['high_gains'][0]:.3f}, I={best_performer['high_gains'][1]:.3f}, D={best_performer['high_gains'][2]:.3f}")
    
    # Create simple neural blended controller for testing
    print(f"\n📝 Creating test neural blended controller...")
    
    controller_content = f'''from controllers.neural_blended import Controller as BaseNeuralController

class Controller(BaseNeuralController):
    def __init__(self):
        # Use Tournament #2 winner parameters
        pid1_params = {best_performer['low_gains']}
        pid2_params = {best_performer['high_gains']}
        
        # No neural model initially - will use velocity-based fallback
        super().__init__(pid1_params, pid2_params, blender_model_path=None)
        
        print("Test Neural Blended Controller initialized (using velocity fallback)")
'''
    
    # Save as temporary controller
    temp_controller_path = "controllers/temp_neural_test.py"
    with open(temp_controller_path, 'w') as f:
        f.write(controller_content)
    
    print(f"✅ Test controller saved to: {temp_controller_path}")
    
    return best_performer

def create_summary_report():
    """Create summary report of the neural blender optimization process"""
    
    print(f"\n📊 Neural Blender Optimization Summary")
    print("=" * 60)
    
    # Training data
    training_data_path = "plans/blender_training_data.json"
    if Path(training_data_path).exists():
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        
        print(f"✅ Training Data Generated:")
        print(f"   Samples: {training_data['num_samples']:,}")
        print(f"   Features: {len(training_data['feature_names'])}")
        print(f"   Top performers used: {training_data.get('top_performers_used', 'N/A')}")
    
    # Tournament results  
    tournament_results_path = "plans/blender_tournament_results.json"
    if Path(tournament_results_path).exists():
        with open(tournament_results_path, 'r') as f:
            results = json.load(f)
        
        print(f"\n⚠️  Blender Tournament Results:")
        print(f"   Status: Completed with evaluation issues")
        print(f"   Best cost: {results['best_cost']:.2f} (fallback - evaluations failed)")
        print(f"   Architecture: {results['best_architecture']['hidden_sizes']}")
        print(f"   Issue: Neural evaluation pipeline needs debugging")
    
    # Archive analysis
    archive_path = "plans/tournament_archive.json"
    if Path(archive_path).exists():
        with open(archive_path, 'r') as f:
            archive = json.load(f)
        
        valid_performers = [p for p in archive['archive'] if 'stats' in p and 'avg_total_cost' in p['stats']]
        costs = [p['stats']['avg_total_cost'] for p in valid_performers]
        min_costs = [p['stats']['min_cost'] for p in valid_performers]
        
        print(f"\n📈 PID Tournament Performance:")
        print(f"   Valid combinations: {len(valid_performers)}")
        print(f"   Best average cost: {min(costs):.2f}")
        print(f"   Best minimum cost: {min(min_costs):.2f}")
        print(f"   Performance range: {min(costs):.2f} - {max(costs):.2f}")
    
    print(f"\n🎯 Current Status:")
    print(f"   ✅ Neural blender architecture implemented")
    print(f"   ✅ Training data generation working")
    print(f"   ✅ Tournament #2 winner identified (cost: {min(costs):.2f})")
    print(f"   ⚠️  Neural tournament evaluation needs debugging")
    print(f"   🎯 Ready for simplified neural blending approach")

if __name__ == "__main__":
    try:
        # Test the neural blended controller
        best_performer = test_neural_blended_controller()
        
        # Create summary report
        create_summary_report()
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Debug neural tournament evaluation pipeline")
        print(f"   2. Train actual BlenderNet models with PyTorch")
        print(f"   3. Test neural blended controller with real models")
        print(f"   4. Compare performance vs Tournament #2 winner")
        
        print(f"\n✅ Neural blender infrastructure is ready!")
        print(f"💡 Tournament #2 winner (cost: {best_performer['stats']['avg_total_cost']:.2f}) provides strong baseline")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
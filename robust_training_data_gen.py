#!/usr/bin/env python3
"""
Robust training data generation for BlenderNet - handles inconsistent archive data
"""

import json
import random
from pathlib import Path

def generate_training_data():
    """Generate training data from tournament archive with error handling"""
    
    print("🔬 Generating BlenderNet training data...")
    
    # Load tournament archive
    archive_path = "plans/tournament_archive.json"
    with open(archive_path, 'r') as f:
        archive = json.load(f)
    
    performers = archive['archive']
    print(f"Found {len(performers)} parameter combinations in archive")
    
    # Filter performers with valid stats
    valid_performers = []
    for p in performers:
        if 'stats' in p and 'avg_total_cost' in p['stats']:
            valid_performers.append(p)
        else:
            print(f"Skipping performer {p.get('id', 'unknown')} - missing stats")
    
    print(f"Found {len(valid_performers)} valid performers with complete stats")
    
    if len(valid_performers) == 0:
        print("❌ No valid performers found!")
        return 0
    
    # Get top 10 performers
    top_performers = sorted(valid_performers, key=lambda x: x['stats']['avg_total_cost'])[:10]
    
    print("Top 5 performers:")
    for i, p in enumerate(top_performers[:5]):
        cost = p['stats']['avg_total_cost']
        min_cost = p['stats'].get('min_cost', 'N/A')
        low_gains = p.get('low_gains', [0, 0, 0])
        high_gains = p.get('high_gains', [0, 0, 0])
        print(f"  {i+1}. Cost: {cost:.2f} (min: {min_cost}) "
              f"Low: [{low_gains[0]:.3f}, {low_gains[1]:.3f}, {low_gains[2]:.3f}] "
              f"High: [{high_gains[0]:.3f}, {high_gains[1]:.3f}, {high_gains[2]:.3f}]")
    
    # Generate training samples
    training_samples = []
    samples_per_combo = 1000  # 10k total samples
    
    print(f"Generating {samples_per_combo} samples per combination...")
    
    for combo_idx, combo in enumerate(top_performers):
        for sample_idx in range(samples_per_combo):
            # Generate realistic vehicle state
            v_ego = max(5, min(70, random.gauss(30, 15)))  # Speed centered around 30 mph
            roll_lataccel = random.uniform(-3, 3)
            a_ego = random.uniform(-2, 2)
            
            # Generate control error
            error = random.uniform(-1, 1)
            error_integral = random.uniform(-0.5, 0.5)
            error_derivative = random.uniform(-0.2, 0.2)
            
            # Generate future plan features
            future_mean = random.uniform(-2, 2)
            future_std = max(0, random.uniform(0, 1.5))
            
            # Create feature vector
            features = [v_ego, roll_lataccel, a_ego, error, error_integral, 
                       error_derivative, future_mean, future_std]
            
            # Compute optimal blend weight (heuristic-based)
            # Use a more sophisticated blending strategy
            speed_normalized = (v_ego - 5) / 65  # Normalize speed to [0,1]
            
            if v_ego < 20:
                # Very low speed: strongly favor low-speed PID
                base_blend = 0.85
            elif v_ego < 35:
                # Low-medium speed: gradually transition
                base_blend = 0.7 - 0.3 * ((v_ego - 20) / 15)
            elif v_ego < 50:  
                # Medium-high speed: favor high-speed PID
                base_blend = 0.4 - 0.2 * ((v_ego - 35) / 15)
            else:
                # High speed: strongly favor high-speed PID
                base_blend = 0.15
            
            # Adjust based on error magnitude and trajectory complexity
            error_magnitude = abs(error)
            if error_magnitude > 0.5:
                # High error: slightly favor more aggressive controller
                base_blend -= 0.1
            
            if future_std > 0.8:
                # Complex trajectory: favor controller with better transient response
                base_blend -= 0.05
            
            # Clamp to valid range
            optimal_blend = max(0.0, min(1.0, base_blend))
            
            training_samples.append((features, optimal_blend))
        
        if (combo_idx + 1) % 2 == 0:
            print(f"  Completed {combo_idx + 1}/{len(top_performers)} combinations")
    
    print(f"Generated {len(training_samples)} training samples")
    
    # Save training data
    training_data = {
        'num_samples': len(training_samples),
        'source_archive': archive_path,
        'top_performers_used': len(top_performers),
        'feature_names': [
            'v_ego', 'roll_lataccel', 'a_ego', 'error',
            'error_integral', 'error_derivative', 
            'future_lataccel_mean', 'future_lataccel_std'
        ],
        'samples': [
            {
                'features': sample[0],
                'blend_weight': sample[1]
            }
            for sample in training_samples
        ]
    }
    
    output_path = "plans/blender_training_data.json"
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"✅ Training data saved to: {output_path}")
    
    # Validate data distribution
    blend_weights = [s[1] for s in training_samples]
    avg_blend = sum(blend_weights) / len(blend_weights)
    low_blend = sum(1 for w in blend_weights if w < 0.3) / len(blend_weights)
    mid_blend = sum(1 for w in blend_weights if 0.3 <= w <= 0.7) / len(blend_weights)  
    high_blend = sum(1 for w in blend_weights if w > 0.7) / len(blend_weights)
    
    print(f"\nTraining data validation:")
    print(f"  Average blend weight: {avg_blend:.3f}")
    print(f"  Low blend (< 0.3):    {low_blend:.1%}")
    print(f"  Mid blend (0.3-0.7):  {mid_blend:.1%}")
    print(f"  High blend (> 0.7):   {high_blend:.1%}")
    
    # Check feature ranges
    features_array = [s[0] for s in training_samples]
    feature_names = training_data['feature_names']
    
    print(f"\nFeature ranges:")
    for i, name in enumerate(feature_names):
        values = [f[i] for f in features_array]
        min_val = min(values)
        max_val = max(values)
        avg_val = sum(values) / len(values)
        print(f"  {name:20}: [{min_val:6.2f}, {max_val:6.2f}] avg={avg_val:6.2f}")
    
    return len(training_samples)

if __name__ == "__main__":
    try:
        num_samples = generate_training_data()
        print(f"\n🎉 Successfully generated {num_samples} training samples!")
        print("🚀 Ready for BlenderNet tournament optimization!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
#!/usr/bin/env python3
"""
Simplified training data generation for BlenderNet
"""

import json
import math
import random
from pathlib import Path

import numpy as np

def generate_training_data():
    """Generate training data from tournament archive"""
    
    print("🔬 Generating BlenderNet training data...")
    
    # Load tournament archive
    archive_path = "plans/tournament_archive.json"
    with open(archive_path, 'r') as f:
        archive = json.load(f)
    
    performers = archive['archive']
    print(f"Found {len(performers)} parameter combinations in archive")
    
    # Get top 10 performers
    top_performers = sorted(performers, key=lambda x: x['stats']['avg_total_cost'])[:10]
    
    print("Top 5 performers:")
    for i, p in enumerate(top_performers[:5]):
        cost = p['stats']['avg_total_cost']
        min_cost = p['stats']['min_cost']
        print(f"  {i+1}. Cost: {cost:.2f} (min: {min_cost:.2f})")
    
    # Generate training samples
    training_samples = []
    samples_per_combo = 1000  # 10k total samples
    
    for combo in top_performers:
        for _ in range(samples_per_combo):
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
            
            # Compute optimal blend weight (heuristic)
            if v_ego < 25:
                optimal_blend = 0.8  # Favor low-speed PID
            elif v_ego > 45:
                optimal_blend = 0.2  # Favor high-speed PID
            else:
                # Blend based on error and complexity
                speed_factor = (v_ego - 25) / 20  # 0 to 1
                error_factor = min(abs(error), 1.0)
                blend = 0.8 - 0.6 * speed_factor + 0.1 * error_factor
                optimal_blend = max(0.0, min(1.0, blend))
            
            training_samples.append((features, optimal_blend))
    
    print(f"Generated {len(training_samples)} training samples")

    # Compute feature statistics for normalization
    feature_array = np.array([s[0] for s in training_samples], dtype=np.float32)
    feature_means = feature_array.mean(axis=0).tolist()
    feature_stds = feature_array.std(axis=0).tolist()

    # Save training data
    training_data = {
        'num_samples': len(training_samples),
        'feature_names': [
            'v_ego', 'roll_lataccel', 'a_ego', 'error',
            'error_integral', 'error_derivative',
            'future_lataccel_mean', 'future_lataccel_std'
        ],
        'feature_stats': {
            'mean': feature_means,
            'std': feature_stds,
        },
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
    
    # Validate data
    blend_weights = [s[1] for s in training_samples]
    low_blend = sum(1 for w in blend_weights if w < 0.3) / len(blend_weights)
    mid_blend = sum(1 for w in blend_weights if 0.3 <= w <= 0.7) / len(blend_weights)  
    high_blend = sum(1 for w in blend_weights if w > 0.7) / len(blend_weights)
    
    print(f"Blend weight distribution:")
    print(f"  Low (< 0.3):   {low_blend:.1%}")
    print(f"  Mid (0.3-0.7): {mid_blend:.1%}")
    print(f"  High (> 0.7):  {high_blend:.1%}")
    
    return len(training_samples)

if __name__ == "__main__":
    try:
        num_samples = generate_training_data()
        print(f"\n🎉 Successfully generated {num_samples} training samples!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
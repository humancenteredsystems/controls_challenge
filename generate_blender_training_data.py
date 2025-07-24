#!/usr/bin/env python3
"""
Generate training data for BlenderNet from PID tournament archive results.

This script analyzes the tournament archive to create supervised learning data
for training neural networks to predict optimal PID blending weights.
"""

import json
import numpy as np
import random
from pathlib import Path
from collections import defaultdict

def load_tournament_archive(archive_path="plans/tournament_archive.json"):
    """Load PID tournament archive"""
    
    if not Path(archive_path).exists():
        raise FileNotFoundError(f"Tournament archive not found: {archive_path}")
    
    with open(archive_path, 'r') as f:
        archive = json.load(f)
    
    return archive

def analyze_archive_performance(archive):
    """Analyze archive to understand performance patterns"""
    
    print("📊 Archive Analysis:")
    print("-" * 40)
    
    performers = archive['archive']
    costs = [p['stats']['avg_total_cost'] for p in performers]
    min_costs = [p['stats']['min_cost'] for p in performers]
    
    print(f"Total parameter combinations: {len(performers)}")
    print(f"Average cost range: {min(costs):.2f} - {max(costs):.2f}")
    print(f"Best minimum cost: {min(min_costs):.2f}")
    print(f"Performance gap: {max(costs)/min(costs):.1f}x")
    
    # Identify top performers
    top_performers = sorted(performers, key=lambda x: x['stats']['avg_total_cost'])[:10]
    
    print(f"\nTop 10 Performers:")
    for i, performer in enumerate(top_performers):
        low_gains = performer['low_gains']
        high_gains = performer['high_gains']
        cost = performer['stats']['avg_total_cost']
        min_cost = performer['stats']['min_cost']
        
        print(f"  {i+1:2d}. Cost: {cost:6.2f} (min: {min_cost:5.2f}) "
              f"Low: [{low_gains[0]:.3f}, {low_gains[1]:.3f}, {low_gains[2]:.3f}] "
              f"High: [{high_gains[0]:.3f}, {high_gains[1]:.3f}, {high_gains[2]:.3f}]")
    
    return top_performers

def compute_optimal_blend_weight(v_ego, error_magnitude, future_complexity, scenario_type="normal"):
    """
    Compute optimal blend weight based on vehicle state and scenario.
    
    This is a heuristic-based approach that would be replaced by actual
    simulation-based optimization in a full implementation.
    
    Args:
        v_ego: Vehicle speed
        error_magnitude: Magnitude of control error
        future_complexity: Complexity of future trajectory
        scenario_type: Type of driving scenario
    
    Returns:
        Optimal blend weight [0,1] for PID1 vs PID2
    """
    
    # Base blending based on speed (PID1 for low speed, PID2 for high speed)
    speed_blend = np.clip((v_ego - 20) / 30, 0, 1)  # Transition from 20-50 mph
    
    # Adjust based on error magnitude
    error_factor = min(error_magnitude, 1.0)
    if error_magnitude > 0.5:
        # High error: favor more aggressive controller (usually PID2)
        error_adjustment = 0.2
    else:
        # Low error: favor smoother controller (usually PID1)  
        error_adjustment = -0.2
    
    # Adjust based on future trajectory complexity
    complexity_factor = min(future_complexity, 1.0)
    if complexity_factor > 0.7:
        # Complex trajectory: favor controller with better transient response
        complexity_adjustment = 0.1
    else:
        # Simple trajectory: favor steady-state performance
        complexity_adjustment = -0.1
    
    # Scenario-specific adjustments
    scenario_adjustment = 0
    if scenario_type == "highway":
        scenario_adjustment = 0.1  # Favor high-speed controller
    elif scenario_type == "city":
        scenario_adjustment = -0.1  # Favor low-speed controller
    
    # Combine all factors
    optimal_blend = speed_blend + error_adjustment + complexity_adjustment + scenario_adjustment
    optimal_blend = np.clip(optimal_blend, 0.0, 1.0)
    
    return optimal_blend

def generate_training_samples(top_performers, num_samples=10000):
    """
    Generate training samples for BlenderNet
    
    Args:
        top_performers: List of top performing PID combinations
        num_samples: Number of training samples to generate
    
    Returns:
        List of (features, optimal_blend_weight) tuples
    """
    
    print(f"\n🔬 Generating {num_samples} training samples...")
    
    training_samples = []
    samples_per_combo = num_samples // len(top_performers)
    
    for combo_idx, combo in enumerate(top_performers):
        print(f"  Processing combo {combo_idx+1}/{len(top_performers)}: "
              f"{samples_per_combo} samples")
        
        for sample_idx in range(samples_per_combo):
            # Generate realistic vehicle state
            v_ego = np.random.gamma(2, 15)  # Speed: gamma distribution biased toward 20-40 mph
            v_ego = np.clip(v_ego, 5, 70)
            
            roll_lataccel = np.random.normal(0, 1.5)  # Lateral acceleration
            roll_lataccel = np.clip(roll_lataccel, -4, 4)
            
            a_ego = np.random.normal(0, 1.0)  # Longitudinal acceleration  
            a_ego = np.clip(a_ego, -3, 3)
            
            # Generate control error (more realistic distribution)
            error = np.random.laplace(0, 0.3)  # Laplace distribution for control errors
            error = np.clip(error, -2, 2)
            
            # Generate error dynamics
            error_integral = np.random.normal(0, 0.2)
            error_integral = np.clip(error_integral, -1, 1)
            
            error_derivative = np.random.normal(0, 0.1) 
            error_derivative = np.clip(error_derivative, -0.5, 0.5)
            
            # Generate future plan characteristics
            future_lataccel_mean = np.random.normal(0, 1.0)
            future_lataccel_mean = np.clip(future_lataccel_mean, -3, 3)
            
            future_lataccel_std = np.random.exponential(0.5)  # Std is always positive
            future_lataccel_std = np.clip(future_lataccel_std, 0, 2)
            
            # Create feature vector (8 dimensions)
            features = [
                v_ego,
                roll_lataccel, 
                a_ego,
                error,
                error_integral,
                error_derivative,
                future_lataccel_mean,
                future_lataccel_std
            ]
            
            # Compute optimal blend weight
            error_magnitude = abs(error)
            future_complexity = future_lataccel_std
            
            # Determine scenario type based on speed and complexity
            if v_ego > 50:
                scenario_type = "highway"
            elif v_ego < 25:
                scenario_type = "city"
            else:
                scenario_type = "normal"
            
            optimal_blend = compute_optimal_blend_weight(
                v_ego, error_magnitude, future_complexity, scenario_type
            )
            
            training_samples.append((features, optimal_blend))
    
    # Add remaining samples to reach target number
    remaining = num_samples - len(training_samples)
    if remaining > 0:
        # Generate additional samples from random top performers
        for _ in range(remaining):
            combo = random.choice(top_performers)
            # Use same generation logic as above
            features, optimal_blend = generate_single_sample()
            training_samples.append((features, optimal_blend))
    
    print(f"✅ Generated {len(training_samples)} training samples")
    return training_samples

def generate_single_sample():
    """Generate a single training sample"""
    
    # Same logic as in the loop above, extracted for reuse
    v_ego = np.clip(np.random.gamma(2, 15), 5, 70)
    roll_lataccel = np.clip(np.random.normal(0, 1.5), -4, 4)
    a_ego = np.clip(np.random.normal(0, 1.0), -3, 3)
    error = np.clip(np.random.laplace(0, 0.3), -2, 2)
    error_integral = np.clip(np.random.normal(0, 0.2), -1, 1)
    error_derivative = np.clip(np.random.normal(0, 0.1), -0.5, 0.5)
    future_lataccel_mean = np.clip(np.random.normal(0, 1.0), -3, 3)
    future_lataccel_std = np.clip(np.random.exponential(0.5), 0, 2)
    
    features = [v_ego, roll_lataccel, a_ego, error, error_integral, 
               error_derivative, future_lataccel_mean, future_lataccel_std]
    
    scenario_type = "highway" if v_ego > 50 else ("city" if v_ego < 25 else "normal")
    optimal_blend = compute_optimal_blend_weight(
        v_ego, abs(error), future_lataccel_std, scenario_type
    )
    
    return features, optimal_blend

def validate_training_data(training_samples):
    """Validate and analyze training data quality"""
    
    print(f"\n🔍 Training Data Validation:")
    print("-" * 40)
    
    features_array = np.array([sample[0] for sample in training_samples])
    labels_array = np.array([sample[1] for sample in training_samples])
    
    print(f"Total samples: {len(training_samples)}")
    print(f"Feature dimensions: {features_array.shape[1]}")
    
    # Feature statistics
    feature_names = [
        'v_ego', 'roll_lataccel', 'a_ego', 'error', 
        'error_integral', 'error_derivative', 
        'future_lataccel_mean', 'future_lataccel_std'
    ]
    
    print(f"\nFeature Statistics:")
    for i, name in enumerate(feature_names):
        mean_val = np.mean(features_array[:, i])
        std_val = np.std(features_array[:, i])
        min_val = np.min(features_array[:, i])
        max_val = np.max(features_array[:, i])
        print(f"  {name:20}: μ={mean_val:6.2f} σ={std_val:5.2f} "
              f"range=[{min_val:6.2f}, {max_val:6.2f}]")
    
    # Label statistics
    print(f"\nBlend Weight Statistics:")
    print(f"  Mean: {np.mean(labels_array):.3f}")
    print(f"  Std:  {np.std(labels_array):.3f}")
    print(f"  Range: [{np.min(labels_array):.3f}, {np.max(labels_array):.3f}]")
    
    # Check for reasonable distribution
    low_blend = np.sum(labels_array < 0.3) / len(labels_array)
    mid_blend = np.sum((labels_array >= 0.3) & (labels_array <= 0.7)) / len(labels_array)
    high_blend = np.sum(labels_array > 0.7) / len(labels_array)
    
    print(f"\nBlend Weight Distribution:")
    print(f"  Low (< 0.3):   {low_blend:.1%}")
    print(f"  Mid (0.3-0.7): {mid_blend:.1%}")
    print(f"  High (> 0.7):  {high_blend:.1%}")

def save_training_data(training_samples, output_path="plans/blender_training_data.json"):
    """Save training data to JSON file"""
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    training_data = {
        'num_samples': len(training_samples),
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
    
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\n💾 Training data saved to: {output_path}")
    print(f"   File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

def main():
    """Generate BlenderNet training data from tournament archive"""
    
    print("🏗️  BlenderNet Training Data Generation")
    print("=" * 50)
    
    try:
        # Load tournament archive
        archive = load_tournament_archive()
        
        # Analyze archive performance
        top_performers = analyze_archive_performance(archive)
        
        # Generate training samples
        training_samples = generate_training_samples(top_performers, num_samples=15000)
        
        # Validate training data
        validate_training_data(training_samples)
        
        # Save training data
        save_training_data(training_samples)
        
        print("\n✅ Training data generation complete!")
        print("🚀 Ready for BlenderNet training and tournament optimization.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating training data: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
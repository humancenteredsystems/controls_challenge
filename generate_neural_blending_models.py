#!/usr/bin/env python3
"""
🧠 Neural Blending Model Generator for Tournament #3
==================================================

Generates working ONNX neural blending models to replace corrupted ones.
The neural networks learn to blend PID1 (low-speed) and PID2 (high-speed) controllers
based on driving context features.

Neural Architecture:
- Input: 8 features (velocity, error, integrals, future plan, etc.)
- Hidden: 16 neurons with ReLU activation  
- Output: 1 blend weight (0.0 = 100% PID2, 1.0 = 100% PID1)

Initial Training Logic:
- Velocity-based: High weight for low speeds, low weight for high speeds
- Error-based: Higher PID1 weight for larger errors (needs aggressive response)
- Stability: Smooth transitions to avoid oscillations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.onnx
from pathlib import Path
import json
import time
import random

class BlenderNet(nn.Module):
    """Neural network for blending PID controllers"""
    
    def __init__(self, input_size=8, hidden_size=16, output_size=1):
        super(BlenderNet, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Ensures output is [0,1]
        
        # Initialize weights to approximate velocity-based blending
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to approximate fallback logic: 0.8 if v < 40 else 0.2"""
        
        # First layer: Focus on velocity (feature 0) and error magnitude (feature 3)
        with torch.no_grad():
            # Initialize first layer weights
            self.fc1.weight.data.normal_(0, 0.1)
            
            # Make velocity feature (index 0) more influential 
            self.fc1.weight.data[:, 0] = torch.randn(16) * 0.5 - 1.0  # Negative bias for high velocity
            
            # Make error feature (index 3) influence blending
            self.fc1.weight.data[:, 3] = torch.randn(16) * 0.3  # Error-based adjustments
            
            # Initialize biases
            self.fc1.bias.data.fill_(0.1)
            
            # Second layer: Map to output range
            self.fc2.weight.data.normal_(0, 0.5)
            self.fc2.bias.data.fill_(0.0)  # Will be adjusted by sigmoid
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Output range [0, 1]
        return x

def generate_training_data(num_samples=1000):
    """Generate synthetic training data based on velocity-based fallback logic"""
    
    features = []
    targets = []
    
    for _ in range(num_samples):
        # Generate realistic driving features
        v_ego = np.random.uniform(5, 80)  # 5-80 mph
        roll_lataccel = np.random.normal(0, 0.2)  # Small roll accelerations
        a_ego = np.random.normal(0, 1.0)  # Longitudinal acceleration
        error = np.random.normal(0, 2.0)  # Lateral acceleration error
        error_integral = np.random.normal(0, 0.5)  # PID integral
        error_derivative = np.random.normal(0, 1.0)  # Error derivative
        future_mean = np.random.normal(0, 1.0)  # Future plan mean
        future_std = np.random.uniform(0, 2.0)  # Future plan std
        
        # Create feature vector
        feature = np.array([
            v_ego, roll_lataccel, a_ego, error,
            error_integral, error_derivative, future_mean, future_std
        ], dtype=np.float32)
        
        # Target based on enhanced velocity logic with error consideration
        base_weight = 0.8 if v_ego < 40 else 0.2
        
        # Adjust based on error magnitude (larger errors need more PID1)
        error_adjustment = min(abs(error) * 0.1, 0.3)
        
        # Adjust based on future plan volatility
        future_adjustment = min(future_std * 0.05, 0.2)
        
        target_weight = np.clip(base_weight + error_adjustment + future_adjustment, 0.0, 1.0)
        
        features.append(feature)
        targets.append(target_weight)
    
    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)

def train_model(model, features, targets, epochs=200):
    """Train the neural blending model"""
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Convert to tensors
    X = torch.FloatTensor(features)
    y = torch.FloatTensor(targets).unsqueeze(1)
    
    print(f"🏋️ Training neural blending model...")
    print(f"   📊 Training samples: {len(features)}")
    print(f"   🎯 Target range: {targets.min():.3f} - {targets.max():.3f}")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"   📈 Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    # Final evaluation
    with torch.no_grad():
        final_outputs = model(X)
        final_loss = criterion(final_outputs, y)
        mae = torch.mean(torch.abs(final_outputs - y))
        
        print(f"   ✅ Training complete!")
        print(f"   📊 Final MSE: {final_loss.item():.6f}")
        print(f"   📊 Final MAE: {mae.item():.6f}")
    
    return model

def export_to_onnx(model, model_path, sample_input):
    """Export trained model to ONNX format"""
    
    model.eval()
    
    # Export to ONNX
    torch.onnx.export(
        model,
        sample_input,
        model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"   💾 Exported to: {model_path}")

def test_onnx_model(model_path):
    """Test the exported ONNX model"""
    
    try:
        import onnxruntime as ort
        
        # Load ONNX model
        session = ort.InferenceSession(str(model_path))
        
        # Test with sample inputs
        test_cases = [
            # [v_ego, roll_lataccel, a_ego, error, error_integral, error_derivative, future_mean, future_std]
            [25.0, 0.1, 0.0, 1.0, 0.2, 0.5, 0.5, 0.8],  # Low speed case
            [65.0, 0.1, 0.0, 1.0, 0.2, 0.5, 0.5, 0.8],  # High speed case
            [45.0, 0.1, 0.0, 3.0, 0.5, 1.0, 1.0, 1.5],  # High error case
        ]
        
        print(f"   🧪 Testing ONNX model...")
        for i, test_input in enumerate(test_cases):
            input_array = np.array([test_input], dtype=np.float32)
            output = session.run(None, {'input': input_array})[0][0]
            
            v_ego = test_input[0]
            error = test_input[3]
            expected = 0.8 if v_ego < 40 else 0.2
            
            weight_value = float(output)
            print(f"      🎯 Test {i+1}: v={v_ego:.0f}mph, error={error:.1f} -> weight={weight_value:.3f} (expected~{expected:.1f})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ONNX test failed: {e}")
        return False

def generate_all_models():
    """Generate all neural blending models"""
    
    print("🧠 Neural Blending Model Generation")
    print("=" * 50)
    
    start_time = time.time()
    
    # Ensure models directory exists
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Generate training data
    print("\n📊 Generating training data...")
    features, targets = generate_training_data(num_samples=2000)
    
    # List of existing corrupted models to replace
    existing_models = list(models_dir.glob("blender_*.onnx"))
    print(f"\n🔍 Found {len(existing_models)} existing models to replace")
    
    if not existing_models:
        # Generate some default model names if none exist
        model_names = [f"blender_{i:08x}.onnx" for i in range(5)]
        print(f"   📝 Creating {len(model_names)} new models")
    else:
        model_names = [model.name for model in existing_models]
        print(f"   📝 Replacing {len(model_names)} corrupted models")
    
    successful_models = 0
    failed_models = 0
    
    for i, model_name in enumerate(model_names):
        print(f"\n🏗️ Creating model {i+1}/{len(model_names)}: {model_name}")
        
        try:
            # Create and train model
            model = BlenderNet()
            
            # Add some variation to each model
            torch.manual_seed(random.randint(0, 10000))
            
            trained_model = train_model(model, features, targets, epochs=150)
            
            # Export to ONNX
            model_path = models_dir / model_name
            sample_input = torch.FloatTensor(features[:1])  # Single sample for export
            export_to_onnx(trained_model, model_path, sample_input)
            
            # Test the model
            if test_onnx_model(model_path):
                file_size = model_path.stat().st_size
                print(f"   ✅ Success! Model size: {file_size:,} bytes")
                successful_models += 1
            else:
                print(f"   ❌ Model failed testing")
                failed_models += 1
                
        except Exception as e:
            print(f"   ❌ Failed to create {model_name}: {e}")
            failed_models += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n🎯 Model Generation Complete!")
    print(f"   ✅ Successful models: {successful_models}")
    print(f"   ❌ Failed models: {failed_models}")
    print(f"   ⏱️  Total time: {elapsed:.1f}s")
    
    # Create summary report
    summary = {
        'timestamp': time.time(),
        'total_models': len(model_names),
        'successful_models': successful_models,
        'failed_models': failed_models,
        'training_samples': len(features),
        'model_architecture': '8->16->1 (ReLU + Sigmoid)',
        'training_epochs': 150,
        'notes': 'Neural blending models for Tournament #3 PID controller blending'
    }
    
    with open('neural_model_generation_report.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   📋 Report saved: neural_model_generation_report.json")
    
    return successful_models > 0

if __name__ == '__main__':
    generate_all_models()
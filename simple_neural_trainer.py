#!/usr/bin/env python3
"""
Simple Neural Blender Trainer - Single Model Approach
=====================================================

Replaces the complex 43-model system with 1 well-trained neural model.
Uses existing patterns from robust_training_data_gen.py and tournament archive.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import time
from pathlib import Path

class BlenderNet(nn.Module):
    """Simple neural network for blending PID controllers"""
    
    def __init__(self, input_size=8, hidden_size=16, output_size=1):
        super(BlenderNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Output range [0,1]
        )
        
        # Initialize weights for velocity-based behavior
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize to approximate velocity-based blending"""
        with torch.no_grad():
            # Focus first layer on velocity (feature 0)
            self.network[0].weight.data[0, 0] = -2.0  # Negative weight for velocity
            self.network[0].bias.data[0] = 1.6  # Bias to favor low speeds
            
            # Initialize other weights small
            self.network[0].weight.data[1:, :].normal_(0, 0.1)
            self.network[2].weight.data.normal_(0, 0.1)
            self.network[2].bias.data.fill_(0.0)
    
    def forward(self, x):
        return self.network(x)

def load_tournament_archive():
    """Load tournament archive using existing pattern"""
    archive_path = "plans/tournament_archive.json"
    try:
        with open(archive_path, 'r') as f:
            archive = json.load(f)
        
        # Filter valid performers (existing pattern)
        valid_performers = [p for p in archive['archive'] 
                           if 'stats' in p and 'avg_total_cost' in p['stats']]
        
        if not valid_performers:
            print("⚠️ No valid performers in archive")
            return []
        
        # Sort by performance (existing pattern)
        return sorted(valid_performers, key=lambda x: x['stats']['avg_total_cost'])
    
    except Exception as e:
        print(f"❌ Error loading tournament archive: {e}")
        return []

def generate_training_data(num_samples=5000):
    """Generate training data using existing pattern from robust_training_data_gen.py"""
    
    print("📊 Generating training data...")
    
    # Load tournament performers (reuse existing pattern)
    performers = load_tournament_archive()
    if not performers:
        print("⚠️ Using fallback training data generation")
        performers = [{'id': 'fallback'}]  # Fallback case
    
    top_performers = performers[:10]  # Top 10 like existing code
    
    training_samples = []
    samples_per_combo = num_samples // len(top_performers)
    
    for combo in top_performers:
        for _ in range(samples_per_combo):
            # Generate realistic vehicle state (existing pattern)
            v_ego = max(5, min(70, np.random.normal(35, 15)))
            roll_lataccel = np.random.uniform(-3, 3)
            a_ego = np.random.uniform(-2, 2)
            
            # Generate control error
            error = np.random.uniform(-2, 2)
            error_integral = np.random.uniform(-0.5, 0.5)
            error_derivative = np.random.uniform(-0.3, 0.3)
            
            # Generate future plan features
            future_mean = np.random.uniform(-2, 2)
            future_std = max(0, np.random.uniform(0, 1.5))
            
            # Create feature vector (same as existing BlenderNet)
            features = [v_ego, roll_lataccel, a_ego, error, 
                       error_integral, error_derivative, future_mean, future_std]
            
            # Simple velocity-based target (what system should learn)
            target_weight = 0.8 if v_ego < 40 else 0.2
            
            # Add some variation based on error magnitude
            if abs(error) > 1.0:
                target_weight += 0.1 if v_ego < 40 else -0.1
            
            # Clamp to valid range
            target_weight = max(0.0, min(1.0, target_weight))
            
            training_samples.append((features, target_weight))
    
    print(f"✅ Generated {len(training_samples)} training samples")
    
    # Validate distribution
    weights = [s[1] for s in training_samples]
    low_weight = sum(1 for w in weights if w > 0.6) / len(weights)
    high_weight = sum(1 for w in weights if w < 0.4) / len(weights)
    
    print(f"   📊 Low-speed bias (>0.6): {low_weight:.1%}")
    print(f"   📊 High-speed bias (<0.4): {high_weight:.1%}")
    
    return training_samples

def train_model(training_samples, epochs=100):
    """Train single neural model"""
    
    print(f"🧠 Training neural model for {epochs} epochs...")
    
    # Convert to PyTorch tensors
    features = torch.FloatTensor([s[0] for s in training_samples])
    targets = torch.FloatTensor([s[1] for s in training_samples]).reshape(-1, 1)
    
    # Create data loader
    dataset = TensorDataset(features, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = BlenderNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_features, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"   📈 Epoch {epoch:3d}: Loss = {avg_loss:.4f}")
    
    return model

def export_to_onnx(model, output_path="models/blender.onnx"):
    """Export to ONNX using existing pattern"""
    
    print(f"📦 Exporting to ONNX: {output_path}")
    
    # Ensure models directory exists
    Path(output_path).parent.mkdir(exist_ok=True)
    
    model.eval()
    
    # Create dummy input for tracing
    dummy_input = torch.randn(1, 8)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
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
    
    # Validate ONNX model
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        
        # Test cases
        test_cases = [
            [25.0, 0.1, 0.0, 1.0, 0.2, 0.1, 0.5, 0.8],  # Low speed -> high weight
            [65.0, 0.1, 0.0, 1.0, 0.2, 0.1, 0.5, 0.8],  # High speed -> low weight
            [40.0, 0.1, 0.0, 2.0, 0.5, 0.2, 1.0, 1.2],  # Boundary case
        ]
        
        print("   🧪 Testing ONNX model:")
        for i, test_input in enumerate(test_cases):
            input_array = np.array([test_input], dtype=np.float32)
            output = session.run(None, {'input': input_array})[0][0][0]
            
            v_ego = test_input[0]
            expected = 0.8 if v_ego < 40 else 0.2
            
            print(f"      🎯 Test {i+1}: v={v_ego:.0f}mph -> weight={output:.3f} (expect~{expected:.1f})")
        
        file_size = Path(output_path).stat().st_size
        print(f"   ✅ ONNX model exported: {file_size:,} bytes")
        return True
        
    except Exception as e:
        print(f"   ❌ ONNX validation failed: {e}")
        return False

def clean_old_models():
    """Remove old fake models"""
    
    models_dir = Path("models")
    if not models_dir.exists():
        return
    
    old_models = list(models_dir.glob("blender_*.onnx"))
    if old_models:
        print(f"🧹 Removing {len(old_models)} old fake models...")
        for model_path in old_models:
            model_path.unlink()
        print("   ✅ Old models removed")

def main():
    """Main training function"""
    
    print("🚀 Simple Neural Blender Trainer")
    print("=" * 50)
    
    start_time = time.time()
    
    # Clean old models
    clean_old_models()
    
    # Generate training data
    training_samples = generate_training_data()
    
    if not training_samples:
        print("❌ No training data generated")
        return
    
    # Train model
    model = train_model(training_samples, epochs=150)
    
    # Export to ONNX
    success = export_to_onnx(model)
    
    elapsed = time.time() - start_time
    
    if success:
        print(f"\n🎉 Neural model training completed in {elapsed:.1f}s")
        print("✅ Single working neural model ready!")
        print("📄 Model saved as: models/blender.onnx")
    else:
        print(f"\n❌ Training failed after {elapsed:.1f}s")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
from pathlib import Path

class BlenderNet(nn.Module):
    """
    Lightweight neural network for learning optimal PID blending weights.
    Input: 8 features (vehicle state + error dynamics + future plan)
    Output: Single blend weight [0,1] for PID1 vs PID2
    """
    def __init__(self, input_size=8, hidden_sizes=[32, 16]):
        super(BlenderNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.extend([
            nn.Linear(prev_size, 1),
            nn.Sigmoid()  # Blend weight must be in [0,1]
        ])
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)
    
    def extract_features(self, state, error, error_integral, error_derivative, future_plan):
        """
        Extract 8-dimensional feature vector from control state
        
        Features:
        [v_ego, roll_lataccel, a_ego, error, error_integral, error_derivative, 
         future_lataccel_mean, future_lataccel_std]
        """
        features = np.array([
            state.v_ego,
            state.roll_lataccel,
            state.a_ego, 
            error,
            error_integral,
            error_derivative,
            np.mean(future_plan.lataccel) if len(future_plan.lataccel) > 0 else 0.0,
            np.std(future_plan.lataccel) if len(future_plan.lataccel) > 0 else 0.0
        ], dtype=np.float32)
        
        return torch.tensor(features).unsqueeze(0)
    
    def export_to_onnx(self, onnx_path, input_size=8):
        """Export trained model to ONNX for production inference"""
        self.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, input_size)
        
        # Export to ONNX
        torch.onnx.export(
            self,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['blend_weight'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'blend_weight': {0: 'batch_size'}
            }
        )
        
        print(f"BlenderNet exported to ONNX: {onnx_path}")
        return onnx_path

def create_random_blender_net():
    """Create a BlenderNet with random architecture for tournament evolution"""
    
    # Random architecture parameters for evolution
    hidden_sizes = []
    num_layers = np.random.randint(2, 4)  # 2-3 hidden layers
    
    for _ in range(num_layers):
        layer_size = np.random.choice([16, 24, 32, 48, 64])
        hidden_sizes.append(layer_size)
    
    return BlenderNet(input_size=8, hidden_sizes=hidden_sizes)

def mutate_blender_net(parent_net, mutation_rate=0.1):
    """Create mutated version of BlenderNet for tournament evolution"""
    
    # Create new net with similar architecture but random initialization
    child_net = BlenderNet(
        input_size=8, 
        hidden_sizes=[32, 16]  # Keep architecture simple for now
    )
    
    # Copy parent weights with mutation
    with torch.no_grad():
        for child_param, parent_param in zip(child_net.parameters(), parent_net.parameters()):
            if np.random.random() < mutation_rate:
                # Add Gaussian noise to weights
                noise = torch.randn_like(parent_param) * 0.1
                child_param.copy_(parent_param + noise)
            else:
                child_param.copy_(parent_param)
    
    return child_net

def train_blender_net(training_data, epochs=100, lr=0.001):
    """
    Train BlenderNet using supervised learning on optimal blending data
    
    Args:
        training_data: List of (features, optimal_blend_weight) tuples
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Trained BlenderNet model
    """
    
    # Create model
    model = BlenderNet()
    
    # Loss and optimizer  
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Convert training data to tensors
    X = torch.stack([torch.tensor(features, dtype=torch.float32) for features, _ in training_data])
    y = torch.tensor([label for _, label in training_data], dtype=torch.float32).unsqueeze(1)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        outputs = model(X)
        loss = criterion(outputs, y)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

if __name__ == "__main__":
    # Test BlenderNet creation and export
    net = create_random_blender_net()
    print(f"Created BlenderNet with architecture: {net}")
    
    # Test feature extraction
    from collections import namedtuple
    State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
    FuturePlan = namedtuple('FuturePlan', ['lataccel'])
    
    test_state = State(roll_lataccel=0.5, v_ego=25.0, a_ego=0.1)
    test_future = FuturePlan(lataccel=[0.2, 0.3, 0.1, -0.1])
    
    features = net.extract_features(test_state, 0.1, 0.05, 0.02, test_future)
    print(f"Extracted features shape: {features.shape}")
    
    # Test forward pass
    blend_weight = net(features)
    print(f"Predicted blend weight: {blend_weight.item():.3f}")
    
    # Test ONNX export
    onnx_path = "test_blender.onnx"
    net.export_to_onnx(onnx_path)
    print(f"ONNX export successful: {Path(onnx_path).exists()}")
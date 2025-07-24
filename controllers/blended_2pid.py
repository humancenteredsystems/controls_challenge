from . import BaseController
from .shared_pid import SpecializedPID
import json
import os
from pathlib import Path

class Controller(BaseController):
    """
    Blended 2-PID controller using velocity-based blending of specialized PIDs
    """
    def __init__(self):
        # Load parameters from json file
        params_file = Path(__file__).parent.parent / "blended_2pid_params.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                params = json.load(f)
            
            low_gains = params['low_gains']
            high_gains = params['high_gains']
            
            print(f"Loaded blended 2-PID parameters (cost: {params.get('best_cost', 'N/A'):.2f})")
        else:
            # Fallback to default parameters
            low_gains = [0.3, 0.03, -0.1]
            high_gains = [0.2, 0.01, -0.05]
            print("Using fallback blended 2-PID parameters")
        
        # Initialize specialized PID controllers
        self.low_speed_pid = SpecializedPID(low_gains[0], low_gains[1], low_gains[2], "LowSpeed")
        self.high_speed_pid = SpecializedPID(high_gains[0], high_gains[1], high_gains[2], "HighSpeed")
        
        print(f"Initialized blended 2-PID controller:")
        print(f"  {self.low_speed_pid}")
        print(f"  {self.high_speed_pid}")
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        v_ego = state.v_ego
        
        # Get outputs from both controllers
        low_output = self.low_speed_pid.update(error)
        high_output = self.high_speed_pid.update(error)
        
        # Simple velocity-based blending logic
        if v_ego < 40:  # Low speed: use 80% low + 20% high
            weights = [0.8, 0.2]
        else:  # High speed: use 20% low + 80% high
            weights = [0.2, 0.8]
        
        # Blend outputs
        blended_output = (weights[0] * low_output + 
                         weights[1] * high_output)
        
        return blended_output
    
    def __repr__(self):
        return f"Blended2PIDController"
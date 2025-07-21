from . import BaseController
import json
import os
from pathlib import Path

class SpecializedPID:
    """A PID controller with configurable gains"""
    def __init__(self, p, i, d, name=""):
        self.p = p
        self.i = i
        self.d = d
        self.name = name
        self.error_integral = 0
        self.prev_error = 0
    
    def update(self, error):
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff
    
    def reset(self):
        self.error_integral = 0
        self.prev_error = 0

class Controller(BaseController):
    """
    Optimized ensemble controller using rule-based blending of specialized PIDs
    """
    def __init__(self):
        # Load optimized parameters from json file
        params_file = Path(__file__).parent.parent / "optimal_params.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                params = json.load(f)
            
            low_gains = params['low_speed_gains']
            high_gains = params['high_speed_gains']
            dynamic_gains = params['dynamic_gains']
            
            print(f"Loaded optimized ensemble parameters (cost: {params['best_cost']:.2f})")
        else:
            # Fallback to default optimized parameters
            low_gains = [0.3, 0.03, -0.1]
            high_gains = [0.2, 0.01, -0.05]
            dynamic_gains = [0.4, 0.1, -0.1]
            print("Using fallback optimized parameters")
        
        # Initialize specialized PID controllers with optimized gains
        self.low_speed_pid = SpecializedPID(low_gains[0], low_gains[1], low_gains[2], "LowSpeed")
        self.high_speed_pid = SpecializedPID(high_gains[0], high_gains[1], high_gains[2], "HighSpeed") 
        self.dynamic_pid = SpecializedPID(dynamic_gains[0], dynamic_gains[1], dynamic_gains[2], "Dynamic")
        
        print(f"Initialized ensemble controller:")
        print(f"  {self.low_speed_pid}")
        print(f"  {self.high_speed_pid}")
        print(f"  {self.dynamic_pid}")
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        v_ego = state.v_ego
        
        # Get outputs from all controllers
        low_output = self.low_speed_pid.update(error)
        high_output = self.high_speed_pid.update(error)
        dynamic_output = self.dynamic_pid.update(error)
        
        # Rule-based blending based on velocity and scenario
        if v_ego < 25:  # Low speed / parking / city
            weights = [0.7, 0.2, 0.1]
        elif v_ego > 55:  # Highway speeds
            weights = [0.1, 0.7, 0.2]
        elif abs(target_lataccel) > 0.7:  # Sharp maneuvers
            weights = [0.2, 0.2, 0.6]
        else:  # Mixed conditions
            weights = [0.4, 0.4, 0.2]
        
        # Blend outputs
        blended_output = (weights[0] * low_output + 
                         weights[1] * high_output + 
                         weights[2] * dynamic_output)
        
        return blended_output
    
    def __repr__(self):
        return f"OptimizedEnsembleController"

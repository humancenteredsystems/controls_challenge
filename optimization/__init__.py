"""
Shared utilities for optimization modules
"""
from typing import List

def generate_blended_controller(low_gains: List[float], high_gains: List[float]) -> str:
    """
    Generate blended 2-PID controller code using proven template from blended_2pid_optimizer.
    
    This uses the exact same controller architecture that achieved 76.81 cost,
    ensuring consistency across all optimizers.
    
    Args:
        low_gains: [P, I, D] gains for low-speed PID
        high_gains: [P, I, D] gains for high-speed PID
        
    Returns:
        Complete controller code as string
    """
    return f'''from controllers import BaseController

class SpecializedPID:
    def __init__(self, p, i, d):
        self.p, self.i, self.d = p, i, d
        self.error_integral = 0
        self.prev_error = 0
    
    def update(self, error):
        dt = 0.1  # Match tinyphysics DEL_T = 0.1 (10 Hz) - CRITICAL TIME STEP FIX
        self.error_integral += error * dt
        error_diff = (error - self.prev_error) / dt
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff

class Controller(BaseController):
    def __init__(self):
        self.low_speed_pid = SpecializedPID({low_gains[0]}, {low_gains[1]}, {low_gains[2]})
        self.high_speed_pid = SpecializedPID({high_gains[0]}, {high_gains[1]}, {high_gains[2]})
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        v_ego = state.v_ego
        
        # Get outputs from both controllers
        low_output = self.low_speed_pid.update(error)
        high_output = self.high_speed_pid.update(error)
        
        # Simple velocity-based blending logic: v_ego < 40 for 80%/20% vs 20%/80%
        if v_ego < 40:  # Low speed: use 80% low + 20% high
            weights = [0.8, 0.2]
        else:  # High speed: use 20% low + 80% high
            weights = [0.2, 0.8]
        
        # Blend outputs (PROPER blending, not summation!)
        blended_output = (weights[0] * low_output + 
                         weights[1] * high_output)
        
        return blended_output
'''

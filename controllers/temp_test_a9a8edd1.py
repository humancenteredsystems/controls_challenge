from . import BaseController

# Import shared PID implementation for consistency
from controllers.shared_pid import SpecializedPID

class Controller(BaseController):
    def __init__(self):
        self.low_speed_pid = SpecializedPID(0.3, 0.03, -0.1)
        self.high_speed_pid = SpecializedPID(0.3, 0.015, -0.06)
        
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
        
        # Blend outputs
        blended_output = (weights[0] * low_output + 
                         weights[1] * high_output)
        
        return blended_output

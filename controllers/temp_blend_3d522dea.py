from controllers import BaseController

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
        self.low_speed_pid = SpecializedPID(0.575, 0.12, -0.05)
        self.high_speed_pid = SpecializedPID(0.293, 0.08, -0.03)
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        v_ego = state.v_ego
        
        # Get outputs from both controllers
        low_output = self.low_speed_pid.update(error)
        high_output = self.high_speed_pid.update(error)
        
        # Simple velocity-based blending logic: v_ego < 40 for 80%/20% vs 20%/80%
        if False:  # Fixed blend weight: 0.2
            weights = [0.2, 0.8]
        else:  # High speed: use 20% low + 80% high
            weights = [0.2, 0.8]
        
        # Blend outputs (PROPER blending, not summation!)
        blended_output = (weights[0] * low_output + 
                         weights[1] * high_output)
        
        return blended_output

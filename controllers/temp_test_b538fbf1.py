from . import BaseController

class SpecializedPID:
    def __init__(self, p, i, d):
        self.p, self.i, self.d = p, i, d
        self.error_integral = 0
        self.prev_error = 0
    
    def update(self, error):
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff

class Controller(BaseController):
    def __init__(self):
        self.low_speed_pid = SpecializedPID(0.283, 0.01, -0.105)
        self.high_speed_pid = SpecializedPID(0.205, 0.007, -0.07)
        self.dynamic_pid = SpecializedPID(0.417, 0.113, -0.155)
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        v_ego = state.v_ego
        
        low_output = self.low_speed_pid.update(error)
        high_output = self.high_speed_pid.update(error)
        dynamic_output = self.dynamic_pid.update(error)
        
        # Enhanced rule-based blending
        if v_ego < 20:  # Very low speed
            weights = [0.8, 0.1, 0.1]
        elif v_ego < 40:  # Low speed
            weights = [0.6, 0.3, 0.1]
        elif v_ego > 70:  # High speed
            weights = [0.05, 0.8, 0.15]
        elif v_ego > 50:  # Medium-high speed
            weights = [0.2, 0.6, 0.2]
        else:  # Medium speed
            weights = [0.4, 0.4, 0.2]
            
        # Adjust for dynamic scenarios
        if abs(target_lataccel) > 1.0:  # Very sharp turns
            weights = [0.1, 0.2, 0.7]
        elif abs(target_lataccel) > 0.5:  # Moderate turns
            weights[2] = min(weights[2] + 0.2, 0.6)
            weights[0] = max(weights[0] - 0.1, 0.1)
            weights[1] = 1.0 - weights[0] - weights[2]
        
        return sum(w * o for w, o in zip(weights, [low_output, high_output, dynamic_output]))

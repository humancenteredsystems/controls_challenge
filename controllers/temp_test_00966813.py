from controllers import BaseController
from controllers.shared_pid import SpecializedPID
from controllers.blending import get_smooth_blend_weight

class Controller(BaseController):
    def __init__(self):
        self.low_speed_pid = SpecializedPID(0.43, 0.096, -0.171)
        self.high_speed_pid = SpecializedPID(0.306, 0.07, -0.036)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        v_ego = state.v_ego

        # Get outputs from both controllers
        low_output = self.low_speed_pid.update(error)
        high_output = self.high_speed_pid.update(error)

        # Calculate smooth blend weights
        blend_weight = get_smooth_blend_weight(v_ego)
        low_weight = 1.0 - blend_weight
        high_weight = blend_weight

        # Blend outputs using smooth weights
        blended_output = (low_weight * low_output + high_weight * high_output)

        return blended_output

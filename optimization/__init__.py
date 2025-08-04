"""
Shared utilities for optimization modules
"""
from typing import List
import numpy as np

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
    # This function now correctly uses the shared SpecializedPID from controllers.shared_pid
    # by virtue of the template string below. The local definition has been removed.
    return f'''from controllers import BaseController
from controllers.shared_pid import SpecializedPID
from utils.blending import get_smooth_blend_weight

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

        # Calculate smooth blend weights
        blend_weight = get_smooth_blend_weight(v_ego)
        low_weight = 1.0 - blend_weight
        high_weight = blend_weight

        # Blend outputs using smooth weights
        blended_output = (low_weight * low_output + high_weight * high_output)

        return blended_output
'''

def generate_neural_blended_controller(pid1_params, pid2_params, onnx_model_path, norm_stats=None):
    """
    Generate neural blended controller code with embedded normalization stats.
    """
    if norm_stats:
        mean_str = np.array2string(np.array(norm_stats['mean']), separator=', ', floatmode='maxprec')
        std_str = np.array2string(np.array(norm_stats['std']), separator=', ', floatmode='maxprec')
        normalization_block = f"""
       self.feature_mean = np.array({mean_str}, dtype=np.float32)
       self.feature_std = np.array({std_str}, dtype=np.float32)
       self.feature_std[self.feature_std == 0] = 1.0  # Avoid division by zero
"""
        feature_processing_block = """
       # Normalize features
       features = (features - self.feature_mean) / self.feature_std
"""
    else:
        normalization_block = ""
        feature_processing_block = ""

    return f'''from controllers import BaseController
from controllers.shared_pid import SpecializedPID
import onnxruntime as ort
import numpy as np

class Controller(BaseController):
   def __init__(self):
       self.pid1 = SpecializedPID({pid1_params[0]}, {pid1_params[1]}, {pid1_params[2]})
       self.pid2 = SpecializedPID({pid2_params[0]}, {pid2_params[1]}, {pid2_params[2]})
       
       self.blender_session = ort.InferenceSession("{onnx_model_path}",
           providers=['CPUExecutionProvider'])
{normalization_block}
   def update(self, target_lataccel, current_lataccel, state, future_plan):
       error = target_lataccel - current_lataccel
       # Update PID controllers and get their outputs
       pid1_output = self.pid1.update(error)
       pid2_output = self.pid2.update(error)
       
       # Extract features for the neural blender
       features = np.array([[
           state.v_ego,
           state.roll_lataccel,
           state.a_ego,
           error,
           self.pid1.error_integral,
           self.pid1.error_derivative, # Use the derivative from the PID controller
           np.mean(future_plan.lataccel) if len(future_plan.lataccel) > 0 else 0.0,
           np.std(future_plan.lataccel) if len(future_plan.lataccel) > 0 else 0.0
       ]], dtype=np.float32)
{feature_processing_block}
       # Run inference
       blend_weight = self.blender_session.run(None, {{'input': features}})[0][0]
       blend_weight = np.clip(float(blend_weight), 0.0, 1.0)
       
       # Blend the PID outputs
       return blend_weight * pid1_output + (1 - blend_weight) * pid2_output
'''

from controllers import BaseController
from controllers.shared_pid import SpecializedPID
import onnxruntime as ort
import numpy as np

class Controller(BaseController):
    """Champion Neural Blended Controller - Stage 2d Winner (Cost: 491.66)"""
    
    def __init__(self):
        # Best PID parameters from Tournament #2
        self.pid1 = SpecializedPID(0.34, 0.027, -0.074, "Champion_PID1")
        self.pid2 = SpecializedPID(0.291, 0.006, -0.057, "Champion_PID2")
        
        # Champion trained neural blender
        self.blender_session = ort.InferenceSession(
            "models/neural_blender_champion.onnx",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        print(f"🏆 Champion Neural Blended Controller loaded (cost: 491.66)")
        print(f"  PID1: P=0.340, I=0.027, D=-0.074")
        print(f"  PID2: P=0.291, I=0.006, D=-0.057")
        print(f"  Neural Architecture: [32, 32]")
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        pid1_output = self.pid1.update(error)
        pid2_output = self.pid2.update(error)
        
        # Neural blending using champion model
        features = np.array([[state.v_ego, state.roll_lataccel, state.a_ego, error,
                             self.pid1.error_integral, error - self.pid1.prev_error,
                             np.mean(future_plan.lataccel) if len(future_plan.lataccel) > 0 else 0.0,
                             np.std(future_plan.lataccel) if len(future_plan.lataccel) > 0 else 0.0]], dtype=np.float32)
        
        blend_weight = self.blender_session.run(None, {'input': features})[0][0]
        blend_weight = np.clip(float(blend_weight), 0.0, 1.0)
        
        return blend_weight * pid1_output + (1 - blend_weight) * pid2_output
    
    def __repr__(self):
        return f"ChampionNeuralBlendedController(cost=491.66)"

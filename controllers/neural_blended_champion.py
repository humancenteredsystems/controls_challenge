from controllers import BaseController
from controllers.shared_pid import SpecializedPID
import onnxruntime as ort
import numpy as np

class Controller(BaseController):
   def __init__(self):
       self.pid1 = SpecializedPID(0.554, 0.036, -0.063)
       self.pid2 = SpecializedPID(0.17, 0.036, -0.063)
       
       self.blender_session = ort.InferenceSession("models/neural_blender_champion.onnx",
           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
   
   def update(self, target_lataccel, current_lataccel, state, future_plan):
       error = target_lataccel - current_lataccel
       pid1_output = self.pid1.update(error)
       pid2_output = self.pid2.update(error)
       
       features = np.array([[state.v_ego, state.roll_lataccel, state.a_ego, error,
                            self.pid1.error_integral, error - self.pid1.prev_error,
                            np.mean(future_plan.lataccel) if len(future_plan.lataccel) > 0 else 0.0,
                            np.std(future_plan.lataccel) if len(future_plan.lataccel) > 0 else 0.0]], dtype=np.float32)
       
       blend_weight = self.blender_session.run(None, {'input': features})[0][0]
       blend_weight = np.clip(float(blend_weight), 0.0, 1.0)
       return blend_weight * pid1_output + (1 - blend_weight) * pid2_output

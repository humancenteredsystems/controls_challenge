from controllers import BaseController
from controllers.shared_pid import SpecializedPID
import onnxruntime as ort
import numpy as np

class Controller(BaseController):
   def __init__(self):
       self.pid1 = SpecializedPID(0.488, 0.07, -0.05)
       self.pid2 = SpecializedPID(0.366, 0.08, -0.03)
       
       self.blender_session = ort.InferenceSession("models/blender_dc936294.onnx",
           providers=['CPUExecutionProvider'])

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

       # Run inference
       blend_weight = self.blender_session.run(None, {'input': features})[0][0]
       blend_weight = np.clip(float(blend_weight), 0.0, 1.0)
       
       # Blend the PID outputs
       return blend_weight * pid1_output + (1 - blend_weight) * pid2_output

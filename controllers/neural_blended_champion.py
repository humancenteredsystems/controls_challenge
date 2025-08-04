from controllers import BaseController
from controllers.shared_pid import SpecializedPID
import onnxruntime as ort
import numpy as np

class Controller(BaseController):
   def __init__(self):
       self.pid1 = SpecializedPID(0.3, 0.03, -0.1)
       self.pid2 = SpecializedPID(0.2, 0.01, -0.05)
       
       self.blender_session = ort.InferenceSession("models/neural_blender_champion.onnx",
           providers=['CPUExecutionProvider'])

       self.feature_mean = np.array([ 1.99516945e+01, -1.13237239e-01,  1.03601195e-01, -1.47993620e-02,
  7.44884536e-02, -2.18505897e-02,  6.71600401e-02,  4.12871450e-01], dtype=np.float32)
       self.feature_std = np.array([12.26183605,  1.74829924,  1.17043579,  0.3279759 ,  0.21340278,
  0.11656978,  0.61870861,  0.41793579], dtype=np.float32)
       self.feature_std[self.feature_std == 0] = 1.0  # Avoid division by zero

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

       # Normalize features
       features = (features - self.feature_mean) / self.feature_std

       # Run inference
       blend_weight = self.blender_session.run(None, {'input': features})[0][0]
       blend_weight = np.clip(float(blend_weight), 0.0, 1.0)
       
       # Blend the PID outputs
       return blend_weight * pid1_output + (1 - blend_weight) * pid2_output

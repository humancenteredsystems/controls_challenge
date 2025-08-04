from controllers import BaseController
from controllers.shared_pid import SpecializedPID
import onnxruntime as ort
import numpy as np

class Controller(BaseController):
   def __init__(self):
       self.pid1 = SpecializedPID(0.599, 0.093, -0.128)
       self.pid2 = SpecializedPID(0.25, 0.061, -0.041)
       
       self.blender_session = ort.InferenceSession("models/neural_blender_champion.onnx",
           providers=['CPUExecutionProvider'])

       self.feature_mean = np.array([ 2.11325378e+01,  3.16461176e-01, -1.37785133e-02,  8.29297453e-02,
  3.48650701e-02, -9.96782910e-03,  2.25655913e-01,  4.12314415e-01], dtype=np.float32)
       self.feature_std = np.array([18.83441544,  1.71287072,  0.67919713,  0.52955788,  0.16270544,
  0.09340752,  0.99589872,  0.43653294], dtype=np.float32)
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

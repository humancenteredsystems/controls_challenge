from controllers import BaseController
from controllers.shared_pid import SpecializedPID
import onnxruntime as ort
import numpy as np

class Controller(BaseController):
   def __init__(self):
       self.pid1 = SpecializedPID(0.25, 0.01, -0.05)
       self.pid2 = SpecializedPID(0.15, 0.08, -0.03)
       
       self.blender_session = ort.InferenceSession("models/neural_blender_champion.onnx",
           providers=['CPUExecutionProvider'])

       self.feature_mean = np.array([ 2.09838181e+01,  1.72638223e-02, -2.48620026e-02, -5.05054332e-02,
  9.47123207e-03, -5.10698697e-03, -5.96792661e-02,  5.04327059e-01], dtype=np.float32)
       self.feature_std = np.array([15.98599911,  1.45740628,  1.04866278,  0.41237071,  0.20246927,
  0.09432463,  0.93433791,  0.46274137], dtype=np.float32)
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

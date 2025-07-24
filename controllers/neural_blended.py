from . import BaseController
from .shared_pid import SpecializedPID
import onnxruntime as ort
import numpy as np
import json
import os
from pathlib import Path

class Controller(BaseController):
    """
    Neural Blended Controller using learned blending weights between two PID controllers
    """
    def __init__(self, pid1_params=None, pid2_params=None, blender_model_path=None):
        
        # Load PID parameters
        if pid1_params is None or pid2_params is None:
            # Load from tournament archive if not provided
            pid1_params, pid2_params = self._load_best_pid_params()
        
        # Initialize PID controllers
        self.pid1 = SpecializedPID(pid1_params[0], pid1_params[1], pid1_params[2], "PID1")
        self.pid2 = SpecializedPID(pid2_params[0], pid2_params[1], pid2_params[2], "PID2")
        
        # Load BlenderNet ONNX model
        if blender_model_path is None:
            blender_model_path = self._find_blender_model()
        
        if blender_model_path and Path(blender_model_path).exists():
            # GPU-first session creation (follows tinyphysics.py pattern)
            session_options = ort.SessionOptions()
            session_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self.blender_session = ort.InferenceSession(
                blender_model_path, 
                sess_options=session_options,
                providers=session_providers
            )
            
            print(f"Neural blended controller initialized with GPU acceleration")
            print(f"  PID1: P={self.pid1.p:.3f}, I={self.pid1.i:.3f}, D={self.pid1.d:.3f}")
            print(f"  PID2: P={self.pid2.p:.3f}, I={self.pid2.i:.3f}, D={self.pid2.d:.3f}")
            print(f"  BlenderNet: {blender_model_path}")
        else:
            # Fallback to simple blending if no neural model
            self.blender_session = None
            print("Warning: No BlenderNet model found, using velocity-based fallback")
    
    def _load_best_pid_params(self):
        """Load best PID parameters from tournament archive"""
        archive_path = Path(__file__).parent.parent / "plans" / "tournament_archive.json"
        
        if archive_path.exists():
            with open(archive_path, 'r') as f:
                archive = json.load(f)
            
            # Get best performer from archive
            best_combo = min(archive['archive'], key=lambda x: x['stats']['avg_total_cost'])
            
            pid1_params = best_combo['low_gains']
            pid2_params = best_combo['high_gains']
            
            print(f"Loaded best PID params from archive (cost: {best_combo['stats']['avg_total_cost']:.2f})")
            
        else:
            # Fallback parameters
            pid1_params = [0.25, 0.12, -0.05]  # Low-speed optimized
            pid2_params = [0.15, 0.08, -0.15]  # High-speed optimized
            print("Using fallback PID parameters")
        
        return pid1_params, pid2_params
    
    def _find_blender_model(self):
        """Find the best BlenderNet model"""
        models_dir = Path(__file__).parent.parent / "models"
        
        # Look for blender models
        blender_models = list(models_dir.glob("blender_*.onnx"))
        
        if blender_models:
            # Return the most recent one (or implement better selection logic)
            return str(max(blender_models, key=lambda p: p.stat().st_mtime))
        
        return None
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        
        # Get PID outputs
        pid1_output = self.pid1.update(error)
        pid2_output = self.pid2.update(error)
        
        if self.blender_session is not None:
            # Neural blending
            blend_weight = self._get_neural_blend_weight(state, error, future_plan)
        else:
            # Fallback to velocity-based blending
            blend_weight = 0.8 if state.v_ego < 40 else 0.2
        
        # Blend PID outputs
        blended_output = blend_weight * pid1_output + (1 - blend_weight) * pid2_output
        
        return blended_output
    
    def _get_neural_blend_weight(self, state, error, future_plan):
        """Get blending weight from neural network"""
        
        # Extract features (same as BlenderNet.extract_features)
        features = np.array([
            state.v_ego,
            state.roll_lataccel,
            state.a_ego,
            error,
            self.pid1.error_integral,  # Use PID1's integral for consistency
            error - self.pid1.prev_error,  # Error derivative
            np.mean(future_plan.lataccel) if len(future_plan.lataccel) > 0 else 0.0,
            np.std(future_plan.lataccel) if len(future_plan.lataccel) > 0 else 0.0
        ], dtype=np.float32).reshape(1, -1)
        
        # Get blend weight from neural network
        try:
            blend_weight = self.blender_session.run(None, {'input': features})[0][0]
            
            # Ensure blend weight is in valid range
            blend_weight = np.clip(blend_weight, 0.0, 1.0)
            
            return float(blend_weight)
            
        except Exception as e:
            print(f"Neural blending failed: {e}, using fallback")
            return 0.8 if state.v_ego < 40 else 0.2
    
    def __repr__(self):
        if self.blender_session is not None:
            return f"NeuralBlendedController(neural_blending=True)"
        else:
            return f"NeuralBlendedController(neural_blending=False, fallback=velocity)"

# For temporary controller creation during tournament evaluation
def create_temp_neural_controller(pid1_params, pid2_params, blender_onnx_path, temp_id):
    """Create temporary neural blended controller file for evaluation"""
    
    controller_content = f'''from controllers.neural_blended import Controller as BaseNeuralController
import numpy as np

# Temporary controller for tournament evaluation
class Controller(BaseNeuralController):
    def __init__(self):
        pid1_params = {pid1_params}
        pid2_params = {pid2_params}
        blender_model_path = "{blender_onnx_path}"
        
        super().__init__(pid1_params, pid2_params, blender_model_path)
'''
    
    temp_path = Path(__file__).parent / f"temp_neural_{temp_id}.py"
    
    with open(temp_path, 'w') as f:
        f.write(controller_content)
    
    return str(temp_path)
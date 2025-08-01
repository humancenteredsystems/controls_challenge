"""
Tournament #3 Neural Blended Controller

Clean, simple controller specifically for Tournament #3 that:
- Uses single blender.onnx model (no complex selection logic)
- Inherits from neural_blended base functionality
- Removes fallback complexity for Tournament #3 use case
- Preserves Tournament #1/#2 architecture intact
"""

from pathlib import Path
from .neural_blended import Controller as BaseController


class Tournament3Controller(BaseController):
    """Tournament #3 specific neural blended controller"""
    
    def _find_blender_model(self):
        """Tournament #3: Use single blender.onnx model"""
        models_dir = Path(__file__).parent.parent / "models"
        blender_path = models_dir / "blender.onnx"
        
        if blender_path.exists() and blender_path.stat().st_size > 1000:
            return str(blender_path)
        
        return None
    
    def __str__(self):
        neural_status = "neural" if self.blender_session is not None else "fallback"
        return f"Tournament3Controller({neural_status})"


# Alias for compatibility
Controller = Tournament3Controller
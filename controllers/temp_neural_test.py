from controllers.neural_blended import Controller as BaseNeuralController

class Controller(BaseNeuralController):
    def __init__(self):
        # Use Tournament #2 winner parameters
        pid1_params = [0.291, 0.12, -0.082]
        pid2_params = [0.15, 0.06, -0.038]
        
        # No neural model initially - will use velocity-based fallback
        super().__init__(pid1_params, pid2_params, blender_model_path=None)
        
        print("Test Neural Blended Controller initialized (using velocity fallback)")

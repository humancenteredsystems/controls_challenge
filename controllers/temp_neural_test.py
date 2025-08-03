from controllers.neural_blended import Controller as BaseNeuralController

class Controller(BaseNeuralController):
    def __init__(self):
        # Use Tournament #2 winner parameters
        pid1_params = [0.265, 0.044, -0.187]
        pid2_params = [0.235, 0.026, -0.059]
        
        # No neural model initially - will use velocity-based fallback
        super().__init__(pid1_params, pid2_params, blender_model_path=None)
        
        print("Test Neural Blended Controller initialized (using velocity fallback)")

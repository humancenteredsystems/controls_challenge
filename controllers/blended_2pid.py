from . import BaseController
from .shared_pid import SpecializedPID
from .blending import get_smooth_blend_weight
import json
from pathlib import Path

class Controller(BaseController):
    """
    Blended 2-PID controller using velocity-based blending of specialized PIDs
    """
    def __init__(self):
        # Load parameters from json file
        params_file = Path(__file__).parent.parent / "blended_2pid_params.json"
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)

            low_gains = params['low_gains']
            high_gains = params['high_gains']

            cost = params.get('best_cost', float('nan'))
            print(f"Loaded blended 2-PID parameters (cost: {cost:.2f})")
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
            raise FileNotFoundError(
                f"Could not load blended_2pid parameters from {params_file}. "
                "Ensure Stage 1 ran and produced a valid params file."
            ) from e
        
        # Initialize specialized PID controllers
        self.low_speed_pid = SpecializedPID(low_gains[0], low_gains[1], low_gains[2], "LowSpeed")
        self.high_speed_pid = SpecializedPID(high_gains[0], high_gains[1], high_gains[2], "HighSpeed")
        
        print(f"Initialized blended 2-PID controller:")
        print(f"  {self.low_speed_pid}")
        print(f"  {self.high_speed_pid}")
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        v_ego = state.v_ego
        
        # Get outputs from both controllers
        low_output = self.low_speed_pid.update(error)
        high_output = self.high_speed_pid.update(error)
        

        # Smooth velocity-based blending with sigmoid transition
        # Threshold: 15 m/s (~33 mph), transition zone: ±2.5 m/s
        # Calculate blend weight (0.0 = all low-speed, 1.0 = all high-speed)
        blend_weight = get_smooth_blend_weight(v_ego)
        low_weight = 1.0 - blend_weight
        high_weight = blend_weight
        
        # Blend outputs using smooth weights
        blended_output = (low_weight * low_output + high_weight * high_output)
        
        return blended_output
    
    def __repr__(self):
        return f"Blended2PIDController"

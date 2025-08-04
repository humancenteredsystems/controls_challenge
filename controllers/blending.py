"""
Blending functions for PID controllers.
"""
import math

def get_smooth_blend_weight(v_ego: float, threshold: float = 15.0, smoothness: float = 1.5) -> float:
    """
    Calculates a smooth blend weight using a sigmoid function.

    Args:
        v_ego: The vehicle's ego speed.
        threshold: The speed at which the blend weight is 0.5.
        smoothness: The steepness of the transition.

    Returns:
        A blend weight between 0.0 and 1.0.
    """
    return 1.0 / (1.0 + math.exp(-(v_ego - threshold) / smoothness))

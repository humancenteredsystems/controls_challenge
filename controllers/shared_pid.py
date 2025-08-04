"""
Shared PID implementation for consistent dt = 0.1 time step handling.
Used by all permanent controller classes to ensure eval.py compatibility.

This module provides the canonical SpecializedPID implementation that all
pipeline stages should use to maintain time step consistency.
"""

class SpecializedPID:
    """
    Canonical PID controller with dt = 0.1 for pipeline consistency.
    
    This implementation ensures all pipeline stages use the same time step
    scaling, which is critical for eval.py compatibility.
    
    Args:
        p (float): Proportional gain
        i (float): Integral gain  
        d (float): Derivative gain
        name (str, optional): Name for debugging/logging
    """
    def __init__(self, p, i, d, name=""):
        self.p = p
        self.i = i  
        self.d = d
        self.name = name
        self.error_integral = 0
        self.prev_error = 0
        self.error_derivative = 0
    
    def update(self, error):
        """
        Update PID controller with new error value.
        
        Uses dt = 0.1 to match tinyphysics DEL_T = 0.1 (10 Hz) for eval.py compatibility.
        
        Args:
            error (float): Current error value
            
        Returns:
            float: PID control output
        """
        dt = 0.1  # Match tinyphysics DEL_T = 0.1 (10 Hz) - CRITICAL FOR eval.py
        self.error_integral += error * dt
        self.error_derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * self.error_derivative
    
    def reset(self):
        """Reset PID controller state."""
        self.error_integral = 0
        self.prev_error = 0
        self.error_derivative = 0
    
    def __repr__(self):
        """String representation for debugging."""
        return f"SpecializedPID(p={self.p:.3f}, i={self.i:.3f}, d={self.d:.3f}, name='{self.name}')"

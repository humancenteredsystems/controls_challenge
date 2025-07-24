#!/usr/bin/env python3

"""
Optimized Blender Controller - Competition Submission

This controller implements the optimal blending strategy discovered through
evolutionary optimization of the Tournament #2 winner parameters.

Performance:
- Final cost: 45.70 (37% improvement over Tournament #2)
- Tournament #2 baseline: 72.24
- Strategy: Speed-threshold blending at 55.33 mph

Optimal Configuration:
- Speed threshold: 55.33 mph (24.74 m/s)
- Low-speed weight: 96.5% (below threshold)
- High-speed weight: 16.3% (above threshold)
- PID1 (low-speed): P=0.291, I=0.120, D=-0.082
- PID2 (high-speed): P=0.150, I=0.060, D=-0.038

Discovered through GPU-accelerated evolutionary optimization with 8 rounds,
population size 15, evaluated on 15 data files per strategy.
"""

from . import BaseController

class Controller(BaseController):
    """
    Competition-ready optimized blender controller.
    
    Implements speed-threshold blending strategy evolved from Tournament #2
    winner parameters. Achieves 45.70 total_cost through intelligent PID
    parameter blending based on vehicle velocity.
    """
    
    def __init__(self):
        # Optimal PID parameters from Tournament #2 winner
        self.pid1_gains = [0.291, 0.120, -0.082]  # Low-speed specialized
        self.pid2_gains = [0.150, 0.060, -0.038]  # High-speed specialized
        
        # Optimal blending strategy parameters
        self.speed_threshold = 24.74  # 55.33 mph in m/s
        self.low_speed_weight = 0.9654  # Weight for PID1 below threshold
        self.high_speed_weight = 0.1634  # Weight for PID1 above threshold
        
        # PID state tracking for both controllers
        self.pid1_integral = 0.0
        self.pid1_prev_error = 0.0
        self.pid2_integral = 0.0
        self.pid2_prev_error = 0.0
        
        # Print statements removed for multiprocessing compatibility with eval.py
        # Controller initialized with optimal blending parameters
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Optimized blender control with speed-threshold strategy.
        
        Uses evolved blending weights based on vehicle velocity:
        - Below 55.33 mph: 96.5% low-speed PID + 3.5% high-speed PID
        - Above 55.33 mph: 16.3% low-speed PID + 83.7% high-speed PID
        
        Args:
            target_lataccel: Desired lateral acceleration
            current_lataccel: Current lateral acceleration  
            state: Current simulator state
            future_plan: Future trajectory plan
            
        Returns:
            Blended control output
        """
        error = target_lataccel - current_lataccel
        dt = 1.0 / 100.0  # 100 Hz control loop
        
        # PID1 (low-speed specialized) controller
        self.pid1_integral += error * dt
        pid1_derivative = (error - self.pid1_prev_error) / dt
        self.pid1_prev_error = error
        
        pid1_output = (
            self.pid1_gains[0] * error +               # Proportional
            self.pid1_gains[1] * self.pid1_integral +  # Integral  
            self.pid1_gains[2] * pid1_derivative       # Derivative
        )
        
        # PID2 (high-speed specialized) controller
        self.pid2_integral += error * dt
        pid2_derivative = (error - self.pid2_prev_error) / dt
        self.pid2_prev_error = error
        
        pid2_output = (
            self.pid2_gains[0] * error +               # Proportional
            self.pid2_gains[1] * self.pid2_integral +  # Integral
            self.pid2_gains[2] * pid2_derivative       # Derivative
        )
        
        # Optimal speed-threshold blending strategy
        if state.v_ego < self.speed_threshold:
            # Below threshold: favor low-speed controller
            pid1_weight = self.low_speed_weight   # 96.5%
            pid2_weight = 1.0 - pid1_weight       # 3.5%
        else:
            # Above threshold: favor high-speed controller  
            pid1_weight = self.high_speed_weight  # 16.3%
            pid2_weight = 1.0 - pid1_weight       # 83.7%
        
        # Optimally blended control output
        blended_output = pid1_weight * pid1_output + pid2_weight * pid2_output
        
        return blended_output
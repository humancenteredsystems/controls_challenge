#!/usr/bin/env python3
"""
Tournament #3: Simple Neural Blending Controller

Your original concept - clean and focused:
1. Take Tournament #2's proven PID parameters (STATIC - 324.83 cost) 
2. Only optimize neural blending weight: weight = neural_network(state)
3. Output = weight * tournament2_low + (1-weight) * tournament2_high

Architecture:
- Tournament #2 PIDs: STATIC (never re-optimize these proven parameters)
- Neural component: Single 8→16→1 network predicting blend weight [0,1]
- Guaranteed minimum: Never worse than Tournament #2's 324.83 cost
- Target: 5-15% improvement through intelligent blending

Updated to use working Tournament #3 neural controller with proper ONNX model loading.
"""

from .tournament3_neural import Tournament3Controller

# Use the working Tournament #3 neural controller
Controller = Tournament3Controller
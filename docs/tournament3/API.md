# Tournament #3 API Reference

## Import Patterns

### Simple Integration (Recommended)
```python
from controllers.tournament3_simple import Controller
```

### Direct Neural Controller
```python
from controllers.tournament3_neural import Tournament3Controller
```

### Legacy Tournament #1/#2 (For Comparison)
```python
from controllers.neural_blended import Controller
```

## Constructor

### Basic Usage
```python
controller = Controller()  # Auto-detects neural models
```

### Status Check
```python
has_neural = controller.blender_session is not None
print(f"Neural status: {'LOADED' if has_neural else 'FALLBACK'}")
```

## Method Signatures

### Primary Control Method
```python
def update(target_lataccel: float, current_lataccel: float, 
           state: State, future_plan: FuturePlan) -> float:
    """
    Returns: Control output (steering adjustment)
    """
```

### Required State Object
```python
from collections import namedtuple
State = namedtuple('State', ['v_ego', 'roll_lataccel', 'a_ego'])
state = State(v_ego=45.0, roll_lataccel=0.1, a_ego=0.0)
```

## Integration Example
```python
from controllers.tournament3_simple import Controller
from collections import namedtuple

# Initialize
controller = Controller()
State = namedtuple('State', ['v_ego', 'roll_lataccel', 'a_ego'])

# Control loop
state = State(v_ego=45, roll_lataccel=0.1, a_ego=0.0)
output = controller.update(1.0, 0.5, state, None)
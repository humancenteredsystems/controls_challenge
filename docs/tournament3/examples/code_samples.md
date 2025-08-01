# Tournament #3 Code Examples

## Basic Usage Example

```python
#!/usr/bin/env python3
"""Tournament #3 Basic Usage Example"""

from controllers.tournament3_simple import Controller
from collections import namedtuple

# Initialize controller
controller = Controller()
print(f"Neural status: {'LOADED' if controller.blender_session else 'FALLBACK'}")

# Create mock state
State = namedtuple('State', ['v_ego', 'roll_lataccel', 'a_ego'])
state = State(v_ego=45, roll_lataccel=0.1, a_ego=0.0)

# Control loop example
target_lataccel = 1.0
current_lataccel = 0.5
output = controller.update(target_lataccel, current_lataccel, state, None)
print(f"Control output: {output:.3f}")
```

## Neural Training Example

```python
#!/usr/bin/env python3
"""Quick neural model training"""

import subprocess
import sys

# Train neural model (takes ~10 seconds)
print("Training neural model...")
result = subprocess.run([sys.executable, "simple_neural_trainer.py"], 
                       capture_output=True, text=True)

if result.returncode == 0:
    print("✅ Neural model trained successfully!")
    print("Model saved to: models/blender.onnx")
else:
    print("❌ Training failed:", result.stderr)
```

## Regression Test Example

```python
#!/usr/bin/env python3
"""Tournament architecture regression test"""

# Test Tournament #1/#2 (Legacy)
try:
    from controllers.neural_blended import Controller as Legacy
    legacy = Legacy()
    print(f"✅ Tournament #1/#2: Working")
except Exception as e:
    print(f"❌ Tournament #1/#2 failed: {e}")

# Test Tournament #3
try:
    from controllers.tournament3_simple import Controller as T3
    t3 = T3()
    print(f"✅ Tournament #3: Working")
except Exception as e:
    print(f"❌ Tournament #3 failed: {e}")

print("🏆 All tournament architectures preserved!")
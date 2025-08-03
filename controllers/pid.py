from . import BaseController
from .shared_pid import SpecializedPID

class Controller(BaseController):
    """Simple PID controller using canonical dt=0.1 scaling."""
    def __init__(self):
        self.pid = SpecializedPID(0.3, 0.05, -0.1, "BaselinePID")

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        return self.pid.update(error)

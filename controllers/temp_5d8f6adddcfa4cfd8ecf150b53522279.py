from . import BaseController

class SpecializedPID:
    def __init__(self, p, i, d):
        self.p, self.i, self.d = p, i, d
        self.error_integral = 0
        self.prev_error = 0

    def update(self, error):
        self.error_integral += error
        diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * diff

class Controller(BaseController):
    def __init__(self):
        self.low_speed_pid = SpecializedPID(0.3964136778358708, 0.04801805174679902, -0.19627334365262142)
        self.high_speed_pid = SpecializedPID(0.29950005413151515, 0.062323787104168187, -0.06735660177091404)
        self.dynamic_pid = SpecializedPID(0.32258623660671254, 0.1309745732407239, -0.2626783743367837)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        err = target_lataccel - current_lataccel
        u1 = self.low_speed_pid.update(err)
        u2 = self.high_speed_pid.update(err)
        u3 = self.dynamic_pid.update(err)
        return u1 + u2 + u3
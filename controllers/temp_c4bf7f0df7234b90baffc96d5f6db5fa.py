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
        self.low_speed_pid = SpecializedPID(0.37656369085810826, 0.0058892646717626, -0.0725140211804912)
        self.high_speed_pid = SpecializedPID(0.1929014719949566, 0.02835005317216678, -0.030288936765394268)
        self.dynamic_pid = SpecializedPID(0.6196488901658052, 0.012277704785640163, -0.34366372734082784)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        err = target_lataccel - current_lataccel
        u1 = self.low_speed_pid.update(err)
        u2 = self.high_speed_pid.update(err)
        u3 = self.dynamic_pid.update(err)
        return u1 + u2 + u3
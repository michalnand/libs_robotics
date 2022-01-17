

class PID:
    def __init__(self, Kp, Ki, Kd, out_range_min, out_range_max, dt = 0.005):
        #PI term parameters
        self.k0 = Kp + Ki*dt
        self.k1 = -Kp

        #derivative term parameters
        self.k0d = Kd/dt
        self.k1d = -2.0*Kd/dt
        self.k2d = Kd/dt

        self.out_range_min = out_range_min
        self.out_range_max = out_range_max

        #iir filter for derivative term
        self.alpha = 2.0*dt/(Kd + 2.0*dt)

        self.reset()

        
    def reset(self, output_initial = 0.0):
        self.e0 = 0.0
        self.e1 = 0.0
        self.e2 = 0.0
        
        self.d0 = 0.0
        self.d1 = 0.0
        
        self.y      = output_initial
        self.y_der  = 0.0

    def step(self, x_setpoint, x_measured):
        #shift error
        self.e2 = self.e1
        self.e1 = self.e0
        self.e0 = x_setpoint - x_measured

        #compute PI term
        y_pi = self.k0*self.e0 + self.k1*self.e1

        #compute low pass filtered D term
        self.y_der = (1.0 - self.alpha)*self.y_der + self.alpha*(self.k0d*self.e0 + self.k1d*self.e1 + self.k2d*self.e2)
       
        #integrate
        self.y = self.y + y_pi + self.y_der

        #saturate
        if self.y < self.out_range_min:
            self.y = self.out_range_min

        if self.y > self.out_range_max:
            self.y = self.out_range_max


        return self.y

from numpy.core.fromnumeric import mean
import torch
import numpy
from dynamical_system import *
import matplotlib.pyplot as plt

class MotorModel(torch.nn.Module):
    def __init__(self, batch_size, motor_constant, resistance, wheel_radius, wheel_mass):
        super().__init__()

        self.batch_size      = batch_size
        self.system_order    = 2
        inputs_count         = 1
        sigma                = 2.0

        j               = 0.5*wheel_mass*(wheel_radius**2)

        self.mean_a          = torch.zeros((self.system_order + inputs_count, self.system_order))
        self.sigma_a         = sigma*torch.ones((self.system_order + inputs_count, self.system_order))

        self.mean_y          = torch.zeros((self.system_order, 1))
        self.sigma_y         = torch.zeros((self.system_order, 1))

        self.mean_a[0][0]                    = -(motor_constant/resistance*motor_constant)*1.0/j
        self.mean_a[self.system_order][0]    =  (motor_constant/resistance)*1.0/j

        self.mean_y[0][0]    = 1.0

        self.new_system()

    def new_system(self):
        self.ds  = DynamicalSystem(self.batch_size, self.mean_a, self.sigma_a, self.mean_y, self.sigma_y)

    def forward(self, state, input, dt = 0.001):

        state, y = self.ds(state, input, dt)

        state = torch.clip(state, -256, 256)

        return state, y


    
if __name__ == "__main__":

    nominal_voltage   = 6       #6Volts
    no_load_speed     = 1000    #1000rpm
    stall_torque      = 0.57    #0.57kg.cm

    wheel_radius      = 32.0*0.5/1000.0    #wheel diameter is 32mm
    wheel_mass        = 0.1               #wheel weight 100 grams
        

    #speed to rad/s
    no_load_speed     = (2.0*numpy.pi/60.0)*no_load_speed

    #torque to Nm
    stall_torque      = 0.09807*stall_torque

    motor_constant    = nominal_voltage/no_load_speed
    resistance        = motor_constant*nominal_voltage/stall_torque

    batch_size        = 64

    motor             = MotorModel(batch_size, motor_constant, resistance, wheel_radius, wheel_mass)
    state             = torch.zeros((batch_size, motor.system_order))

    
    t_max = 1500
    dt    = 0.001


    l_time      = numpy.zeros((t_max, batch_size))
    l_voltage   = numpy.zeros((t_max, batch_size))
    l_speed     = numpy.zeros((t_max, batch_size))
    
    time = 0
    
    for t in range(t_max):

        v = 0.0
        if t > 0.1*t_max:
            v = 0.5*nominal_voltage
        if t > 0.2*t_max:
            v = nominal_voltage
        if t > 0.5*t_max:
            v = 0.0
        if t > 0.75*t_max:
            v = nominal_voltage

        input    = torch.ones((batch_size, 1))*v
        state, y = motor(state, input, dt)

        y_      = y[:,0].detach().to("cpu").numpy().copy()
        speed   = y_*60.0/(2.0*numpy.pi)

        l_time[t]       = time
        l_voltage[t]    = v
        l_speed[t]      = speed

        time+= dt

    plt.subplot(211)
    plt.ylabel('voltage [V]')
    plt.xlabel('time [s]')
    plt.plot(l_time, l_voltage, color='salmon')

    plt.subplot(212)
    plt.ylabel('speed [RPM]')
    plt.xlabel('time [s]')
    plt.plot(l_time, l_speed, color='deepskyblue')
    
    plt.show()
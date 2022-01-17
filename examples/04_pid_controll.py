import numpy
import torch
import LibsRobotics
import matplotlib.pyplot as plt



nominal_voltage   = 6       #6Volts
no_load_speed     = 1000    #1000rpm
stall_torque      = 0.57    #0.57kg.cm

wheel_radius      = 32.0*0.5/1000.0     #wheel diameter is 32mm
wheel_mass        = 0.1                 #wheel weight 100 grams
    

#speed to rad/s
no_load_speed     = (2.0*numpy.pi/60.0)*no_load_speed

#torque to Nm
stall_torque      = 0.09807*stall_torque

motor_constant    = nominal_voltage/no_load_speed
resistance        = motor_constant*nominal_voltage/stall_torque

#motors count
batch_size        = 64 

motor             = LibsRobotics.MotorModel(batch_size, motor_constant, resistance, wheel_radius, wheel_mass)
state             = torch.zeros((batch_size, motor.system_order))






import torch
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")

from LibsRobotics.controll.dynamical_system import *
from LibsRobotics.controll.ode_solver       import *

'''
spring mass harmonic oscilator

dx = Ax + Bu 
y  = Cu

A - state matrix
B - input matrix
C - output matrix

matrices consist of mean + std parts
'''
m       = 1.5   #weight mass [kg]
damping = 1.3
t       = 0.5   #period [s]
k       = (2.0*3.141592654*m/t)**2.0    #spring stiffness 


#state matrix A
mat_a   = torch.FloatTensor([[-damping, -k/m], [1.0, 0.0]])
sigma_a = torch.FloatTensor([[0.1,  0.1], [0.0, 0.0]])

#input matrix
mat_b   = torch.FloatTensor([[1.0/m, 0.0], [0.0, 0.0]])
sigma_b = torch.FloatTensor([[0.1,   0.0], [0.0, 0.0]])

#output matrix
mat_c   = torch.FloatTensor([[0.0, 1.0]])
sigma_c = torch.FloatTensor([[0.0, 0.0]])

#16 oscilators in batch
batch_size = 16

#create dynamical system
ds = DynamicalSystem(batch_size, mat_a, sigma_a, mat_b, sigma_b, mat_c, sigma_c)

#solver = ODESolverEuler(ds)
solver = ODESolverRK4(ds)

#random initial state
x  = torch.randn((batch_size, mat_a.shape[0]), requires_grad=True)

#zero control signal
u  = torch.zeros((batch_size, mat_a.shape[0]), requires_grad=True)

#200Hz sampling rate
dt          = 1.0/200.0
time        = 0
x_result    = []
y_result    = []

for i in range(1000):
    x, y = solver.step(x, u, dt)

    x_result.append(time)
    y_result.append(y[0].detach().to("cpu").numpy())

    time+= dt

#plor result

plt.plot(x_result, y_result)
plt.xlabel("time     [s]")
plt.ylabel("position [m]")
plt.show()
import torch
import sys
sys.path.append("../..")

from LibsRobotics.controll.dynamical_system import *
from LibsRobotics.controll.ode_solver       import *
from LibsRobotics.common.plot_response      import *


'''
spring mass harmonic oscilator

dx = Ax + Bu 
y  = Cu

A - state matrix
B - input matrix
C - output matrix

matrices consist of mean + std parts
'''
m       = 0.05   #weight mass [kg]
damping = 45
t       = 0.1   #period [s]
k       = (2.0*3.141592654*m/t)**2.0    #spring stiffness 
amp     = 1.0


#state matrix A
mat_a   = torch.FloatTensor([[-damping/m, -k/m], [1.0, 0.0]])
sigma_a = torch.FloatTensor([[0.1,  0.1], [0.0, 0.0]])

#input matrix
mat_b   = torch.FloatTensor([[amp/m], [0.0]])
sigma_b = torch.FloatTensor([[0.1  ], [0.0]])

#output matrix
mat_c   = torch.FloatTensor([[1.0, 0.0], [0.0, 1.0]])
sigma_c = torch.FloatTensor([[0.0, 0.0], [0.0, 0.0]])

#64 oscilators in batch
batch_size = 64

#create dynamical system
ds = DynamicalSystem(batch_size, mat_a, sigma_a, mat_b, sigma_b, mat_c, sigma_c)

#solver = ODESolverEuler(ds)
solver = ODESolverRK4(ds)


time_steps = 1000

#zero initial state
x  = torch.zeros((batch_size, mat_a.shape[0]))

#unit step from sample 100
u_trajectory             = torch.zeros((time_steps, batch_size, mat_b.shape[1]))
u_trajectory[100:, :, :] = 1.0

t_trajectory = torch.zeros(time_steps)

y_trajectory = torch.zeros((time_steps, batch_size, mat_c.shape[0]))

#200Hz sampling rate
dt          = 1.0/200.0
time        = 0



for i in range(time_steps):
   
    x, y = solver.step(x, u_trajectory[i], dt)

    t_trajectory[i] = time
    y_trajectory[i] = y

    time+= dt


plot_output(t_trajectory, u_trajectory, y_trajectory, "force [N]", ["velocity [m/s]", "position [m]"], "spring_mass.png")

'''
u_trajectory = u_trajectory*0.224808943
y_trajectory = y_trajectory*torch.tensor([[3.2808399, 39.3700787]])
plot_output(t_trajectory, u_trajectory, y_trajectory, "force [lb]", ["velocity [ft/s]", "position [inch]"], "spring_mass.png")
'''
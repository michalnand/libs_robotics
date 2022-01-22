import torch
import sys
sys.path.append("../..")

from LibsRobotics.controll.dynamical_system import *
from LibsRobotics.controll.ode_solver       import *
from LibsRobotics.controll.plot_response      import *


#input parameters
v_nom       = 6.0                           #nominal voltage,       6[V]
friction    = 0.0                           #friction force,        [Nms]
w_nom       = 1000.0*2.0*torch.pi/60.0      #no load speed,         1000[rpm]
torque      = 0.57*0.09807                  #stall torque,          0.57[kg.cm]

wr          = 32.0*0.5/1000.0               #wheel diameter,        32[mm]
wm          = 10.0/1000.0                   #wheel + rotor mass,    10[grams]
#i_max       = 1.6                          #max stall current,     1.6[A]


k       = v_nom/w_nom       #motor constant, [Vs/rad]

i_max   = torque/k
r       = v_nom/i_max       #resistance [ohm]

#j       = 0.5*wm*(wr**2)    #inertia momentum
j       = wm*(wr**2)    #inertia momentum

a   = -((k**2)/r + friction)*(1.0/j)
b   = k/(r*j)


mat_a   = torch.FloatTensor([[a]])
sigma_a = torch.FloatTensor([[0.01]])

#input matrix
mat_b   = torch.FloatTensor([[b]])
sigma_b = torch.FloatTensor([[0.01]])

#output matrix
mat_c   = torch.FloatTensor([[1.0]])
sigma_c = torch.FloatTensor([[0.0]])



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
u_trajectory[100:, :, :] = v_nom

t_trajectory = torch.zeros(time_steps)

y_trajectory = torch.zeros((time_steps, batch_size, mat_c.shape[0]))

#200Hz sampling rate
dt          = 1.0/10000.0
time        = 0



for i in range(time_steps):
   
    x, y = solver.step(x, u_trajectory[i], dt)

    t_trajectory[i] = time
    y_trajectory[i] = y

    time+= dt


y_trajectory = y_trajectory*60.0/(2.0*torch.pi)

plot_output(t_trajectory, u_trajectory, y_trajectory, "input [V]", ["output [rpm]"], "dc_motor.png")

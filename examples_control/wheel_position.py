import torch
import sys
sys.path.append("..")

import LibsControll


#wheel position controll
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


mat_a   = torch.FloatTensor([[a, 0.0],      [1.0, 0.0]])
sigma_a = torch.FloatTensor([[0.0, 0.0],    [0.0, 0.0]]) 

#input matrix
mat_b   = torch.FloatTensor([[b,    0.0],  [0.0, 0.0]])
sigma_b = torch.FloatTensor([[0.0,  0.0],  [0.0, 0.0]])

#output matrix
mat_c   = torch.FloatTensor([[1.0, 0.0], [0.0, 1.0]])
sigma_c = torch.FloatTensor([[0.0, 0.0], [0.0, 0.0]])



#64 plants in batch
batch_size = 64

#create dynamical system
ds = LibsControll.DynamicalSystem(batch_size, mat_a, sigma_a, mat_b, sigma_b, mat_c, sigma_c)

#solver = ODESolverEuler(ds)
solver = LibsControll.ODESolverRK4(ds)


time_steps = 500

#zero initial state
x  = torch.zeros((batch_size, mat_a.shape[0]))

#unit step from sample 100
u_trajectory             = torch.zeros((time_steps, batch_size, mat_b.shape[1]))
u_trajectory[100:, :, 0] = v_nom

t_trajectory = torch.zeros(time_steps)

y_trajectory = torch.zeros((time_steps, batch_size, mat_c.shape[0]))

#200Hz sampling rate
dt          = 1.0/1000.0
time        = 0


for i in range(time_steps):
   
    x, y = solver.step(x, u_trajectory[i], dt)

    t_trajectory[i] = time
    y_trajectory[i] = y

    time+= dt


#conversion, rad/s to rpm, rad to degrees
conv = torch.tensor([[60.0/(2.0*torch.pi), 360.0/(2.0*torch.pi)]])

y_trajectory = y_trajectory*conv

LibsControll.plot_output(t_trajectory, u_trajectory, y_trajectory, ["input [V]"], ["speed [rpm]", "position [degrees]"], "imgs/wheel_position.png")

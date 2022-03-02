import torch
import sys
sys.path.append("..")

import LibsControll

inputs_count    = 2
outputs_count   = 6
hidden_count    = 8


t0 = LibsControll.LoadTrajectory([2, 3], [4, 5, 6, 7], "run_0.txt")




mat_a   = torch.randn([[a]])
sigma_a = torch.randn([[0.1]])

#input matrix
mat_b   = torch.randn([[b]])
sigma_b = torch.randn([[0.1]])

#output matrix
mat_c   = torch.randn([[1.0]])
sigma_c = torch.randn([[0.0]])


#64 plants in batch
batch_size = 64

#create dynamical system
ds = LibsControll.DynamicalSystem(batch_size, mat_a, sigma_a, mat_b, sigma_b, mat_c, sigma_c)

#solver = ODESolverEuler(ds)
solver = LibsControll.ODESolverRK4(ds)


time_steps = 1000

#zero initial state
x  = torch.zeros((batch_size, mat_a.shape[0]))

#unit step from sample 100
u_trajectory             = torch.zeros((time_steps, batch_size, mat_b.shape[1]))
u_trajectory[100:, :, 0] = v_nom

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

LibsControll.plot_output(t_trajectory, u_trajectory, y_trajectory, ["input [V]"], ["output [rpm]"], "imgs/dc_motor.png")

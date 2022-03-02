import torch
import sys
sys.path.append("..")

import LibsControll

k_spring = 1.0
mass_1   = 1.0
mass_2   = 1.0

k_spring_noise  = 0.1
mass_1_noise    = 0.1
mass_2_noise    = 0.1


#64 plants in batch
batch_size = 64

#create dynamical system
ds = LibsControll.plant.TwoCarts(batch_size, k_spring, mass_1, mass_2, k_spring_noise, mass_1_noise, mass_2_noise)

#solver = ODESolverEuler(ds)
solver = LibsControll.ODESolverRK4(ds)


time_steps = 1000

#zero initial state
x  = torch.zeros((batch_size, ds.mat_a.shape[1]))

#unit step from sample 100
u_trajectory             = torch.zeros((time_steps, batch_size, ds.mat_b.shape[2]))
u_trajectory[100:, :, 0] = 1.0

t_trajectory = torch.zeros(time_steps)

y_trajectory = torch.zeros((time_steps, batch_size, ds.mat_c.shape[1]))

#200Hz sampling rate
dt          = 1.0/100.0
time        = 0



for i in range(time_steps):
   
    x, y = solver.step(x, u_trajectory[i], dt)

    t_trajectory[i] = time
    y_trajectory[i] = y

    time+= dt



print(">>> ", y_trajectory.shape)
LibsControll.plot_output(t_trajectory, u_trajectory, y_trajectory, ["input [N]"], ["position 1 [m]", "velocity 1 [m/s]", "position 2 [m]", "velocity 2 [m/s]"], "imgs/two_carts.png")

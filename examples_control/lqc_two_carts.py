import torch
import numpy
import sys
sys.path.append("..")

import LibsControll

k_spring        = 1.0
mass_1          = 0.5
mass_2          = 0.05 

k_spring_noise  = 0.1
mass_1_noise    = 0.1
mass_2_noise    = 0.1


steps_count = 1000

#64 plants in batch
batch_size = 64

#create dynamical system
plant = LibsControll.plant.TwoCarts(batch_size, k_spring, mass_1, mass_2, k_spring_noise, mass_1_noise, mass_2_noise)

#solver = ODESolverEuler(plant)
solver = LibsControll.ODESolverRK4(plant)


required_dim    = 4
plant_outputs   = plant.mat_c.shape[1]
plant_inputs    = plant.mat_b.shape[2]

amplitude_max = 1.0

dt = 1.0/50.0


required_generator      = LibsControll.SignalUnitStep(steps_count, required_dim, period = 5, randomise=True, amplitudes = [amplitude_max, 0, 0, 0])
noise_generator         = LibsControll.SignalGaussianNoise(steps_count, plant_outputs, amplitudes = [0.05*amplitude_max])

controller              = LibsControll.LinearQuadraticController(required_dim, plant_outputs, plant_inputs, 0.1)
#controller              = LibsControll.LinearQuadraticControllerHidden(required_dim, plant_outputs, plant_inputs, 4, 0.1)
#controller              = LibsControll.NonLinearController(required_dim, plant_outputs, plant_inputs)

optimizer = torch.optim.Adam(controller.parameters(), lr=0.1)


position_weight = 1.0
velocity_weight = 0.001

for i in range(100):
    torch.manual_seed(numpy.random.randint(1000000000))
    plant.new_system()

    required_trajectory = required_generator.sample_batch(batch_size)
    required_trajectory = torch.from_numpy(required_trajectory).float()

    noise_trajectory    = noise_generator.sample_batch(batch_size)
    noise_trajectory    = torch.from_numpy(noise_trajectory).float()


    controller_u_trajectory, plant_y_trajectory = LibsControll.closed_loop_response(plant, controller, required_trajectory, None, noise_trajectory, dt)

    loss = position_weight*((required_trajectory[:,:,0] - plant_y_trajectory[:,:,0])**2).mean()
    #loss+= velocity_weight*((required_trajectory[:,:,3] - plant_y_trajectory[:,:,3])**2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(i, loss)

    
    if i%10 == 0:
        torch.manual_seed(0)
        plant.new_system()

        required_trajectory = torch.zeros((steps_count, batch_size, required_dim)).float()
        required_trajectory[steps_count//10:, :, 0] = amplitude_max
        

        controller_u_trajectory, plant_y_trajectory  = LibsControll.closed_loop_response(plant, controller, required_trajectory, None, None, dt)


        time_trajectory = torch.tensor(range(steps_count))*dt

        LibsControll.plot_controll_output(time_trajectory, controller_u_trajectory, required_trajectory, plant_y_trajectory, ["force [N]"], [ "position 1 [m]",  "velocity 1 [m/2]", "position 2 [m]", "velocity 2 [m/s]"], "imgs/lqc_two_carts/" + str(i))
    


import numpy
import torch
import sys
sys.path.append("..")

import LibsControll


v_nom = 1.0

free_run_rpm  = 1180.0
time_constant = 92*0.001 

k   = free_run_rpm/v_nom
a   = -1.0/time_constant
b   = k/time_constant

w_max   = 1000

mat_a   = torch.FloatTensor([[a]])
sigma_a = torch.FloatTensor([[0.1]])

#input matrix
mat_b   = torch.FloatTensor([[b]])
sigma_b = torch.FloatTensor([[0.1]])

#output matrix
mat_c   = torch.FloatTensor([[1.0]])
sigma_c = torch.FloatTensor([[0.0]])


#64 plants in batch
batch_size  = 64
steps_count = 256

dt = 1.0/250.0


#create dynamical system
plant = LibsControll.DynamicalSystem(batch_size, mat_a, sigma_a, mat_b, sigma_b, mat_c, sigma_c)


plant_inputs    = plant.mat_b.shape[2]
plant_outputs   = plant.mat_c.shape[2]
required_dim    = plant.mat_c.shape[2]

amplitude_max   = v_nom

required_generator      = LibsControll.SignalUnitStep(steps_count, required_dim, period = 5, randomise=True, amplitudes = [amplitude_max])
noise_generator         = LibsControll.SignalGaussianNoise(steps_count, plant_outputs, amplitudes = [0.05*amplitude_max])

controller              = LibsControll.LinearQuadraticController(required_dim, plant_outputs, plant_inputs)


optimizer = torch.optim.Adam(controller.parameters(), lr=0.001)

controller_weight   = 0.1
speed_weight        = 1.0


for i in range(100):
    torch.manual_seed(numpy.random.randint(1000000000))
    plant.new_system() 

    required_trajectory = required_generator.sample_batch(batch_size)
    required_trajectory = torch.from_numpy(required_trajectory).float()

    noise_trajectory    = noise_generator.sample_batch(batch_size)
    noise_trajectory    = torch.from_numpy(noise_trajectory).float()


    controller_u_trajectory, plant_y_trajectory = LibsControll.closed_loop_response(plant, controller, required_trajectory, None, noise_trajectory, min_value = -1.0, max_value = 1.0, dt = dt)

    loss = ((required_trajectory[:,:,0] - plant_y_trajectory[:,:,0])**2).mean()
    loss+= (controller_u_trajectory**2).mean()

    loss_controller = controller_weight*(controller_u_trajectory**2).mean()
    loss_speed      = speed_weight*((required_trajectory[:,:,0] - plant_y_trajectory[:,:,0])**2).mean()

    loss = loss_controller + loss_speed


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(i, loss)

    
    if i%10 == 0:
        torch.manual_seed(0)
        plant.new_system()

        required_trajectory = torch.zeros((steps_count, batch_size, required_dim)).float()
        required_trajectory[steps_count//2:, :, 0] = w_max
        

        controller_u_trajectory, plant_y_trajectory  = LibsControll.closed_loop_response(plant, controller, required_trajectory, None, None, dt)

        #required_trajectory     = required_trajectory*60.0/(2.0*torch.pi)
        #plant_y_trajectory      = plant_y_trajectory*60.0/(2.0*torch.pi)

        time_trajectory = torch.tensor(range(steps_count))*dt

        print("min max ", torch.min(controller_u_trajectory), torch.max(controller_u_trajectory))

        
        LibsControll.plot_controll_output(time_trajectory, controller_u_trajectory, required_trajectory, plant_y_trajectory, ["input [u]"], ["speed [rpm]"], "imgs/lqc_first_order/" + str(i))
        
        print(controller.controll_mat)
        print("\n\n\n")

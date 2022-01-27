import numpy
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
wm          = 100.0/1000.0                  #wheel + rotor mass,    100[grams]
j           = 0.5*wm*(wr**2)

#25% parameters noise
parameter_noise_level = 0.25

v_nom_noise     = parameter_noise_level
w_nom_noise     = parameter_noise_level
torque_noise    = parameter_noise_level
friction_noise  = parameter_noise_level
j_noise         = parameter_noise_level


#64 plants in batch
batch_size  = 64
steps_count = 512

dt = 1.0/200.0


#create dynamical system
#plant       = LibsControll.DynamicalSystem(batch_size, mat_a, sigma_a, mat_b, sigma_b, mat_c, sigma_c)

plant = LibsControll.plant.Servo(batch_size, v_nom, friction, w_nom, torque, j, v_nom_noise, w_nom_noise, torque_noise, friction_noise, j_noise)


plant_inputs    = plant.mat_b.shape[2]
plant_outputs   = plant.mat_c.shape[2]
required_dim    = plant.mat_c.shape[2]

amplitude_max   = 3.141592654

required_generator      = LibsControll.SignalUnitStep(steps_count, required_dim, period = 5, randomise=True, amplitudes = [0.0, amplitude_max])
noise_generator         = LibsControll.SignalGaussianNoise(steps_count, plant_outputs, amplitudes = [0.05, 0.05*amplitude_max])

#controller              = LibsControll.LinearQuadraticController(required_dim, plant_outputs, plant_inputs)

controller              = LibsControll.NonLinearController(required_dim, plant_outputs, plant_inputs)


optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)

controller_weight   = 0.1
speed_weight        = 0.0
position_weight     = 1.0



for i in range(200):
    torch.manual_seed(numpy.random.randint(1000000000))
    plant.new_system()

    required_trajectory = required_generator.sample_batch(batch_size)
    required_trajectory = torch.from_numpy(required_trajectory).float()

    noise_trajectory    = noise_generator.sample_batch(batch_size)
    noise_trajectory    = torch.from_numpy(noise_trajectory).float()


    controller_u_trajectory, plant_y_trajectory = LibsControll.closed_loop_response(plant, controller, required_trajectory, None, noise_trajectory, dt)

    dif           = required_trajectory - plant_y_trajectory

    loss_controller = controller_weight*(controller_u_trajectory**2).mean()
    loss_speed      = speed_weight*(dif[:,:,0]**2).mean()
    loss_position   = position_weight*(dif[:,:,1]**2).mean()

    loss = loss_controller + loss_speed + loss_position


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(i, loss)

    
    if i%10 == 0:
        torch.manual_seed(0)
        plant.new_system()

        required_trajectory = torch.zeros((steps_count, batch_size, required_dim)).float()
        required_trajectory[steps_count//2:, :, 1] = amplitude_max
        

        controller_u_trajectory, plant_y_trajectory  = LibsControll.closed_loop_response(plant, controller, required_trajectory, None, None, dt)

        required_trajectory     = required_trajectory*180/3.141592654
        plant_y_trajectory      = plant_y_trajectory*180/3.141592654

        time_trajectory = torch.tensor(range(steps_count))*dt

        LibsControll.plot_controll_output(time_trajectory, controller_u_trajectory, required_trajectory, plant_y_trajectory, ["voltage [V]"], ["speed [degrees/s]", "position [degrees]"], "imgs/rnn_servo/" + str(i))
    
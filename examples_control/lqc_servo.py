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
#i_max       = 1.6                          #max stall current,     1.6[A]


k       = v_nom/w_nom       #motor constant, [Vs/rad]

i_max   = torque/k
r       = v_nom/i_max       #resistance [ohm]

#j       = 0.5*wm*(wr**2)    #inertia momentum
j       = wm*(wr**2)    #inertia momentum

a   = -((k**2)/r + friction)*(1.0/j)
b   = k/(r*j)


mat_a   = torch.FloatTensor([[a, 0.0],      [1.0, 0.0]])
sigma_a = torch.FloatTensor([[0.1, 0.0],    [0.0, 0.0]]) 

#input matrix
mat_b   = torch.FloatTensor([[b],   [0.0]])
sigma_b = torch.FloatTensor([[0.1], [0.0]])

#output matrix
mat_c   = torch.FloatTensor([[1.0, 0.0], [0.0, 1.0]])
sigma_c = torch.FloatTensor([[0.1, 0.0], [0.0, 0.0]])



#64 plants in batch
batch_size  = 64
steps_count = 512

dt = 1.0/200.0


#create dynamical system
plant       = LibsControll.DynamicalSystem(batch_size, mat_a, sigma_a, mat_b, sigma_b, mat_c, sigma_c)


plant_inputs    = mat_b.shape[1]
plant_outputs   = mat_c.shape[1]
required_dim    = mat_c.shape[1]

amplitude_max   = 3.141592654

required_generator      = LibsControll.SignalUnitStep(steps_count, required_dim, period = 5, randomise=True, amplitudes = [0.0, amplitude_max])
noise_generator         = LibsControll.SignalGaussianNoise(steps_count, plant_outputs, amplitudes = [0.05, 0.05*amplitude_max])

controller              = LibsControll.LinearQuadraticController(required_dim, plant_outputs, plant_inputs)


optimizer = torch.optim.Adam(controller.parameters(), lr=0.05)

speed_weight    = 0.01
position_weight = 1.0



for i in range(100):
    torch.manual_seed(numpy.random.randint(1000000000))
    plant.new_system()

    required_trajectory = required_generator.sample_batch(batch_size)
    required_trajectory = torch.from_numpy(required_trajectory).float()

    noise_trajectory    = noise_generator.sample_batch(batch_size)
    noise_trajectory    = torch.from_numpy(noise_trajectory).float()


    controller_u_trajectory, plant_y_trajectory = LibsControll.closed_loop_response(plant, controller, required_trajectory, None, noise_trajectory, dt)

    loss_speed    = speed_weight*((required_trajectory[:,:,0] - plant_y_trajectory[:,:,0])**2).mean()
    loss_position = position_weight*((required_trajectory[:,:,1] - plant_y_trajectory[:,:,1])**2).mean()

    loss = loss_speed + loss_position

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

        LibsControll.plot_controll_output(time_trajectory, controller_u_trajectory, required_trajectory, plant_y_trajectory, ["voltage [V]"], ["speed [degrees/s]", "position [degrees]"], "imgs/lqc_servo/" + str(i))
    
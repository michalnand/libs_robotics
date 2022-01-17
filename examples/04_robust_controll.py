import numpy
import torch
import LibsRobotics
import matplotlib.pyplot as plt

samples_count       = 512
channels_count      = 1
components_count    = 5

batch_size          = 64

nominal_voltage   = 6       #6Volts
no_load_speed     = 1000    #1000rpm
stall_torque      = 0.57    #0.57kg.cm

wheel_radius      = 32.0*0.5/1000.0     #wheel diameter is 32mm
wheel_mass        = 0.1                 #wheel weight 100 grams
    

#speed to rad/s
no_load_speed     = (2.0*numpy.pi/60.0)*no_load_speed

#torque to Nm
stall_torque      = 0.09807*stall_torque

motor_constant    = nominal_voltage/no_load_speed
resistance        = motor_constant*nominal_voltage/stall_torque


testing_signal      = LibsRobotics.TestingSignal(samples_count, channels_count, components_count)


class ModelController(torch.nn.Module):
    def __init__(self, inputs_count, outputs_count):
        super(ModelController, self).__init__()

        self.gru        = torch.nn.GRU(inputs_count + outputs_count, 32, batch_first = False)
        self.output     = torch.nn.Linear(32, outputs_count)
       
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)
    
    #state shape = (sequence, batch, features)
    def forward(self, state, x):
        y, state_new = self.gru(x, state)
        
        y = self.output(y[0])

        return state_new, y


steps_max = 100

plant       = LibsRobotics.MotorModel(batch_size, motor_constant, resistance, wheel_radius, wheel_mass, parameters_sigma = 0.5)


for step in range(steps_max):
    target_input    = testing_signal.sample_batch(batch_size, batch_first = False)
    target_input_t  = torch.from_numpy(target_input)

    plant.new_system()

    controller  = ModelController(1, 1)
    optimizer   = torch.optim.Adam(controller.parameters(), lr= 0.001)


    controller_state    = torch.zeros((1, batch_size, 32), requires_grad=True)
    plant_output        = torch.zeros((1, batch_size, 1), requires_grad=True)
    plant_state         = torch.zeros((batch_size, plant.system_order), requires_grad=True)

    loss = 0.0
    for n in range(samples_count):

        target_t = target_input_t[n].unsqueeze(0)

        x = torch.cat([target_t, plant_output], dim=2)

        controller_state, controller_output = controller(controller_state, x)


        plant_state, plant_output  = plant(plant_state, controller_output, dt = 0.005)

        plant_output = plant_output.unsqueeze(0)

        error = target_t - plant_output
        loss+= (error**2).mean()



    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(step, loss)
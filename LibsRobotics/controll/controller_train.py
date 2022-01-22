import torch
import numpy

from .dynamical_system      import *
from .ode_solver            import *
from .controller            import *
from .plot_response         import *   


def noise_func(x):

    noise = torch.randn_like(x)

    return x + 0.1*noise



class ControllerTrain:
 
    def __init__(self, batch_size, required_signal_generator, dynamical_system, controller_order, loss_func):


        self.batch_size                     = batch_size
        self.required_signal_generator      = required_signal_generator
        self.ds                             = dynamical_system
        self.loss_func                      = loss_func

        #create linear dynamical system and its solver
        self.solver = ODESolverRK4(self.ds)


        #extract parameters
        self.plant_order             = self.ds.mat_a.shape[1]
        self.plant_inputs_count      = self.ds.mat_b.shape[2]
        self.plant_outputs_count     = self.ds.mat_c.shape[1]
        self.controller_order        = controller_order
        self.controller_min_output   = -8.0
        self.controller_max_output   = 8.0


        required_trajectory   = self.required_signal_generator.sample_batch(self.batch_size, batch_first = False)

        self.required_trajectory_inputs_count = required_trajectory.shape[2]
       


        print("batch_size                           = ", self.batch_size)
        print("plant_order                          = ", self.plant_order)
        print("plant_inputs_count                   = ", self.plant_inputs_count)
        print("plant_outputs_count                  = ", self.plant_outputs_count)
        print("required_trajectory_inputs_count     = ", self.required_trajectory_inputs_count)
        print("controller_order                     = ", self.controller_order)


        #create controller
        controller_inputs_count = self.required_trajectory_inputs_count + self.plant_outputs_count + self.plant_inputs_count
        
        self.controller = LinearController(controller_inputs_count, self.plant_inputs_count, self.controller_order)
        #self.controller = NonLinearController(controller_inputs_count, self.plant_inputs_count, self.controller_order)
      
        self.optimizer  = torch.optim.Adam(self.controller.parameters(), lr=0.01)


    def train(self, iterations, dt = 1.0/200.0, result_path = "./"):

        for iteration in range(iterations):
            torch.manual_seed(numpy.random.randint(1000000000))
            self.ds.new_system()

            required_trajectory   = self.required_signal_generator.sample_batch(self.batch_size, batch_first = False)
            required_trajectory   = torch.from_numpy(required_trajectory)

            #initial conditions

            #random plant state
            plant_x        = torch.randn((self.batch_size, self.plant_order), requires_grad=True)


            plant_y_trajectory, controller_y_trajectory = self.obtain_trajectory(required_trajectory, plant_x, noise_func, dt)

            loss = self.loss_func(required_trajectory, plant_y_trajectory, controller_y_trajectory)

            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(iteration, loss)

            
            if iteration%10 == 0:
                torch.manual_seed(0)
                self.ds.new_system()

                test_trajectory_length = 200

                #unit step from sample 100
                step_trajectory             = torch.zeros((test_trajectory_length, self.batch_size, self.required_trajectory_inputs_count))
                step_trajectory[100:, :, :] = 1.0*self.required_signal_generator.amp_max

                #zero plant state
                plant_x        = torch.zeros((self.batch_size, self.plant_order), requires_grad=True)


                plant_y_trajectory, controller_y_trajectory = self.obtain_trajectory(step_trajectory, plant_x, None, dt)

                t_trajectory = torch.tensor(range(test_trajectory_length))*dt

                file_name = result_path + "/plots/response_" + str(iteration).zfill(5) + ".png"

                #step_trajectory     = step_trajectory*60.0/(2.0*torch.pi)
                #plant_y_trajectory  = plant_y_trajectory*60.0/(2.0*torch.pi)

                step_trajectory     = step_trajectory*180.0/torch.pi
                plant_y_trajectory  = plant_y_trajectory[:,:,1].unsqueeze(2)*180.0/torch.pi
                plot_controll_output(t_trajectory, controller_y_trajectory, step_trajectory, plant_y_trajectory, ["position [degrees]"], file_name)
                

                torch.save(self.controller.state_dict(), result_path + "controller/controller.pt")
                #print(self.controller.to_string())
            

    def obtain_trajectory(self, required_trajectory, plant_x, noise_func, dt):

        steps_count = required_trajectory.shape[0]

        #zero plant output
        plant_y        = torch.zeros((self.batch_size, self.plant_outputs_count), requires_grad=True)

        #zero controller state
        controller_x   =  torch.zeros((self.batch_size, self.controller_order), requires_grad=True)

        controller_y    = torch.zeros((self.batch_size, self.plant_inputs_count), requires_grad=False)

        controller_y_trajectory = torch.zeros((steps_count, self.batch_size, self.plant_inputs_count))
        plant_y_trajectory      = torch.zeros((steps_count, self.batch_size, self.plant_outputs_count))

        for i in range(steps_count):

            if noise_func is not None:
                plant_y_noised = noise_func(plant_y)
            else:
                plant_y_noised = plant_y

            #create controller input
            #take required value and plant output
            #due controller output saturation, also take saturated output (ability to learn antiwindup)
            controller_input = torch.cat([required_trajectory[i], plant_y_noised, controller_y], dim=1)


            #proces controller update
            controller_x, controller_y  = self.controller(controller_x, controller_input)

            #saturate actuator
            controller_y = torch.clamp(controller_y, self.controller_min_output, self.controller_max_output)

            controller_y_trajectory[i] = controller_y

            #proces dynamical plant update
            plant_x, plant_y = self.solver.step(plant_x, controller_y, dt)

            plant_y_trajectory[i] = plant_y



        return plant_y_trajectory, controller_y_trajectory

        
        

        '''
        #unit step from sample 100
        step_trajectory             = torch.zeros((steps_count, batch_size, self.plant_outputs_count))
        step_trajectory[100:, :, :] = 1.0

        '''

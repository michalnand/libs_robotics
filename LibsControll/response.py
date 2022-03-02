import torch
from .ode_solver import *

def closed_loop_response(plant, controller, required_trajectory, disturbance = None, noise = None, min_value = -10**10, max_value = 10**10, dt = 1.0/200.0):

    steps               = required_trajectory.shape[0]
    batch_size          = required_trajectory.shape[1]
    plant_outputs_count = plant.mat_c.shape[1]

    #Runge-Kutta solver
    ode_solver  = ODESolverRK4(plant)

    plant_y             = torch.zeros((batch_size, plant_outputs_count), requires_grad=True).float()
    plant_y_trajectory  = torch.zeros((steps, batch_size, plant_outputs_count)).float()

    controller_u_trajectory = torch.zeros((steps, batch_size, plant.mat_b.shape[1])).float()
 
    plant_x             = torch.zeros((batch_size, plant.mat_a.shape[1])).float()

    if hasattr(controller, "hidden_dim"):
        controller_x = torch.zeros((batch_size, controller.hidden_dim)).float()
    else:
        controller_x = None

    for n in range(steps):
        required_state  = required_trajectory[n]

        #add disturbance into plant output
        if disturbance is not None:
            plant_y = plant_y + disturbance[n]

        #add noise to similate not accurate state reading
        if noise is not None:
            plant_y = plant_y + noise[n]
        
        #obtain controller output
        if hasattr(controller, "hidden_dim"):
            controller_u, controller_x    = controller(required_state, plant_y, controller_x)
        else:
            controller_u    = controller(required_state, plant_y)

        controller_u = torch.clip(controller_u, min_value, max_value)


        #obtain plant output
        plant_x, plant_y = ode_solver.step(plant_x, controller_u, dt)

        controller_u_trajectory[n]  = controller_u
        plant_y_trajectory[n]       = plant_y
    
    return controller_u_trajectory, plant_y_trajectory


import torch
from .dynamical_system import *

class MotorModel(torch.nn.Module):
    def __init__(self, batch_size, motor_constant, resistance, wheel_radius, wheel_mass, parameters_sigma = 2.0, system_order = 2):
        super().__init__()

        self.batch_size     = batch_size
        self.system_order   = system_order
        inputs_count        = 1 

        j                   = 0.5*wheel_mass*(wheel_radius**2)

        self.mean_a          = torch.zeros((self.system_order + inputs_count, self.system_order))
        self.sigma_a         = parameters_sigma*torch.ones((self.system_order + inputs_count, self.system_order))

        self.mean_y          = torch.zeros((self.system_order, 1))
        self.sigma_y         = torch.zeros((self.system_order, 1))

        self.mean_a[0][0]                    = -(motor_constant/resistance*motor_constant)*1.0/j
        self.mean_a[self.system_order][0]    =  (motor_constant/resistance)*1.0/j

        self.mean_y[0][0]    = 1.0

        self.new_system()

    def new_system(self):
        self.ds  = DynamicalSystem(self.batch_size, self.mean_a, self.sigma_a, self.mean_y, self.sigma_y)

    def forward(self, state, input, dt = 0.001):

        state, y    = self.ds(state, input, dt)
        state       = torch.clip(state, -256, 256)

        return state, y


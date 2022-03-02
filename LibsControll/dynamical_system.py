import torch


'''
create batch of linear dynamical system :

dx = Ax + Bu 
y  = Cu


with mean given by matrices     mat_a, mat_b, mat_c
and sweeped by noise given      sigma_a, sigma_b, sigma_c

this noise simulates system identification uncertaininty



A - state matrix, [mat_a, var_a]
B - input matrix, [mat_b, var_b]
C - output matrix, [mat_c, var_c]

d_state[dx], output[y] = forward(state[x], controll[u])
'''
class DynamicalSystem(torch.nn.Module):
    def __init__(self, batch_size, mat_a, var_a, mat_b, var_b, mat_c, var_c):
        super().__init__()

        self.initial_mat_a      = mat_a
        self.initial_var_a      = var_a

        self.initial_mat_b      = mat_b
        self.initial_var_b      = var_b

        self.initial_mat_c      = mat_c
        self.initial_var_c      = var_c

        self.batch_size         = batch_size

        self.new_system()

    def new_system(self): 
        a_value = self.initial_mat_a*(1.0 + self.initial_var_a*(2.0*torch.rand((self.batch_size, ) + self.initial_var_a.shape) - 1.0)).float()
        b_value = self.initial_mat_b*(1.0 + self.initial_var_b*(2.0*torch.rand((self.batch_size, ) + self.initial_var_b.shape) - 1.0)).float()
        c_value = self.initial_mat_c*(1.0 + self.initial_var_c*(2.0*torch.rand((self.batch_size, ) + self.initial_var_c.shape) - 1.0)).float()

        self.mat_a  = torch.nn.parameter.Parameter(a_value, requires_grad=True)
        self.mat_b  = torch.nn.parameter.Parameter(b_value, requires_grad=True)
        self.mat_c  = torch.nn.parameter.Parameter(c_value, requires_grad=True)



    def forward(self, x, u):
        x_      = x.unsqueeze(2)
        u_      = u.unsqueeze(2)

        dx = torch.bmm(self.mat_a, x_) + torch.bmm(self.mat_b, u_)

        dx = dx.squeeze(2)

        y  = torch.bmm(self.mat_c, x_)
        y  = y.squeeze(2)

        return dx, y




class LinearModel(torch.nn.Module):
    def __init__(self, inputs_count, outputs_count, hidden_size, init_range = 0.01):
        super().__init__()

        self.mat_a         = torch.nn.parameter.Parameter(init_range*torch.randn((hidden_size, hidden_size)),  requires_grad=True)
        self.mat_b         = torch.nn.parameter.Parameter(init_range*torch.randn((hidden_size, inputs_count)), requires_grad=True)
        self.mat_c         = torch.nn.parameter.Parameter(init_range*torch.randn((outputs_count, hidden_size)),requires_grad=True)
        
        
        self.y_noise_std    = torch.nn.parameter.Parameter(init_range*torch.randn((outputs_count)), requires_grad=True)
        self.y_noise_mean   = torch.nn.parameter.Parameter(torch.zeros((outputs_count)), requires_grad=True)


    def forward(self, x, u):
       
        x_      = x.unsqueeze(2)
        u_      = u.unsqueeze(2)

        dx = torch.bmm(self.mat_a, x_) + torch.bmm(self.mat_b, u_)

        dx = dx.squeeze(2)

        y  = torch.bmm(self.mat_c, x_)
        y  = y.squeeze(2)

        noise = self.y_noise_std*torch.randn_like(y) + self.y_noise_mean
        y = y + noise

        return dx, y




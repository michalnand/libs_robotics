import torch


'''
create batch of linear dynamical system :

dx = Ax + Bu 
y  = Cu


with mean given by matrices     mat_a, mat_b, mat_c
and sweeped by noise given      sigma_a, sigma_b, sigma_c

this noise simulates system identification uncertaininty



A - state matrix, [mat_a, sigma_a]
B - input matrix, [mat_b, sigma_b]
C - output matrix, [mat_c, sigma_c]

d_state[dx], output[y] = forward(state[x], controll[u])
'''
class DynamicalSystem(torch.nn.Module):
    def __init__(self, batch_size, mat_a, sigma_a, mat_b, sigma_b, mat_c, sigma_c):
        super().__init__()

        self.initial_mat_a   = mat_a
        self.initial_sigma_a = sigma_a

        self.initial_mat_b   = mat_b
        self.initial_sigma_b = sigma_b

        self.initial_mat_c   = mat_c
        self.initial_sigma_c = sigma_c

        self.batch_size      = batch_size

        self.new_system()

    def new_system(self): 
        
        '''
        a_value = self.initial_mat_a + self.initial_sigma_a*torch.randn((self.batch_size, ) + self.initial_sigma_a.shape)
        b_value = self.initial_mat_b + self.initial_sigma_b*torch.randn((self.batch_size, ) + self.initial_sigma_b.shape)
        c_value = self.initial_mat_c + self.initial_sigma_c*torch.randn((self.batch_size, ) + self.initial_sigma_c.shape)
        '''

        a_value = self.initial_mat_a*(1.0 + self.initial_sigma_a*(2.0*torch.rand((self.batch_size, ) + self.initial_sigma_a.shape) - 1.0))
        b_value = self.initial_mat_b*(1.0 + self.initial_sigma_b*(2.0*torch.rand((self.batch_size, ) + self.initial_sigma_b.shape) - 1.0))
        c_value = self.initial_mat_c*(1.0 + self.initial_sigma_c*(2.0*torch.rand((self.batch_size, ) + self.initial_sigma_c.shape) - 1.0))

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



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

        self.new_system(batch_size, mat_a, sigma_a, mat_b, sigma_b, mat_c, sigma_c)

    def new_system(self, batch_size, mat_a, sigma_a, mat_b, sigma_b, mat_c, sigma_c): 
        
        a_value = mat_a + sigma_a*torch.randn((batch_size, ) + sigma_a.shape)
        b_value = mat_b + sigma_b*torch.randn((batch_size, ) + sigma_b.shape)
        c_value = mat_c + sigma_c*torch.randn((batch_size, ) + sigma_c.shape)

        self.mat_a  = torch.nn.parameter.Parameter(a_value, requires_grad=True)
        self.mat_b  = torch.nn.parameter.Parameter(b_value, requires_grad=True)
        self.mat_c  = torch.nn.parameter.Parameter(c_value, requires_grad=True)

        self.mat_a.to(mat_a.device)
        self.mat_b.to(mat_b.device)
        self.mat_c.to(mat_c.device)

    def forward(self, x, u):
        x_      = x.unsqueeze(2)
        u_      = u.unsqueeze(2)

        dx = torch.bmm(self.mat_a, x_) + torch.bmm(self.mat_b, u_)

        dx = dx.squeeze(2)

        y  = torch.bmm(self.mat_c, x_)
        y  = y.squeeze(2)

        return dx, y



import torch

class TwoCarts(torch.nn.Module):
    def __init__(self, batch_size, spring_k, mass_1, mass_2, spring_k_noise = 0.0, mass_1_noise = 0.0, mass_2_noise = 0.0):
        super().__init__()

        self.batch_size = batch_size

        self.spring_k   = spring_k
        self.mass_1     = mass_1
        self.mass_2     = mass_2

        self.spring_k_noise   = spring_k_noise
        self.mass_1_noise     = mass_1_noise
        self.mass_2_noise     = mass_2_noise
        
        self.new_system()
 


    def new_system(self):
        spring_k    = self.spring_k*(1.0 + self.spring_k_noise*self._rnd((self.batch_size)))
        mass_1      = self.mass_1*(1.0 + self.mass_1_noise*self._rnd((self.batch_size)))
        mass_2      = self.mass_2*(1.0 + self.mass_2_noise*self._rnd((self.batch_size)))
       
        mat_a   = torch.zeros((self.batch_size, 4, 4)).float()
        mat_b   = torch.zeros((self.batch_size, 4, 1)).float()
        mat_c   = torch.zeros((self.batch_size, 4, 4)).float()

        mat_a[range(self.batch_size),0,1] = 1.0
        mat_a[range(self.batch_size),1,0] = -(spring_k/mass_1)[range(self.batch_size)]
        mat_a[range(self.batch_size),1,2] = (spring_k/mass_1)[range(self.batch_size)]
        #mat_a[range(self.batch_size),2,3] = 1.0
        #mat_a[range(self.batch_size),3,0] = (spring_k/mass_2)[range(self.batch_size)]
        #mat_a[range(self.batch_size),3,2] = -(spring_k/mass_2)[range(self.batch_size)]

        mat_b[range(self.batch_size),1,0] = (1.0/mass_1)[range(self.batch_size)]

        mat_c[range(self.batch_size),0,0] = 1.0
        mat_c[range(self.batch_size),1,1] = 1.0
        mat_c[range(self.batch_size),2,2] = 1.0
        mat_c[range(self.batch_size),3,3] = 1.0

        self.mat_a  = torch.nn.parameter.Parameter(mat_a, requires_grad=True)
        self.mat_b  = torch.nn.parameter.Parameter(mat_b, requires_grad=True)
        self.mat_c  = torch.nn.parameter.Parameter(mat_c, requires_grad=True)





    def forward(self, x, u):
        x_      = x.unsqueeze(2)
        u_      = u.unsqueeze(2)

        dx = torch.bmm(self.mat_a, x_) + torch.bmm(self.mat_b, u_)

        dx = dx.squeeze(2)

        y  = torch.bmm(self.mat_c, x_)
        y  = y.squeeze(2)

        return dx, y


    def _rnd(self, shape):
        return 2.0*torch.rand(shape) - 1.0






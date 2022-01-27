import torch

class Servo(torch.nn.Module):
    def __init__(self, batch_size, v_nom, friction, w_nom, torque, j, v_nom_noise = 0.0, w_nom_noise = 0.0, torque_noise = 0.0, friction_noise = 0.0, j_noise = 0.0):
        super().__init__()

        self.batch_size = batch_size

        self.v_nom      = v_nom
        self.w_nom      = w_nom
        self.torque     = torque
        self.friction   = friction
        self.j          = j

        self.v_nom_noise    = v_nom_noise
        self.w_nom_noise    = w_nom_noise
        self.torque_noise   = torque_noise
        self.friction_noise = friction_noise
        self.j_noise        = j_noise

        self.new_system()



    def new_system(self):
        v_nom    = self.v_nom*(1.0 + self.v_nom_noise*self._rnd((self.batch_size)))
        w_nom    = self.w_nom*(1.0 + self.w_nom_noise*self._rnd((self.batch_size)))
        torque   = self.torque*(1.0+ self.torque_noise*self._rnd((self.batch_size)))
        friction = self.friction*(1.0+ self.friction_noise*self._rnd((self.batch_size)))
        j        = self.j*(1.0+ self.j_noise*self._rnd((self.batch_size))) 

        k       = v_nom/w_nom       #motor constant, [Vs/rad]

        i_max   = torque/k
        r       = v_nom/i_max       #resistance [ohm]

        a       = -((k**2)/r + friction)*(1.0/j)
        b       = k/(r*j)

        mat_a   = torch.zeros((self.batch_size, 2, 2)).float()
        mat_b   = torch.zeros((self.batch_size, 2, 1)).float()
        mat_c   = torch.zeros((self.batch_size, 2, 2)).float()

        mat_a[range(self.batch_size),0,0] = a[range(self.batch_size)]
        mat_a[range(self.batch_size),1,0] = 1.0
        
        mat_b[range(self.batch_size),0,0] = b[range(self.batch_size)]

        mat_c[range(self.batch_size),0,0] = 1.0
        mat_c[range(self.batch_size),1,1] = 1.0

        #mat_a   = torch.FloatTensor([[a, 0.0],      [1.0, 0.0]]).float()
        #mat_b   = torch.FloatTensor([[b],           [0.0]]).float()
        #mat_c   = torch.FloatTensor([[1.0, 0.0],    [0.0, 1.0]]).float()

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






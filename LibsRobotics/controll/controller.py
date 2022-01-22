import torch

'''
inputs_count    = measured variables from controlled ssytem
outputs_count   = controller outputs count
internal_size   = internal state size (e.g. 2..3x inputs_count)

linear controller, using matrices : 

state matrix A(internal_size, internal_size)
input matrix B(inputs_count, internal_size)
output matrix C(internal_size, outputs_count)


controller output : 

dx      = xA    + uB
x_new   = x     + dx 

y       = xC


note : 
matrices are transposed, to avoid transpose operations in forward pass, for faster run
(can we avoid this?)
'''
class LinearController(torch.nn.Module):
    def __init__(self, inputs_count, outputs_count, internal_size, init_range = 0.01, device = "cpu"):
        super().__init__()

        self.inputs_count   = inputs_count
        self.outputs_count  = outputs_count
        self.internal_size  = internal_size

        #matrices are in transpsoed form for faster computation 
        
        mat_a = init_range*torch.randn((self.internal_size, self.internal_size))
        mat_b = init_range*torch.randn((self.inputs_count,  self.internal_size))
        mat_c = init_range*torch.randn((self.internal_size, self.outputs_count))
        
        
        self.mat_a  = torch.nn.parameter.Parameter(mat_a, requires_grad=True)
        self.mat_b  = torch.nn.parameter.Parameter(mat_b, requires_grad=True)
        self.mat_c  = torch.nn.parameter.Parameter(mat_c, requires_grad=True)

        self.mat_a.to(device)
        self.mat_b.to(device)
        self.mat_c.to(device)

    def to_string(self):

        result = "internal_state_size = " + str(self.internal_size) + "\n"
        result+= "inputs_count = " + str(self.inputs_count) + "\n"
        result+= "outputs_count = " + str(self.outputs_count) + "\n"
        result+= "\n"

        result+= "mat_a = \n"
        mat_a = self.mat_a.detach().to("cpu").numpy()
        for j in range(self.internal_size):
            for i in range(self.internal_size):
                result+= str(mat_a[j][i]) + " "
            result+= "\n"
        result+= "\n"


        result+= "mat_b = \n"
        mat_b = self.mat_b.detach().to("cpu").numpy()
        for j in range(self.inputs_count):
            for i in range(self.internal_size):
                result+= str(mat_b[j][i]) + " "
            result+= "\n"
        result+= "\n"


        result+= "mat_c = \n"
        mat_c = self.mat_c.detach().to("cpu").numpy()
        for j in range(self.internal_size):
            for i in range(self.outputs_count):
                result+= str(mat_c[j][i]) + " "
            result+= "\n"
        result+= "\n"

        return result


    def forward(self, x, u):
        x_new   = torch.mm(x, self.mat_a) + torch.mm(u, self.mat_b)
        y       = torch.mm(x_new, self.mat_c)

        return x_new, y
        



'''
inputs_count    = measured variables from controlled ssytem
outputs_count   = controller outputs count
internal_size   = internal state size (e.g. 2..3x inputs_count)

uses GRU for non linear modeeling and linear layer for output
'''
class NonLinearController(torch.nn.Module):
    def __init__(self, inputs_count, outputs_count, internal_size = 64, device = "cpu"):
        super().__init__()

        self.inputs_count   = inputs_count
        self.outputs_count  = outputs_count
        self.internal_size  = internal_size

        self.gru    = torch.nn.GRU(self.inputs_count, self.internal_size, batch_first = True)
        self.lin    = torch.nn.Linear(self.internal_size, self.outputs_count)

        self.gru.to(device)
        self.lin.to(device)

   
    def forward(self, x, u):
        x  = x.unsqueeze(0)
        u  = u.unsqueeze(1)

        _, hn = self.gru(u, x)

        #take final hidden state
        hn = hn.squeeze(0)

        y = self.lin(hn)
        return hn, y


if __name__ == "__main__":

    inputs_count    = 6
    outputs_count   = 4
    internal_size   = 64

    #controller = LinearController(inputs_count, outputs_count, internal_size)
    controller = NonLinearController(inputs_count, outputs_count, internal_size)

    batch_size = 20

    x = torch.randn((batch_size, internal_size))

    for i in range(10):
        u = torch.randn((batch_size, inputs_count))

        x, y = controller(x, u)
        
        print(y.shape, x.shape)
        print(y[0])
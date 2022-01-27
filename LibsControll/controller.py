import torch

'''
apply controll law : u = x[required_state, plant_output]*controll_mat

required_dim        - required value dimension (controller input)
plant_output_dim    - plant outputs count (plant matrix C rows) (controller input)
plant_input_dim     - plant inputs count  (plant matrix B rows) (controller output)
'''
class LinearQuadraticController(torch.nn.Module):
    def __init__(self, required_dim, plant_output_dim, plant_input_dim):
        super().__init__()

        controll_mat        = 0.001*torch.randn((required_dim + plant_output_dim, plant_input_dim)).float()
        self.controll_mat   = torch.nn.parameter.Parameter(controll_mat, requires_grad=True)

      
    def to_string(self):
        result = ""
        result+= "controller_inputs = " + str(self.controll_mat.shape[0]) + "\n"
        result+= "controller_output = " + str(self.controll_mat.shape[1]) + "\n"
        result+= "\n"

        result+= "controll_mat = \n"
        controll_mat = self.controll_mat.detach().to("cpu").numpy()
        for j in range(self.controll_mat.shape[0]):
            for i in range(self.controll_mat.shape[1]):
                result+= str(controll_mat[j][i]) + " "
            result+= "\n"
        result+= "\n"

        return result

    def forward(self, required_state, plant_state):
        x   = torch.cat([required_state, plant_state], dim=1)
        y   = torch.mm(x, self.controll_mat)
        return y




'''
apply controll law : 
h_new = [required_state, plant_output, h]*H
u     = h*C

required_dim        - required value dimension (controller input)
plant_output_dim    - plant outputs count (plant matrix C rows) (controller input)
plant_input_dim     - plant inputs count  (plant matrix B rows) (controller output)
'''



class LinearQuadraticControllerHidden(torch.nn.Module):
    def __init__(self, required_dim, plant_output_dim, plant_input_dim, hidden_dim = 4):
        super().__init__()

        self.hidden_dim     = hidden_dim

        self.plant_input_dim = plant_input_dim

        controll_mat        = 0.001*torch.randn((required_dim + plant_output_dim + hidden_dim, hidden_dim + plant_input_dim)).float()
        self.controll_mat   = torch.nn.parameter.Parameter(controll_mat, requires_grad=True)

  
    def forward(self, required_state, plant_state, hidden_state):
        x   = torch.cat([required_state, plant_state, hidden_state], dim=1)

        tmp  = torch.mm(x, self.controll_mat)

        hidden_new, y = torch.split(tmp, [self.hidden_dim,  self.plant_input_dim], dim=1)

        return y, hidden_new



'''
class LinearQuadraticControllerHidden(torch.nn.Module):
    def __init__(self, required_dim, plant_output_dim, plant_input_dim, hidden_dim = 4):
        super().__init__()

        self.hidden_dim     = hidden_dim

        h_mat        = 0.01*torch.randn((required_dim + plant_output_dim + hidden_dim, hidden_dim)).float()
        g_mat        = 0.01*torch.randn((hidden_dim, plant_input_dim)).float()

        self.h_mat   = torch.nn.parameter.Parameter(h_mat, requires_grad=True)
        self.g_mat   = torch.nn.parameter.Parameter(g_mat, requires_grad=True)

  
    def forward(self, required_state, plant_state, hidden_state):
        x   = torch.cat([required_state, plant_state, hidden_state], dim=1)

        hidden_new  = torch.mm(x, self.h_mat)
        y           = torch.mm(hidden_new, self.g_mat)
        
        return y, hidden_new
'''

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
apply controll law : u = x[required_state, plant_output]*controll_mat

required_dim        - required value dimension
plant_output_dim    - plant outputs count (plant matrix C rows), controller inputs count
plant_input_dim     - plant inputs count  (plant matrix B rows), controller outputs count

uses GRU for non linear modeeling and linear layer for output
'''
class NonLinearController(torch.nn.Module):
    def __init__(self, required_dim, plant_output_dim, plant_input_dim, hidden_dim = 16, device = "cpu"):
        super().__init__()

        self.hidden_dim = hidden_dim

        inputs_count = required_dim + plant_output_dim

        self.gru    = torch.nn.GRU(inputs_count, self.hidden_dim, batch_first = True)
        self.lin    = torch.nn.Linear(self.hidden_dim, plant_input_dim)

        self.gru.to(device)
        self.lin.to(device)

   
    def forward(self, required_state, plant_state, hidden_state):

        x  = torch.cat([required_state, plant_state], dim=1).unsqueeze(1)

        #rnn step
        _, hidden_new = self.gru(x, hidden_state.unsqueeze(0))

        #take final hidden state
        hidden_new = hidden_new.squeeze(0)

        y = self.lin(hidden_new)

        return y, hidden_new


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
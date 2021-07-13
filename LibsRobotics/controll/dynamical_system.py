import torch


class DynamicalSystem(torch.nn.Module):
    def __init__(self, batch_size, mean_a, sigma_a, mean_y, sigma_y):
        super().__init__()

        self.new_system(batch_size, mean_a, sigma_a, mean_y, sigma_y) 
       
    def new_system(self, batch_size, mean_a, sigma_a, mean_y, sigma_y):
        eps = 0.0000000001

        a_value = (sigma_a + eps)*torch.randn((batch_size, mean_a.shape[0], mean_a.shape[1])) + mean_a
        self.a  = torch.nn.parameter.Parameter(a_value, requires_grad=True)

        y_value = (sigma_y + eps)*torch.randn((batch_size, mean_y.shape[0], mean_y.shape[1])) + mean_y
        self.y  = torch.nn.parameter.Parameter(y_value, requires_grad=True)

        self.a.to(mean_a.device) 
        self.y.to(mean_y.device)
        

    def forward(self, state, input, dt):
        state_      = state.unsqueeze(1)
        input_      = input.unsqueeze(1)
        x           = torch.cat([state_, input_], dim = 2)
        
        ds          = torch.bmm(x, self.a)*dt
        state_new   = state + ds.squeeze(1)
        y           = torch.bmm(state_new.unsqueeze(1), self.y).squeeze(1)

        return state_new, y
    

    def print_matrices(self):
        print(self.a)
        #print(self.y)
        print("\n")

if __name__ == "__main__":

    inputs_count    = 5
    outputs_count   = 2 
    system_order    = 3

    batch_size      = 64

    mean_a  = torch.randn((system_order + inputs_count, system_order))
    sigma_a = 0.01*torch.ones((system_order + inputs_count, system_order))

    mean_y  = torch.randn((system_order, outputs_count))
    sigma_y = 0.01*torch.ones((system_order, outputs_count))

    model = DynamicalSystem(batch_size, mean_a, sigma_a, mean_y, sigma_y)
    model.print_matrices()

    state = torch.randn((batch_size, system_order))
    input = torch.randn((batch_size, inputs_count))

    state_new, y = model(state, input, 0.01)
    
    print(state.shape, input.shape, state_new.shape, y.shape)


    
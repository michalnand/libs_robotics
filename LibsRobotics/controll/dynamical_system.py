import torch


class DynamicalSystem(torch.nn.Module):
    def __init__(self, inputs_count, outputs_count, system_order):
        super().__init__()
        
        self.a = torch.nn.Linear(system_order + inputs_count, system_order, bias=False)
        self.y = torch.nn.Linear(system_order, outputs_count, bias=False)
        

        print(inputs_count, outputs_count)

    def forward(self, state, input):
        x           = torch.cat([state, input], dim = 1)
        
        state_new   = state + self.a(x)*0.0001

        return state_new, self.y(state_new) 
    

    def print_matrices(self):
        print(self.a.weight.data)
        print(self.y.weight.data)
        print("\n")


    
import torch


class DynamicalSystem(torch.nn.Module):
    def __init__(self, inputs_count, outputs_count):
        super().__init__()
        
        self.a = torch.nn.Linear(inputs_count + outputs_count, outputs_count)
        

    def forward(self, state, input):
        x = torch.cat([state, input], dim = 1)
        return self.a(x) 
    

    def print_matrices(self):
        print(self.a.weight.data)
        print(self.a.bias.data)
        print("\n")


    
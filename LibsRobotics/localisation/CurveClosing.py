import torch


class Model(torch.nn.Module):

    def __init__(self, input_shape, output_shape, hidden_units = 16):
        super(Model, self).__init__()

        self.device = "cpu"
        
        self.gru        = torch.nn.GRU(input_size=2*input_shape[0], hidden_size=hidden_units, batch_first=True)
        self.linear     = torch.nn.Linear(hidden_units, output_shape[0])

        self.gru.to(self.device)
        self.linear.to(self.device)

        print(self.gru)
        print(self.linear)

    def forward(self, x):   
        x_in = torch.transpose(x, 0, 1)
        x_in = torch.transpose(x_in, 1, 2)
        x_in = x_in.reshape( (x_in.shape[0], x_in.shape[1], x_in.shape[2]*x_in.shape[3]) )

        gru_y, _    = self.gru(x_in)
        y           = self.linear(gru_y)

        return y

    


if __name__ == "__main__":
    batch_size  = 32
    seq_length  = 256
    input_shape = (7, )
    output_shape= (3, ) #dx, dy, dangle

    input   = torch.randn((2, batch_size, seq_length,) + input_shape)

    model   = Model(input_shape, (3, ))

    y       = model.forward(input)

    print(y.shape)
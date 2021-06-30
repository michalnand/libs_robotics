import dynamical_system
import torch
import numpy
from matplotlib import pyplot as plt


inputs_count    = 2
outputs_count   = 2
system_order    = 4

batch_size      = 512
sequence_length = 64

#create reference dynamical system - load data
reference_model = dynamical_system.DynamicalSystem(inputs_count, outputs_count, system_order)

#create model
trained_model   = dynamical_system.DynamicalSystem(inputs_count, outputs_count, system_order)

optimizer       = torch.optim.Adam(trained_model.parameters(), lr=0.1)


testing_input = torch.randn((sequence_length, 1, inputs_count))


def plot_testing(num):
    output_ref      = torch.zeros((sequence_length, 1, outputs_count))
    state_ref       = torch.zeros((1, system_order))

    output_trained  = torch.zeros((sequence_length, 1, outputs_count))
    state_trained   = torch.zeros((1, system_order))

    for t in range(sequence_length): 
        state_ref, y_ref    = reference_model.forward(state_ref, testing_input[t].detach())
        output_ref[t]       = y_ref

        state_trained, y_trained = trained_model.forward(state_trained, testing_input[t].detach())
        output_trained[t]        = y_trained
    
    output_ref_np       = output_ref.squeeze(1).squeeze(1).detach().to("cpu").numpy()
    output_trained_np   = output_trained.squeeze(1).squeeze(1).detach().to("cpu").numpy()

    output_ref_np       = numpy.transpose(output_ref_np)
    output_trained_np   = numpy.transpose(output_trained_np)

    print(output_ref_np.shape)
    print(output_trained_np.shape)

    plt.clf()
    plt.ion()
    plt.show()

    plt.plot(output_ref_np[0], output_ref_np[1])
    plt.plot(output_trained_np[0], output_trained_np[1])
    plt.draw()
    plt.pause(0.001)
    plt.savefig("images/frame_" + str(num) + ".png")

for epoch in range(100):

    input           = torch.randn((sequence_length, batch_size, inputs_count))
  
    output_ref      = torch.zeros((sequence_length, batch_size, outputs_count))
    state_ref       = torch.zeros((batch_size, system_order))

    output_trained  = torch.zeros((sequence_length, batch_size, outputs_count))
    state_trained   = torch.zeros((batch_size, system_order))

    for t in range(sequence_length): 
        state_ref, y_ref    = reference_model.forward(state_ref, input[t].detach())
        output_ref[t]       = y_ref

        state_trained, y_trained = trained_model.forward(state_trained, input[t].detach())
        output_trained[t]        = y_trained


    loss = (output_ref.detach() - output_trained)**2
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("loss = ", loss)

    reference_model.print_matrices()
    trained_model.print_matrices()
    print("\n\n\n\n")

    plot_testing(epoch)

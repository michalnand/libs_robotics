import dynamical_system
import torch

inputs_count    = 4
outputs_count   = 4

batch_size      = 128
sequence_length = 256

#create reference dynamical system - load data
reference_model = dynamical_system.DynamicalSystem(inputs_count, outputs_count)

#create model
trained_model   = dynamical_system.DynamicalSystem(inputs_count, outputs_count)

optimizer       = torch.optim.Adam(trained_model.parameters(), lr=0.02)

input           = torch.randn((sequence_length, batch_size, inputs_count))

for epoch in range(100):

    output_ref      = torch.randn((sequence_length, batch_size, outputs_count))
    state_ref       = torch.randn((batch_size, outputs_count))

    output_trained  = torch.randn((sequence_length, batch_size, outputs_count))
    state_trained   = torch.randn((batch_size, outputs_count))

    for t in range(sequence_length):
        state_ref           = reference_model.forward(state_ref, input[t].detach())
        output_ref[t]       = state_ref

        state_trained       = trained_model.forward(state_trained, input[t].detach())
        output_trained[t]   = state_trained


    loss = (output_ref.detach() - output_trained)**2
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("loss = ", loss)

    reference_model.print_matrices()
    trained_model.print_matrices()

    print("\n\n\n\n")
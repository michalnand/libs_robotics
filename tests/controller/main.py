import torch
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")

from LibsRobotics.controll.controller import *


inputs_count    = 7
outputs_count   = 5
internal_size   = 17

controller = LinearController(inputs_count, outputs_count, internal_size)
#controller = NonLinearController(inputs_count, outputs_count, internal_size)

batch_size = 19

x = torch.randn((batch_size, internal_size))

for i in range(10):
    u = torch.randn((batch_size, inputs_count))

    x, y = controller(x, u)
    
    print(y.shape, x.shape)
    print(y[0])


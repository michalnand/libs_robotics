import torch
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")

from LibsRobotics.controll.controller_train import *
from LibsRobotics.controll.testing_signal   import *


def loss_func(target_trajectory, output_trajectory, controller_output):

    target_trajectory   = target_trajectory[:,:,0]
    output_trajectory   = output_trajectory[:,:,1]

    diff  = target_trajectory - output_trajectory
    loss  = (diff**2).mean()
    
    
    r = 0.05*(torch.abs(target_trajectory) + 0.1)

    overshoot   = (torch.abs(output_trajectory) - torch.abs(target_trajectory)) > r
    overshoot   = overshoot.detach()

    loss+= (overshoot*(diff**2)).mean()
    
    return loss

'''
#spring mass harmonic oscilator
m       = 0.05   #weight mass [kg]
damping = 2.3
t       = 0.1   #period [s]
k       = (2.0*3.141592654*m/t)**2.0    #spring stiffness 
gain    = 3.7 
 
#state matrix A
mat_a   = torch.FloatTensor([[-damping, -k/m], [1.0, 0.0]])
sigma_a = torch.FloatTensor([[0.1,  0.1], [0.0, 0.0]])

#input matrix 
mat_b   = torch.FloatTensor([[gain/m], [0.0]])
sigma_b = torch.FloatTensor([[0.1  ], [0.0]])

#output matrix
mat_c   = torch.FloatTensor([[0.0, 1.0]])
sigma_c = torch.FloatTensor([[0.0, 0.0]])
'''

'''
#DC motor
#input parameters
v_nom       = 6.0                           #nominal voltage,       6[V]
friction    = 0.0                           #friction force,        [Nms]
w_nom       = 1000.0*2.0*torch.pi/60.0      #no load speed,         1000[rpm]
torque      = 0.57*0.09807                  #stall torque,          0.57[kg.cm]

wr          = 32.0*0.5/1000.0               #wheel diameter,        32[mm]
wm          = 100.0/1000.0                  #wheel + rotor mass,    1-0[grams]
#i_max       = 1.6                          #max stall current,     1.6[A]


k       = v_nom/w_nom       #motor constant, [Vs/rad]

i_max   = torque/k
r       = v_nom/i_max       #resistance [ohm]

#j       = 0.5*wm*(wr**2)    #inertia momentum
j       = wm*(wr**2)    #inertia momentum

a   = -((k**2)/r + friction)*(1.0/j)
b   = k/(r*j)


mat_a   = torch.FloatTensor([[a]])
sigma_a = torch.FloatTensor([[0.2]]) 

#input matrix
mat_b   = torch.FloatTensor([[b]])
sigma_b = torch.FloatTensor([[0.2]])

#output matrix
mat_c   = torch.FloatTensor([[1.0]])
sigma_c = torch.FloatTensor([[0.0]])
'''



#wheel position controll
#input parameters
v_nom       = 6.0                           #nominal voltage,       6[V]
friction    = 0.0                           #friction force,        [Nms]
w_nom       = 1000.0*2.0*torch.pi/60.0      #no load speed,         1000[rpm]
torque      = 0.57*0.09807                  #stall torque,          0.57[kg.cm]

wr          = 32.0*0.5/1000.0               #wheel diameter,        32[mm]
wm          = 10.0/1000.0                   #wheel + rotor mass,    10[grams]
#i_max       = 1.6                          #max stall current,     1.6[A]


k       = v_nom/w_nom       #motor constant, [Vs/rad]

i_max   = torque/k
r       = v_nom/i_max       #resistance [ohm]

#j       = 0.5*wm*(wr**2)    #inertia momentum
j       = wm*(wr**2)    #inertia momentum

a   = -((k**2)/r + friction)*(1.0/j)
b   = k/(r*j)

param_var = 0.02

mat_a   = torch.FloatTensor([[a, 0.0],      [1.0, 0.0]])
sigma_a = torch.FloatTensor([[param_var, 0.0],    [0.0, 0.0]]) 

#input matrix
mat_b   = torch.FloatTensor([[b],    [0.0]])
sigma_b = torch.FloatTensor([[param_var],  [0.0]])

#output matrix
mat_c   = torch.FloatTensor([[1.0, 0.0], [0.0, 1.0]])
sigma_c = torch.FloatTensor([[0.0, 0.0], [0.0, 0.0]])

#mat_c   = torch.FloatTensor([[0.0, 1.0]]) 
#sigma_c = torch.FloatTensor([[0.0, 0.0]])



batch_size          = 64
controller_order    = 64

steps_count         = 512
epoch_count         = 200


#create dynamical system
dynamical_system    = DynamicalSystem(batch_size, mat_a, sigma_a, mat_b, sigma_b, mat_c, sigma_c)


signal_generator    = UnitStepSignal(steps_count, mat_b.shape[1], period = 5, randomise=True, amp_max = 3.141592654)


trainer = ControllerTrain(batch_size, signal_generator, dynamical_system, controller_order, loss_func)



trainer.train(epoch_count, 1.0/200.0, "results_servo/")
 
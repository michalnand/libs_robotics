import torch
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")

from LibsRobotics.controll.lqr import *


#inverted pendulum
M = 0.5
m = 0.2
b = 0.1
I = 0.006
g = 9.8
l = 0.3

p = I*(M+m)+M*m*(l**2)

mat_a = [   [0,      1,              0,           0,],
            [0, -(I+m*(l**2))*b/p,  ((m**2)*g*(l**2))/p,   0],
            [0,      0,             0,           1],
            [0, -(m*l*b)/p,       m*g*l*(M+m)/p,  0] ]
            
mat_b = [   [ 0],
            [(I+m*(l**2))/p], 
            [0],
            [m*l/p] ]

mat_c = [   [ 1, 0, 0, 0],
            [0, 0, 1, 0] ]

mat_a = numpy.array(mat_a)
mat_b = numpy.array(mat_b)
mat_c = numpy.array(mat_c)


q       = numpy.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0] , [0.0, 0.0, 0.0, 0.0]])
r       = 1.0


k = lqr_solve(mat_a, mat_b, q, r)


print(k) # -1.0000   -1.6567   18.6854    3.4594

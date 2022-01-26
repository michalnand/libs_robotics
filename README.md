# libs_robotics
libs for robotics - localisation, mapping ...

## IMU

- complementary filter for sensor fusion
- low level driver [mpu6050](LibsDevices/mpu6050.h)
- IMU driver for sensor fussion [imu](LibsEmbedded/imu.h)
- firmware code [fw](tests/imu_avr_firmware)
- GUI PC app [app](tests/imu_visualisation)

![imu architecture](doc/imu/architecture.png)

![imu fusion](doc/imu/fusion.png)

![imu fusion](doc/imu/animation.gif)

## linear quadratic regulator in pytorch

- pytorch is used for controller optimisation
- linear dynamical system (described as state space)
- whole graph is differentiable, so torch can optimise whole problem
- Runge-Kutta is used for solving ODE's
- loss specifies controller behaviour

### block diagram of controll loop

![controller](doc/robust_controll/lqr.png)


### controlling servo

- 2nd order dynamics
- angular velocity  and wheel position 
- input is voltage into motor

### system dynamics
![servo_dynamics](doc/robust_controll/eqn_servo.gif)
 
### controller

![controller](doc/robust_controll/eqn_servo_controller.gif)





$$
\begin{pmatrix}
dx_0 \\
dx_1
\end{pmatrix} = 

\begin{pmatrix}
\alpha && 0 \\
1 && 0
\end{pmatrix} 

\begin{pmatrix}
x_0 \\
x_1 
\end{pmatrix} 

+ 

\begin{pmatrix}
\beta \\
0 
\end{pmatrix} 

\begin{pmatrix}
u
\end{pmatrix} 
\\
\begin{pmatrix}
\omega \\
\theta
\end{pmatrix} = 

\begin{pmatrix}
1 && 0 \\
0 && 1
\end{pmatrix} 

\begin{pmatrix}
x_0 \\
x_1 
\end{pmatrix} 
$$

$$
\begin{pmatrix}
u 
\end{pmatrix} = 

\begin{pmatrix}
k_0 && k_1 && k_2 && k_3
\end{pmatrix} 

\begin{pmatrix}
\omega_{req} \\
\theta_{req} \\
\omega_{out} \\
\theta_{out} \\
\end{pmatrix} 
$$

![controller](doc/robust_controll/servo_control.gif)

## robust controll in pytorch

- pytorch is used for controller optimisation
- linear dynamical system (described as state space)
- controller can be linear or non-liear (GRU, LSTM ...)
- whole graph is differentiable, so torch can optimise whole problem
- Runge-Kutta is used for solving ODE's
- loss specifies controller behaviour

### computational graph overview (loss ommitted)
![controller](doc/robust_controll/overview.png)


### linear controller is just 3 matrix multiplications
[code](LibsRobotics/controll/controller.py)
![controller](doc/robust_controll/linearcontrollerdesc.png)
![controller](doc/robust_controll/linearcontroller.png)

### servo controll with 20% parameters uncertaininty, using non-linear 64 units GRU
![controller](doc/robust_controll/servo_controll_gru.gif)




## particle filter

estimate position for local measurement, known map and relative position change

<img src="doc/particle_filter.gif" width="800">


```python
#map size
height      = 150
width       = 150

particles_count = 512

#create same random grid map, numpy array (height, width, features)
map = LibsRobotics.MapGrid(height, width, 1)
map.random()

#create particle filter
partcile_filter = LibsRobotics.ParticleFilter(map.get_map(), particles_count)


#main loop
while True:
    dx, dy      = ... #obtain robot dx, dy

    observation = ... #obtain environment sensor observation (features)

    #compute particle filter
    partcile_filter.step(observation, dx, dy)

    #output
    ... = partcile_filter.position_x
    ... = partcile_filter.position_y
```


## points cloud matching

find transformation to match two point clouds
(scale, offset and rotation)

<img src="doc/points_matching.gif" width="800">

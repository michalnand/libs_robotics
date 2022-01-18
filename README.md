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

# libs_robotics
libs for robotics - localisation, mapping ...


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
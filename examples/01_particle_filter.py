import numpy
import cv2
import LibsRobotics

#fourcc = cv2.VideoWriter_fourcc(*'XVID') 
#writer = cv2.VideoWriter("particle_filter.avi", fourcc, 25.0, (512, 512)) 


def render(map, robot_x, robot_y, estimated_x, estimated_y, particles_x, particles_y):
    height  = map.shape[0]
    width   = map.shape[1]
    particles_count = len(particles_x)

    image = numpy.zeros((3, height, width))
    image[0] = 0.3*map.copy()[:,:,0]
    image[1] = 0.3*map.copy()[:,:,0]
    image[2] = 0.3*map.copy()[:,:,0]


    for p in range(particles_count):
        py = int(particles_y[p]*height)
        px = int(particles_x[p]*width)

        image[0][py][px] = 0.0
        image[1][py][px] = 0.2
        image[2][py][px] = 0.0

    image[0][robot_y][robot_x] = 0.0
    image[1][robot_y][robot_x] = 0.0
    image[2][robot_y][robot_x] = 1.0

    pos_y = int(estimated_y*height)
    pos_x = int(estimated_x*width)

    image[0][pos_y][pos_x] = 1.0
    image[1][pos_y][pos_x] = 1.0
    image[2][pos_y][pos_x] = 1.0

    image = numpy.swapaxes(image, 0, 2)

    image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_NEAREST)
    
    window_name = "particle filter"
    
    cv2.imshow(window_name, image) 
    cv2.waitKey(1)

    #writer.write((image*255).astype(numpy.uint8) )



height      = 150
width       = 150

particles_count = 512

map = LibsRobotics.MapGrid(height, width)
map.random()

print(map.get_map()[0, 0])


partcile_filter = LibsRobotics.ParticleFilter(map.get_map(), particles_count)


dx = 0.0
dy = 0.0

x = 0.2
y = 0.2

way     = 0
d_pos   = 0.01

while True:
    #square-like robot move
    if way == 0:
        dx = d_pos
        dy = 0.0
        x+= dx
        if x >= 0.8:
            way = 1
    elif way == 1:
        dx = 0.0
        dy = d_pos
        y+= dy
        if y >= 0.8:
            way = 2
    elif way == 2:
        dx = -d_pos
        dy = 0.0
        x+= dx
        if x <= 0.2:
            way = 3
    elif way == 3:
        dx = 0.0
        dy = -d_pos
        y+= dy
        if y <= 0.2:
            way = 0

    
    rx = int(round(x*(width-1)))
    ry = int(round(y*(height-1)))

    observation = map.map[ry][rx]

    partcile_filter.step(observation, dx, dy)

    render( map.get_map(), 
            rx, ry, 
            partcile_filter.position_x, partcile_filter.position_y, 
            partcile_filter.particles_x, partcile_filter.particles_y)


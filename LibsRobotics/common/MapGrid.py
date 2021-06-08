import numpy


class MapGrid:

    def __init__(self, height, width, channels = 1):
        self.height      = height
        self.width       = width
        self.map         = numpy.zeros((height, width, channels))

    def get_map(self):
        return self.map

    def random(self):
        count = (self.height+self.width)
        
        for _ in range(count):
            y0 = numpy.random.randint(0, self.width)
            x0 = numpy.random.randint(0, self.height)
            y1 = numpy.random.randint(y0, self.width)
            x1 = numpy.random.randint(x0, self.height)

            value = numpy.random.randint(0, 2)

            for y in range(y1 - y0):
                for x in range(x1 - x0):
                    self.map[y + y0][x + x0][0] = value
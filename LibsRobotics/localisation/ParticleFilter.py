from os import WEXITED
from typing import overload
import numpy


class ParticleFilter:

    def __init__(self, map, particles_count = 1024):

        self.map                = map
        self.particles_count    = particles_count

        self.particles_x        = numpy.random.uniform(0.0, 1.0, particles_count)
        self.particles_y        = numpy.random.uniform(0.0, 1.0, particles_count)

        self.weights            = numpy.ones(particles_count)

        self.position_x = 0.5
        self.position_y = 0.5

    def step(self, observation, dx, dy):
        self._move_particles(dx, dy)

        weights = self._particles_weights(observation)
        
        k = 0.05
        self.weights    = (1.0 - k)*self.weights + k*weights


        self.position_x = ((self.particles_x*self.weights).sum())/self.weights.sum()
        self.position_y = ((self.particles_y*self.weights).sum())/self.weights.sum()

        self.particles_x, self.particles_y = self._resample(weights)

        
    
    def _move_particles(self, dx, dy):
        self.particles_x        = numpy.clip(self.particles_x + dx,     0.0, 0.9999)
        self.particles_y        = numpy.clip(self.particles_y + dy,     0.0, 0.9999)


    def _particles_weights(self, observation):
        x       = (self.particles_x*self.map.shape[1]).astype(int)
        y       = (self.particles_y*self.map.shape[0]).astype(int)

        x       = numpy.clip(x, 0, self.map.shape[1] - 1)
        y       = numpy.clip(y, 0, self.map.shape[0] - 1)

        m       = self.map[y,x]

        dist    = ((observation - m)**2).mean(axis=1)
        dist    = dist - dist.max()

        weights  = numpy.exp(-dist)
       
        return weights


    def _resample(self, weights):
        
        indices = numpy.random.choice(self.particles_count, self.particles_count, p=weights/weights.sum())

        noise_x = 0.005*numpy.random.randn(self.particles_count)
        noise_y = 0.005*numpy.random.randn(self.particles_count)

        particles_x = numpy.take(self.particles_x, indices)
        particles_y = numpy.take(self.particles_y, indices)

        particles_x = numpy.clip(particles_x + noise_x, 0, 0.9999)
        particles_y = numpy.clip(particles_y + noise_y, 0, 0.9999)
        
        return particles_x, particles_y 

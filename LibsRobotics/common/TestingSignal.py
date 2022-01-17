import numpy

class TestingSignal:
    def __init__(self, samples_count, channels_count, components_count = 5):
        self.samples_count      = samples_count
        self.channels_count     = channels_count
        self.components_count   = components_count


    def sample_batch(self, batch_size = 32, batch_first = False):
        result  = numpy.zeros((self.samples_count, batch_size, self.channels_count), dtype=numpy.float32)

        weights = numpy.random.randn(self.components_count, batch_size, self.channels_count)
        weights = weights/weights.sum(axis=0)

        mode    = numpy.random.randint(0, 2)

        mode = 0

        for j in range(self.components_count):

            if mode == 0:
                periods  = numpy.random.randint(1 + self.samples_count//10, self.samples_count, (batch_size, self.channels_count))
                phases   = numpy.random.randint(0, self.samples_count, (batch_size, self.channels_count))
                signal   = self._square(batch_size, self.samples_count, periods, phases)
            elif mode == 1:
                freqs    = numpy.random.randint(1, self.samples_count//32, (batch_size, self.channels_count))*(2.0*numpy.pi)/self.samples_count
                phases   = 2.0*numpy.pi*numpy.random.rand()
                signal   = self._sine(batch_size, self.samples_count, freqs, phases)
            elif mode == 2:
                means    = 2.0*numpy.random.rand() - 1.0
                sigmans  = 2.0*numpy.random.rand() + 0.01
                signal   = self._noise(batch_size, self.samples_count, means, sigmans)

            result+= weights[j]*signal
        
        r = numpy.max(numpy.abs(result), axis=2).reshape((self.samples_count, batch_size, 1))

        result = result/r

        if batch_first:
            result = numpy.swapaxes(result, 0, 1)

        return result

    def _square(self, batch_size, samples_count, periods, phases):
        result = numpy.zeros((samples_count, batch_size, self.channels_count), dtype=numpy.float32)

        for n in range(samples_count):
            tmp         = (n + phases)%periods > periods/2
            result[n]   = 2.0*tmp - 1.0


        return result

    def _sine(self, batch_size, samples_count, freqs, phases):
        result = numpy.zeros((samples_count, batch_size, self.channels_count), dtype=numpy.float32)

        for n in range(samples_count):
            result[n] = numpy.sin(n*freqs + phases)

        return result

    def _noise(self, batch_size, samples_count, mean, sigma):
        result = numpy.zeros((samples_count, batch_size, self.channels_count), dtype=numpy.float32)

        for n in range(samples_count):
            result[n] = mean + sigma*numpy.random.randn()

        return result


class SquareSignal:
    def __init__(self, samples_count, channels_count, period = 5, randomise = False):
        self.samples_count      = samples_count
        self.channels_count     = channels_count
        self.period             = period
        self.randomise          = randomise


    def sample_batch(self, batch_size = 32, batch_first = False):
        
        if self.randomise:
            periods  = numpy.random.randint(1 + self.samples_count//10, self.samples_count, (batch_size, self.channels_count))
            phases   = numpy.random.randint(0, self.samples_count, (batch_size, self.channels_count))
        else:   
            periods  = self.period*numpy.ones((batch_size, self.channels_count))
            phases   = numpy.zeros((batch_size, self.channels_count))

        result = self._square(batch_size, self.samples_count, periods, phases)
           
        if batch_first:
            result = numpy.swapaxes(result, 0, 1)

        return result

    def _square(self, batch_size, samples_count, periods, phases):
        result = numpy.zeros((samples_count, batch_size, self.channels_count), dtype=numpy.float32)

        for n in range(samples_count):
            tmp         = (n + phases)%periods > periods/2
            result[n]   = 2.0*tmp - 1.0

        return result

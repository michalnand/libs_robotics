import numpy



class SignalUnitStep:
    def __init__(self, samples_count, channels_count, period = 5, randomise = False, amplitudes = [1.0]):
        self.samples_count      = samples_count
        self.channels_count     = channels_count
        self.period             = period
        self.randomise          = randomise
        self.amplitudes         = numpy.array(amplitudes).reshape((1, len(amplitudes)))


    def sample_batch(self, batch_size = 32, batch_first = False):
        if self.randomise:
            amplitudes  = self.amplitudes*(2.0*numpy.random.rand(batch_size, self.channels_count) - 1.0)
            phases      = numpy.random.randint(0, int(self.samples_count*0.8), (batch_size, self.channels_count))
        else:
            amplitudes  = self.amplitudes*numpy.ones((batch_size, self.channels_count))
            phases      = numpy.zeros((batch_size, self.channels_count))

        result = self._square(batch_size, self.samples_count, amplitudes, phases)
           
        if batch_first:
            result = numpy.swapaxes(result, 0, 1)

        return result

    def _square(self, batch_size, samples_count, amplitudes, phases):
        result = numpy.zeros((samples_count, batch_size, self.channels_count), dtype=numpy.float32)

        for n in range(samples_count):
            result[n]   = amplitudes*(n > phases)

        return result


class SignalGaussianNoise:
    def __init__(self, samples_count, channels_count, amplitudes):
        self.samples_count      = samples_count
        self.channels_count     = channels_count
        self.amplitudes         = numpy.array(amplitudes).reshape((1, len(amplitudes)))


    def sample_batch(self, batch_size = 32, batch_first = False): 
        result = self.amplitudes*numpy.random.randn(self.samples_count, batch_size, self.channels_count)
           
        if batch_first:
            result = numpy.swapaxes(result, 0, 1)

        return result


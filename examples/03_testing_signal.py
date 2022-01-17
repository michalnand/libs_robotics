import numpy
import LibsRobotics
import matplotlib.pyplot as plt

samples_count       = 1024
channels_count      = 3
components_count    = 5

batch_size          = 32

#testing_signal      = LibsRobotics.TestingSignal(samples_count, channels_count, components_count)
testing_signal      = LibsRobotics.SquareSignal(samples_count, channels_count, period=100, randomise=False)

samples = testing_signal.sample_batch(batch_size)

print("shape = ", samples.shape)

plt.ylabel('value')
plt.xlabel('sample')
plt.plot(samples[:,0,0], color='deepskyblue')

plt.show()
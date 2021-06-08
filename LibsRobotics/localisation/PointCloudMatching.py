import numpy
import torch


class PointCloudMatchingModel(torch.nn.Module):

    def __init__(self):
        super(PointCloudMatchingModel, self).__init__()

        self.alpha    = torch.nn.Parameter(torch.zeros((), device="cpu", requires_grad=True))
        self.scale    = torch.nn.Parameter(torch.ones((2 ), device="cpu", requires_grad=True))
        self.offset   = torch.nn.Parameter(torch.zeros((2 ), device="cpu", requires_grad=True))

    #points shape : (points, 2)
    def forward(self, points):

        x = torch.transpose(points, 0, 1)

        x_new = x[0]*torch.cos(self.alpha) - x[1]*torch.sin(self.alpha)
        y_new = x[0]*torch.sin(self.alpha) + x[1]*torch.cos(self.alpha)

        x_new = x_new*self.scale[0] + self.offset[0]
        y_new = y_new*self.scale[1] + self.offset[1]

        return torch.stack([x_new, y_new], dim=1)



class PointCloudMatching:

    def __init__(self):
        self.model      = PointCloudMatchingModel()
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def step(self, points_a, points_b):

        points_a_t = torch.from_numpy(points_a)
        points_b_t = torch.from_numpy(points_b)

        points_b_prediction = self.model(points_a_t)

        self.optimizer.zero_grad()

        loss = self._compute_loss(points_b_t, points_b_prediction)
        loss.backward()
        self.optimizer.step()

        return points_b_prediction.detach().to("cpu").numpy()

    def _compute_loss(self, points_ref, points_pred):
        distances = torch.cdist(points_ref, points_pred, p=2)**2

        dist_min  = torch.min(distances, dim=1)[0]
        return dist_min.mean()

from matplotlib import pyplot as plt, scale

def random_transform(points):
    x = numpy.transpose(points)

    alpha  = 0.1*numpy.random.randn()
    scale  = 1.0 + numpy.random.rand(2)
    offset = 2.0*numpy.random.randn(2)

    x_new = x[0]*numpy.cos(alpha) - x[1]*numpy.sin(alpha)
    y_new = x[0]*numpy.sin(alpha) + x[1]*numpy.cos(alpha)

    x_new = x_new*scale[0] + offset[0]
    y_new = y_new*scale[1] + offset[1]

    return numpy.stack([x_new, y_new], axis=1)




if __name__ == "__main__":

    pcm = PointCloudMatching()

    points_a = numpy.random.randn(64, 2)
    points_b = random_transform(points_a)

    plt.ion()
    plt.show()

    for i in range(1000):
        points_b_prediction = pcm.step(points_a, points_b)

        pb_target       = numpy.transpose(points_b)
        pb_prediction   = numpy.transpose(points_b_prediction)

        if i%10 == 0:
            plt.clf()
            plt.plot(pb_target[0], pb_target[1], 'o', color='skyblue')
            plt.plot(pb_prediction[0], pb_prediction[1], 'o', color='red')
            plt.draw()
            plt.savefig("./images/" + str(i) + ".png")
            plt.pause(0.001)

        

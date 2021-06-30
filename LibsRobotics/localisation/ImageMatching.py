import torch
import numpy

class Model(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fc_inputs_count = (input_shape[1]//32)*(input_shape[2]//32)*128*2

        self.layers_features = [
            self.conv_bn(input_shape[0], 32, 2),
            
            self.conv_bn(32, 64, 1),
            self.conv_bn(64, 64, 2),

            self.conv_bn(64, 128, 1),
            self.conv_bn(128, 128, 2),

            self.conv_bn(128, 128, 1),
            self.conv_bn(128, 128, 2),

            self.conv_bn(128, 128, 1),
            self.conv_bn(128, 128, 2),
            
            torch.nn.Flatten()
        ]

        self.layers_output = [
            torch.nn.Linear(fc_inputs_count, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_shape[0])
        ]

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_features[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.layers_features[i].bias)

        for i in range(len(self.layers_output)):
            if hasattr(self.layers_output[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_output[i].weight)
                torch.nn.init.zeros_(self.layers_output[i].bias)


        self.model_features = torch.nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_output = torch.nn.Sequential(*self.layers_output)
        self.model_output.to(self.device)

        print(self.model_features)
        print(self.model_output)

    def forward(self, x):
        xt          = torch.transpose(x, 0, 1)

        features_0  = self.model_features(xt[0])
        features_1  = self.model_features(xt[1])

        features    = torch.cat([features_0, features_1], dim=1)
        
        return self.model_output(features)

    def eval_features(self, x):
        return self.model_features(x)

    def eval_output(self, features_0, features_1):
        features    = torch.cat([features_0, features_1], dim=1)
        return self.model_output(features)

    def save(self, path):
        torch.save(self.model_features.state_dict(), path + "model_features.pt") 
        torch.save(self.model_output.state_dict(), path + "model_output.pt") 

    def load(self, path):
        self.model_features.load_state_dict(torch.load(path + "model_features.pt", map_location = self.device))
        self.model_output.load_state_dict(torch.load(path + "model_output.pt", map_location = self.device))
        
        self.model_features.eval() 
        self.model_output.eval() 

    def conv_bn(self, inputs, outputs, stride):
        return torch.nn.Sequential(
                torch.nn.Conv2d(inputs, outputs, kernel_size = 3, stride = stride, padding = 1),
                torch.nn.BatchNorm2d(outputs),
                torch.nn.ReLU(inplace=True))


class ImageMatching:

    def __init__(self):
        self.features   = 4
        self.model      = Model((3, 256, 256), (self.features, ))


    def run(self, images):
        shape = images.shape
        images_t    = torch.from_numpy(images).float().to(self.model.device)

        features_t  = self.model.eval_features(images_t)


        result_t    = torch.zeros((shape[0], shape[0], self.features)).to(self.model.device)

        for i in range(shape[0]):

            features_repeat_t = features_t[i].unsqueeze(0).repeat((shape[0], 1))
            y = self.model.eval_output(features_repeat_t, features_t)

            result_t[i] = y

        return result_t.detach().to("cpu").numpy()


if __name__ == "__main__":
    im = ImageMatching()

    images = numpy.random.randn(16, 3, 256, 256)

    y = im.run(images)

    print(y.shape)

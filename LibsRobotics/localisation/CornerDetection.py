import torch
import numpy
import sys
from PIL import Image

def make_fast_kernel(size = 7, count = 16, count_k=12):
    result = numpy.zeros((count, 1, size, size))

    for k in range(count):
        phase  = 2.0*numpy.pi*k/(count)
        center = size/2
        radius = size/2 

        for i in range(count_k):
            phi = 2.0*numpy.pi*i/(count)
           
            y = center + radius*numpy.cos(phi + phase)
            x = center + radius*numpy.sin(phi + phase)

            y = numpy.clip(y, 0, size-1)
            x = numpy.clip(x, 0, size-1)

            result[k][0][int(y)][int(x)] = 1


    print(result)

    return result

if __name__ == "__main__":
    numpy.set_printoptions(threshold=sys.maxsize)
    fast_kernel = make_fast_kernel()

    
    gaussian_kernel = [ 
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ]

    gaussian_smoothing = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2, stride=1)
    gaussian_smoothing.weight.data[:,:] = torch.tensor(gaussian_kernel)/(3*273.0)
    gaussian_smoothing.bias.data[:]     = 0.0

    fast_detection = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, padding=3)
    fast_detection.weight.data = torch.tensor(fast_kernel).float()/12.0
    fast_detection.bias.data[:]     = 0.0
   
    img_orig = Image.open("image.jpg")
    image    = numpy.array(img_orig.convert("L"))
    img_rgb  = numpy.array(img_orig)

    image_t = torch.from_numpy(image)/256.0
    image_t = image_t.unsqueeze(0)
 

    x = image_t.unsqueeze(0)
    y0 = gaussian_smoothing(x)
    y1 = fast_detection(y0)

    threshold = 0.01

    dif = torch.abs(y0 - y1)

    y_res = 1.0*((dif > threshold).sum(dim=1) > 0.0)

    print(y_res.shape)

    
    y_np = y_res[0].detach().to("cpu").numpy()
    y_np = numpy.clip(y_np, 0, 1)*255

    img_res = img_rgb
    img_res[:,:,0] = img_res[:,:,0] + y_np

    print(">>> ", img_rgb.shape, y_np.shape, img_res.shape)

    image = Image.fromarray(img_res)
    image.show()

    
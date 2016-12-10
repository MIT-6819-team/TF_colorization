import numpy as np
from PIL import Image
from skimage import color
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
import time

quantized_array = np.load('pts_in_hull.npy')

def get_colorized_image( image, prediction ,use_skiimage_conversion=False):
    print "Get colorized image", use_skiimage_conversion
    T = 0.38
    epsilon = 1e-8

    annealed_mean = np.exp( np.log(prediction + epsilon) / T )
    annealed_mean /= np.sum(annealed_mean, axis = 2).reshape((256,256,1))

    predicted_coloring = np.dot(annealed_mean, quantized_array)
    colorized_image = np.zeros( (256,256,3) )
    colorized_image[:,:,0:1] = image
    colorized_image[:,:,1:] = predicted_coloring

    if use_skiimage_conversion:
        rgb_image = 255 * color.lab2rgb(colorized_image)
    else:
        rgb_image = 255 * _lab_to_rgb(colorized_image)

    return Image.fromarray(rgb_image.astype(np.uint8) )

def _lab_to_rgb(lab_image):
    print "Start lab to rgb. This may take a while."
    lt = time.time()
    rgb_image = np.zeros([256, 256, 3])

    for i in xrange(256**2):
        x = i/256
        y = i%256
        lab = lab_image[x][y]
        rgb = convert_color(LabColor(*lab), sRGBColor)
        rgb_image[x][y] = np.array([rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b])

    print "end", (time.time() - lt)

    return rgb_image

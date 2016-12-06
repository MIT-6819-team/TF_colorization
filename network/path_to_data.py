import numpy as np
import scipy.misc

from skimage import color, io

ab_to_dist = {}


def image_path_to_image_and_distribution_tensor(path):
    '''Converts an image path to a LAB image and a [64, 64, 313] tensor of color distribution values.'''
    img = io.imread(path)
    img = scipy.misc.imresize(img, (64, 64))
    img = color.rgb2lab(img)

    assert img.shape == (64, 64, 3)

    img = img[:, :, 1:3]
    dist = np.zeros([64, 64, 313])

    h, w, _ = dist.shape
    for x in range(w):
        for y in range(h):
            dist[x][y] = _map_ab_to_distribution(tuple(np.floor(img[x][y]).tolist()))

    return img, dist


def _gaussian(x, var):
    '''Gaussian on np array'''
    return np.exp(-(x**2) / (2 * var**2))


def _precompute_distributions():
    '''Precomputes the distribution we want for each integer a,b value.'''
    global ab_to_dist
    print "Precomputing distributions... will take a second"

    quantized_array = np.load('/afs/csail.mit.edu/u/k/kocabey/TF_colorization/network/pts_in_hull.npy')

    for a in range(-120, 120):
        for b in range(-120, 120):
            tiled_ab = np.tile([a, b], (313, 1))

            distances = np.linalg.norm(quantized_array - tiled_ab, axis=1)
            d = distances.copy()
            d.sort()

            low_values = (distances > np.tile(d[4], (313)))
            gaussian_distances = _gaussian(distances, 5)
            gaussian_distances[low_values] = 0

            dist = gaussian_distances / np.sum(gaussian_distances)

            ab_to_dist[(a, b)] = dist
    print "Done"


def _map_ab_to_distribution(ab):
    '''Map an integer (a,b) tuple to a 313 deep distribution.'''
    if len(ab_to_dist) == 0:
        _precompute_distributions()

    return ab_to_dist[ab]
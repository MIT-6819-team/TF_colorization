import ujson, gzip
import collections
import random
import io
import math
import numpy as np
import tensorflow as tf
from skimage import color, io

# Parameters from the paper
LAMBDA = 0.5
SIGMA = 5

# Parameters for loading
PERCENT_DATASET_TO_CHECK = 0.01
QUANITZATION_SIZE = 10
SATURATION_LIMIT = .1
DATASET_ROOT = '/root/persistant_data/datasets/imagenet/train256/'

quantized_regions = np.load('pts_in_hull.npy')

counts = collections.Counter()
greyscale = 0

def get_data(path):
    global counts, greyscale
    raw_image = io.imread(path)
    assert raw_image.shape == (256,256,3)

    img = color.rgb2lab(raw_image)

    image = img[:,:,1:3]
    quantized_image = np.round(image/10)*10

    for i, region in enumerate(quantized_regions):
        counts[i]+= np.count_nonzero(quantized_image == region)

def compute_counts():
    f = gzip.open('../../dataset_indexes/imagenet_train_256_saturation_values.json.gz', 'rt')
    files = ujson.load(f)
    random_files = files.keys()
    random.shuffle(random_files)

    for i, path in enumerate(random_files):
        if files[path] >= SATURATION_LIMIT:
            get_data(DATASET_ROOT + path.strip('\n'))

        if i%1000 == 0:
            print (i)

        if i >= len(files) * PERCENT_DATASET_TO_CHECK:
            break

    reweighted_counts = _apply_reweighting(counts)
    quantized_counts_as_array = list(reweighted_counts.values())
    np.save('quantized_counts.npy', quantized_counts_as_array)
    print (quantized_counts_as_array)

def _apply_reweighting(counts):
    blurred_counts = []
    for i in xrange(313):
        for j in xrange(313):
            blurred_counts[j] += _gaussian(_distance_of_indicies(i,j)) * counts[i]
    blurred_counts /= sum(blurred_counts)

    combined_counts = [ 1 / ((1 - LAMBDA) * blurred_counts[i] + LAMBDA / 313.) for i in xrange(313)]
    combined_counts /= sum(combined_counts)

    return combined_counts

def _distance_of_indicies(i, j):
    a = quantized_regions[i]
    b = quantized_regions[j]
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

compute_counts()

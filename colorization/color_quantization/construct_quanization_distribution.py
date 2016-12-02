import ujson, gzip
import collections
import random
import io
import numpy as np
import tensorflow as tf
from skimage import color, io

PERCENT_DATASET_TO_CHECK = 0.001
QUANITZATION_SIZE = 10
SATURATION_LIMIT = .1
DATASET_ROOT = '/root/persistant_value/datasets/'

counts = collections.Counter()
greyscale = 0

def get_data(path):
  global counts, greyscale
  raw_image = io.imread(path)
  assert raw_image.shape == (256,256,3):
    
  img = color.rgb2lab(raw_image)

  image = img[:,:,1:3]
  quantized_image = np.round(image/10)*10
  
  for i, region in enumerate(quantized_regions):
    counts[i]+= np.count_nonzero(quantized_image == region)

def compute_counts():
  f = gzip.open('../../imagenet_train_256_saturation_values.json.gz', 'rt')
  files = random.shuffle(ujson.load(f))

  for i, (path, saturation) in enumerate(f):
    get_data(DATASET_ROOT + path.strip('\n'))

    if i%1000 == 0:
      print i

    if i >= len(files) * PERCENT_DATASET_TO_CHECK:
      break

  quantized_counts_as_array = list(counts.values())
  np.save('quantized_counts.npy', quantized_counts_as_array)
  print quantized_counts_as_array
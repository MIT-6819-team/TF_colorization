import ujson
import gzip
import numpy as np
import random
import time
from path_to_data import image_path_to_image_and_distribution_tensor


class DataLoader(object):
    SATURATION_THRESHOLD = 0.1
    OUTPUT_IMAGE_SIZE = 64
    INPUT_IMAGE_SIZE = 256


    def __init__(self, batch_size, use_imagenet=True):
	self.batch_size = batch_size
        self._load_paths_and_threshold(use_imagenet)
        self.current_datapoint_index = 0

	if use_imagenet:
	    self.root = '/data/vision/torralba/yusuf/imagenet/data/images/train256/'
	else:
	    print "Don't know places root"

    def next_batch(self, batch_size):
        x_batch = np.zeros((batch_size, self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, 1))
        y__batch = np.zeros((batch_size, self.OUTPUT_IMAGE_SIZE, self.OUTPUT_IMAGE_SIZE, 313))

        for i in range(batch_size):
            path = self.all_paths[self.current_datapoint_index]
            x, y_ = image_path_to_image_and_distribution_tensor(self.root + path)

            x_batch[i, ...] = x.reshape((256, 256, 1))
            y__batch[i, ...] = y_

            self.current_datapoint_index += 1
            if self.current_datapoint_index >= len(self.all_paths):
                self.current_datapoint_index = 0

        return x_batch, y__batch

    def get_test_batch(self):
        lt = time.time()
        x_batch = np.zeros((len(self.test_batch), self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, 1))
        y__batch = np.zeros((len(self.test_batch), self.OUTPUT_IMAGE_SIZE, self.OUTPUT_IMAGE_SIZE, 313))

        for i in range(len(self.test_batch)):
            path = self.test_batch[i]
            x, y_ = image_path_to_image_and_distribution_tensor(self.root + path)

            x_batch[i, ...] = x.reshape((256, 256, 1))
            y__batch[i, ...] = y_

        print "Batch loading took ", (time.time() - lt)

        return x_batch, y__batch

    def _load_paths_and_threshold(self, use_imagenet):
        '''Loads all the paths and removes those below the saturation threshold.'''
        source = 'imagenet_train_256_saturation_values.json.gz' if use_imagenet else 'places_2_256_training_saturation_index.json.gz'
        f = ujson.load(gzip.open('../dataset_indexes/' + source, 'rt'))
        self.all_paths = [path for path in f.keys() if f[path] > self.SATURATION_THRESHOLD]

        self.test_batch = self.all_paths[:self.batch_size]

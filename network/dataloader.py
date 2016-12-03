import ujson
import gzip
import numpy as np
from path_to_data import image_path_to_image_and_distribution_tensor


class DataLoader(object):
    SATURATION_THRESHOLD = 0.1
    INPUT_IMAGE_SIZE = 256

    def __init__(self, use_imagenet=True):
        self._load_paths_and_threshold(use_imagenet)
        self.current_datapoint_index = 0

    def next_batch(self, batch_size):
        x_batch = np.zeros((batch_size, self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, 1))
        y__batch = np.zeros((batch_size, self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, 313))

        for i in range(batch_size):
            path = self.all_paths[self.current_datapoint_index]
            x, y_ = image_path_to_image_and_distribution_tensor(path)

            x_batch[i, ...] = x
            y__batch[i, ...] = y_

            self.current_datapoint_index += 1
            if self.current_datapoint_index >= len(self.all_datapoints):
                self.current_datapoint_index = 0

        return x_batch, y__batch

    def _load_paths_and_threshold(self, use_imagenet):
        '''Loads all the paths and removes those below the saturation threshold.'''
        source = 'imagenet_train_256_saturation_values.json' if use_imagenet else 'places_2_256_training_saturation_index.json'
        f = ujson.load(gzip.open('../dataset_indexes/' + source, 'rt'))
        self.all_paths = [path for path in f.keys() if f[path] > self.SATURATION_THRESHOLD]

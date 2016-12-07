import ujson
import gzip
import numpy as np
import random
import threading
import time
from path_to_data import image_path_to_image_and_distribution_tensor


class DataLoader(object):
    SATURATION_THRESHOLD = 0.1
    OUTPUT_IMAGE_SIZE = 64
    INPUT_IMAGE_SIZE = 256

    def __init__(self, batch_size, use_imagenet=True, use_winter=True):
        self.batch_size = batch_size
        self._load_paths_and_threshold(use_imagenet)

        self.training_batches = []
        self.batches_available = threading.Semaphore(0)
        self.DESIRED_QUEUED_BATCHES = 3

    	if use_imagenet:
            if use_winter:
    	        self.root = '/root/persistant_data/datasets/imagenet/train256/'
            else:
                self.root = '/data/vision/torralba/yusuf/imagenet/data/images/train256/'
    	else:
    	    print "Don't know places root"

    def next_batch(self):
        """Gets the next batch from the dataset and starts loading others in parallel."""
        # Make sure that we always have enough batches precomputed
        for b in xrange(self.DESIRED_QUEUED_BATCHES - len(self.training_batches)):
          threading.Thread(target = self._load_batch).start()

        self.batches_available.acquire()  # Wait for a new batch
        data_x, data_y_ = self.training_batches[0]
        del self.training_batches[0]

        return data_x, data_y_

    def get_test_batch(self):
        x_batch = np.zeros((len(self.test_batch), self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, 1))
        y__batch = np.zeros((len(self.test_batch), self.OUTPUT_IMAGE_SIZE, self.OUTPUT_IMAGE_SIZE, 313))

        for i in range(len(self.test_batch)):
            path = self.test_batch[i]
            x, y_ = image_path_to_image_and_distribution_tensor(self.root + path)

            x_batch[i, ...] = x.reshape((256, 256, 1))
            y__batch[i, ...] = y_

        return x_batch, y__batch

    def _load_batch(self):
        """Load the next batch, queue it, and increase the semaphore."""
        lt = time.time()
        x_batch = np.zeros((self.batch_size, self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, 1))
        y__batch = np.zeros((self.batch_size, self.OUTPUT_IMAGE_SIZE, self.OUTPUT_IMAGE_SIZE, 313))

        for i in range(self.batch_size):
          path = self.all_paths[int(random.random() * len(self.all_paths))]
          x, y_ = image_path_to_image_and_distribution_tensor(self.root + path)

          x_batch[i, ...] = x.reshape((256, 256, 1))
          y__batch[i, ...] = y_

        self.training_batches.append((x_batch, y__batch))
        self.batches_available.release()
        print "Batch loaded in parallel ", (time.time() - lt)

    def _load_paths_and_threshold(self, use_imagenet):
        '''Loads all the paths and removes those below the saturation threshold.'''
        source = 'imagenet_train_256_saturation_values.json.gz' if use_imagenet else 'places_2_256_training_saturation_index.json.gz'
        f = ujson.load(gzip.open('../dataset_indexes/' + source, 'rt'))
        self.all_paths = [path for path in f.keys() if f[path] > self.SATURATION_THRESHOLD]

        self.test_batch = self.all_paths[:self.batch_size]

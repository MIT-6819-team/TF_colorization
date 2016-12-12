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
    TRAIN_SOURCE = '../dataset_indexes/imagenet_train_256_saturation_values.json.gz'
    VALIDATION_SOURCE = '../dataset_indexes/imagenet_human_validation_set.json'
    CATEGORY_SOURCE = '../dataset_indexes/imagenet_train_256_category_paths_reweighted.json.gz'
    # This is the per-example probability that an example will be pulled from a specific category.
    # This encourages that images from the same category show up together more often.
    CATEGORY_REWEIGHT_ALPHA = 0.1
    

    def __init__(self, batch_size, use_imagenet=True, batching_style="reweighted"):
        self.batch_size = batch_size
        self.batching_style = batching_style
        self._load_paths_and_threshold(use_imagenet)

        self.training_batches = []
        self.batches_available = threading.Semaphore(0)
        self.DESIRED_QUEUED_BATCHES = 3

    	if use_imagenet:    
            self.root = '../../datasets/imagenet/train256/'
            self.validation_root = '../../datasets/imagenet/val256/'
        else:
    	    print "Don't know where the places_2 root is!"
            raise
            
    def get_filenames_for_batch(self):
        '''
        Returns a list of filenames. Uses value from self.batching_style.
        '''
        paths = []
        if self.batching_style == "reweighted":
            # Pick a category to focus on.
            focus_category = self.categories[int(random.random()*len(self.categories))]
            category_filenames = self.category_index[focus_category]
            
            for i in xrange(self.batch_size):
                if random.random() <= self.CATEGORY_REWEIGHT_ALPHA:
                    paths.append(category_filenames[int(random.random()*len(category_filenames))])
                else:
                    paths.append(self.all_paths[int(random.random() * len(self.all_paths))])
        else:
          paths = [self.all_paths[int(random.random() * len(self.all_paths))] for i in xrange(self.batch_size)]
        return paths

        
    def next_batch(self):
        """Gets the next batch from the dataset and starts loading others in parallel."""
        # Make sure that we always have enough batches precomputed
        for b in xrange(self.DESIRED_QUEUED_BATCHES - len(self.training_batches)):
          threading.Thread(target = self._load_batch).start()

        self.batches_available.acquire()  # Wait for a new batch
        data_x, data_y_, data_y_rebalance = self.training_batches[0]
        del self.training_batches[0]

        return data_x, data_y_, data_y_rebalance

    def get_validation_batch(self):
        x_batch = np.zeros((len(self.validation_paths), self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, 1))
        y__batch = np.zeros((len(self.validation_paths), self.OUTPUT_IMAGE_SIZE, self.OUTPUT_IMAGE_SIZE, 313))
        gt_batch = np.zeros((len(self.validation_paths), self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, 3))

        for i in range(len(self.validation_paths)):
            path = self.validation_paths[i]
            x, y_, _, ground_truth= image_path_to_image_and_distribution_tensor(self.validation_root + path)

            x_batch[i, ...] = x.reshape((256, 256, 1))
            y__batch[i, ...] = y_
            gt_batch[i, ...] = ground_truth

        return x_batch, y__batch, gt_batch

    def _load_batch(self):
        """Load the next batch, queue it, and increase the semaphore."""
        lt = time.time()
        x_batch = np.zeros((self.batch_size, self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, 1))
        y__batch = np.zeros((self.batch_size, self.OUTPUT_IMAGE_SIZE, self.OUTPUT_IMAGE_SIZE, 313))
        y_reweight_batch = np.zeros((self.batch_size, self.OUTPUT_IMAGE_SIZE, self.OUTPUT_IMAGE_SIZE))
        
        paths = self.get_filenames_for_batch()
        
        for i,path in enumerate(paths):
          x, y_, y_reweight, _ = image_path_to_image_and_distribution_tensor(self.root + path)

          x_batch[i, ...] = x.reshape((256, 256, 1))
          y__batch[i, ...] = y_
          y_reweight_batch[i, ...] = y_reweight

        self.training_batches.append((x_batch, y__batch, y_reweight_batch))
        self.batches_available.release()
        print "Batch loaded in parallel ", (time.time() - lt)

    def _load_paths_and_threshold(self, use_imagenet):
        '''Loads all the paths and removes those below the saturation threshold.'''
        
        if self.batching_style == "reweighted":
            # Take the file list from the reweighted category list
            f = ujson.load(gzip.open(self.CATEGORY_SOURCE, 'rt'))
            self.all_paths = []
            for category, paths in f.items():
                self.all_paths += [path for path in paths if path > self.SATURATION_THRESHOLD]
        else:
            f = ujson.load(gzip.open(self.TRAIN_SOURCE, 'rt'))
            self.all_paths = [path for path in f.keys() if f[path] > self.SATURATION_THRESHOLD]

        vf = ujson.load(open(self.VALIDATION_SOURCE, 'rt'))
        self.validation_paths = [path for path in vf.keys() if vf[path] > self.SATURATION_THRESHOLD]
        

        self.category_index = ujson.load(gzip.open(self.CATEGORY_SOURCE, 'rt'))
        self.categories = self.category_index.keys()

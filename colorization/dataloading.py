import tensorflow as tf
import cv2
import numpy as np
from skimage import color

WIDTH = 100
HEIGHT = 100
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 3


def rgb_to_lab(rbg):
  return color.rgb2lab(rgb)

def img_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def rescale_img(img):
    return cv2.resize(img,(WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)

class DataLoader(object):
    def __init__(self):
        data_dir = 'images/'
        self._idx = 0

        self.colored_images = []
        self.grey_images = []

        for name in ['images/i1.jpg']:
          color_img = rescale_img(cv2.imread(name))
          gray_img = img_to_gray(color_img)#.reshape(WIDTH, HEIGHT, 1)
          self.colored_images.append(color_img)
          self.grey_images.append(gray_img)

        self.num = len(self.colored_images)
        
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, HEIGHT, WIDTH, INPUT_CHANNELS)) 
        labels_batch = np.zeros((batch_size, HEIGHT, WIDTH, OUTPUT_CHANNELS))
          
        for i in range(batch_size):
            # when your dataset is huge, you might need to load images on the fly
            # you might also want data augmentation
            # images_batch[i, ...] = self.grey_images[self._idx].reshape((HEIGHT*WIDTH*INPUT_CHANNELS))
            # img = tf.reshape(self.grey_images[self._idx], (HEIGHT, WIDTH, INPUT_CHANNELS))
            # print 'src',self.grey_images[self._idx]
            img = np.expand_dims(self.grey_images[self._idx], axis=2)
            # print 'img',img
            images_batch[i, ...] = img

            labels_batch[i, ...] = self.colored_images[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
        
        # images_batch = tf.reshape(images_batch, (None, HEIGHT, WIDTH, INPUT_CHANNELS))

        return images_batch, labels_batch
    
    def load_test(self):
        return self.dataset.test.images.reshape((-1, self.h, self.w, self.c)), self.dataset.test.labels

import tensorflow as tf
import cv2
import numpy as np

WIDTH = 100
HEIGHT = 100
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 3

def img_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def rescale_img(img):
    return cv2.resize(img,(WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)

import subprocess

def sendmessage(message):
    subprocess.Popen(['notify-send', message])
    return

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

def setup_tensorflow_graph():
  x = tf.placeholder(tf.float32, shape=[None,WIDTH,HEIGHT,1])#WIDTH*HEIGHT])
  y_ = tf.placeholder(tf.float32, shape=[None,WIDTH,HEIGHT,OUTPUT_CHANNELS]) #WIDTH*HEIGHT*OUTPUT_CHANNELS])

  # x_image = tf.reshape(x, [-1,WIDTH,HEIGHT,1])
  x_image = x

  W_conv1 = weight_variable([5, 5, 1, 64])
  b_conv1 = bias_variable([64])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5, 5, 64, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_conv3 = weight_variable([5, 5, 64, 32])
  b_conv3 = bias_variable([32])

  h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
  h_pool3 = max_pool_2x2(h_conv3)

  W_conv4 = weight_variable([5, 5, 32, 3])
  b_conv4 = bias_variable([3])

  h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
  h_pool4 = max_pool_2x2(h_conv4)

  # y_output = tf.reshape(h_pool3, [-1,WIDTH*HEIGHT*OUTPUT_CHANNELS])
  y_output = h_pool4

  print x
  print y_
  print x_image
  print h_conv1
  print h_conv2
  print h_conv3
  print y_output

  return x, y_, y_output

  # W_fc1 = weight_variable([WIDTH*HEIGHT * 64, 1024])
  # b_fc1 = bias_variable([1024])

  # h_pool2_flat = tf.reshape(h_pool2, [-1, WIDTH*HEIGHT*64])
  # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # keep_prob = tf.placeholder(tf.float32)
  # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # W_fc2 = weight_variable([1024, 10])
  # b_fc2 = bias_variable([10])

  # y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

def run_training():
  print "Construct Session"
  with tf.Session() as sess:
    x, y_, y_output = setup_tensorflow_graph()
    cross_entropy = tf.reduce_sum(tf.squared_difference(y_, y_output))#tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_output, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_sum(tf.squared_difference(y_, y_output))#tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()

    dataset = DataLoader()
    print "Begin training"
    for i in range(5000):
      batch = dataset.next_batch(1)
      if i%1 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1]})#, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})#, keep_prob: 0.5})

    save_path = saver.save(sess, "model.ckpt")
    print("Model saved in file: %s" % save_path)

  sendmessage('Done training')


def colorize_img():
  with tf.Session() as sess:
    x, y_, y_output = setup_tensorflow_graph()
    saver = tf.train.Saver()
    dataset = DataLoader()

    saver.restore(sess, "model.ckpt")
    # x = img_to_gray(rescale_img(cv2.imread(name)))
    for i in range(1):
      batch = dataset.next_batch(1)
      img = sess.run(y_output, feed_dict={x: batch[0]})
      img = img.reshape((100, 100, 3))
      print img.shape
      cv2.imshow('y',img)
      cv2.waitKey(0)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 1, 1, 1], padding='SAME') #was [1, 2, 2, 1]


run_training()
# colorize_img()
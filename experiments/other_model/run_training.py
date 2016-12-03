#TODO: RGB -> Lab
#TODO: Randomly crop/scale as done in paper

import tensorflow as tf
import cv2
import numpy as np
from skimage import color

def run_training():
  # Should base on https://github.com/richzhang/colorization/blob/master/train/solver.prototxt

  print "Construct Session"
  with tf.Session() as sess:
    x, y_, y_output = setup_tensorflow_graph()
    cross_entropy = tf.reduce_sum(tf.squared_difference(y_, y_output))#tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_output, y_))
    train_step = tf.train.AdamOptimizer(3.16e-5).minimize(cross_entropy) # TODO: Set params equal to correct values
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



run_training()

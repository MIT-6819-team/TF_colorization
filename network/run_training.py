#TODO: Loss function, optimizer, etc.

import tensorflow as tf
from construct_graph import setup_tensorflow_graph
from dataloader import DataLoader

def run_training():
  '''Runs training.'''
  with tf.Session() as sess:
    x, y_, y_output = setup_tensorflow_graph()
    saver = tf.train.Saver()
    dataset = DataLoader()

    sess.run(tf.initialize_all_variables())

    for i in range(1):
      data_x, data_y_ = dataset.next_batch(32)
      img = sess.run(y_output, feed_dict={x: batch[0]})

run_training()

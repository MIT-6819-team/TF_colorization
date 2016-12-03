import tensorflow as tf
from construct_graph import setup_tensorflow_graph
from lab_to_distribution import image_path_to_image_and_distribution_tensor
from dataloader import DataLoader

def run_training():
  '''Runs training.'''
  with tf.Session() as sess:
    x, y_, y_output = setup_tensorflow_graph()
    saver = tf.train.Saver()
    dataset = DataLoader()

    saver.restore(sess, "model.ckpt")
    #TODO : make this
    for i in range(1):
      data_x, data_y_ = dataset.next_batch(32)
      img = sess.run(y_output, feed_dict={x: batch[0]})

run_training()

import tensorflow as tf
import construct_graph
from dataloader import DataLoader
from distribution_to_image import get_colorized_image
import warnings
warnings.filterwarnings('ignore')
import time
import threading

training_batches = []
batches_available = threading.Semaphore(0)
DESIRED_BATCHES = 3

def load_batch(dataset):
  lt = time.time()
  data_x, data_y_ = dataset.next_batch()
  training_batches.append((data_x, data_y_))
  batches_available.release()
  print "Batch loaded in parallel ", (time.time() - lt)

def run_training(BATCH_SIZE = 32, ITERATIONS = 3000):
  f = open('log.txt', 'w')

  with tf.Session() as sess:
    x, y_, y_output = construct_graph.setup_tensorflow_graph(BATCH_SIZE)

    saver = tf.train.Saver()
    dataset = DataLoader(BATCH_SIZE)
    dataset.next_batch() # hack to precompute, remove this

    loss = construct_graph.loss_function(y_output, y_)
    prediction = construct_graph.get_prediction(y_output)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    sess.run(tf.initialize_all_variables())

    for i in xrange(ITERATIONS):
      lt = time.time()
      for b in xrange(DESIRED_BATCHES - len(training_batches)):
        t = threading.Thread(target = load_batch, args = (dataset,))
        t.start()

      batches_available.acquire()
      data_x, data_y_ = training_batches[0]
      del training_batches[0]

      lt2 = time.time()
      _, loss_res = sess.run([train_step, loss], feed_dict={x: data_x, y_: data_y_})
      lt3 = time.time()

      #f.write(str(loss_res) + '\n'); f.flush()

      if i % 1000 == 0:
        _colorize_and_save_test_images(sess, dataset, prediction, i/1000, x)
        saver.save( sess, 'model/model')

      print "Iteration ", i, "Data loading: ", (lt2 - lt), "Backprop: ", (lt3 - lt2), "Full", (time.time() - lt), "Accuracy:", loss_res


def _colorize_and_save_test_images(sess, dataset, prediction, iteration, x):
  test_image_batch, _ = dataset.get_test_batch()
  test_image_predictions = sess.run( prediction,  feed_dict = {x: test_image_batch} )
  for i in range(test_image_batch.shape[0]):
    get_colorized_image(test_image_batch[i], test_image_predictions[i]).save('images/' + str(i) + '_' + str(iteration) + '.jpg')

run_training()

import tensorflow as tf
import construct_graph
from dataloader import DataLoader
from distribution_to_image import get_colorized_image
import warnings
warnings.filterwarnings('ignore')
import time

RESTORE_FROM_X_ITERATIONS = 61000

def run_training(BATCH_SIZE = 16, ITERATIONS = 99999999999):
  f = open('log.txt', 'w')

  with tf.Session() as sess:

    print "Setup dataloader"
    saver = tf.train.Saver()
    data_x, data_y = DataLoader(BATCH_SIZE)

    print "Setup graph"
    y_output = construct_graph.setup_tensorflow_graph(data_x, BATCH_SIZE)
    loss = construct_graph.loss_function(y_output, data_y)
    prediction = construct_graph.get_prediction(y_output)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
 
    sess.run(tf.initialize_all_variables())
    if RESTORE_FROM_X_ITERATIONS:
        print "Restoring model."
        saver.restore(sess, "model/model")
    else:
        print "Starting model from scratch."


    for i in xrange(ITERATIONS):
      lt = time.time()

      #data_x, data_y_ = dataset.next_batch()

      lt2 = time.time()
      _, loss_res = sess.run([train_step, loss])
      lt3 = time.time()

      if i % 1000 == 0 and i != 0:
        _colorize_and_save_test_images(sess, dataset, prediction, (i + RESTORE_FROM_X_ITERATIONS)/1000, x)
        saver.save( sess, 'model/model')

      print "Iteration ", i, "Data loading: ", (lt2 - lt), "Backprop: ", (lt3 - lt2), "Full", (time.time() - lt), "Accuracy:", loss_res


def _colorize_and_save_test_images(sess, dataset, prediction, iteration, x):
  test_image_batch, _ = dataset.get_test_batch()
  test_image_predictions = sess.run( prediction,  feed_dict = {x: test_image_batch} )
  for i in range(test_image_batch.shape[0]):
    get_colorized_image(test_image_batch[i], test_image_predictions[i]).save('images/' + str(i) + '_' + str(iteration) + '.jpg')

run_training()

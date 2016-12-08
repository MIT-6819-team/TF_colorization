import tensorflow as tf
import construct_graph
from dataloader import DataLoader
from distribution_to_image import get_colorized_image
import warnings
warnings.filterwarnings('ignore')
import time

def run_training(BATCH_SIZE = 32, ITERATIONS = 99999999999, RESTORE_FROM_MODEL = True, REWEIGHT_COLOR_CLASSES = False):
  with tf.Session() as sess:
    x, y_, y_output, rebalance_ = construct_graph.setup_tensorflow_graph(BATCH_SIZE)

    print "Setup dataloader"
    saver = tf.train.Saver()
    dataset = DataLoader(BATCH_SIZE)

    print "Setup graph"
    if REWEIGHT_COLOR_CLASSES:
        loss = construct_graph.weighted_loss_function(y_output, y_, rebalance_)
        model_name = "rebalanced_model/model"
    else:
        loss = construct_graph.loss_function(y_output, y_)
        model_name = "model/model"

    f = open('reweighted_iterations_log.txt' if REWEIGHT_COLOR_CLASSES else 'iterations_log.txt', 'ra')

    prediction = construct_graph.get_prediction(y_output)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    sess.run(tf.initialize_all_variables())
    starting_iterations = 0
    if RESTORE_FROM_MODEL:
        print "Restoring model."
        saver.restore(sess, model_name)
        starting_iterations = int(f.readlines()[-1].rstrip('\n'))
        print "Starting from iteration", starting_iterations
    else:
        print "Starting model from scratch."
        thread.sleep(3)

    for i in xrange(starting_iterations, ITERATIONS):
      lt = time.time()

      data_x, data_y_, data_y_rebalance = dataset.next_batch()

      lt2 = time.time()
      _, loss_res = sess.run([train_step, loss], feed_dict={x: data_x, y_: data_y_, rebalance_: data_y_rebalance})
      lt3 = time.time()

      if i % 1000 == 0:
        _colorize_and_save_test_images(sess, dataset, prediction, (i)/1000, x, REWEIGHT_COLOR_CLASSES)
        saver.save( sess, model_name)
        f.write(str(i))
        f.flush()

      print "Iteration ", i, "Data loading: ", (lt2 - lt), "Backprop: ", (lt3 - lt2), "Full", (time.time() - lt), "Accuracy:", loss_res


def _colorize_and_save_test_images(sess, dataset, prediction, iteration, x, REWEIGHT_COLOR_CLASSES):
  test_image_batch, _ = dataset.get_test_batch()
  test_image_predictions = sess.run( prediction,  feed_dict = {x: test_image_batch} )
  for i in range(test_image_batch.shape[0]):
      image_folder = 'reweight_images/' if REWEIGHT_COLOR_CLASSES else 'images/'
      get_colorized_image(test_image_batch[i], test_image_predictions[i]).save(image_folder + str(i) + '_' + str(iteration) + '.jpg')

run_training()

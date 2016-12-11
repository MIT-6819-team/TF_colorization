#!/usr/bin/env python
import tensorflow as tf
import construct_graph
from dataloader import DataLoader
from distribution_to_image import get_colorized_image
import warnings
warnings.filterwarnings('ignore')
import time, os
from joblib import Parallel, delayed

TEST_NAME = "probablistic_reweighted_batching"
BATCHING_STYLE = "probablistic"

def run_training(BATCH_SIZE = 32, ITERATIONS = 99999999999, RESTORE_FROM_MODEL = True, REWEIGHT_COLOR_CLASSES = True):
  print "Run training for test '{}' using batch style {}. Reweight: {} Batch Size: {}.".format(TEST_NAME, BATCHING_STYLE, REWEIGHT_COLOR_CLASSES, BATCH_SIZE)

  with tf.Session() as sess:
    x, y_, y_output, rebalance_ = construct_graph.setup_tensorflow_graph(BATCH_SIZE)

    print "Setup dataloader"
    saver = tf.train.Saver()
    dataset = DataLoader(BATCH_SIZE, batching_style=BATCHING_STYLE)

    print "Setup graph"
    logfile_name = 'tests/{}/iterations_log.txt'.format(TEST_NAME)
    model_name = "tests/{}/model".format(TEST_NAME)
    # Ensure that the directory we want to save our stuff in exists.
    if not os.path.exists("tests/{}".format(TEST_NAME)):
        os.makedirs("tests/{}".format(TEST_NAME))
        os.makedirs("tests/{}/model/".format(TEST_NAME))
        os.makedirs("tests/{}/images/".format(TEST_NAME))
    

    loss = construct_graph.weighted_loss_function(y_output, y_, rebalance_) if REWEIGHT_COLOR_CLASSES else construct_graph.loss_function(y_output, y_)

    prediction = construct_graph.get_prediction(y_output)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    sess.run(tf.initialize_all_variables())
    starting_iterations = 0
    if RESTORE_FROM_MODEL:
        print "Trying to restore model."
        try:
            with open(logfile_name, 'r') as f:
                starting_iterations = int(f.readlines()[-1].rstrip('\n'))
            print "Restoring model from iteration", starting_iterations
            saver.restore(sess, "{}_step_{}".format(model_name, starting_iterations))
        except:
            print "ERROR loading model. Ignoring and starting model from scratch."
    else:
        print "Starting model from scratch."

    for i in xrange(starting_iterations, ITERATIONS):
      lt = time.time()

      data_x, data_y_, data_y_rebalance = dataset.next_batch()

      lt2 = time.time()
      _, loss_res = sess.run([train_step, loss], feed_dict={x: data_x, y_: data_y_, rebalance_: data_y_rebalance})
      lt3 = time.time()

      if i % 1000 == 0:
        print "Generating images of hold out set."
        _colorize_and_save_test_images(sess, dataset, prediction, (i)/1000, x, REWEIGHT_COLOR_CLASSES)
        print "Saving the model."
        saver.save( sess, "{}_step_{}".format(model_name,i))
        with open(logfile_name, 'a') as f:
            f.write(str(i)+'\n')

      print "Iteration ", i, "Data loading: ", (lt2 - lt), "Backprop: ", (lt3 - lt2), "Full", (time.time() - lt), "Accuracy:", loss_res
    
def workerfunc(test_im, test_im_predictions, x, itr, folder):
    return get_colorized_image(test_im, test_im_predictions, False).save(folder + str(x) + '_' + str(itr) + '.jpg')


def _colorize_and_save_test_images(sess, dataset, prediction, iteration, x, REWEIGHT_COLOR_CLASSES):
  test_image_batch, _, z = dataset.get_validation_batch()
  test_image_predictions = sess.run( prediction,  feed_dict = {x: test_image_batch} )
  image_folder = 'tests/{}/images/'.format(TEST_NAME)
    
  Parallel(n_jobs=4)(delayed(workerfunc)(test_image_batch[i], test_image_predictions[i], i, iteration,image_folder) for i in xrange(test_image_batch.shape[0]))

run_training()

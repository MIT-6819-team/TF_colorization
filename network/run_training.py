#!/usr/bin/env python
import tensorflow as tf
import construct_graph
from dataloader import DataLoader
from distribution_to_image import get_colorized_image
import warnings
warnings.filterwarnings('ignore')
import time
from joblib import Parallel, delayed

def run_training(BATCH_SIZE = 32, ITERATIONS = 99999999999, RESTORE_FROM_MODEL = True, REWEIGHT_COLOR_CLASSES = True):
  print "Run training! Reweight: ", REWEIGHT_COLOR_CLASSES, " Batch Size: ", BATCH_SIZE

  with tf.Session() as sess:
    x, y_, y_output, rebalance_ = construct_graph.setup_tensorflow_graph(BATCH_SIZE)

    print "Setup dataloader"
    saver = tf.train.Saver()
    dataset = DataLoader(BATCH_SIZE)

    print "Setup graph"
    logfile_name = 'reweighted_iterations_log.txt' if REWEIGHT_COLOR_CLASSES else 'iterations_log.txt'
    if REWEIGHT_COLOR_CLASSES:
        loss = construct_graph.weighted_loss_function(y_output, y_, rebalance_)
        model_name = "rebalanced_model/model"
    else:
        loss = construct_graph.loss_function(y_output, y_)
        model_name = "model/model"

    prediction = construct_graph.get_prediction(y_output)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    sess.run(tf.initialize_all_variables())
    starting_iterations = 0
    if RESTORE_FROM_MODEL:
        print "Trying to restore model."
        try:
            saver.restore(sess, model_name)
            with open(logfile_name, 'r') as f:
                starting_iterations = int(f.readlines()[-1].rstrip('\n'))
            print "Starting from iteration", starting_iterations
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
        print "Generated colorized images."
        _colorize_and_save_test_images(sess, dataset, prediction, (i)/1000, x, REWEIGHT_COLOR_CLASSES)
        saver.save( sess, model_name)
        with open(logfile_name, 'a') as f:
            f.write(str(i)+'\n')

      print "Iteration ", i, "Data loading: ", (lt2 - lt), "Backprop: ", (lt3 - lt2), "Full", (time.time() - lt), "Accuracy:", loss_res
    
def workerfunc(test_im, test_im_predictions, x, itr, folder):
    return get_colorized_image(test_im, test_im_predictions, False).save(folder + str(x) + '_' + str(itr) + '.jpg')


def _colorize_and_save_test_images(sess, dataset, prediction, iteration, x, REWEIGHT_COLOR_CLASSES):
  test_image_batch, _ = dataset.get_test_batch()
  test_image_predictions = sess.run( prediction,  feed_dict = {x: test_image_batch} )
  image_folder = 'reweight_images/' if REWEIGHT_COLOR_CLASSES else 'images/'
    
  Parallel(n_jobs=4)(delayed(workerfunc)(test_image_batch[i], test_image_predictions[i], i, iteration,image_folder) for i in xrange(test_image_batch.shape[0]))

run_training()

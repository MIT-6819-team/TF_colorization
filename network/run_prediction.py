import tensorflow as tf
import construct_graph
from dataloader import DataLoader
from distribution_to_image import get_colorized_image
import warnings
warnings.filterwarnings('ignore')
import time
from joblib import Parallel, delayed
from PIL import Image
import numpy as np


def run_prediction(REWEIGHT_COLOR_CLASSES = True):
    BATCH_SIZE = 20
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

        sess.run(tf.initialize_all_variables())

        print "Restoring model."
        saver.restore(sess, model_name)

        print "Running validation set through model"

        _colorize_and_save_test_images(sess, dataset, prediction, x, REWEIGHT_COLOR_CLASSES)

def workerfunc(test_im, test_im_predictions, x, folder, ground_truth_img):
    print ground_truth_img.shape
    Image.fromarray(ground_truth_img.astype(np.uint8)).save(folder + str(x) + '_gt.jpg')
    return get_colorized_image(test_im, test_im_predictions, False).save(folder + str(x) + '.jpg')

def _colorize_and_save_test_images(sess, dataset, prediction, x, REWEIGHT_COLOR_CLASSES):
  validation_image_batch, ground_truth_image_batch = dataset.get_validation_batch()
  validation_image_predictions = sess.run( prediction,  feed_dict = {x: validation_image_batch} )
  image_folder = 'reweight_images_validation/' if REWEIGHT_COLOR_CLASSES else 'images_validation/'

  Parallel(n_jobs=4)(delayed(workerfunc)(validation_image_batch[i], validation_image_predictions[i], i,image_folder, ground_truth_image_batch[i]) for i in xrange(validation_image_batch.shape[0]))

run_prediction()

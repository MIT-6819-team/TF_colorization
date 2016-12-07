import tensorflow as tf

def setup_tensorflow_graph(image_, BATCH_SIZE):

  #image_ = tf.placeholder( tf.float32, shape = [None, 256, 256, 1] )
  output_ = tf.placeholder( tf.float32, shape = [None, 64, 64, 313])

  W1_1 = weight_variable([3,3,1,64])
  b1_1 = bias_variable([64])
  conv1_1 = conv2d( image_, W1_1, 1 ) + b1_1
  conv1_1 = tf.nn.relu(conv1_1)

  W1_2 = weight_variable([3,3,64,64])
  b1_2 = bias_variable([64])
  conv1_2 = conv2d( conv1_1, W1_2, 2 ) + b1_2
  conv1_2 = tf.nn.relu(conv1_2)

  conv1_2 = tf.contrib.layers.batch_norm(conv1_2)

  W2_1 = weight_variable([3,3,64,128])
  b2_1 = bias_variable([128])
  conv2_1 = conv2d( conv1_2, W2_1, 1 ) + b2_1
  conv2_1 = tf.nn.relu(conv2_1)

  W2_2 = weight_variable([3,3,128,128])
  b2_2 = bias_variable([128])
  conv2_2 = conv2d( conv2_1, W2_2, 2 ) + b2_2
  conv2_2 = tf.nn.relu(conv2_2)

  conv2_2 = tf.contrib.layers.batch_norm(conv2_2)

  W3_1 = weight_variable([3,3,128,256])
  b3_1 = bias_variable([256])
  conv3_1 = conv2d( conv2_2, W3_1, 1 ) + b3_1
  conv3_1 = tf.nn.relu(conv3_1)

  W3_2 = weight_variable([3,3,256,256])
  b3_2 = bias_variable([256])
  conv3_2 = conv2d( conv3_1, W3_2, 1 ) + b3_2
  conv3_2 = tf.nn.relu(conv3_2)

  W3_3 = weight_variable([3,3,256,256])
  b3_3 = bias_variable([256])
  conv3_3 = conv2d( conv3_2, W3_3, 2 ) + b3_3
  conv3_3 = tf.nn.relu(conv3_3)

  conv3_3 = tf.contrib.layers.batch_norm(conv3_3)

  W4_1 = weight_variable([3,3,256,512])
  b4_1 = bias_variable([512])
  conv4_1 = conv2d( conv3_3, W4_1, 1 ) + b4_1
  conv4_1 = tf.nn.relu(conv4_1)

  W4_2 = weight_variable([3,3,512,512])
  b4_2 = bias_variable([512])
  conv4_2 = conv2d( conv4_1, W4_2, 1 ) + b4_2
  conv4_2 = tf.nn.relu(conv4_2)

  W4_3 = weight_variable([3,3,512,512])
  b4_3 = bias_variable([512])
  conv4_3 = conv2d( conv4_2, W4_3, 1 ) + b4_3
  conv4_3 = tf.nn.relu(conv4_3)

  conv4_3 = tf.contrib.layers.batch_norm(conv4_3)

  W5_1 = weight_variable([3,3,512,512])
  b5_1 = bias_variable([512])
  conv5_1 = tf.nn.atrous_conv2d( conv4_3, W5_1, 2, padding = 'SAME') + b5_1
  conv5_1 = tf.nn.relu(conv5_1)

  W5_2 = weight_variable([3,3,512,512])
  b5_2 = bias_variable([512])
  conv5_2 = tf.nn.atrous_conv2d( conv5_1, W5_2, 2, padding = 'SAME') + b5_2
  conv5_2 = tf.nn.relu(conv5_2)

  W5_3 = weight_variable([3,3,512,512])
  b5_3 = bias_variable([512])
  conv5_3 = tf.nn.atrous_conv2d( conv5_2, W5_3, 2, padding = 'SAME') + b5_3
  conv5_3 = tf.nn.relu(conv5_3)

  conv5_3 = tf.contrib.layers.batch_norm(conv5_3)

  W6_1 = weight_variable([3,3,512,512])
  b6_1 = bias_variable([512])
  conv6_1 = tf.nn.atrous_conv2d( conv5_3, W6_1, 2, padding = 'SAME') + b6_1
  conv6_1 = tf.nn.relu(conv6_1)

  W6_2 = weight_variable([3,3,512,512])
  b6_2 = bias_variable([512])
  conv6_2 = tf.nn.atrous_conv2d( conv6_1, W6_2, 2, padding = 'SAME') + b6_2
  conv6_2 = tf.nn.relu(conv6_2)

  W6_3 = weight_variable([3,3,512,512])
  b6_3 = bias_variable([512])
  conv6_3 = tf.nn.atrous_conv2d( conv6_2, W6_3, 2, padding = 'SAME') + b6_3
  conv6_3 = tf.nn.relu(conv6_3)

  conv6_3 = tf.contrib.layers.batch_norm(conv6_3)

  W7_1 = weight_variable([3,3,512,256])
  b7_1 = bias_variable([256])
  conv7_1 = conv2d( conv6_3, W7_1, 1 ) + b7_1
  conv7_1 = tf.nn.relu(conv7_1)

  W7_2 = weight_variable([3,3,256,256])
  b7_2 = bias_variable([256])
  conv7_2 = conv2d( conv7_1, W7_2, 1 ) + b7_2
  conv7_2 = tf.nn.relu(conv7_2)

  W7_3 = weight_variable([3,3,256,256])
  b7_3 = bias_variable([256])
  conv7_3 = conv2d( conv7_2, W7_3, 1 ) + b7_3
  conv7_3 = tf.nn.relu(conv7_3)

  conv7_3 = tf.contrib.layers.batch_norm(conv7_3)

  conv7_3 = tf.image.resize_images(conv7_3, [64,64])

  W8_1 = weight_variable([4,4,256,128])
  b8_1 = bias_variable([128])
  conv8_1 = conv2d( conv7_3, W8_1, 1 ) + b8_1
  conv8_1 = tf.nn.relu(conv8_1)

  W8_2 = weight_variable([3,3,128,128])
  b8_2 = bias_variable([128])
  conv8_2 = conv2d( conv8_1, W8_2, 1 ) + b8_2
  conv8_2 = tf.nn.relu(conv8_2)

  W8_3 = weight_variable([3,3,128,128])
  b8_3 = bias_variable([128])
  conv8_3 = conv2d( conv8_2, W8_3, 1 ) + b8_3
  conv8_3 = tf.nn.relu(conv8_3)

  W_ab = weight_variable([1,1,128,313])
  b_ab = bias_variable([313])
  conv_ab = conv2d( conv8_3, W_ab, 1 ) + b_ab
  output = tf.nn.relu(conv_ab)

  return image_, output_, output

def loss_function(output, output_):
  loss = tf.nn.softmax_cross_entropy_with_logits( output,  output_ )
  return tf.reduce_mean(loss)

def weighted_loss_function(output, output_):
    quantized_frequencies = np.load('../preprocessing/reweighting_vector.npy')

    loss = tf.nn.softmax_cross_entropy_with_logits( output,  output_ )
    return tf.reduce_mean(loss)

def get_prediction( output ):
  prediction = tf.nn.softmax( output )
  return tf.image.resize_images( prediction, [256,256] )

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, s):
  return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')

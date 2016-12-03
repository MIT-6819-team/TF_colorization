# TODO: Lab quantization, boost priors, loss

def setup_tensorflow_graph():
  # Reconstruct the architecture from https://github.com/richzhang/colorization/blob/master/models/colorization_train_val_v2.prototxt
  # (I think this is the right one? Not sure)
  x = tf.placeholder(tf.float32, shape=[None,WIDTH,HEIGHT,1])
  y_ = tf.placeholder(tf.float32, shape=[None,WIDTH,HEIGHT,OUTPUT_CHANNELS])

  layer_1_1 = conv_layer(x, prev_kernels=3, kernels=64, kernel_size=3, stride=1)
  layer_1_2 = conv_layer(layer_1_1, prev_kernels=64, kernels=64, kernel_size=3, stride=2)
  layer_1_bn = batch_norm(layer_1_2)

  layer_2_1 = conv_layer(layer_1_bn, prev_kernels=64, kernels=128, kernel_size=3, stride=1)
  layer_2_2 = conv_layer(layer_2_1, prev_kernels=128, kernels=128, kernel_size=3, stride=2)
  layer_2_bn = batch_norm(layer_2_2)

  layer_3_1 = conv_layer(layer_2_bn, prev_kernels=128, kernels=256, kernel_size=3, stride=1)
  layer_3_2 = conv_layer(layer_3_1, prev_kernels=256, kernels=256, kernel_size=3, stride=1)
  layer_3_3 = conv_layer(layer_3_2, prev_kernels=256, kernels=256, kernel_size=3, stride=2)
  layer_3_bn = batch_norm(layer_3_3)

  layer_4_1 = conv_layer(layer_3_bn, prev_kernels=256, kernels=512, kernel_size=3, stride=1)
  layer_4_2 = conv_layer(layer_4_1, prev_kernels=512, kernels=512, kernel_size=3, stride=1)
  layer_4_3 = conv_layer(layer_4_2, prev_kernels=512, kernels=512, kernel_size=3, stride=1)
  layer_4_bn = batch_norm(layer_4_3)

  layer_5_1 = conv_layer(layer_4_bn, prev_kernels=256, kernels=512, kernel_size=3, stride=1, dilation=2)
  layer_5_2 = conv_layer(layer_5_1, prev_kernels=512, kernels=512, kernel_size=3, stride=1, dilation=2)
  layer_5_3 = conv_layer(layer_5_2, prev_kernels=512, kernels=512, kernel_size=3, stride=1, dilation=2)
  layer_5_bn = batch_norm(layer_5_3)

  layer_6_1 = conv_layer(layer_5_bn, prev_kernels=512, kernels=512, kernel_size=3, stride=1, dilation=2)
  layer_6_2 = conv_layer(layer_6_1, prev_kernels=512, kernels=512, kernel_size=3, stride=1, dilation=2)
  layer_6_3 = conv_layer(layer_6_2, prev_kernels=512, kernels=512, kernel_size=3, stride=1, dilation=2)
  layer_6_bn = batch_norm(layer_6_3)

  layer_7_1 = conv_layer(layer_6_bn, prev_kernels=512, kernels=512, kernel_size=3, stride=1)  
  layer_7_2 = conv_layer(layer_7_1, prev_kernels=512, kernels=512, kernel_size=3, stride=1)
  layer_7_3 = conv_layer(layer_7_2, prev_kernels=512, kernels=512, kernel_size=3, stride=1)
  layer_7_bn = batch_norm(layer_7_3)

  layer_8_1 = conv_layer(layer_7_bn, prev_kernels=512, kernels=256, kernel_size=4, stride=2)  
  layer_8_2 = conv_layer(layer_8_1, prev_kernels=256, kernels=256, kernel_size=3, stride=1)
  layer_8_3 = conv_layer(layer_8_2, prev_kernels=256, kernels=256, kernel_size=3, stride=1)

  layer_8_313 = conv_layer(layer_8_3, prev_kernels=256, kernels=313, kernel_size=1, stride=1)  
  layer_8_313_rh = 2.606 * layer_8_313 # TODO: Why is this done? Is this right?
  layer8_313_softmax = tf.nn.softmax(layer_8_313_rh)
  class8_ab = conv_layer(layer8_313_softmax, prev_kernels=313, kernels=2, kernel_size=1, stride=1)  

  y_output = h_pool4
  return x, y_, y_output

def conv_layer(prev_layer, prev_kernels, kernels, kernel_size, stride, dilation=None):
  W_conv = weight_variable([kernel_size, kernel_size, prev_kernels, kernels])
  b_conv = bias_variable([kernels])

  return tf.nn.relu(conv2d(prev_layer, W_conv2, stride, dilation) + b_conv2)

def resize_layer(prev_layer, size)
  return tf.image.resize_bilinear(h_pool1, size)#, align_corners=None, name=None)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride, dilation=None):
  if dilation is None:
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
  else:
    return tf.nn.atrous_conv2d(x, W, dilation, padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 1, 1, 1], padding='SAME') #was [1, 2, 2, 1]

# def conv_and_resize_layer(prev_layer, prev_kernels, kernels, kernel_size, resize_size):
#   conv_l = conv_layer(prev_layer, prev_kernels, kernels, kernel_size)
#   return resize_layer(conv_l, resize_size)
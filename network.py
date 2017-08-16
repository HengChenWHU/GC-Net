import tensorflow as tf
from config import Config
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from main import HEIGHT, WIDTH

MOVING_AVERAGE_DECAY = 0.99
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00001
CONV_WEIGHT_STDDEV = 0.05
GC_VARIABLES = 'gc_variables'
UPDATE_OPS_COLLECTION = 'gc_update_ops'  # training ops
DISPARITY = 192
NUM_RES_BLOCK = 8   # totally 8 resnet blocks

# wrapper for 2d convolution op
def conv_2d(x, c):
  ksize = c['ksize']
  stride = c['stride']
  filters_out = c['conv_filters_out']

  filters_in = x.get_shape()[-1]
  shape = [ksize, ksize, filters_in, filters_out]
  weights = tf.get_variable('weights',
                         shape=shape,
                         dtype='float32',
                         initializer=tf.contrib.layers.xavier_initializer(),
			 regularizer=tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY),
			 collections=[tf.GraphKeys.GLOBAL_VARIABLES, GC_VARIABLES],
			 trainable=True)
  bias = tf.get_variable('bias', [filters_out], 'float32', tf.constant_initializer(0.05, dtype='float'))
  x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
  return tf.nn.bias_add(x, bias)

def conv_3d(x, c):
  ksize = c['ksize']
  stride = c['stride']
  filters_out = c['conv_filters_out']
  filters_in = x.get_shape()[-1]
  shape = [ksize, ksize, ksize, filters_in, filters_out]

  weights = tf.get_variable('weights',
                         shape=shape,
                         dtype='float32',
                         initializer=tf.contrib.layers.xavier_initializer(),
			 regularizer=tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY),
                         collections=[tf.GraphKeys.GLOBAL_VARIABLES, GC_VARIABLES],
                         trainable=True)
  bias = tf.get_variable('bias', [filters_out], 'float32', tf.constant_initializer(0.05, dtype='float'))
  x = tf.nn.conv3d(x, weights, [1, stride, stride, stride, 1], padding='SAME')
  return tf.nn.bias_add(x, bias)

def deconv_3d(x, c):
  ksize = c['ksize']
  stride = c['stride']
  filters_out = c['conv_filters_out']
  filters_in = x.get_shape()[-1]

  # must have as_list to get a python list!
  x_shape = x.get_shape().as_list()
  depth = x_shape[1] * stride
  height = x_shape[2] * stride
  width = x_shape[3] * stride
  output_shape = [1, depth, height, width, filters_out]
  strides = [1, stride, stride, stride, 1]
  shape = [ksize, ksize, ksize, filters_out, filters_in]

  initializer = tf.contrib.layers.xavier_initializer()
  weights = tf.get_variable('weights',
                          shape=shape,
                          dtype='float32',
                          initializer=tf.contrib.layers.xavier_initializer(),
			  regularizer=tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY),
                         collections=[tf.GraphKeys.GLOBAL_VARIABLES, GC_VARIABLES],
                         trainable=True)
  bias = tf.get_variable('bias', [filters_out], 'float32', tf.constant_initializer(0.05, dtype='float32'))
  x = tf.nn.conv3d_transpose(x, weights, output_shape=output_shape, strides=strides, padding='SAME')
  return tf.nn.bias_add(x, bias)

# wrapper for batch-norm op
def bn(x, c):
  x_shape = x.get_shape()
  params_shape = x_shape[-1:]

  axis = list(range(len(x_shape) - 1))

  beta = tf.get_variable('beta',
                         shape=params_shape,
                         initializer=tf.zeros_initializer(),
			 dtype='float32',
			 collections=[tf.GraphKeys.GLOBAL_VARIABLES, GC_VARIABLES],
			 trainable=True)
  gamma = tf.get_variable('gamma',
                          shape=params_shape,
                          initializer=tf.ones_initializer(),
			  dtype='float32',
                         collections=[tf.GraphKeys.GLOBAL_VARIABLES, GC_VARIABLES],
                         trainable=True)

  moving_mean = tf.get_variable('moving_mean',
                              shape=params_shape,
                              initializer=tf.zeros_initializer(),
			      dtype='float32',
                              collections=[tf.GraphKeys.GLOBAL_VARIABLES, GC_VARIABLES],
                              trainable=False)
  moving_variance = tf.get_variable('moving_variance',
                                  shape=params_shape,
                                  initializer=tf.ones_initializer(),
				  dtype='float32',
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES, GC_VARIABLES],
                                  trainable=False)

  # These ops will only be performed when training.
  mean, variance = tf.nn.moments(x, axis)
  update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                             mean, BN_DECAY)
  update_moving_variance = moving_averages.assign_moving_average(
                                        moving_variance, variance, BN_DECAY)
  tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
  tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

  mean, variance = control_flow_ops.cond(
    c['is_training'], lambda: (mean, variance),
    lambda: (moving_mean, moving_variance))

  x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

  return x


# resnet block
def stack(x, c):
  shortcut = x
  with tf.variable_scope('block_A'):
    x = conv_2d(x, c)
    x = bn(x, c)
    x = tf.nn.relu(x)
  with tf.variable_scope('block_B'):
    x = conv_2d(x, c)
    x = bn(x, c)
    x = shortcut + x
    x = tf.nn.relu(x)
  return x

# siamese structure
def _build_resnet(x, c):

  with tf.variable_scope('downsample'):
    c['conv_filters_out'] = 32
    c['ksize'] = 5
    c['stride'] = 2
    x = conv_2d(x, c)
    x = bn(x, c)
    x = tf.nn.relu(x)

  c['ksize'] = 3
  c['stride'] = 1

  with tf.variable_scope('resnet'):
    for i in xrange(NUM_RES_BLOCK):
      with tf.variable_scope('resnet' + str(i+1)):
        x = stack(x, c)
  
  x = conv_2d(x, c)
  x = bn(x, c)
  x = tf.nn.relu(x)

  return x


def inference(left_x, right_x, is_training):
  c = Config()
  c['is_training'] = tf.convert_to_tensor(is_training,
                                          dtype = 'bool',
                                          name = 'is_training')
  c['conv_filters_out'] = 32

  with tf.variable_scope("siamese") as scope:
    left_features = _build_resnet(left_x, c)
    scope.reuse_variables()
    right_features = _build_resnet(right_x, c)

  # Cost Volumn
  cost_vol = []
  left_features = tf.squeeze(left_features)
  right_features = tf.squeeze(right_features)
  for d in xrange(1, DISPARITY/2+1):
    paddings = [[0,0], [d,0], [0,0]]
    for k in xrange(c['conv_filters_out']):
      left_feature = tf.slice(left_features, [0, 0, k], [HEIGHT/2, WIDTH/2, 1])
      right_feature = tf.slice(right_features, [0, 0, k], [HEIGHT/2, WIDTH/2, 1])
      right_feature = tf.slice(right_feature, [0, d, 0], [HEIGHT/2, WIDTH/2-d, 1])
      right_feature = tf.pad(right_feature, paddings, "CONSTANT")
      # feature_pair = tf.concat([left_feature, right_feature], 3)
      cost_vol.append(left_feature)
      cost_vol.append(right_feature)
  cost_vol = tf.stack(cost_vol)
  cost_vol = tf.reshape(cost_vol, shape=(1, DISPARITY/2, 2*c['conv_filters_out'], HEIGHT/2, WIDTH/2))
  cost_vol = tf.transpose(cost_vol, [0, 1, 3, 4, 2])

  # 3d convolution
  with tf.variable_scope("3dconv"):
    c['ksize'] = 3
    c['stride'] = 1
    c['conv_filters_out'] = 32
    with tf.variable_scope('Conv_3d_' + str(0)):
      with tf.variable_scope('A'):
        x = conv_3d(cost_vol, c)
        x = bn(x, c)
        x = tf.nn.relu(x)

      with tf.variable_scope('B'):
        x = conv_3d(x, c)
        x = bn(x, c)
        x20 = tf.nn.relu(x)
	print('x20_1' + str(x20.get_shape()))
      c['stride'] = 2
      c['conv_filters_out'] = 64

      with tf.variable_scope('C'):
        x = conv_3d(x20, c)
        x = bn(x, c)
        x = tf.nn.relu(x)

    c['conv_filters_out'] = 64
    with tf.variable_scope('Conv_3d_' + str(1)):
      c['stride'] = 1

      with tf.variable_scope('A'):
        x = conv_3d(x, c)
        x = bn(x, c)
        x = tf.nn.relu(x)

      with tf.variable_scope('B'):
        x = conv_3d(x, c)
        x = bn(x, c)
        x23 = tf.nn.relu(x)
	print('x23_1' + str(x23.get_shape()))
      c['stride'] = 2

      with tf.variable_scope('C'):
        x = conv_3d(x23, c)
        x = bn(x, c)
        x = tf.nn.relu(x)

    with tf.variable_scope('Conv_3d_' + str(2)):
      c['stride'] = 1

      with tf.variable_scope('A'):
        x = conv_3d(x, c)
        x = bn(x, c)
        x = tf.nn.relu(x)

      with tf.variable_scope('B'):
        x = conv_3d(x, c)
        x = bn(x, c)
        x26 = tf.nn.relu(x)
	print('x26_1' + str(x26.get_shape()))
      c['stride'] = 2

      with tf.variable_scope('C'):
        x = conv_3d(x26, c)
        x = bn(x, c)
        x = tf.nn.relu(x)

    with tf.variable_scope('Conv_3d_' + str(3)):
      c['stride'] = 1

      with tf.variable_scope('A'):
        x = conv_3d(x, c)
        x = bn(x, c)
        x = tf.nn.relu(x)

      with tf.variable_scope('B'):
        x = conv_3d(x, c)
        x = bn(x, c)
        x29 = tf.nn.relu(x)
	print('x29_1' + str(x29.get_shape()))
      c['stride'] = 2
      c['conv_filters_out'] = 128

      with tf.variable_scope('C'):
        x = conv_3d(x29, c)
        x = bn(x, c)
        x = tf.nn.relu(x)

      c['stride'] = 1
      with tf.variable_scope('D'):
        c['stride'] = 1
        x = conv_3d(x, c)
        x = bn(x, c)
        x = tf.nn.relu(x)

      with tf.variable_scope('E'):
        x = conv_3d(x, c)
        x = bn(x, c)
        x = tf.nn.relu(x)
	print('xfinal' + str(x.get_shape()))
  # 3d deconvolution
  with tf.variable_scope("deconv"):
    c['stride'] = 2
    c['conv_filters_out'] = 64
    c['ksize'] = 3
    with tf.variable_scope('A'):
      x = deconv_3d(x, c)
      x = bn(x, c)
      x = tf.nn.relu(x)
      x = x + x29

      print('x29' + str(x29.get_shape()))
    with tf.variable_scope('B'):
      x = deconv_3d(x, c)
      x = bn(x, c)
      x = tf.nn.relu(x)
      x = x + x26
      print('x26' + str(x26.get_shape()))

    with tf.variable_scope('C'):
      x = deconv_3d(x, c)
      x = bn(x, c)
      x = tf.nn.relu(x)
      x = tf.slice(x, [0, 0, 0, 0, 0], [1,48,135,240,64])
      print('x23' + str(x23.get_shape()))
      x = x + x23

    c['conv_filters_out'] = 32
    with tf.variable_scope('D'):
      x = deconv_3d(x, c)
      x = bn(x, c)
      x = tf.nn.relu(x)
      x = x + x20

    c['conv_filters_out'] = 1
    with tf.variable_scope('E'):
      x = deconv_3d(x, c)

  x = tf.squeeze(x)
  x = -x

  return x

#!/usr/bin/python

import tensorflow as tf
import numpy as np
from network import *
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import readPFM
from PIL import Image
import argparse


WIDTH = 960 
HEIGHT = 540


def parse_args():
  parser = argparse.ArgumentParser(description="Tune stereo matching network")
  parser.add_argument('--phase', default='train', help='train or test')
  parser.add_argument('--gpu', default='0', help='state the index of gpu: 0, 1, 2 or 3')
  parser.add_argument('--output_dir', default='/data/jack/flyingthings3d/gc-net/log')
  parser.add_argument('--learning_rate', default=0.001, type=float)
  parser.add_argument('--max_steps', default=50000, type=int)
  parser.add_argument('--pretrain', default='false', help='true or false')
  args = parser.parse_args()
  return args

# predict the final disparity map
def computeSoftArgMin(logits):
  softmax = tf.nn.softmax(logits)
  disp = tf.range(1, DISPARITY+1, 1)
  disp = tf.cast(disp, tf.float32)
  disp_mat = []
  for i in xrange(WIDTH*HEIGHT):
    disp_mat.append(disp)
  disp_mat = tf.reshape(tf.stack(disp_mat), [HEIGHT, WIDTH, DISPARITY])
  result = tf.multiply(softmax, disp_mat)
  result = tf.reduce_sum(result, 2)
  return result

def loss(logits, labels):
  mask = tf.cast(labels<=DISPARITY, dtype=tf.bool)
  loss_ = tf.abs(tf.subtract(logits, labels))
  loss_ = tf.where(mask, loss_, tf.zeros_like(loss_))
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  loss_sum = tf.reduce_sum(loss_)
  mask = tf.cast(mask, tf.float32)
  loss_mean = tf.div(loss_sum, tf.reduce_sum(mask))

  loss_final = tf.add_n([loss_mean] + regularization_losses)

  return loss_final

def normalizeRGB(img):
  img=img.astype(float)
  for i in range(3):
	  minval=img[:, :, i].min()
	  maxval=img[:, :, i].max()
	  if minval!=maxval:
		  img[:, :, i]=(img[:, :, i]-minval)/(maxval-minval)
  return img		

def train(dataset, args):
  tf.logging.set_verbosity(tf.logging.ERROR)

  with tf.device('/gpu:' + args.gpu):
    left_img = tf.placeholder(tf.float32, shape=(1, None, None, 3))
    right_img = tf.placeholder(tf.float32, shape=(1, None, None, 3))
    labels = tf.placeholder(tf.float32, shape=(None, None))

    logits = inference(left_img, right_img, is_training=True)
    logits = tf.transpose(logits, [1, 2, 0])
    
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    result = computeSoftArgMin(logits)

    loss_ = loss(result, labels)

    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # keep moving average for loss
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    # tf.scalar_summary('loss;', ema.average(loss_))
    tf.summary.scalar('loss: ', loss_)

    optimizer = tf.train.RMSPropOptimizer(args.learning_rate, decay=0.9, epsilon=1e-8)
    gradients = optimizer.compute_gradients(loss_)
    # this step also adds 1 to global_step
    apply_gradient_optimizer = optimizer.apply_gradients(gradients, global_step=global_step)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)

    # train op contains optimizer, batchnorm and averaged loss
    train = tf.group(apply_gradient_optimizer, batchnorm_updates_op)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.initialize_all_variables()
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type='BFC'

    sess = tf.Session(config=config)
    # keep training from last checkpoint
    if args.pretrain == 'true':
      new_saver = tf.train.Saver()
      print("================== Loading checkpoint ==================")
      new_saver.restore(sess, tf.train.latest_checkpoint(args.output_dir))
    else:
      sess.run(init)

    summary_writer = tf.summary.FileWriter(args.output_dir+'log', sess.graph)

    print("================== Start training ==================")
    # x <= 22390
    for x in xrange(args.max_steps + 1):
      start_time = time.time()

      with open("train.lst", "r") as lst:
        img = lst.readlines()[x]
        img = img.strip()
	img = img.split('\t')
      
      imgL = np.array(Image.open(img[0]).convert("RGB"))
      print("After open(imgL size): " + str(np.shape(imgL)))
      imgL = normalizeRGB(imgL)
      imgL = imgL[:, :, :]
      print("Before expand(imgL size): " + str(np.shape(imgL)))
      imgL = np.expand_dims(imgL, axis=0)
      print("After expand(imgL size): " + str(np.shape(imgL)))

      imgR = np.array(Image.open(img[1]).convert("RGB"))
      imgR = normalizeRGB(imgR)
      imgR = imgR[:, :, :]
      imgR = np.expand_dims(imgR, axis=0)

      imgGround = readPFM.load_pfm(img[2])
      imgGround = imgGround[:, :]
      print("imgGround size: " + str(np.shape(imgGround)))

      step = sess.run(global_step)
      i = [loss_]

      write_summary = step % 100
      if write_summary or step==0:
        i.append(summary_op)
      print("Before sess.run")
      output = sess.run(i, feed_dict={left_img: imgL, right_img: imgR, labels: imgGround})
      print("After sess.run")
      loss_value = output[0] 

      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        format_str = ('step %d, loss = %.2f (%.3f sec/batch)')
        print(format_str % (step, loss_value, duration))

      # Save the model checkpoint periodically.
      if (step > 1 and step % 200 == 0) or step == args.max_steps:
        checkpoint_path = os.path.join(args.output_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=global_step)

def test(dataset, args):
    # new_saver = tf.train.import_meta_graph("model.ckpt-141601.meta")
    
    # construct exactly same variables as training phase
    print "================ Building the same network as training phase ==============="
    left_img = tf.placeholder(tf.float32, shape=(1, HEIGHT, WIDTH, None))
    right_img = tf.placeholder(tf.float32, shape=(1, HEIGHT, WIDTH, None))
    labels = tf.placeholder(tf.float32, shape=(HEIGHT, WIDTH))

    logits = inference(left_img, right_img, is_training=True)
    logits = tf.transpose(logits, [1, 2, 0])
    
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    result = computeSoftArgMin(logits)

    loss_ = loss(result, labels)
    
    sess = tf.Session()
    new_saver = tf.train.Saver()
    print "================= Loading checkpoint =============="
    new_saver.restore(sess, tf.train.latest_checkpoint(output_dir))

    with open("train.lst", "r") as lst:
      randomNumber = 0
      randomH = 0
      randomW = 0
      img = lst.readlines()[randomNumber]
      img = img.strip()
      img = img.split('\t')
      
      imgL = np.array(Image.open(img[0]))
      imgL = normalizeRGB(imgL)
      imgL = imgL[randomH:randomH+HEIGHT, randomW:randomW+WIDTH, :]
      imgL = np.expand_dims(imgL, axis=0)
     
      imgR = np.array(Image.open(img[1]))
      imgR = normalizeRGB(imgR)
      imgR = imgR[randomH:randomH+HEIGHT, randomW:randomW+WIDTH, :]
      imgR = np.expand_dims(imgR, axis=0)

      imgGround = readPFM.load_pfm(img[2])
      imgGround = imgGround[randomH:randomH+HEIGHT, randomW:randomW+WIDTH]
      print imgGround
    
    print "Start testing for one image.................."  
    o = sess.run([result, loss_], feed_dict={left_img: imgL, right_img: imgR, labels: imgGround})
    test_img = o[0].squeeze()
    test_img = np.asarray(test_img, np.uint8)
    test_img = Image.fromarray(test_img)
    print "result is: ", o[0]
    print "loss is: ", o[1]
    print "Saving test.png..................."
    test_img.save("test.png")
 

def main(_):
  global HEIGHT
  global WIDTH

  dataset = "/data/jack/flyingthings3d/"
  args = parse_args()

  if args.phase == 'train':
    train(dataset, args)
  else:
    test(dataset, args)

if __name__ == '__main__':
  tf.app.run()

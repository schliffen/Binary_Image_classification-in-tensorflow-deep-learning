#
#
#

import sys, os
import tensorflow as tf
from mynet.mynet_training import train_and_validate
from mynet.mynet_testing import test_cnn_net
#
cwd = os.getcwd()
sys.path.append(os.path.realpath(cwd))
tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.app.flags
#
# parameter setting with flags
flags.DEFINE_bool('train', True, 'run training')
flags.DEFINE_bool('test', False, 'run testing')
#
flags.DEFINE_integer('num_classes', 2, 'number of classes')
flags.DEFINE_integer('channels', 3, 'number of channels')
flags.DEFINE_integer('epochs', 500, 'number of epochs')
flags.DEFINE_integer('num_steps', 100, 'number of iteration steps')
flags.DEFINE_integer('num_tests', 10, 'number of testing iamges (maxumam number is 81)')
flags.DEFINE_integer('batch_size', 10, 'batch size of training data')
flags.DEFINE_integer('valid_batch', 40, 'batch size of validation data')
flags.DEFINE_float('learning_rate', .005, 'initial learning rate')
flags.DEFINE_integer('showing_step', 10, 'after how many steps to show results')
flags.DEFINE_integer('model_saving_step', 2000, 'after how many steps to save results automatically')
flags.DEFINE_integer('n_threads', 1, 'number of threads')
flags.DEFINE_bool('tensorboard', False, 'whether to load tensorboard in testing')
#
flags.DEFINE_string('log_path', cwd + '/mynet/log/', 'path to my models log (training)')
flags.DEFINE_string('chkpnt_path', cwd + '/mynet/chkpnt', 'path to my models checkpoints')
flags.DEFINE_string('traning_path', cwd + '/making_data/training_susand_01.tfrecords', 'path to training data')
flags.DEFINE_string('validate_path', cwd + '/making_data/validate_susand_01.tfrecords', 'path to my validation data')
flags.DEFINE_string('testing_path', cwd + '/making_data/testing_susand_01.tfrecords', 'path to my testing data')
#
FLAGS = flags.FLAGS
#
# loading training function on my network ----
if FLAGS.train:
    train_and_validate(FLAGS)

# testing my network ----
if FLAGS.test:
    test_cnn_net(FLAGS)

# defining CNN MODEL
#
# imports
import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from tensorflow.contrib.slim import fully_connected as dencel
from tensorflow.contrib.slim import conv2d
#
# predifining filters and biases for conv layers for clean coding
class model_utils():
    def __init__(self, FLAGS):
        # MODEL specific Configurations
        self.flag = FLAGS
        self.SEED = 1415926 # seed for random numbre generation!! lucky number!! (recall pi)
        self.patch_size = 5
        self.depth = 4
        self.num_hidden = 32
        self.dropout_prob = 0.75
        self.conv_layers = 3
        self.stddev = 0.2
        self.dim1 = 7*7*512
        self.dim2 = 1024
        self.dim3 = 1024
        self.weight_bias = {
            'wt_1' : tf.Variable(tf.random_normal([self.dim1, self.dim2], stddev=0.1), name='wt_1'),
            'bs_1' : tf.Variable(tf.random_normal([self.dim2], stddev=0.1), name='bs_1'),
            'wt_2' : tf.Variable(tf.random_normal([self.dim2, self.dim3], stddev=0.1), name='wt_2'),
            'bs_2' : tf.Variable(tf.random_normal([self.dim3], stddev=0.1), name='bs_2'),
            'wt_3' : tf.Variable(tf.random_normal([self.dim3, self.flag.num_classes], stddev=0.1), name='wt_3'),
            'bs_3' : tf.Variable(tf.random_normal([self.flag.num_classes], stddev=0.1), name='bs_3')
        }
#
    # parsing data
    def parse_record(self, buff_data, repeat, is_train):
        #
        reader = tf.TFRecordReader()
        fname_queue = tf.train.string_input_producer([buff_data], num_epochs=repeat)
        _,serialized_example = reader.read(fname_queue)
        # namig root according to type of dataset
        if is_train:
            root  = 'training'
        else:
            root  = 'validate'
        feature = {
            root + '/image' : tf.FixedLenFeature([], dtype=tf.string),
            root + '/label' : tf.FixedLenFeature([], dtype=tf.string),
            root + '/height': tf.FixedLenFeature([], dtype=tf.int64),
            root + '/width' : tf.FixedLenFeature([], dtype=tf.int64)
        }
        # reading height and weight of image as numpy format
        reader = tf.python_io.tf_record_iterator(buff_data)
        # as the shape of all data are same, I extract the shape from one of them
        for s in reader:
            s_img = tf.train.Example().FromString(s)
            height = s_img.features.feature[root +'/height'].int64_list.value[0]
            width = s_img.features.feature[root + '/width'].int64_list.value[0]
            break
        # loading image and label data as tensors to feed them directly
        record = tf.parse_single_example(serialized_example, features = feature)
        image_dt = tf.decode_raw(record[root + '/image'], tf.float32)
        image_dt = tf.reshape(image_dt, shape=(height, width, self.flag.channels))
        image_dt = tf.image.per_image_standardization(image_dt)
        label = tf.decode_raw(record[root + '/label'], tf.float32)
        label = tf.reshape(label, [2])
        return  image_dt, label

    def build_filter(self, shape, name):
        filter = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(filter, name=name)

    def build_bias(self, shape, name):
        bias= tf.constant(0.1, shape=(shape,))
        return tf.Variable(bias, name=name)

    def img_preproc(self, img):
        # do preprocessing here
        return img

    def maxpool_layer(self, img):
        return tf.nn.max_pool(img, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # convolution layer
    def build_conv_layer(self, img, shape, name):
        conv = tf.nn.conv2d(img, self.build_filter(shape, name), strides=[1, 1, 1, 1], padding='SAME')
        conv = slim.batch_norm(conv)
        bias = tf.nn.bias_add(conv, self.build_bias(shape[3], name))
            # using RELU activation function
        relu = tf.nn.relu(bias)
        return relu
    # starting main model
    def cnn_net(self, img):
        #Convolutional Layer 1
        conv_1 = self.build_conv_layer(img,[6,6,3,32],   'conv_1')
        #Pooling Layer 1
        conv_1_pooling = self.maxpool_layer(conv_1)
        #Convolutional Layer 2
        conv_2 = self.build_conv_layer(conv_1_pooling, [6,6,32,64],  'conv_2')
        #Pooling Layer 2
        conv_2_pooling = self.maxpool_layer(conv_2)
        #Convolutional Layer 3
        conv_3 = self.build_conv_layer(conv_2_pooling,[6,6,64,64], 'conv_3')
        #Pooling Layer 3
        conv_3_pooling = self.maxpool_layer(conv_3)
        #Convolutional Layer 4 + RELU
        conv_4 = self.build_conv_layer(conv_3_pooling, [6,6,64,128], 'conv_4')
        #Pooling Layer 4
        conv_4_pooling = self.maxpool_layer(conv_4)
        #
        # Entering to Dence layers
        #
        conv_4_flat = tf.reshape(conv_4_pooling,[-1,7*7*512])
        #
        full_layer_one = tf.nn.relu(tf.add(tf.matmul(conv_4_flat, self.weight_bias['wt_1']), self.weight_bias['bs_1']))
        #Dropout Layer 1
        full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=.85)
        #F C 1
        full_layer_two = tf.nn.relu(tf.add(tf.matmul(full_one_dropout, self.weight_bias['wt_2']), self.weight_bias['bs_2']))
        #Dropout Layer 1
        full_two_dropout = tf.nn.dropout(full_layer_two,keep_prob=.5)
        #Output Layer,containing 2 output nodes.
        logits  = tf.add(tf.matmul(full_two_dropout, self.weight_bias['wt_3']), self.weight_bias['bs_3'])
        return logits, self.weight_bias
    # Auto Encoder
    def Encode(self, input_img):
        # I will use conv layer to imporove performance
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE) as scope:
            conv_l = conv2d(input_img, 32, [3,3], 2, weights_regularizer=slim.l1_l2_regularizer(.001),
                            scope='conv_1')
            conv_l = conv2d(conv_l, 64, [3, 3], 2, weights_regularizer=slim.l1_l2_regularizer(.005),
                            scope='conv_2')
            conv_l = conv2d(conv_l, 128, [3, 3], 4, weights_regularizer=slim.l1_regularizer(.005),
                            scope=scope)
        return conv_l
    # Decoder
    def Decode(self, Decoded_img):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as scope:
            conv_d = slim.conv2d_transpose(Decoded_img, 64, [3, 3], 4,
                                           weights_regularizer=slim.l1_l2_regularizer(.001),
                                           scope='dconv_1')
            conv_d = slim.conv2d_transpose(conv_d, 32, [3, 3], 2,
                                           weights_regularizer=slim.l1_l2_regularizer(.005),
                                           scope='dconv_2')
            conv_d = slim.conv2d_transpose(conv_d, 3, [3, 3], 2,
                                           weights_regularizer=slim.l1_regularizer(.005),
                                           scope=scope)
        return conv_d



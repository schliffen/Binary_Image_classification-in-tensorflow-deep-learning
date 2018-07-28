#
# in this file I'm goinf to test trained networks
#
# imports
import sys, os, glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
# appending dirs
# configuration

# preparing testing data
def data_generator(FLAGS):
    feature = {'testing/image': tf.FixedLenFeature([], dtype=tf.string),
               'testing/label': tf.FixedLenFeature([], dtype=tf.string),
               'testing/height': tf.FixedLenFeature([], dtype=tf.int64),
               'testing/width': tf.FixedLenFeature([], dtype=tf.int64)
               }
    reader = tf.python_io.tf_record_iterator(FLAGS.testing_path)
    for samples in reader:
        # getting sample data
        sample_data = tf.train.Example().FromString(samples)
        # getting the shape of images
        height = sample_data.features.feature['testing/height'].int64_list.value[0]
        width = sample_data.features.feature['testing/width'].int64_list.value[0]
        shape = (height, width, FLAGS.channels)
        # extracting image
        s_image = np.fromstring(sample_data.features.feature['testing/image'].
                                      bytes_list.value[0], dtype=np.float32).reshape(shape)
        # extracting label
        s_label = np.fromstring(sample_data.features.feature['testing/label'].
                                      bytes_list.value[0], dtype=np.float32)
        yield s_image, s_label

def representer(img, treu_label, pred_label):

        true_name = 'sushi' if int(np.argmax(treu_label))==0 else 'sandwich'
        # feeding data to trained network
        l_name = 'sushi'  if np.argmax(pred_label[0][0])==0 else  'sandwich'
        print('Predicted probability {}, Predicted label {},True name: {}'
              .format(np.max(pred_label[0][0]), l_name, true_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.title('Classified Image')
        color = 'g' if l_name==true_name else 'red'
        plt.text(-0, -10, r'Predic: ' + l_name, fontsize=10, color = color)
        plt.text(170, -10, r'True: ' + true_name, fontsize=10, color = 'g')
        plt.imshow(img)
        plt.show()


def test_cnn_net(FLAGS):
    prediction_graph = tf.Graph()
    with prediction_graph.as_default():
        # loading netweork -----------------------
        assert FLAGS.chkpnt_path, '`train_dir` is missing.'
        ckp_net = tf.train.latest_checkpoint(FLAGS.chkpnt_path)
        # to check this
        ckpt = tf.train.get_checkpoint_state(FLAGS.chkpnt_path)
        meta_file = ckp_net.split('/')[-1]

        with tf.Session(graph=prediction_graph) as sess:
            # Load the graph with the trained states
            saver = tf.train.import_meta_graph(FLAGS.chkpnt_path + '/' + meta_file + '.meta', import_scope=None)
            saver.restore(sess, ckp_net)
            # Get the tensors by their variable name
            image_tensor = prediction_graph.get_tensor_by_name('input_port:0')
            #image_tensor = prediction_graph.get_tensor_by_name('input_port:0')
            y_out =  prediction_graph.get_tensor_by_name('training/softmax_output:0')
            #
            data_gen = data_generator(FLAGS)
            # to initialize the parameters of the loaded graph
            for sample in range(FLAGS.num_tests):
                img, label = next(data_gen)
                output = sess.run([y_out], feed_dict={image_tensor: np.expand_dims(img, axis=0)})
                # representing results
                representer(img, label, output)


#if __name__ == '__main__':
#    test_cnn_net(FLAGS)
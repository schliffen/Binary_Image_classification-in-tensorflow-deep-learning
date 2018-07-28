#
# This code is applied for training
#
# imports
import sys, os, glob
import tensorflow as tf
import numpy as np
from mynet.model_util import model_utils as m_util

# =============================== Importing data for train ===============================
# parsing tfrecords and getting Tensor form of images and labels
def train_and_validate(FLAGS):
    util = m_util(FLAGS)
    train_img, train_label = util.parse_record(FLAGS.traning_path, FLAGS.epochs*FLAGS.num_steps, True)
    # making queue
    img_input, label_input  = tf.train.shuffle_batch([train_img, train_label],
                                                     batch_size=FLAGS.batch_size, num_threads=FLAGS.n_threads,
                                                     #enqueue_many=True,
                                                     allow_smaller_final_batch=True,
                                                     capacity= FLAGS.epochs + 3 * FLAGS.batch_size,
                                                     min_after_dequeue=5, name='input_port')
    # queue runner for validation
    # I did the same process for preparing validation data
    valid_img, valid_label = util.parse_record(FLAGS.validate_path, FLAGS.epochs*FLAGS.num_steps, False) #epochs*num_steps//showing_step
    val_img_input, val_label_input  = tf.train.shuffle_batch([valid_img, valid_label],
                                                             batch_size=FLAGS.valid_batch, num_threads=FLAGS.n_threads,
                                                             #enqueue_many=True,
                                                             allow_smaller_final_batch=True,
                                                             capacity = FLAGS.epochs + 3 * FLAGS.valid_batch,
                                                             min_after_dequeue=5, name='val_input_port')
    #
    # =============================== TRAIN VARIABLES AND MEASURES ===============================
    # To increase performance I did not use placeholders and feeding process
    # importing model and other training functions
    #
    with tf.name_scope('training'):
        # encoding images
        #coded_img = util.Decode(util.Encode(img_input))
        # coded_img
        # feeding decoder results
        logits, weigh_bias = util.cnn_net(img_input) #  train_holder
        prob = tf.nn.softmax(logits, name='softmax_output') # this output is to load for testing
    #
    with tf.name_scope('loss') as scope:
        # network loss
        ur_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label_input), name=scope) # train_labels
        # Lyanpunov (L2) type regularization to avoid overfitting problems in training
        regularizers = (tf.nn.l2_loss(weigh_bias['wt_1']) +
                    tf.nn.l2_loss(weigh_bias['bs_1']) +
                    tf.nn.l2_loss(weigh_bias['wt_2']) +
                    tf.nn.l2_loss(weigh_bias['bs_2']) +
                    tf.nn.l2_loss(weigh_bias['wt_3']) +
                    tf.nn.l2_loss(weigh_bias['bs_3']))
        # regularized loss
        loss = ur_loss + 5e-4 * regularizers
        #
        # Auto Encoders loss
        #ae_loss = tf.losses.mean_squared_error(img_input, coded_img)
        ae_loss = 0
        # total loss
        total_loss = loss + ae_loss

    # total optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)
    # prediction
    with tf.name_scope('Accuracy'):
        val_logit = util.cnn_net(val_img_input)[0]
        predictions = tf.argmax(val_logit, 1)
        true_label = tf.argmax(val_label_input, 1)
        equality = tf.equal(predictions, true_label)
        # my accuracy
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        # accuracy with tf metrics
        tfmetrics = tf.metrics.accuracy(true_label, predictions)[1]
    #
    #    ==================================== TRAINING ITERATIONS ================================
    # defining saver to keep checkpoints
    saver = tf.train.Saver(max_to_keep=3,  keep_checkpoint_every_n_hours=2)
    # checking for existing meta files
    pre_chkpnt = tf.train.latest_checkpoint(FLAGS.chkpnt_path)
    #
    with tf.Session() as sess:
        # writing for tensorboard
        tn_board_writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)
        # preparing summaries
        with tf.name_scope("summaries"):
            # reporting network loss
            tf.summary.scalar("ae_loss", ae_loss)
            # reporting network loss
            tf.summary.scalar("net_loss", loss)
            tf.summary.histogram("histogram_loss",loss)
            tf.summary.scalar("accuracy", accuracy)
            tf.summary.histogram("histogram_accuracy", accuracy)
            tf.summary.scalar("tf_metrics", tfmetrics)
        #
        merge_all = tf.summary.merge_all() # merging summaries
        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        sess.run(init_op)
        #
        # checking if checkpoint exist to load and retrain it
        try:
            meta_file = pre_chkpnt.split('/')[-1]
            restorer = tf.train.import_meta_graph(FLAGS.chkpnt_path + '/' + meta_file + '.meta')
            restorer.restore(sess, pre_chkpnt)
            print('saved model is loaded to continue training ...')
        except:
            print('meta file was not found, training from scratch ...')
        # defining queue runner
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for epoch in range(FLAGS.epochs):
            try:
                for step in range(FLAGS.num_steps):
                    # feeding data
                    results,  train_loss, net_loss, _ = sess.run([merge_all, loss, total_loss,  optimizer]) #
                    # saving data for representation
                    tn_board_writer.add_summary(results, step + FLAGS.num_steps*FLAGS.epochs) #
                    if step % FLAGS.showing_step== 0:
                        # validation
                        valid_accuracy, tfmeter = sess.run([accuracy , tfmetrics])
                        # saving checkpoints here
                        saver.save(sess, FLAGS.chkpnt_path+'/smdl', write_meta_graph=True, write_state=True, meta_graph_suffix= 'meta',  global_step=step + FLAGS.num_steps*epoch)
                        # printing results
                        print('epoch {}, step {}, train loss: {}'.format(epoch, step, train_loss))
                        print('network loss: {}, auto encoder loss: {}'.format(net_loss, train_loss - net_loss))
                        print('-------- validation accuracy: {}, tf accuracy metic {}'.format(valid_accuracy, tfmeter))
                        #
            except tf.errors.OutOfRangeError:
                print('there is not enough data to feed')
        # saving at the end of training loops
        saver.save(sess, FLAGS.chkpnt_path+'/smdl', write_meta_graph=True, write_state=True, meta_graph_suffix= 'meta', global_step=step + FLAGS.num_steps*epoch)

    os.system('tensorboard --logdir=FLAGS.log_path')
    print('Training is Done!!')


#output = sess.run('summing:0')
#print(output)
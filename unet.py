import math
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time

from config import NET_INPUT_SIZE
from data import full_pipeline


def get_time():
    return time.strftime("%y_%m_%d__%H_%M_%S", time.gmtime())

def log(writer, value, name, step):
    summary_ = tf.Summary(value=[
        tf.Summary.Value(tag=name, simple_value=value),
    ])
    writer.add_summary(summary_, step)
    writer.flush()

def logs(writer, values, names, step):
    for val, name in zip(values, names):
        log(writer, val, name, step)


class UnetTrainer(object):
    def __init__(self, conv_layers=None, conv_kernel_sizes=None, conv_batch_norms=None,
                 learning_rate=0.01, optimizer='adam'):
        # if conv_layers is None:
        #     conv_layers = [16, 32]
        # if conv_kernel_sizes is None:
        #     conv_kernel_sizes = [3 for _ in conv_layers]
        #
        # assert len(conv_layers) == len(conv_batch_norms)
        #
        self.learning_rate = learning_rate
        # self.conv_layers = conv_layers
        # self.fc_layers = fc_layers
        # self.conv_kernel_sizes = conv_kernel_sizes
        # self.conv_batch_norms = conv_batch_norms
        # self.fc_batch_norms = fc_batch_norms
        self.optimizer = optimizer

    def train_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.train_step, self.loss, self.accuracy],
                                feed_dict={self.x: batch_xs, self.y_target: batch_ys, self.is_training: True})
        return results[1:]

    def validate_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.loss, self.accuracy],
                                feed_dict={self.x: batch_xs, self.y_target: batch_ys, self.is_training: False})
        return results

    def downscale(self, signal):
        return tf.layers.max_pooling2d(signal, pool_size=2, strides=2)

    def upscale(self, signal, kernel_size=3):
        num_filters = signal.get_shape()[-1] // 2
        return tf.layers.conv2d_transpose(
            inputs=signal,
            filters=num_filters,
            strides=2,
            kernel_size=kernel_size,
            padding='SAME'
        )

    def conv_bn_relu(self, signal, num_filters, kernel_size=3):
        signal = tf.layers.conv2d(
            inputs=signal,
            filters=num_filters,
            kernel_size=kernel_size,
            padding='SAME'
        )
        # TODO: batch norm
        # if use_batch_norm:
        #     signal = self.apply_batch_normalization(signal)
        return tf.nn.relu(signal)

    def conv_1x1(self, signal, num_filters):
        return tf.layers.conv2d(
            inputs=signal,
            filters=num_filters,
            kernel_size=1,
            padding='SAME'
        )
        # TODO: batch norm?
        return tf.nn.relu(signal)

    def create_model(self):
        self.x = tf.placeholder(tf.float32, [None, NET_INPUT_SIZE[0], NET_INPUT_SIZE[1], 1], name='x')
        self.y_target = tf.placeholder(tf.int64, [None, NET_INPUT_SIZE[0], NET_INPUT_SIZE[1]], name='y_target')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        signal = self.x
        print_shape = lambda: print('shape', signal.get_shape())

        num_filters = 66
        kernel_size = 3

        layers_stack = []
        push = lambda: layers_stack.append(signal)
        pop = lambda: layers_stack.pop()

        # for idx, (num_filters, kernel_size, use_batch_norm) \
        #         in enumerate(zip(self.conv_layers, self.conv_kernel_sizes, self.conv_batch_norms)):

        # Conv layers
        print('Going down')
        print_shape()
        signal = self.conv_bn_relu(signal, 64)
        signal = self.conv_bn_relu(signal, 64)
        print_shape()
        push()
        signal = self.downscale(signal)
        print_shape()
        signal = self.conv_bn_relu(signal, 128)
        signal = self.conv_bn_relu(signal, 128)

        print('Going up')
        print_shape()
        signal = self.upscale(signal)
        high_res_layer = pop()
        print('Upscaled layer shape:', signal.get_shape())
        print('Previous layer shape:', high_res_layer.get_shape())
        signal = tf.concat([high_res_layer, signal], axis=-1)
        print_shape()
        signal = self.conv_bn_relu(signal, 64)
        signal = self.conv_bn_relu(signal, 64)
        print_shape()

        # Output
        signal = self.conv_1x1(signal, 66)
        print_shape()

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=signal, labels=self.y_target))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_target, tf.argmax(signal, axis=-1)), tf.float32))

        if self.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
        else:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Include batch norm updates
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(self.loss)

        print('list of variables', list(map(lambda x: x.name, tf.global_variables())))

    def store_parameters(self, filename):
        params = [
            # self.conv_layers,
            # self.fc_layers,
            # self.conv_kernel_sizes,
            # self.conv_batch_norms,
            # self.fc_batch_norms,
            self.learning_rate,
            self.optimizer
        ]
        with open(filename, 'w') as f:
            f.write(str(params) + '\n')

    def train(self, epochs_n=100, dir_base='out', save_path=None):
        self.create_model()

        log_dir_base = dir_base + '/unet' + get_time() + '/'
        os.makedirs(os.path.dirname(log_dir_base), exist_ok=True)
        self.store_parameters(log_dir_base + 'params')

        train_dataset, valid_dataset, batches_per_epoch_train, batches_per_epoch_valid = full_pipeline()
        train_batch_getter = train_dataset.make_one_shot_iterator().get_next()
        valid_batch_getter = valid_dataset.make_one_shot_iterator().get_next()

        with tf.Session() as self.sess:
            summary_writer = tf.summary.FileWriter(log_dir_base + 'train/', self.sess.graph)
            valid_summary_writer = tf.summary.FileWriter(log_dir_base + 'valid/', self.sess.graph)

            tf.global_variables_initializer().run()  # initialize variables

            losses = []
            try:

                for epoch_idx in range(epochs_n):
                    print('Epoch', epoch_idx, 'starts')
                    train_losses = []
                    valid_losses = []
                    for batch_idx in range(batches_per_epoch_train):
                        batch_xs, batch_ys = self.sess.run(train_batch_getter)
                        # TODO: connect ^ v ^ v ^
                        vloss = self.train_on_batch(batch_xs, batch_ys)
                        train_losses.append(vloss)

                        print('    Batch', batch_idx, vloss)

                    for batch_idx in range(batches_per_epoch_valid):
                        batch_xs, batch_ys = self.sess.run(valid_batch_getter)
                        vloss = self.validate_on_batch(batch_xs, batch_ys)
                        valid_losses.append(vloss)
                        print('    [VALID] Batch', batch_idx, vloss)

                    print('Epoch', epoch_idx, 'ended')
                    epoch_train_stats = np.mean(np.asarray(train_losses), axis=1)
                    epoch_valid_stats = np.mean(np.asarray(train_losses), axis=1)

                    print('Epoch training:', epoch_train_stats)
                    print('Epoch validation:', epoch_valid_stats)
                    logs(summary_writer, epoch_train_stats, ['loss', 'acc'], epoch_idx)
                    logs(valid_summary_writer, epoch_valid_stats, ['loss', 'acc'], epoch_idx)

                    # if batch_idx % 100 == 0:
                    #     print('Batch {batch_idx}: mean_loss {mean_loss}'.format(
                    #         batch_idx=batch_idx, mean_loss=np.mean(losses[-200:], axis=0))
                    #     )
                    #     test_results = self.sess.run([self.loss, self.accuracy],
                    #                                         feed_dict={self.x: mnist.test.images,
                    #                                                    self.y_target: mnist.test.labels})
                    #     print('Test results', test_results)
                    #     logs(test_summary_writer, test_results, ['loss', 'acc'], batch_idx)


            except KeyboardInterrupt:
                print('Stopping training!')
                pass

            # # Test trained model
            # print('Test results', self.sess.run([self.loss, self.accuracy], feed_dict={self.x: mnist.test.images,
            #                                     self.y_target: mnist.test.labels}))

            if save_path is not None:
                self.store(save_path)

    def store(self, path='./model/model.ckpt'):
        """ Must be called within a started session. """
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path)
        print("Model saved in path: %s" % save_path)

    def restore(self, path='./model/model.ckpt'):
        """ Must be called within a started session. """
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        print("Model restored.")

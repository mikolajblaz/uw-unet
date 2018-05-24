import math
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time

import config
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
        signal = tf.layers.conv2d_transpose(
            inputs=signal,
            filters=num_filters,
            strides=2,
            kernel_size=kernel_size,
            padding='SAME'
        )
        signal = tf.layers.batch_normalization(signal, training=self.is_training)
        return tf.nn.relu(signal)

    def conv_bn_relu(self, signal, num_filters, kernel_size=3):
        signal = tf.layers.conv2d(
            inputs=signal,
            filters=num_filters,
            kernel_size=kernel_size,
            padding='SAME'
        )
        signal = tf.layers.batch_normalization(signal, training=self.is_training)
        return tf.nn.relu(signal)

    def conv_1x1(self, signal, num_filters):
        return tf.layers.conv2d(
            inputs=signal,
            filters=num_filters,
            kernel_size=1,
            padding='SAME'
        )
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

        self.filters_nums = filters_nums = [64, 128, 256, 512, 1024]
        last_level = filters_nums[-1]
        filters_nums = filters_nums[:-1]

        # for idx, (num_filters, kernel_size, use_batch_norm) \
        #         in enumerate(zip(self.conv_layers, self.conv_kernel_sizes, self.conv_batch_norms)):

        # Conv layers
        print('Going down')
        print_shape()
        for num_filters in filters_nums:
            signal = self.conv_bn_relu(signal, num_filters)
            signal = self.conv_bn_relu(signal, num_filters)
            print_shape()
            push()
            signal = self.downscale(signal)
            print_shape()


        signal = self.conv_bn_relu(signal, last_level)
        signal = self.conv_bn_relu(signal, last_level)

        print('Going up')
        print_shape()
        for num_filters in filters_nums[::-1]:
            signal = self.upscale(signal)
            high_res_layer = pop()
            print('Upscaled layer shape:', signal.get_shape())
            print('Previous layer shape:', high_res_layer.get_shape())
            signal = tf.concat([high_res_layer, signal], axis=-1)
            print_shape()
            signal = self.conv_bn_relu(signal, num_filters)
            signal = self.conv_bn_relu(signal, num_filters)
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

        print('list of variables', list(map(lambda x: x.name, tf.global_variables())), flush=True)

    def store_parameters(self, filename):
        params = [
            self.learning_rate,
            self.optimizer,
            self.filters_nums,
            config.NET_INPUT_SIZE,
            config.BATCH_SIZE
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

            try:
                start_time = new_time = time.time()
                for epoch_idx in range(epochs_n):
                    last_time = new_time
                    print('Epoch', epoch_idx, 'starts', flush=True)
                    train_losses = []
                    valid_losses = []
                    for batch_idx in range(batches_per_epoch_train):
                        batch_xs, batch_ys = self.sess.run(train_batch_getter)
                        # TODO: connect ^ v ^ v ^
                        vloss = self.train_on_batch(batch_xs, batch_ys)
                        train_losses.append(vloss)
                        if batch_idx % 50 == 0:
                            print('    Batch {}/{}: {}'.format(batch_idx, batches_per_epoch_train, np.mean(train_losses, axis=0)), flush=True)

                    for batch_idx in range(batches_per_epoch_valid):
                        batch_xs, batch_ys = self.sess.run(valid_batch_getter)
                        vloss = self.validate_on_batch(batch_xs, batch_ys)
                        valid_losses.append(vloss)
                        if batch_idx % 50 == 0:
                            print('    [VALID] Batch {}/{}: {}'.format(batch_idx, batches_per_epoch_valid, np.mean(valid_losses, axis=0)), flush=True)

                    new_time = time.time()
                    print('Epoch', epoch_idx, 'ended after', int(new_time - last_time), 'seconds', flush=True)
                    epoch_train_stats = np.mean(train_losses, axis=0)
                    epoch_valid_stats = np.mean(valid_losses, axis=0)

                    print('Epoch training:', epoch_train_stats)
                    print('Epoch validation:', epoch_valid_stats)
                    print()
                    logs(summary_writer, epoch_train_stats, ['loss', 'acc'], epoch_idx)
                    logs(valid_summary_writer, epoch_valid_stats, ['loss', 'acc'], epoch_idx)

            except KeyboardInterrupt:
                print('Stopping training!')

            training_time = time.time() - start_time
            print('Training ended after', int(training_time), 'seconds')

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

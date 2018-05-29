import math
import numpy as np
import os
import tensorflow as tf
import time

import config
from config import NET_INPUT_SIZE
from data import full_pipeline
from palette import PALETTE


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
    def __init__(self, filters_nums=None, learning_rate=0.01, optimizer='adam'):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        if filters_nums is None:
            filters_nums = [64, 128, 256, 512, 1024]
        self.filters_nums = filters_nums

    def train_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.train_step, self.loss, self.accuracy],
                                feed_dict={self.x: batch_xs, self.y_target: batch_ys, self.is_training: True})
        return results[1:]

    def validate_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.valid_loss, self.valid_accuracy, self.images_summ],
                                feed_dict={self.x: batch_xs, self.y_target: batch_ys, self.is_training: False})
        return results[:2], results[2]

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

        layers_stack = []
        push = lambda: layers_stack.append(signal)
        pop = lambda: layers_stack.pop()

        last_level = self.filters_nums[-1]
        filters_nums = self.filters_nums[:-1]

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

        # Validation
        def unbatch(signal):
            output_shape = tf.shape(signal)
            # shape: (8, x, x, 66)
            signal = tf.reshape(signal, (-1, config.AUGMENT_FACTOR, output_shape[1], output_shape[2], output_shape[3]))
            # shape: (4, 2, x, x, 66)
            return signal

        output_shape = tf.shape(signal)
        valid_grouped_output = tf.reshape(signal, (-1, config.AUGMENT_FACTOR, output_shape[1], output_shape[2], output_shape[3]))
        print('Validation shape:', valid_grouped_output.get_shape())
        normal = valid_grouped_output[:, 0]
        unflipped = tf.reverse(valid_grouped_output[:, 1], axis=[-2])
        unaugmented = tf.stack([normal, unflipped])
        # shape: (2, 4, x, x, 66)
        valid_mean_output = tf.reduce_mean(unaugmented, axis=0)
        # shape: (4, x, x, 66)

        labels_shape = tf.shape(self.y_target)
        # shape: (8, x, x)
        labels_unbatched =  tf.reshape(self.y_target, (-1, config.AUGMENT_FACTOR, labels_shape[1], labels_shape[2]))
        # shape: (4, 2, x, x)
        labels = labels_unbatched[:, 0]
        # shape: (4, x, x)

        self.valid_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=valid_mean_output, labels=labels))
        self.valid_predictions = tf.argmax(valid_mean_output, axis=-1)
        # shape: (4, x, x)
        self.valid_accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, self.valid_predictions), tf.float32))

        # Image summaries
        input_shape = tf.shape(self.x)
        # shape: (8, x, x, 1)
        input_unbatched =  tf.reshape(self.x, (-1, config.AUGMENT_FACTOR, input_shape[1], input_shape[2]))
        # shape: (4, 2, x, x)
        input = input_unbatched[:, 0]
        # shape: (4, x, x)

        self.images_summ = self.output_image_summaries(input, labels, self.valid_predictions)

        print('list of variables', list(map(lambda x: x.name, tf.global_variables())), flush=True)

    def output_image_summaries(self, input, ground_truth, output):
        # All arguments are tensors of shape: (4, x, x)
        input_rgb = tf.cast(tf.stack([input] * 3, axis=-1), tf.uint8)
        # shape: (4, x, x, 3)

        concat_labels = tf.concat([
            ground_truth,
            output
        ], axis=-1)
        # shape: (4, x, 2*x)

        # Apply palette
        params = np.asarray(PALETTE, dtype='uint8')
        shape = tf.shape(concat_labels)  # (4, x, 2*x)
        rgb_labels = tf.gather_nd(
            params=params,
            indices=tf.reshape(concat_labels, (-1, shape[1], shape[2], 1))
        )
        # (4, x, 2*x, 3)

        concat_all = tf.concat([
            input_rgb,
            rgb_labels
        ], axis=-2)
        # shape: (4, x, 3*x, 3)

        print('Concat images shape:', concat_all.get_shape())
        return tf.summary.image('img', concat_all, max_outputs=config.BATCH_SIZE)

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

    def train(self, epochs_n=100, dir_base='out', save_path='model/model.ckpt'):
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
                        vloss = self.train_on_batch(batch_xs, batch_ys)
                        train_losses.append(vloss)
                        if batch_idx % 50 == 0:
                            print('    Batch {}/{}: {}'.format(batch_idx, batches_per_epoch_train, np.mean(train_losses, axis=0)), flush=True)
                            logs(summary_writer, np.mean(train_losses, axis=0), ['loss', 'acc'], epoch_idx + batch_idx / batches_per_epoch_train)

                    for batch_idx in range(batches_per_epoch_valid):
                        batch_xs, batch_ys = self.sess.run(valid_batch_getter)
                        vloss, images_summ_ = self.validate_on_batch(batch_xs, batch_ys)
                        valid_losses.append(vloss)
                        valid_summary_writer.add_summary(images_summ_, epoch_idx + batch_idx / batches_per_epoch_valid)
                        valid_summary_writer.flush()
                        if batch_idx % 50 == 0:
                            print('    [VALID] Batch {}/{}: {}'.format(batch_idx, batches_per_epoch_valid, np.mean(valid_losses, axis=0)), flush=True)
                            logs(valid_summary_writer, np.mean(valid_losses, axis=0), ['loss', 'acc'], epoch_idx + batch_idx / batches_per_epoch_valid)


                    new_time = time.time()
                    print('Epoch', epoch_idx, 'ended after', int(new_time - last_time), 'seconds', flush=True)
                    epoch_train_stats = np.mean(train_losses, axis=0)
                    epoch_valid_stats = np.mean(valid_losses, axis=0)

                    print('Epoch training:', epoch_train_stats)
                    print('Epoch validation:', epoch_valid_stats)
                    print()
                    logs(summary_writer, epoch_train_stats, ['loss', 'acc'], epoch_idx + 1)
                    logs(valid_summary_writer, epoch_valid_stats, ['loss', 'acc'], epoch_idx + 1)

                    if save_path is not None:
                        self.store(log_dir_base + 'model/epoch_{}.ckpt'.format(epoch_idx))

            except KeyboardInterrupt:
                print('Stopping training!')

            training_time = time.time() - start_time
            print('Training ended after', int(training_time), 'seconds')

            if save_path is not None:
                self.store(log_dir_base + save_path)

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

    # def visualize_predictions(self, model_path):
    #     with tf.Session() as self.sess:
    #         self.restore(model_path)
    #
    #         summary_writer = tf.summary.FileWriter('vis/', self.sess.graph)

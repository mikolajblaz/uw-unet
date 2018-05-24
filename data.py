import math
import os
import numpy as np
import tensorflow as tf

import config

# Files handling helpers
def read_image(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image_resized = tf.image.resize_images(image_decoded, config.NET_INPUT_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    return image_resized

def read_label(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=1)
    image_resized = tf.image.resize_images(image_decoded, config.NET_INPUT_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image_resized[:, :, 0]

def read_both(img_lbl):
    img = read_image(img_lbl[0])
    lbl = read_label(img_lbl[1])
    return img, lbl

def basename2img_lbl(basename):
    img_filename = os.path.abspath(TRAIN_IMGS_DIR + basename + '.jpg')
    lbl_filename = os.path.abspath(TRAIN_LABL_DIR + basename + '.png')
    return img_filename, lbl_filename

def img_lbl_filenames(img_lbl_dir):
    img_dir = os.path.join(img_lbl_dir, 'images/')
    lbl_dir = os.path.join(img_lbl_dir, 'labels_plain/')
    print(img_dir, lbl_dir)
    imgs_basenames = [os.path.splitext(name)[0] for name in os.listdir(img_dir)]
    img_lbl_filenames = [
        (os.path.abspath(img_dir + basename + '.jpg'),
         os.path.abspath(lbl_dir + basename + '.png'))
        for basename in imgs_basenames
    ]
    return img_lbl_filenames

def construct_dataset_train(img_lbl_filenames):
    num_samples = len(img_lbl_filenames)
    dataset = tf.data.Dataset.from_tensor_slices(img_lbl_filenames)
    dataset = dataset.map(read_both)
    dataset = dataset.shuffle(buffer_size=config.SHUFFLE_BUFFER)
    dataset = dataset.repeat()
    dataset = dataset.batch(3)      # overlapping epochs, but that's ok
    dataset = dataset.prefetch(1)
    return dataset

def construct_dataset_valid(img_lbl_filenames):
    dataset = tf.data.Dataset.from_tensor_slices(img_lbl_filenames)
    dataset = dataset.map(read_both)
    dataset = dataset.repeat()
    dataset = dataset.batch(3)      # overlapping epochs, but that's ok
    dataset = dataset.prefetch(1)
    return dataset

def train_valid_split(seq, train_ratio=0.7):
    np.random.shuffle(seq)
    train_size = int(len(seq) * train_ratio)
    return seq[:train_size], seq[train_size:]


def full_pipeline(img_lbl_dir=config.ASSIGNMENT_ROOT_DIR + 'training/'):
    img_lbls = img_lbl_filenames(img_lbl_dir)
    # TODO: remove
    img_lbls = img_lbls[:1000]
    img_lbls_train, img_lbls_valid = train_valid_split(img_lbls)

    # img_lbls_train = list(range(10))
    # img_lbls_valid = list(range(10, 30))

    batches_per_epoch_train = math.ceil(len(img_lbls_train) / config.BATCH_SIZE)
    batches_per_epoch_valid = math.ceil(len(img_lbls_valid) / config.BATCH_SIZE)

    return construct_dataset_train(img_lbls_train), construct_dataset_valid(img_lbls_valid), \
           batches_per_epoch_train, batches_per_epoch_valid

class Dataset:
    def __init__(self, assignment_root_dir=config.ASSIGNMENT_ROOT_DIR):
        self.root_dir = assignment_root_dir

        self.train_imgs_dir = assignment_root_dir + 'training/images/'
        self.train_labl_dir = assignment_root_dir + 'training/labels_plain/'

        filenames = os.listdir(self.train_imgs_dir)

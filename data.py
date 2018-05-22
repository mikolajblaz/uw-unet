import os
import tensorflow as tf


from config import NET_INPUT_SIZE, ASSIGNMENT_ROOT_DIR

# Files handling helpers
def get_read_image_fn(jpeg=True):
    decode_fn = tf.image.decode_jpeg if jpeg else tf.image.decode_png
    def read_image(filename):
        image_string = tf.read_file(filename)
        image_decoded = decode_fn(image_string, channels=1)
        image_resized = tf.image.resize_images(image_decoded, NET_INPUT_SIZE)
        return image_resized
    return read_image

def basename2img_lbl(basename):
    img_filename = os.path.abspath(TRAIN_IMGS_DIR + basename + '.jpg')
    lbl_filename = os.path.abspath(TRAIN_LABL_DIR + basename + '.png')
    return img_filename, lbl_filename

def read_both(img_lbl):
    img = img_lbl[0]
    lbl = img_lbl[1]
    return get_read_image_fn(True)(img), get_read_image_fn(False)(lbl)

def img_lbl_filenames(img_lbl_dir):
    img_dir = os.path.join(img_lbl_dir, 'images/')
    lbl_dir = os.path.join(img_lbl_dir, 'labels/')
    print(img_dir, lbl_dir)
    imgs_basenames = [os.path.splitext(name)[0] for name in os.listdir(img_dir)]
    img_lbl_filenames = [
        (os.path.abspath(img_dir + basename + '.jpg'),
         os.path.abspath(lbl_dir + basename + '.png'))
        for basename in imgs_basenames
    ]
    return img_lbl_filenames

def construct_dataset(img_lbl_filenames):
    dataset = tf.data.Dataset.from_tensor_slices(img_lbl_filenames)
    # dataset = dataset.shuffle(buffer_size=TODO)
    dataset = dataset.map(read_both)
    dataset = dataset.repeat()
    dataset = dataset.batch(3)
    return dataset

def full_pipeline(img_lbl_dir=ASSIGNMENT_ROOT_DIR + 'training/'):
    img_lbls = img_lbl_filenames(img_lbl_dir)
    return construct_dataset(img_lbls)

class Dataset:
    def __init__(self, assignment_root_dir=ASSIGNMENT_ROOT_DIR):
        self.root_dir = assignment_root_dir

        self.train_imgs_dir = assignment_root_dir + 'training/images/'
        self.train_labl_dir = assignment_root_dir + 'training/labels/'

        filenames = os.listdir(self.train_imgs_dir)

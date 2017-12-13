#ref from http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

from PIL import Image
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt


mirror = False

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_filename = 'validation.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# Let's collect the real images to later on compare
# to the reconstructed ones
NUM_CLASS = 8277
f = open('./validation_shuffle_8277.txt')
index = 0
while True:
# for _ in range(2):
    line = f.readline()
    if not line:
        break
    index += 1
    print(index)

    img_path, annotation = line.split()

    img = np.array(Image.open("../DB/" + img_path))
    # plt.imshow(img)
    # plt.show()

    annotation = np.array(int(annotation)).reshape(-1)
    annotation = np.eye(NUM_CLASS + 1)[annotation][0]
    annotation = annotation.astype(int)
    annotation = np.array(annotation, dtype=np.int32)
    # print(annotation.size)

    # The reason to store image sizes was demonstrated
    # in the previous example -- we have to know sizes
    # of images to later read raw serialized string,
    # convert to 1d array and convert to respective
    # shape that image used to have.
    height = img.shape[0]
    width = img.shape[1]

    img_raw = img.tostring()
    annotation_raw = annotation.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw),
        'mask_raw': _bytes_feature(annotation_raw)}))

    writer.write(example.SerializeToString())

if mirror:
    f.seek(0)
    while True:
    # for _ in range(2):
        line = f.readline()
        if not line:
            break
        index += 1
        print(index)

        img_path, annotation = line.split()

        img = np.array(Image.open("../DB/" + img_path))
        img = np.fliplr(img)
        # plt.imshow(img)
        # plt.show()

        annotation = np.array(int(annotation)).reshape(-1)
        annotation = np.eye(NUM_CLASS + 1)[annotation][0]
        annotation = annotation.astype(int)
        annotation = np.array(annotation, dtype=np.int32)

        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,
        # convert to 1d array and convert to respective
        # shape that image used to have.
        height = img.shape[0]
        width = img.shape[1]

        img_raw = img.tostring()
        annotation_raw = annotation.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw)}))

        writer.write(example.SerializeToString())


writer.close()

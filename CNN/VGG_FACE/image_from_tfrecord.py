#ref from http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
import tensorflow as tf

def read_and_decode(filename_queue, im_height, im_width, num_channel, num_class, batch_size, num_threads):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.int32)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image_shape = tf.stack([height, width, num_channel])
    annotation_shape = tf.stack([num_class + 1])

    image = tf.reshape(image, image_shape)
    image = tf.cast(image, tf.float32)
    annotation = tf.reshape(annotation, annotation_shape)
    image.set_shape((im_height, im_width, num_channel))
    annotation.set_shape((num_class + 1))
    annotation = tf.cast(annotation, tf.float32)

    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.

    # min_after_dequeue = 10000
    # images, annotations = tf.train.shuffle_batch([image, annotation],
    #                                              batch_size=batch_size,
    #                                              capacity=min_after_dequeue + 3 * batch_size,
    #                                              num_threads=num_threads,
    #                                              min_after_dequeue=min_after_dequeue)

    images, annotations = tf.train.batch([image, annotation], batch_size, num_threads=num_threads)
    return images, annotations
# This file is part of Channel Capacity Estimator,
# licenced under GNU GPL 3 (see file License.txt).
# Homepage: http://pmbm.ippt.pan.pl/software/cce 

from collections import Counter
import numpy as np
import tensorflow as tf


def weight_loss(neighb_count: np.array, labels: np.array,
                weights: list) -> float:
    """Calculates loss for given neighbors and weights.

    Parameters
    ----------
    neighb_count : numpy array
        describes for each point the number of neighbors with each label.
        Shape: (number of data points, number of labels)

    labels : numpy array
        label for each point. Shape: (number of data points, )

    weights : list
        weight of each label. Length: number of labels

    Returns
    -------
    float
        calculated loss
    """
    # reset graph before each run
    tf.reset_default_graph()

    num_data, num_labels = neighb_count.shape
    label_counts = np.zeros([num_labels])
    for label, count in Counter(labels).most_common():
        label_counts[label] = count

    # neighbors matrix
    neigh_matx = tf.constant(neighb_count, dtype=tf.float32)

    # label count vector
    label_cnts = tf.constant(label_counts, dtype=tf.float32)

    # weights
    w = tf.constant(weights, dtype=tf.float32)

    # weight lookup list
    w_list = tf.reduce_sum(tf.one_hot(labels, num_labels) * w, axis=1)

    # label counts lookup list
    label_cnts_list = tf.reduce_sum(tf.one_hot(labels, num_labels) * label_cnts,
                                    axis=1)
    nx = w * num_data

    ny = label_cnts_list / w_list * \
         tf.reduce_sum(neigh_matx * (w/label_cnts), axis=1)

    loss = (tf.reduce_sum(tf.digamma(nx) * w) \
          + tf.reduce_sum(tf.digamma(ny) * w_list / label_cnts_list))

    with tf.Session() as sess:
        return sess.run(loss)

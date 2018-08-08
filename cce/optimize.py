# This file is part of Channel Capacity Estimator,
# licenced under GNU GPL 3 (see file License.txt).
# Homepage: http://pmbm.ippt.pan.pl/software/cce 

from collections import Counter
import numpy as np
import tensorflow as tf

def weight_optimizer(neighb_count, labels) -> (float, list):
    """Returns loss and optimized weights for given neighbors description.

    Parameters
    ----------
    neighb_count : numpy array of shape (# of data points, # of labels)
        describes for each point the number of neighbors with each label

    labels : numpy array of shape (# of data points, )
        label for each point

    Returns
    -------
    float
        loss
    list
        weight for each label
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

    # logits -- to be optimized
    logits = tf.Variable(np.ones(num_labels), dtype=tf.float32)

    # weights
    w = tf.nn.softmax(logits)

    # weight lookup list
    w_list = tf.reduce_sum(tf.one_hot(labels, num_labels) * w, axis=1)

    # label cnts lookup list
    label_cnts_list = tf.reduce_sum(tf.one_hot(labels, num_labels)
                                    * label_cnts, axis=1)
    nx = w * num_data

    ny = label_cnts_list / w_list \
         * tf.reduce_sum(neigh_matx * (w/label_cnts), axis=1)

    loss = (tf.reduce_sum(tf.digamma(nx) * w) +
            tf.reduce_sum(tf.digamma(ny) * w_list / label_cnts_list))

    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #print("Starting training...")
        for _ in range(5000):
            curr_loss, curr_w, __ = sess.run([loss, w, train])
            #if _ % 250 == 0:
            #    print("steps: %s, loss: %s, w: %s"
            #        % (_, curr_loss, curr_w))
        #print("Done.")
        return sess.run([loss, w])

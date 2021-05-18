from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf


def loss_function(real, pred):

    cross_entropy = SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    loss = cross_entropy(y_true=real, y_pred=pred)
    mask = tf.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask * loss
    loss = tf.reduce_mean(loss)

    return loss

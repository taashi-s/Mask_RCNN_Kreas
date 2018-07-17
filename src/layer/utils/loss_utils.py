"""
Loss Utils modeule
"""

import keras.backend as KB
import tensorflow as tf
import loss_utils as lu

def mask_labels_mean_loss(labels, preds):
    """
    Mask Labels mean loss
    """
    loss = KB.binary_crossentropy(labels, preds)
    return lu.losses_mean(loss)


"""
Mask Loss Utils modeule
"""

import keras.backend as KB
import tensorflow as tf

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from subnetwork import utils as u

def mask_labels_mean_loss(labels, preds):
    """
    Mask Labels mean loss
    """
    loss = KB.binary_crossentropy(labels, preds)
    return u.loss_utils.losses_mean(loss)

"""
TODO : Write description
Mask Loss Layer Module
"""

import tensorflow as tf
import keras.backend as KB
from keras.layers.core import Lambda

from utils import loss_utils as lu

class MaskLoss():
    """
    TODO : Write description
    Mask Loss Layer class
    """

    def __init__(self):
        self.__layer = Lambda(lambda inputs: self.__mask_loss(*inputs)
                              , output_shape=self.__mask_loss_output_shape)


    def __call__(self, inputs):
        return self.__layer(inputs)


    def __mask_loss(self, cls_labels, msk_labels, preds):
        positive_ids = tf.where(cls_labels > 0)
        target_batch_ids = KB.cast(positive_ids[:, 0], 'int32')
        target_region_ids = KB.cast(positive_ids[:, 1], 'int32')
        target_class_ids = KB.cast(tf.gather_nd(cls_labels, positive_ids), 'int32')
        target_pred_ids = KB.stack((target_batch_ids, target_region_ids, target_class_ids), axis=1)

        target_label = tf.gather_nd(msk_labels, positive_ids)
        target_pred = tf.gather_nd(preds, target_pred_ids)
        return lu.mask_labels_mean_loss(target_label, target_pred)


    def __mask_loss_output_shape(self, _):
        return [None, 1]

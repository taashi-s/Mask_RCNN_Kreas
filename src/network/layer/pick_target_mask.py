"""
TODO : Write description
Pick Target Mask Layer Module
"""

import tensorflow as tf
import keras.backend as KB
from keras.layers.core import Lambda

class PickTaretMask():
    """
    TODO : Write description
    Pick Target Mask Layer class
    """

    def __init__(self):
        self.__layer = Lambda(lambda inputs: self.__pick_target_mask(*inputs)
                              , output_shape=self.__pick_target_mask_output_shape)


    def __call__(self, inputs):
        return self.__layer(inputs)


    def __pick_target_mask(self, cls_labels, masks):
        mask_shape = KB.shape(masks)
        batch = mask_shape[0]
        reg_num = mask_shape[1]
        mask_h = mask_shape[3]
        mask_w = mask_shape[4]
        dim1 = KB.flatten(KB.repeat(KB.expand_dims(KB.arange(batch)), reg_num))
        dim2 = KB.tile(KB.arange(reg_num), [batch])
        dim3 = KB.cast(KB.flatten(cls_labels), tf.int32)
        ids = KB.stack([dim1, dim2, dim3], axis=1)
        squeezed_masks = tf.gather_nd(masks, ids)
        return KB.reshape(squeezed_masks, [batch, reg_num, mask_h, mask_w])


    def __pick_target_mask_output_shape(self, inputs):
        return [None, inputs[1][1], inputs[1][3], inputs[1][4]]

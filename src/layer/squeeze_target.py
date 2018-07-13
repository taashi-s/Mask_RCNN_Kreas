"""
TODO : Write description
Squeeze Target Layer Module
"""

import tensorflow as tf
import keras.backend as KB
from keras.layers.core import Lambda

class SqueezeTaret():
    """
    TODO : Write description
    Squeeze Target Layer class
    """

    def __init__(self, batch_size=5):
        self.__batch_size = batch_size
        self.__layer = Lambda(lambda inputs: self.__squeeze_target(*inputs)
                              , output_shape=self.__squeeze_target_output_shape)


    def __call__(self, inputs):
        return self.__layer(inputs)


    def __squeeze_target(self, regions, classes, offsets):
        target_clss = []
        target_ofss = []
        target_regs = []

        zip_data = self.__zip_by_batch(regions, classes, offsets, self.__batch_size)
        for data in zip_data:
            data_s = self.__shaping_inputs(*data)
            target_data = self.__get_target_data(*data_s)

            target_reg, target_ofs, target_cls, target_msk = target_data
            target_clss.append(target_cls)
            target_ofss.append(target_ofs)
            target_regs.append(target_reg)
        return [KB.stack(target_clss), KB.stack(target_ofss)
                , KB.stack(target_regs)]

    def __zip_by_batch(self, regions, classes, offsets, batch_size):
        split_regions = tf.split(regions, batch_size)
        split_classes = tf.split(classes, batch_size)
        split_offsets = tf.split(offsets, batch_size)
        return zip(split_regions, split_classes, split_offsets)


    def __shaping_inputs(self, reg, cls, ofs):
        reg_2d = KB.squeeze(reg, 0)
        cls_2d = KB.squeeze(cls, 0)
        ofs_2d = KB.squeeze(ofs, 0)
        return reg_2d, cls_2d, ofs_2d


    def __get_target_data(self, reg, cls, ofs):
        ious = RegionsUtils(reg).calc_iou(reg_lbl)
        positive_ids = self.__get_positive(ious)
        negative_ids = self.__get_negative(ious, KB.shape(positive_ids)[0])

        target_reg_ids = KB.concatenate((positive_ids, negative_ids))
        max_iou_ids = KB.argmax(ious, axis=1)
        target_reg_lbl_ids = KB.gather(max_iou_ids, target_reg_ids)
        target_cls_lbl_ids = KB.gather(max_iou_ids, positive_ids)

        target_reg = KB.gather(reg, target_reg_ids)
        target_ofs = self.__get_target_offset(reg_lbl, target_reg_lbl_ids, target_reg)
        target_cls = self.__get_target_class_label(cls_lbl, target_cls_lbl_ids, negative_ids)
        target_msk = self.__get_target_mask_label(msk_lbl, target_reg_lbl_ids, target_reg)
        return self.__padding_data(target_reg, target_ofs, target_cls, target_msk)


    def __get_positive(self, ious):
        max_iou = KB.max(ious, axis=1)
        ids = KB.flatten(tf.where(max_iou >= self.__th))
        count = round(self.__count_per_batch * self.__ratio)
        return self.__get_shuffle_sample(ids, count)


    def __get_negative(self, ious, positive_count):
        max_iou = KB.max(ious, axis=1)
        ids = KB.flatten(tf.where((self.__excl_th <= max_iou) & (max_iou < self.__th)))
        count = self.__count_per_batch - positive_count
        return self.__get_shuffle_sample(ids, count)


    def __get_shuffle_sample(self, sample, count):
        sample_num = KB.shape(sample)[0]
        limit = KB.minimum(count, sample_num)
        shuffle_sample = tf.random_shuffle(sample)[:limit]
        return KB.switch(sample_num > 0, shuffle_sample, sample)


    def __get_target_offset(self, reg_lbl, target_reg_lbl_ids, target_reg):
        target_reg_lbl = KB.gather(reg_lbl, target_reg_lbl_ids)
        return RegionsUtils(target_reg_lbl).calc_offset(target_reg)


    def __get_target_class_label(self, cls_lbl, target_cls_lbl_ids, negative_ids):
        target_cls = KB.squeeze(KB.cast(KB.gather(cls_lbl, target_cls_lbl_ids), 'int32'), 1)
        padding = KB.zeros([KB.shape(negative_ids)[0]], dtype='int32')
        return KB.expand_dims(KB.concatenate((target_cls, padding)), 1)


    def __get_target_mask_label(self, msk_lbl, target_reg_lbl_ids, target_reg):
        target_msk = KB.gather(msk_lbl, target_reg_lbl_ids)
        target_msk_4d = tf.expand_dims(target_msk, -1)
        ids = KB.arange(0, KB.shape(target_reg)[0])
        target_msk_crop = tf.image.crop_and_resize(target_msk_4d, target_reg, ids
                                                   , (self.__mask_size_h, self.__mask_size_w))
        target_msk_3d = tf.squeeze(target_msk_crop, axis=3)
        return tf.round(target_msk_3d)


    def __padding_data(self, regs, ofss, clss, msks):
        padding_count = KB.maximum(self.__count_per_batch - KB.shape(regs)[0], 0)
        paddings = [(0, padding_count), (0, 0)]
        mask_shape = (self.__count_per_batch, self.__mask_size_h, self.__mask_size_w)
        padding_regs = KB.reshape(tf.pad(regs, paddings), (self.__count_per_batch, 4))
        padding_ofss = KB.reshape(tf.pad(ofss, paddings), (self.__count_per_batch, 4))
        padding_clss = KB.reshape(tf.pad(clss, paddings), (self.__count_per_batch, 1))
        padding_msks = KB.reshape(tf.pad(msks, [(0, padding_count), (0, 0), (0, 0)]), mask_shape)
        return padding_regs, padding_ofss, padding_clss, padding_msks


    def __squeeze_target_output_shape(self, inputs):
        return []

"""
TODO : Write description
Detection Target Region Mask Layer Module
"""

import tensorflow as tf
from keras.layers.core import Lambda
import keras.backend as KB
import keras.utils.conv_utils as KCUtils

from utils.regions_utils import RegionsUtils


class DetectionTargetRegionMask():
    """
    TODO : Write description
    Detection Target Region Mask Layer class
    """

    def __init__(self, positive_threshold=0.5, positive_ratio=0.33, image_shape=None
                 , batch_size=5, exclusion_threshold=0.1, count_per_batch=64
                 , mask_size=28, name='detection_target_region_mask', **kwargs):
        super(DetectionTargetRegionMask, self).__init__(**kwargs)
        self.__th = positive_threshold
        self.__excl_th = exclusion_threshold
        self.__count_per_batch = count_per_batch
        self.__ratio = positive_ratio
        self.__image_shape = image_shape
        self.__batch_size = batch_size
        (mask_h, mask_w) = KCUtils.normalize_tuple(mask_size, 2, 'mask_size')
        self.__mask_size_h = mask_h
        self.__mask_size_w = mask_w
        self.__layer = Lambda(lambda inputs: self.__detection_target_region_mask(*inputs)
                              , output_shape=self.__output_shape
                              , name=name)


    def __call__(self, inputs):
        return self.__layer(inputs)

    def __detection_target_region_mask(self, cls_labels, reg_labels, msk_labels, regions):
        norm_reg_labels = reg_labels
        if self.__image_shape is not None:
            img_h, img_w, _ = self.__image_shape
            norm_reg_labels = RegionsUtils(reg_labels).normalize(img_h, img_w)

        target_clss = []
        target_ofss = []
        target_regs = []
        target_msks = []

        zip_data = self.__zip_by_batch(cls_labels, norm_reg_labels, msk_labels
                                       , regions, self.__batch_size)
        for data in zip_data:
            data_s = self.__shaping_inputs(*data)
            target_data = self.__get_target_data(*data_s)

            target_reg, target_ofs, target_cls, target_msk = target_data
            target_clss.append(target_cls)
            target_ofss.append(target_ofs)
            target_regs.append(target_reg)
            target_msks.append(target_msk)
        return [KB.stack(target_clss), KB.stack(target_ofss)
                , KB.stack(target_msks), KB.stack(target_regs)]

    def __zip_by_batch(self, cls_labels, reg_labels, msk_labels, regions, batch_size):
        split_cls_labels = tf.split(cls_labels, batch_size)
        split_reg_labels = tf.split(reg_labels, batch_size)
        split_msk_labels = tf.split(msk_labels, batch_size)
        split_regions = tf.split(regions, batch_size)
        return zip(split_cls_labels, split_reg_labels, split_msk_labels, split_regions)


    def __shaping_inputs(self, cls_label, reg_label, msk_label, region):
        cls_label_1d = KB.squeeze(KB.squeeze(cls_label, 0), 1)
        reg_label_2d = KB.squeeze(reg_label, 0)
        msk_label_3d = KB.squeeze(msk_label, 0)
        region_2d = KB.squeeze(region, 0)

        target_lbl_ids = KB.flatten(tf.where(KB.any(reg_label_2d, axis=1)))
        target_reg_ids = KB.flatten(tf.where(KB.any(region_2d, axis=1)))

        cls_lbl = KB.gather(cls_label_1d, target_lbl_ids)
        reg_lbl = KB.gather(reg_label_2d, target_lbl_ids)
        msk_lbl = KB.gather(msk_label_3d, target_lbl_ids)
        reg = KB.gather(region_2d, target_reg_ids)
        return cls_lbl, reg_lbl, msk_lbl, reg


    def __get_positive(self, max_ious, count):
        ids = KB.flatten(tf.where(max_ious >= self.__th))
        return self.__get_shuffle_sample(ids, count)


    def __get_negative(self, max_ious, count):
        ids = KB.flatten(tf.where((self.__excl_th <= max_ious) & (max_ious < self.__th)))
        return self.__get_shuffle_sample(ids, count)


    def __get_shuffle_sample(self, sample, count):
        sample_num = KB.shape(sample)[0]
        limit = KB.minimum(count, sample_num)
        shuffle_sample = tf.random_shuffle(sample)[:limit]
        return KB.switch(sample_num > 0, shuffle_sample, sample)


    def __get_target_data(self, cls_lbl, reg_lbl, msk_lbl, reg):
        ious = RegionsUtils(reg).calc_iou(reg_lbl)
        max_ious = KB.max(ious, axis=1)
        positive_count = round(self.__count_per_batch * self.__ratio)
        negative_count = self.__count_per_batch - positive_count
        positive_ids = self.__get_positive(max_ious, positive_count)
        negative_ids = self.__get_negative(max_ious, negative_count)

        target_reg_ids = KB.concatenate((positive_ids, negative_ids))
        max_iou_ids = KB.argmax(ious, axis=1)
        target_reg_lbl_ids = KB.gather(max_iou_ids, target_reg_ids)
        target_cls_lbl_ids = KB.gather(max_iou_ids, positive_ids)

        target_reg = KB.gather(reg, target_reg_ids)
        target_ofs = self.__get_target_offset(reg_lbl, target_reg_lbl_ids, target_reg)
        target_cls = self.__get_target_class_label(cls_lbl, target_cls_lbl_ids, KB.shape(negative_ids)[0])
        target_msk = self.__get_target_mask_label(msk_lbl, target_reg_lbl_ids, target_reg)
        return self.__padding_data(target_reg, target_ofs, target_cls, target_msk)


    def __get_target_offset(self, reg_lbl, target_reg_lbl_ids, target_reg):
        target_reg_lbl = KB.gather(reg_lbl, target_reg_lbl_ids)
        return RegionsUtils(target_reg_lbl).calc_offset(target_reg)


    def __get_target_class_label(self, cls_lbl, target_cls_lbl_ids, negative_count):
        target_cls = KB.gather(cls_lbl, target_cls_lbl_ids)
        target_cls_c = KB.cast(target_cls, 'int32')
        padding = KB.zeros([negative_count], dtype='int32')
        return KB.concatenate((target_cls_c, padding))


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
        reg_paddings = [(0, padding_count), (0, 0)]
        cls_paddings = [(0, padding_count)]
        mask_shape = (self.__count_per_batch, self.__mask_size_h, self.__mask_size_w)
        padding_regs = KB.reshape(tf.pad(regs, reg_paddings), (self.__count_per_batch, 4))
        padding_ofss = KB.reshape(tf.pad(ofss, reg_paddings), (self.__count_per_batch, 4))
        padding_clss = KB.reshape(tf.pad(clss, cls_paddings), (self.__count_per_batch, 1))
        padding_msks = KB.reshape(tf.pad(msks, [(0, padding_count), (0, 0), (0, 0)]), mask_shape)
        return padding_regs, padding_ofss, padding_clss, padding_msks


    def __output_shape(self, input_shape):
        return [(None, self.__count_per_batch, 1)
                , (None, self.__count_per_batch, 4)
                , (None, self.__count_per_batch, self.__mask_size_h, self.__mask_size_w)
                , (None, self.__count_per_batch, 4)
               ]

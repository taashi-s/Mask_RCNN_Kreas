"""
TODO : Write description
Squeeze Target Layer Module
"""

import tensorflow as tf
import keras.backend as KB
from keras.layers.core import Lambda

from utils.regions_utils import RegionsUtils

class SqueezeTarget():
    """
    TODO : Write description
    Squeeze Target Layer class
    """

    def __init__(self, batch_size=5, image_shape=None
                 , squeeze_threshold=0.7, max_pred_count=50, nms_threshold=0.3
                 , refinement_std_dev=None):
        self.__batch_size = batch_size
        self.__image_shape = image_shape
        self.__th = squeeze_threshold
        self.__max_count = max_pred_count
        self.__nms_th = nms_threshold
        self.__ref_std = [0.1, 0.1, 0.2, 0.2]
        if refinement_std_dev is not None:
            self.__ref_std = refinement_std_dev
        self.__layer = Lambda(lambda inputs: self.__squeeze_target(*inputs)
                              , output_shape=self.__squeeze_target_output_shape)


    def __call__(self, inputs):
        return self.__layer(inputs)


    def __squeeze_target(self, regions, classes, offsets):
        target_real_regs = []
        target_reg_preds = []
        target_cls_preds = []
        target_cls_idss = []

        zip_data = self.__zip_by_batch(regions, classes, offsets, self.__batch_size)
        for data in zip_data:
            data_s = self.__shaping_inputs(*data)
            target_data = self.__get_target_data(*data_s)

            target_real_reg, target_reg_pred, target_cls_pred, target_cls_ids = target_data
            target_real_regs.append(target_real_reg)
            target_reg_preds.append(target_reg_pred)
            target_cls_preds.append(target_cls_pred)
            target_cls_idss.append(target_cls_ids)
        return [KB.stack(target_real_regs), KB.stack(target_reg_preds)
                , KB.stack(target_cls_preds), KB.stack(target_cls_idss)]


    def __zip_by_batch(self, regions, classes, offsets, batch_size):
        split_regions = tf.split(regions, batch_size)
        split_classes = tf.split(classes, batch_size)
        split_offsets = tf.split(offsets, batch_size)
        return zip(split_regions, split_classes, split_offsets)


    def __shaping_inputs(self, reg_pred, cls_pred, ofs_pred):
        reg_pred_2d = KB.squeeze(reg_pred, 0)
        cls_pred_2d = KB.squeeze(cls_pred, 0)
        ofs_pred_2d = KB.squeeze(ofs_pred, 0)
        return reg_pred_2d, cls_pred_2d, ofs_pred_2d


    def __get_target_data(self, reg_pred, cls_pred, ofs_pred):
        positive_preds = self.__get_positive_class_preds(reg_pred, cls_pred, ofs_pred)
        object_preds = self.__get_object_region_preds(*positive_preds)
        sorted_preds = self.__sort_preds_by_class(*object_preds)
        sorted_reg, sorted_cls, sorted_ofs, sorted_cls_ids = sorted_preds
        real_reg = self.__get_real_regions(sorted_reg, sorted_ofs)
        nms_preds = self.__get_nms_preds(real_reg, sorted_reg, sorted_cls, sorted_cls_ids)
        return self.__padding_data(*nms_preds)


    def __gather_data(self, data_list, ids):
        gather_data = []
        for data in data_list:
            gather_data.append(tf.gather(data, ids))
        return gather_data

    def __get_positive_class_preds(self, reg_pred, cls_pred, ofs_pred):
        reg_num, _ = cls_pred.get_shape()

        max_cls_ids = KB.cast(KB.argmax(cls_pred, axis=-1), 'int32')
        max_ids_2d = KB.cast(KB.stack([KB.arange(reg_num), max_cls_ids], axis=1), 'int32')

        max_cls_pred = tf.gather_nd(cls_pred, max_ids_2d)
        max_ofs_pred = tf.gather_nd(ofs_pred, max_ids_2d) * KB.variable(self.__ref_std)

        positive_cls_ids, _ = tf.unique(tf.where(max_cls_pred >= self.__th)[:, 0])
        return self.__gather_data([reg_pred, max_cls_pred, max_ofs_pred, max_cls_ids]
                                  , positive_cls_ids)


    def __get_object_region_preds(self, reg_pred, cls_pred, ofs_pred, cls_ids):
        object_cls_ids, _ = tf.unique(tf.where(cls_ids > 0)[:, 0])
        return self.__gather_data([reg_pred, cls_pred, ofs_pred, cls_ids], object_cls_ids)


    def __sort_preds_by_class(self, reg_pred, cls_pred, ofs_pred, cls_ids):
        target_num, _ = cls_pred.get_shape()
        _, sort_cls_ids = tf.nn.top_k(cls_pred, k=target_num, sorted=True)
        return self.__gather_data([reg_pred, cls_pred, ofs_pred, cls_ids], sort_cls_ids)


    def __get_real_regions(self, reg_pred, ofs_pred):
        offset_reg = RegionsUtils(reg_pred).offsets(ofs_pred)
        if self.__image_shape is not None:
            offset_reg = RegionsUtils(offset_reg).denormalize(*self.__image_shape)
        return KB.cast(offset_reg, 'float32')


    def __get_nms_preds(self, real_reg, reg_pred, cls_pred, cls_ids):
        nms_ids = tf.image.non_max_suppression(real_reg, cls_pred
                                               , max_output_size=self.__max_count
                                               , iou_threshold=self.__nms_th)
        return self.__gather_data([real_reg, reg_pred, cls_pred, cls_ids], nms_ids)


    def __padding_data(self, real_reg, reg_pred, cls_pred, cls_ids):
        target_count = (real_reg.get_shape())[0]
        padding_count = KB.cast(KB.maximum(self.__max_count - target_count, 0), 'int32')
        padding_shape = [0, padding_count]
        reg_shape = (self.__max_count, 4)
        cls_shape = (self.__max_count, 1)
        padding_real_reg = KB.reshape(tf.pad(real_reg, [padding_shape, [0, 0]]), reg_shape)
        padding_reg_pred = KB.reshape(tf.pad(reg_pred, [padding_shape, [0, 0]]), reg_shape)
        padding_cls_pred = KB.reshape(tf.pad(cls_pred, [padding_shape]), cls_shape)
        padding_cls_ids = KB.reshape(tf.pad(cls_ids, [padding_shape]), cls_shape)
        return padding_real_reg, padding_reg_pred, padding_cls_pred, padding_cls_ids


    def __squeeze_target_output_shape(self, _):
        return [(None, self.__max_count, 4)
               , (None, self.__max_count, 4)
               , (None, self.__max_count, 1)
               , (None, self.__max_count, 1)
               ]

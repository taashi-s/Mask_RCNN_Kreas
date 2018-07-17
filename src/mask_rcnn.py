"""
TODO : Write description
Faster R-CNN Module
"""

import tensorflow as tf
from keras.models import Model
from keras.engine.topology import Input
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import plot_model
import keras.utils.conv_utils as KCUtils

from subnetwork.faster_rcnn.src.faster_rcnn import FasterRCNN, TrainTarget, RoIPooling
from subnetwork.faster_rcnn.src.faster_rcnn import ClassLoss, RegionLoss
from layer.detection_target_region_mask import DetectionTargetRegionMask
from layer.mask_loss import MaskLoss
from layer.pick_target_mask import PickTaretMask

class MaskRCNN():
    """
    TODO : Write description
    Mask R-CNN class
    """

    def __init__(self, input_shape, class_num, anchors
                 , batch_size=5, mask_size=14
                 , is_predict=False, train_taegets=None):
        self.__input_shape = input_shape
        train_head = TrainTarget.HEAD in train_taegets
        train_rpn = TrainTarget.RPN in train_taegets

        faster_rcnn = FasterRCNN(input_shape, class_num, anchors
                                 , batch_size, is_predict, train_taegets)

        _, backbone = faster_rcnn.get_backbone_network()
        inputs, rpn = faster_rcnn.get_rpn_network()
        _, _, rpn_prop_regs = rpn
        outputs = []
        if train_rpn and not is_predict:
            inputs, outputs = faster_rcnn.get_rpn_loss_network()

        if train_head and not is_predict:
            mask_h, mask_w = KCUtils.normalize_tuple(mask_size, 2, 'mask_size')
            inputs_cls = Input(shape=[None, 1], dtype='int32')
            inputs_reg = Input(shape=[None, 4], dtype='float32')
            inputs_msk = Input(shape=[None, mask_h, mask_w], dtype='float32')
            inputs += [inputs_cls, inputs_reg]

            dtrm = DetectionTargetRegionMask(positive_threshold=0.5, positive_ratio=0.33
                                             , image_shape=self.__input_shape, batch_size=batch_size
                                             , exclusion_threshold=0.1, count_per_batch=64
                                             , mask_size=mask_size
                                            )([inputs_cls, inputs_reg, inputs_msk, rpn_prop_regs])
            dtrm_cls_labels, dtrm_ofs_labels, dtrm_msk_labels, dtrm_regions = dtrm

            clsses, offsets = faster_rcnn.head_net(backbone, dtrm_regions, class_num
                                                   , batch_size=batch_size)
            masks = self.__mask_net(backbone, dtrm_regions, class_num, batch_size=batch_size)

            cls_losses = ClassLoss()([dtrm_cls_labels, clsses])
            reg_losses = RegionLoss()([dtrm_cls_labels, dtrm_ofs_labels, offsets])
            msk_losses = MaskLoss()([dtrm_cls_labels, dtrm_msk_labels, masks])

            outputs += [cls_losses, reg_losses, msk_losses]

        if is_predict:
            classes, offsets = faster_rcnn.head_net(backbone, rpn_prop_regs, class_num
                                                    , batch_size=batch_size)
            sqzt = SqueezeTarget()([rpn_prop_regs, classes, offsets])
            sqzt_, sqzt_, sqzt_, sqzt_ = sqzt
            masks = self.__mask_net(backbone, , class_num, batch_size=batch_size)
            target_masks = PickTaretMask()([, masks])

            outputs += [rpn_prop_regs, classes, offsets, target_masks]

        self.__network = (inputs, outputs)
        self.__model = Model(inputs=inputs, outputs=outputs)

        for output in outputs:
            self.__model.add_loss(tf.reduce_mean(output))


    def __mask_net(self, fmaps, regions, class_num, batch_size=5):
        """
        for backbone is ResNet/FPN
        """
        # TODO : change kernel for 14 * 14, stride=2
        roi_pool = RoIPooling(image_shape=self.__input_shape
                              , batch_size=batch_size)([fmaps, regions])

        conv_layers = roi_pool
        for _ in range(4):
            conv_layers = TimeDistributed(Conv2D(265, 3, padding='same'))(conv_layers)
            conv_layers = TimeDistributed(BatchNormalization())(conv_layers)
            conv_layers = TimeDistributed(Activation('relu'))(conv_layers)

        deconv = TimeDistributed(Conv2DTranspose(256, 2, strides=2, activation='relu'))(conv_layers)
        masks = TimeDistributed(Conv2D(class_num, 1, 1, activation='sigmoid'))(deconv)
        return masks


    def get_network(self):
        """
        TODO : Write description
        get_model
        """
        return self.__network


    def get_model(self):
        """
        TODO : Write description
        get_model
        """
        return self.__model


    def default_compile(self):
        """
        TODO : Write description
        default_compile
        """
        self.__model.compile(optimizer=SGD(momentum=0.9, decay=0.0001)
                             , loss=[None] * len(self.__model.outputs), metrics=[])


    def get_model_with_default_compile(self):
        """
        TODO : Write description
        get_model_with_default_compile
        """
        self.default_compile()
        return self.__model


    def draw_model_summary(self, file_name='model.png'):
        plot_model(self.__model, to_file=file_name)


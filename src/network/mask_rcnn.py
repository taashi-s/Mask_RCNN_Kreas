"""
TODO : Write description
Faster R-CNN Module
"""

import tensorflow as tf
import keras.backend as KB
from keras.models import Model
from keras.engine.topology import Input
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Activation, Reshape, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
import keras.utils.conv_utils as KCUtils

from .subnetwork import FasterRCNN, TrainTarget, RoIPooling, ClassLoss, RegionLoss
from .layer import DetectionTargetRegionMask, MaskLoss, SqueezeTarget, PickTaretMask

class MaskRCNN():
    """
    TODO : Write description
    Mask R-CNN class
    """

    def __init__(self, input_shape, class_num, anchors=None
                 , batch_size=5, mask_size=28, roi_pool_size=14
                 , is_predict=False, train_targets=None):
        self.__input_shape = input_shape
        if train_targets is None:
            train_targets = []
        train_head = TrainTarget.HEAD in train_targets
        train_rpn = TrainTarget.RPN in train_targets

        faster_rcnn = FasterRCNN(input_shape, class_num, anchors
                                 , batch_size, is_predict, train_targets)

        first_layer, backbone = faster_rcnn.get_backbone_network()
        inputs, rpn = faster_rcnn.get_rpn_network()
        rpn_cls_probs, rpn_regions, rpn_prop_regs = rpn
        outputs = []
        if train_rpn and not is_predict:
            inputs, outputs = faster_rcnn.get_rpn_loss_network()

        if train_head and not is_predict:
            image_h, image_w, _ = self.__input_shape
            inputs_cls = Input(shape=[None, 1], dtype='int32', name='head_cls_input')
            inputs_reg = Input(shape=[None, 4], dtype='float32', name='head_reg_input')
            inputs_msk = Input(shape=[None, image_h, image_w], dtype='float32', name='head_msk_input')
            inputs += [inputs_cls, inputs_reg, inputs_msk]

            dtrm = DetectionTargetRegionMask(positive_threshold=0.5, positive_ratio=0.33
                                             , image_shape=self.__input_shape, batch_size=batch_size
                                             , exclusion_threshold=0.1, count_per_batch=20
                                             , mask_size=mask_size
                                            )([inputs_cls, inputs_reg, inputs_msk, rpn_prop_regs])
            dtrm_cls_labels, dtrm_ofs_labels, dtrm_msk_labels, dtrm_regions = dtrm

            clsses, offsets = faster_rcnn.head_net(backbone, dtrm_regions, class_num
                                                   , trainable=train_head, batch_size=batch_size)
            masks = self.__mask_net(backbone, dtrm_regions, class_num
                                    , trainable=train_head, batch_size=batch_size)

            cls_losses = ClassLoss()([dtrm_cls_labels, clsses])
            reg_losses = RegionLoss()([dtrm_cls_labels, dtrm_ofs_labels, offsets])
            msk_losses = MaskLoss()([dtrm_cls_labels, dtrm_msk_labels, masks])

            outputs += [cls_losses, reg_losses, msk_losses]

        if is_predict:
            classes, offsets = faster_rcnn.head_net(backbone, rpn_prop_regs, class_num
                                                    , trainable=train_head, batch_size=batch_size)
            sqzt = SqueezeTarget(batch_size=5, image_shape=self.__input_shape
                                 , squeeze_threshold=0.7, max_pred_count=10, nms_threshold=0.3
                                 , refinement_std_dev=None
                                )([rpn_prop_regs, classes, offsets])
            sqzt_real_reg, sqzt_reg_pred, sqzt_cls_pred, sqzt_cls_ids = sqzt
            masks = self.__mask_net(backbone, sqzt_reg_pred, class_num
                                    , batch_size=batch_size, roi_pool_size=roi_pool_size
                                    , trainable=False, mask_size=mask_size)
            target_masks = PickTaretMask()([sqzt_cls_pred, masks])

            outputs += [sqzt_real_reg, sqzt_cls_pred, sqzt_cls_ids, target_masks
                        , sqzt_reg_pred, rpn_regions, rpn_cls_probs]

        self.__network = (inputs, outputs)
        self.__model = Model(inputs=inputs, outputs=outputs)

        if not is_predict:
            for output in outputs:
                self.__model.add_loss(tf.reduce_mean(output))
        else:
            dummy_loss = Lambda(lambda x: KB.constant(0.0), name='dummy_loss')(first_layer)
            self.__model.add_loss(tf.reduce_mean(dummy_loss))


    def __mask_net(self, fmaps, regions, class_num, batch_size=5, roi_pool_size=14
                   , trainable=True, mask_size=28):
        """
        for backbone is ResNet/FPN
        """
        roi_pool = RoIPooling(batch_size=batch_size, pooling=roi_pool_size
                              , image_shape=self.__input_shape
                              , name='poi_pooling_mask'
                             )([fmaps, regions])

        conv_layers = roi_pool
        for _ in range(4):
            #conv_layers = TimeDistributed(Conv2D(256, 3, padding='same'
            conv_layers = TimeDistributed(Conv2D(128, 3, padding='same'
                                                 , trainable=trainable))(conv_layers)
            conv_layers = TimeDistributed(BatchNormalization())(conv_layers)
            conv_layers = TimeDistributed(Activation('relu'))(conv_layers)

        #deconv = TimeDistributed(Conv2DTranspose(256, 2, strides=2, activation='relu'
        deconv = TimeDistributed(Conv2DTranspose(128, 2, strides=2, activation='relu'
                                                 , trainable=trainable))(conv_layers)
        fc_conv = TimeDistributed(Conv2D(class_num, 1, activation='sigmoid'
                                         , trainable=trainable))(deconv)

        mask_h, mask_w = KCUtils.normalize_tuple(mask_size, 2, 'mask_size')
        masks = Reshape((-1, class_num, mask_h, mask_w))(fc_conv)
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
        #self.__model.compile(optimizer=SGD(momentum=0.9, decay=0.0001)
        self.__model.compile(optimizer=Adam()
                             , loss=[None] * len(self.__model.outputs), metrics=['tmpa'])


    def get_model_with_default_compile(self):
        """
        TODO : Write description
        get_model_with_default_compile
        """
        self.default_compile()
        return self.__model


    def draw_model_summary(self, file_name='model.png'):
        """
        TODO : Write description
        draw_model_summary
        """
        plot_model(self.__model, to_file=file_name)


    @staticmethod
    def get_backbone_output_shape(input_shape):
        """
        get_backbone_output_shape
        """
        return FasterRCNN.get_backbone_output_shape(input_shape)

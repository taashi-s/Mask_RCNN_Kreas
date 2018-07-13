"""
TODO : Write description
Faster R-CNN Module
"""

from enum import Enum
import tensorflow as tf
from keras.models import Model
from keras.engine.topology import Input
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import plot_model

from subnetwork.faster_rcnn.src.faster_rcnn import FasterRCNN, TrainTarget, RoIPooling

class MaskRCNN():
    """
    TODO : Write description
    Mask R-CNN class
    """

    def __init__(self, input_shape, class_num, anchors
                 , batch_size=5, is_predict=False, train_taegets=None):
        self.__input_shape = input_shape
        train_head = TrainTarget.HEAD in train_taegets

        faster_rcnn = FasterRCNN(input_shape, class_num, anchors
                                 , batch_size, is_predict, train_taegets)

        inputs, outputs = faster_rcnn.get_network()
        _, backbone = faster_rcnn.get_backbone_network()
        _, prev_head = faster_rcnn.get_prev_head_network()
        _, _, prev_head_regs = prev_head

        masks = self.__mask_net(backbone, prev_head_regs, class_num, batch_size=batch_size)
        if train_head and not is_predict:
            # TODO : Add Loss outputs
            # TODO : Add inputs, outputs
            pass

        if is_predict:
            outputs += [masks]

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
        for i in range(4):
            conv_layers = TimeDistributed(Conv2D(265, 3, padding='same'))(conv_layers)
            conv_layers = TimeDistributed(BatchNormalization())(conv_layers)
            conv_layers = TimeDistributed(Activation('relu'))(conv_layers)

        deconv = TimeDistributed(Conv2DTranspose(256, 2, strides=2, activation='relu'))
        masks = TimeDistributed(Conv2D(class_num, 1, 1, activation='sigmoid'))
        return masks


    def __mask_loss(self, mask_labels, class_ids, preds):
        """
        TODO : Layer
        """
        return []


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


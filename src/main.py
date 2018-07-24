""" main """

import numpy as np
import os
import math
from enum import Enum
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

from network.mask_rcnn import MaskRCNN, TrainTarget
from network.subnetwork.faster_rcnn import rpn_input_data
from data.coco_dataset import COCODataset, GenerateTarget

#INPUT_SHAPE = (1024, 1024, 3)
INPUT_SHAPE = (256, 256, 3)
BATCH_SIZE = 2
EPOCHS = 100

DIR_MODEL = '.'
FILE_MODEL = 'MaskRCNN_Model'
EXT_MODEL = '.hdf5'

class Train_Mode(Enum):
    """
    Train Mode Class
    """
    STEP1 = 1
    STEP2 = 2
    STEP3 = 3
    STEP4 = 4


def train(mode):
    """ train """
    print('execute train')

    # memo :
    #    | Backbone | RPN | Head |
    # 1. |    o     |  o  |  x   |
    # 2. |    o     |  x  |  o   |
    # 3. |    x     |  o  |  x   |
    # 4. |    x     |  o  |  o   |
    train_targets = [TrainTarget.BACKBONE, TrainTarget.RPN, TrainTarget.HEAD]
    gener_targets = [GenerateTarget.RPN_INPUT, GenerateTarget.HEAD_INPUT]
    if mode == Train_Mode.STEP1:
        train_targets = [TrainTarget.BACKBONE, TrainTarget.RPN]
        gener_targets = [GenerateTarget.RPN_INPUT]
    elif mode == Train_Mode.STEP2:
        train_targets = [TrainTarget.BACKBONE, TrainTarget.HEAD]
        gener_targets = [GenerateTarget.HEAD_INPUT]
    elif mode == Train_Mode.STEP3:
        train_targets = [TrainTarget.RPN]
        gener_targets = [GenerateTarget.RPN_INPUT]
    elif mode == Train_Mode.STEP4:
        train_targets = [TrainTarget.RPN, TrainTarget.HEAD]
        gener_targets = [GenerateTarget.RPN_INPUT, GenerateTarget.HEAD_INPUT]

    anchors = rpn_input_data.get_anchors(INPUT_SHAPE)

    network = MaskRCNN(INPUT_SHAPE, 2
                       , train_targets= train_targets, is_predict=False
                      )
    print('model compiling ...')
    model = network.get_model_with_default_compile()
    #network.draw_model_summary(file_name=os.path.join(os.pardir, 'ModelLayers.png'))
    #model.summary()
    print('... compiled')

    model_filename_base = os.path.join(DIR_MODEL, FILE_MODEL)
    model_filename = model_filename_base + EXT_MODEL
    if mode == Train_Mode.STEP1:
        model_filename = model_filename_base + '_Step1' + EXT_MODEL
    elif mode == Train_Mode.STEP2:
        model.load_weights(model_filename_base + '_Step1' + EXT_MODEL, by_name=True)
        model_filename = model_filename_base + '_Step2' + EXT_MODEL
    elif mode == Train_Mode.STEP3:
        model.load_weights(model_filename_base + '_Step2' + EXT_MODEL, by_name=True)
        model_filename = model_filename_base + '_Step3' + EXT_MODEL
    elif mode == Train_Mode.STEP4:
        model.load_weights(model_filename_base + '_Step3' + EXT_MODEL, by_name=True)
        model_filename = model_filename_base + '_Step4' + EXT_MODEL

    print('dataset create ...')
    dataset = COCODataset(categories=['cat'])
    data_num = dataset.data_size()
    train_data_num = math.floor(data_num * 0.8)
    valid_data_num = data_num - train_data_num
    basedata = dataset.get_data_list()
    basedata_train = basedata[:train_data_num]
    basedata_valid = basedata[train_data_num:]
    train_generator = dataset.generator(anchors, INPUT_SHAPE, batch_size=BATCH_SIZE
                                        , target_data_list=basedata_train
                                        , genetate_targets=gener_targets
                                       )
    valid_generator = dataset.generator(anchors, INPUT_SHAPE, batch_size=BATCH_SIZE
                                        , target_data_list=basedata_valid
                                        , genetate_targets=gener_targets
                                       )
    print('... created')

    print('fix ...')
    his = model.fit_generator(train_generator
                              , steps_per_epoch=math.ceil(train_data_num / BATCH_SIZE)
                              , epochs=EPOCHS
                              , verbose=1
                              , use_multiprocessing=True
                              #, callbacks=callbacks
                              , validation_data=valid_generator
                              , validation_steps=math.ceil(valid_data_num / BATCH_SIZE)
                             )
    print('model saveing ...')
    model.save_weights(model_filename)
    print('... saved')
    plotLearningCurve(his)


def plotLearningCurve(history):
    """ plotLearningCurve """
    x = range(EPOCHS)
    pyplot.plot(x, history.history['loss'], label="loss")
    pyplot.title("loss")
    pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#    pyplot.show()
    pyplot.savefig('LearningCurve.png')


def predict():
    """ predict """
    network = MaskRCNN(INPUT_SHAPE, 2, is_predict=True)
    model = network.get_model()
    model.load_weights(os.path.join(DIR_MODEL, FILE_MODEL), by_name=True)
    print('execute predict')


if __name__ == '__main__':
    train(Train_Mode.STEP1)
    train(Train_Mode.STEP2)
    train(Train_Mode.STEP3)
    train(Train_Mode.STEP4)
#    predict()

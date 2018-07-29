""" main """

import numpy as np
import os
import math
from enum import Enum
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

import keras.backend.tensorflow_backend as KB_tf
import keras.callbacks
import tensorflow as tf

from network.mask_rcnn import MaskRCNN, TrainTarget
from network.subnetwork.faster_rcnn import rpn_input_data
from data.coco_dataset import COCODataset, GenerateTarget, COCOCategory
from data.predict_dataset import PredictDataset
from data.utils.data_utils import DataUtils
from history_checkpoint_callback import HistoryCheckpoint


INPUT_SHAPE = (512, 512, 3)
#INPUT_SHAPE = (256, 256, 3)
BATCH_SIZE = 10
EPOCHS = 1000

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

    session = tf.Session('')
    KB_tf.set_session(session)
    KB_tf.set_learning_phase(1)

    backbone_shape = MaskRCNN.get_backbone_output_shape(INPUT_SHAPE)
    anchors = rpn_input_data.get_anchors(INPUT_SHAPE, backbone_shape)
    network = MaskRCNN(INPUT_SHAPE, 2
                       , anchors=anchors, batch_size=BATCH_SIZE, mask_size=28, roi_pool_size=14
                       , is_predict=False, train_targets= train_targets
                      )
    print('model compiling ...')
    model = network.get_model_with_default_compile()
    #network.draw_model_summary(file_name=os.path.join(os.pardir, 'ModelLayers.png'))
    #model.summary()
    print('... compiled')

    model_filename_base = os.path.join(DIR_MODEL, FILE_MODEL)
    pre_step_surfix = ''
    step_surfix = ''
    epochs = EPOCHS
    if mode == Train_Mode.STEP1:
        step_surfix = 'Step1'
        epochs = 2500
    elif mode == Train_Mode.STEP2:
        pre_step_surfix = 'Step1'
        step_surfix = 'Step2'
        epochs = 500
    elif mode == Train_Mode.STEP3:
        pre_step_surfix = 'Step2'
        step_surfix = 'Step3'
        epochs = 2500
    elif mode == Train_Mode.STEP4:
        pre_step_surfix = 'Step3'
        step_surfix = 'Step4'
        epochs = 4000

    if mode != Train_Mode.STEP1:
        model.load_weights(model_filename_base + '_' + pre_step_surfix + EXT_MODEL, by_name=True)
    model_filename = model_filename_base + '_' + step_surfix + EXT_MODEL

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
    callbacks = [keras.callbacks.TensorBoard(log_dir='./log/'
                                             , histogram_freq=0
                                             , write_graph=True
                                             , write_images=True)
                , keras.callbacks.ModelCheckpoint(filepath=model_filename
                                                  , verbose=1
                                                  , save_weights_only=True
                                                  , save_best_only=True
                                                  , period=20
                                                 )
                , HistoryCheckpoint(filepath='LearningCurve' + '_{history}_' + step_surfix + '.png'
                                   , verbose=1
                                   , period=20
                                   )
                , keras.callbacks.CSVLogger('train_log_' + step_surfix + '.csv')
                ]
    print('... created')

    print('fix ...')
    his = model.fit_generator(train_generator
                              , steps_per_epoch=math.ceil(train_data_num / BATCH_SIZE)
                              , epochs=epochs
                              , verbose=1
                              , use_multiprocessing=False # True
                              , callbacks=callbacks
                              , validation_data=valid_generator
                              , validation_steps=math.ceil(valid_data_num / BATCH_SIZE)
                             )
    print('model saveing ...')
    model.save_weights(model_filename)
    print('... saved')
    #saveLearningCurve(his, surfix=step_surfix)


def saveLearningCurve(history, surfix=None):
    """ saveLearningCurve """
    x = range(EPOCHS)
    pyplot.plot(x, history.history['loss'], label="loss")
    pyplot.title("loss")
    pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    lc_name = 'LearningCurve'
    if surfix is not None:
        lc_name += '_' + surfix
    pyplot.savefig(lc_name + '.png')
    pyplot.close()


def predict():
    """ predict """
    backbone_shape = MaskRCNN.get_backbone_output_shape(INPUT_SHAPE)
    anchors = rpn_input_data.get_anchors(INPUT_SHAPE, backbone_shape)
    network = MaskRCNN(INPUT_SHAPE, 2
                       , anchors=anchors, batch_size=BATCH_SIZE, mask_size=28, roi_pool_size=14
                       , is_predict=True
                      )
    print('model compiling ...')
    model = network.get_model_with_default_compile()
    print('... compiled')
    print('model weights loading ...')
    model_filename_base = os.path.join(DIR_MODEL, FILE_MODEL)
    pre_step_surfix = 'Step4'
    model.load_weights(model_filename_base + '_' + pre_step_surfix + EXT_MODEL, by_name=True)
    print('... loaded')

    print('predict data create ...')
    dataset = COCODataset(categories=['cat'], load_data=False)
    pred_dataset = PredictDataset(INPUT_SHAPE[:2], with_resize=True)
    print('... created')

    print('predict ...')
    input_names, inputs = pred_dataset.get_data_list()
    dummy_count = BATCH_SIZE - (pred_dataset.data_size() % BATCH_SIZE)
    input_names, inputs = input_names + input_names[:dummy_count], inputs + inputs[:dummy_count]
    inputs = np.array(inputs)
    preds = model.predict(inputs, BATCH_SIZE)
    print('... predicted')
    
    print('outputs saveing ...')
    zip_preds = network.zip_predict_by_batch(preds, len(inputs))
    zip_data = zip(input_names, inputs, zip_preds)
    for k, (name, img, pred) in enumerate(zip_data):
        regs, scrs, lbls, msks, rois, _, _ = pred

        lbls = np.squeeze(lbls, axis=1)
        positive_ids = np.where(lbls > 0)
        print('positive_ids : ', positive_ids)
        #regs = regs[positive_ids]
        #lbls = lbls[positive_ids]
        #scrs = scrs[positive_ids]
        #msks = msks[positive_ids]
        #rois = rois[positive_ids]
        lbl_names = []
        for c in lbls:
            ctg = dataset.get_category(dataset.class_index_to_category_id(c))
            lbl_name = '%2d' % c
            if isinstance(ctg, COCOCategory):
                lbl_name = ctg.name
            lbl_names.append(lbl_name)
        data = DataUtils(img, lbl_names, regs, msks, scores=scrs, rois=rois)
        save_path = os.path.join(pred_dataset.get_output_dir_name(), name)
        print('%3d' % k, ' : ', save_path)
        data.save(save_path)
    print('... saved')

    print('execute predict')


def aaaaa(tag):
    print('###############################################')
    print('################## ', tag, ' ###################')
    print('###############################################')


if __name__ == '__main__':
#    aaaaa('Step 1')
#    train(Train_Mode.STEP1)
#    aaaaa('Step 2')
#    train(Train_Mode.STEP2)
#    aaaaa('Step 3')
#    train(Train_Mode.STEP3)
#    aaaaa('Step 4')
#    train(Train_Mode.STEP4)
    aaaaa('Preds ')
    predict()

""" main """

import numpy as np
import os
from network.mask_rcnn import MaskRCNN, TrainTarget

INPUT_SHAPE = (256, 256, 3)
BATCH_SIZE = 256
EPOCHS = 100

DIR_MODEL = '.'
FILE_MODEL = 'MaskRCNN_Model.hdf5'

def train():
    """ train """
    print('execute train')

    # TODO
    train_inputs = None
    train_teachers = None
    test_inputs = None
    test_teachers = None

    train_taegets=[TrainTarget.BACKBONE, TrainTarget.RPN, TrainTarget.HEAD]

    network = MaskRCNN(INPUT_SHAPE, 2
                       , train_taegets= None, is_predict=True
                       #, train_taegets= train_taegets, is_predict=False
                      )
    model = network.get_model_with_default_compile()
    network.draw_model_summary(file_name=os.path.join(os.pardir, 'ModelLayers.png'))
    model.summary()
#    his = model.fit(train_inputs, train_teachers
#                    , batch_size=BATCH_SIZE
#                    , epochs=EPOCHS
#                    , validation_data=(test_inputs, test_teachers)
#                    , verbose=1)
#    model.save_weights(os.path.join(DIR_MODEL, FILE_MODEL))

def predict():
    """ predict """
    print('execute predict')

if __name__ == '__main__':
    train()
    predict()

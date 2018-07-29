"""
Predict Dataset Module
"""
import os
import glob
import cv2

from .utils import rpn_input_data as rpn_data
from .utils.image_utils import ImageUtils
from .utils.data_utils import DataUtils

class PredictDataset():
    """
    Predict Dataset Class
    """
    PREDICT_DIR = 'predict'
    INPUT_DIR = 'inputs'
    OUTPUT_DIR = 'outputs'

    def __init__(self, image_size=None, data_dir=None, load_data=True, with_resize=False):
        self.__dir = os.path.dirname(os.path.abspath(__file__))
        if data_dir is not None:
            self.__dir = data_dir
        self.__img_list = []
        self.__name_list = []
        self.__image_size = image_size
        if load_data:
            self.load_data(with_resize)


    def load_data(self, with_resize=False):
        """
        load_data
        """
        dir_name = os.path.join(self.__dir, self.PREDICT_DIR, self.INPUT_DIR)
        files = glob.glob(os.path.join(dir_name, '*.png'))
        files += glob.glob(os.path.join(dir_name, '*.jpg'))
        files += glob.glob(os.path.join(dir_name, '*.jpeg'))
        for file in files:
          self.__add_data(file, with_resize)


    def __add_data(self, file_path, with_resize=False):
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if with_resize and (self.__image_size is not None):
            img = cv2.resize(img, self.__image_size)
        self.__name_list.append(os.path.basename(file_path))
        self.__img_list.append(img)


    def data_size(self):
        """
        data_size
        """
        return len(self.__name_list)


    def get_data_list(self):
        """
        get_data_list
        """
        return self.__name_list, self.__img_list


    def get_output_dir_name(self):
        """
        get_output_dir_name
        """
        return os.path.join(self.__dir, self.PREDICT_DIR, self.OUTPUT_DIR)




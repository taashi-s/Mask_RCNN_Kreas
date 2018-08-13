import os
import cv2
import random
import warnings
import numpy as np
import skimage.io as io
import scipy.ndimage as scndi
import matplotlib.pyplot as plot
from PIL import Image
from enum import Enum
from pycocotools.coco import COCO

from .utils import rpn_input_data as rpn_data
from .utils.image_utils import ImageUtils
from .utils.data_utils import DataUtils

class GenerateTarget(Enum):
    """
    TODO : Write description
    Generate Target Enum
    """
    RPN_INPUT = 2
    HEAD_INPUT = 3


class COCOData():
    """
    COCO Data
    """
    def __init__(self, img_id, path, height, width, annotations):
        self.id = img_id
        self.path = path
        self.height = height
        self.width = width
        self.annotations = annotations


class COCOCategory():
    """
    COCO Category
    """
    def __init__(self, ctg_id, name):
        self.id = ctg_id
        self.name = name


class COCODataset():
    """
    COCO Dataset Class
    """
    ANNOTATION_DIR = 'annotations'
    ANNOTATION_FILE_PREFIX = 'instances_'
    ANNOTATION_FILE_EXT = '.json'
    IMAGE_DIR = 'images'

    def __init__(self, data_dir=None, data_type='val2017', categories=None, load_data=True):
        self.__dir = os.path.dirname(os.path.abspath(__file__))
        if data_dir is not None:
            self.__dir = data_dir
        self.__type = data_type
        self.__ann_json_path = os.path.join(self.__dir
                                            , self.ANNOTATION_DIR
                                            , self.__get_annotaion_filename()
                                           )
        self.__coco = COCO(self.__ann_json_path)
        self.__image_dir = os.path.join(self.__dir, self.IMAGE_DIR, self.__type)
        self.__update_categories_info(categories)
        self.__data_list = []
        if load_data:
            self.load_data()
            print('### COCODataset load data : ', self.data_size())


    def __get_annotaion_filename(self):
        return self.ANNOTATION_FILE_PREFIX + self.__type + self.ANNOTATION_FILE_EXT


    def __update_categories_info(self, categories):
        self.__ctg_ids = []
        if categories is None:
            self.__ctg_ids = sorted(self.__coco.getCatIds())
        else:
            self.__ctg_ids = self.__coco.getCatIds(catNms=categories)
        self.__categories = []
        for cid in self.__ctg_ids:
            self.__categories.append(COCOCategory(cid, self.__coco.loadCats(cid)[0]["name"]))


    def category_id_to_class_index(self, ctg_id):
        """
        category_id_to_class_index
        """
        id = None
        if ctg_id in self.__ctg_ids:
            id = self.__ctg_ids.index(ctg_id)
        return id


    def class_index_to_category_id(self, ind):
        """
        class_index_to_category_id
        """
        id = ind
        if len(self.__ctg_ids) > ind:
            id = self.__ctg_ids[ind]
        return id


    def get_category(self, ctg_id):
        """
        """
        for ctg_info in self.__categories:
            if ctg_info.id == ctg_id:
                return ctg_info
        return None


    def load_data(self):
        """
        load_data
        """
        image_ids = []
        for cid in self.__ctg_ids:
            image_ids.extend(list(self.__coco.getImgIds(catIds=[cid])))
        image_ids = list(set(image_ids))

        for iid in image_ids:
            self.__add_data(iid)


    def __add_data(self, image_id):
        image = self.__coco.imgs[image_id]
        image_path = os.path.join(self.__image_dir, image['file_name'])
        image_h = image['height']
        image_w = image['width']
        image_ann_ids = self.__coco.getAnnIds(imgIds=[image_id], catIds=self.__ctg_ids
                                              , iscrowd=False)
        image_anns = self.__coco.loadAnns(image_ann_ids)
        self.__data_list.append(COCOData(image_id, image_path, image_h, image_w, image_anns))


    def data_size(self):
        """
        data_size
        """
        return len(self.__data_list)


    def shuffle_data(self):
        """
        shuffle_data
        """
        return random.shuffle(self.__data_list)


    def get_coco(self):
        """
        get_coco
        """
        return self.__coco


    def get_data(self, id):
        """
        get_data
        """
        if self.data_size() <= id:
            return None

        return self.__data_list[id]


    def get_data_list(self):
        """
        get_data_list
        """
        return self.__data_list


    def generator(self, anchors, image_shape, max_objects=10, batch_size=None, target_data_list=None, genetate_targets=None):
        """
        keras data generator
        """
        data_list = self.get_data_list()
        if target_data_list is not None:
            data_list = target_data_list

        if batch_size is None:
            batch_size = self.data_size()
        if genetate_targets is None:
            genetate_targets = []
        include_rpns = GenerateTarget.RPN_INPUT in genetate_targets
        include_heads = GenerateTarget.HEAD_INPUT in genetate_targets
        image_list = []

        while True:
            random.shuffle(data_list)
            for data in data_list:
                if (image_list == []) or (len(image_list) >= batch_size):
                    image_list = []
                    rpn_classes_list = []
                    rpn_offsets_list = []
                    classes_list = []
                    regions_list = []
                    masks_list = []

                data_inputs = self.generate_data(data, image_shape, max_objects, anchors
                                                 , include_rpns, include_heads)
                img, cls_label, ofs_label, clss, regs, msks = data_inputs
                if img is None:
                    continue

                image_list.append(img)
                rpn_classes_list.append(cls_label)
                rpn_offsets_list.append(ofs_label)
                classes_list.append(clss)
                regions_list.append(regs)
                masks_list.append(msks)

                if len(image_list) >= batch_size:
                    inputs = [np.array(image_list)]
                    outputs = []
                    if include_rpns:
                        inputs += [np.array(rpn_classes_list), np.array(rpn_offsets_list)]
                    if include_heads:
                        inputs += [np.array(classes_list), np.array(regions_list)]
                        inputs += [np.array(masks_list)]

                    # print('')
                    # for k, inp in enumerate(inputs):
                    #    print('input(', k, ')>>> ', np.shape(inp))

                    yield inputs, outputs


    def generate_data(self, data, image_shape, max_objects, anchors
                      , include_rpn_inputs, include_head_inputs):
        """
        generate_data
        """
        error_return = None, None, None, None, None, None
        height, width, channel = image_shape
        reg_min_cofficent = 1 / 20
        reg_min_h = height * reg_min_cofficent
        reg_min_w = width * reg_min_cofficent

        img = io.imread(data.path)
        if len(img.shape) < channel:
            return error_return

        resize_data = ImageUtils().resize_with_keeping_aspect(img, height, width)
        resize_img, scale, padding = resize_data

        clss_tmp = []
        regs_tmp = []
        msks_tmp = []

        for annotation in data.annotations:
            mask = self.__coco.annToMask(annotation)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mask = scndi.zoom(mask, zoom=[scale, scale], order=0)
            resize_mask = np.pad(mask, padding, mode='constant')

            posivive_poss = np.where(resize_mask == 1)
            reg = np.array([np.min(posivive_poss[0]), np.min(posivive_poss[1])
                            , np.max(posivive_poss[0]), np.max(posivive_poss[1])])
            if (reg[2] - reg[0]) < reg_min_h or (reg[3] - reg[1]) < reg_min_w:
                continue

            clss_tmp.append(self.category_id_to_class_index(annotation['category_id']))
            regs_tmp.append(reg)
            msks_tmp.append(resize_mask)

        if clss_tmp == []:
            return error_return

        if len(clss_tmp) > max_objects:
            clss_tmp = clss_tmp[:max_objects]

        if len(regs_tmp) > max_objects:
            regs_tmp = regs_tmp[:max_objects]

        if len(msks_tmp) > max_objects:
            msks_tmp = msks_tmp[:max_objects]

        ofs_label, cls_label = None, None
        if include_rpn_inputs:
            ofs_label, cls_label = rpn_data.make_inputs(anchors, np.array(regs_tmp), height, width)

        clss, regs, msks = None, None, None
        if include_head_inputs:
            clss = np.zeros([max_objects])
            regs = np.zeros([max_objects, 4])
            msks = np.zeros([max_objects, height, width])

            clss[:len(clss_tmp)] = clss_tmp
            regs[:len(regs_tmp), :] = regs_tmp
            msks[:len(msks_tmp), :, :] = msks_tmp

            clss = np.expand_dims(clss, axis=1)

        return resize_img, cls_label, ofs_label, clss, regs, msks


    def show_image_with_label(self, image_id, image_shape, anchors, max_objects=10):
        """
        show_image_with_label
        """
        target = self.__data_list[image_id]
        if self.data_size() < image_id:
            return False
        image_filename = os.path.basename(target.path)
        data = self.generate_data(target, image_shape, max_objects, anchors, True, True)
        img, _, _, clss, regs, msks = data
        if img is None:
            print('data is None : ', image_filename)
            return
        idx_pos = np.where(np.any(regs, axis=1))[0]
        cls_names = [(self.get_category(c)).name for c in clss]
        DataUtils(img, cls_names, regs[idx_pos], msks[idx_pos]).show()


    def save_image_with_label(self, image_id, image_shape, anchors, save_dir, max_objects=10):
        """
        save_image_with_label
        """
        target = self.__data_list[image_id]
        if self.data_size() < image_id:
            return False
        image_filename = os.path.basename(target.path)
        data = self.generate_data(target, image_shape, max_objects, anchors, True, True)
        img, _, _, clss, regs, msks = data
        if img is None:
            print('data is None : ', image_filename)
            return
        idx_pos = np.where(np.any(regs, axis=1))[0]
        cls_names = [(self.get_category(c)).name for c in clss]
        DataUtils(img, cls_names, regs[idx_pos], msks[idx_pos]).show()


    def show_data_image(self, id, with_annotations=False):
        """
        data_size
        """
        if self.data_size() <= id:
            return False

        data = self.__data_list[id]
        img = io.imread(data.path)
        io.imshow(img)
        if with_annotations:
            self.__coco.showAnns(data.annotations)
        io.show()
        return True


if __name__ == '__main__':
    """
    """
    dataset = COCODataset(categories=['cat'])
    data_len = dataset.data_size()
    """
    for i in range(5):
        n = random.randrange(0, data_len)
        #dataset.show_data_image(n, with_annotations=True)
        d = dataset.get_data(n)
        print('data[%3d] : '%(n), '(', d.width, ', ', d.height, ')')
    """
    print('data count : ', data_len)
    data_list = dataset.get_data_list()
    print('create anchors ...')
    INPUT_SHAPE = (512, 512, 3)
    ACS = rpn_data.get_anchors(INPUT_SHAPE)
    print('... finish')
    hs = []
    ws = []
    for d in data_list:
        hs.append(d.height)
        ws.append(d.width)
#    print('max h : ', max(hs))
#    print('max w : ', max(ws))
    loop = 5
    if data_len > loop:
        ddd = []
        """
        for iii in range(loop):
            ddd.append(dataset.generate_data(data_list[iii], INPUT_SHAPE, 2, ACS, True, True))

        for iii in range(loop):
            img, _, _, clss, regs, msks = ddd[iii]
            DataUtils(img, clss, regs, msks).show()
        """
        SAVE_DIR = os.path.join(os.path.dirname(__file__), 'tmp')
        id_ofs = 118
        for iii in range(data_len - id_ofs):
            print('[Save] : ', iii + id_ofs)
            dataset.save_image_with_label(iii + id_ofs, INPUT_SHAPE, ACS, SAVE_DIR)


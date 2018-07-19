
import os
import numpy as np
from numpy import random as nprand
from pycocotools.coco import COCO
import skimage.io as io
from PIL import Image
import matplotlib.pyplot as plot


class COCODataset():
    """
    COCODataset Class
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
        #self.__categories = categories
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
            self.__categories.append({'id' : cid
                                      , 'name' : self.__coco.loadCats(cid)[0]["name"]
                                     })


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
        image_ann_ids = self.__coco.getAnnIds(imgIds=[image_id], catIds=self.__ctg_ids)
        image_anns = self.__coco.loadAnns(image_ann_ids)
        data = {'id' : image_id
                , 'path' : image_path
                , 'height' : image_h
                , 'width' : image_w
                , 'annotataions' : image_anns
               }
        self.__data_list.append(data)


    def get_coco(self):
        """
        get_coco
        """
        return self.__coco

    def get_data_list(self):
        """
        get_data_list
        """
        return self.__data_list


if __name__ == '__main__':
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    data_type = 'val2017'
    annotation_file = '%s/annotations/instances_%s.json'%(data_dir, data_type)

    coco = COCO(annotation_file)
    categories = coco.loadCats(coco.getCatIds())
    cat_names = [cat['name'] for cat in categories]
    # print('COCO categories : \n', cat_names)

    cat_ids = coco.getCatIds(catNms=['cat'])
    cat_img_ids = coco.getImgIds(catIds=cat_ids)
    img_data = coco.loadImgs(cat_img_ids[nprand.randint(0,len(cat_img_ids))])[0]
    img_path = '%s/images/%s/%s'%(data_dir, data_type, img_data['file_name'])
    img = io.imread(img_path)
    io.imshow(img)
    # img = Image.open(img_path)
    # img.show()
    ann_ids = coco.getAnnIds(imgIds=img_data['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    coco.showAnns(anns)
    io.show()
    """
    dataset = COCODataset(categories=['cat'])
    data_list = dataset.get_data_list()
    coco = dataset.get_coco()

    if len(data_list) > 0:
        img = io.imread(data_list[0]['path'])
        io.imshow(img)
        coco.showAnns(data_list[0]['annotataions'])
        io.show()



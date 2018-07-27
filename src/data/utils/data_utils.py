"""
Data Utils Module
"""

import cv2
import numpy as np
import skimage.io as io


class DataUtils():
    """
    Data Utils Class
    """
    def __init__(self, image, classes, regions, masks, scores=None, rois=None):
        self.__image = image
        self.__clsses = classes
        self.__regions = regions
        self.__masks = masks
        self.__rois = rois
        self.__scores = [None for _ in range(len(classes))]
        if scores is not None:
            self.__scores = scores


    def show(self, with_regions=True, with_masks=True):
        """
        show
        """
        io.imshow(self.make_image(with_regions=with_regions, with_masks=with_masks))
        io.show()


    def save(self, save_path, with_regions=True, with_masks=True):
        """
        save
        """
        cv2.imwrite(save_path, self.make_image(with_regions=with_regions, with_masks=with_masks))


    def make_image(self, with_regions=True, with_masks=True):
        """
        make_image
        """
        img = np.flip(self.__image, axis=2).astype(np.uint8)
        zip_data = zip(self.__clsses, self.__scores, self.__regions, self.__masks)
        col = [i for i in range(255)[::(255 // self.__regions.shape[0] - 1)]]
        for k, (cla, scr, reg, msk) in enumerate(zip_data):
            reg = reg.astype(np.uint32)
            msk = msk.astype(np.uint8)
            color = (col[k], col[::-1][k], 0)
            reg_lt = (reg[1], reg[0])
            reg_rb = (reg[3], reg[2])

            caption = 'class(' + cla + ')'
            if scr is not None:
                caption += ' : .3f' % scr
            cv2.putText(img, caption, reg_lt, cv2.FONT_HERSHEY_PLAIN, 1, color)
            cv2.rectangle(img, reg_lt, reg_rb, color, 2)
            for ch in range(3):
                img[:, :, ch] = np.where(msk == 1
                                         , img[:, :, ch] * 0.5 + 0.5 * color[ch] * 255
                                         , img[:, :, ch])
        return img


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
    def __init__(self, image, classes, regions, masks):
        self.__image = image
        self.__clsses = classes
        self.__regions = regions
        self.__masks = masks


    def show(self, with_regions=True, with_masks=True):
        img = np.flip(self.__image, axis=2).astype(np.uint8)
        idx_pos = np.where(np.any(self.__regions, axis=1))[0]
        regs = self.__regions[idx_pos]
        msks = self.__masks[idx_pos]
        c = [i for i in range(255)[::(255 // regs.shape[0] - 1)]]
        i = 0
        for reg, mask in zip(regs, msks):
            reg = reg.astype(np.uint32)
            mask = mask.astype(np.uint8)
            color = (c[i], c[::-1][i], 0)
            cv2.rectangle(img, (reg[1], reg[0]), (reg[3], reg[2]), color, 2)
            mask = np.dstack([mask, mask, mask])
            mask[:, :, 0][mask[:, :, 0] == 1] = color[0]
            mask[:, :, 1][mask[:, :, 1] == 1] = color[1]
            mask[:, :, 2][mask[:, :, 2] == 1] = color[2]
            cv2.addWeighted(mask, 1, img, 1, 0, img)
            i += 1
        io.imshow(img)
        io.show()


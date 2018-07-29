"""
Data Utils Module
"""

import cv2
import numpy as np
import skimage.io as io
import scipy.misc

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
        img_h, img_w = img.shape[0], img.shape[1]
        zip_data = zip(self.__clsses, self.__scores, self.__regions, self.__masks)
        col = [i for i in range(255)[::(255 // self.__regions.shape[0] - 1)]]
        for k, (cla, scr, reg, msk) in enumerate(zip_data):
            reg = reg.astype(np.uint32)
#            msk = msk.astype(np.uint8)
            color = (col[k], col[::-1][k], 0)
            reg_lt = (reg[1], reg[0])
            reg_rb = (reg[3], reg[2])
            if (reg[0] < 0 or img_h < reg[0]) or (reg[2] < 0 or img_h < reg[2]) or (reg[1] < 0 or img_w < reg[1]) or (reg[3] < 0 or img_w < reg[3]):
                  continue

            caption = 'class(' + cla + ')'
            if scr is not None:
                caption += ' : .3f' % scr
            cv2.putText(img, caption, reg_lt, cv2.FONT_HERSHEY_PLAIN, 1, color)
            cv2.rectangle(img, reg_lt, reg_rb, color, 2)
            self.add_mask(img, msk, reg, color, (img.shape[0], img.shape[1]))
#            for ch in range(3):
#                img[:, :, ch] = np.where(msk == 1
#                                         , img[:, :, ch] * 0.5 + 0.5 * color[ch] * 255
#                                         , img[:, :, ch])
        return img


    def add_mask(self, dest_img, mask, bbox, color, image_shape):
        threshold = 0.5
        y1, x1, y2, x2 = bbox
        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0:
          return
        mask = scipy.misc.imresize(mask, (h, w),interp='bilinear').astype(np.float32)
        mask /= 255.0
        mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

        _y1, _x1, _y2, _x2 = max(0, y1), max(0, x1), min(image_shape[0], y2), \
            min(image_shape[1], x2)
        d_y1, d_x1, d_y2, d_x2 = _y1 - y1, _x1 - x1, _y2 - y2, _x2 - x2
        mask = mask[d_y1:h + d_y2, d_x1:w + d_x2]

        fullsize_mask = np.zeros(image_shape, dtype=np.uint8)
        fullsize_mask[_y1:_y2, _x1:_x2] = mask

        mask_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        mask_image[:, :] = color
        mask_image = cv2.bitwise_and(mask_image, mask_image, mask=fullsize_mask)
        cv2.addWeighted(mask_image, 1.5, dest_img, 1, 0, dest_img)

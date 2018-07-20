"""
image utils Module
"""
import numpy as np
import skimage.transform as ski_trans

class ImageUtils():
    """
    Image Utils Class
    """
    def __init__(self):
        pass


    def resize_with_keeping_aspect(self, image, height, width):
        """
        resize_with_keeping_aspect
        """
        src_h, src_w = image.shape[:2]
        scale = self.__get_resize_scale(image, height, width)
        resize_h, resize_w = round(src_h * scale), round(src_w * scale)
        if scale != 1:
            resize_img = ski_trans.resize(image, (resize_h, resize_w)
                                          , mode="constant", preserve_range=True)


        top_pad, bottom_pad = self.__split_to_two_int(height - resize_h)
        left_pad, right_pad = self.__split_to_two_int(width - resize_w)

        padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
        pad_img = np.pad(resize_img, padding + [(0, 0)], mode='constant')
        # org_img_pos = (top_pad, left_pad, resize_h + top_pad, resize_w + left_pad)
        return pad_img.astype(image.dtype), scale, padding


    def __get_resize_scale(self, image, height, width):
        src_h, src_w = image.shape[:2]
        scale_h = max(0, height / src_h)
        scale_w = max(0, width / src_w)
        scale = scale_h
        if scale_w < scale_h:
            scale = scale_w
        return scale


    def __split_to_two_int(self, src):
        a = src // 2
        return a, src - a

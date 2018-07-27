"""
COCO Dataset Chk Module
"""

import os

from data.coco_dataset import COCOData, COCOCategory, COCODataset
from data.utils import rpn_input_data as rpn_data

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
        SAVE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'tmp_1')
        id_ofs = 0
        for iii in range(data_len - id_ofs):
            print('[Save] : ', iii + id_ofs)
            dataset.save_image_with_label(iii + id_ofs, INPUT_SHAPE, ACS, SAVE_DIR)



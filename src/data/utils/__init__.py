import sys
import os

# TODO : refactoring path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'
                                                      , '..'
                                                      , 'network'
                                                      , 'subnetwork'
                                                      #, 'faster_rcnn'
                                                      #, 'src'
                                                      #, 'network'
                                                      #, 'subnetwork'
                                                      #, 'rpn'
                                                      #, 'layers'
                                                      #, 'utils'
                                                      ))

print(sys.path)

from .image_utils import ImageUtils
from .data_utils import DataUtils
#from layers.utils import RegionsUtils
#from rpn import rpn_input_data
from faster_rcnn import rpn_input_data
import sys
import os

# TODO : refactoring path
aaa = os.path.join(os.path.dirname(__file__), '..'
                                                      , '..'
                                                      , 'network'
                                                      , 'subnetwork'
                                                      , 'faster_rcnn'
                                                      , 'src'
                                                      , 'network'
                                                      , 'subnetwork'
                                                      , 'rpn'
                                                      , 'layers'
                                                      , 'utils'
                                                      )
print(aaa)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'
                                                      , '..'
                                                      , 'network'
                                                      , 'subnetwork'
                                                      , 'faster_rcnn'
                                                      , 'src'
                                                      , 'network'
                                                      , 'subnetwork'
                                                      , 'rpn'
                                                      #, 'layers'
                                                      #, 'utils'
                                                      ))

from .image_utils import ImageUtils
from .data_utils import DataUtils
from layers.utils import RegionsUtils
import rpn_input_data

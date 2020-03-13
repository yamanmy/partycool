import numpy as np
import cv2
from matplotlib import pyplot as plt

from watershed.py import *

#Test for the watershed function

def test_watershed():
	result = watershed('/Users/margheritataddei/Desktop/DIRECT/PROJECT/partycool/example_images/cut_images/sem_1_cut_zoom2.jpg', 0.1)
	assert result == <matplotlib.image.AxesImage at 0x123c23650>, 'Wrong result'

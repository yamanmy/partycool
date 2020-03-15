import numpy as np
import cv2

from watershed import *

def test_watershed():
	image= '/Users/margheritataddei/Desktop/DIRECT/PROJECT/partycool/example_images/cut_images/zoom/sem_2_cut_zoom.jpg'
	result = watershed(image)
	assert result.size == 8400, "Wrong size of the contours matrix"
	assert result.shape == (80, 105), "Wrong shape of the countours matrix"

#The end 

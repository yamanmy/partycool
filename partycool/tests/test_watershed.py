import numpy as np
import cv2

from watershed import *

def test_watershed():
	image= './partycool/example_images/watershed_trial/ws_sem_2_0.jpg'
	result = watershed(image)
	assert result.size == 8400, "Wrong size of the contours matrix"
	assert result.shape == (80, 105), "Wrong shape of the countours matrix"

#The end 

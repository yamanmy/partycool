import cv2
from partycool import partycool


#Testing database
test = cv2.imread('../../example_images/sem_1.jpg',0)

#Test for boundary detection

def test_boundary_detection():
    result = partycool.boundary_detection(test)
    
    assert type(result) == int, 'Wrong output type in boundary detection function'
    assert result == 593, "Wrong output result in boundary detection function"
    return

#Test for corner detection

def test_corner_detection():
    result = partycool.corner_detection(test,1)
    
    assert result == 1.1848341232227488, "Wrong output result of the corner detection function"
    assert result.shape == (), "Wrong output shape of the corner detection function"
    assert result.size == 1, "Wrong output size of the corner detection function"
    return

#Test img_pread function

def test_img_pread():
    result = partycool.img_pread(test)
    
    assert result.shape == (2048, 3072), "Wrong output shape of the img_pread function"
    assert result.size == 6291456, "Wrong  output size of the img_pread function"
    
    return

#Testing filetered dataframe used for testing the following functions



#Test contour capture function

def test_contour_capture():
    test_filtered = partycool.img_pread(test)
    result_cnt = partycool.contour_capture(test_filtered)
    
    assert len(result_cnt) == 316, 'wrong output'
    
    return

#Testing dataframe from contour capture for the following functions



#Test shape radar function

def test_shape_radar():
    test_filtered = partycool.img_pread(test)
    result_cnt = partycool.contour_capture(test_filtered)
    result = partycool.shape_radar(result_cnt, test)
    
    assert len(result) == 2188, 'Wrong output'
    
    return

#Test the wrapping function of partycool summary

def test_partycool_summary():
    test_filtered = partycool.img_pread(test)
    result_cnt = partycool.contour_capture(test_filtered)
    result_df = partycool.partycool_summary(result_cnt)
    
    assert len(result_df.columns) == 5 or len(result_df.columns) == 8, 'wrong output dataframe!'
    
    return

#Test the watershed function


def test_watershed():
    test_img = '../../example_images/sem_1.jpg'
    result = partycool.watershed(test_img)
    
    assert result.size == 6721536, "Wrong output size of the watershed result matrix"
    assert result.shape == (2188, 3072), "Wrong output shape of the watershade result matrix"
    
    return




#THE END
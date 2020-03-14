import numpy as np
import cv2

#Function to calculate the area of borders of two close objects
#The watershed function needs the image and we iterate over the iteration in the opening function (o_iter), the iterations in the background detection (s_iter) and on i which define how much of the area of the particle is considered.
#The smaller the i, the larger would be the final boundaries
#The larger the i, the smaller will be the final boundaries
#After a lot of trials with different images we found out that the best contours are found when i is between 0.1 and 0.7, o_iter is between 1 and 5 and s_iter is between 1 and 2. 

def watershed(image):
    my_range = np.arange(0.0, 0.7, 0.1)
    img_3channel = cv2.imread(image, 1)
    img = cv2.imread(image, 0)
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    for o_iter in range(1,5):
        opening = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel, iterations = o_iter)
    for s_iter in range(1,2):
        sure_bg = cv2.dilate(opening,kernel,iterations= s_iter)
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    for i in my_range:
        ret, sure_fg = cv2.threshold(dist_transform,i*dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        ret, contours = cv2.connectedComponents(sure_fg)
        contours = contours+1
        contours[unknown==255] = 0
        contours = cv2.watershed(img_3channel ,contours)
        img[contours == -1] = [0]
    
    return contours


#The end

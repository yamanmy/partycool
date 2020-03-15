# Image pretreatment
def boundary_detection(img, thres = 20):
        '''
        thres: threshold to distinguish the scale bar background with particle background by grey scale
        for now only work for SEM img, needs update if apply to different imgs
        scan from upper to bottom, also needs update if need scan from left to right
        
        img: input image in gray scale
        thres: threshold for contrast of distinguishing the boundary, i.e larger thres means higher contrast for boundary
        '''
        mode_list = []
        for line in range(len(img)):
            mode = stats.mode(img[line])
            mode_list.append(int(mode[0]))

            if line >= 1:
                mode_mean = mean(mode_list)
                if mode_mean - int(mode[0]) >= thres:
                    boundary = line
                    break

        return boundary


# Watershed for distinguish shapes -- beta 
def watershed(image):
    my_range = np.arange(0.0, 0.7, 0.1)
    img_3channel = cv2.imread(image, 1)
    img = cv2.imread(image, 0)
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    skel = np.zeros(th.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7,7))
    open = cv2.morphologyEx(th, cv2.MORPH_OPEN, element)
    temp = cv2.subtract(th, open)
    eroded = cv2.erode(th, element)
    skel = cv2.bitwise_or(skel,temp)
    erod = eroded.copy()
    for s_iter in range(1,5):
        sure_bg = cv2.dilate(erod,element,iterations= s_iter)
        dist_transform = cv2.distanceTransform(erod,cv2.DIST_L2,5)
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

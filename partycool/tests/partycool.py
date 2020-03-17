import numpy as np
import cv2
import skimage
import math
from skimage import io
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.morphology import reconstruction
from scipy import stats
from statistics import mean 
from collections import OrderedDict
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

#Optional modules
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform

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

#Scale bar detection and calculation
def corner_detection(img, actual_len):
    """
    This function is used to find the length of each pixel in nm.
    The unit of the output length_each_pixel is nm
    
    img: input image in gray scale
    actual_len: real length in micrometer
    """ 
    
    def dilated_image(img,sigma=1):
        """
        filtering regional maxima to find bright features by 
        using gaussian filter and reconstruction
        simga: standard deviation for Gaussian kernel 
        """
        # Convert to float: Important for subtraction later which won't work with uint8
        img = gaussian_filter(img, sigma)
        seed = np.copy(img, sigma)
        seed[1:-1, 1:-1] = img.min()
        mask = img

        dilated = reconstruction(seed, mask, method='dilation')
        return dilated
    
    actual_len = actual_len*1000
    height = img.shape[0]
    width = img.shape[1]
    #find the bottom part of the SEM image. Here we used the return refunction 
    ime = img[boundary_detection(dilated_image(img,1)): , : ]
    
    # find the smallest area of interest
    boundary_v = []
    thres = 100
    for i in range(ime.shape[1]):
        if ime[:,i][0] > thres:
            boundary_v.append(i)
    
    #determine the smaller one of the scale bar region
    ime = img[boundary_detection(dilated_image(img,1)): , boundary_v[-1]+10: ]
    
    boundary_h = []
    for i in range(ime.shape[0]):
        if ime[i,:][0] > thres:
            boundary_h.append(i)
    ime = img[boundary_detection(dilated_image(img,1)):boundary_detection(dilated_image(img,1))+boundary_h[0] , boundary_v[-1]+10: ]
    
    tform = AffineTransform()
    image = warp(ime,tform.inverse)
    coords = corner_peaks(corner_harris(image))
    coords_subpix = corner_subpix(image, coords)
    
    #get the length of the scale bar
    #length_scale_bar = abs(coords[0][1] - coords[1][1])
    
    scales = []
    threshold = 500
    for i in range(len(coords)):
        for j in range(len(coords)):
            if j <= i:
                continue
            else:
                if coords[i][0] == coords[j][0]:
                    scale = abs(coords[i][1] - coords[j][1])
                    if scale > threshold:
                        scales.append((coords[i][0],scale))
                    else:
                        continue
                else:
                    continue
    scalebar = []
    for i in range(len(scales)):
        n_count = 0
        for j in range(len(scales)):
            if scales[i][0] == scales[j][0]:
                n_count += 1
            else:
                continue
        if n_count == 1:
            scalebar.append(scales[i][1])
        else:
            continue
    
    for i in range(len(scalebar)):
        num = scalebar.count(scalebar[i])
        if num >= 2:
            final_scale = scalebar[i]
        else:
            continue
    
    length_each_pixel = actual_len/final_scale
    
    return length_each_pixel

#Image read and contour capture module
def img_pread(img, thres = 20, cut = True):
    '''
    Pretreatment for the picture to get a dilated and boundary cutted image
    
    img: input image in gray scale
    thres: threshold for contrast distinguishing the boundary
    cut: boolean value to set if the img be cutted
    '''
    #Pretreatment for the boundary detection
    image = img
    image = gaussian_filter(image, 1)
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    dilated = reconstruction(seed, mask, method='dilation')
    
    if cut == True:
        image = mask - dilated
        bound = boundary_detection(dilated)
        img_c = image[:bound,:]
        img_c = img_c.astype(np.uint8)
    else:
        img_c = image
    
    return img_c


def contour_capture(img, 
                    noise_factor = 0.25,
                    thresh_method = cv2.THRESH_BINARY,
                    area_thresh = 300):
    '''
    The function captures the contours from the given imgs
    Returns contours
    
    img: input image in gray scale
    noise_factor: factor used to set threshold for the threshold function
    thresh_method: please refer to cv2.threshold
    area_thresh: threshold to ignore noise contours
    '''
    _, threshold = cv2.threshold(img, img.max() * noise_factor, img.max(), thresh_method)
    contours, _=cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= area_thresh]
    
    return contours


def peri_avg(contours):
    '''
    Assistant function for determine the average perimeter from all contours
    
    contours: contours from the image
    '''
    peri_c_tot = 0
    for c in contours:
        peri_c_tot += cv2.arcLength(c, True)
    avg_peri = peri_c_tot / len(contours)

    return avg_peri


#Main module for shape detection
def shape_radar(contours, img, thresh_di = 1.09, thres_poly = 1.75):
    '''
    Takes input from contour_capture
    return a annotated img from setted threshold
    Model tunning is possible by using different predictions provided below
    
    contours: contours from the image
    img: dilated image from previous function
    '''

    plt.figure(figsize=(20,16))
    dilated_c = img.copy()
    dilated_c = cv2.cvtColor(dilated_c,cv2.COLOR_GRAY2RGB)
    avg_c = peri_avg(contours)
    
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        #Optional predictors
        area = cv2.contourArea(c)
        len_c = round(cv2.arcLength(c, True), 1)
        r_area_len = round((area/len_c),1)
        r_peri = len_c / avg_c

        if r_peri <= thresh_di:
            cv2.drawContours(dilated_c, [box], 0, (255, 255, 255), 3)
        elif r_peri > thresh_di and r_peri <= thres_poly:
            if area > 900:
                cv2.putText(dilated_c, 'dimer', (c[0][0][0], c[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0), 3)
                cv2.drawContours(dilated_c, [box], 0, (255, 0, 0), 3)
            else:
                cv2.drawContours(dilated_c, [box], 0, (255, 255, 255), 3)
        elif r_peri > thres_poly:
            cv2.putText(dilated_c, 'polymer', (c[0][0][0], c[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0), 3) 
            cv2.drawContours(dilated_c, [box], 0, (0, 255, 0), 3)

    return dilated_c


def partycool_summary(contours, convert_factor = 0, enable_poly = True, thresh_di = 1.09, thres_poly = 1.75, spherical= False ):
    '''
    returns a dataframe that summarized the particle information
    '''
    
    if enable_poly == True:
        contours = contours
    elif enable_poly == False:
        avg_c = peri_avg(contours)
        contours = [contour for contour in contours if (cv2.arcLength(contour, True) / avg_c) <= thresh_di]
    
    
    
    loc_x = []
    loc_y = []
    aspect_r = []
    area_l = []
    arealen_r = []
    category_l = []
    cX_au=[]
    cY_au=[]
    height = []
    width = []
    com_arr = np.empty((0, 2))
    angles = []
    
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h
        area = cv2.contourArea(c)
        avg_c = peri_avg(contours)
        len_c = round(cv2.arcLength(c, True), 1)
        r_peri = len_c / avg_c
        # center of mass of each particle
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY= int(M["m01"] / M["m00"])
        else:
          # set values as what you need in the situation
            cX, cY = 1,1  
        
        #particle arientation
             
        try:
            (com), _, angle = cv2.fitEllipse(c)
        except:
            continue
        com = np.array(com)
        com_arr = np.append(com_arr, [com], axis=0)
        angles.append(int(angle))
    
        
        category = ''

        if r_peri <= thresh_di:
            category = 'monomer'
        elif r_peri > thresh_di and r_peri <= thres_poly:
            if area > 900:
                category = 'dimer'
            else:
                category = 'monomer'
        elif r_peri > thres_poly:
            category = 'polymer'

        loc_x.append(x)
        loc_y.append(y)
        aspect_r.append(float('%.2g' % aspect_ratio))
        area_l.append(int(area))
        category_l.append(category)
        cX_au.append(cX)
        cY_au.append(cY)
        height.append(w)
        width.append(h)
    
    #aspact ratio correction
    aspect_ratio = []
    for i in range(len(aspect_r)):
        if aspect_r[i] < 1:
            aspect_ratio.append(float('%.2g' % (1/aspect_r[i])))
        else:
            aspect_ratio.append(float('%.2g' % aspect_r[i]))
    #Radius calculation
    #Conversion part
    if convert_factor != 0:
        area_list = [int(a * (convert_factor ** 2)) for a in area_l]
        height_list = [int(a*convert_factor) for a in width]
        width_list = [int(a*convert_factor) for a in height]
        #Radius calculation
        r = [int(math.sqrt(a/math.pi)) for a in area_list]
    elif convert_factor == 0:
        area_list = [int(a) for a in area_l]
        height_list = [int(a) for a in width]
        width_list = [int(a) for a in height]
        r = [int(math.sqrt(a/math.pi)) for a in area_list]
        
    if spherical==True:
        result_dict = {'Cx' : cX_au, 
                       'Cy' : cY_au,
                       'Area (nm\u00b2)' : area_list,
                       'Radius (nm)' : r,
                       'Category' : category_l}
    elif spherical==False:
          result_dict = {'Cx' : cX_au, 
                         'Cy' : cY_au,
                         'Height (nm)' : width_list,
                         'Width (nm)' : height_list,
                         'Aspect Ratio' : aspect_ratio,
                         'Area (nm\u00b2)' : area_list,
                         'Angles (°)' :angles,
                         'Category' : category_l}
                     #change the 'Area (nm\u00b2)' to area_list        
                
    result_df = pd.DataFrame.from_dict(result_dict)
    
    return result_df


def partycool_plots(df, interactive = False):
    '''
    dataframe: df from partycool_summary
    aspect: what user want to see
    interactive: define whether the plots be interactive
    '''
    if interactive == False:
        plt.rc('xtick', labelsize = 16)
        plt.rc('xtick.major', pad = 15)
        plt.rc('ytick', labelsize = 20)
        plt.rc('lines', lw = 2.2, markersize = 7.5)
        plt.rc('legend', fontsize = 16)
        plt.rc('axes', linewidth=2.2)

        fig, ax = plt.subplots(2, 2, figsize=(16, 12))

        ax[0][0].hist(df['Angles (°)'], bins=40, color='purple',label = 'Area')
        ax[0][0].set_xlabel('Angles (°)', fontsize = 18)
        ax[0][0].set_ylabel('Count', fontsize = 18)
        ax[0][0].grid(True)

        ax[0][1].hist(df['Aspect Ratio'], bins=40, color='purple',label = 'Area')
        ax[0][1].set_xlabel('Aspect Ratio', fontsize = 18)
        ax[0][1].set_ylabel('Count', fontsize = 18)
        ax[0][1].grid(True)

        ax[1][0].hist(df['Height (nm)'], bins=40, color='purple',label = 'Area')
        ax[1][0].set_xlabel('Length of height (nm)', fontsize = 18)
        ax[1][0].set_ylabel('Count', fontsize = 18)
        ax[1][0].grid(True)

        ax[1][1].hist(df['Width (nm)'], bins=40, color='purple',label = 'Area')
        ax[1][1].set_xlabel('Length of width (nm)', fontsize = 18)
        ax[1][1].set_ylabel('Count', fontsize = 18)
        ax[1][1].grid(True)
        
    elif interactive == True:
        fig = make_subplots(rows=2, cols=2)
        subplot_titles = ['Area', 'Aspect Ratio', 'Height', 'Width']
        xaxis_title=['Angles','Aspect Ratio','Height (nm)', 'Width (nm)']
        yaxis_title='count'
        rows = 2
        cols = 2
        trace = [go.Histogram(x=df['Angles (°)'], marker_color='#800080',opacity=0.75),\
                 go.Histogram( x=df['Aspect Ratio'],opacity=0.75),\
                go.Histogram(x=df['Height (nm)'],opacity=0.75),go.Histogram(x=df['Width (nm)'],opacity=0.75)]

        for i, col in enumerate(subplot_titles):
            r = int(np.ceil(((i+1)/cols)))
            c = i%2+1
            fig.add_trace(trace[i], row=r, col=c)
            fig.update_xaxes(title_text=xaxis_title[i], row=r, col=c)
            fig.update_yaxes(title_text=yaxis_title, row=r, col=c)
            fig.update_layout(bargap=0.15,showlegend=False) 
        fig.show()
    
    #pie chart
    count_mono = 0
    count_di = 0
    count_poly = 0
    for i in range(len(df['Category'])):
        if df['Category'][i] =='monomer':
            count_mono +=1
        elif df['Category'][i] =='dimer':
            count_di += 1
        elif df['Category'][i] =='polymer':
            count_poly += 1
    colors = ['gold', 'darkorange', 'lightgreen']
    fig = go.Figure(data=[go.Pie(labels=['Monomer','Dimer','Polymer'],
                                 values=[count_mono, count_di, count_poly],title='Categories')])
    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=14,
                      marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    fig.show()
    
    return


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

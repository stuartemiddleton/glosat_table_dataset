# !/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# (c) Copyright University of Southampton, 2020
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Created By : Juliusz Ziomek
# Created Date : 2020/09/09
# Project : GloSAT
#
######################################################################

import cv2
import numpy as np
from sklearn.cluster import KMeans

#Ratios that indicate an edge
VERTICAL_EDGE = 1/5
HORIZONTAL_EDGE = 10
CELL_DIM_THRESHOLD = 0.05

#Character or cell decision
MIN_THRESHOLD = 0.01
MAX_CELL_AREA = 0.25

def put_box(image,box,colour,text=None,thickness=2):
    image= cv2.rectangle(image,((int)(box[0]),(int)(box[1])),((int)(box[2]),(int)(box[3])),colour,thickness=thickness)
    
    if text:
        image = cv2.putText(image,text,((int)(box[0]),(int)(box[1])),cv2.FONT_HERSHEY_SIMPLEX,fontScale = 1,color = colour,thickness = thickness)

    return image

def put_line(image,start,end,colour,thickness=3):
    image = cv2.line(image,start,end,color=colour,thickness=thickness)
    return image

def erosion(image,kernel_size=3,iters = 1):
    image = cv2.erode(image.astype(np.uint8), np.ones((kernel_size,kernel_size), np.uint8) , iterations=iters)
    return image

def find_contours(image,mode=cv2.RETR_EXTERNAL):
    
    ret, thresh = cv2.threshold(image.astype(np.uint8), 50, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, mode, cv2.CHAIN_APPROX_TC89_KCOS)

    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        boxes.append([x,y,x+w,y+h])

    return boxes

def filter_pixels(labels_im,label):
    '''
    Filter the mask, removing the pixels not belonging to the layer.
    '''

    filtered_im = np.zeros_like(labels_im)

    filtered_im[labels_im==label] = 255

    return filtered_im

def normalise_contrast(img):
    '''
    Performs local contrast normalisation
    '''
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_normalised = clahe.apply(img)

    return img_normalised

def preprocess(image):
    image = normalise_contrast(image)
    image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)[1]  
    return image

def postprocess(region,image_shape,area_threshold=0.001,dim_threshold=0.02):
    area = lambda box: (box[3] - box[1]) * (box[2] - box[0])

    if area(region)< area((0,0) + image_shape) * area_threshold:
        return False
    
    x1,y1,x2,y2 = region
    h,w = image_shape

    if x2-x1<w*dim_threshold or y2-y1<h*dim_threshold:
        return False
    
    return True

def remove_lowest(labels_im,threshold=None):
    '''
    Removed labels from labelled imaged that have number of pixels below the threshold.
    If threshold is None, it is calculated as the mean number of pixels among the labels.
    '''
    labels_im = labels_im.copy()

    labels,counts = count_components(labels_im)

    threshold =  threshold if threshold else sum(labels)/len(labels) 

    to_remove = set()

    for i in range(len(labels)):
        if counts[i]<threshold:
            to_remove.add(labels[i])
    
    for x in range(labels_im.shape[0]):
        for y in range(labels_im.shape[1]):
            if labels_im[x,y] in to_remove:
                labels_im[x,y] = 0

    return labels_im

def count_components(labels_im):
    '''
    Count how many pixels belongs to label in a laballed image.
    Returns list containg labels and list contatining number of occurences in the image of corresponding label from labels list.
    '''
    labels,counts = np.unique(labels_im,return_counts=True)

    return labels,counts

def get_components(img):
    num_labels, labels_im = cv2.connectedComponents(img)
    labels = np.unique(remove_lowest(labels_im))
    return labels, labels_im

def cc_extraction(image):
    image = preprocess(image)
    labels, mask = get_components(image)

    cc = []
    for label in labels:
        if label==0:
            continue

        filtered_mask = filter_pixels(mask,label)
        filtered_mask = erosion(filtered_mask)
        boxes = find_contours(filtered_mask)
        
        for no,box in enumerate(boxes):
            if not postprocess(box,image.shape):
                continue
            else:
                cc.append(box)

    return cc

def divide_cells_or_characters(cells_or_characters,image_shape):
    area = lambda box: (box[3]-box[1]) * (box[2] - box[0])
    cells_or_characters_areas = [area(c) for c in cells_or_characters]

    cells, characters = [], []
    threshold = run_kmeans_areas(np.array(cells_or_characters_areas).reshape(-1,1))

    threshold = max(threshold,MIN_THRESHOLD * image_shape[0] * image_shape[1]) 

    for c in cells_or_characters:
        if area(c)>MAX_CELL_AREA*image_shape[0] * image_shape[1]:
            continue

        if area(c)>=threshold :
            cells.append(c)

        else:
            characters.append(c)

    return cells,characters

def find_background_contours(img_original,mode=cv2.RETR_LIST):
    img = cv2.erode(img_original, np.ones((3,3), np.uint8) , iterations=1)
    img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1]  

    num_labels, labels_im = cv2.connectedComponents(img)
    
    boxes = find_contours(invert_image(labels_im),mode=mode)
    cells_or_characters = []

    for no,box in enumerate(boxes):

            #Do not process boxes outside the image
            if any(box[i]>=img_original.shape[(i+1)%2] for i in range(4)):
                continue
            
            #Check if box is not an edge
            if not(((box[2]-box[0])/(box[3]-box[1]) > HORIZONTAL_EDGE and (box[3]-box[1])<CELL_DIM_THRESHOLD * img_original.shape[1]) or ((box[2]-box[0])/(box[3]-box[1]) < VERTICAL_EDGE) and (box[2]-box[0])<CELL_DIM_THRESHOLD * img_original.shape[0] ) :
                cells_or_characters.append(box)

    return cells_or_characters

def invert_image(image):
    image = image.copy()
    image[image>0] = -1
    image += 1
    image *= 255
    return image

def run_kmeans_areas(areas):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(areas)
    return np.mean(kmeans.cluster_centers_)

def non_text_removal(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    cells_or_characters = find_background_contours(image)
    _, characters = divide_cells_or_characters(cells_or_characters,image.shape)
    mask = np.zeros_like(image)

    for box in characters:
        mask[box[1]:box[3],box[0]:box[2]] = 1
    
    image *= mask

    image[image==0] = 255

    return image

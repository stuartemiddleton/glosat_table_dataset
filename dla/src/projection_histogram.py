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

import os
import cv2
import time
import numpy as np
import argparse

from scipy import ndimage
from PIL import Image
from sklearn.cluster import KMeans

try:
    from dla.src.image_utils import divide_cells_or_characters, find_background_contours, cc_extraction
    from dla.src.xml_utils import save_ICDAR_xml, load_ICDAR_xml
except:
    from image_utils import divide_cells_or_characters, find_background_contours, cc_extraction
    from xml_utils import save_ICDAR_xml, load_ICDAR_xml

BIN_WIDTH = 10
PEAK_THRESHOLD = 0.2


def run_kmeans(points,mode):
    kmeans = KMeans(n_clusters=2, random_state=0,max_iter=1).fit(points)

    if mode=="min":
        return kmeans.labels_==np.argmin(kmeans.cluster_centers_)
    elif mode=="max":
        return kmeans.labels_==np.argmax(kmeans.cluster_centers_)

def historgram_partitioning(text_regions,image_shape,axis,plot=False,median_filter=True,invert=False,clustering=True):
    #Implemented algorithm described here: http://www.dlib.org/dlib/november14/klampfl/11klampfl.html

    projection = []

    for region in text_regions:
        if axis=="x":
            projection += [i for i in range(region[0],region[2]+1)]
        elif axis=="y":
            projection += [i for i in range(region[1],region[3]+1)]

    histogram,bins = np.histogram(projection,bins=np.arange(image_shape[1]//BIN_WIDTH) * BIN_WIDTH)

    if median_filter:
        histogram = ndimage.median_filter(histogram,size=5)

    maxima = []
    minima = []

    max_ = max(histogram)

    diff = np.diff(histogram)

    edges = np.where(diff!=0)[0]

    for i in range(len(edges)-1):
        if diff[edges[i]] * diff[edges[i+1]] >= 0:
            diff[edges[i+1]] += diff[edges[i]]

        elif abs(diff[edges[i]])>PEAK_THRESHOLD * max_ or abs(diff[edges[i+1]])>PEAK_THRESHOLD * max_:
            if  diff[edges[i]]<0 and diff[edges[i+1]] > 0 :
                minima.append((edges[i] + edges[i+1])//2)

            if  diff[edges[i]]>0 and diff[edges[i+1]] < 0 :
                maxima.append((edges[i] + edges[i+1])//2)


    minima = np.array(minima)
    if np.unique(minima).size >3 and clustering:
        minima = minima[run_kmeans(histogram[minima].reshape(-1,1),mode="min")]      

    maxima = np.array(maxima)
    if np.unique(maxima).size >3 and clustering:
        maxima = maxima[run_kmeans(histogram[maxima].reshape(-1,1),mode="max")]    

    if invert:
        return (np.array(maxima) * BIN_WIDTH).tolist()

    return (np.array(minima)*BIN_WIDTH).tolist()


def table_analysis(image,cc_cols=False,cc_rows=False):
    cells_n_characters = find_background_contours(image)
    _, characters = divide_cells_or_characters(cells_n_characters,image.shape)   

    columns = historgram_partitioning(characters,image.shape,axis="x")
    rows = historgram_partitioning(characters,image.shape,axis="y")

    cc = cc_extraction(image)
    v_boundaries = [[c[0],c[1],c[0]+1,c[3]] for c in cc] + [[c[2],c[1],c[2]+1,c[3]] for c in cc]
    h_boundaries = [[c[0],c[1],c[2],c[1]+1] for c in cc] + [[c[0],c[3],c[2],c[3]+1] for c in cc]

    v_edges = historgram_partitioning(v_boundaries,image.shape,axis="x",median_filter=False,invert=True,clustering=False)
    h_edges = historgram_partitioning(h_boundaries,image.shape,axis="y",median_filter=False,invert=True,clustering=False)

    return columns + (v_edges if cc_cols else []), rows + (h_edges if cc_rows else [])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('table_path')
    parser.add_argument('save_to')
    parser.add_argument('--cc_rows',type=bool,dest="use_cc_rows",action="store_true")
    parser.add_argument('--cc_cols',type=bool,dest="use_cc_cols",action="store_true")
    args = parser.parse_args()

    image_path = args.image_path
    table_path = args.table_path
    save_to = args.save_to

    use_cc_cols = args.use_cc_cols
    use_cc_rows = args.use_cc_rows

    for image_name in os.listdir(image_path):
        if image_name.endswith(".xml"):
            continue
        if image_name.split(".")[-2] + ".xml" in os.listdir(table_path):
            tables = load_ICDAR_xml(table_path + image_name.split(".")[-2] + ".xml")
        else:
            continue

        image = cv2.imread(image_path + image_name)

        cols = []
        rows = []
        tabs = []

        for table in tables:
            table = tuple(max(pos,0) for pos in table["region"])
            tabs.append(table)

            image_cut = image[(int)(table[1]):(int)(table[3]),(int)(table[0]):(int)(table[2])]
            output = table_analysis(cv2.cvtColor(image_cut,cv2.COLOR_BGR2GRAY),cc_rows=use_cc_rows,cc_cols=use_cc_cols)

            cols.append([col + table[0] for col in output[0]]  + [table[0],table[2]])
            rows.append([row + table[1] for row in output[1]]  + [table[1],table[3]])

        save_ICDAR_xml(tabs,cols,rows,save_to + image_name.split(".")[-2] + ".xml")
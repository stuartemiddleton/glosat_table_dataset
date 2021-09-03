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

import numpy as np
import warnings
from sklearn.cluster import DBSCAN

area = lambda box: (box[2]-box[0]) * (box[3] - box[1]) if box[2]>=box[0] and box[3]>=box[1] else 0

def run_dbs_1D(cells:list, eps:int,include_outliers=True,min_samples=2) -> list: 
    '''
    Runs DBScan in 1d and returns the average values for each label.
    If outliers are detected (label = -1), each of them is appended to the average values.
    '''

    centers = np.array([cells]).reshape(-1,1)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(centers)

    examples_by_label = {label:[] for label in labels}
    mean_by_label = dict()

    for no, center in enumerate(centers):
        examples_by_label[labels[no]].append(center)

    for label in examples_by_label:
        if label!= -1 :
            mean_by_label[label] = sum(examples_by_label[label])/len(examples_by_label[label])


    return list(mean_by_label.values()) + (examples_by_label[-1] if -1 in examples_by_label.keys() and include_outliers else [])


def reconstruct_table(cells:list,table:list,eps:int) -> (list,list):
    '''
    Reconstructs the cells, given the table region using DBScan with hyperparmeter eps.

    '''
    table_width = table[2] - table[0]
    table_height = table[3] - table[1]

    #Normalise cells
    cells = [[(cell[0]-table[0])/table_width,(cell[1]-table[1])/table_height,(cell[2]-table[0])/table_width,(cell[3]-table[1])/table_height] for cell in cells]

    cells_x = [0,1]
    cells_y = [0,1]
    for cell in cells:
        cells_x += [cell[0], cell[2]]
        cells_y += [cell[1], cell[3]]   

    eps_x, eps_y = check_hyperparams(cells,eps)    

    rows = run_dbs_1D(cells_y,eps_y)
    cols = run_dbs_1D(cells_x,eps_x)

    rows = [(int)(row * table_height) + table[1] for row in rows]
    cols = [(int)(col * table_width) + table[0] for col in cols]

    return rows,cols

def reconstruct_table_coarse_and_fine(coarse_cells:list,fine_cells:list,table:list,eps:int) -> (list,list):
    '''
    Reconstructs the cells, given the table region using DBScan with hyperparmeter eps.

    '''
    table_width = table[2] - table[0]
    table_height = table[3] - table[1]

    rows = []
    cols = []

    if fine_cells!=[]:
        #Normalise cells
        fine_cells = [[(cell[0]-table[0])/table_width,(cell[1]-table[1])/table_height,(cell[2]-table[0])/table_width,(cell[3]-table[1])/table_height] for cell in fine_cells]

        cells_x = [0,1]
        cells_y = [0,1]
        for cell in fine_cells:
            cells_x += [cell[0], cell[2]]
            cells_y += [cell[1], cell[3]]

        fine_eps_x, fine_eps_y = check_hyperparams(fine_cells,eps)

        rows += run_dbs_1D(cells_y,fine_eps_y)
        cols += run_dbs_1D(cells_x,fine_eps_x)

    if coarse_cells!=[]:
        coarse_cells = [[(cell[0]-table[0])/table_width,(cell[1]-table[1])/table_height,(cell[2]-table[0])/table_width,(cell[3]-table[1])/table_height] for cell in coarse_cells]

        cells_x = [0,1]
        cells_y = [0,1]
        for cell in coarse_cells:
            cells_x += [cell[0], cell[2]]
            cells_y += [cell[1], cell[3]]  

        eps_x, eps_y = check_hyperparams(coarse_cells,eps)  

        rows += run_dbs_1D(cells_y,eps_y)
        cols += run_dbs_1D(cells_x,eps_x)

    if fine_cells!=[]:
        rows = run_dbs_1D(rows,fine_eps_y)
        cols = run_dbs_1D(cols,fine_eps_x)
    
    elif coarse_cells!=[]:
        rows = run_dbs_1D(rows,eps_y)
        cols = run_dbs_1D(cols,eps_x)

    rows = [(int)(row * table_height) + table[1] for row in rows]
    cols = [(int)(col * table_width) + table[0] for col in cols]

    return rows,cols

def check_hyperparams(cells:list,eps:int) -> (int,int):
    '''
    Check whether the eps paramter is smaller than avarega width and height of cell.
    If one of those conditions is violated, prints a warning.
    Returns adjusted hyperparameters for x and y.
    '''
    diff_x, diff_y = [], []
    for cell in cells:
        diff_x.append(cell[2] - cell[0])
        diff_y.append(cell[3] - cell[1])

    avg_diff_x = sum(diff_x)/len(diff_x)
    avg_diff_y = sum(diff_y)/len(diff_y)
    
    if avg_diff_x/2<eps:
        warnings.warn("Hyperparameter eps = {} larger than half of average cell size in x. Changing to {}".format(eps,avg_diff_x/2),RuntimeWarning)
        eps_x = avg_diff_x/2
    else:
        eps_x = eps

    if avg_diff_y/2<eps:
        warnings.warn("Hyperparameter eps = {} larger than half of average cell size in y. Changing to {}".format(eps,avg_diff_y/2),RuntimeWarning)
        eps_y =  avg_diff_y/2
    else:
        eps_y = eps

    return eps_x,eps_y

def how_much_contained(box1:list,box2:list) -> int:
    '''
    Checks how much of the first box lies inside the second one.
    '''

    area1 = area(box1)

    intersection_box = [max(box1[0],box2[0]),
                        max(box1[1],box2[1]),
                        min(box1[2],box2[2]),
                        min(box1[3],box2[3])]

    intersection_area = area(intersection_box)

    return intersection_area/(area1)

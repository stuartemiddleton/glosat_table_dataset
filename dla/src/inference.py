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

from mmdet.apis import init_detector, inference_detector
import mmcv
import dla.src.table_structure_analysis as tsa
import dla.src.xml_utils as xml_utils
from dla.src.image_utils import put_box, put_line
import argparse
import os
import cv2
import collections
import numpy as np

THRESHOLD = 0.5
CLASSES = ("table_body","cell","full_table","header","heading")

parser = argparse.ArgumentParser()
parser.add_argument('table_checkpoint',type=str)
parser.add_argument('--cell_checkpoint',type=str,dest='cell_checkpoint')
parser.add_argument('--coarse_cell_checkpoint',type=str,dest='coarse_cell_checkpoint')
parser.add_argument('--load_from',type=str,dest='load_from')
parser.add_argument('--out',type=str,dest='out')
parser.add_argument('--voc',type=bool,dest='voc')
parser.add_argument('--visual',type=bool,dest='visual')
parser.add_argument('--raw_cells',type=bool,dest='raw_cells')
parser.add_argument('--skip_headers',type=bool,dest='skip_headers')
args = parser.parse_args()

# Load model
table_checkpoint_file = args.table_checkpoint
cell_checkpoint_file = args.cell_checkpoint
coarse_cell_checkpoint_file = args.coarse_cell_checkpoint
path = args.load_from
save_to = args.out
raw_cells = args.raw_cells
image_list = os.listdir(path)
config_file = "dla/config/cascadeRCNN.py"
segment_headers = args.skip_headers

cells_by_image = collections.defaultdict(list)
coarse_cells_by_image = collections.defaultdict(list)

if cell_checkpoint_file:
    model = init_detector(config_file, cell_checkpoint_file, device='cuda:0')

    for image_name in image_list:

        if args.visual:
            image = cv2.imread(os.path.join(path,image_name))

            width, height,_ = image.shape
        
        # Run Inference
        result = inference_detector(model, os.path.join(path,image_name))

        #Process cells
        cells_by_image[image_name] = result[CLASSES.index("cell")].tolist()

    del model

if coarse_cell_checkpoint_file:
    model = init_detector(config_file, coarse_cell_checkpoint_file, device='cuda:0')


    for image_name in image_list:

        if args.visual:
            image = cv2.imread(os.path.join(path,image_name))

            width, height,_ = image.shape
        
        # Run Inference
        result = inference_detector(model, os.path.join(path,image_name))

        #Process cells
        coarse_cells_by_image[image_name] += result[CLASSES.index("cell")].tolist()

    del model

model = init_detector(config_file, table_checkpoint_file, device='cuda:0')

for image_name in image_list:
    
    if args.visual:
        image = cv2.imread(os.path.join(path,image_name))

        width, height,_ = image.shape
    else:
        width, height = 1000, 1000

    # Run Inference
    result = inference_detector(model,os.path.join(path,image_name))

    #Process table headings
    headings = []
    for box in result[CLASSES.index("heading")]:
        if box[4]>THRESHOLD :
            headings.append(box[0:4])

            if args.visual:
                put_box(image,box,(255,0,255),"heading")

    #Process table headers
    headers = []
    for box in result[CLASSES.index("header")]:
        if box[4]>THRESHOLD :
            headers.append(box[0:4])

            if args.visual:
                put_box(image,box,(255,0,0),"header")

    #Process table bodies
    tables = []
    for box in result[CLASSES.index("table_body")]:
        if box[4]>THRESHOLD :
            tables.append(box[0:4])

            if args.visual:
                put_box(image,box,(0,0,255),"table_body")

    #Process tables
    full_tables = []
    for box in result[CLASSES.index("full_table")]:
        if box[4]>THRESHOLD :
            full_tables.append(box[0:4])

            if all(tsa.how_much_contained(table,box)<0.5 for table in tables):
                tables.append(box[0:4])
                if args.visual:
                    put_box(image,box,(0,0,255),"table_body")

            if args.visual:
                put_box(image,box,(0,255,255),"Full_Table")
    
    for table in tables:
        if all(tsa.how_much_contained(table,full_table)<0.5 for full_table in full_tables):
            full_tables.append(table)


    if raw_cells:
        cells = []
        for box in cells_by_image[image_name]:
            if box[4]>THRESHOLD :
                cells.append(box[0:4])

                if args.visual:
                    put_box(image,box,(0,0,255),"cell")

        xml_utils.save_VOC_xml_from_cells(headings,headers,tables,full_tables,cells,save_to + image_name.split(".")[-2] + ".xml",width,height)
    else:
    #Process cells

        rows_by_table = []
        cols_by_table = []
        full_table_by_table = []

        for table in tables:
            cells = []
            coarse_cells = []

            found_fulltable = False
            for full_table in full_tables:
                if tsa.how_much_contained(table,full_table)>0.5:
                    full_table_by_table.append(full_table)
                    found_fulltable = True
                    break
            
            if not found_fulltable:
                full_table_by_table.append(table)


            if cell_checkpoint_file:
                for box in cells_by_image[image_name]:
                    cell = box[0:4]

                    if box[4]>THRESHOLD:
                        if tsa.how_much_contained(cell,table if not segment_headers else full_table_by_table[-1])>0.5:
                            cells.append(cell)
                            #if args.visual:
                            #    put_box(image,box,(0,0,0))
            
            if coarse_cell_checkpoint_file:
                for box in coarse_cells_by_image[image_name]:
                    cell = box[0:4]

                    if box[4]>THRESHOLD:
                        if tsa.how_much_contained(cell,table if not segment_headers else full_table_by_table[-1])>0.5:
                            coarse_cells.append(cell)
                            #if args.visual:
                            #    put_box(image,box,(0,0,0))
            
            if cells != [] or coarse_cells!=[]:
                if coarse_cell_checkpoint_file:
                    rows, cols = tsa.reconstruct_table_coarse_and_fine(coarse_cells,cells,table if segment_headers else full_table_by_table[-1],eps=0.02)
                else:
                    rows, cols = tsa.reconstruct_table(cells,table if segment_headers else full_table_by_table[-1],eps=0.02)
            else:
                rows,cols = [],[]

            if args.visual:
                for row in rows:
                    put_line(image,((int)(table[0]),(int)(row)),((int)(table[2]),(int)(row)),colour=(0,255,0))

                for col in cols:
                    put_line(image,((int)(col),(int)(table[1])),((int)(col),(int)(table[3])),colour=(0,255,0))
            
            rows_by_table.append(rows)
            cols_by_table.append(cols)

        if not(args.voc):
            xml_utils.save_ICDAR_xml(full_table_by_table,cols_by_table,rows_by_table,save_to + image_name.split(".")[-2] + ".xml")
        else:
            xml_utils.save_VOC_xml(headings,headers,tables,full_tables,cols_by_table,rows_by_table,save_to + image_name.split(".")[-2] + ".xml",width,height)

    if args.visual:
        cv2.imwrite(save_to + "out_%s"%(image_name),image)

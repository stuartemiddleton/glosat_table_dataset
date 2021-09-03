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

THRESHOLD = 0.5
CLASSES = ("bordered_table","cell","borderless_table")

parser = argparse.ArgumentParser()
parser.add_argument('table_checkpoint',type=str)
parser.add_argument('--load_from',type=str,dest='load_from')
parser.add_argument('--use_cells',type=bool,dest='use_cells',default=False)
parser.add_argument('--out',type=str,dest='out')
parser.add_argument('--voc',type=bool,dest='voc')
parser.add_argument('--visual',type=bool,dest='visual')
args = parser.parse_args()

# Load model
table_checkpoint_file = args.table_checkpoint
path = args.load_from
save_to = args.out
image_list = os.listdir(path)
config_file = "dla/config/cascadeRCNN.py"


model = init_detector(config_file, table_checkpoint_file, device='cuda:0')

for image_name in image_list:
    
    if args.visual:
        image = cv2.imread(os.path.join(path,image_name))

        width, height,_ = image.shape

    # Run Inference
    result = inference_detector(model,os.path.join(path,image_name))

    #Process tables
    bordered_tables = []
    for box in result[CLASSES.index("bordered_table")]:
        if box[4]>THRESHOLD :
            bordered_tables.append(box[0:4].tolist())

            if args.visual:
                put_box(image,box,(0,255,255),"Full_Table")

    borderless_tables = []
    for box in result[CLASSES.index("borderless_table")]:
        if box[4]>THRESHOLD :
            borderless_tables.append(box[0:4].tolist())

            if args.visual:
                put_box(image,box,(0,255,255),"Full_Table")

    full_tables = []

    #If borderless table and bordered table are both detected, save only the one with greater area
    for bordered_table in bordered_tables:
        borderless_match = None
        for borderless_table in borderless_tables:
            if tsa.IoU(bordered_table,borderless_table)>0.5:
                if tsa.area(borderless_table)>tsa.area(bordered_table):
                    borderless_match = borderless_table

        if borderless_match:
            full_tables.append(borderless_match)
        else:
            full_tables.append(bordered_table)

    for borderless_table in borderless_tables:
        bordered_match = None
        for bordered_table in bordered_tables:
            if tsa.IoU(borderless_table,bordered_table)>0.5:
                if tsa.area(borderless_table)<tsa.area(bordered_table):
                    bordered_match = bordered_table

        if not(bordered_match) and borderless_table not in full_tables:
            full_tables.append(borderless_table)


    # Run Inference
    result = inference_detector(model, os.path.join(path,image_name))

    
    cells = []
    rows_by_table = []
    cols_by_table = []

    if args.use_cells:
        #Process cells
        for box in result[CLASSES.index("cell")]:
            if box[4]>THRESHOLD :
                cells.append(box[0:4].tolist())

                if args.visual:
                    put_box(image,box,(0,255,255),"Cell")
        
        for table in full_tables:
            cells = []
            for box in cells:
                cell = box[0:4].tolist()

                if box[4]>THRESHOLD:
                        if tsa.how_much_contained(cell,table)>0.5:
                            cells.append(cell)
                            if args.visual:
                                put_box(image,box,(0,255,0))
            
            if cells != []:
                rows, cols = tsa.reconstruct_table(cells,table,eps=0.02)
            else:
                rows,cols = [], []

            if args.visual:
                for row in rows:
                    put_line(image,((int)(table[0]),(int)(row)),((int)(table[2]),(int)(row)),colour=(0,255,0))

                for col in cols:
                    put_line(image,((int)(col),(int)(table[1])),((int)(col),(int)(table[3])),colour=(0,255,0))
            
            rows_by_table.append(rows)
            cols_by_table.append(cols)
    else:
        rows_by_table = [[] for _ in full_tables]
        cols_by_table = [[] for _ in full_tables]

    if not(args.voc):
        xml_utils.save_ICDAR_xml(full_tables,cols_by_table,rows_by_table,save_to + image_name.split(".")[-2] + ".xml")
    else:
        xml_utils.save_VOC_xml([],[],[],full_tables,cols_by_table,rows_by_table,save_to + image_name.split(".")[-2] + ".xml",width,height)

    if args.visual:
        cv2.imwrite(save_to + "out_%s"%(image_name),image)

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

try:
    import xml_utils as xml_utils
except:
    import dla.src.xml_utils as xml_utils

import argparse
import os

area = lambda box: (box[2]-box[0]) * (box[3] - box[1]) if box[2]>=box[0] and box[3]>=box[1] else 0

def IoU(box1,box2):

    area1 = area(box1)
    area2 = area(box2)

    intersection_box = [max(box1[0],box2[0]),
                        max(box1[1],box2[1]),
                        min(box1[2],box2[2]),
                        min(box1[3],box2[3])]

    intersection_area = area(intersection_box)

    return intersection_area/(area1 + area2 - intersection_area)

def calculate_scores(output,gt,IoU_threshold):
    scores = {"true_pos":0,"false_pos":0,"false_neg":0,"mean_IoU":0}
    duplicated_positives = 0
    total_IoU = 0

    for gt_object in gt:
        object_detected = False
        for output_object in output:
            if IoU(gt_object,output_object)>IoU_threshold:
                total_IoU += IoU(gt_object,output_object)
                if object_detected:
                    duplicated_positives += 1
                else:
                    object_detected = True
                    scores["true_pos"] += 1

        if not object_detected:
            scores["false_neg"] += 1
        
    scores["mean_IoU"] = total_IoU/(scores["true_pos"] + duplicated_positives) if total_IoU else None
    scores["false_pos"] = max(len(output) - scores["true_pos"] - duplicated_positives,0)

    return scores

parser = argparse.ArgumentParser()
parser.add_argument('gt',type=str)
parser.add_argument('output',type=str)
parser.add_argument('--IoU_threshold',type=float)
args = parser.parse_args()

IoU_threshold = args.IoU_threshold if args.IoU_threshold else 0.5

gt_path = args.gt if args.gt.endswith("/") else args.gt + "/"
output_path = args.output if args.output.endswith("/") else args.output + "/"

table_scores = {"true_pos":0,"false_pos":0,"false_neg":0,"mean_IoU":0}
cell_scores = {"true_pos":0,"false_pos":0,"false_neg":0,"mean_IoU":0}
table_no = 0
cell_no = 0

for file in os.listdir(output_path):
    if file in os.listdir(gt_path):
        output = xml_utils.load_ICDAR_xml(output_path + file)
        gt = xml_utils.load_ICDAR_xml(gt_path + file)

        output_tables = []
        output_cells = []
        gt_tables = []
        gt_cells = []

        for table in output:
            output_tables.append(table["region"])
            output_cells += table["cells"]

        for table in gt:
            gt_tables.append(table["region"])
            gt_cells += table["cells"]

        new_table_scores = calculate_scores(output_tables,gt_tables,IoU_threshold )
        new_cell_scores =calculate_scores(output_cells,gt_cells,IoU_threshold )

        for score_type in new_table_scores:
            table_scores[score_type] += new_table_scores[score_type] if new_table_scores[score_type] else 0
            cell_scores[score_type] += new_cell_scores[score_type] if new_cell_scores[score_type] else 0
        
        if new_table_scores["mean_IoU"]:
            table_no += 1
        if new_cell_scores["mean_IoU"]:
            cell_no += 1

precision_tables = table_scores["true_pos"] / (table_scores["true_pos"] + table_scores["false_pos"]) if (table_scores["true_pos"] + table_scores["false_pos"])!=0 else None
precision_cells = cell_scores["true_pos"] / (cell_scores["true_pos"] + cell_scores["false_pos"]) if (cell_scores["true_pos"] + cell_scores["false_pos"]) !=0 else None
recall_tables = table_scores["true_pos"] / (table_scores["true_pos"] + table_scores["false_neg"]) if (table_scores["true_pos"] + table_scores["false_neg"]) !=0 else None
recall_cells = cell_scores["true_pos"] / (cell_scores["true_pos"] + cell_scores["false_neg"]) if (cell_scores["true_pos"] + cell_scores["false_neg"]) !=0 else None

print("Table:","Precision:",precision_tables,"Recall:",recall_tables,"Mean IoU:",table_scores["mean_IoU"]/table_no if table_no>0 else None)
print("Cell:","Precision:",precision_cells,"Recall:",recall_cells,"Mean IoU:",cell_scores["mean_IoU"]/cell_no if cell_no>0 else None)
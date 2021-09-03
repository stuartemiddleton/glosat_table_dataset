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

    if area1==0 and area2==0:
        return 0

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
row_scores = {"true_pos":0,"false_pos":0,"false_neg":0,"mean_IoU":0}
col_scores = {"true_pos":0,"false_pos":0,"false_neg":0,"mean_IoU":0}
table_no = 0
row_no = 0
col_no = 0 

PROXIMITY_FACTOR = 5


for file in os.listdir(output_path):
    if file in os.listdir(gt_path):
        output = xml_utils.load_ICDAR_xml_lines(output_path + file)
        gt = xml_utils.load_ICDAR_xml_lines(gt_path + file)

        output_tables = []

        gt_tables = []

        total_rows = 0
        total_cols = 0
        foundtable_rows = 0
        foundtable_cols = 0

        for table in output:
            output_tables.append(table["region"])
            total_rows += len(table["rows"][1:])
            total_cols += len(table["cols"][1:])

        for table in gt:
            gt_tables.append(table["region"])

        new_table_scores = calculate_scores(output_tables,gt_tables,IoU_threshold)

        for score_type in new_table_scores:
            table_scores[score_type] += new_table_scores[score_type] if new_table_scores[score_type] else 0
        
        if new_table_scores["mean_IoU"]:
            table_no += 1

        for gt_table in gt:
            match_found = False
            for out_table in output:
                if IoU(out_table["region"],gt_table["region"])>IoU_threshold:
                    
                    match_found = True

                    gt_table["rows"] = gt_table["rows"][1:]
                    out_table["rows"] = out_table["rows"][1:]
                    gt_table["cols"] = gt_table["cols"][1:]
                    out_table["cols"] = out_table["cols"][1:]

                    foundtable_rows += len(out_table["rows"])
                    foundtable_cols += len(out_table["cols"])

                    row_proximity = (gt_table["region"][3] - gt_table["region"][1]) /(len(gt_table["rows"])+2) / PROXIMITY_FACTOR
                    col_proximity = (gt_table["region"][2] - gt_table["region"][0]) / (len(gt_table["cols"])+2)/ PROXIMITY_FACTOR
                    
                    gt_rows = [[gt_table["region"][0],row - row_proximity,gt_table["region"][2],row + row_proximity] for row in gt_table["rows"]]
                    gt_cols = [[col - col_proximity,gt_table["region"][1],col + col_proximity,gt_table["region"][3]] for col in gt_table["cols"]]

                    out_rows = [[gt_table["region"][0],row - row_proximity,gt_table["region"][2],row + row_proximity] for row in out_table["rows"]]
                    out_cols = [[col - col_proximity,gt_table["region"][1],col + col_proximity,gt_table["region"][3]] for col in out_table["cols"]]

                    new_row_scores = calculate_scores(out_rows,gt_rows,IoU_threshold)
                    new_col_scores = calculate_scores(out_cols,gt_cols,IoU_threshold)

                    for score_type in new_row_scores:
                        row_scores[score_type] += new_row_scores[score_type] if new_row_scores[score_type] else 0

                    for score_type in new_col_scores:
                        col_scores[score_type] += new_col_scores[score_type] if new_col_scores[score_type] else 0
                    
                    if new_row_scores["mean_IoU"]:
                        row_no += 1

                    if new_col_scores["mean_IoU"]:
                        col_no += 1
                    break
                
            if not(match_found):
                row_scores["false_neg"] += len(gt_table["rows"][1:])
                col_scores["false_neg"] += len(gt_table["cols"][1:])
        
        row_scores["false_pos"] += total_rows - foundtable_rows
        col_scores["false_pos"] += total_cols - foundtable_cols

                

precision_tables = table_scores["true_pos"] / (table_scores["true_pos"] + table_scores["false_pos"]) if (table_scores["true_pos"] + table_scores["false_pos"])!=0 else None
precision_rows = row_scores["true_pos"] / (row_scores["true_pos"] + row_scores["false_pos"]) if (row_scores["true_pos"] + row_scores["false_pos"]) !=0 else None
precision_cols = col_scores["true_pos"] / (col_scores["true_pos"] + col_scores["false_pos"]) if (col_scores["true_pos"] + col_scores["false_pos"]) !=0 else None
recall_tables = table_scores["true_pos"] / (table_scores["true_pos"] + table_scores["false_neg"]) if (table_scores["true_pos"] + table_scores["false_neg"]) !=0 else None
recall_rows = row_scores["true_pos"] / (row_scores["true_pos"] + row_scores["false_neg"]) if (row_scores["true_pos"] + row_scores["false_neg"]) !=0 else None
recall_cols = col_scores["true_pos"] / (col_scores["true_pos"] + col_scores["false_neg"]) if (col_scores["true_pos"] + col_scores["false_neg"]) !=0 else None

print("Table:","Precision:",precision_tables,"Recall:",recall_tables,"Mean IoU:",table_scores["mean_IoU"]/table_no if table_no else None)
print("row:","Precision:",precision_rows,"Recall:",recall_rows,"Mean IoU:",row_scores["mean_IoU"]/row_no if row_no else None)
print("col:","Precision:",precision_cols,"Recall:",recall_cols,"Mean IoU:",col_scores["mean_IoU"]/col_no if col_no else None)
print("row F1",2*precision_rows*recall_rows/(precision_rows+recall_rows),"col F1",2*precision_cols*recall_cols/(precision_cols+recall_cols))
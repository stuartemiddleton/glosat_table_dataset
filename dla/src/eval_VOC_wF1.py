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

CLASSES = ("heading","header","full_table","table_body","cell")

try:
    import xml_utils as xml_utils
except:
    import dla.src.xml_utils as xml_utils

import argparse
import os
import collections

area = lambda box: (box[2]-box[0]) * (box[3] - box[1]) if box[2]>=box[0] and box[3]>=box[1] else 0

def IoU(box1,box2):

    box1 = [box1[0],box1[1],(box1[2] + 1) if box1[0]==box1[2] else box1[2],(box1[3] + 1) if box1[1]==box1[3] else box1[3]]

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
args = parser.parse_args()

IoU_thresholds = {0.6,0.7,0.8,0.9}

F1scores = collections.defaultdict(lambda:0)

for IoU_threshold in IoU_thresholds:

    gt_path = args.gt if args.gt.endswith("/") else args.gt + "/"
    output_path = args.output if args.output.endswith("/") else args.output + "/"

    scores = {c : {"true_pos":0,"false_pos":0,"false_neg":0,"mean_IoU":0} for c in CLASSES}
    no = {c : 0 for c in CLASSES}


    for file in os.listdir(output_path):
        if file in os.listdir(gt_path):
            output = xml_utils.load_VOC_xml(output_path + file)
            gt = xml_utils.load_VOC_xml(gt_path + file)

            output_objects = {c:[] for c in CLASSES}
            gt_objects = {c:[] for c in CLASSES}
    
            for obj in output:
                    output_objects[obj["name"]].append(obj["bbox"])
            
            for obj in gt:
                    gt_objects[obj["name"]].append(obj["bbox"])

            for class_ in CLASSES:
                new_scores = calculate_scores(output_objects[class_],gt_objects[class_],IoU_threshold)

                for score_type in new_scores:
                    scores[class_][score_type] += new_scores[score_type] if new_scores[score_type] else 0
            
                if new_scores["mean_IoU"]:
                    no[class_] += 1

                #if class_ == "full_table":
                #    if new_scores["false_neg"]>1 or new_scores["false_pos"]>1:
                #        print(file)

    #print("\n IoU threshold:",IoU_threshold,"\n")
    for class_ in CLASSES:
        precision = scores[class_]["true_pos"] / (scores[class_]["true_pos"] + scores[class_]["false_pos"]) if (scores[class_]["true_pos"] + scores[class_]["false_pos"])!=0 else None

        recall = scores[class_]["true_pos"] / (scores[class_]["true_pos"] + scores[class_]["false_neg"]) if (scores[class_]["true_pos"] + scores[class_]["false_neg"]) !=0 else None

        
        #print(class_,"Precision:",precision,"Recall:",recall,"Mean IoU:",scores[class_]["mean_IoU"]/no[class_] if no[class_]>0 else None)

        F1scores[class_] += 2 * precision * recall / (precision + recall) * IoU_threshold if precision!=None and recall!=None else 0

print("\n\n")
for class_ in CLASSES:
    print(class_,"Avg.W.F1",F1scores[class_]/sum(IoU_thresholds))

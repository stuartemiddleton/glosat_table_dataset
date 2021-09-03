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

import xml.etree.ElementTree as ET
import time

def save_ICDAR_xml(tables:list,cols:list,rows:list,filename:str):
    '''
    Saves the output in ICDAR-style xml.
    - tables_with_cells - dictonaries in form {k:v} where k is a tuple representing table bounding box and v is a list of boxes representing table cells
    - filename - string containing filename under which the xml should be saved
    '''

    root = ET.Element('document')

    for no,table in enumerate(tables):
        
        tab = xml_add_table(root,table)

        cells, row_by_cell, col_by_cell = cells_from_lines(rows[no],cols[no])
        
        for no,cell in enumerate(cells):
            xml_add_cell(tab,cell,row_by_cell[no],col_by_cell[no],rowSpan=1,colSpan=1)

    tree = ET.ElementTree(root)
    tree.write(filename)    


def box_to_item(box):
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = (int)(x0), (int)(y0), (int)(x1), (int)(y1)

    result = (str)(x0) + "," + (str)(y0)
    result += " " + (str)(x0) + "," + (str)(y1)
    result += " " + (str)(x1) + "," + (str)(y0)
    result += " " + (str)(x1) + "," + (str)(y1)

    return result

def item_to_box(item):

    for info in item:
        if info.tag.endswith("Coords"):
            coords = info
    try:
        points = coords.attrib["points"].split(" ")
    except:
        points = coords.attrib["point"].split(" ")
    
    points = [((float)(p.split(",")[0]),(float)(p.split(",")[1])) for p in points]

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    
    return min(x),min(y),max(x),max(y)

def cells_from_lines(rows,cols):

    junctions = []

    for row in sorted(rows):
        junctions_in_row = []
        for col in sorted(cols):
            junctions_in_row.append([col,row])

        junctions.append(junctions_in_row)

    cells = []
    row_by_cell = []
    col_by_cell = []

    for i in range(len(junctions)-1):
        for j in range(len(junctions[i])-1):
            cells += [junctions[i][j] + junctions[i+1][j+1]]
            row_by_cell.append(i)
            col_by_cell.append(j)

    return cells, row_by_cell, col_by_cell

def cell_to_lines(cells):
    rows_by_pixels = dict()
    cols_by_pixels = dict()

    for cell in cells:
        if cell.attrib["start-row"] in rows_by_pixels:
            rows_by_pixels[cell.attrib["start-row"]].append(item_to_box(cell)[1])
        else:
            rows_by_pixels[cell.attrib["start-row"]] = [item_to_box(cell)[1]]

        if cell.attrib["start-col"] in cols_by_pixels:
            cols_by_pixels[cell.attrib["start-col"]].append(item_to_box(cell)[0])
        else:
            cols_by_pixels[cell.attrib["start-col"]] = [item_to_box(cell)[0]]

    return [sum(rows_by_pixels[row])/len(rows_by_pixels[row])for row in rows_by_pixels],[sum(cols_by_pixels[col])/len(cols_by_pixels[col])for col in cols_by_pixels]


def xml_add_table(root,table):
    tab = ET.SubElement(root,"table")
    tab.attrib["id"] = "Table_" + (str)(time.time()*1000000)

    coords = ET.SubElement(tab,"Coords")
    coords.attrib["points"] = box_to_item(table) 

    return tab

def xml_add_cell(table,cell,row,col,rowSpan=1,colSpan=1):
    c = ET.SubElement(table,"cell")
    c_coords = ET.SubElement(c,"Coords")
    c_coords.attrib["points"] = box_to_item(cell)
    c.attrib["id"] = "TableCell_" + (str)(time.time()*1000000)
    c.attrib["start-row"] = (str)(row)
    c.attrib["start-col"] = (str)(col)
    c.attrib["end-row"] = (str)(row + rowSpan)
    c.attrib["end-col"] = (str)(col + colSpan)

def get_ICDAR_summary(filename):
    root = ET.parse(filename).getroot()

    doc_type = root.attrib["type"]

    tables = []

    for element in root:
        if element.tag == "table":
            table_type = element.attrib["type"]
            table_region = item_to_box(element)
            cells = []
            header_cells = 0
            for child in element:  
                if child.tag.endswith("cell"):
                    if "header" in child.attrib:
                        header_cells += 1    
                    cells.append(item_to_box(child))
                
            tables.append({"cells":cells,"region":table_region,"type":table_type.replace("_","-"),"header_no":header_cells})

    return tables, doc_type

def load_ICDAR_xml(filename):
    root = ET.parse(filename).getroot()

    tables = []

    for element in root:
        if element.tag == "table":
            table_region = item_to_box(element)
            cells = []
            for child in element:  
                if child.tag.endswith("cell"):    
                    cells.append(item_to_box(child))
                
            tables.append({"cells":cells,"region":table_region})

    return tables

def load_ICDAR_xml_lines(filename):
    root = ET.parse(filename).getroot()

    tables = []

    for element in root:
        if element.tag == "table":
            table_region = item_to_box(element)
            cells = []
            for child in element:  
                if child.tag.endswith("cell"):    
                    cells.append(child)
                
            rows,cols = cell_to_lines(cells)
            tables.append({"rows":rows,"cols":cols,"region":table_region})

    return tables

def load_VOC_xml(filename):
    root = ET.parse(filename).getroot()

    objects = []

    for element in root:
        if element.tag == "object":
            for child in element:
                if child.tag == "name":
                    name = child.text

                if child.tag == "bndbox":
                    for dim in child:
                        if dim.tag == "xmin":
                            xmin = (int)(dim.text)

                        if dim.tag == "xmax":
                            xmax = (int)(dim.text)
                        
                        if dim.tag == "ymin":
                            ymin = (int)(dim.text)

                        if dim.tag == "ymax":
                            ymax = (int)(dim.text)

            objects.append({"name":name,"bbox":[xmin,ymin,xmax,ymax]})
            
    return objects

def add_VOC_object(bbox,name_text,root):
    obj = ET.SubElement(root,"object")
    name = ET.SubElement(obj,"name")
    pose = ET.SubElement(obj,"pose")
    truncated = ET.SubElement(obj,"truncated")
    difficult = ET.SubElement(obj,"difficult")

    name.text = name_text
    pose.text = "Unspecified"
    truncated.text = "0"
    difficult.text = "0"
    bndbox = ET.SubElement(obj,"bndbox")

    xmin = ET.SubElement(bndbox,"xmin")
    ymin = ET.SubElement(bndbox,"ymin")
    xmax = ET.SubElement(bndbox,"xmax")
    ymax = ET.SubElement(bndbox,"ymax")
    
    xmin.text = (str)((int)(bbox[0]))
    ymin.text = (str)((int)(bbox[1]))
    xmax.text = (str)((int)(bbox[2]))
    ymax.text = (str)((int)(bbox[3]))


def add_VOC_intro(root,width_,height_,filename):
    folder = ET.SubElement(root,"folder")
    folder.text = "JPEGImages"
    filename = ET.SubElement(root,"filename")
    filename.text = filename
    path = ET.SubElement(root,"path")
    path.text = "VOC/"
    source = ET.SubElement(root,"source")
    database = ET.SubElement(source,"database")
    database.text = "GloSAT"
    size = ET.SubElement(root,"size")
    width = ET.SubElement(size,"width")
    height = ET.SubElement(size,"height")
    depth = ET.SubElement(size,"depth")

    width.text = (str)(width_)
    height.text = (str)(height_)
    depth.text = "3"

    segmented = ET.SubElement(root,"segmented")
    segmented.text = "0"


def save_VOC_xml(headings:list,headers:list,bodies:list,full_tables:list,cols:list,rows:list,filename:str,width:int,height:int):
    root = ET.Element('annotation')
    add_VOC_intro(root,width,height,filename)    

    for heading in headings:
        add_VOC_object(heading,"heading",root)

    for header in headers:
        add_VOC_object(header,"header",root)

    for no,body in enumerate(full_tables):
        add_VOC_object(body,"full_table",root)

    for no,body in enumerate(bodies):
        add_VOC_object(body,"table_body",root)

        cells, _, _ = cells_from_lines(rows[no],cols[no])
            
        for cell in cells:
            add_VOC_object(cell,"cell",root)

    tree = ET.ElementTree(root)
    tree.write(filename)    

def save_VOC_xml_from_cells(headings:list,headers:list,bodies:list,full_tables:list,cells:list,filename:str,width:int,height:int):
    root = ET.Element('annotation')
    add_VOC_intro(root,width,height,filename)    

    for heading in headings:
        add_VOC_object(heading,"heading",root)

    for header in headers:
        add_VOC_object(header,"header",root)

    for no,body in enumerate(full_tables):
        add_VOC_object(body,"full_table",root)

    for no,body in enumerate(bodies):
        add_VOC_object(body,"table_body",root)

            
    for cell in cells:
        add_VOC_object(cell,"cell",root)

    tree = ET.ElementTree(root)
    tree.write(filename)    
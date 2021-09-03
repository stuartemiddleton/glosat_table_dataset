import os
import collections
import xml_utils as xml_utils
import pandas as pd

summaries = []
dataset_root = '/media/DATA/GloSAT_dataset_fine'
files = set()

for source in os.listdir( dataset_root ) :

    cells = 0
    tables = 0
    images = 0
    header_cells = 0
    headers = 0
    table_per_type = collections.defaultdict(lambda:0)
    docs_per_type = collections.defaultdict(lambda:0)

    if "labels" not in os.listdir(os.path.join(dataset_root,source)):
        continue

    for file in os.listdir(os.path.join(dataset_root,source,"labels")):
        if file.endswith(".xml"):

            if file.strip(".xml") + ".jpg" not in os.listdir(os.path.join(dataset_root,source)) and file.strip(".xml") + ".JPG" not in os.listdir(os.path.join(dataset_root,source)):
                continue
            
            images += 1

            icdar_parsed, doc_type = xml_utils.get_ICDAR_summary( os.path.join(dataset_root,source,"labels",file) )
            
            docs_per_type[doc_type] += 1

            for entry in icdar_parsed:
                tables += 1
                cells += len(entry["cells"])
                header_cells += entry["header_no"]
                headers += entry["header_no"]!=0
                table_per_type[entry["type"]] += 1 

    summary = pd.DataFrame([[source,images,cells,tables,header_cells,headers]],columns=["source","images","cells","tables","header_cells","headers"])

    for table_type in pd.unique(list(table_per_type.keys())):
        summary[table_type + " tables"] = table_per_type[table_type]

    for doc_type in pd.unique(list(docs_per_type.keys())):
        summary[doc_type + " docs"] = docs_per_type[doc_type]

    summaries.append(summary)

summaries = pd.concat(summaries).fillna(0)


print("\n\n",summaries)

#summaries.to_csv("dataset_summary_fine")

print("\n\n",summaries.sum(axis=0))
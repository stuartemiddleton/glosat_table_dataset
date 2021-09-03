import os
import xml_utils as xml_utils

#
# note: this is not needed for GloSAT dataset as it is shipped with VOC, ICDAR and Transkribus xml labels
#

voc_dir = '/home/juliuszziomek/Documents/Python/GloSAT_fine_test/VOC2007_test_noheader/Annotations/'
icdar_dir = '/home/juliuszziomek/Documents/Python/GloSAT_fine_test/VOC2007/ICDAR/'

if not os.path.exists( icdar_dir ) :
	raise Exception( 'model dir not found : ' + repr(icdar_dir) )
if not os.path.exists( voc_dir ) :
	os.mkdir( voc_dir )

# see inference.py code for the basis of this code
CLASSES = ("table_body","cell","full_table","header","heading")
area = lambda box: (box[2]-box[0]) * (box[3] - box[1]) if box[2]>=box[0] and box[3]>=box[1] else 0

for file in os.listdir( icdar_dir ) :
    if file.endswith(".xml"):

        icdar_parsed = xml_utils.load_ICDAR_xml( icdar_dir + '/' + file )

        tables = []
        cells = []

        for entry in icdar_parsed:
            tables.append(entry["region"])
            cells += entry["cells"]
            

        xml_utils.save_VOC_xml_from_cells(headings=[],headers=[],bodies=tables,full_tables=tables,cells=cells,filename=os.path.join(voc_dir,file),width=1000,height=1000)

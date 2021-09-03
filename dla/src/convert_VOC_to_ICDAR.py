import os, random, codecs
import table_structure_analysis as tsa
import xml_utils as xml_utils
from image_utils import put_box, put_line
import argparse

#
# note: this is not needed for GloSAT dataset as it is shipped with VOC, ICDAR and Transkribus xml labels
#
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('voc',type=str)
	parser.add_argument('icdar',type=str)
	args = parser.parse_args()

	voc_dir = args.voc
	icdar_dir = args.icdar

	if not os.path.exists( voc_dir ) :
		raise Exception( 'model dir not found : ' + repr(voc_dir) )
	if not os.path.exists( icdar_dir ) :
		os.mkdir( icdar_dir )

	# see inference.py code for the basis of this code
	CLASSES = ("table_body","cell","full_table","header","heading")
	area = lambda box: (box[2]-box[0]) * (box[3] - box[1]) if box[2]>=box[0] and box[3]>=box[1] else 0

	for file in os.listdir( voc_dir ) :
		if not(file.endswith(".xml")):
    			continue

		# load VOC xml
		# voc_parsed  = [ {"name":name,"bbox":[xmin,ymin,xmax,ymax]}, ... ]
		voc_parsed = xml_utils.load_VOC_xml( voc_dir + '/' + file )

		headings = []
		headers = []
		tables = []
		full_tables = []
		predicted_cells = []

		for entry in voc_parsed :
			if entry['name'] == 'header' :
				headers.append( entry['bbox'] )
			elif entry['name'] == 'table_body' :
				tables.append( entry['bbox'] )
			elif entry['name'] == 'heading' :
				headings.append( entry['bbox'] )
			elif entry['name'] == 'full_table' :
				full_tables.append( entry['bbox'] )
			elif entry['name'] == 'cell' :
				predicted_cells.append( entry['bbox'] )

		for table in tables:
				if all(tsa.how_much_contained(table,full_table)<0.5 for full_table in full_tables):
						full_tables.append(table)

		rows_by_table = []
		cols_by_table = []

		for table in full_tables:
			cells = []
			for cell in predicted_cells :

				# assign a cell to a table if cell area > 0 AND cell overlap with table is > 50%
				if area(cell) > 0 :
					if tsa.how_much_contained(cell,table)>0.5:
						cells.append(cell)

			if cells != []:
				rows, cols = tsa.reconstruct_table(cells,table,eps=0.02)
			else:
				rows,cols = [],[]

			rows_by_table.append(rows)
			cols_by_table.append(cols)

		xml_utils.save_ICDAR_xml( full_tables, cols_by_table, rows_by_table, icdar_dir +'/' + file )


import os, random, codecs

# source dataset
#    <dataset_dir>/<source>/VOC_annotations ==> VOC xml labels
#    <dataset_dir>/<source>/labels ==> ICDAR xml labels
#    <dataset_dir>/<source>/page ==> Transkribus xml labels (original source)
dataset_dir = '/data/glosat_table_dataset/datasets/GloSAT_dataset_coarse'
#dataset_dir = '/data/glosat_table_dataset/datasets/GloSAT_dataset_fine'


# resulting model dir structure
#    <model_dir>_train/VOC2007/JPEGImages
#    <model_dir>_train/VOC2007/Annotations
#    <model_dir>_train/VOC2007/ICDAR
#    <model_dir>_train/VOC2007/Transkribus
#    <model_dir>_test/VOC2007/JPEGImages
#    <model_dir>_test/VOC2007/Annotations
#    <model_dir>_test/VOC2007/ICDAR
#    <model_dir>_test/VOC2007/Transkribus
model_dir = '/data/glosat_table_dataset/dla_models/model_table_det_full_table'
#model_dir = '/data/glosat_table_dataset/dla_models/model_table_det_enhanced'
#model_dir = '/data/glosat_table_dataset/dla_models/model_table_struct_coarse'
#model_dir = '/data/glosat_table_dataset/dla_models/model_table_struct_fine'

# images will be copied as 66% train 33% test
train_dir = model_dir + '_train'
holdout_dir = model_dir + '_test'

data_sources = {
	"20cr_DWR_MO/": dataset_dir + "/20cr_DWR_MO",
	"20cr_DWR_NOAA/": dataset_dir +"/20cr_DWR_NOAA",
	"20cr_Kubota/": dataset_dir + "/20cr_Kubota",
	"20cr_Natal_Witnes/": dataset_dir + "/20cr_Natal_Witnes",
	"DWR/": dataset_dir + "/DWR",
	"WR_10_years/": dataset_dir + "/WR_10_years",
	"WesTech/": dataset_dir + "/WesTech_Rodgers",
	"WR_Devon_Extern/": dataset_dir +"/WR_Devon_Extern",
	"Ben_Nevis/": dataset_dir +"/Ben_Nevis"}

if not os.path.exists( train_dir ) :
	os.mkdir( train_dir )
if not os.path.exists( train_dir + "/VOC2007" ) :
	os.mkdir( train_dir + "/VOC2007" )
if not os.path.exists( train_dir + "/VOC2007/Annotations" ) :
	os.mkdir( train_dir + "/VOC2007/Annotations" )
if not os.path.exists( train_dir + "/VOC2007/ICDAR" ) :
	os.mkdir( train_dir + "/VOC2007/ICDAR" )
if not os.path.exists( train_dir + "/VOC2007/Transkribus" ) :
	os.mkdir( train_dir + "/VOC2007/Transkribus" )
if not os.path.exists( train_dir + "/VOC2007/JPEGImages" ) :
	os.mkdir( train_dir + "/VOC2007/JPEGImages" )
if not os.path.exists( train_dir + "/VOC2007/ImageSets" ) :
	os.mkdir( train_dir + "/VOC2007/ImageSets" )

if not os.path.exists( holdout_dir ) :
	os.mkdir( holdout_dir )
if not os.path.exists( holdout_dir + "/VOC2007" ) :
	os.mkdir( holdout_dir + "/VOC2007" )
if not os.path.exists( holdout_dir + "/VOC2007/Annotations" ) :
	os.mkdir( holdout_dir + "/VOC2007/Annotations" )
if not os.path.exists( holdout_dir + "/VOC2007/ICDAR" ) :
	os.mkdir( holdout_dir + "/VOC2007/ICDAR" )
if not os.path.exists( holdout_dir + "/VOC2007/Transkribus" ) :
	os.mkdir( holdout_dir + "/VOC2007/Transkribus" )
if not os.path.exists( holdout_dir + "/VOC2007/JPEGImages" ) :
	os.mkdir( holdout_dir + "/VOC2007/JPEGImages" )
if not os.path.exists( holdout_dir + "/VOC2007/ImageSets" ) :
	os.mkdir( holdout_dir + "/VOC2007/ImageSets" )

list_test = []
list_train  = []

for data_source in data_sources.values():
	available_files = []
	for file in os.listdir(data_source):
		if file.endswith(".jpg") and file.strip(".jpg") + ".xml" in os.listdir(data_source + "/VOC_annotations/"):
			available_files.append(file)

	for _ in range(int(len(available_files) * 0.75)):
		file = random.choice(available_files)
		os.system("cp {} {}".format(data_source + '/' + file,train_dir + "/VOC2007/JPEGImages/" + file))		
		os.system("cp {} {}".format(data_source + "/VOC_annotations/" + file.strip(".jpg") + ".xml",train_dir + "/VOC2007/Annotations/" + file.strip(".jpg") + ".xml"))		
		os.system("cp {} {}".format(data_source + "/labels/" + file.strip(".jpg") + ".xml",train_dir + "/VOC2007/ICDAR/" + file.strip(".jpg") + ".xml"))		
		os.system("cp {} {}".format(data_source + "/page/" + file.strip(".jpg") + ".xml",train_dir + "/VOC2007/Transkribus/" + file.strip(".jpg") + ".xml"))		
		list_train.append( int( file.strip(".jpg") ) )
		available_files.remove(file)

	for _ in range(len(available_files)):
		file = random.choice(available_files)
		os.system("cp {} {}".format(data_source + '/' + file,holdout_dir + "/VOC2007/JPEGImages/" + file))		
		os.system("cp {} {}".format(data_source + "/VOC_annotations/" + file.strip(".jpg") + ".xml", holdout_dir + "/VOC2007/Annotations/" + file.strip(".jpg") + ".xml"))		
		os.system("cp {} {}".format(data_source + "/labels/" + file.strip(".jpg") + ".xml", holdout_dir + "/VOC2007/ICDAR/" + file.strip(".jpg") + ".xml"))		
		os.system("cp {} {}".format(data_source + "/page/" + file.strip(".jpg") + ".xml", holdout_dir + "/VOC2007/Transkribus/" + file.strip(".jpg") + ".xml"))		
		list_test.append( int( file.strip(".jpg") ) )
		available_files.remove(file)

	list_train = sorted( list_train )
	write_handle = codecs.open( train_dir + "/VOC2007/ImageSets/main.txt", 'w', 'utf-8', errors = 'replace' )
	for image_id in list_train :
		write_handle.write( str(image_id) + '\n' )
	write_handle.close()

	list_test = sorted( list_test )
	write_handle = codecs.open( holdout_dir + "/VOC2007/ImageSets/main.txt", 'w', 'utf-8', errors = 'replace' )
	for image_id in list_test :
		write_handle.write( str(image_id) + '\n' )
	write_handle.close()

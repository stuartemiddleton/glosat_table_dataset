## GloSAT Historical Measurement Table Dataset
Dataset containing scanned historical measurement table documents from ship logs and land measurement stations. Annotations provided in this dataset are designed to allow finergrained table detection and table structure recognition models to be trained and tested. Annotations are region boundaries for tables, cells, headings, headers and captions.

This dataset release includes code to train models on a training split, to use trained model checkpoints for inference and to evaluate interred results on a test split. Pretrained models used in the published HIP-2021 paper are included in the dataset so results can be easily reproduced without training the model checkpoints yourself.

Instructions and code can be found on this github repository. Examples of some processed scanned pages from different model types can be seen in the examples folder.

Dataset and model checkpoint files can be downloaded from Zendo. Dataset files should be checked into the datasets dir. Model checkpoint files should be checked into the models dir. Zendo dataset https://doi.org/10.5281/zenodo.5363456

A snapshot release of the github site can also be downloaded from Zendo.

Data sourced for a total of 500 annotated images. Original images sourced with permission from UK Met Office, US NOAA and weatheerrescue.org (University of Reading).

| Source ID; Region; Timeframe | Images / Tables / Headers | Page Style; Table Style |
| ---------------------------- | ------------------------- | ----------------------- |
| 20cr_DWR_MO; India; 1970s | 24 / 31 / 31 | Printed; Borderless |
| 20cr_DWR_NOAA; India; 1930s | 24 / 24 / 24 | Printed; Semi-bordered |
| 20cr_Kubota; Philippines; 1900s | 24 / 28 / 28 | Printed; Semi-bordered |
| 20cr_Natal_Witness; Africa; 1870s | 26 / 26 / 26 | Printed; Semi-bordered |
| Ben Nevis; UK; 1890s | 97 / 137 / 82 | Printed; Semi-bordered |
| DWR; UK and world; 1900s | 93 / 139 / 139 | Mixed; Semi-bordered |
| WesTech Rodgers; Arctic; 1880s | 82 / 164 / 82 | Mixed; Semi-bordered |
| WR_10_years; UK; 1830s to 1930s | 97 / 129 / 129 | Mixed; Bordered |
| WR_Devon_Extern; UK; 1890s to 1940s | 33 / 33/ 33 | Mixed; Bordered |
| Total | 500 / 710 / 573 | |

This work can be cited as:

Ziomek. J. Middleton, S.E. GloSAT Historical Measurement Table Dataset: Enhanced Table Structure Recognition Annotation for Downstream Historical Data Rescue, 6th International Workshop on Historical Document Imaging and Processing (HIP-2021), Sept 5-6, 2021, Lausanne, Switzerland

A pre-print of the HIP-2021 paper can be found on the authors website https://www.southampton.ac.uk/~sem03/HIP_2021.pdf

This work is part of the GloSAT project https://www.glosat.org/ and supported by the Natural Environment Research Council (NE/S015604/1). The authors acknowledge the use of the IRIDIS High Performance Computing Facility, and associated support services at the University of Southampton, in the completion of this work.

# Installation under Ubuntu 20.04LTS

```
cd /data/glosat_table_dataset

# install conda see https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
chmod +x Anaconda3-2020.07-Linux-x86_64.sh
./Anaconda3-2020.07-Linux-x86_64.sh
conda list

# create open-mmlab env in conda
conda create --yes --use-local -n open-mmlab python=3.7 -y
conda init bash
conda config --set auto_activate_base false
conda activate open-mmlab

# check you have python 3.7 via conda
python3 -V

# install cuda and torch
conda install --yes -c anaconda cudatoolkit=10.0
python3 -m pip install --user torch==1.4.0 torchvision==0.5.0

# install mmdetection
# note: mmdetection tutorials are located at https://mmdetection.readthedocs.io/en/latest/index.html
# note: delete the ./build dir if re-installing
python3 -m pip install --user mmcv terminaltables
git clone --branch v1.2.0 https://github.com/open-mmlab/mmdetection.git

cd /data/glosat_table_dataset/mmdetection
python3 -m pip install --user -r requirements/optional.txt
rm -rf build
python3 -m pip install --user pillow==6.2.1
python3 setup.py install --user
python3 setup.py develop --user
python3 -m pip install --user -r "requirements.txt"
python3 -m pip install --user mmcv==0.4.3

# In case, sklearn is not able to install use the following command
# export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
python3 -m pip install --user sklearn #scikit-learn

python3 -m pip install --user pycocotools

# manually install the mmcv model file hrnetv2_w32-dc9eeb4f.pth as later versions of mmcv have changed the download link from AWS to aliyun cloud provider (so it breaks on download from old AWS link)
cp /data/glosat_table_dataset/models/hrnetv2_w32-dc9eeb4f.pth /home/sem03/.cache/torch/checkpoints/hrnetv2_w32-dc9eeb4f.pth

# install glosat dataset files into mmdetection
cp -R /data/glosat_table_dataset/dla /data/glosat_table_dataset/mmdetection
cd /data/glosat_table_dataset/mmdetection
python3 dla/src/install.py

# change default classes for voc to GloSAT table classes
nano /data/glosat_table_dataset/mmdetection/mmdet/datasets/voc.py
	CLASSES = ('table_body','cell','full_table','header','heading')

# prepare the VOC training data files for GloSAT dataset (course and fine)
cd /data/glosat_table_dataset/datasets/GloSAT_dataset_coarse
unzip 20cr_DWR_MO.zip
unzip 20cr_DWR_NOAA.zip
unzip 20cr_Kubota.zip
unzip 20cr_Natal_Witnes.zip
unzip Ben_Nevis.zip
unzip DWR.zip
unzip WesTech_Rodgers.zip
unzip WR_10_years.zip
unzip WR_Devon_Extern.zip

cd /data/glosat_table_dataset/datasets/GloSAT_dataset_fine
unzip 20cr_DWR_MO.zip
unzip 20cr_DWR_NOAA.zip
unzip 20cr_Kubota.zip
unzip 20cr_Natal_Witnes.zip
unzip Ben_Nevis.zip
unzip DWR.zip
unzip WesTech_Rodgers.zip
unzip WR_10_years.zip
unzip WR_Devon_Extern.zip
```

# Train models

This is only needed if you are not using the available pretrained model checkpoints.

```
cd /data/glosat_table_dataset/mmdetection

#
# Train >> Table Detection Model
#


# Table Detection Model >> full_table
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab

mkdir /data/glosat_table_dataset/dla_models
nano dla/src/construct_VOC.py
	dataset_dir = '/data/glosat_table_dataset/datasets/GloSAT_dataset_coarse'
	model_dir = '/data/glosat_table_dataset/dla_models/model_table_det_full_table'
python3 dla/src/construct_VOC.py

ls -la /data/glosat_table_dataset/dla_models/model_table_det_full_table_train/VOC2007
ls -la /data/glosat_table_dataset/dla_models/model_table_det_full_table_test/VOC2007

nano dla/config/cascadeRCNN_full_table_only.py
	model_dir='/data/glosat_table_dataset/dla_models/model_table_det_full_table_train'
	resume_from = None
	total_epochs = 601
	# do less epochs for testing
nohup python3 tools/train.py dla/config/cascadeRCNN_full_table_only.py --work_dir /data/glosat_table_dataset/dla_models/model_table_det_full_table_train > /data/glosat_table_dataset/mmdetection/dla_train.log 2>&1 &

la -la /data/glosat_table_dataset/dla_models/model_table_det_full_table_train/*.pth
cp /data/glosat_table_dataset/dla_models/model_table_det_full_table_train/epoch_601.pth /data/glosat_table_dataset/dla_models/model_table_det_full_table_train/best_model.pth
rm /data/glosat_table_dataset/dla_models/model_table_det_full_table_train/epoch_*.pth

# resume training if needed from a specific epoch
python3 tools/train.py dla/config/cascadeRCNN_full_table_only.py --work_dir /data/glosat_table_dataset/dla_models/model_table_det_full_table_train --resume_from /data/glosat_table_dataset/dla_models/model_table_det_full_table_train/epoch_50.pth

# Table Detection Model >> full table, header, caption (enhanced)
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab

mkdir /data/glosat_table_dataset/dla_models
nano dla/src/construct_VOC.py
	dataset_dir = '/data/glosat_table_dataset/datasets/GloSAT_dataset_coarse'
	model_dir = '/data/glosat_table_dataset/dla_models/model_table_det_enhanced'
python3 dla/src/construct_VOC.py

ls -la /data/glosat_table_dataset/dla_models/model_table_det_enhanced_train/VOC2007
ls -la /data/glosat_table_dataset/dla_models/model_table_det_enhanced_test/VOC2007

nano dla/config/cascadeRCNN_ignore_cells.py
	model_dir='/data/glosat_table_dataset/dla_models/model_table_det_enhanced_train'
	resume_from = None
	total_epochs = 601
	# do less epochs for testing
nohup  python3 tools/train.py dla/config/cascadeRCNN_ignore_cells.py --work_dir /data/glosat_table_dataset/dla_models/model_table_det_enhanced_train > /data/glosat_table_dataset/mmdetection/dla_train.log 2>&1 &

la -la /data/glosat_table_dataset/dla_models/model_table_det_enhanced_train/*.pth
cp /data/glosat_table_dataset/dla_models/model_table_det_enhanced_train/epoch_601.pth /data/glosat_table_dataset/dla_models/model_table_det_enhanced_train/best_model.pth
rm /data/glosat_table_dataset/dla_models/model_table_det_enhanced_train/epoch_*.pth

#
# Train >> Table Structure Recognition Model
#

# Table Structure Recognition Model >> coarse segmentation cells
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab

nano dla/src/construct_VOC.py
	dataset_dir = '/data/glosat_table_dataset/datasets/GloSAT_dataset_coarse'
	model_dir = '/data/glosat_table_dataset/dla_models/model_table_struct_coarse'
python3 dla/src/construct_VOC.py

ls -la /data/glosat_table_dataset/dla_models/model_table_struct_coarse_train/VOC2007
ls -la /data/glosat_table_dataset/dla_models/model_table_struct_coarse_test/VOC2007

nano dla/config/cascadeRCNN_ignore_all_but_cells.py
	model_dir='/data/glosat_table_dataset/dla_models/model_table_struct_coarse_train'
	resume_from = None
	total_epochs = 601
	# do less epochs for testing
nohup python3 tools/train.py dla/config/cascadeRCNN_ignore_all_but_cells.py --work_dir /data/glosat_table_dataset/dla_models/model_table_struct_coarse_train > /data/glosat_table_dataset/mmdetection/dla_train.log 2>&1 &

la -la /data/glosat_table_dataset/dla_models/model_table_struct_coarse_train/*.pth
cp /data/glosat_table_dataset/dla_models/model_table_struct_coarse_train/epoch_601.pth /data/glosat_table_dataset/dla_models/model_table_struct_coarse_train/best_model.pth
rm /data/glosat_table_dataset/dla_models/model_table_struct_coarse_train/epoch_*.pth


# Table Structure Recognition Model >> individual cells (needs reduced memory model)
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab

nano dla/src/construct_VOC.py
	dataset_dir = '/data/glosat_table_dataset/datasets/GloSAT_dataset_fine'
	model_dir = '/data/glosat_table_dataset/dla_models/model_table_struct_fine'
python3 dla/src/construct_VOC.py

ls -la /data/glosat_table_dataset/dla_models/model_table_struct_fine_train/VOC2007
ls -la /data/glosat_table_dataset/dla_models/model_table_struct_fine_test/VOC2007

nano dla/config/cascadeRCNN_ignore_all_but_cells.py
	model_dir='/data/glosat_table_dataset/dla_models/model_table_struct_fine_train'
	resume_from = None
	total_epochs = 601
	# do less epochs for testing
	type='CascadeRCNNFrozenRPN'
nohup python3 tools/train.py dla/config/cascadeRCNN_ignore_all_but_cells.py --work_dir /data/glosat_table_dataset/dla_models/model_table_struct_fine_train > /data/glosat_table_dataset/mmdetection/dla_train.log 2>&1 &

la -la /data/glosat_table_dataset/dla_models/model_table_struct_fine_train/*.pth
cp /data/glosat_table_dataset/dla_models/model_table_struct_fine_train/epoch_601.pth /data/glosat_table_dataset/dla_models/model_table_struct_fine_train/best_model.pth
rm /data/glosat_table_dataset/dla_models/model_table_struct_fine_train/epoch_*.pth

```

In order to change more advanced training settings, one has to edit cascadeRCNN.py config file.
To change the total epoch number, please edit the total_epochs (line 247) variable.
To change the learning rate and optimiser settings, edit the optimizer dictionary (line 288).
Full documentation of the training options can be found here: https://mmdetection.readthedocs.io/en/latest/getting_started.html#train-a-model.

To ignore class, set the dataset type (line 192) to "IgnoringVOCDataset". This give the option to add ignore keyword in data pipelines, eg. ignore = ("cell").

To reduce memory footprint, one can use CascadeRCNNFrozen or CascadeRCNNFrozenRPN.
The first one has all backbone layers frozen. The second one has also RPN network frozen.
To use them simply change the 'type' key value in model dictonary ('model = dict(...)',) in config file.
The type string should be changed to either 'CascadeRCNNFrozen' or 'CascadeRCNNFrozenRPN'.


# Infer and evaluate using models

Commands provided for using both the available pretrained models and ones trained using previous section.

```
cd /data/glosat_table_dataset/mmdetection
mkdir /data/glosat_table_dataset/dla_results

#
# Infer >> Pretrained Model >> Table Detection (GloSAT dataset Test split)
#


# Pretrained Model >> CascadeTabNet original model ( not reported in HIP 2021 paper, downloaded from https://github.com/DevashishPrasad/CascadeTabNet )
# note: inference_original.py is needed not inference.py as the original CascadeTabNet model has different classes (Borderless, bordered etc.), so this separate script is needed to have correct labels.
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab
mkdir /data/glosat_table_dataset/dla_results/cascadetabnet_original_table_det
python3 dla/src/inference_original.py /data/glosat_table_dataset/models/cascadetabnet_epoch_14.pth --load_from /data/glosat_table_dataset/datasets/Test/JPEGImages --out /data/glosat_table_dataset/dla_results/cascadetabnet_original_table_det/ --visual True

python3 dla/src/eval_ICDAR.py /data/glosat_table_dataset/dla_results/cascadetabnet_original_table_det /data/glosat_table_dataset/datasets/Test/Coarse/ICDAR --IoU_threshold 0.1


# Pretrained Model >> CascadeTabNet original model fine-tuned on GloSAT dataset Train split (fulltables only)
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab
mkdir /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_fulltables_table_det
python3 dla/src/inference.py /data/glosat_table_dataset/models/model_fulltables_only_GloSAT.pth --load_from /data/glosat_table_dataset/datasets/Test/JPEGImages --out /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_fulltables_table_det/ --visual True

python3 dla/src/eval_ICDAR.py /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_fulltables_table_det /data/glosat_table_dataset/datasets/Test/Coarse/ICDAR --IoU_threshold 0.1

# Pretrained Model >> CascadeTabNet original model fine-tuned on GloSAT dataset Train split (full table, header, caption)
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab
mkdir /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_det
python3 dla/src/inference.py /data/glosat_table_dataset/models/model_tables_enchanced_GloSAT.pth --load_from /data/glosat_table_dataset/datasets/Test/JPEGImages --out /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_det/ --visual True

python3 dla/src/eval_ICDAR.py /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_det /data/glosat_table_dataset/datasets/Test/Coarse/ICDAR --IoU_threshold 0.1


#
# Infer >> Pretrained Model >> Table Detection (ICDAR dataset and aggregated dataset)
#


# Pretrained Model >> CascadeTabNet original model fine-tuned on cTDaR19 dataset Train split
# model checkpoint = /data/glosat_table_dataset/models/model_tables_ICDAR.pth
# download test data from ICDAR website ( https://zenodo.org/record/2649217#.YSjA2YhKiUk ) and follow the same procedure as for GloSAT models

# Pretrained Model >> CascadeTabNet original model fine-tuned on aggregated dataset Train split
# model checkpoint = /data/glosat_table_dataset/models/model_tables_both.pth
# download image test data from ICDAR website ( https://zenodo.org/record/2649217#.YSjA2YhKiUk ), aggregate with GloSAT dataset Test split and follow the same procedure as for GloSAT models


#
# Infer >> Trained Model >> Table Detection
#


# Trained Model >> fulltable model
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab
mkdir /data/glosat_table_dataset/dla_results/trained_model_fulltables_table_det
python3 dla/src/inference.py /data/glosat_table_dataset/dla_models/model_table_det_full_table_train/best_model.pth --load_from /data/glosat_table_dataset/datasets/Test/JPEGImages --out /data/glosat_table_dataset/dla_results/trained_model_fulltables_table_det/ --visual True

python3 dla/src/eval_ICDAR.py /data/glosat_table_dataset/dla_results/trained_model_fulltables_table_det /data/glosat_table_dataset/datasets/Test/Coarse/ICDAR --IoU_threshold 0.1


# Trained Model >> enhanced model
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab
mkdir /data/glosat_table_dataset/dla_results/trained_model_enhanced_table_det
python3 dla/src/inference.py /data/glosat_table_dataset/dla_models/model_table_det_enhanced_train/best_model.pth --load_from /data/glosat_table_dataset/datasets/Test/JPEGImages --out /data/glosat_table_dataset/dla_results/trained_model_enhanced_table_det/ --visual True

python3 dla/src/eval_ICDAR.py /data/glosat_table_dataset/dla_results/trained_model_enhanced_table_det /data/glosat_table_dataset/datasets/Test/Coarse/ICDAR --IoU_threshold 0.1


#
# Infer >> Pretrained Model >> Table Structure Recognition (GloSAT dataset Test split)
#


# Pretrained Model >> GloSAT (coarse segmentation cells) table detected >> CascadeTabNet original model fine-tuned on GloSAT dataset Train split (fulltables only, coarse cells, post-processing, table det)
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab
mkdir /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells
python3 dla/src/inference.py /data/glosat_table_dataset/models/model_fulltables_only_GloSAT.pth --cell_checkpoint /data/glosat_table_dataset/models/model_coarsecell_GloSAT.pth --load_from /data/glosat_table_dataset/datasets/Test/JPEGImages --out /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells/ --visual True

python3 dla/src/eval_ICDAR_wF1.py /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells /data/glosat_table_dataset/datasets/Test/Coarse/ICDAR
python3 dla/src/eval_ICDAR.py /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells /data/glosat_table_dataset/datasets/Test/Coarse/ICDAR --IoU_threshold 0.5
python3 dla/src/eval_rows_n_cols_only.py /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells /data/glosat_table_dataset/datasets/Test/Coarse/ICDAR --IoU_threshold 0.5


# Pretrained Model >> GloSAT (individual cells) table detected >> CascadeTabNet original model fine-tuned on GloSAT dataset Train split (fulltables only, individual cells, post-processing, table det)
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab
mkdir /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct
python3 dla/src/inference.py /data/glosat_table_dataset/models/model_fulltables_only_GloSAT.pth --cell_checkpoint /data/glosat_table_dataset/models/model_finecell_GloSAT.pth --coarse_cell_checkpoint /data/glosat_table_dataset/models/model_coarsecell_GloSAT.pth --load_from /data/glosat_table_dataset/datasets/Test/JPEGImages --out /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct/ --visual True

python3 dla/src/eval_ICDAR_wF1.py /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct /data/glosat_table_dataset/datasets/Test/Fine/ICDAR
python3 dla/src/eval_ICDAR.py /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct /data/glosat_table_dataset/datasets/Test/Fine/ICDAR --IoU_threshold 0.5
python3 dla/src/eval_rows_n_cols_only.py /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct /data/glosat_table_dataset/datasets/Test/Fine/ICDAR --IoU_threshold 0.5


# Pretrained Model >> GloSAT (coarse segmentation cells) table provided >> CascadeTabNet original model fine-tuned on GloSAT dataset Train split (fulltables only, coarse cells, post-processing, table provided)
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab
mkdir /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells_table_provided
python3 dla/src/inference_regiongiven.py /data/glosat_table_dataset/datasets/Test/Coarse/VOC_without_headercells --cell_checkpoint /data/glosat_table_dataset/models/model_coarsecell_GloSAT.pth --load_from /data/glosat_table_dataset/datasets/Test/JPEGImages --out /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells_table_provided/ --visual True

python3 dla/src/eval_ICDAR_wF1.py /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells_table_provided /data/glosat_table_dataset/datasets/Test/Coarse/ICDAR
python3 dla/src/eval_ICDAR.py /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells_table_provided /data/glosat_table_dataset/datasets/Test/Coarse/ICDAR --IoU_threshold 0.5
python3 dla/src/eval_rows_n_cols_only.py /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells_table_provided /data/glosat_table_dataset/datasets/Test/Coarse/ICDAR --IoU_threshold 0.5


# Pretrained Model >> GloSAT (individual cells) table provided >> CascadeTabNet original model fine-tuned on GloSAT dataset Train split (fulltables only, individual cells, post-processing, table provided)
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab
mkdir /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_table_provided
python3 dla/src/inference_regiongiven.py /data/glosat_table_dataset/datasets/Test/Coarse/VOC_without_headercells --cell_checkpoint /data/glosat_table_dataset/models/model_finecell_GloSAT.pth --coarse_cell_checkpoint /data/glosat_table_dataset/models/model_coarsecell_GloSAT.pth --load_from /data/glosat_table_dataset/datasets/Test/JPEGImages --out /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_table_provided/ --visual True

python3 dla/src/eval_ICDAR_wF1.py /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_table_provided /data/glosat_table_dataset/datasets/Test/Fine/ICDAR
python3 dla/src/eval_ICDAR.py /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_table_provided /data/glosat_table_dataset/datasets/Test/Fine/ICDAR --IoU_threshold 0.5
python3 dla/src/eval_rows_n_cols_only.py /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_table_provided /data/glosat_table_dataset/datasets/Test/Fine/ICDAR --IoU_threshold 0.5


# Pretrained Model without any post processing >> GloSAT (coarse segmentation cells) table detected
# note: no post processing means VOC format is output so needs a different eval script
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab
mkdir /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells_no_post
python3 dla/src/inference.py /data/glosat_table_dataset/models/model_fulltables_only_GloSAT.pth --cell_checkpoint /data/glosat_table_dataset/models/model_coarsecell_GloSAT.pth --load_from /data/glosat_table_dataset/datasets/Test/JPEGImages --out /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells_no_post/ --visual True --raw_cells True

python3 dla/src/eval_VOC_wF1.py /data/glosat_table_dataset/datasets/Test/Coarse/VOC_without_headercells ../dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells_no_post
python3 dla/src/eval_VOC.py /data/glosat_table_dataset/datasets/Test/Coarse/VOC_without_headercells ../dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells_no_post --IoU_threshold 0.5


# Pretrained Model without any post processing >> GloSAT (individual cells) table detected
# note: no post processing means VOC format is output so needs a different eval script
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab
mkdir /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_no_post
python3 dla/src/inference.py /data/glosat_table_dataset/models/model_fulltables_only_GloSAT.pth --cell_checkpoint /data/glosat_table_dataset/models/model_finecell_GloSAT.pth --coarse_cell_checkpoint /data/glosat_table_dataset/models/model_coarsecell_GloSAT.pth --load_from /data/glosat_table_dataset/datasets/Test/JPEGImages --out /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_no_post/ --visual True --raw_cells True

python3 dla/src/eval_VOC_wF1.py /data/glosat_table_dataset/datasets/Test/Fine/VOC_without_headercells ../dla_results/cascadetabnet_GloSAT_table_struct_no_post
python3 dla/src/eval_VOC.py /data/glosat_table_dataset/datasets/Test/Fine/VOC_without_headercells ../dla_results/cascadetabnet_GloSAT_table_struct_no_post --IoU_threshold 0.5


# Pretrained Model without any post processing >> GloSAT (coarse segmentation cells) table provided
# note: no post processing means VOC format is output so needs a different eval script
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab
mkdir /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells_table_provided_no_post
python3 dla/src/inference_regiongiven.py /data/glosat_table_dataset/datasets/Test/Coarse/VOC_without_headercells --cell_checkpoint /data/glosat_table_dataset/models/model_coarsecell_GloSAT.pth --load_from /data/glosat_table_dataset/datasets/Test/JPEGImages --out /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells_table_provided_no_post/ --visual True --raw_cells True

python3 dla/src/eval_VOC_wF1.py /data/glosat_table_dataset/datasets/Test/Coarse/VOC_without_headercells ../dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells_table_provided_no_post
python3 dla/src/eval_VOC.py /data/glosat_table_dataset/datasets/Test/Coarse/VOC_without_headercells ../dla_results/cascadetabnet_GloSAT_table_struct_coarse_cells_table_provided_no_post --IoU_threshold 0.5


# Pretrained Model without any post processing >> GloSAT (individual cells) table provided
# note: no post processing means VOC format is output so needs a different eval script
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab
mkdir /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_table_provided_no_post
python3 dla/src/inference_regiongiven.py /data/glosat_table_dataset/datasets/Test/Coarse/VOC_without_headercells --cell_checkpoint /data/glosat_table_dataset/models/model_finecell_GloSAT.pth --coarse_cell_checkpoint /data/glosat_table_dataset/models/model_coarsecell_GloSAT.pth --load_from /data/glosat_table_dataset/datasets/Test/JPEGImages --out /data/glosat_table_dataset/dla_results/cascadetabnet_GloSAT_table_struct_table_provided_no_post/ --visual True --raw_cells True

python3 dla/src/eval_VOC_wF1.py /data/glosat_table_dataset/datasets/Test/Fine/VOC_without_headercells ../dla_results/cascadetabnet_GloSAT_table_struct_table_provided_no_post
python3 dla/src/eval_VOC.py /data/glosat_table_dataset/datasets/Test/Fine/VOC_without_headercells ../dla_results/cascadetabnet_GloSAT_table_struct_table_provided_no_post --IoU_threshold 0.5


#
# Infer >> Pretrained Model >> Table Structure Recognition (ICDAR dataset and aggregated dataset)
#


# Pretrained Model >> CascadeTabNet original model fine-tuned on cTDaR19 dataset Train split
# model checkpoint = /data/glosat_table_dataset/models/model_cell_ICDAR.pth
# download image test data from ICDAR website ( https://zenodo.org/record/2649217#.YSjA2YhKiUk ) and follow the same procedure as for GloSAT models

# Pretrained Model >> CascadeTabNet original model fine-tuned on aggregated dataset Train split
# model checkpoint (coarse) = /data/glosat_table_dataset/models/model_coarsecell_both.pth
# model checkpoint (fine) = /data/glosat_table_dataset/models/model_finecell_both.pth
# download image test data from ICDAR website ( https://zenodo.org/record/2649217#.YSjA2YhKiUk ) and aggregate with GloSAT dataset Test split and follow the same procedure as for GloSAT models


#
# Infer >> Trained Model >> Table Structure Recognition
#


# Trained Model >> coarse segmentation cells
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab
mkdir /data/glosat_table_dataset/dla_results/trained_model_coarse_table_struct
python3 dla/src/inference.py /data/glosat_table_dataset/dla_models/model_table_det_full_table_train/best_model.pth --cell_checkpoint /data/glosat_table_dataset/dla_models/model_table_struct_coarse_train/best_model.pth --load_from /data/glosat_table_dataset/datasets/Test/JPEGImages --out /data/glosat_table_dataset/dla_results/trained_model_coarse_table_struct/ --visual True

python3 dla/src/eval_ICDAR_wF1.py /data/glosat_table_dataset/dla_results/trained_model_coarse_table_struct /data/glosat_table_dataset/datasets/Test/Coarse/ICDAR
python3 dla/src/eval_ICDAR.py /data/glosat_table_dataset/dla_results/trained_model_coarse_table_struct /data/glosat_table_dataset/datasets/Test/Coarse/ICDAR --IoU_threshold 0.5
python3 dla/src/eval_rows_n_cols_only.py /data/glosat_table_dataset/dla_results/trained_model_coarse_table_struct /data/glosat_table_dataset/datasets/Test/Coarse/ICDAR --IoU_threshold 0.5


# Trained Model >> individual cells
cd /data/glosat_table_dataset/mmdetection
conda activate open-mmlab
mkdir /data/glosat_table_dataset/dla_results/trained_model_fine_table_struct
python3 dla/src/inference.py /data/glosat_table_dataset/dla_models/model_table_det_full_table_train/best_model.pth --cell_checkpoint /data/glosat_table_dataset/dla_models/model_table_struct_fine_train/best_model.pth --coarse_cell_checkpoint /data/glosat_table_dataset/dla_models/model_table_struct_coarse_train/best_model.pth --load_from /data/glosat_table_dataset/datasets/Test/JPEGImages --out /data/glosat_table_dataset/dla_results/trained_model_fine_table_struct/ --visual True

python3 dla/src/eval_ICDAR_wF1.py /data/glosat_table_dataset/dla_results/trained_model_fine_table_struct /data/glosat_table_dataset/datasets/Test/Fine/ICDAR
python3 dla/src/eval_ICDAR.py /data/glosat_table_dataset/dla_results/trained_model_fine_table_struct /data/glosat_table_dataset/datasets/Test/Fine/ICDAR --IoU_threshold 0.5
python3 dla/src/eval_rows_n_cols_only.py /data/glosat_table_dataset/dla_results/trained_model_fine_table_struct /data/glosat_table_dataset/datasets/Test/Fine/ICDAR --IoU_threshold 0.5



```

Three scripts exist for inference:
- inference_original.py - for original model (original)
- inference.py - for pretrained model (B2)
- inference_regiongiven.py - for pretrained with table region given explicitly (B1)

For (original) and (B2), the first argument is the checkpoint file for model used to predict tables.
For region_given (B1), the first argument is path to folder with VOC annotations to read table regions from

Besides that optional arguments can be passed:

For all: 
--visual will generate images with predicted bounding boxes
--voc will generate VOC formatted xml as opposed to ICDAR-19 formatted xml

For original:
--use_cells will use the cells provided by original model (no post-processing)

For pretrained models (B1 and B2):
--cell_checkpoint specifies the model which will be used for cell prediction, cells skipped if not specified
--coarse_cell_chceckpoint specifies the model to use for coarse-assist cell prediction
--raw_cells will skip all post-processing on cells (if not given, post-processing is applied, by default pp is used)
--skip_headers will only segment table bodies, if not given whole tables (inlc. headers) are segmented (by default headers are segmented)

# Latest eval results

These results use the same model checkpoints as used for the HIP 2021 paper but with latest small improvements to post processing code.

Table Detection

| Model | Precision | Recall | F1 |
| ----- | --------- | ------ | -- |
| CascadeTabNet original (no fine tuning) | 0.97 | 1.0 | 0.98 |
| CascadeTabNet + fine tuning on (full table) | 1.0 | 1.0 | 1.0 |
| CascadeTabNet + fine tuning on (full table, header, caption) | 1.0 | 1.0 | 1.0 |

Table Structure Recognition

| Model | Automated Table Detection | Weighted Average F1 Score | Row F1 Score | Col F1 Score |
| ----- | ------------------------- | ------------------------- | ------------ | ------------ |
| GloSAT (coarse segmentation cells) CascadeTabNet + post-processing | Yes | 0.74 | 0.87 | 0.95 |
| GloSAT (coarse segmentation cells) CascadeTabNet + post-processing | No | 0.75 | 0.87 | 0.95 |
| GloSAT (coarse segmentation cells) CascadeTabNet | Yes | 0.37 | | |
| GloSAT (coarse segmentation cells) CascadeTabNet | No | 0.37 | | |
| GloSAT (individual cells) CascadeTabNet + post-processing | Yes | 0.38 | 0.57 | 0.92 |
| GloSAT (individual cells) CascadeTabNet + post-processing | No | 0.39 | 0.58 | 0.92 |
| GloSAT (individual cells) CascadeTabNet | Yes | 0.05 | | |
| GloSAT (individual cells) CascadeTabNet | No | 0.05 | | |


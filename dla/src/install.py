import os
import sys

os.system(f"cp dla/src/installation_files/dataset__init__.py mmdet/datasets/__init__.py")
os.system(f"cp dla/src/installation_files/ignoringvoc.py mmdet/datasets/ignoringvoc.py")
os.system(f"cp dla/src/installation_files/mean_ap.py mmdet/core/evaluation/mean_ap.py")
os.system(f"cp dla/src/installation_files/inference.py mmdet/apis/inference.py")

os.system(f"cp dla/src/installation_files/detectors__init__.py mmdet/models/detectors/__init__.py")
os.system(f"cp dla/src/installation_files/cascade_rcnn_frozen.py mmdet/models/detectors/cascade_rcnn_frozen.py")
os.system(f"cp dla/src/installation_files/cascade_rcnn_frozenrpn.py mmdet/models/detectors/cascade_rcnn_frozenrpn.py")

sys.stdout.write("Installation complete. \n")

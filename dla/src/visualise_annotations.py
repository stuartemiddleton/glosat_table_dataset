import argparse
import cv2

try:
    from dla.src.image_utils import put_box 
    from dla.src.xml_utils import load_VOC_xml
except ModuleNotFoundError:
    from image_utils import put_box
    from xml_utils import load_VOC_xml

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",type=str,default="")
    parser.add_argument("--annotation",type=str,default="")

    args = parser.parse_args()

    image_file = args.image
    annotation_file = args.annotation

    image = cv2.imread(image_file)
    #image = cv2.imread("segmentation_.jpg")

    objects = load_VOC_xml(annotation_file)


    for object_ in objects:
        name,box = object_["name"], object_["bbox"]
        #if name=="header":
        #    put_box(image,box,colour=(0,0,0),thickness=10)
        #if name=="heading":
        #    put_box(image,box,colour=(0,0,0),thickness=10)
        #if name=="full_table":
        #    put_box(image,box,colour=(0,0,0),thickness=10)
        if name=="table_body":
            put_box(image,box,colour=(25,25,25))
        if name=="cell":
            put_box(image,box,colour=(25,25,25))
    
    cv2.imwrite("vis_90.jpg",image)


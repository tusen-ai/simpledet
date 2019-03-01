# -*- coding: utf-8 -*-
"""
    This script allows you to transfer your own data from your own data format to coco format.

    Attention: This is not the official format, it does not require licenses and other redundant info, but can generate
    coco-like dataset which can be accepted by Simpledet.

    TODO: You should reimplement the code from line 31 to the end, this file only describe the format of dataset
    and the way to save it.
"""

import json
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python data_to_coco.py infile outfile")
        exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # The whole coco dataset
    dataset = {
        'licenses': [],
        'info': {},
        'categories': [],   # Required
        'images': [],       # Required
        'annotations': []   # Required
    }

    # TODO: class_map maps the class, which would be added into dataset['categories']
    class_map = {
        "box": 1,
        "can": 2,
        "bottle": 3
    }
    for class_name, class_id in class_map.items():
        dataset['categories'].append({
            'id': class_id,
            'name': class_name,
            'supercategory': 'supercategory_name'
        })

    # TODO: Load your own data
    self_data_list = []
    with open(input_file, 'r') as in_file:
        for line in in_file:
            self_data_list.append(json.loads(line))

    # TODO: Dataset images info, normally you should implement an iter here to append the info
    dataset['images'].append({
        'coco_url': '',
        'date_captured': '',
        'file_name': '',    # Required (str)    image file name
        'flickr_url': '',
        'id': int(),        # Required (int)    id of image
        'license': '',
        'width': int(),     # Required (int)    width of image
        'height': int()     # Required (int)    height of image
    })

    # TODO: Dataset annotation info, normally you should implement an iter here to append the info
    dataset["annotations"].append({
        'area': int(),          # Required (int)    image area
        'bbox': [int()] * 4,    # Required (int)    one of the image bboxes
        'category_id': int(),   # Required (int)    class id of this bbox
        'id': int(),            # Required (int)    bbox id in this image
        'image_id': int(),      # Required (int)    image id of this bbox
        'iscrowd': 0,           # Optional, required only if you want to train for semantic segmentation
        'segmentation': []      # Optional, required only if you want to train for semantic segmentation
    })

    with open(output_file, 'w') as ofile:
        json.dump(dataset, ofile, sort_keys=True, indent=2)


if __name__ == '__main__':
    main()

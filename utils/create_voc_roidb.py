import argparse
import os
import glob
import pickle
import json
import xml.etree.ElementTree as ET

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Generate SimpleDet GroundTruth Database for PASCAL VOC dataset')
    parser.add_argument('--data-dir', help='Path to VOC-like dataset', type=str)
    parser.add_argument('--label-map', help='A json file containing the map from class name to training id', 
        type=str, default="data/label_map/voc_label_map.json")
    parser.add_argument('--split', help='Dataset split', type=str, default=None)

    args = parser.parse_args()
    with open(args.label_map) as f:
        label_map = json.load(f)
    return args.data_dir, label_map, args.split


def create_roidb(data_dir, label_map, split):
    # sanity check
    if not os.path.exists(data_dir):
        raise Exception("{} is not accessible".format(data_dir))
    for subdir in ["Annotations", "JPEGImages", "ImageSets"]:
        if not os.path.exists(os.path.join(data_dir, subdir)):
            raise Exception("{}/{} is not accessible".format(data_dir, subdir))
    
    if split is not None:
        subset = set()
        with open("{}/ImageSets/Main/{}.txt".format(data_dir, split)) as f:
            for line in f:
                subset.add("{}/Annotations/{}.xml".format(data_dir, line.strip()))
    else:
        subset = glob.glob("{}/Annotations/*.xml".format(data_dir))

    roidb = []
    for i, anno_name in enumerate(sorted(subset)):
        tree = ET.parse(anno_name)
        root = tree.getroot()
        h = int(root.find("size/height").text)
        w = int(root.find("size/width").text)
        filename = root.find("filename").text
        image_url = os.path.abspath(os.path.join(data_dir, "JPEGImages", filename))
        assert os.path.exists(image_url)
        im_id = i
        gt_class, gt_bbox = list(), list()
        for obj in root.findall("object"):
            gt_class.append(label_map[obj.find("name").text])
            x1 = float(obj.find("bndbox/xmin").text)
            y1 = float(obj.find("bndbox/ymin").text)
            x2 = float(obj.find("bndbox/xmax").text)
            y2 = float(obj.find("bndbox/ymax").text)
            gt_bbox.append([x1, y1, x2, y2])

        roidb.append(dict(
            gt_class=np.array(gt_class, dtype=np.float32),
            gt_bbox=np.array(gt_bbox, dtype=np.float32),
            flipped=False,
            h=h,
            w=w,
            image_url=image_url,
            im_id=im_id))

    dataset_name = os.path.basename(data_dir).lower()
    if split is not None:
        roidb_name = "data/cache/{}_{}.roidb".format(dataset_name, split)
    else:
        roidb_name = "data/cache/{}.roidb".format(dataset_name)

    with open(roidb_name, "wb") as f:
        pickle.dump(roidb, f)


if __name__ == "__main__":
    os.makedirs("data/cache", exist_ok=True)
    create_roidb(*parse_args())

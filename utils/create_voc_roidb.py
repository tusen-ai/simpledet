import argparse
import os
import glob
import pickle
import json
import xml.etree.ElementTree as ET

import numpy as np


label_map = json.load(open("data/label_map/voc_label_map.json"))


def parse_args():
    parser = argparse.ArgumentParser(description='Generate SimpleDet GroundTruth Database for PASCAL VOC dataset')
    parser.add_argument('--data-dir', help='Path to VOCdevkit', type=str, default="data/VOCdevkit")
    parser.add_argument('--subset', help='VOC2007 or VOC2012', type=str, default="VOC2007")

    args = parser.parse_args()
    return args.data_dir, args.subset


def create_roidb(data_dir, subset):
    if not os.path.exists(os.path.join(data_dir, subset)):
        raise Exception("Can not find VOCdevkit in {}".format(os.path.join(data_dir, subset)))

    roidb = []
    for i, filename in enumerate(sorted(glob.glob("{}/{}/Annotations/*.xml".format(data_dir, subset)))):
        tree = ET.parse(filename)
        root = tree.getroot()
        h = int(root.find("size/height").text)
        w = int(root.find("size/width").text)
        image_url = os.path.abspath(filename.replace("Annotations", "JPEGImages").replace(".xml", ".jpg"))
        im_id = i + 1
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

    with open("{}/{}/ImageSets/Main/trainval.txt".format(data_dir, subset)) as f:
        trainval = set()
        for line in f:
            trainval.add(line.strip())

    if subset != "VOC2012":
        with open("{}/{}/ImageSets/Main/test.txt".format(data_dir, subset)) as f:
            test = set()
            for line in f:
                test.add(line.strip())
    else:
        test = set()

    roidb_trainval, roidb_test = list(), list()
    for rec in roidb:
        if os.path.basename(rec["image_url"]).replace(".jpg", "") in trainval:
            roidb_trainval.append(rec)
        elif os.path.basename(rec["image_url"]).replace(".jpg", "") in test:
            roidb_test.append(rec)

    with open("data/cache/{}_trainval.roidb".format(subset.lower()), "wb") as f:
        pickle.dump(roidb_trainval, f)

    if subset != "VOC2012":
        with open("data/cache/{}_test.roidb".format(subset.lower()), "wb") as f:
            pickle.dump(roidb_test, f)


if __name__ == "__main__":
    data_dir, subset = parse_args()
    os.makedirs("data/cache", exist_ok=True)
    create_roidb(data_dir, subset)

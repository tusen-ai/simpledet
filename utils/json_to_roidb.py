import argparse
import json
import pickle as pkl
from os.path import basename

import numpy as np


def parse_argument():
    parser = argparse.ArgumentParser("Convert json gt to roidb")
    parser.add_argument("--json", type=str, required=True)
    args = parser.parse_args()
    return args.json


def json_to_roidb(json_path):
    with open(json_path) as f:
        json_gt = json.load(f)

    for obj in json_gt:
        obj["gt_class"] = np.array(obj["gt_class"], dtype=np.float32)
        obj["gt_bbox"] = np.array(obj["gt_bbox"], dtype=np.float32)
    with open("data/cache/%s.roidb" % basename(json_path).replace("json", "roidb"), "wb") as fout:
        pkl.dump(json_gt, fout)


if __name__ == "__main__":
    json_to_roidb(parse_argument())

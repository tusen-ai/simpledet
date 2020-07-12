import argparse
import os
import pickle as pkl
import numpy as np
import random
from PIL import Image
import concurrent.futures
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Generate SimpleDet GroundTruth Database for Crowdhuman-like dataset')
    parser.add_argument('--dataset', help='dataset name', type=str)
    parser.add_argument('--dataset-split', help='dataset split, e.g. train, val', type=str)
    parser.add_argument('--num-threads', help='number of threads to process', default=4, type=int)

    args = parser.parse_args()
    return args.dataset, args.dataset_split, args.num_threads

def load_func(fpath):
    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records

def decode_annotations(rec_id):
    global dataset_path
    img_id = records[rec_id]['ID']
    img_url = dataset_path + 'images/' + img_id + '.jpg'
    assert os.path.exists(img_url)
    im = Image.open(img_url)
    im_w, im_h = im.width, im.height

    gt_box = records[rec_id]['gtboxes']
    gt_box_len = len(gt_box)
    valid_bboxes = []
    valid_classes = []
    for ii in range(gt_box_len):
        each_data = gt_box[ii]
        x, y, w, h = each_data['fbox']

        if w <= 0 or h <= 0:
            continue

        x1 = x; y1 = y; x2 = x + w; y2 = y + h

        valid_bbox = [x1, y1, x2, y2]
        valid_bboxes.append(valid_bbox)
        if each_data['tag'] == 'person':
            tag = 1
        else:
            tag = -2
        if 'extra' in each_data:
            if 'ignore' in each_data['extra']:
                if each_data['extra']['ignore'] != 0:
                    tag = -2
        valid_classes.append(tag)

    valid_bboxes = np.array(valid_bboxes).reshape(-1, 4)
    valid_classes = np.array(valid_classes).reshape(-1,)

    valid_num = valid_bboxes.shape[0]
    rand_ind = np.arange(valid_num)
    np.random.shuffle(rand_ind)
    gt_bbox = valid_bboxes[rand_ind]
    gt_class = valid_classes[rand_ind]

    roi_rec = {
        'image_url': img_url,
        'im_id': rec_id,
        'id': img_id,
        'h': im_h,
        'w': im_w,
        'gt_class': gt_class,
        'gt_bbox': gt_bbox,
        'flipped': False
    }
    return roi_rec, gt_bbox.shape[0]

if __name__ == "__main__":
    dataset_name, dataset_type, num_threads = parse_args()

    dataset_path = 'data/%s/' % dataset_name
    ch_file_path = dataset_path + 'annotations/annotation_%s.odgt' % dataset_type
    json_file_path = dataset_path + 'annotations/annotation_%s.json' % dataset_type

    records = load_func(ch_file_path)
    print("Loading Annotations Done")

    roidbs = []; num_bbox = 0
    rec_ids = list(range(len(records)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        for index, res_data in enumerate(executor.map(decode_annotations, rec_ids)):
            roidb, cnt = res_data
            roidbs.append(roidb)
            num_bbox += cnt
            if index % 1000 == 0:
                print("Finished %d/%d" % (index, len(rec_ids)))
    print("Parsing Bbox Number: %d" % num_bbox)
    os.makedirs("data/cache", exist_ok=True)
    with open("data/cache/%s_%s.roidb" % (dataset_name, dataset_type), "wb") as fout:
        pkl.dump(roidbs, fout)

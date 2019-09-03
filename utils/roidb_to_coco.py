import json
import tempfile

from pycocotools.coco import COCO
from operator_py.detectron_bbox_utils import xyxy_to_xywh


def roidb_to_coco(roidb):
    # The whole coco dataset
    dataset = {
        'categories': [],
        'images': [],
        'annotations': []
    }

    category_ids = set()
    obj_id = 0
    for roirec in roidb:
        dataset['images'].append({
            'id': roirec['im_id'], 
            'width': roirec['w'], 
            'height': roirec['h']
        })
        roirec['gt_bbox'] = xyxy_to_xywh(roirec['gt_bbox'])
        for bbox, cls in zip(roirec['gt_bbox'], roirec['gt_class']):
            x, y, h, w = bbox.tolist()
            dataset["annotations"].append({
                'area': h * w,
                'bbox': [x, y, h, w],
                'category_id': float(cls),
                'id': obj_id,
                'image_id': roirec['im_id'],
                'iscrowd': 0
            })
            obj_id += 1
            category_ids.add(float(cls))
    for class_id in category_ids:
        dataset['categories'].append({
            'id': class_id,
            'name': class_id,
            'supercategory': 'none'
        })
    
    with tempfile.NamedTemporaryFile(mode="w") as f:
        json.dump(dataset, f)
        f.flush()
        coco = COCO(f.name)

    return coco
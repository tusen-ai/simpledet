from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import mxnet as mx

from core.detection_input import DetectionAugmentation


class ResizeCrop2DImageBbox(DetectionAugmentation):
    def __init__(self, pResize):
        super(ResizeCrop2DImageBbox, self).__init__()
        self.p = pResize
    
    def apply(self, input_record):
        p = self.p

        image = input_record["image"]
        gt_bbox = input_record["gt_bbox"].astype(np.float32)

        assert p.short == p.long
        output_size = p.short

        # compute the accurate scale_factor using rounded scaled image size
        height, width = image.shape[:2]
        max_image_size = float(max(height, width))
        image_scale = float(output_size) / max_image_size        
        scaled_height = int(float(height) * image_scale)
        scaled_width = int(float(width) * image_scale)
        # resize input image and crop it to the output size
        scaled_image = cv2.resize(image, (scaled_width, scaled_height), 
                                  interpolation=cv2.INTER_LINEAR)
        scaled_image = scaled_image[
            0:0 + output_size,
            0:0 + output_size, :]
        
        # resize boxes and crop it to the output size
        gt_bbox[:, :4] = gt_bbox[:, :4] * image_scale
        bbox_offset = np.stack([0, 0,
                                0, 0,])
        gt_bbox[:, :4] -= np.reshape(bbox_offset, [1, 4])
        gt_bbox[:, :4] = np.clip(gt_bbox[:, :4], 0, output_size)

        input_record["image"] = scaled_image
        input_record["gt_bbox"] = gt_bbox
        input_record["im_info"] = (scaled_image.shape[0], scaled_image.shape[1], image_scale)        


class RandResizeCrop2DImageBbox(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarry(n, 5)
    output: image, ndarray(h', w', rgb)
            im_info, tuple(h', w', scale)
            gt_bbox, ndarray(n, 5)
    """

    def __init__(self, pResize):
        super(RandResizeCrop2DImageBbox, self).__init__()
        self.p = pResize  # type: ResizeParam

    def apply(self, input_record):
        p = self.p

        image = input_record["image"]
        gt_bbox = input_record["gt_bbox"].astype(np.float32)

        assert p.short == p.long
        # select a random scale factor
        scale_min = p.scale_min
        scale_max = p.scale_max
        output_size = p.short
        random_scale_factor = np.random.uniform(scale_min, scale_max)
        scaled_size = int(random_scale_factor * output_size)

        # recompute the accurate scale_factor using rounded scaled image size
        height, width = image.shape[:2]
        max_image_size = float(max(height, width))
        image_scale = float(scaled_size) / max_image_size

        # select non-zero random offset (x, y) if scaled image is large than output_size
        scaled_height = int(float(height) * image_scale)
        scaled_width = int(float(width) * image_scale)
        offset_y = float(scaled_height - output_size)
        offset_x = float(scaled_width - output_size)
        offset_y = max(0.0, offset_y) * np.random.uniform(0, 1)
        offset_x = max(0.0, offset_x) * np.random.uniform(0, 1)
        offset_y = int(offset_y)
        offset_x = int(offset_x)

        # resize input image and crop it to the output size
        scaled_image = cv2.resize(image, (scaled_width, scaled_height), 
                                  interpolation=cv2.INTER_LINEAR)
        scaled_image = scaled_image[
            offset_y:offset_y + output_size,
            offset_x:offset_x + output_size, :]
        
        # resize boxes and crop it to the output size
        gt_bbox[:, :4] = gt_bbox[:, :4] * image_scale
        bbox_offset = np.stack([offset_x, offset_y,
                                offset_x, offset_y,])
        gt_bbox[:, :4] -= np.reshape(bbox_offset, [1, 4])
        gt_bbox[:, :4] = np.clip(gt_bbox[:, :4], 0, output_size)

        input_record["image"] = scaled_image
        input_record["gt_bbox"] = gt_bbox
        input_record["im_info"] = (scaled_image.shape[0], scaled_image.shape[1], image_scale)
        

if __name__ == "__main__":
    import six.moves.cPickle as pkl
    import time

    import pycocotools.mask as mask_util

    from core.detection_input import ReadRoiRecord, \
        ConvertImageFromHwcToChw, Flip2DImageBbox, Pad2DImageBbox, \
        RenameRecord, AnchorTarget2D, AnchorLoader

    from models.NASFPN.input import RandResizeCrop2DImageBbox


    class ResizeParam:
        short = 640
        long = 640
        scale_min = 0.8
        scale_max = 1.2


    class PadParam:
        short = 640
        long = 640
        max_num_gt = 100


    class AnchorTarget2DParam:
        class generate:
            short = 640 // 16
            long = 640 // 16
            stride = 16
            scales = (2, 4, 8, 16, 32)
            aspects = (0.5, 1.0, 2.0)

        class assign:
            allowed_border = 0
            pos_thr = 0.7
            neg_thr = 0.3
            min_pos_thr = 0.0

        class sample:
            image_anchor = 256
            pos_fraction = 0.5

    class RenameParam:
        mapping = dict(image="data")

    transform = [
        ReadRoiRecord(None),
        RandResizeCrop2DImageBbox(ResizeParam),
        Flip2DImageBbox(),
        Pad2DImageBbox(PadParam),
        ConvertImageFromHwcToChw(),
        AnchorTarget2D(AnchorTarget2DParam),
        RenameRecord(RenameParam.mapping)
    ]

    DEBUG = True

    with open("data/cache/coco_val2017.roidb", "rb") as fin:
        roidb = pkl.load(fin)
        roidb = [rec for rec in roidb if rec["gt_bbox"].shape[0] > 0]
        roidb = [roidb[i] for i in np.random.choice(len(roidb), 20, replace=False)]

        print(roidb[0])
        flipped_roidb = []
        for rec in roidb:
            new_rec = rec.copy()
            new_rec["flipped"] = True
            flipped_roidb.append(new_rec)
        roidb = roidb + flipped_roidb

        loader = AnchorLoader(roidb=roidb,
                              transform=transform,
                              data_name=["data", "im_info", "gt_bbox"],
                              label_name=["rpn_cls_label", "rpn_reg_target", "rpn_reg_weight"],
                              batch_size=2,
                              shuffle=False,
                              kv=None)


        tic = time.time()
        while True:
            try:
                data_batch = loader.next()
                if DEBUG:
                    import uuid
                    print(data_batch.provide_data)
                    print(data_batch.provide_label)
                    print(data_batch.data[0].shape)
                    print(data_batch.label[1].shape)
                    print(data_batch.label[2].shape)
                    data = data_batch.data[0]
                    gt_bbox = data_batch.data[2]
                    for i, (im, bbox) in enumerate(zip(data, gt_bbox)):
                        im = im.transpose((1, 2, 0))[:, :, ::-1].asnumpy()
                        im = np.uint8(im)
                        valid_instance = np.where(bbox[:, -1] != -1)[0]
                        bbox = bbox[valid_instance].asnumpy()
                        for j, bbox_j in enumerate(bbox):
                            x1, y1, x2, y2 = bbox_j[:4].astype(int)
                            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.imwrite(str(uuid.uuid4()) + '.jpg', im)
            except StopIteration:
                toc = time.time()
                print("{} samples/s".format(len(roidb) / (toc - tic)))
                break

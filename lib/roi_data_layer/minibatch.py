# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# import numpy.random as npr
from scipy.misc import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import pdb


def get_minibatch(roidb, num_classes, random_scale_inds):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    # random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
    #                 size=num_images)
    assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
            format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob, im_blob_ir, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob, 'data_ir': im_blob_ir}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes_ir = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes_rgb = np.empty((len(gt_inds), 5), dtype=np.float32)

    gt_boxes_ir[:, 0:4] = roidb[0]['boxes_ir'][gt_inds, :] * im_scales[0]
    gt_boxes_ir[:, 4] = roidb[0]['gt_classes'][gt_inds]

    gt_boxes_rgb[:, 0:4] = roidb[0]['boxes_rgb'][gt_inds, :] * im_scales[0]
    gt_boxes_rgb[:, 4] = roidb[0]['gt_classes'][gt_inds]

    blobs['gt_boxes_ir'] = gt_boxes_ir
    blobs['gt_boxes_rgb'] = gt_boxes_rgb
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)

    blobs['img_id'] = roidb[0]['img_id']

    return blobs


def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
  scales.
  """
    num_images = len(roidb)

    processed_ims = []
    processed_ims_ir = []
    im_scales = []
    for i in range(num_images):
        # im = cv2.imread(roidb[i]['image'])
        im = imread(roidb[i]['image'])
        ir_im = imread(roidb[i]['ir_image'])

        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
        if len(ir_im.shape) == 2:
            ir_im = ir_im[:, :, np.newaxis]
            ir_im = np.concatenate((ir_im, ir_im, ir_im), axis=2)

        # flip the channel, since the original one using cv2
        # rgb -> bgr
        im = im[:, :, ::-1]
        ir_im = ir_im[:, :, ::-1]

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            ir_im = ir_im[:, ::-1, :]

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        ir_im, im_scale = prep_im_for_blob(ir_im, cfg.PIXEL_MEANS, target_size,
                                           cfg.TRAIN.MAX_SIZE)

        im_scales.append(im_scale)
        processed_ims.append(im)
        processed_ims_ir.append(ir_im)

    # Create a blob to hold the input images
    blob, blob_ir = im_list_to_blob(processed_ims, processed_ims_ir)

    return blob, blob_ir, im_scales

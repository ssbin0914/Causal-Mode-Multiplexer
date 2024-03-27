# --------------------------------------------------------
# Pytorch FPN implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang, based on code from faster R-CNN
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.io as sio
import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import time
import cv2
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import xml.etree.ElementTree as ET

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms, soft_nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections
# from model.fpn.resnet import resnet
from model.fpn.resnet_w_rgb_branch import resnet
import pdb

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('exp_name', type=str, default=None, help='experiment name')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="weights",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=2, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--types', dest='types',
                        help='ir/rgb/all', default="none",
                        type=str)
    parser.add_argument('--UKLoss', dest='UKLoss',
	                    help='Whether to use UK Loss (ON/OFF)',
                        default='ON', type=str)
    parser.add_argument('--uncertainty', dest='uncertainty',
                        help='whether to use uncertainty (ON/OFF)',
                        default='ON', type=str)
    parser.add_argument('--hyper', dest='hyper',
                        help='hyperparameter',
                        default=5.0, type=float)

    parser.add_argument('--soft_nms', help='whether use soft_nms', action='store_true')
    args = parser.parse_args()
    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_0712_trainval"
        args.imdbval_name = "voc_0712_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "kaist":
        args.imdb_name = "kaist_train"
        args.imdbval_name = "kaist_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "CVC":
        args.imdb_name = "cvc_train"
        args.imdbval_name = "cvc_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    args.cfg_file = "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    if args.exp_name is not None:
        input_dir = args.load_dir + "/" + args.net + "/" + args.dataset + '/' + args.exp_name
    else:
        input_dir = args.load_dir + "/" + args.net + "/" + args.dataset + '/' + args.exp_name
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'fpn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'vgg16':
        fpn = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fpn = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fpn = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fpn = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fpn.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fpn.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    im_data_ir = torch.FloatTensor(1)
    gt_boxes_rgb = torch.FloatTensor(1)


    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        im_data_ir = im_data_ir.cuda()

    # make variable
    im_data = Variable(im_data, volatile=True)
    im_info = Variable(im_info, volatile=True)
    num_boxes = Variable(num_boxes, volatile=True)
    gt_boxes = Variable(gt_boxes, volatile=True)
    im_data_ir = Variable(im_data_ir, volatile=True)
    gt_boxes_rgb = Variable(gt_boxes_rgb, volatile=True)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fpn.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.0
    else:
        thresh = 0.0

    save_name = 'faster_rcnn_10'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    fpn.eval()

#    matlab_score = []
#    matlab_uncer_cls = []
#    matlab_uncer_loc = []
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for i in range(num_images):
        data = data_iter.next()

        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])
        im_data_ir.data.resize_(data[5].size()).copy_(data[5])
        gt_boxes_rgb.data.resize_(data[6].size()).copy_(data[6])
       
        det_tic = time.time()

#         if vis:
#             rois, cls_prob, bbox_pred, \
#             final_roi_feat, roi_feat = fpn(im_data, im_info, gt_boxes, gt_boxes_rgb, num_boxes, im_data_ir, args.checksession, args.UKLoss, args.uncertainty, args.hyper)
#
#             final_roi_feat = final_roi_feat.data.cpu().numpy()
#             roi_feat = roi_feat.data.cpu().numpy()
# #            relation_matrix = relation_matrix.data.cpu().numpy()

        # rois, cls_prob, bbox_pred, \
        # _, _,pooled_feat_rgb, pooled_feat_ir, tie, te, nde = fpn(im_data, im_info, gt_boxes, gt_boxes_rgb, num_boxes, im_data_ir, args.checksession, args.UKLoss, args.uncertainty, args.hyper)
        rois, cls_prob, bbox_pred, \
        _, _,pooled_feat_rgb, pooled_feat_ir = fpn(im_data, im_info, gt_boxes, gt_boxes_rgb, num_boxes, im_data_ir, args.checksession, args.UKLoss, args.uncertainty, args.hyper)
        # print(tie.size())
        # print(cls_prob.size())
        # det_toc = time.time()
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        pooled_feat_rgb = pooled_feat_rgb.data.cpu().numpy()
        pooled_feat_ir = pooled_feat_ir.data.cpu().numpy()
#        var_cls_ir_epi_part = var_cls_ir_epi_part.data.cpu().numpy()
#        var_cls_rgb_epi_part = var_cls_rgb_epi_part.data.cpu().numpy()

#        sio.savemat('./Uncertainty/' + str(img_ind) + '_' + 'final_roi_feat.mat', {'final_roi_feat':final_roi_feat_})
        #uncer_loc = torch.sqrt(torch.clamp(torch.exp(uncer_loc), min=1e-18))
        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = boxes

        pred_boxes /= data[1][0][2]#.cuda()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im = cv2.imread(imdb.image_path_at(i)[1])  # 1: RGB
            im2show = np.copy(im)
            im_ir = cv2.imread(imdb.image_path_at(i)[0])  # 0: IR
            im2show_ir = np.copy(im_ir)
        file_name = imdb.image_path_at(i)[1].split('/')[-1].split('.')[0]  # 1: RGB
        file_name_ir = imdb.image_path_at(i)[0].split('/')[-1].split('.')[0]  # 0: IR
#        sio.savemat('./Feature_Original/' + str(file_name), {'pooled_feat_rgb':pooled_feat_rgb, 'pooled_feat_ir':pooled_feat_ir})

        #########################################################
        f = open('Detection_Result' + '/' + file_name + '.txt', 'wt')
        #########################################################

        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                # tie_scores = tie[inds]
                # te_scores = te[inds]
                # nde_scores = nde[inds]
                _, order = torch.sort(cls_scores, 0, True)

                # print(cls_scores[0])
                # print(cls_scores[4010])  # same as 'print(_[0])'
                # print(_[0])
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

#                 if vis:
#                     final_roi_feat_tmp = final_roi_feat[inds]
#                     roi_feat_tmp = roi_feat[inds]
# #                    relation_matrix_tmp = relation_matrix[inds]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                # tie_scores = tie_scores.unsqueeze(1)
                # tie_scores = tie_scores[order]
                # te_scores = te_scores.unsqueeze(1)
                # te_scores = te_scores[order]
                # nde_scores = nde_scores.unsqueeze(1)
                # nde_scores = nde_scores[order]
                # if vis:
                #     final_roi_feat_tmp = final_roi_feat_tmp[order]
                #     roi_feat_tmp = roi_feat_tmp[order]
#                    relation_matrix_tmp = relation_matrix_tmp[order]

                if args.soft_nms:
                    np_dets = cls_dets.cpu().numpy().astype(np.float32)
                    keep = soft_nms(np_dets, cfg.TEST.SOFT_NMS_METHOD)  # np_dets will be changed in soft_nms
                    keep = torch.from_numpy(keep).type_as(cls_dets).int()
                    cls_dets = torch.from_numpy(np_dets).type_as(cls_dets)
                else:
                    keep = nms(cls_dets, cfg.TEST.NMS)
                
#                cls_dets = cls_dets[keep.view(-1).long()]
#                 if vis:
#                     final_roi_feat_tmp = final_roi_feat_tmp[keep.view(-1).long()]
#                     roi_feat_tmp = roi_feat_tmp[keep.view(-1).long()]
#                    relation_matrix_tmp = relation_matrix_tmp[keep.view(-1).long()]


                cls_dets = cls_dets[keep.view(-1).long()]
                # tie_scores = tie_scores[keep.view(-1).long()]
                # te_scores = te_scores[keep.view(-1).long()]
                # nde_scores = nde_scores[keep.view(-1).long()]
                # print(cls_dets[:6, 4])
                # print(te_scores[:6, :])
                # print(nde_scores[:6, :])
                # print(tie_scores[:6, :])
                # print('###################################################################')

                ########### Written by Jung Uk ##########
                # with open('Detection_Result' + '/' + file_name + '.txt', 'wt') as f:
                #     for k in xrange(cls_dets.shape[0]):
                #         f.write('person {:.4f} {:.4f} {:.4f} {:.4f} {:.8f}\n'.format(cls_dets[k,0], cls_dets[k,1], cls_dets[k,2], cls_dets[k,3], cls_dets[k,4]*100.0))

                for k in xrange(cls_dets.shape[0]):
                    scores_ = cls_dets[k, 4]
                    f.write('person {:.4f} {:.4f} {:.4f} {:.4f} {:.8f}\n'.format(cls_dets[k, 0], cls_dets[k, 1],
                                                                                 cls_dets[k, 2], cls_dets[k, 3],
                                                                                 cls_dets[k, 4] * 100.0))
                f.close()

                if vis:
                    # full_filename = os.path.join('data', 'KAIST_PED', 'Annotations', 'lwir', file_name + '.txt')
                    # with open(full_filename) as f:
                    #     lines = f.readlines()
                    # for ii, obj in enumerate(lines):
                    #     if ii == 0:
                    #         continue
                    #     info = obj.split()
                    #     if info[0] == 'person' or info[0] == 'cyclist' or info[0] == 'people':
                    #         x1 = float(info[1])
                    #         y1 = float(info[2])
                    #         ## written by Jung Uk - be careful - original : x2, y2
                    #         x2 = min(float(info[3]) + float(info[1]), 640.0 - 1.0)
                    #         y2 = min(float(info[4]) + float(info[2]), 512.0 - 1.0)
                    #         cv2.rectangle(im2show, (int(x1),int(y1)), (int(x2),int(y2)), (0, 255, 0), 2)
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.7)
                    im2show_ir = vis_detections(im2show_ir, imdb.classes[j], cls_dets.cpu().numpy(), 0.7)


#                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), uncertainty_loc.data.cpu().numpy(), j, 0.7)

                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array
                f.close()

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic
        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                         .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

        if vis:
            cv2.imwrite('images/' + file_name + '.png', im2show)
            cv2.imwrite('images_ir/' + file_name_ir + '.png', im2show_ir)
#           assert 1==0
            #pdb.set_trace()
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)

#    sio.savemat('np_struct_arr.mat', {'matlab_score': matlab_score, 'matlab_uncer_cls': matlab_uncer_cls, 'matlab_uncer_loc': matlab_uncer_loc})

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)


    print('Evaluating detections')

    results = []
    overthresh = 0.5
    imdb.evaluate_detections(all_boxes, output_dir)#, overthresh)
    #results.append(recall_val)
#    print('Overthresh: ', overthresh)

#    results = []
#    overthresh = np.arange(0.5, 1.0, 0.05)
#    for t in overthresh:
#        recall_val = imdb.evaluate_detections(all_boxes, output_dir, t)
#        results.append(recall_val)
#    print('Overthresh: ', overthresh)
#    print('Recall: ', results)
#    print('mean : ', sum(results) / len(results))


#    print('Evaluating detections')
#    imdb.evaluate_detections(all_boxes, output_dir)

    end = time.time()
    print("test time: %0.4fs" % (end - start))

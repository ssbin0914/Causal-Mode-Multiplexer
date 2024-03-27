# --------------------------------------------------------
# Pytorch FPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang, based on code from faster R-CNN
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import logging

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from model.utils.net_utils import vis_detections
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient
import cv2
# from model.fpn.resnet import resnet
from model.fpn.resnet_w_rgb_branch import resnet
from tensorboardX import SummaryWriter
from model.utils.summary import *
import pdb
#from torch.nn.utils import clip_grad_norm_

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('exp_name', type=str, default='ResNet-101', help='experiment name')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='kaist', type=str)
    parser.add_argument('--net', dest='net',
                        help='res101, res152, etc',
                        default='res101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=3, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="weights", )
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=4, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--lscale', dest='lscale',
                        help='whether use large scale',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regressio',
                        action='store_true')
    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.008, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=2, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=2, type=int)
    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=2, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=4, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=3685, type=int)
    # log and diaplay
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        help='whether use tensorflow tensorboard',
                        default=False, type=bool)
    parser.add_argument('--types', dest='types',
                        help='day/night/all',
                        default='none', type=str)
    parser.add_argument('--UKLoss', dest='UKLoss',
	                    help='Whether to use Uncertainty-KL Loss (ON/OFF)',
						default='ON', type=str)
    parser.add_argument('--uncertainty', dest='uncertainty',
                        help='Whether to use uncertainty (ON/OFF)',
                        default='ON', type=str)
    parser.add_argument('--hyper', dest='hyper',
                        help='hyperparameter',
                        default=5.0, type=float)

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        num_data = train_size
        self.num_per_batch = int(num_data / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if num_data % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, num_data).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        # rand_num = torch.arange(self.num_per_batch).long().view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


def _print(str, logger=None):
    print(str)
    if logger is None:
        return
    logger.info(str)


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.use_tfboard:
        writer = SummaryWriter(comment=args.exp_name)

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                         'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_0712_trainval"
        args.imdbval_name = "voc_0712_test"
        args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                         'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                         'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                         'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":

        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                         'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "kaist":
        args.imdb_name = "kaist_train"
        args.imdbval_name = "kaist_test"
        args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                         'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "CVC":
        args.imdb_name = "cvc_train"
        args.imdbval_name = "cvc_test"
        args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                         'MAX_NUM_GT_BOXES', '50']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.lscale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    # logging.info(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    # _print('{:d} roidb entries'.format(len(roidb)), logging)
    _print('{:d} roidb entries'.format(len(roidb)))

    if args.exp_name is not None:
        output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + '/' + args.exp_name
    else:
        output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes_ir = torch.FloatTensor(1)
    gt_boxes_rgb = torch.FloatTensor(1)
    im_data_ir = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes_ir = gt_boxes_ir.cuda()
        gt_boxes_rgb = gt_boxes_rgb.cuda()
        im_data_ir = im_data_ir.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes_ir = Variable(gt_boxes_ir)
    gt_boxes_rgb = Variable(gt_boxes_rgb)
    im_data_ir = Variable(im_data_ir)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.net == 'res101':
        FPN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        FPN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        FPN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    FPN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(FPN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        load_name = os.path.join(output_dir,
                                 'fpn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        _print("loading checkpoint %s" % (load_name), )
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        FPN.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        _print("loaded checkpoint %s" % (load_name), )

    if args.mGPUs:
        FPN = nn.DataParallel(FPN)

    if args.cuda:
        FPN.cuda()

    iters_per_epoch = int(train_size / args.batch_size)

    for epoch in range(args.start_epoch, args.max_epochs+1):
        # setting to train mode
        FPN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma
        if epoch % (13) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)

        for step in range(iters_per_epoch):
            data = data_iter.next()

            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes_ir.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])
            im_data_ir.data.resize_(data[4].size()).copy_(data[4])
            gt_boxes_rgb.data.resize_(data[5].size()).copy_(data[5])

            FPN.zero_grad()
            # try:
            if args.UKLoss == 'ON':
                _, _, _, rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_bbox_mix, RCNN_loss_cls_mix, kl_loss, \
                roi_labels = FPN(im_data, im_info, gt_boxes_ir, gt_boxes_rgb, num_boxes, im_data_ir, args.session, args.types, args.UKLoss, args.uncertainty, args.hyper)

            elif args.UKLoss == 'OFF':
                _, _, _, rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                roi_labels = FPN(im_data, im_info, gt_boxes_ir, gt_boxes_rgb, num_boxes, im_data_ir, args.session, args.types, args.UKLoss, args.uncertainty, args.hyper)

            else:
                print("Wrong type for UK Loss")
                assert 1==0

#            _, _, _, rpn_loss_cls, rpn_loss_box, \
#            RCNN_loss_cls, RCNN_loss_bbox, _, \
#            roi_labels = FPN(im_data, im_info, gt_boxes_ir, gt_boxes_rgb, num_boxes, im_data_ir, args.session)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

            if args.UKLoss == 'ON':
                loss = loss + RCNN_loss_bbox_mix.mean() + RCNN_loss_cls_mix.mean() + kl_loss.mean()

            loss_temp += loss.data[0]

            # backward
            optimizer.zero_grad()
            #########################################
            # # a = list(FPN.parameters())[0].data
            # # print(a[0])
            # # print(a)
            # # print(list(FPN.parameters())[0].grad)
            # print(list(FPN.parameters())[0])
            #########################################
            loss.backward()
            torch.nn.utils.clip_grad_norm(FPN.parameters(), 10.)

            optimizer.step()

            #########################################
            # # b = list(FPN.parameters())[0].data
            # # print(list(FPN.parameters())[0].grad)
            # print(list(FPN.parameters())[0])
            #########################################

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().data[0]
                    loss_rpn_box = rpn_loss_box.mean().data[0]

                    loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
                    RCNN_loss_bbox = RCNN_loss_bbox.mean().data[0]

                    if args.UKLoss == 'ON':
                        kl_loss = kl_loss.mean().data[0]
                        RCNN_loss_bbox_mix = RCNN_loss_bbox_mix.mean().data[0]
                        RCNN_loss_cls_mix = RCNN_loss_cls_mix.mean().data[0]

                    fg_cnt = torch.sum(roi_labels.data.ne(0))
                    bg_cnt = roi_labels.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.data[0]
                    loss_rpn_box = rpn_loss_box.data[0]

                    loss_rcnn_cls = RCNN_loss_cls.data[0]
                    RCNN_loss_bbox = RCNN_loss_bbox.data[0]
#                    RCNN_loss_bbox_mix = RCNN_loss_bbox_mix.data[0]
#                    RCNN_loss_cls_mix = RCNN_loss_cls_mix.data[0]

                    if args.UKLoss == 'ON':
                        kl_loss = kl_loss.data[0]
                        RCNN_loss_bbox_mix = RCNN_loss_bbox_mix.data[0]
                        RCNN_loss_cls_mix = RCNN_loss_cls_mix.data[0]

                    fg_cnt = torch.sum(roi_labels.data.ne(0))
                    bg_cnt = roi_labels.data.numel() - fg_cnt
                _print("Backbone : %s, UKLoss : %s, uncertainty : %s, hyper : %s" %(args.net, args.UKLoss, args.uncertainty, str(args.hyper)))
                _print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                       % (args.session, epoch, step, iters_per_epoch, loss_temp, lr), )
                _print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start), )

                if args.UKLoss == 'ON':
                    _print("\t\t\trpn_cls: %.4f, rpn_box_ir: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f, rcnn_box_mix: %.4f, rcnn_cls_mix: %.4f, kl_loss: %.4f" \
                           % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, RCNN_loss_bbox, RCNN_loss_bbox_mix, RCNN_loss_cls_mix, kl_loss))

                elif args.UKLoss == 'OFF':
                    _print("\t\t\trpn_cls: %.4f, rpn_box_ir: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f" %(loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, RCNN_loss_bbox))

                else:
                    _print("Wrong type for UK Loss")
                    assert 1==0

#                _print("\t\t\trpn_cls: %.4f, rpn_box_ir: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f" \
#                       % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, RCNN_loss_bbox))

                loss_temp = 0
                start = time.time()

        if args.mGPUs:
            save_name = os.path.join(output_dir, 'fpn_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': FPN.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
        else:
            save_name = os.path.join(output_dir, 'fpn_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': FPN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
        _print('save model: {}'.format(save_name), )

        end = time.time()
        print(end - start)

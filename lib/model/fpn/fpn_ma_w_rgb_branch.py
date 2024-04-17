import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import scipy.sparse as sp
import torchvision.utils as vutils
from model.utils.config import cfg
from model.rpn.rpn_fpn import _RPN_FPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _smooth_l1_loss_epi, _smooth_l1_loss_penalty, _crop_pool_layer, \
    _affine_grid_gen, _affine_theta, loc_uncertainty_loss
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes, bbox_decode
import time
import pdb
from pygcn.layers import GraphConvolution
from pygcn.att_model import GAT

eps = 1e-12


class GradMulConst(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.const, None


def grad_mul_const(x, const):
    return GradMulConst.apply(x, const)


no_cuda = False
is_cuda = not no_cuda and torch.cuda.is_available()


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if is_cuda:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + Variable(sample_gumbel(logits.size()))
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=0.1, hard=True):
    y = logits
    if not hard:
        return y
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard


class _FPN(nn.Module):
    def __init__(self, classes, class_agnostic):
        super(_FPN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.thresh = nn.Threshold(0.8, 0)
        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        self.RCNN_rpn = _RPN_FPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()
        self.relu = nn.ReLU(inplace=True)
        self.constant = nn.Parameter(torch.FloatTensor([0.0]))

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        def weights_init(m, mean, stddev, truncated=False):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        normal_init(self.RCNN_toplayer, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer3, 0, 0.01, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.1, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score_ir_branch, 0, 0.1, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred_ir_branch, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score_rgb_branch, 0, 0.1, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred_rgb_branch, 0, 0.01, cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_top, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def pairwise_distances(self, x):
        x_norm = (x ** 2).sum(-1).unsqueeze(-1)
        y = x
        y_norm = x_norm.transpose(1, 2)
        asd = 2.0 * torch.bmm(x, torch.transpose(y, 1, 2))
        dist = x_norm + y_norm - 2.0 * torch.bmm(x, torch.transpose(y, 1, 2))
        return torch.clamp(dist, min=0)

    def _PyramidRoI_Feat(self, feat_maps, rois, im_info):
        img_area = im_info[0][0] * im_info[0][1]
        h = rois.data[:, 4] - rois.data[:, 2] + 1
        w = rois.data[:, 3] - rois.data[:, 1] + 1
        roi_level = torch.log(torch.sqrt(h * w) / 224.0) / np.log(2)
        roi_level = torch.floor(roi_level + 4)

        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5

        if cfg.POOLING_MODE == 'align':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().squeeze()
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                feat = self.RCNN_roi_align(feat_maps[i], rois[idx_l], scale)
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]
        else:
            print('error')
            assert 1 == 0

        return roi_pool_feat

    def fusion(self, f, ir, rgb, f_fact=False, ir_fact=False, rgb_fact=False):
        f, ir, rgb = self.transform(f, ir, rgb, f_fact, ir_fact, rgb_fact)

        f = torch.sigmoid(f)
        ir = torch.sigmoid(ir)
        rgb = torch.sigmoid(rgb)

        z = f * ir * rgb
        z = 1.0 * torch.log(z + eps)

        return z

    def transform(self, f, ir, rgb, f_fact=False, ir_fact=False, rgb_fact=False):
        if not f_fact:
            f = self.constant * torch.ones_like(f).cuda()

        if not ir_fact:
            ir = self.constant * torch.ones_like(ir).cuda()

        if not rgb_fact:
            rgb = self.constant * torch.ones_like(rgb).cuda()

        return f, ir, rgb

    def forward(self, im_data, im_info, gt_boxes_ir, gt_boxes_rgb, num_boxes, im_data_ir, session, types='None',
                UKLoss='ON', uncertainty='ON', hyper=5.0):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes_ir = gt_boxes_ir.data
        gt_boxes_rgb = gt_boxes_rgb.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        ############ RGB part ##############
        c1 = self.RCNN_layer0(im_data)
        c2 = self.RCNN_layer1(c1)
        c3 = self.RCNN_layer2(c2)
        c4 = self.RCNN_layer3(c3)
        c5 = self.RCNN_layer4(c4)

        p5 = self.RCNN_toplayer(c5)
        p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
        p4 = self.RCNN_smooth1(p4)
        p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
        p3 = self.RCNN_smooth2(p3)
        p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
        p2 = self.RCNN_smooth3(p2)
        ####################################

        ############## IR part #############
        c1_ir = self.RCNN_layer0_ir(im_data_ir)
        c2_ir = self.RCNN_layer1_ir(c1_ir)
        c3_ir = self.RCNN_layer2_ir(c2_ir)
        c4_ir = self.RCNN_layer3_ir(c3_ir)
        c5_ir = self.RCNN_layer4_ir(c4_ir)

        p5_ir = self.RCNN_toplayer_ir(c5_ir)
        p4_ir = self._upsample_add(p5_ir, self.RCNN_latlayer1_ir(c4_ir))
        p4_ir = self.RCNN_smooth1_ir(p4_ir)
        p3_ir = self._upsample_add(p4_ir, self.RCNN_latlayer2_ir(c3_ir))
        p3_ir = self.RCNN_smooth2_ir(p3_ir)
        p2_ir = self._upsample_add(p3_ir, self.RCNN_latlayer3_ir(c2_ir))
        p2_ir = self.RCNN_smooth3_ir(p2_ir)
        ####################################

        p6_ir = self.maxpool2d(p5_ir)
        p6 = self.maxpool2d(p5)

        rpn_feature_maps_rgb = [p2, p3, p4, p5, p6]
        rpn_feature_maps_ir = [p2_ir, p3_ir, p4_ir, p5_ir, p6_ir]

        mrcnn_feature_maps_rgb = [p2, p3, p4, p5]
        mrcnn_feature_maps_ir = [p2_ir, p3_ir, p4_ir, p5_ir]

        rois_ir_, rpn_loss_cls_ir, rpn_loss_bbox_ir = self.RCNN_rpn(rpn_feature_maps_ir, im_info, gt_boxes_ir, num_boxes)
        rois_rgb_, rpn_loss_cls_rgb, rpn_loss_bbox_rgb = self.RCNN_rpn(rpn_feature_maps_rgb, im_info, gt_boxes_rgb, num_boxes)
        rpn_loss_cls = rpn_loss_cls_ir + rpn_loss_cls_rgb
        rpn_loss_bbox = rpn_loss_bbox_ir + rpn_loss_bbox_rgb

        rois_ir = torch.cat((rois_ir_, rois_rgb_), 1)

        # if it is training phrase, then use ground truth bboxes for refining
        if self.training:
            roi_data_ir = self.RCNN_proposal_target(rois_ir, gt_boxes_ir, gt_boxes_rgb, num_boxes)
            rois_ir, rois_label, gt_assign, rois_target_ir, rois_target_rgb, rois_inside_ws, rois_outside_ws = roi_data_ir
            rois_ir = rois_ir.view(-1, 5)

            rois_label = rois_label.view(-1).long()

            rois_ir = Variable(rois_ir)
            rois_label = Variable(rois_label)

            rois_target_ir = Variable(rois_target_ir.view(-1, rois_target_ir.size(2)))
            rois_target_rgb = Variable(rois_target_rgb.view(-1, rois_target_rgb.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target_ir = None
            rois_target_rgb = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            rois_ir = rois_ir.view(-1, 5)
            rois_ir = Variable(rois_ir)

        roi_pool_feat_ir = self._PyramidRoI_Feat(mrcnn_feature_maps_ir, rois_ir, im_info)
        roi_pool_feat_rgb = self._PyramidRoI_Feat(mrcnn_feature_maps_rgb, rois_ir, im_info)

        roi_pool_feat = torch.cat((roi_pool_feat_ir, roi_pool_feat_rgb), 1)
        roi_pool_feat = self.RCNN_reduce(roi_pool_feat)
        pooled_feat = self._head_to_tail(roi_pool_feat)  # ( , )

        ### Uncertainty Calculation ###
        if UKLoss == 'ON':
            if self.training:
                mean_cls_ir_epi = []
                mean2_cls_ir_epi = []
                mean_cls_rgb_epi = []
                mean2_cls_rgb_epi = []

                iteration = 20

                for ii in range(iteration):
                    roi_pool_feat_ir_epi = F.dropout(roi_pool_feat_ir, p=0.5, training=True)
                    pooled_feat_ir_epi = self._head_to_tail_ir_dropout(roi_pool_feat_ir_epi.detach())
                    cls_score_ir_epi = self.RCNN_cls_score_ir(pooled_feat_ir_epi.detach())
                    cls_prob_ir_epi = F.softmax(cls_score_ir_epi.detach())

                    mean_cls_ir_epi.append(cls_prob_ir_epi ** 2)
                    mean2_cls_ir_epi.append(cls_prob_ir_epi)

                    roi_pool_feat_rgb_epi = F.dropout(roi_pool_feat_rgb, p=0.5, training=True)
                    pooled_feat_rgb_epi = self._head_to_tail_rgb_dropout(roi_pool_feat_rgb_epi.detach())
                    cls_score_rgb_epi = self.RCNN_cls_score_rgb(pooled_feat_rgb_epi.detach())
                    cls_prob_rgb_epi = F.softmax(cls_score_rgb_epi.detach())

                    mean_cls_rgb_epi.append(cls_prob_rgb_epi ** 2)
                    mean2_cls_rgb_epi.append(cls_prob_rgb_epi)

                mean1s_cls_ir_epi = torch.stack(mean_cls_ir_epi, dim=0).mean(dim=0)
                mean2s_cls_ir_epi = torch.stack(mean2_cls_ir_epi, dim=0).mean(dim=0)
                mean1s_cls_rgb_epi = torch.stack(mean_cls_rgb_epi, dim=0).mean(dim=0)
                mean2s_cls_rgb_epi = torch.stack(mean2_cls_rgb_epi, dim=0).mean(dim=0)

                var_cls_ir_epi = (mean1s_cls_ir_epi - mean2s_cls_ir_epi ** 2)[:, 1].mean()
                var_cls_rgb_epi = (mean1s_cls_rgb_epi - mean2s_cls_rgb_epi ** 2)[:, 1].mean()

                var_cls_ir_epi_part = (var_cls_ir_epi) / (var_cls_ir_epi + var_cls_rgb_epi + 1e-10)
                var_cls_rgb_epi_part = (var_cls_rgb_epi) / (var_cls_ir_epi + var_cls_rgb_epi + 1e-10)

        pooled_feat_ir = self._head_to_tail_ir(roi_pool_feat_ir)
        bbox_pred_ir = self.RCNN_bbox_pred_ir(pooled_feat_ir)
        cls_score_ir = self.RCNN_cls_score_ir(pooled_feat_ir)

        pooled_feat_rgb = self._head_to_tail_rgb(roi_pool_feat_rgb)
        bbox_pred_rgb = self.RCNN_bbox_pred_rgb(pooled_feat_rgb)
        cls_score_rgb = self.RCNN_cls_score_rgb(pooled_feat_rgb)

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_ir_view = bbox_pred_ir.view(bbox_pred_ir.size(0), int(bbox_pred_ir.size(1) / 4), 4)
            bbox_pred_ir_select = torch.gather(bbox_pred_ir_view, 1,
                                               rois_label.long().view(rois_label.size(0), 1, 1).expand(
                                                   rois_label.size(0), 1, 4))
            bbox_pred_ir = bbox_pred_ir_select.squeeze(1)
            bbox_pred_rgb_view = bbox_pred_rgb.view(bbox_pred_rgb.size(0), int(bbox_pred_rgb.size(1) / 4), 4)
            bbox_pred_rgb_select = torch.gather(bbox_pred_rgb_view, 1,
                                                rois_label.long().view(rois_label.size(0), 1, 1).expand(
                                                    rois_label.size(0), 1, 4))
            bbox_pred_rgb = bbox_pred_rgb_select.squeeze(1)

        RCNN_loss_bbox_ir1 = 0
        RCNN_loss_cls_ir1 = 0
        RCNN_loss_bbox_rgb1 = 0
        RCNN_loss_cls_rgb1 = 0
        if self.training:
            RCNN_loss_bbox_ir1 = _smooth_l1_loss(bbox_pred_ir, rois_target_ir, rois_inside_ws, rois_outside_ws)
            RCNN_loss_cls_ir1 = F.cross_entropy(cls_score_ir, rois_label)
            RCNN_loss_bbox_rgb1 = _smooth_l1_loss(bbox_pred_rgb, rois_target_rgb, rois_inside_ws, rois_outside_ws)
            RCNN_loss_cls_rgb1 = F.cross_entropy(cls_score_rgb, rois_label)

        ############################# IR branch #############################
        roi_pool_feat_ir_temp = grad_mul_const(roi_pool_feat_ir, 0.0)
        pooled_feat_ir_branch = self._head_to_tail_ir_branch(roi_pool_feat_ir_temp)
        cls_score_ir_branch = self.RCNN_cls_score_ir_branch(pooled_feat_ir_branch)
        bbox_pred_ir_branch = self.RCNN_bbox_pred_ir_branch(pooled_feat_ir_branch)
        cls_prob_ir_branch = F.softmax(cls_score_ir_branch)

        RCNN_loss_cls_ir_branch = 0
        RCNN_loss_bbox_ir_branch = 0
        RCNN_loss_bbox_mix_ir_branch = 0
        RCNN_loss_cls_mix_ir_branch = 0

        if self.training:
            # classification loss
            RCNN_loss_cls_ir_branch = F.cross_entropy(cls_score_ir_branch, rois_label)
            # bounding box regression L1 loss
            RCNN_loss_bbox_ir_branch = _smooth_l1_loss(bbox_pred_ir_branch, rois_target_ir, rois_inside_ws,
                                                rois_outside_ws)
            RCNN_loss_bbox_mix_ir_branch = RCNN_loss_bbox_ir_branch
            RCNN_loss_cls_mix_ir_branch = RCNN_loss_cls_ir_branch
        ######################################################################

        ############################# RGB branch #############################
        roi_pool_feat_rgb_temp = grad_mul_const(roi_pool_feat_rgb, 0.0)
        pooled_feat_rgb_branch = self._head_to_tail_rgb_branch(roi_pool_feat_rgb_temp)
        cls_score_rgb_branch = self.RCNN_cls_score_rgb_branch(pooled_feat_rgb_branch)
        bbox_pred_rgb_branch = self.RCNN_bbox_pred_rgb_branch(pooled_feat_rgb_branch)
        cls_prob_rgb_branch = F.softmax(cls_score_rgb_branch)

        RCNN_loss_cls_rgb_branch = 0
        RCNN_loss_bbox_rgb_branch = 0
        RCNN_loss_bbox_mix_rgb_branch = 0
        RCNN_loss_cls_mix_rgb_branch = 0

        if self.training:
            # classification loss
            RCNN_loss_cls_rgb_branch = F.cross_entropy(cls_score_rgb_branch, rois_label)
            # bounding box regression L1 loss
            RCNN_loss_bbox_rgb_branch = _smooth_l1_loss(bbox_pred_rgb_branch, rois_target_rgb, rois_inside_ws,
                                                           rois_outside_ws)
            RCNN_loss_bbox_mix_rgb_branch = RCNN_loss_bbox_rgb_branch
            RCNN_loss_cls_mix_rgb_branch = RCNN_loss_cls_rgb_branch
        #######################################################################

        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)

        te = self.fusion(cls_score, cls_score_ir_branch, cls_score_rgb_branch, f_fact=True, ir_fact=True, rgb_fact=True)
        te_prob = F.softmax(te)
        te_prob_all = te_prob.view(batch_size, rois_ir.size(0)/batch_size, -1)

        ir_gumbel = gumbel_softmax(cls_prob_ir_branch)
        rgb_gumbel = gumbel_softmax(cls_prob_rgb_branch)
        ir_prob_diff = ir_gumbel[:, 1] - ir_gumbel[:, 0]
        rgb_prob_diff = rgb_gumbel[:, 1] - rgb_gumbel[:, 0]
        bias_detect = (-1) * ir_prob_diff * rgb_prob_diff
        bias_detect = self.relu(bias_detect)

        nde = self.fusion(cls_score, cls_score_ir_branch, cls_score_rgb_branch, f_fact=False, ir_fact=True, rgb_fact=False)
        tie = te - bias_detect.unsqueeze(1) * nde
        tie_prob = F.softmax(tie)
        tie_prob_all = tie_prob.view(batch_size, rois_ir.size(0)/batch_size, -1)

        if UKLoss == 'ON':
            distance_matrix_rgb = torch.mm(pooled_feat_rgb, pooled_feat_rgb.transpose(0, 1))
            distance_matrix_rgb = F.normalize(distance_matrix_rgb.view(1, -1), dim=-1, p=1)

            distance_matrix_ir = torch.mm(pooled_feat_ir, pooled_feat_ir.transpose(0, 1))
            distance_matrix_ir = F.normalize(distance_matrix_ir.view(1, -1), dim=-1, p=1)

        if self.training and not types == 'none':
            if types == 'rgb':
                kl_loss = 5.0 * torch.sum(distance_matrix_rgb.detach() * torch.log(
                    (distance_matrix_rgb.detach() + 1e-10) / (distance_matrix_ir + 1e-10)))
                assert 1 == 0
            elif types == 'ir':
                kl_loss = 5.0 * torch.sum(distance_matrix_ir.detach() * torch.log(
                    (distance_matrix_ir.detach() + 1e-10) / (distance_matrix_rgb + 1e-10)))
                assert 1 == 0
            elif types == 'all':
                if uncertainty == 'ON':
                    kl_loss1 = 5.0 * var_cls_ir_epi_part.detach() * torch.sum(
                        distance_matrix_rgb * torch.log((distance_matrix_rgb + 1e-10) / (distance_matrix_ir + 1e-10)))
                    kl_loss2 = 5.0 * var_cls_rgb_epi_part.detach() * torch.sum(
                        distance_matrix_ir * torch.log((distance_matrix_ir + 1e-10) / (distance_matrix_rgb + 1e-10)))
                    kl_loss = kl_loss1 + kl_loss2
                elif uncertainty == 'OFF':
                    kl_loss1 = hyper * torch.sum(
                        distance_matrix_rgb * torch.log((distance_matrix_rgb + 1e-10) / (distance_matrix_ir + 1e-10)))
                    kl_loss2 = hyper * torch.sum(
                        distance_matrix_ir * torch.log((distance_matrix_ir + 1e-10) / (distance_matrix_rgb + 1e-10)))
                    kl_loss = kl_loss1 + kl_loss2
                else:
                    print("Wrong type of args.uncertainty")
                    assert 1 == 0
            else:
                print(types)
                assert 1 == 0

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        if self.training:
            RCNN_loss_cls = F.cross_entropy(tie, rois_label)
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target_ir, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))

        if UKLoss == 'ON':
            RCNN_loss_bbox_mix = RCNN_loss_bbox_rgb1 + RCNN_loss_bbox_ir1
            RCNN_loss_cls_mix = RCNN_loss_cls_rgb1 + RCNN_loss_cls_ir1

        RCNN_loss_bbox_tot = RCNN_loss_bbox_ir_branch + RCNN_loss_bbox + RCNN_loss_bbox_rgb_branch
        RCNN_loss_cls_tot = RCNN_loss_cls_ir_branch + RCNN_loss_cls + RCNN_loss_cls_rgb_branch

        RCNN_loss_bbox_mix_tot = RCNN_loss_bbox_mix_ir_branch + RCNN_loss_bbox_mix + RCNN_loss_bbox_mix_rgb_branch
        RCNN_loss_cls_mix_tot = RCNN_loss_cls_mix_ir_branch + RCNN_loss_cls_mix + RCNN_loss_cls_mix_rgb_branch


        if self.training:
            rois_label = rois_label.view(batch_size, -1)

        rois_ir = rois_ir.view(batch_size, -1, rois_ir.size(1))
        if self.training:
            if UKLoss == 'ON':
                return rois_ir, tie_prob_all, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls_tot, RCNN_loss_bbox_tot, RCNN_loss_bbox_mix_tot, RCNN_loss_cls_mix_tot, kl_loss, rois_label
            elif UKLoss == 'OFF':
                return rois_ir, tie_prob_all, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls_tot, RCNN_loss_bbox_tot, rois_label
            else:
                print("Wrong type for UK Loss")
                assert 1 == 0
        else:
            return rois_ir, tie_prob_all, bbox_pred, roi_pool_feat_rgb, roi_pool_feat_rgb, pooled_feat_rgb, pooled_feat_ir

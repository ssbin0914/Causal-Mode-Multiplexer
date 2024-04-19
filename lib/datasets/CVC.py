from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


# <<<< obsolete


class CVC(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'CVC_' + image_set)
        self._image_set = image_set  # train / test
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path  # data/CVC-14
        self._data_path = self._devkit_path  # data/CVC-14
        self._classes = ('__background__',  # always index 0
                         'person')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.tif'
        self._image_index = self._load_image_set_index()  # File Name without extensions such as txt
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._devkit_path), \
            'bdd_data path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path_ir = os.path.join(self._data_path, 'JPEGImages', 'lwir',
                                     index + self._image_ext)
        image_path_vis = os.path.join(self._data_path, 'JPEGImages', 'visible',
                                      index + self._image_ext)

        assert os.path.exists(image_path_ir), \
            'Path does not exist: {}'.format(image_path_ir)
        assert os.path.exists(image_path_vis), \
            'Path does not exist: {}'.format(image_path_vis)

        return image_path_ir, image_path_vis

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/Main/trainval.txt
        # self._data_path + /ImageSets/Main/test.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where BDD is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'KAIST_PED')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        # /data/cache/bdd_trainval_gt_roidb.pkl
        # /data/cache/bdd_test_gt_roidb.pkl
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if self._image_set == 'test':
            gt_roidb = [self._load_CVC_annotation(index)
                        for index in self.image_index]
        elif self._image_set == 'train':
            gt_roidb = [self._load_CVC_annotation(index)
                        for index in self.image_index]
        #            gt_roidb = [self._load_pascal_annotation(index)
        #                        for index in self.image_index]
        else:
            assert 1 == 0

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_CVC_annotation(self, index):
        """
        Load image and bounding boxes info from txt file in the caltech format.
        """
        # filename = os.path.join(self._data_path, self._image_set, 'Annotations' + self._data_filter + self._imp_type, index + '.txt')
        filename_lwir = os.path.join(self._data_path, 'Annotations', 'lwir', index + '.txt')
        filename_visible = os.path.join(self._data_path, 'Annotations', 'visible', index + '.txt')

#        print(filename_lwir)

        with open(filename_lwir) as f:
            lines = f.readlines()
        with open(filename_visible) as fv:
            lines_visible = fv.readlines()

        id_pers_ir = []
        for i, obj in enumerate(lines):
#            if i == 0:
#                continue
            info = obj.split()

            x1 = max(float(info[0]) - 0.5 * float(info[2]), 0)
            y1 = max(float(info[1]) - 0.5 * float(info[3]), 0)
            ## written by Jung Uk - be careful - original : x2, y2
            x2 = min(float(info[0]) + 0.5 * float(info[2]), 640.0-1.0)
            y2 = min(float(info[1]) + 0.5 * float(info[3]), 471.0-1.0)


#            x1 = float(info[0])
#            y1 = float(info[1])
#            ## written by Jung Uk - be careful - original : x2, y2
#            x2 = min(float(info[2]) + float(info[0]), 640.0 - 1.0)
#            y2 = min(float(info[3]) + float(info[1]), 471.0 - 1.0)

            if x2 <= x1 or y2 <= y1:
                continue
            else:
                id_pers_ir.append(info[9])

        id_pers_visible = []
        for i, obj in enumerate(lines_visible):
#            if i == 0:
#                continue
            info = obj.split()

            x1 = max(float(info[0]) - 0.5 * float(info[2]), 0)
            y1 = max(float(info[1]) - 0.5 * float(info[3]), 0)
            ## written by Jung Uk - be careful - original : x2, y2
            x2 = min(float(info[0]) + 0.5 * float(info[2]), 640.0-1.0)
            y2 = min(float(info[1]) + 0.5 * float(info[3]), 471.0-1.0)


#            x1 = float(info[0])
#            y1 = float(info[1])
#            ## written by Jung Uk - be careful - original : x2, y2
#            x2 = min(float(info[2]) + float(info[0]), 640.0 - 1.0)
#            y2 = min(float(info[3]) + float(info[1]), 471.0 - 1.0)

            if x2 <= x1 or y2 <= y1:
                continue
            else:
                id_pers_visible.append(info[9])
        
        ir_to_remove = []
        for ir_id in id_pers_ir:
            if ir_id not in id_pers_visible:
                ir_to_remove.append(ir_id)

        vis_to_remove = []
        for vis_id in id_pers_visible:
            if vis_id not in id_pers_ir:
                vis_to_remove.append(vis_id)

        if (len(id_pers_ir) - len(ir_to_remove)) == (len(id_pers_visible) - len(vis_to_remove)):
            num_pers = len(id_pers_ir) - len(ir_to_remove)
            for r in ir_to_remove:
                id_pers_ir.remove(r)
            for r in vis_to_remove:
                id_pers_visible.remove(r)
        else:            
            print("Wrong GT boxes matching between IR and Visible")
            assert 1==0

#            if info[0] == 'person' or info[0] == 'cyclist':#or info[0] == 'people' or info[0] == 'person?':
#                x1 = float(info[1])
#                y1 = float(info[2])
#                ## written by Jung Uk - be careful - original : x2, y2
#                x2 = min(float(info[3]) + float(info[1]), 640.0 - 1.0)
#                y2 = min(float(info[4]) + float(info[2]), 512.0 - 1.0)

#                if x2 <= x1 or y2 <= y1:
#                    continue
#                else:
#                    num_pers += 1
#            elif info[0] == 'person?a' or info[0] == 'unpaired' or info[0] == 'people' or info[0] == 'person?':
#                continue
#            else:
#                print(info)
#                raise NotImplementedError

        ##### Ground Truth IR and RGB - reference #####
        boxes_ir = np.zeros((num_pers, 4), dtype=np.uint16)
        boxes_rgb = np.zeros((num_pers, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_pers), dtype=np.int32)
        overlaps = np.zeros((num_pers, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_pers), dtype=np.float32)

        ##### IR - Load object bounding boxes into a data frame.
        ixp_ir = 0
        for i, obj in enumerate(lines):
            # Make pixel indexes 0-based
#            if i == 0:
#                continue
            info = obj.split()

            x1 = max(float(info[0]) - 0.5 * float(info[2]), 0)
            y1 = max(float(info[1]) - 0.5 * float(info[3]), 0)
            ## written by Jung Uk - be careful - original : x2, y2
            x2 = min(float(info[0]) + 0.5 * float(info[2]), 640.0-1.0)
            y2 = min(float(info[1]) + 0.5 * float(info[3]), 471.0-1.0)


#            x1 = float(info[0])
#            y1 = float(info[1])
#            x2 = min(float(info[2]) + float(info[0]), 640.0 - 1.0)
#            y2 = min(float(info[3]) + float(info[1]), 471.0 - 1.0)

#            if info[0] == 'person' or info[0] == 'cyclist':# or info[0] == 'people' or info[0] == 'person?':

            if info[9] in id_pers_ir:
                cls = self._class_to_ind['person']
                boxes_ir[ixp_ir, :] = [x1, y1, x2, y2]
                gt_classes[ixp_ir] = cls
                overlaps[ixp_ir, cls] = 1.0
                seg_areas[ixp_ir] = (x2 - x1 + 1) * (y2 - y1 + 1)
                ixp_ir = ixp_ir + 1
            else:
                continue

        ##### RGB - Load object bounding boxes into a data frame.
        ixp_vis = 0
        for i, obj in enumerate(lines_visible):
            # Make pixel indexes 0-based
#            if i == 0:
#                continue
            info = obj.split()

            x1 = max(float(info[0]) - 0.5 * float(info[2]), 0)
            y1 = max(float(info[1]) - 0.5 * float(info[3]), 0)
            ## written by Jung Uk - be careful - original : x2, y2
            x2 = min(float(info[0]) + 0.5 * float(info[2]), 640.0-1.0)
            y2 = min(float(info[1]) + 0.5 * float(info[3]), 471.0-1.0)


#            x1 = float(info[0])
#            y1 = float(info[1])
#            x2 = min(float(info[2]) + float(info[0]), 640.0 - 1.0)
#            y2 = min(float(info[3]) + float(info[1]), 471.0 - 1.0)

#            if info[0] == 'person' or info[0] == 'cyclist':# or info[0] == 'people' or info[0] == 'person?':
            if info[9] in id_pers_visible:
                boxes_rgb[ixp_vis, :] = [x1, y1, x2, y2]
                ixp_vis = ixp_vis + 1
            else:
                continue

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes_ir': boxes_ir,
                'boxes_rgb': boxes_rgb,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        # TODO: Run the annotations creator for both validation and training labels in the same folder
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            # print('x1: ' + str(x1) + ', y1: ' + str(y1) + ', x2: ' + str(x2) + ', y2: ' + str(y2))

            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult

            cls = 1  # self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            if x2 < x1:
                print(index)
                print(obj.find('name').text)
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_bdd_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        # data/bdd_data/results/bdd/Main/<comp_id>_det_trainval.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results', 'bdd', 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_bdd_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} BDD results file'.format(cls))
            filename = self._get_bdd_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output', thresh=0.5):
        annopath = os.path.join(
            self._devkit_path,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False
        print('BDD metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_bdd_results_file_template().format(cls)
            rec, prec, ap, mr = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=thresh,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        print('miss rate: ' + str(mr))
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir, thresh=0.5):
        self._write_bdd_results_file(all_boxes)
        self._do_python_eval(output_dir, thresh)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_bdd_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

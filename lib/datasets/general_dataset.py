# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# Modifications to pascal_voc.py for using with custom dataset KITTI
import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid

from general_dataset_eval import general_dataset_eval, determine_sample_mode
from fast_rcnn.config import cfg

class general_dataset(imdb):
    def __init__(self, image_set, faster_rcnn_exp):
        imdb.__init__(self, image_set)
        self.faster_rcnn_exp = faster_rcnn_exp
        self._image_set = image_set
        self._classes = self._get_classname_list()
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        # specific config options
        self.config = {'cleanup'     : False,
                       'use_salt'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}
        
        if self.faster_rcnn_exp.use_validation_experiments:
            self.image_set_folder = os.path.join(self.faster_rcnn_exp.misc.image_id_directory, 'temp')
            self.results_foldername = os.path.join(self.faster_rcnn_exp.output_directory, self.faster_rcnn_exp.experiment_name, 'results', 'temp')
            self.image_set_file = os.path.join(self.faster_rcnn_exp.misc.image_id_directory, 'temp', self._image_set + '.txt')
            self.config['cleanup'] = True
        else:
            Split = str(int(self.faster_rcnn_exp.train_data_fraction*100)) + '-' + str(int((100-self.faster_rcnn_exp.train_data_fraction*100)))
            self.image_set_folder = os.path.join(self.faster_rcnn_exp.misc.image_id_directory, Split)
            self.results_foldername = os.path.join(self.faster_rcnn_exp.output_directory, self.faster_rcnn_exp.experiment_name, 'results', Split)
            self.image_set_file = os.path.join(self.faster_rcnn_exp.misc.image_id_directory, Split, self._image_set + '.txt')

        if not os.path.exists(self.results_foldername):
            os.makedirs(self.results_foldername)

        #ADD OTHER IMAGE EXTENSIONS HERE
        self._image_ext = ['.jpg','.png','.JPEG']
        self._image_index = self._load_image_set_index()

    def _get_classname_list(self):
        classname_list = ()
        classname_list = classname_list + ('__background__',)
        with open(self.faster_rcnn_exp.class_names_file, 'r') as c_n_f:
            for classname in c_n_f.readlines():
                classname_list = classname_list + (classname.strip('\n'),)
        return classname_list

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self.faster_rcnn_exp.misc.image_data_directory, index + ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        imageset_file = os.path.join(self.image_set_folder, self._image_set + '.txt')
        assert os.path.exists(imageset_file), \
                'Path does not exist: {}'.format(imageset_file)
        with open(imageset_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        """

        # Don't use any cache
        gt_roidb = [self.load_general_dataset_annotation(index)
                    for index in self.image_index]

        return gt_roidb

    def load_general_dataset_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the kitti format.
        """
        #@ filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        filename = os.path.join(self.faster_rcnn_exp.misc.annotation_directory, index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        # Exclude the samples labeled as dontcare
        objs = [obj for obj in objs if obj.find('name').text.lower().strip != 'dontcare']
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        
        # Training Data sample mode should be accounted for (eg: easy(1),moderate(2) and hard(3) in kitti)
        keepInds = []
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            name = obj.find('name').text.lower().strip()
            #Train only for those classes mentioned in class_names_file
            if name not in self._classes:
                continue
            bbox = obj.find('bndbox')
            # pixel indices are 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            #EXTEND FUNCTIONALITY HERE
            #UPDATE ANNOTATION MODES HERE
            #add any extra annotation fields corresponding to new 'dataset_type'
            #add any extra annotation filters corresponding to new 'dataset_type'
            if self.faster_rcnn_exp.dataset_type.lower() == 'kitti':
                truncated = float(obj.find('truncated').text)
                occluded = int(obj.find('occluded').text)
                sample_mode = determine_sample_mode(self.faster_rcnn_exp.dataset_type, truncated, occluded, y1, y2)
                if sample_mode <= self.faster_rcnn_exp.train_mode:
                    keepInds.append(ix)
            else:
                sample_mode = determine_sample_mode(self.faster_rcnn_exp.dataset_type, -1, -1, -1, -1)
                if sample_mode == self.faster_rcnn_exp.train_mode:
                    keepInds.append(ix)

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        boxes = boxes[keepInds]
        gt_classes = gt_classes[keepInds]
        overlaps = overlaps[keepInds]
        seg_areas = seg_areas[keepInds]

        overlaps = scipy.sparse.csr_matrix(overlaps)

        # print 'INDEX:'
        # print index
        # print 'BOXES:'
        # print boxes
        # print 'GTCLASSES:'
        # print gt_classes

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_general_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = 'comp4_det_' + self._image_set + '_{:s}.txt'
        #@ foldername = os.path.join(self._data_path, 'results', 'Fast-RCNN', self.TrainSplit)
        path = os.path.join(self.results_foldername, filename)
        return path

    def _write_general_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if (cls == '__background__'):
                continue
            print 'Writing {} - {} results file'.format(cls, self.faster_rcnn_exp.experiment_name)
            filename = self._get_general_results_file_template().format(cls)
            with open(filename, 'w') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # Generally 0-based indices are used
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3]))
        return

    def _do_python_eval(self):
        annopath = os.path.join(self.faster_rcnn_exp.misc.annotation_directory, '{:s}.xml')

        aps = []
        # The PASCAL VOC metric changed in 2010 to area of AP curve
        print 'VOC07 metric? ' + ('Yes' if self.faster_rcnn_exp.use07metric else 'No')

        for i, cls in enumerate(self._classes):
            if (cls == '__background__'):
                continue
            filename = self._get_general_results_file_template().format(cls)
            
            rec, prec, ap = general_dataset_eval(self.faster_rcnn_exp.dataset_type, filename, annopath, self.image_set_file, cls, sample_mode=self.faster_rcnn_exp.test_mode, ovthresh=0.7, use_07_metric=self.faster_rcnn_exp.use07metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
            #     cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        
        mAP = np.mean((aps))
        print('Mean AP = {:.4f}'.format(mAP))
        print('~~~~~~~~')
        print('Faster RCNN Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(mAP))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code for VOC')
        print('--------------------------------------------------------------')


        return mAP, aps

    # def _do_matlab_eval(self, output_dir='output'):
    #     print '-----------------------------------------------------'
    #     print 'Computing results with the official MATLAB eval code.'
    #     print '-----------------------------------------------------'
    #     path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
    #                         'VOCdevkit-matlab-wrapper')
    #     cmd = 'cd {} && '.format(path)
    #     cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    #     cmd += '-r "dbstop if error; '
    #     cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
    #            .format(self._data_path, self._get_comp_id(),
    #                    self._image_set, output_dir)
    #     print('Running:\n{}'.format(cmd))
    #     status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes):
    #def evaluate_detections(self, all_boxes, output_dir):
        self._write_general_results_file(all_boxes)
        
        if self.faster_rcnn_exp.evaluate:
            mAP, aps = self._do_python_eval()
        else:
            mAP = -1
            aps = []
            print ('Evaluation mode is set to False. Set `evaluate` to True in config file to get the mAP score.')
        # if self.config['matlab_eval']:
        #     self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_general_results_file_template().format(cls)
                os.remove(filename)

        return mAP, aps
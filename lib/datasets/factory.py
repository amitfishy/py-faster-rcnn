# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

from pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.general_dataset import general_dataset

import numpy as np

__sets = {}
#@# Set up voc_<year>_<split> using selective search "fast" mode
#@for year in ['2007', '2012']:
#@    for split in ['train', 'val', 'trainval', 'test']:
#@        name = 'voc_{}_{}'.format(year, split)
#@        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

#@# Set up coco_2014_<split>
#@for year in ['2014']:
#@    for split in ['train', 'val', 'minival', 'valminusminival']:
#@        name = 'coco_{}_{}'.format(year, split)
#@        __sets[name] = (lambda split=split, year=year: coco(split, year))

#@# Set up coco_2015_<split>
#@for year in ['2015']:
#@    for split in ['test', 'test-dev']:
#@        name = 'coco_{}_{}'.format(year, split)
#@        __sets[name] = (lambda split=split, year=year: coco(split, year))

#@# Basketball set from ImageNet is used for experimenting with custom Datasets
#@from datasets.basketball import basketball
#@basketball_data_path = '/home/amitsinha/data/ImageNet-basketball'

#@for split in ['train', 'val']:
    #@name = '{}_{}'.format('basketball', split)
    #@__sets[name] = (lambda split=split: basketball(split, basketball_data_path))

#@# KITTI dataset is used for training car models
#@from datasets.kitti import kitti
#@kitti_data_path = '/home/amitsinha/data/PASCAL-VOC-CUSTOM-KITTI/VOC2012-CUSTOM-KITTI'
#@# Fraction of train data in the trainval set.
#@TrainSplitFraction = 0.8 #default is use 80% for training and 20% for validation
#@use07metric = True #11 point AP because kitti uses the same
#@kitti_difficulty = 'Easy' #Easy, Moderate, Hard

#@for split in ['train', 'val', 'trainval', 'test']:
    #@name = '{}_{}'.format('kitti', split)
    #@__sets[name] = (lambda split=split: kitti(split, TrainSplitFraction, kitti_data_path, kitti_difficulty, use07metric))    

def init_new_dataset(faster_rcnn_exp):
    if faster_rcnn_exp.train:
        split = faster_rcnn_exp.train_split
        name = faster_rcnn_exp.experiment_name + '_' + split
        __sets[name] = (lambda split=split: general_dataset(split, faster_rcnn_exp))
    if faster_rcnn_exp.test:
        split = faster_rcnn_exp.test_split
        name = faster_rcnn_exp.experiment_name + '_' + split
        __sets[name] = (lambda split=split: general_dataset(split, faster_rcnn_exp))
    return

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

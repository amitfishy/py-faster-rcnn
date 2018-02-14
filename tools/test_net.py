#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb, init_new_dataset
import caffe
import argparse
import pprint
import time, os, sys

def test_controller(faster_rcnn_exp):
    init_new_dataset(faster_rcnn_exp)

    test_proto_file = faster_rcnn_exp.test_proto_file
    faster_rcnn_cfg = faster_rcnn_exp.cfg_file
    cfg_from_file(faster_rcnn_cfg)

    GPU_ID = faster_rcnn_exp.misc.gpu_id
    
    if faster_rcnn_exp.use_trained_weights_test:
        caffemodel = faster_rcnn_exp.trained_model_path
    else:
        caffemodel = faster_rcnn_exp.weights_file_test
    
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)

    print test_proto_file
    print caffemodel

    net = caffe.Net(test_proto_file, str(caffemodel), caffe.TEST)
    net.name = os.path.splitext(os.path.basename(caffemodel))[0]

    imdb_name = faster_rcnn_exp.experiment_name + '_' + faster_rcnn_exp.test_split
    imdb = get_imdb(imdb_name)

    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    mAP, aps = test_net(net, imdb)

    return mAP, aps
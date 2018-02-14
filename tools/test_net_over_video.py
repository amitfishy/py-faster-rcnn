#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net_video
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb, init_new_dataset
import caffe
import argparse
import pprint
import time, os, sys

# def parse_args():
#     """
#     Parse input arguments
#     """
#     parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
#     parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
#                         default=0, type=int)
#     parser.add_argument('--def', dest='prototxt',
#                         help='prototxt file defining the network',
#                         default=None, type=str)
#     parser.add_argument('--net', dest='caffemodel',
#                         help='model to test',
#                         default=None, type=str)
#     parser.add_argument('--cfg', dest='cfg_file',
#                         help='optional config file', default=None, type=str)
#     parser.add_argument('--wait', dest='wait',
#                         help='wait until net file exists',
#                         default=True, type=bool)
#     parser.add_argument('--imdb', dest='imdb_name',
#                         help='dataset to test',
#                         default='voc_2007_test', type=str)
#     parser.add_argument('--comp', dest='comp_mode', help='competition mode',
#                         action='store_true')
#     parser.add_argument('--set', dest='set_cfgs',
#                         help='set config keys', default=None,
#                         nargs=argparse.REMAINDER)
#     parser.add_argument('--vis', dest='vis', help='visualize detections',
#                         action='store_true')
#     parser.add_argument('--savevid', dest='savevid', help='save detection video',
#                         action='store_true')
#     parser.add_argument('--num_dets', dest='max_per_image',
#                         help='max number of detections per image',
#                         default=100, type=int)
#     parser.add_argument('--input_video', dest='input_video',
#                         help='input video to be processed',
#                         default='NoVideo', type=str)    

#     if len(sys.argv) == 1:
#         parser.print_help()
#         sys.exit(1)

#     args = parser.parse_args()
#     return args

# if __name__ == '__main__':
#     args = parse_args()

#     print('Called with args:')
#     print(args)

#     if args.cfg_file is not None:
#         cfg_from_file(args.cfg_file)
#     if args.set_cfgs is not None:
#         cfg_from_list(args.set_cfgs)

#     cfg.GPU_ID = args.gpu_id

#     print('Using config:')
#     pprint.pprint(cfg)

#     while not os.path.exists(args.caffemodel) and args.wait:
#         print('Waiting for {} to exist...'.format(args.caffemodel))
#         time.sleep(10)

#     caffe.set_mode_gpu()
#     caffe.set_device(args.gpu_id)
#     net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
#     net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

#     imdb = get_imdb(args.imdb_name)
#     imdb.competition_mode(args.comp_mode)
#     if not cfg.TEST.HAS_RPN:
#         imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

#     test_net_video(net, imdb, VideoName=args.input_video, max_per_image=args.max_per_image, vis=args.vis, savevid=args.savevid)

def test_controller_video(faster_rcnn_exp, InputVideoPath, OutputVideoPath):
    #init_new_dataset(faster_rcnn_exp)

    test_proto_file = faster_rcnn_exp.test_proto_file
    faster_rcnn_cfg = faster_rcnn_exp.cfg_file
    cfg_from_file(faster_rcnn_cfg)

    GPU_ID = faster_rcnn_exp.misc.gpu_id
    
    if faster_rcnn_exp.use_train_weights_for_video:
        caffemodel = faster_rcnn_exp.trained_model_path
    else:
        caffemodel = faster_rcnn_exp.weights_file_for_video
    
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)

    print test_proto_file
    print caffemodel

    net = caffe.Net(str(test_proto_file), str(caffemodel), caffe.TEST)
    net.name = os.path.splitext(os.path.basename(caffemodel))[0]

    # imdb_name = faster_rcnn_exp.experiment_name + '_' + faster_rcnn_exp.test_split
    # imdb = get_imdb(imdb_name)

    # imdb = get_imdb(imdb_name)

    # if not cfg.TEST.HAS_RPN:
    #     imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    test_net_video(net, faster_rcnn_exp.class_names_file, InputVideoPath=InputVideoPath, OutputVideoPath=OutputVideoPath, thresh=faster_rcnn_exp.detection_threshold_for_video, savevid=True)

    return
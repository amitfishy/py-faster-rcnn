#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb, init_new_dataset
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys
import os

import google.protobuf.text_format

# def parse_args():
#     """
#     Parse input arguments
#     """
#     parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
#     parser.add_argument('--gpu', dest='gpu_id',
#                         help='GPU device id to use [0]',
#                         default=0, type=int)
#     parser.add_argument('--solver', dest='solver',
#                         help='solver prototxt',
#                         default=None, type=str)
#     parser.add_argument('--iters', dest='max_iters',
#                         help='number of iterations to train',
#                         default=40000, type=int)
#     parser.add_argument('--weights', dest='pretrained_model',
#                         help='initialize with pretrained model weights',
#                         default=None, type=str)
#     parser.add_argument('--cfg', dest='cfg_file',
#                         help='optional config file',
#                         default=None, type=str)
#     parser.add_argument('--imdb', dest='imdb_name',
#                         help='dataset to train on',
#                         default='voc_2007_trainval', type=str)
#     parser.add_argument('--rand', dest='randomize',
#                         help='randomize (do not use a fixed seed)',
#                         action='store_true')
#     parser.add_argument('--set', dest='set_cfgs',
#                         help='set config keys', default=None,
#                         nargs=argparse.REMAINDER)

#     if len(sys.argv) == 1:
#         parser.print_help()
#         sys.exit(1)

#     args = parser.parse_args()
#     return args

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

# def modify_model_name(model_orig_path, mod_prefix, max_iters):
#     dir_name = os.path.dirname(model_orig_path)
#     modified_file_name = mod_prefix + '_' + str(max_iters) + '.caffemodel'
#     modified_model_path = os.path.join(dir_name, modified_file_name)

#     os.rename(model_orig_path, modified_model_path)
#     return modified_model_path


#@if __name__ == '__main__':
def train_controller(faster_rcnn_exp):
    #@args = parse_args()

    #@print('Called with args:')
    #@print(args)

    #@ if args.cfg_file is not None:
    #@     cfg_from_file(args.cfg_file)
    #@ if args.set_cfgs is not None:
    #@     cfg_from_list(args.set_cfgs)

    #@ cfg.GPU_ID = args.gpu_id

    #@ print('Using config:')
    #@ pprint.pprint(cfg)

    #@ if not args.randomize:
    #@     # fix the random seeds (numpy and caffe) for reproducibility
    #@     np.random.seed(cfg.RNG_SEED)
    #@     caffe.set_random_seed(cfg.RNG_SEED)

    #@ # set up caffe
    #@ caffe.set_mode_gpu()
    #@ caffe.set_device(args.gpu_id)

    #@ imdb, roidb = combined_roidb(args.imdb_name)
    #@ print '{:d} roidb entries'.format(len(roidb))

    #@ output_dir = get_output_dir(imdb)
    #@ print 'Output will be saved to `{:s}`'.format(output_dir)

    #@ train_net(args.solver, roidb, output_dir,
    #@           pretrained_model=args.pretrained_model,
    #@           max_iters=args.max_iters)
    init_new_dataset(faster_rcnn_exp)

    solver = faster_rcnn_exp.solver_proto_file
    faster_rcnn_cfg = faster_rcnn_exp.cfg_file
    cfg_from_file(faster_rcnn_cfg)

    GPU_ID = faster_rcnn_exp.misc.gpu_id

    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)

    #faster_rcnn_exp.name is same as _image_set
    imdb_name = faster_rcnn_exp.experiment_name + '_' + faster_rcnn_exp.train_split
    imdb, roidb = combined_roidb(imdb_name)

    print '{:d} roidb entries'.format(len(roidb))

    if faster_rcnn_exp.use_validation_experiments:
        subfolder = 'temp'
    else:
        subfolder = str(int(faster_rcnn_exp.train_data_fraction*100)) + '-' + str(int(100-faster_rcnn_exp.train_data_fraction*100))
    output_model_dir = os.path.join(faster_rcnn_exp.output_directory, faster_rcnn_exp.experiment_name, 'models', subfolder)
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    print 'Output will be saved to `{:s}`'.format(output_model_dir)

    if faster_rcnn_exp.use_pretrained_weights:
        pretrained_model_file = faster_rcnn_exp.pretrained_weights_file
        assert os.path.exists(pretrained_model_file), 'Pretrained model file does not exist: `{:s}`'.format(pretrained_model_file)
    else:
        pretrained_model_file = None

    max_iters = faster_rcnn_exp.num_iterations

    model_paths = train_net(solver, roidb, output_model_dir, faster_rcnn_exp.trained_model_filename, pretrained_model_file, max_iters)
    #modified_model_path = modify_model_name(model_paths[-1], faster_rcnn_exp.output_model_prefix, max_iters)
    
    print '------------------------------'
    print 'Trained Model is Stored in Path: `{:s}`'.format(model_paths[-1])
    print '------------------------------'

    return model_paths[-1];
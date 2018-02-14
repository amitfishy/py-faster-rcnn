# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np


def determine_sample_mode(dataset_type, truncated, occluded, y1, y2):
    #EXTEND FUNCTIONALITY HERE
    #UPDATE ANNOTATION MODES HERE
    #Indicate the specific annotation parameters for certain categories here
    #Remeber to associate it to a new dataset_type
    if dataset_type.lower() == 'kitti':
    #Run kitti's easy moderate hard 2d detection
        if ((y2-y1) >= 40) and (truncated <= 0.15) and (occluded == 0):
            sample_mode = 1
        elif ((y2-y1) >= 25) and (truncated <= 0.3) and ((occluded == 0) or (occluded == 1)):
            sample_mode = 2
        elif((y2-y1) >= 25) and (truncated <= 0.5) and ((occluded == 0) or (occluded == 1) or (occluded == 2)):
            sample_mode = 3
        else:
            #include all data
            sample_mode = 4
    #Any other data
    else:
        #include all data for any other dataset (apart from kitti)
        sample_mode = 1

    return sample_mode

def parse_rec(dataset_type, filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    #EXTEND FUNCTIONALITY HERE
    #UPDATE ANNOTATION MODES HERE
    #Update any additional fields that are not included in standard annotations
    #Remeber to associate it to a new dataset_type
    if dataset_type.lower()  == 'kitti':
        #kitti categories - easy, moderate, hard
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text.lower().strip()
            obj_struct['pose'] = float(obj.find('pose').text)
            obj_struct['truncated'] = float(obj.find('truncated').text)
            obj_struct['occluded'] = int(obj.find('occluded').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            sample_mode = determine_sample_mode(dataset_type, obj_struct['truncated'], obj_struct['occluded'], int(bbox.find('ymin').text), int(bbox.find('ymax').text))
            obj_struct['sample_mode'] = sample_mode
            objects.append(obj_struct)
    else:
        #generic category - takes all data regardless, and has basic parameters
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            sample_mode = determine_sample_mode(dataset_type, -1, -1, -1, -1)
            obj_struct['sample_mode'] = sample_mode
            objects.append(obj_struct)

    return objects

def general_dataset_ap(rec, prec, use_07_metric=False):
    """ ap = kitti_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def general_dataset_eval(dataset_type,
             detpath,
             annopath,
             imagesetfile,
             classname,
             sample_mode,
             ovthresh=0.7,
             use_07_metric=False):
    """
    Top level function that does the PASCAL VOC evaluation.
    
    dataset_type: string which indicates the kind of annotations and annotation filtering methods
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    sample_mode: level described by dataset_type
    ovthresh: Overlap threshold (default = 0.7)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(dataset_type, annopath.format(imagename))
        if i % 100 == 0:
            print 'Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        #difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)

        DontCareR = [obj for obj in recs[imagename] if obj['name'] == 'dontcare']
        DontCarebbox = np.array([x['bbox'] for x in DontCareR])
        DontCaredet = [False] * len(DontCareR)
        #EXTEND FUNCTIONALITY HERE
        #UPDATE ANNOTATION MODES HERE
        #Indicate how data should be filtered for testing here
        #Remeber to associate it to a new dataset_type
        if dataset_type.lower() == 'kitti':
            ValidInds = np.array([x['sample_mode']<=sample_mode for x in R]).astype(np.bool)
        else:
            ValidInds = np.array([x['sample_mode']==sample_mode for x in R]).astype(np.bool)
        
        #npos = npos + sum(ValidInds) - sum(ValidInds & difficult)
        npos = npos + sum(ValidInds)
        class_recs[imagename] = {'bbox': bbox,
                                 'DontCarebbox': DontCarebbox,
                                 'ValidInds': ValidInds,
                                 'det': det,
                                 'DontCaredet': DontCaredet}

    # read dets
    # detfile = detpath.format(classname)
    with open(detpath, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    # list of indices to ignore in evaluation
    delInds = []
    #EXTEND FUNCTIONALITY HERE
    #UPDATE ANNOTATION SAMPLE MODES HERE
    #Filter Bounding Boxes based on Size (Height)
    #Remeber to associate it to a new dataset_type
    if dataset_type.lower() == 'kitti':
        if sample_mode == 1:
            MinHeight = 40
        else:
            MinHeight = 25
    else:
        MinHeight = 25

    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)

        if (bb[3] - bb[1]) < MinHeight:
            delInds.append(d)
            continue

        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        DontCare_ovmax = -np.inf
        DontCareBBGT = R['DontCarebbox'].astype(float)   
        
        if DontCareBBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(DontCareBBGT[:, 0], bb[0])
            iymin = np.maximum(DontCareBBGT[:, 1], bb[1])
            ixmax = np.minimum(DontCareBBGT[:, 2], bb[2])
            iymax = np.minimum(DontCareBBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (DontCareBBGT[:, 2] - DontCareBBGT[:, 0] + 1.) *
                   (DontCareBBGT[:, 3] - DontCareBBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            DontCare_ovmax = np.max(overlaps)
            DontCare_jmax = np.argmax(overlaps)


        if ovmax >= ovthresh:
            if R['ValidInds'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                if R['det'][jmax] == 0:
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            if (DontCare_ovmax >= ovthresh) and (R['DontCaredet'][DontCare_jmax] == 0):
                R['DontCaredet'][DontCare_jmax] = 1
            else:
                fp[d] = 1.

        if (fp[d] == 0) and (tp[d] == 0):
            delInds.append(d)

    #delete inds from fp and tp
    np.delete(fp, delInds)
    np.delete(tp, delInds)

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = general_dataset_ap(rec, prec, use_07_metric)

    return rec, prec, ap

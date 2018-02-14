import _init_paths
from fast_rcnn.test import detect_image_online
from fast_rcnn.config import cfg, cfg_from_file
import caffe
import os

class frcnn_online_det():
	def __init__(self, faster_rcnn_exp):
		self.class_names_file = faster_rcnn_exp.class_names_file
		self.detection_thresh_online = faster_rcnn_exp.detection_thresh_online
		self.nms_thresh_online = faster_rcnn_exp.nms_thresh_online
		self.test_proto_file = faster_rcnn_exp.test_proto_file
		self.faster_rcnn_cfg = faster_rcnn_exp.cfg_file
		cfg_from_file(self.faster_rcnn_cfg)

		self.GPU_ID = faster_rcnn_exp.misc.gpu_id
		caffe.set_mode_gpu()
		caffe.set_device(self.GPU_ID)

		self.caffemodel = faster_rcnn_exp.weights_file_online
		self.net = caffe.Net(self.test_proto_file, self.caffemodel, caffe.TEST)
		self.net.name = os.path.splitext(os.path.basename(self.caffemodel))[0]

	def det(self, im):
		all_dets = detect_image_online(self.net, im, self.class_names_file, self.detection_thresh_online, self.nms_thresh_online)

		return all_dets
#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
import random
import torch
import xml.etree.ElementTree as ET

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',),'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
global select_id
select_id = 0
def vis_detections(im, class_name, dets,image_name, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    im_name = image_name.split('.')[0]
    
    im = im[:, :, (2, 1, 0)]
    im_copy = im.copy()
    image_folder='images/'
    jpge_folder = '/home/yanxp/yarley/yanxp/dataset/VOCdevkit2007/VOC2007/JPEGImages/'
    xml_file = '/home/yanxp/yarley/yanxp/dataset/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt' 
    xml_folder = '/home/yanxp/yarley/yanxp/dataset/VOCdevkit2007/VOC2007/Annotations' 
    
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        proposal = im_copy[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:]
        cv2.imwrite(image_folder+im_name+'_'+class_name+'.png',proposal)
        proposal_height=proposal.shape[0]        
        proposal_width=proposal.shape[1]
        total_select=3
        fs=open(xml_file,'r').readlines()
	for i in range(total_select):
            global select_id
	    select_id += 1
	    if select_id > len(fs):
		select_id =0
            xml = fs[select_id].strip('\n')+'.xml'
	    txt = open(os.path.join(xml_folder, xml),'r')
	    if class_name not in txt:
	        select_im_path = os.path.join(jpge_folder, xml.split('.')[0]+'.jpg')
                select_im = cv2.imread(select_im_path)
                if select_im.shape[0]>proposal_height and select_im.shape[1]>proposal_width:
		    start_y=random.randint(0,select_im.shape[0]-proposal_height)
                    start_x=random.randint(0,select_im.shape[1]-proposal_width)
                    combination_im=select_im.copy()
                    combination_im[start_y:start_y+proposal_height, start_x:start_x+proposal_width, :] = proposal[0:proposal_height,0:proposal_width,:]
		    cv2.imwrite('combination_image/'+xml.split('.')[0]+'.jpg',combination_im)
            
	    
def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join('/home/yanxp/yarley/yanxp/dataset/VOCdevkit2012/VOC2012/JPEGImages', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]
        vis_detections(im, cls, dets, image_name,thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    saved_model = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0] %(70000 if dataset == 'pascal_voc' else 110000))

    print (saved_model)
    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(21,
                          tag='default', anchor_scales=[8, 16, 32])

    net.load_state_dict(torch.load(saved_model))

    net.eval()
    net.cuda()

    print('Loaded network {:s}'.format(saved_model))

    trainval="/home/yanxp/yarley/yanxp/dataset/VOCdevkit2012/VOC2012/ImageSets/Main/trainval.txt"
    fs=open(trainval,'r')
    for filename in fs.readlines():
	im_name=filename.strip('\n')+'.jpg'
        demo(net, im_name)

    #plt.show()

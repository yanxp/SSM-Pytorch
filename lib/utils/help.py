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

def softmax(ary):
    ary = ary.flatten()
    expa = np.exp(ary)
    dom = np.sum(expa)
    return expa/dom

def choose_model(dir):
    '''                                                                                                            
    get the latest model in in dir'''
    lists = os.listdir(dir)
    lists.sort(key= lambda fn:os.path.getmtime(os.path.join(dir,fn)))
    return lists[-1]

def calcu_iou(A,B):
    '''
    calculate two box's iou
    '''
    width = min(A[2],B[2])-max(A[0],B[0]) + 1
    height = min(A[3],B[3])-max(A[1],B[1]) + 1
    if width<=0 or height<=0:
        return 0
    Aarea =(A[2]-A[0])*(A[3]-A[1]+1) 
    Barea =(B[2]-B[0])*(B[3]-B[1]+1)
    iner_area = width* height
    return iner_area/(Aarea+Barea-iner_area)

def load_model(net_file ,path):
    '''
    return caffe.Net'''
    import caffe
    net = caffe.Net(net_file, path, caffe.TEST)    
    return net
def judge_y(score):
    '''return :
    y:np.array len(score)
    '''
    y=[]
    for s in score:
        if s==1 or np.log(s)>np.log(1-s):
           y.append(1)
        else:
           y.append(-1)
    return np.array(y, dtype=np.int)
def detect_im(net, detect_idx, imdb,clslambda):
    roidb = imdb.roidb
    allBox =[]; allScore = [];  allY=[] ; al_idx = []
    for i in detect_idx:
        imgpath = imdb.image_path_at(i)
        im = cv2.imread(imgpath)
        height = im.shape[0]; width=im.shape[1]

        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im)
        timer.toc()
        
        BBox=[] # all eligible boxes for this img
        Score=[] # every box in BBox has k*1 score vector
        Y = []
        CONF_THRESH = 0.5 # if this is high then no image can enter al, but low thresh leads many images enter al
        NMS_THRESH = 0.3
        if np.amax(scores[:,1:])<CONF_THRESH:
           al_idx.append(i)
           continue
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(torch.from_numpy(dets), NMS_THRESH)
            dets = dets[keep.numpy(), :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
  
            if len(inds) == 0 :
                #for idx in range(len(keep)):
                #    bbox = dets[idx, :4]
                #    BBox.append(bbox)
                #    k = keep[idx]
                #    Score.append(scores[k].copy())
                #    Y.append(judge_y(scores[k]))               
                continue
#            vis_detections(im, cls, dets, thresh=CONF_THRESH)
            for j in inds:
                bbox = dets[j, :4]
                BBox.append(bbox)
                # find which region this box deriving from
                k = keep[j]
                Score.append(scores[k].copy())
                Y.append(judge_y(scores[k]))
                y = Y[-1]
  
        allBox.append(BBox[:]); allScore.append(Score[:]); allY.append(Y[:])
    return np.array(allScore), np.array(allBox), np.array(allY), al_idx
def judge_uv(loss, gamma, clslambda):
    '''
    return 
    u: scalar
    v: R^kind vector
    '''
    lsum = np.sum(loss)
    dim = loss.shape[0]
    v_val = np.zeros((dim,))

    if(lsum > gamma):
        return False, v_val
    elif lsum < gamma:
        for i,l in enumerate(loss):
            if l > clslambda[i]:
                v_val[i]=0
            else:
                v_val[i]=1-l/clslambda[i]
    return True, v_val
def image_cross_validation(model,roidb,labeledsample,curr_roidb,pre_box,pre_cls):
    '''
    implement image cross validation function
    to choose the highest consistant proposal
    '''
    total_select = 5 # total_select images to paste
    curr_select = 0
    cross_validation = 0
    curr_im = cv2.imread(curr_roidb['image'])
    # crop proposal from image
    bbox = pre_box
    im_proposal = curr_im[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:]
    proposal_height = im_proposal.shape[0]
    proposal_width = im_proposal.shape[1]
    avg_score = 0
    if proposal_width<=0 or proposal_height<=0:
        return False, 0
    for i in labeledsample:
        pre_select_roidb = roidb[i]
        pre_select_cls = pre_select_roidb['gt_classes']
        # select image 
        if pre_cls not in pre_select_cls:
            select_im = cv2.imread(pre_select_roidb['image'])
            if proposal_height > select_im.shape[0] or proposal_width> select_im.shape[1]:
                continue
            # resize the proposal
            if proposal_height>0.6*select_im.shape[0] and proposal_width>0.6*select_im.shape[1]:
                resize_proposal_im =cv2.resize(src=im_proposal,dsize=(int(select_im.shape[1]*0.6),int(select_im.shape[0]*0.6)),interpolation=cv2.INTER_LINEAR)
            elif proposal_height<0.2*select_im.shape[0] and proposal_width<0.2*select_im.shape[1]:
                resize_proposal_im = cv2.resize(src=im_proposal,dsize=(int(select_im.shape[1]*0.6),int(select_im.shape[0]*0.6)),interpolation=cv2.INTER_LINEAR)
            else:
                resize_proposal_im = im_proposal.copy()
            proposal_height_resize =  resize_proposal_im.shape[0]
            proposal_width_resize = resize_proposal_im.shape[1]
            pasted_image = select_im.copy()
            start_y = random.randint(0,select_im.shape[0]-proposal_height_resize)
            start_x = random.randint(0,select_im.shape[1]-proposal_width_resize)
            original_boxex = [start_x,start_y,start_x+proposal_width_resize,start_y+proposal_height_resize]
            pasted_image[start_y:start_y+proposal_height_resize,start_x:start_x+proposal_width_resize,:] = resize_proposal_im[0:proposal_height_resize,0:proposal_width_resize,:]
            # redetect pasted_image
            # curr_select += 1
            pred_scores_pasted,pred_boxes_pasted = im_detect(model,pasted_image)
            boxes_pasted_index = pred_scores_pasted[:,pre_cls].argmax()
            pred_lattent_score = pred_scores_pasted[boxes_pasted_index,pre_cls]
            pred_lattent_boxes = pred_boxes_pasted[boxes_pasted_index,4*int(pre_cls):4*(int(pre_cls) + 1)]
            if len(pred_lattent_boxes) == 0 :
                continue
            overlape_iou = calcu_iou(original_boxex,pred_lattent_boxes)
            curr_select += 1
#            import time
#            t0 = time.time()
#            cv2.imwrite('pasted/'+str(t0)+'.jpg',pasted_image)
            if pred_lattent_score > 0.5 and overlape_iou > 0.5:
                cross_validation += 1
                avg_score += pred_lattent_score
            else:
                cross_validation += 0
            if curr_select >= total_select:
                break
        else:
            continue
    if cross_validation > total_select/2:
        return True,avg_score/cross_validation
    else:
        return False,0
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    plt.switch_backend('Agg')
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig,ax = plt.subplots()
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    import time
    t0 = time.time()
    fig = plt.gcf()
    fig.savefig('images/'+str(t0)+'.jpg')

def blur_image(roidbs,ss_candidate_idx):
    '''
    blur regions except BBox
    '''
    def _handle(roi, idx):
        imgpath = roi['image'].split('/')[-1]
        im = cv2.imread(roi['image'])
        im_bbox = []
        for box in roi['boxes']:
            box = map(int, box)
            im_bbox.append(im[box[1]:box[3], box[0]:box[2]])
        new_im = cv2.blur(im, (25,25))
        for i, box in enumerate(roi['boxes']):
            box = map(int, box)
#        cv2.rectangle(new_im,(box[0],box[1]),(box[2],box[3]),(255,0,0),3)
            new_im[box[1]:box[3], box[0]:box[2]] = im_bbox[i]
    
        path = 'tmpdata/{}'.format(imgpath)
        cv2.imwrite(path, new_im)
        assert os.path.exists(path), "didnt save successfully"
        roi['image'] = path
        return roi
    print ('blur inrelevent regions')
    res_roidb = []
    for i in range(len(roidbs)):
        if len(roidbs[i]['boxes'])>0 and i in ss_candidate_idx and not roidbs[i]['flipped']:
            res_roidb.append(roidbs[i].copy())
            res_roidb[i] = _handle(res_roidb[i], i)
        else:
            res_roidb.append(roidbs[i].copy())
    return res_roidb

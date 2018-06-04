# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import torch
from model.train_val import get_training_roidb, train_net, SolverWrapper,update_training_roidb
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import roi_data_layer.roidb as rdl_roidb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys
import scipy.sparse
import scipy.io as sio
from utils.help import *
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
from bitmap import BitMap
import logging
import operator

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str)
  parser.add_argument('--imdbval', dest='imdbval_name',
                      help='dataset to validate on',
                      default='voc_2007_test', type=str)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152, mobile',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
######################## begin #################################
  parser.add_argument('--enable_al', help='whether or not to use al process',action='store_true',default=False)
  parser.add_argument('--enable_ss', help='whether or not to use ss process',action='store_true',default=True)
######################### end ##################################
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
    roidb = get_training_roidb(imdb)
  return imdb, roidb
######################## begin #################################
def get_Imdbs(imdb_names):
    imdbs = [get_imdb(s) for s in imdb_names.split('+')]
    for imdb in imdbs:
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print ('set proposal method:{:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    return datasets.imdb.Imdbs(imdbs)
######################### end ###################################
if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
############################## begin #######################################
  # train set
#  imdb, roidb = combined_roidb(args.imdb_name)
  imdb = get_Imdbs(args.imdb_name)

  # total num_images
  total_num = imdb.num_images
  # initial num_images 
  initialnum = imdb[imdb.item_name(0)].num_images
  # unlabeled num_images 
  remainnum = imdb[imdb.item_name(1)].num_images

  unflippedImdb = imdb[imdb.item_name(1)]
  print('total num:{}, initial num:{}'.format(total_num,initialnum)) 

  bitmapImdb = BitMap(total_num)
  
  roidb = get_training_roidb(imdb)

  print('{:d} roidb entries'.format(len(roidb)))

  # output directory where the models are saved
  output_dir = get_output_dir(imdb, args.tag)
  print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where the summaries are saved during training
  tb_dir = get_output_tb_dir(imdb, args.tag)
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  _, valroidb = combined_roidb(args.imdbval_name)
  print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip

  # load network
  if args.net == 'vgg16':
    net = vgg16()
  elif args.net == 'res50':
    net = resnetv1(num_layers=50)
  elif args.net == 'res101':
    net = resnetv1(num_layers=101)
  elif args.net == 'res152':
    net = resnetv1(num_layers=152)
  elif args.net == 'mobile':
    net = mobilenetv1()
  else:
    raise NotImplementedError
  # some statistic record
  al_num = 0
  ss_num = 0
  initial_num = len(imdb[imdb.item_name(0)].roidb)
  print ('All VOC2007 images use for initial train after flipped, imagenumbers:%d'%(initial_num))
  for i in range(initialnum):                        
      bitmapImdb.set(i)
  train_roidb = imdb[imdb.item_name(0)].roidb
  # pretrained-model
  pretrained_model_name = args.weight
  
  # some parameters
  tao = args.max_iters
  gamma = 0.15;  clslambda = np.array([-np.log(0.9)]*imdb.num_classes)
  # train record
  loopcounter = 0; train_iters = 0;iters_sum = train_iters
  # control al proportion
  al_proportion_checkpoint = [x*initialnum for x in np.linspace(0.3,2,10)]
  # control ss proportion with respect to al proportion
  ss_proportion_checkpoint = [x*initialnum for x in np.linspace(0.2,2,10)]
  # get solver object
  sw = SolverWrapper(net, imdb, train_roidb, valroidb, output_dir, tb_dir,
                     pretrained_model=pretrained_model_name)
  train_iters = 70000
  iters_sum = train_iters;
  sw.train_model(iters_sum)
  while(True):
      # detact unlabeledidx samples
      unlabeledidx = list(set(range(total_num))-set(bitmapImdb.nonzero()))
      # detect labeledidx
      labeledsample = list(set(bitmapImdb.nonzero()))
      pretrained_model_name = choose_model(output_dir)
      # load latest trained model
      saved_model = os.path.join(output_dir,pretrained_model_name)
      net.create_architecture(21,
                          tag='default', anchor_scales=[8, 16, 32])
      
      net.load_state_dict(torch.load(saved_model))
      print('load lastest model:{} sucessfully!'.format(pretrained_model_name))
      net.eval()
      net.cuda()
      print('Process detect the unlabeled images ...')
      # return detect results of the unlabeledidx samples with the latest model
      scoreMatrix,boxRecord,yVecs, al_idx = detect_im(net,unlabeledidx,unflippedImdb,clslambda)
      unlabeledidx = [ x for x in unlabeledidx if x not in al_idx ]
      # record some detect results for updatable
      al_candidate_idx = []  # record al samples index in imdb
      ss_candidate_idx = []   # record ss samples index in imdb
      discard_num = 0
      ss_fake_gt = []  # record fake labels for ss
      cls_loss_sum = np.zeros((imdb.num_classes,)) # record loss for each cls
      cls_sum = 0 # used for update clslambda
      ss_avg_score = []
      print('Process Self-supervised Sample Mining...')
      for i in range(len(unlabeledidx)):
          im_boxes = []
          im_cls=[]
          cls_sum += len(boxRecord[i])
          ss_accum_score = 0 # ss_scores of per image
          for j,box in enumerate(boxRecord[i]):
              # score of a box
              boxscore = scoreMatrix[i][j]
              # fake label box 
              y = yVecs[i][j]
              # the fai function 
              loss = -((1+y)/2*np.log(boxscore)+(1-y)/2*np.log(1-boxscore+1e-30))
              
              cls_loss_sum += loss
              # choose v,v_val by loss
              v,v_val = judge_uv(loss,gamma,clslambda)
              # SS process
              if v:
                  if(np.sum(y==1)==1 and np.where(y==1)[0]!= 0):
                     # add Imgae Cross Validation
                     pre_cls = np.where(y==1)[0]
                     pre_box = box
                     curr_roidb = roidb[unlabeledidx[i]]
                     cross_validate,avg_score = image_cross_validation(net,roidb,labeledsample,curr_roidb,pre_box,pre_cls)
                     if cross_validate:
                         im_boxes.append(box)
                         im_cls.append(np.where(y==1)[0])
                         ss_accum_score += float(avg_score)
                     else:
                         continue
                  else:
                     discard_num += 1
                     continue
              else: # AL process
                  # add image sample to al candidate
                  al_candidate_idx.append(unlabeledidx[i])
                  im_boxes=[]
                  im_cls = []
                  break
          # replace the fake ground truth for the ss_candidate                                                         
          if len(im_boxes) != 0:
              ss_avg_score.append(ss_accum_score/len(img_boxes))
              ss_candidate_idx.append(unlabeledidx[i])
              overlaps = np.zeros((len(im_boxes), imdb.num_classes), dtype=np.float32)
              for i in range(len(im_boxes)):
                  overlaps[i, im_cls[i]]=1.0
              overlaps = scipy.sparse.csr_matrix(overlaps)
              ss_fake_gt.append({'boxes':np.array(im_boxes),'gt_classes':np.array(im_cls,dtype=np.int).flatten(),'gt_overlaps':overlaps, 'flipped':False})
      if (args.enable_al and len(al_candidate_idx)<=10) or iters_sum>args.max_iters:
          print ("all process finish at loop ",loopcounter)
          print ('the num of al_candidate :',len(al_candidate_idx)) 
          print ("total traning process stop at ",iters_sum)
          break
      al_candidate_idx.extend(al_idx)
      # 50% enter AL
      r = np.random.rand(len(al_candidate_idx))
      al_candidate_idx = [x for i,x in enumerate(al_candidate_idx) if r[i]>0.5]
      # re-rank according to consistency-score
      ss_avg_score = np.reshape(np.array(ss_avg_score),(-1,))
      ss_avg_idx = np.argsort(ss_avg_score)[::-1]
      ss_candidate_idx = [ss_candidate_idx[i] for i in ss_avg_idx]

      if args.enable_al:
          # control al proportion
          if al_num + len(al_candidate_idx)>= al_proportion_checkpoint[0]:
              al_candidate_idx = al_candidate_idx[:int(al_proportion_checkpoint[0]-al_num)]
              tmp = al_proportion_checkpoint.pop(0)
              al_proportion_checkpoint.append(tmp)
              logging.info('al proportion:{}%%,model name:{}'.format(tmp/imdb.num_images,pretrained_model_name))
          print ('samples for AL:',len(al_candidate_idx))
      else:
          al_candidate_idx = []
      if args.enable_ss:
          # control ss proportion
          print('ss_num:',ss_num,'ss_candidate_idx:',len(ss_candidate_idx),'ss_proportion_checkpoint:',ss_proportion_checkpoint[0])
          if ss_num+len(ss_candidate_idx)>=ss_proportion_checkpoint[0]:
              ss_candidate_idx = ss_candidate_idx[:int(ss_proportion_checkpoint[0]-ss_num)]
              ss_fake_gt = ss_fake_gt[:int(ss_proportion_checkpoint[0]-ss_num)]
              tmp = ss_proportion_checkpoint.pop(0)
              ss_proportion_checkpoint.append(tmp)
              print ('ss_proportion_checkpoint: {}%% samples for al, model name:{}'.format(tmp/initial_num,pretrained_model_name ))
          print ('samples for SS:',len(ss_candidate_idx))
      else:
          ss_candidate_idx = []
          ss_fake_gt = []
      # record the proportion of al and ss
      al_num += len(al_candidate_idx)
      ss_num += len(ss_candidate_idx)
      ss_factor = float(ss_num/initial_num)
      al_factor = float(al_num/initial_num)
      total_process_num = al_num + ss_num + discard_num
      logging.info('last model name :{},AL amount:{}/{},al_factor:{},SS amount: {}/{},ss_factor:{},total_process_num:{}'.format(pretrained_model_name,al_num,initial_num,al_factor,ss_num,initial_num,ss_factor,total_process_num))

      # generate training set for next loop  
      for idx in al_candidate_idx:
          bitmapImdb.set(idx)
      next_train_idx = bitmapImdb.nonzero()
      next_train_idx.extend(ss_candidate_idx)
      # update the roidb with ss_fake_gt
      train_roidb = update_training_roidb(imdb,ss_candidate_idx,ss_fake_gt)

      next_train_idx = [x + initialnum for x in next_train_idx if x>=initialnum] + [x + initialnum + remainnum for x in next_train_idx if x>=initialnum]

      train_idx = list(np.arange(initialnum*2))
      next_train_idx.extend(train_idx)

      train_roidb = [train_roidb[i] for i in next_train_idx]
      print('next train roidb :',len(train_roidb)) 
      # stop condition
      loopcounter +=1
      if iters_sum<=tao:
          clslambda=0.9*clslambda - 0.1*np.log(softmax(cls_loss_sum/(cls_sum+1e-30)))
          gamma = min(gamma+0.05,1)
          cls_loss_sum = 0.0        
      train_iters = 20000
      iters_sum += train_iters
      sw.update_roidb(train_roidb)
      sw.train_model(iters_sum)

################################## end ################################################

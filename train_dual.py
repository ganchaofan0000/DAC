from __future__ import division, absolute_import
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import argparse
import torch.optim as optim
import time
import random
from sklearn.preprocessing import normalize
from models.image_encoder import Img_encoder, HeadNet_dual_fused
from models.pt_encoder import Pt_encoder
from tools.dataset import ModelNet_Dataset
from tools.utils import calculate_accuracy
from losses.cross_modal_loss import CrossModalLoss
from losses.CCL_loss import CCL_log_loss
# from losses.sup_cross_modal_loss import SMG_dual_loss
from tools.utils import MovingAverage, Divide_distribution_dual
import warnings
import scipy
from tools.utils import Accstorage
warnings.filterwarnings('ignore',category=FutureWarning)
from torch.utils.tensorboard import SummaryWriter

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

class GCELoss(torch.nn.Module):
    def __init__(self, num_classes=40, q=0.7, gpu=None):
        super(GCELoss, self).__init__()
        # self.device = torch.device('cuda:%s'%gpu) if gpu else torch.device('cpu')
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().cuda()
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()

def compute_weight(img_pred, pt_pred):
    joint_pred = torch.argmax(pt_pred+img_pred, dim=1)
    img_pred = torch.argmax(img_pred, dim=1)
    pt_pred = torch.argmax(pt_pred, dim=1)
    # joint_pred = torch.argmax(pt_pred+img_pred, dim=1)
    weight_mask = torch.full((joint_pred.shape[0],),0.5).cuda()
    ###high credits:
    high_credits = (img_pred==pt_pred)&(joint_pred==pt_pred)
    low_credits = (img_pred!=pt_pred)&(joint_pred!=pt_pred)&(joint_pred!=img_pred)
    weight_mask[high_credits]=1.0
    weight_mask[low_credits]=0.1
    return weight_mask
    
def training_selected(args,
             epoch, 
             img_net, 
             pt_net,
             model,
             train_trainloader_clean,
             train_trainloader_noisy,
             optimizer_img, 
             optimizer_pt, 
             optimizer_model, 
             optimizer_cmc, 
             cls_criterion, 
             inst_criterion, 
             sem_criterion,
             MA_epoch,
             writer,
             iteration,
             Acc_dict):


    pt_net.train()
    # mesh_net.train()
    model.train()
    img_net.train()
    noisy_train_iter = iter(train_trainloader_noisy)
    start_time = time.time()
    corrected_labels=[]
    img_corrected_labels=[]
    pt_corrected_labels=[]
    joint_corrected_labels=[]
    
    gt_labels=[]
    
    
    for data in train_trainloader_clean:
        img_list_c, pt_feat_c, target_c, target_ori_c, indices_c = data
        
        if target_c.shape[0]==1:
            continue
        
        try:
            img_list_n, pt_feat_n, target_n, target_ori_n, indices_n = noisy_train_iter.next()
            assert target_n.shape[0] != 1
        except:
            noisy_train_iter = iter(train_trainloader_noisy)
            img_list_n, pt_feat_n, target_n, target_ori_n, indices_n = noisy_train_iter.next()
        
        optimizer_img.zero_grad()
        optimizer_pt.zero_grad()
        # optimizer_mesh.zero_grad()
        optimizer_model.zero_grad()
        optimizer_cmc.zero_grad()
        
        
        
        img_feat_c, img_feat_v_c = img_list_c[0], img_list_c[1] 
        img_feat_c = Variable(img_feat_c).to(torch.float32).to('cuda')
        img_feat_v_c = Variable(img_feat_v_c).to(torch.float32).to('cuda')
        pt_feat_c = Variable(pt_feat_c).to(torch.float32).to('cuda')
        # pt_feat_c = pt_feat_c.permute(0,2,1)
        target_c = Variable(target_c).to(torch.long).to('cuda')
        target_ori_c = Variable(target_ori_c).to(torch.long).to('cuda')
        
        img_feat_n, img_feat_v_n = img_list_n[0], img_list_n[1] 
        img_feat_n = Variable(img_feat_n).to(torch.float32).to('cuda')
        img_feat_v_n = Variable(img_feat_v_n).to(torch.float32).to('cuda')
        pt_feat_n = Variable(pt_feat_n).to(torch.float32).to('cuda')
        # pt_feat_n = pt_feat_n.permute(0,2,1)
        target_n = Variable(target_n).to(torch.long).to('cuda')
        target_ori_n = Variable(target_ori_n).to(torch.long).to('cuda')
        
        img_feat = torch.cat((img_feat_c, img_feat_n), dim=0)
        img_feat_v = torch.cat((img_feat_v_c, img_feat_v_n), dim=0)
        
        
        pt_feat = torch.cat((pt_feat_c, pt_feat_n), dim=0)
        target = torch.cat((target_c, target_n), dim=0)
        target_ori = torch.cat((target_ori_c, target_ori_n), dim=0)
        indices = torch.cat((indices_c, indices_n), dim=0)
        
        clean_num = img_feat_c.shape[0]
        noisy_num = img_feat_n.shape[0]
        mask_clean = torch.arange(clean_num+noisy_num, dtype=torch.int32).cuda()
        mask_clean = mask_clean < clean_num
         
        ##noisy label
        y_clean = torch.nn.functional.one_hot(target_c, args.num_classes).float().cuda()
        
        y_n = torch.nn.functional.one_hot(target, args.num_classes).float().cuda()
        _img_feat, _pt_feat = img_net(img_feat, img_feat_v), pt_net(pt_feat)
        _img_pred, _pt_pred, _joint_pred = model(_img_feat, _pt_feat, jointed=args.jointed)
        
        ##linear pred label
        y_img_pred = F.softmax(_img_pred, dim=1)
        img_pred_labels = torch.argmax(y_img_pred, dim=1)
        y_pt_pred = F.softmax(_pt_pred, dim=1)
        pt_pred_labels = torch.argmax(y_pt_pred, dim=1)
        
        y_joint_pred = F.softmax(_joint_pred, dim=1)
        joint_pred_labels = torch.argmax(y_joint_pred, dim=1)
        
        
        ###weighted correction
        weight_mask = None
        if args.weighted:
            with torch.no_grad():
                weight_mask = compute_weight(y_img_pred, y_pt_pred)
                weight_mask = weight_mask[:,None].cpu()
                # print(weight_mask)
        
        
        if args.modal_name=="joint":
            y_hat = y_joint_pred
        elif args.modal_name=="img":
            y_hat = y_img_pred
        elif args.modal_name=="pt":
            y_hat = y_pt_pred
        else:
            y_hat = (y_img_pred+y_pt_pred)/2
        
        ###moving-average
        if args.ma:
            with torch.no_grad():
                y_e = y_hat.clone().detach().cpu()
                # print(y_e.shape)
                y_hat = MA_epoch.update(y_e, indices,epoch, weight = weight_mask)
                y_hat = y_hat.cuda()
        
        if not args.corrected:
            y_final_one_hot = y_n
            y_final = target
            w = None
        else:
            y_final_one_hot = y_hat
            y_final_one_hot[mask_clean,:] = y_clean
            y_final = torch.argmax(y_hat, dim=1)
            w = None
        
        ##clean set
        
        _pt_pred_s = _pt_pred[mask_clean]
        _img_pred_s = _img_pred[mask_clean]
        _joint_pred_s = _joint_pred[mask_clean]
        _img_feat_s = _img_feat[mask_clean]
        _pt_feat_s = _pt_feat[mask_clean]
        y_final_one_hot_s = y_final_one_hot[mask_clean]
        y_final_s = y_final[mask_clean]
        prior = torch.ones(args.num_classes)/args.num_classes
        prior = prior.cuda()
        # regularization     
        pt_pred_mean = torch.softmax(_pt_pred_s, dim=1).mean(0)
        pt_penalty = torch.sum(prior*torch.log(prior/pt_pred_mean))
        pt_cls_loss = -torch.mean(torch.sum(F.log_softmax(_pt_pred_s, dim=1) * y_final_one_hot_s, dim=1)) + pt_penalty
        
        # regularization     
        img_pred_mean = torch.softmax(_img_pred_s, dim=1).mean(0)
        img_penalty = torch.sum(prior*torch.log(prior/img_pred_mean))
        img_cls_loss = -torch.mean(torch.sum(F.log_softmax(_img_pred_s, dim=1) * y_final_one_hot_s, dim=1)) + img_penalty
        
        # regularization     
        joint_pred_mean = torch.softmax(_joint_pred_s, dim=1).mean(0)
        joint_penalty = torch.sum(prior*torch.log(prior/joint_pred_mean))
        joint_crc_loss = -torch.mean(torch.sum(F.log_softmax(_joint_pred_s, dim=1) * y_final_one_hot_s, dim=1)) + joint_penalty
        
        if not args.uni_loss:
            cls_loss_c = (pt_cls_loss + img_cls_loss + 2*joint_crc_loss)/2
        elif args.modal_name=="img":
            cls_loss_c = img_cls_loss
        elif args.modal_name=="pt":
            cls_loss_c = pt_cls_loss
    
        
        
        
        sem_loss_c, centers = sem_criterion(torch.cat((_img_feat_s, _pt_feat_s), dim = 0), torch.cat((y_final_s, y_final_s), dim = 0),  epoch)
        
        inst_loss_c = inst_criterion(torch.cat((_img_feat_s, _pt_feat_s), dim = 0))
        
        loss_c = args.w_cls_c * cls_loss_c + args.w_sem_c * sem_loss_c + args.w_inst_c * inst_loss_c
        
        _pt_pred_s = _pt_pred[~mask_clean]
        _img_pred_s = _img_pred[~mask_clean]
        _joint_pred_s = _joint_pred[~mask_clean]
        _img_feat_s = _img_feat[~mask_clean]
        _pt_feat_s = _pt_feat[~mask_clean]
        y_final_one_hot_s = y_final_one_hot[~mask_clean]
        y_final_s = y_final[~mask_clean]
        
        
        pt_pred_mean = torch.softmax(_pt_pred_s, dim=1).mean(0)
        pt_penalty = torch.sum(prior*torch.log(prior/pt_pred_mean))
        pt_cls_loss = -torch.mean(torch.sum(F.log_softmax(_pt_pred_s, dim=1) * y_final_one_hot_s, dim=1)) + pt_penalty
        
        # regularization     
        img_pred_mean = torch.softmax(_img_pred_s, dim=1).mean(0)
        img_penalty = torch.sum(prior*torch.log(prior/img_pred_mean))
        img_cls_loss = -torch.mean(torch.sum(F.log_softmax(_img_pred_s, dim=1) * y_final_one_hot_s, dim=1)) + img_penalty
        
        # regularization     
        joint_pred_mean = torch.softmax(_joint_pred_s, dim=1).mean(0)
        joint_penalty = torch.sum(prior*torch.log(prior/joint_pred_mean))
        joint_crc_loss = -torch.mean(torch.sum(F.log_softmax(_joint_pred_s, dim=1) * y_final_one_hot_s, dim=1)) + joint_penalty
        
        if not args.uni_loss:
            cls_loss_n = (pt_cls_loss + img_cls_loss + 2*joint_crc_loss)/2
        elif args.modal_name=="img":
            cls_loss_n = img_cls_loss
        elif args.modal_name=="pt":
            cls_loss_n = pt_cls_loss
    
        
        
        
        sem_loss_n, centers = sem_criterion(torch.cat((_img_feat_s, _pt_feat_s), dim = 0), torch.cat((y_final_s, y_final_s), dim = 0),  epoch)
        
        inst_loss_n = inst_criterion(torch.cat((_img_feat_s, _pt_feat_s), dim = 0))
        
        loss_n = args.w_cls_n * cls_loss_n + args.w_sem_n * sem_loss_n + args.w_inst_n * inst_loss_n
        
        loss = loss_c + args.wn*loss_n
        
        
        loss.backward()

        optimizer_pt.step()
        optimizer_img.step()
        # optimizer_mesh.step()
        optimizer_model.step()

        optimizer_cmc.step()
        
        img_acc = calculate_accuracy(y_img_pred, target_ori)
        pt_acc = calculate_accuracy(y_pt_pred, target_ori)
        # mesh_acc = calculate_accuracy(y_mesh_pred, target_ori)
        joint_pred = (y_img_pred+y_pt_pred)/2
        joint_acc = calculate_accuracy(joint_pred, target_ori)
        joint_m_acc = calculate_accuracy(y_joint_pred, target_ori)
        
        ema_acc = calculate_accuracy(y_final_one_hot, target_ori)
        
        ####corrected acc
        img_acc_n = calculate_accuracy(y_img_pred[~mask_clean], target_ori[~mask_clean])
        pt_acc_n = calculate_accuracy(y_pt_pred[~mask_clean], target_ori[~mask_clean])
        # mesh_acc_n = calculate_accuracy(y_mesh_pred[~mask_clean], target_ori[~mask_clean])
        joint_acc_n = calculate_accuracy(joint_pred[~mask_clean], target_ori[~mask_clean])
        # mesh_acc_n = calculate_accuracy(y_mesh_pred[~mask_clean], target_ori[~mask_clean])
        joint_acc_c = calculate_accuracy(joint_pred[mask_clean], target_ori[mask_clean])
        ema_acc_n = calculate_accuracy(y_final_one_hot[~mask_clean], target_ori[~mask_clean])
        
        

        writer.add_scalar('Loss/loss', loss.item(), iteration)
        
        

        if iteration % args.per_print == 0:
            print("[%d]  img_acc: %.4f  pt_acc: %.4f  joint_acc: %.4f jm_acc: %.4f ema_acc: %.4f ema_acc_n: %f" % (iteration, img_acc, pt_acc, joint_acc, joint_m_acc, ema_acc, ema_acc_n))
            
            
            start_time = time.time()
            

        iteration = iteration + 1
        corrected_labels.extend(y_final[~mask_clean].clone().detach().cpu().numpy().tolist())
        img_corrected_labels.extend(img_pred_labels[~mask_clean].clone().detach().cpu().numpy().tolist())
        pt_corrected_labels.extend(pt_pred_labels[~mask_clean].clone().detach().cpu().numpy().tolist())
        joint_corrected_labels.extend(joint_pred_labels[~mask_clean].clone().detach().cpu().numpy().tolist())
        
        gt_labels.extend(target_ori[~mask_clean].clone().detach().cpu().numpy().tolist())
        
        # target_ori[~mask_clean]
    
    corrected_labels = np.array(corrected_labels)
    img_corrected_labels = np.array(img_corrected_labels)
    pt_corrected_labels = np.array(pt_corrected_labels)
    joint_corrected_labels = np.array(joint_corrected_labels)
    
    gt_labels = np.array(gt_labels)
    
    corrected_acc = (corrected_labels==gt_labels).sum()/gt_labels.shape[0]
    
    img_corrected_acc = (img_corrected_labels==gt_labels).sum()/gt_labels.shape[0]
    
    pt_corrected_acc = (pt_corrected_labels==gt_labels).sum()/gt_labels.shape[0]
    
    joint_corrected_acc = (joint_corrected_labels==gt_labels).sum()/gt_labels.shape[0]
    
    
    all_acc = {'img_acc': img_acc,
                    'pt_acc': pt_acc,
                    'joint_acc':joint_acc,
                    'ema_acc': ema_acc,
                    'corrected_acc': corrected_acc,
                    'img_corrected_acc': img_corrected_acc,
                    'pt_corrected_acc': pt_corrected_acc,
                    'joint_corrected_acc': joint_corrected_acc
                    }
    Acc_dict.add(all_acc)
    # writer.add_scalar('Acc/corrected_acc_epoch',corrected_acc, epoch)
    
    writer.add_scalars('Acc/all_corrected_acc',
                          {'corrected_acc': corrected_acc,
                            'img_corrected_acc': img_corrected_acc,
                            'pt_corrected_acc': pt_corrected_acc,
                            'joint_corrected_acc': joint_corrected_acc
                           }, epoch)
    return iteration


def warmup(args,
             epoch,
             img_net, 
             pt_net,
             model, 
             train_trainloader,
             optimizer_img, 
             optimizer_pt, 
             optimizer_model,
             optimizer_centloss, 
             cls_criterion, 
             inst_criterion, 
             sem_criterion,
             writer):

    pt_net.train(True)
    model.train(True)
    img_net.train(True)
    # mesh_net.train(True)
    
    conf_penalty = NegEntropy()
    
    
    
    iteration = epoch*len(train_trainloader)
    iteration_all = args.epochs*len(train_trainloader)
    start_time = time.time()
    
    for data in train_trainloader:
        # image, point cloud, noisy labels, original labels (True labels for val.).
        img_list, pt_feat, target, target_ori, indices = data
        
        img_feat, img_feat_v = img_list[0], img_list[1] 
        
        img_feat = Variable(img_feat).to(torch.float32).to('cuda')
        img_feat_v = Variable(img_feat_v).to(torch.float32).to('cuda')
        
        
        pt_feat = Variable(pt_feat).to(torch.float32).to('cuda')
        # pt_feat = pt_feat.permute(0,2,1)
        target = Variable(target).to(torch.long).to('cuda')

        optimizer_img.zero_grad()
        optimizer_pt.zero_grad()
        # optimizer_mesh.zero_grad()
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        
        _img_feat, _pt_feat = img_net(img_feat, img_feat_v), pt_net(pt_feat)
        # _mesh_feat = mesh_net(centers, corners, normals, neighbor_index)
        _img_pred, _pt_pred, _joint_pred = model(_img_feat, _pt_feat, jointed=args.jointed)
        # print(_joint_pred.shape)
        # compute loss
        pt_cls_loss = cls_criterion(_pt_pred, target)
        img_cls_loss = cls_criterion(_img_pred, target)
        joint_cls_loss = cls_criterion(_joint_pred, target)
        
        # mesh_ce_loss = ce_criterion(_mesh_pred, target)
        pt_penalty = conf_penalty(_img_pred)
        img_penalty = conf_penalty(_pt_pred)
        joint_penalty = conf_penalty(_joint_pred)
        
        
        penalty = pt_penalty+img_penalty+joint_penalty
        
        if not args.uni_loss:
            cls_loss = (pt_cls_loss + img_cls_loss+2*joint_cls_loss)/2
        elif args.modal_name=="img":
            cls_loss = img_cls_loss
        elif args.modal_name=="pt":
            cls_loss = pt_cls_loss
        
        
        # sup_feat = torch.cat([_img_feat, _pt_feat], dim=0)
        # # sup_feat = torch.cat([_img_feat_proj, _pt_feat_proj], dim=0)
        # sup_feat = torch.nn.functional.normalize(sup_feat, dim=-1)

        sem_loss, centers = sem_criterion(torch.cat((_img_feat, _pt_feat), dim = 0), torch.cat((target, target), dim = 0), epoch)
        
        inst_loss = inst_criterion(torch.cat((_img_feat, _pt_feat), dim = 0))
        
        if args.noise_type=="asy":
            loss = args.w_cls_c * (cls_loss+penalty) + args.w_sem_c * sem_loss + args.w_inst_c * inst_loss
        else:
            loss = args.w_cls_c * cls_loss + args.w_sem_c * sem_loss + args.w_inst_c * inst_loss
        
        
        

        loss.backward()

        # optimizer_head.step()
        optimizer_img.step()
        optimizer_pt.step()
        # optimizer_mesh.step()
        optimizer_model.step()

        optimizer_centloss.step()

        writer.add_scalar('Loss/loss', loss.item(), iteration)
        if iteration % args.per_print == 0:
            print("[%d/%d]  loss: %f  sem_loss: %f  cls_loss: %f  inst_loss: %f time: %f" % (iteration, iteration_all, loss.item(), sem_loss.item(), cls_loss.item(), inst_loss.item(), time.time()-start_time))
            start_time = time.time()
            

        iteration = iteration + 1
        
    return iteration


def fx_calc_map_label(view_1, view_2, label_test):
    dist = scipy.spatial.distance.cdist(view_1, view_2, 'cosine') #rows view_1 , columns view 2 
    ord = dist.argsort()
    # print("sort dist finished.....")
    numcases = dist.shape[0]
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(numcases):
            if label_test[i] == label_test[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)


def test(args,
        epoch,
        img_net, 
        pt_net, 
        eval_loader):
    
    pt_net.eval()
    # head_net.eval()
    img_net.eval()
    # batch_size = args.eval_batch_size
    img_feat_list_2 = np.zeros((len(eval_loader.dataset), 256))
    img_feat_list_4 = np.zeros((len(eval_loader.dataset), 256))
    
    pt_feat_list = np.zeros((len(eval_loader.dataset), 256))
    label = np.zeros((len(eval_loader.dataset)))
    #################################
    iteration = 0
    for data in eval_loader:
        img_list, pt_feat, noisy_label, ori_label, indices = data
        batch_size = ori_label.shape[0]
        
        img_feat1, img_feat2, img_feat3, img_feat4 = img_list    
        img_feat1 = Variable(img_feat1).to(torch.float32).to('cuda')
        img_feat2 = Variable(img_feat2).to(torch.float32).to('cuda')
        img_feat3 = Variable(img_feat3).to(torch.float32).to('cuda')
        img_feat4 = Variable(img_feat4).to(torch.float32).to('cuda')
        
        pt_feat = Variable(pt_feat).to(torch.float32).to('cuda')
        # pt_feat = pt_feat.permute(0,2,1)

        ori_label = Variable(ori_label).to(torch.long).to('cuda')
        ##########################################
        img_net = img_net.to('cuda')
        _img_feat1 = img_net(img_feat1, img_feat2)
        _img_feat2 = img_net(img_feat3, img_feat4)
        pt_net = pt_net.to('cuda')
        
        _pt_feat = pt_net(pt_feat)
        
        feat_4_views = 0.5*(_img_feat1+_img_feat2)
        img_feat_list_2[iteration:iteration+batch_size, :] = _img_feat1.data.cpu().numpy()
        img_feat_list_4[iteration:iteration+batch_size, :] = feat_4_views.data.cpu().numpy()
        
        pt_feat_list[iteration:iteration+batch_size, :] = _pt_feat.data.cpu().numpy()
        label[iteration:iteration+batch_size] = ori_label.data.cpu().numpy()
        iteration = iteration+batch_size
    
    img_eval = normalize(img_feat_list_2, norm='l1', axis=1)
    pt_eval = normalize(pt_feat_list, norm='l1', axis=1)
    
    i2p_acc = fx_calc_map_label(img_eval,pt_eval, label)
    i2p_acc = round(i2p_acc*100,2)
    
    p2i_acc = fx_calc_map_label(pt_eval,img_eval, label)
    p2i_acc = round(p2i_acc*100,2)
    with open('warm_up.txt','a+') as file0:
        print("View 2, Eval Epoch%s: I2P:%.2f, P2I:%.2f"%(epoch, i2p_acc, p2i_acc), file=file0)
    print("View 2, Eval Epoch%s: I2P:%.2f, P2I:%.2f"%(epoch, i2p_acc, p2i_acc))
    
    img_eval = normalize(img_feat_list_4, norm='l1', axis=1)
    
    i2p_acc = fx_calc_map_label(img_eval,pt_eval, label)
    i2p_acc = round(i2p_acc*100,2)
    
    p2i_acc = fx_calc_map_label(pt_eval,img_eval, label)
    p2i_acc = round(p2i_acc*100,2)
    with open('warm_up.txt','a+') as file0:
        print("View 4, Eval Epoch%s: I2P:%.2f, P2I:%.2f"%(epoch, i2p_acc, p2i_acc), file=file0)
    print("View 4, Eval Epoch%s: I2P:%.2f, P2I:%.2f"%(epoch, i2p_acc, p2i_acc))
    
    return i2p_acc, p2i_acc


def training(args):
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    img_net = Img_encoder(pre_trained = None)
    pt_net = Pt_encoder(args)
    model = HeadNet_dual_fused(num_classes=args.num_classes,fused_type=args.fused_type)
    
    img_net.train(True)
    img_net = img_net.to('cuda')
    pt_net.train(True)
    pt_net = pt_net.to('cuda')
    model.train(True)
    model = model.to('cuda')
    

    writer = SummaryWriter(os.path.join(args.save, 'summary'))
    
    #cross entropy loss for classification
    cls_criterion = nn.CrossEntropyLoss()
    
    sem_criterion = CCL_log_loss(num_classes=args.num_classes, feat_dim=256, temperature=args.center_temp)
    

    inst_criterion = CrossModalLoss(modal_num=2)
    
    optimizer_img = optim.Adam(img_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_pt = optim.Adam(pt_net.parameters(), lr=args.lr_pt, weight_decay=args.weight_decay)
    # optimizer_mesh = optim.Adam(mesh_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_centloss = optim.Adam(sem_criterion.parameters(), lr=args.lr_center)

    train_set = ModelNet_Dataset(dataset = args.dataset, num_points = args.num_points, num_classes=args.num_classes, 
                                 dataset_dir=args.dataset_dir,  partition='train', 
                                 noise_rate=args.noise_rate, noise_type=args.noise_type, use_cross=args.use_cross)
    eval_set = ModelNet_Dataset(dataset=args.dataset, num_points = args.num_points, num_classes=args.num_classes,
                                  dataset_dir=args.dataset_dir, partition="test", 
                                  noise_rate=args.noise_rate, noise_type=args.noise_type, use_cross=args.use_cross)

    data_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    eval_data_loader = torch.utils.data.DataLoader(eval_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=1)
    
    print("train batch number:%s"%(len(data_loader)))
    print("eval batch number:%s"%(len(eval_data_loader)))
    ###MovingAverage
    MA_epoch = MovingAverage(num_data=train_set.__len__(), num_classes=args.num_classes, beta=args.beta, warm_up=args.warm_up)
    MA_epoch.initial_bank(data_loader)
    
    
    Acc_dict = Accstorage()
    ##gt divide
    gt_clean, gt_noisy = train_set.get_gt_divide()
    gt_clean, gt_noisy = torch.from_numpy(gt_clean), torch.from_numpy(gt_noisy)
    best_I2P = 0.0
    best_P2I = 0.0
    clean_indices, noisy_indices=[],[]
    clean_scores = []
    
    warmup_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    print("num batch: ",len(warmup_loader))
    print('Warmuping Network......')
    
    iteration = 0
    start_time = time.time()
    
    divide_acc_list = torch.zeros(args.epochs)
    divide_rec_list = torch.zeros(args.epochs)
    data_acc_list = torch.zeros(args.epochs)
    i2p_epochs = torch.zeros(args.epochs)
    p2i_epochs = torch.zeros(args.epochs)
    
    for epoch in range(args.epochs):
        
        if epoch == args.warm_up:
            lr = 0.0001
            # lr_pt = args.lr_pt * (0.1 ** (epoch // args.lr_step))
            print('New  Learning Rate:     ' + str(lr))
            for param_group in optimizer_img.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_pt.param_groups:
                param_group['lr'] = lr
            # for param_group in optimizer_mesh.param_groups:
            #     param_group['lr'] = lr
            for param_group in optimizer_model.param_groups:
                param_group['lr'] = lr

        # update the learning rate of the center loss
        if epoch == args.warm_up:
            lr_center = 0.0001
            print('New  Center LR:     ' + str(lr_center))
            for param_group in optimizer_centloss.param_groups:
                param_group['lr'] = lr_center
                
        if (epoch%args.lr_step) == 0:
            lr = args.lr * (0.1 ** (epoch // args.lr_step))
            # lr_pt = args.lr_pt * (0.1 ** (epoch // args.lr_step))
            print('New  Learning Rate:     ' + str(lr))
            for param_group in optimizer_img.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_pt.param_groups:
                param_group['lr'] = lr
            # for param_group in optimizer_mesh.param_groups:
            #     param_group['lr'] = lr
            for param_group in optimizer_model.param_groups:
                param_group['lr'] = lr

        # update the learning rate of the center loss
        if (epoch%args.lr_step) == 0:
            lr_center = args.lr_center * (0.1 ** (epoch // args.lr_step))
            print('New  Center LR:     ' + str(lr_center))
            for param_group in optimizer_centloss.param_groups:
                param_group['lr'] = lr_center
        
        if epoch < args.warm_up:     # warm up  
            # train_set = FeatureDataloader(root_dir="/media/cf/dataset/objaverse/", num_classes=args.num_classes, partition='train')
            
            ###注意warmup的时候采用的是CEloss
            iteration = warmup(args, epoch, img_net, pt_net, model, 
                   warmup_loader, optimizer_img, optimizer_pt, optimizer_model, optimizer_centloss, 
                   cls_criterion, inst_criterion, sem_criterion, writer)
            
        
        else:
            
            if (epoch >= args.warm_up and epoch <= args.warm_up+16) or (epoch%20==0): 
                print('Training Model......')
                
                # print(clean_indices.shape, noisy_indices.shape)
                batch_size_clean = ((clean_indices.shape[0]/(clean_indices.shape[0]+noisy_indices.shape[0]))*args.batch_size//2)*2
                batch_size_clean = max(int(batch_size_clean), args.batch_size//2)
                # batch_size_clean = args.batch_size//2
                
                # batch_size_clean = int(batch_size_clean)
                
                batch_size_noisy = int(args.batch_size - batch_size_clean)
                # batch_size_noisy = args.batch_size
                
                clean_part_rate = batch_size_clean/(batch_size_clean + batch_size_noisy)
                print("clean and noisy batch size:%.2f,%.2f"%(batch_size_clean, batch_size_noisy))
                # print("clean part: %s, noisy part: %s"%(clean_part_rate, 1-clean_part_rate))
                train_set_clean = ModelNet_Dataset(dataset=args.dataset, num_points=args.num_points, num_classes=args.num_classes,
                                    dataset_dir=args.dataset_dir, partition="train", 
                                    noise_rate=args.noise_rate, select_indices=clean_indices, 
                                    weight_scores=None, noise_type=args.noise_type, use_cross=args.use_cross)
                train_loader_clean = torch.utils.data.DataLoader(train_set_clean, batch_size=batch_size_clean, shuffle=True, num_workers=8, drop_last=True)
                
                train_set_noisy = ModelNet_Dataset(dataset=args.dataset, num_points=args.num_points, num_classes=args.num_classes,
                                    dataset_dir=args.dataset_dir, partition="train", 
                                    noise_rate=args.noise_rate, select_indices=noisy_indices, 
                                    weight_scores=None, noise_type=args.noise_type, use_cross=args.use_cross)
                train_loader_noisy = torch.utils.data.DataLoader(train_set_noisy, batch_size=batch_size_noisy, shuffle=True, num_workers=8, drop_last=True)
                
                # print("label num batch: ",len(train_loader_clean))
                # print("unlabel num batch: ",len(train_loader_noisy))
                # args.weight_clean = len(train_loader_clean)/(len(train_loader_clean)+len(train_loader_noisy))
                # args.weight_noisy = 1-args.weight_clean
                # print("clean and noisy learning weight:%.2f,%.2f"%(args.weight_clean, args.weight_noisy))
            ##training use MeanAbsoluteError
            iteration = training_selected(args, epoch, img_net, pt_net, model, 
                   train_loader_clean, train_loader_noisy, optimizer_img, optimizer_pt, optimizer_model, optimizer_centloss, 
                   cls_criterion, inst_criterion, sem_criterion, MA_epoch, writer, iteration, Acc_dict)
        
        if epoch%10==0:
            print('\n==== eval %s epoch ===='%(epoch))
            # if epoch%10==0:
            with torch.no_grad(): 
                i2p_acc, p2i_acc = test(args, epoch, img_net, pt_net, eval_data_loader)
                i2p_epochs[epoch] = i2p_acc
                p2i_epochs[epoch] = p2i_acc
                if i2p_acc>best_I2P:
                    best_I2P = i2p_acc
                    best_P2I = p2i_acc
                    print('----------------- Save The Network ------------------------')
                    with open(args.save + 'best-head_net.pkl', 'wb') as f:
                        torch.save(model, f)
                    with open(args.save + 'best-img_net.pkl', 'wb') as f:
                        torch.save(img_net, f)
                    with open(args.save + 'best-pt_net.pkl', 'wb') as f:
                        torch.save(pt_net, f)
                
            writer.add_scalar('mAP/i2p_acc',i2p_acc, epoch)
            writer.add_scalar('mAP/p2i_acc',p2i_acc, epoch)
        if (epoch >= 0 and epoch <= args.warm_up+10) or (epoch >= args.warm_up-1 and (epoch+1)%20==0): 
            print('\n==== Divide %s epoch training data ===='%(epoch+1)) 
            ###get centers
            cmc_centers = sem_criterion.get_centers()
            eval_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
            scores_ce = Divide_distribution_dual(args, img_net, pt_net, model, epoch, eval_loader, cmc_centers)
            
            clean_scores = torch.from_numpy(scores_ce)
            
            clean_indices = torch.nonzero(clean_scores>args.p_threshold).squeeze()

            # 获取为False的元素的索引
            noisy_indices = torch.nonzero(clean_scores<=args.p_threshold).squeeze()
            
            args.weight_clean = clean_indices.shape[0]/(clean_indices.shape[0]+noisy_indices.shape[0])
            args.weight_noisy = 1-args.weight_clean
            print("clean and noisy learning weight:%.2f,%.2f"%(args.weight_clean, args.weight_noisy))
            
            # print(clean_indices.shape, noisy_indices.shape)
            pred_clean = clean_scores>args.p_threshold
            # num_data = gt_clean.shape[0]
            divide_acc = (((gt_clean == pred_clean)&pred_clean).sum())/pred_clean.sum()
            divide_rec = (((gt_clean == pred_clean)&gt_clean).sum())/gt_clean.sum()
            data_acc = ((gt_clean == pred_clean).sum())/pred_clean.shape[0]
            
            f_score = 2*divide_acc*divide_rec/(divide_rec+divide_acc)
            ##计算divide的acc
            writer.add_scalar('Acc/divide_acc',divide_acc, epoch)
            writer.add_scalar('Acc/divide_rec',divide_rec, epoch)
            writer.add_scalar('Acc/f_score',f_score, epoch)
            writer.add_scalar('Acc/data_acc',data_acc, epoch)
            
            
            ##计算divide的acc
            writer.add_scalar('Acc/divide_acc',divide_acc, epoch)
            print("Divide acc: %.4f"%divide_acc.item())
            print("Divide rec: %.4f"%divide_rec.item())
            print("Data acc: %.4f"%data_acc.item())
            
            clean_rate = clean_indices.shape[0]/(clean_indices.shape[0]+noisy_indices.shape[0])
            noisy_rate = noisy_indices.shape[0]/(clean_indices.shape[0]+noisy_indices.shape[0])
            
            print("\nclean vs noisy: %.3f vs %.3f"%(clean_rate, noisy_rate))
            
            divide_acc_list[epoch]=divide_acc
            divide_rec_list[epoch]=divide_rec
            data_acc_list[epoch]=data_acc
            
            
        
        
        
    
    print('----------------- Save The Network ------------------------')
    with open(os.path.join(args.save, 'last-head_net.pkl'), 'wb') as f:
                    torch.save(model, f)
    with open(os.path.join(args.save, 'last-img_net.pkl'), 'wb') as f:
        torch.save(img_net, f)
    with open(os.path.join(args.save, 'last-pt_net.pkl'), 'wb') as f:
        torch.save(pt_net, f)
    
    
    with open('%s_ab_p.txt'%args.dataset,'a+') as file0:
        print("Noise rate: %.2f, Noise type: %s, Save: %s"%(args.noise_rate, args.noise_type, args.save), file=file0)
        print("Best Acc: %.2f, %.2f"%(best_I2P, best_P2I), file=file0)
    print("Best Acc: %.2f, %.2f"%(best_I2P, best_P2I))
    
    # Accstorage.save_acc(args.save)
    np.save(os.path.join(args.save, 'divide-acc.npy'), divide_acc_list.numpy())
    np.save(os.path.join(args.save, 'divide-rec.npy'), divide_rec_list.numpy())
    np.save(os.path.join(args.save, 'data-acc.npy'), data_acc_list.numpy())
    np.save(os.path.join(args.save, 'i2p_acc.npy'), i2p_epochs.numpy())
    np.save(os.path.join(args.save, 'p2i_acc.npy'), p2i_epochs.numpy())
    
    Acc_dict.save_acc(save_dir=args.save)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Cross Modal Retrieval for Point Cloud, Mesh, and Image Models')

    parser.add_argument('--dataset', type=str, default='ModelNet40', metavar='dataset',
                        help='ModelNet10 or ModelNet40')

    parser.add_argument('--dataset_dir', type=str, default='/mnt/sdb/chaofan/data/ModelNet40/', metavar='dataset_dir',
                        help='dataset_dir')

    parser.add_argument('--num_classes', type=int, default=40, metavar='num_classes',
                        help='10 or 40')

    parser.add_argument('--batch_size', type=int, default=128, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--eval_batch_size', type=int, default=20, metavar='batch_size',
                        help='Size of batch)')
    
    parser.add_argument('--bank_size', type=int, default=16, metavar='batch_size',
                    help='Size of batch)')

    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of episode to train ')
    
    parser.add_argument('--warm_up', type=int, default=10, metavar='N',
                        help='number of episode to train ')
    #optimizer
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    
    parser.add_argument('--lr_pt', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')

    parser.add_argument('--lr_step', type=int,  default=100,
                        help='how many iterations to decrease the learning rate')

    parser.add_argument('--lr_center', type=float, default=0.001, metavar='LR',
                        help='learning rate for center loss (default: 0.5)')
                                         
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')

    #DGCNN
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings') 
    
    parser.add_argument('--mix_alpha', type=float, default=5.0, metavar='mix_alpha',
                        help='mix_alpha' )  

    #loss
    parser.add_argument('--w_sem_c', type=float, default=1, metavar='weight_center',
                        help='weight center (default: 1.0)')

    parser.add_argument('--w_cls_c', type=float, default=0.1, metavar='weight_ce',
                        help='weight ce' ) 

    parser.add_argument('--w_inst_c', type=float, default=0.1, metavar='weight_mse',
                        help='weight mse' )
    
    parser.add_argument('--w_sem_n', type=float, default=1, metavar='weight_center',
                        help='weight noisy center (default: 1.0)')

    parser.add_argument('--w_cls_n', type=float, default=1, metavar='weight_ce',
                        help='weight cls (noisy)' ) 
    
    parser.add_argument('--wn', type=float, default=1, metavar='weight_ce',
                        help='weight S_n' ) 

    parser.add_argument('--w_inst_n', type=float, default=10, metavar='weight_mse',
                        help='weight inst (noisy)' )

    parser.add_argument('--weight_decay', type=float, default=1e-3, metavar='weight_decay',
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--per_save', type=int,  default=50,
                        help='how many iterations to save the model')

    parser.add_argument('--per_print', type=int,  default=20,
                        help='how many iterations to print the loss and accuracy')
                        
    parser.add_argument('--save', type=str,  default='./checkpoints_our/ModelNet40/',
                        help='path to save the final model')
    
    parser.add_argument('--model_path', type=str, default='/mnt/sdb/chaofan/project/CrossPoint/dgcnn_cls_best.pth', metavar='N',
                        help='Pretrained model path')

    parser.add_argument('--gpu_id', type=str,  default='0,1',
                        help='GPU used to train the network')
    
    parser.add_argument('--pretrained', action='store_true', help='use pretrained')
    
    parser.add_argument('--weight_clean', type=float, default=1, metavar='weight_clean',
                        help='weight clean' )   # 0.1

    parser.add_argument('--weight_noisy', type=float, default=1, metavar='weight_noisy',
                        help='weight noisy' ) #20 - 10   50
    parser.add_argument('--mixup', action='store_true', help='features mixup for center classifier')
    parser.add_argument('--linear_mixup', action='store_true', help='features mixup for linear classifier')
    parser.add_argument('--bank_mixup', action='store_true', help='features mixup for center classifier')
    
    parser.add_argument('--corrected', action='store_true', help='GPU used to train the network')
    parser.add_argument('--ma', action='store_true', help='epoch moving-average')
    
    parser.add_argument('--noise_rate', type=float, default=0.0, metavar='noise_rate',
                        help='noise rate')
    
    parser.add_argument('--center_temp', type=float, default=0.07, metavar='noise_rate',
                        help='noise rate')
    
    parser.add_argument('--beta', type=float, default=0.9, metavar='moving average beta',
                        help='weight mse' )
    
    parser.add_argument('--noise_type', type=str, default='sym',
                        help='sym or asy')

    parser.add_argument('--log', type=str,  default='log/',
                        help='path to the log information')
    
    parser.add_argument('--pretrained_path', type=str,  default=None,
                        help='path to the log information')
    
    parser.add_argument('--modal_name', type=str,  default='joint',
                        help='joint, img, mesh, pt')
    
    parser.add_argument('--fused_type', type=str,  default=None,
                        help='add, concat, attention')
    
    parser.add_argument('--use_cross', action='store_true', help='features mixup for center classifier')
    parser.add_argument('--weighted', action='store_true', help='weighted noisy sample')
    
    parser.add_argument('--jointed', action='store_true', help='weighted noisy sample')
    parser.add_argument('--uni_loss', action='store_true', help='weighted noisy sample')
    
    
    
    

    args = parser.parse_args()
    seed=2000
    random.seed(seed)
    torch.manual_seed(seed)
    print(args)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # torch.backends.cudnn.enabled = False
    training(args)

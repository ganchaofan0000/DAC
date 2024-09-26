import torch
import torch.nn as nn
import math
import torch.nn.functional as F


    
class CCL_log_loss(nn.Module):
    
    def __init__(self, num_classes = 10, feat_dim=256, warmup=15, temperature=0.07):
        super(CCL_log_loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.warmup = warmup
        
        self.temperature = temperature
        self.alpha=0.5
        self.a=0.2
        
        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

    def forward(self, features, labels, w=None, weight=None):
        
        # print(x.shape, labels.shape)
        # batch_size = features.size(0)
        
        labels = labels.contiguous().view(-1, 1)
        features = F.normalize(features, p=2, dim=1)
        
        centers = self.centers
        centers = F.normalize(centers, p=2, dim=1)
        
        centers_labels = torch.arange(self.num_classes, dtype=labels.dtype).cuda()
        
        mask = torch.eq(labels, centers_labels.T).float().cuda()
        # print(mask.shape)
        
        anchor_dot_contrast = torch.div(
            torch.matmul(features, centers.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        
        # mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)
        # mask_sum = torch.where(mask_sum == 0, torch.tensor(1).cuda(), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = - mean_log_prob_pos
        if weight is not None:
            loss = loss*weight
        loss = loss.mean()

        return loss, self.centers
    
    def forward_robust(self, x, labels, w=None):
        
        # print(x.shape, labels.shape)
        batch_size = x.size(0)
        
        # normalization
        x = F.normalize(x, p=2, dim=1)
        centers = self.centers
        centers = F.normalize(centers, p=2, dim=1)
        # print(centers.shape)
        # print(x.shape)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().cuda()
        all_centers_sum = torch.div(centers.sum(dim=0).unsqueeze(0).repeat(batch_size,1),self.num_classes)  # [288,512]
        compute_center = centers.unsqueeze(0).repeat(batch_size,1,1)  # [288,40,512]
        compute_one_hot = label_one_hot.unsqueeze(2).repeat(1,1,self.feat_dim) # [288,40,512]
        # one_centers_sum = torch.div(torch.mul(compute_center, compute_one_hot).sum(1),(self.num_classes/(self.num_classes+1)))
        
        # loss = torch.div((torch.exp(torch.mul(all_centers_sum,x).sum(1)) - torch.exp(torch.mul(one_centers_sum,x).sum(1))),self.temperature).sum(0) / batch_size
        one_centers_sum = torch.mul(compute_center, compute_one_hot).sum(1)
        loss = (-torch.abs(torch.div(torch.add(torch.exp(torch.mul(all_centers_sum,x).sum(1)) - torch.exp(torch.mul(one_centers_sum,x).sum(1)),self.a),self.alpha))).sum(0)/batch_size 
        
        # loss = (1-v) * loss_1 + v * loss_2

        return loss, self.centers

    def get_centers(self):
        return self.centers.clone().detach()
    
    @torch.no_grad()
    def center_classifier(self, img_features, pt_features):
        
        img_features = F.normalize(img_features, p=2, dim=1)
        pt_features = F.normalize(pt_features, p=2, dim=1)
        centers =  F.normalize(self.centers.clone().detach(), p=2, dim=1)
        
        scores_img = torch.matmul(img_features, centers.T)
        scores_pt = torch.matmul(pt_features, centers.T)
        
        return scores_img, scores_pt

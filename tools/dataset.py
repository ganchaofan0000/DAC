import os
import sys
import glob
import h5py
import json
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch
import numpy as np

# from tools.visualize import showpoints

def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)


def load_data(partition,dataset_dir):
    all_data = []
    all_label = []
    img_lst = []
    if partition=="train":
        orders=[3,4,0,1,2]
    else:
        orders=[0,1]
    h5_list = [os.path.join(dataset_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s%s.h5'%(partition, index)) for index in orders]
    for h5_name in h5_list:
        split = h5_name[-4]
        # print(split)
        jason_name = dataset_dir+'modelnet40_ply_hdf5_2048/ply_data_' + partition +'_' + split + '_id2file.json'

        with open(jason_name) as json_file:
            images = json.load(json_file)        
        img_lst = img_lst + images
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0) # DEBUG:
    all_label = all_label.flatten()
    print(len(all_data), len(all_label), len(img_lst))
    return all_data, all_label, img_lst


def load_modelnet10_data(partition,dataset_dir):
    all_data = []
    all_label = []
    img_lst = []
    # a1=["/mnt/sdb/chaofan/data/ModelNet40/modelnet10_hdf5_2048/train1.h5", "/mnt/sdb/chaofan/data/ModelNet40/modelnet10_hdf5_2048/train0.h5"]
    for h5_name in sorted(glob.glob(os.path.join(dataset_dir, 'modelnet10_hdf5_2048', '%s*.h5'%partition)), reverse=True):
        # print(h5_name)
    # for h5_name in a1:
        split = h5_name[-4]
        jason_name = dataset_dir+'modelnet10_hdf5_2048/'+partition + split + '_id2file.json'
        with open(jason_name) as json_file:
            images = json.load(json_file)
        img_lst = img_lst + images
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0) # DEBUG:
    all_label = all_label.flatten()
    print(len(all_data), len(all_label), len(img_lst))
    return all_data, all_label, img_lst



def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.rand()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def random_scale(pointcloud, scale_low=0.8, scale_high=1.25):
    N, C = pointcloud.shape
    scale = np.random.uniform(scale_low, scale_high)
    pointcloud = pointcloud*scale
    return pointcloud

class ModelNet_Dataset(Dataset):
    def __init__(self, dataset, num_points, num_classes, dataset_dir, partition='train', 
                 noise_type="sym",noise_rate=0.0, select_indices=None, weight_scores=None, use_cross=False):
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.select_indices = select_indices
        self.weight_scores = weight_scores
        self.partition = partition
        self.use_cross = use_cross
        fea_path = os.path.join(self.dataset_dir, "pc_crosspoint_%s.pt"%self.partition)
        
        embedding = torch.load(fea_path, map_location='cpu')
        
        obj_name_list = embedding["us"]
        obj_fea = embedding["feats"]
        obj_fea = [a for a in obj_fea]
        
        ###转为dict
        self.name2fea_dict = {key: value for key, value in zip(obj_name_list, obj_fea)}
        
        if self.dataset == 'ModelNet40':
            self.data, self.label1, self.img_lst = load_data(partition,self.dataset_dir)
            data_dir = os.path.join(self.dataset_dir, 'modelnet40_ply_hdf5_2048/%s_pt_feat_small.npy'%(self.partition))
            ori_label_dir = os.path.join(self.dataset_dir, 'modelnet40_ply_hdf5_2048/%s_ori_label.npy'%(self.partition))
            
            if partition=="train" and noise_rate>0:
                if noise_type=="sym":
                    label_dir = os.path.join(self.dataset_dir, 'modelnet40_ply_hdf5_2048/%s_label_%s.npy'%(self.partition, int(noise_rate*100)))
                else:
                    label_dir = os.path.join(self.dataset_dir, 'modelnet40_ply_hdf5_2048/%s_asy_label_%s.npy'%(self.partition, int(noise_rate*100)))
                    
            else:
                label_dir = ori_label_dir
            
            self.data = np.load(data_dir)
            self.label = np.load(label_dir)
            self.ori_label = np.load(ori_label_dir)
            
            print((self.label1==self.ori_label).sum()/self.ori_label.shape[0])
        else:
            self.data, self.label1, self.img_lst = load_modelnet10_data(partition,self.dataset_dir)
            
            # print(self.label1[:20])
            
            # dir_name = "modelnet10_hdf5_2048"
            data_dir = os.path.join(self.dataset_dir, 'modelnet10_hdf5_2048/%s_pt_feat_small.npy'%(self.partition))
            ori_label_dir = os.path.join(self.dataset_dir, 'modelnet10_hdf5_2048/%s_ori_label.npy'%(self.partition))
            
            if partition=="train" and noise_rate>0:
                if noise_type=="sym":
                    label_dir = os.path.join(self.dataset_dir, 'modelnet10_hdf5_2048/%s_label_%s.npy'%(self.partition, int(noise_rate*100)))
                else:
                    label_dir = os.path.join(self.dataset_dir, 'modelnet10_hdf5_2048/%s_asy_label_%s.npy'%(self.partition, int(noise_rate*100)))
                    
            else:
                label_dir = ori_label_dir
            
            self.data = np.load(data_dir)
            self.label = np.load(label_dir)
            self.ori_label = np.load(ori_label_dir)
            # print(self.ori_label[:20])
            # print((self.label1==self.ori_label).sum()/self.ori_label.shape[0])
        self.num_points = num_points
        self.partition = partition
        self.num_classes=num_classes
        
        print(np.sum(self.ori_label==self.label)/self.ori_label.shape[0])
        if select_indices is not None:
            print("select data")
            self.data, self.label = self.data[select_indices], self.label[select_indices]
            self.ori_label = self.ori_label[select_indices]
            self.img_lst = [self.img_lst[indice] for indice in select_indices]
            if self.weight_scores is not None:
                self.weight_scores = [self.weight_scores[indice] for indice in select_indices]
            
        
        self.img_train_transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.Resize(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.img_test_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_data(self, item):

        # Get Image Data first
        names = self.img_lst[item]
        names = names.split('/')
        
        #randomly select one image from the 179 images for each object
        img_list=[]
        img_index_list=[]
        if self.partition=="train":
            num_view = 2
        else:
            num_view = 4
        for view in range(num_view):
            img_idx = random.randint(0, 179)
            while img_idx in img_index_list:
                img_idx = random.randint(0, 179)
            img_index_list.append(img_idx)
            img_names =self.dataset_dir+'ModelNet40-Images-180/%s/%s/%s.%d.png' % (names[0], names[1][:-4], names[1][:-4], img_idx)
            
            # img_names = self.dataset_dir+'ModelNet40-Images-180/%s/%s/%s.%d.png' % ( names[0], names[1][:-4], names[1][:-4], img_idx)
            img = Image.open(img_names).convert('RGB')
            img_list.append(img)

        label = int(self.label[item])
        # print(len(img_list))
        # obj_name = names[1][:-4]
        # pointcloud = self.name2fea_dict[obj_name]
        if self.use_cross:
            obj_name = names[1][:-4]
            pointcloud = self.name2fea_dict[obj_name]
        else:
            pointcloud = self.data[item].astype(np.float32)
        if self.partition == 'train':
            img_list = [self.img_train_transform(img_per) for img_per in img_list]
        else:
            # img = self.img_test_transform(img)
            # img2 = self.img_test_transform(img2)
            img_list = [self.img_test_transform(img_per) for img_per in img_list]

        return pointcloud, label, img_list


    def get_mesh(self, item):
        
        names = self.img_lst[item]
        names = names.split('/')
        mesh_path = self.dataset_dir+'ModelNet40_Mesh/' + names[0] + '/%s/'%(self.partition) + names[1][:-4] + '.npz'

        data = np.load(mesh_path)
        face = data['face']
        neighbor_index = data['neighbor_index']

        # data augmentation
        if self.partition == 'train':
            sigma, clip = 0.01, 0.05
            jittered_data = np.clip(sigma * np.random.randn(*face[:, :12].shape), -1 * clip, clip)
            face = np.concatenate((face[:, :12] + jittered_data, face[:, 12:]), 1)

        # fill for n < max_faces with randomly picked faces
        max_faces = 1024 
        num_point = len(face)
        if num_point < max_faces:
            fill_face = []
            fill_neighbor_index = []
            for i in range(max_faces - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        return centers, corners, normals, neighbor_index

    def check_exist(self, item):
        names = self.img_lst[item]
        names = names.split('/')
        mesh_path = self.dataset_dir+'ModelNet40_Mesh/' + names[0] + '/%s/'%(self.partition) + names[1][:-4] + '.npz'

        return os.path.isfile(mesh_path)

    def __getitem__(self, item):
        
        while not self.check_exist(item):
            idx = random.randint(0, len(self.data)-1)
            item = idx

        pt, target, img_list= self.get_data(item)
        # centers, corners, normals, neighbor_index = self.get_mesh(item)
        # pt = torch.from_numpy(pt)
        target_ori = int(self.ori_label[item])
        
        if self.weight_scores is not None:
            weight_score = self.weight_scores[item]
        
        if self.select_indices is not None:
            item = self.select_indices[item]

        
        if self.weight_scores is not None:
        
            return img_list, pt, target, target_ori, weight_score, item
        
        return img_list, pt, target, target_ori, item

    def __len__(self):
        return self.data.shape[0]
    
    
    def get_gt_divide(self):
        gt_clean = self.ori_label==self.label
        gt_noisy = self.ori_label!=self.label
        return gt_clean, gt_noisy
    
    def get_img_list(self):
        if self.dataset != "ModelNet40":
            return self.ori_label, self.img_lst
        return self.ori_label, self.label, self.img_lst


class MINIST3D(Dataset):
    def __init__(self, dataset, num_points, num_classes, dataset_dir, partition='train', 
                 noise_type="sym",noise_rate=0.0, select_indices=None, weight_scores=None, use_cross=False):
        self.dataset=dataset
        self.num_points=num_points
        self.dataset_dir=dataset_dir
        self.select_indices = select_indices
        self.num_classes = num_classes
        print(noise_rate)
        if partition == 'train':
            self.img_feat = np.load(os.path.join(self.dataset_dir, 'train_img_feat.npy'))
            self.pt_feat = np.load(os.path.join(self.dataset_dir,'train_pt_feat.npy'))  
            # self.label = np.load(os.path.join(self.dataset_dir,'train_label_%s.npy'%(int(noise_rate*100))))
            ori_label_dir = os.path.join(self.dataset_dir,'train_ori_label.npy')
            # self.ori_label = np.load(os.path.join(self.dataset_dir,'train_ori_label.npy'))
            if partition=="train" and noise_rate>0.0:
                if noise_type=="sym":
                    label_dir = os.path.join(self.dataset_dir, 'train_label_%s.npy'%(int(noise_rate*100)))
                else:
                    label_dir = os.path.join(self.dataset_dir, 'train_asy_label_%s.npy'%(int(noise_rate*100)))
                    
            else:
                label_dir = ori_label_dir
            
            self.label = np.load(label_dir)
            self.ori_label = np.load(ori_label_dir)
            
            # print((self.label==self.ori_label).sum()/self.ori_label.shape[0])
        else:
            self.img_feat = np.load(os.path.join(self.dataset_dir,'test_img_feat.npy'))
            self.pt_feat = np.load(os.path.join(self.dataset_dir,'test_pt_feat.npy'))      
            self.ori_label = np.load(os.path.join(self.dataset_dir,'test_ori_label.npy'))
            self.label = self.ori_label
        
        if select_indices is not None:
            print("select data")
            self.img_feat, self.pt_feat = self.img_feat[select_indices], self.pt_feat[select_indices]
            self.label, self.ori_label = self.label[select_indices], self.ori_label[select_indices]
            # self.img_lst = [self.img_lst[indice] for indice in select_indices]

        
        print(np.sum(self.ori_label==self.label)/self.ori_label.shape[0])
        
    def __getitem__(self, item):
        img_feat =  self.img_feat[item]
        pt_feat =  self.pt_feat[item]
        label = self.label[item]
        ori_label = self.ori_label[item]
        
        if self.select_indices is not None:
            item = self.select_indices[item]

        return img_feat, pt_feat, label, ori_label, item
    def __len__(self):
        return self.label.shape[0]
    
    def get_gt_divide(self):
        gt_clean = self.ori_label==self.label
        gt_noisy = self.ori_label!=self.label
        return gt_clean, gt_noisy
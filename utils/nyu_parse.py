from __future__ import division
from __future__ import print_function
import copy
import csv
import json
import numpy as np
import scipy.linalg
import scipy.io as sio
import os
import os.path as osp
import cPickle as pickle
import cPickle as pkl
import torch
from torch.autograd import Variable
from . import transformations
from . import bbox_utils
import pdb
from collections import defaultdict

valid_object_classes = [
    'bed', 'sofa', 'table', 'chair', 'desk', 'television',
    # 'cabinet', 'counter', 'refridgerator', 'night_stand', 'toilet', 'bookshelf', 'shelves', 'bathtub'
]

object_class2index = {'bed' : 1, 'sofa' :2, 'table' :3, 
    'chair':4 , 'desk':5, 'television':6,
}

def get_split(save_dir, image_names=None, train_ratio=0.7, val_ratio=0.28):
    ''' Loads saved splits if they exist. Otherwise creates a new split.

    Args:
        save_dir: Absolute path of where splits should be saved
    '''
    split_file = osp.join(save_dir, 'nyu_split.pkl')
    if os.path.isfile(split_file):
        return pickle.load(open(split_file, 'rb'))
    else:
        list.sort(image_names)
        image_names = np.ndarray.tolist(np.random.permutation(image_names))
        n_images = len(image_names)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        splits = {
            'train': image_names[0:n_train],
            'val': image_names[n_train:(n_train + n_val)],
            'test': image_names[(n_train + n_val):]
        }
        pickle.dump(splits, open(split_file, 'wb'))
        return splits

def scale_transform(s):
    return np.diag([s, s, s, 1])


def trans_transform(t):
    t = t.reshape(3)
    tform = np.eye(4)
    tform[0:3, 3] = t
    return tform

def invertTransformation(transform):
    invertedTransform = np.eye(4)
    invertedTransform[0:3, 0:3] = transform[0:3, 0:3].transpose()
    tinv = -1*np.matmul(transform[0:3, 0:3].transpose(), transform[0:3, 3:])
    invertedTransform[0:3, 3] = tinv.squeeze()
    return invertedTransform.copy()


def euler_to_rotation_matrix(theta, phi, psi):
    '''
    theta, phi, and psi are in degrees. 
    theta is rotation about x-axis
    phi is rotation about y-axis
    psi is rotation about z-axis
    '''
    rotx = np.array([[1,              0,                    0,     0],
                     [0,        np.cos(theta), -np.sin(theta),     0],
                     [0,        np.sin(theta),  np.cos(theta),     0],
                     [0,                 0,           0,           1]])

    roty = np.array([[np.cos(phi),    0,       np.sin(phi),        0],
                     [0,              1,            0     ,        0],
                     [-np.sin(phi),   0,       np.cos(phi),        0],
                     [0,                 0,           0,           1]])
    
    rotz = np.array([[np.cos(psi), -np.sin(psi),          0,       0],
                     [np.sin(psi),  np.cos(psi),          0,       0],
                     [0,                0,                1,       0],
                     [0,                 0,           0,           1]])
    rot =  np.matmul(np.matmul(rotx, roty), rotz)
    return rot

def get_sym_angles_from_axis(sym_type = ''):
    syms = [(0, 0, 0)]
    if sym_type == 'xz':
        syms.append((0, np.pi, 0))
    elif sym_type == 'xy':
        syms.append((0, 0, np.pi))
    elif sym_type == 'yz':
        syms.append((np.pi, 0, 0))
    elif sym_type == 'xyz':
        syms.append((np.pi, 0, 0))
        syms.append((0, np.pi, 0))
        syms.append((0, 0, np.pi))
    return syms


def codify_room_data(object_data, obj_loader, use_shape_code=False, max_object_classes=10):
    n_objects = len(object_data)
    codes = []
    bboxes = []
    for nx in range(n_objects):
        model_name = object_data[nx]['basemodel'].replace(".mat", "")
        if (use_shape_code):
            shape = obj_loader.lookup(model_name, 'code')
        else:
            shape = obj_loader.lookup(model_name, 'voxels')
        vox_shift = trans_transform(np.ones((3, 1)) * 0.5)
        vox2obj = np.matmul(
            np.matmul(
                trans_transform(obj_loader.lookup(model_name, 'translation')),
                scale_transform(obj_loader.lookup(model_name, 'scale')[0, 0])
            ), vox_shift)

        syms = get_sym_angles_from_axis(obj_loader.lookup(model_name, 'sym'))
        quat_vals = []
        cams2vox_object = []
        vox2cams_object = []
        for sym_angles in syms:
            theta, phi, psi  = sym_angles
            sym_rot = euler_to_rotation_matrix(theta, phi, psi)
            scale = np.array(object_data[nx]['scale'], copy=True, dtype=np.float32)
            trans = np.array(object_data[nx]['trans'], copy=True, dtype=np.float32)
            rot_val = np.array(object_data[nx]['pose_full'], copy=True, dtype=np.float32)
            rot_val =  np.pad(rot_val, (0, 1), 'constant')
            rot_val[3,3] = 1
            obj2cam = np.matmul(trans_transform(trans) , np.array(rot_val, copy=True))
            obj2cam = np.matmul(obj2cam, sym_rot)
            vox2cam = np.matmul(obj2cam, vox2obj)
            trans = np.array(vox2cam[0:3, 3], copy=True, dtype=np.float32)
            vox2cams_object.append(vox2cam)
            cams2vox_object.append(invertTransformation(vox2cam))
            quat_val = transformations.quaternion_from_matrix(rot_val, isprecise=True)
            quat_vals.append(quat_val)

        bbox = np.array(object_data[nx]['bbox'], copy=True, dtype=np.float32)
        quat_val = np.stack(quat_vals).copy()
        transform_cam2vox = np.stack(cams2vox_object).astype(np.float32)
        transform_vox2cam = np.stack(vox2cams_object).astype(np.float32)
        amodal_bbox = np.zeros((8,3))
        
        code = {'shape' : shape,
        'vox2cam' : vox2cam,
        'scale' : scale,
        'quat' : quat_val,
        'trans' : trans,
        'class' : np.array([object_class2index[object_data[nx]['cls']]]),
        'transform_cam2vox' : transform_cam2vox,
        'transform_vox2cam' : transform_vox2cam,
        'amodal_bbox' : amodal_bbox,
        }
        codes.append(code)
        bboxes.append(bbox)

    if len(bboxes) > 0:
        bboxes = np.vstack(bboxes)

    return codes, bboxes

def select_ids(object_data, metaloader=None, min_pixels=0):
    n_objects = len(object_data)
    objects  = []
    for nx in range(n_objects):
        obj = object_data[nx]
        if obj['cls'] in valid_object_classes:
            objects.append(obj)

    return objects

class ObjectLoader:
    '''Pre-loading class to facilitate object lookup'''

    def __init__(self, object_dir):
        self._object_dir = object_dir
        object_names = [f.replace(".mat","") for f in os.listdir(object_dir) if "mat" in f]
        list.sort(object_names)
        self._object_names = object_names
        self._curr_obj_id = None
        self._preloaded_data = {}
        self._predloaded_syms = defaultdict(str)

    def lookup(self, obj_id, field):
        if obj_id != self._curr_obj_id:
            self._curr_obj_id = obj_id
            if obj_id in self._preloaded_data.keys():
                self._curr_obj_data = self._preloaded_data[obj_id]
            else:
                self._curr_obj_data = sio.loadmat(osp.join(
                    self._object_dir, obj_id + '.mat'))
        return copy.copy(self._curr_obj_data[field])

    def preload(self):
        with open('symmetry_nyu.pkl') as f:
            preloaded_syms = pickle.load(f)
        for ox in range(len(self._object_names)):
            obj_id = self._object_names[ox]
            obj_data = sio.loadmat(osp.join(
                self._object_dir, obj_id + '.mat'))
            obj_data_new = {}
            obj_data_new['scale'] = obj_data['scale'].copy()
            obj_data_new['translation'] = obj_data['translation'].copy()
            obj_data_new['voxels']  = obj_data['voxels'].copy()
            # for k in obj_data.keys():
            #     obj_data_new[k] = obj_data[k].clone()
            self._preloaded_data[obj_id] = obj_data_new
            if obj_id in preloaded_syms.keys():
                self._preloaded_data[obj_id]['sym'] = preloaded_syms[obj_id]
            else:
                self._preloaded_data[obj_id]['sym'] = ''
        return


# class ObjectLoader:
#     def __init__(self, object_dir):
#         self._object_dir = object_dir
#         object_names = [f.replace('.mat','') for f in os.listdir(object_dir)]
#         list.sort(object_names)
#         self._object_names = object_names
#         self._curr_obj_id = None
#         self._preloaded_data = {}
#         self._predloaded_syms = defaultdict(str)

#     def lookup(self, obj_id, field):
#         if obj_id != self._curr_obj_id:
#             self._curr_obj_id = obj_id
#             if obj_id in self._preloaded_data.keys():
#                 self._curr_obj_data = self._preloaded_data[obj_id]
#             else:
#                 self._curr_obj_data = sio.loadmat(osp.join(
#                     self._object_dir, obj_id + '.mat'))
#         return copy.copy(self._curr_obj_data[field])

#     def preload(self):
#         # with open('symmetry2.pkl') as f:
#         #     preloaded_syms = pickle.load(f)
        
#         for ox in range(len(self._object_names)):
#             obj_id = self._object_names[ox]
#             obj_data = sio.loadmat(osp.join(
#                 self._object_dir, obj_id + '.mat'))
#             obj_data_new = {}
#             obj_data_new['surfaces']  = obj_data['surfaces'].copy()
#             obj_data_new['comp']  = obj_data['comp'].copy()
#             self._preloaded_data[obj_id] = obj_data_new
#             # if obj_id in preloaded_syms.keys():
#             #     self._preloaded_data[obj_id]['sym'] = preloaded_syms[obj_id]
#             # else:
#             #     self._preloaded_data[obj_id]['sym'] = ''
#         return


class MetaLoader:
    def __init__(self, ):
        return
    def lookup(self, obj_id, field='nyuv2_40class'):
        obj_class =obj_id.split('_')[0]
        return obj_class


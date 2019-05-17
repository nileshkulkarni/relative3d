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

curr_path = osp.dirname(osp.abspath(__file__))
symmetry_file = osp.join(curr_path, '../cachedir/symmetry_suncg.pkl')
valid_object_classes = [
    'bed', 'sofa', 'table', 'chair', 'desk', 'television',
]

object_class2index = {'bed' : 1, 'sofa' :2, 'table' :3, 
    'chair':4 , 'desk':5, 'television':6,
}

list.sort(valid_object_classes)



def define_spatial_image(img_height, img_width, spatial_scale):
        img_height = int(img_height * spatial_scale)
        img_width = int(img_width * spatial_scale)
        spatial_h = torch.arange(0, img_height).unsqueeze(1).expand(torch.Size([img_height, img_width]))
        spatial_w = torch.arange(0, img_width).unsqueeze(0).expand(torch.Size([img_height, img_width]))
        spatial_h /= img_height
        spatial_w /= img_width
        spatial_image = torch.stack([spatial_h, spatial_w])
        return spatial_image


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

def load_json(json_file):
    '''
    Parse a json file and return a dictionary.

    Args:
        json_file: Absolute path of json file
    Returns:
        var: json data as a nested dictionary
    '''
    with open(json_file) as f:
        var = json.load(f)
        return var


# ------------ Cameras -------------#
# ----------------------------------#
def read_node_indices(node_file):
    '''
    Returns a dictionary mapping node index of entity name.

    Args:
        node_file: Absolute path of node indices file
    Returns:
        node_dict: output mapping
    '''
    node_dict = {}
    with open(node_file) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        for c in content:
            index, entity = c.split()
            node_dict[int(index)] = entity
    return node_dict


def read_camera_pose(cam_file):
    '''
    Returns a list of camera poses stored in the cam_file.

    Args:
        cam_file: Absolute path of camera file
    Returns:
        cam_data: List of cameras, each camera pose is a list of 12 numbers
    '''
    with open(cam_file) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        cam_data = [[float(v) for v in l.split()] for l in content]
        return cam_data


def cam_intrinsic():
    '''
    SUNCG rendering intrinsic matrix.

    Returns:
        3 X 3 intrinsic matrix
    '''
    return np.array([
        [517.97, 0, 320],
        [0, 517.97, 240],
        [0, 0, 1]
    ])


def campose_to_extrinsic(campose):
    '''
    Obtain an extrinsic matrix from campose format.

    Based on sscnet implementation
    (https://github.com/shurans/sscnet/blob/master/matlab_code/utils/camPose2Extrinsics.m)
    Args:
        campose: list of 12 numbers indicting the camera pose
    Returns:
        extrinsic: 4 X 4 extrinsic matrix
    '''
    tv = np.array(campose[3:6])
    uv = np.array(campose[6:9])
    rv = np.cross(tv, uv)
    trans = np.array(campose[0:3]).reshape(3, 1)

    extrinsic = np.concatenate((
        rv.reshape(3, 1),
        -1 * uv.reshape(3, 1),
        tv.reshape(3, 1),
        trans.reshape(3, 1)), axis=1)
    extrinsic = np.concatenate((extrinsic, np.array([[0, 0, 0, 1]])), axis=0)
    return extrinsic


# --------- Intersection -----------#
# ----------------------------------#
def dehomogenize_coord(point):
    nd = point.size
    return point[0:nd - 1] / point[nd]


def is_inside_box(point, bbox):
    c1 = np.all(point.reshape(-1) >= np.array(bbox['min']))
    c2 = np.all(point.reshape(-1) <= np.array(bbox['max']))
    return c1 and c2


def intersects(obj_box, grid_box, transform):
    '''
    Check if the trnsformed obj_box intersects the grid_box.

    Args:
        obj_box: 3D bbox of object in world frame
        grid_box: grid in camera frame
        transform: extrinsic matrix for world2camera frame
    Returns:
        True/False indicating if the boxes intersect
    '''
    for ix in range(8):
        point = [0, 0, 0, 1]
        point[0] = obj_box['min'][0] if ((ix // 1) % 2 == 0) else obj_box['max'][0]
        point[1] = obj_box['min'][1] if ((ix // 2) % 2 == 0) else obj_box['max'][1]
        point[2] = obj_box['min'][2] if ((ix // 4) % 2 == 0) else obj_box['max'][2]
        pt_transformed = np.matmul(transform, np.array(point).reshape(4, 1))
        if is_inside_box(dehomogenize_coord(pt_transformed), grid_box):
            return True

    return False


# --------- House Parsing ----------#
# ----------------------------------#
def select_ids(house_data, bbox_data, min_sum_dims=0, min_pixels=0, meta_loader=None):
    '''
    Selects objects which are indexed by bbox_data["node_ids"].

    Args:
        house_data: House Data
        bbox_data: ids for desired objects, and the number of pixels for each
    Returns:
        house_data with nodes only for the selected objects
    '''
    house_copy = {}
    room_node = {}


    for k in house_data.keys():
        if (k != 'levels'):
            house_copy[k] = copy.deepcopy(house_data[k])
            room_node[k] = copy.deepcopy(house_data[k])
    house_copy['levels'] = [{}]  # empty dict
    room_node['nodeIndices'] = []
    house_copy['levels'][0]['nodes'] = [room_node]



    bboxes = bbox_data['bboxes']
    node_ids = bbox_data['ids']
    n_pixels = bbox_data['nPixels']
    n_selected = 0
    selected_inds = []
    select_node_ids = []
    for ix in range(n_pixels.size):
        lx = int(node_ids[ix] // 10000) - 1
        nx = int(node_ids[ix] % 10000) - 1

        obj_node = house_data['levels'][lx]['nodes'][nx]
        select_node = obj_node['type'] == 'Object'

        if min_sum_dims > 0 and select_node:
            select_node = sum(obj_node['bbox']['max']) >= (sum(obj_node['bbox']['min']) + min_sum_dims)

        if min_pixels > 0 and select_node:
            select_node = n_pixels[ix] > min_pixels

        if (meta_loader is not None) and select_node:
            object_class = meta_loader.lookup(obj_node['modelId'], field='nyuv2_40class')
            select_node = object_class in valid_object_classes
            # select_node = True

        if select_node:
            # syms = obj_loader.lookup(model_name, 'sym')
            mirrored =  (obj_node.has_key('isMirrored') and obj_node['isMirrored'] == 1)
            # print("{} , {}, {}, {}, {}".format(ix, object_class, obj_node['modelId'], obj_node['id'], (obj_node.has_key('isMirrored') and obj_node['isMirrored'] == 1)))
            selected_inds.append(ix)
            select_node_ids.append(node_ids[ix][0])
            n_selected = n_selected + 1
            obj_node['class'] = object_class2index[object_class]

            house_copy['levels'][0]['nodes'][0]['nodeIndices'].append(n_selected)
            house_copy['levels'][0]['nodes'].append(copy.deepcopy(obj_node))


    objects_bboxes  = np.copy(bboxes[selected_inds, :]).astype(np.float)
    paths = [[[] for i in range(len(select_node_ids))] for j in range(len(select_node_ids))]
    object_locations = np.zeros((len(select_node_ids), 3))

    return house_copy, objects_bboxes


def select_layout(house_data):
    '''
    Selects all room nodes while ignoring objects.

    Args:
        house_data: House Data
    Returns:
        house_data with nodes only for the rooms
    '''
    house_copy = copy.deepcopy(house_data)
    for lx in range(len(house_data['levels'])):
        nodes_layout = []
        for nx in range(len(house_data['levels'][lx]['nodes'])):
            node = house_data['levels'][lx]['nodes'][nx]
            if node['type'] == 'Room':
                node_copy = copy.deepcopy(node)
                node_copy['nodeIndices'] = None
                nodes_layout.append(node_copy)
        house_copy['levels'][ls]['nodes'] = nodes_layout
    return house_copy


# -------- Voxel Processing --------#
# ----------------------------------#
def prune_exterior_voxels(voxels, voxel_coords, grid_box, cam2world, slack=0.05):
    '''
    Sets occupancies of voxels outside grid box to 0.

    Args:
        voxels: grid indicating occupancy at each location
        voxel_coords: 4 X grid_size voxel coordinates (in camera frame)
        grid_box: area of interest (in world frame)
        cam2world: transformation matrix
        slack: The grid_box is assumed to be a bit bigger by this fraction
    Returns:
        voxels with some occupancies outside set to 0
    '''
    min_dims = np.array(grid_box['min']).reshape(-1, 1)
    max_dims = np.array(grid_box['max']).reshape(-1, 1)
    box_slack = (max_dims - min_dims) * slack
    min_dims -= box_slack
    max_dims += box_slack

    ndims = min_dims.size
    cam2world = cam2world[0:ndims - 1, :]

    voxel_coords = np.matmul(cam2world, voxel_coords.reshape(ndims, -1))
    is_inside = (voxel_coords >= min_dims) & (voxel_coords <= max_dims)
    is_inside = np.all(is_inside, axis=0, keep_dims=False).reshape(voxels.shape)
    return np.multiply(voxels, is_inside.astype(voxels.dtype))


# ------------- Codify -------------#
# ----------------------------------#
def scale_transform(s):
    return np.diag([s, s, s, 1])


def trans_transform(t):
    t = t.reshape(3)
    tform = np.eye(4)
    tform[0:3, 3] = t
    return tform


def homogenize_coordinates(locations):
    '''

    :param locations: N x 3 array of locations
    :return: homogeneous coordinates
    '''
    ones = np.ones((locations.shape[0], 1))
    homogenous_location = np.concatenate((locations, ones), axis=1)
    return homogenous_location


def dehomogenize_coordinates(points):
    n, nd = points.shape
    points = points[:, 0:nd - 1] / (points[:, nd - 1].reshape(-1, 1))
    return points


def normalize_coordinates_to_image_plane(points):
    n, nd = points.shape
    points = points / (points[:, nd -1].reshape(-1, 1))
    return points

def transform_coordinates(transformation_matrix, locations):
    '''
    Homogenize location, transform them, dehomgenize.
    :param transformation_matrix:  4 x 4 matrix
    :param locations: N x 3
    :return: N x 3
    '''
    transformed_locations = homogenize_coordinates(locations)  ## N x 4
    transformed_locations = np.matmul(transformation_matrix, transformed_locations.transpose()).transpose()
    transformed_locations = dehomogenize_coordinates(transformed_locations).copy()

    return transformed_locations


def transform_to_image_coordinates(points, intrinsic_matrix):
    normalized_points = normalize_coordinates_to_image_plane(points)
    image_coordinates = np.matmul(intrinsic_matrix, normalized_points.transpose()).transpose()
    return image_coordinates[:,0:2].copy()




# def invertTransformation(transform):
#     invertedTransform = np.eye(4)
#     invertedTransform[0:3, 0:3] = transform[0:3, 0:3].transpose()
#     invertedTransform[0:3, 3] = -transform[0:3, 3]

#     return invertedTransform


def invertTransformation(transform):
    invertedTransform = np.eye(4)
    invertedTransform[0:3, 0:3] = transform[0:3, 0:3].transpose()
    tinv = -1*np.matmul(transform[0:3, 0:3].transpose(), transform[0:3, 3:])
    invertedTransform[0:3, 3] = tinv.squeeze()
    return invertedTransform.copy()


def convert_bbox_max_min_to_corners(bbox_max, bbox_min):
    corner_points = np.zeros((8, 3), dtype=np.float32)
    corner_points[0, :] = np.array([bbox_min[0], bbox_min[1], bbox_min[2]], dtype=np.float32)
    corner_points[1, :] = np.array([bbox_min[0], bbox_max[1], bbox_min[2]], dtype=np.float32)
    corner_points[2, :] = np.array([bbox_max[0], bbox_max[1], bbox_min[2]], dtype=np.float32)
    corner_points[3, :] = np.array([bbox_max[0], bbox_min[1], bbox_min[2]], dtype=np.float32)

    corner_points[4, :] = np.array([bbox_min[0], bbox_min[1], bbox_max[2]], dtype=np.float32)
    corner_points[5, :] = np.array([bbox_min[0], bbox_max[1], bbox_max[2]], dtype=np.float32)
    corner_points[6, :] = np.array([bbox_max[0], bbox_max[1], bbox_max[2]], dtype=np.float32)
    corner_points[7, :] = np.array([bbox_max[0], bbox_min[1], bbox_max[2]], dtype=np.float32)
    return corner_points


def convert_2D_corners_bbox_max_min(corners):
    bbmin = np.min(corners,axis = 0) 
    bbmax = np.max(corners, axis = 0)
    return np.concatenate([bbmin, bbmax])


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

def convert_bbox_max_min_to_corners(bbox_max, bbox_min):
    corner_points = np.zeros((8, 3), dtype=np.float32)
    corner_points[0, :] = np.array([bbox_min[0], bbox_min[1], bbox_min[2]], dtype=np.float32)
    corner_points[1, :] = np.array([bbox_min[0], bbox_max[1], bbox_min[2]], dtype=np.float32)
    corner_points[2, :] = np.array([bbox_max[0], bbox_max[1], bbox_min[2]], dtype=np.float32)
    corner_points[3, :] = np.array([bbox_max[0], bbox_min[1], bbox_min[2]], dtype=np.float32)

    corner_points[4, :] = np.array([bbox_min[0], bbox_min[1], bbox_max[2]], dtype=np.float32)
    corner_points[5, :] = np.array([bbox_min[0], bbox_max[1], bbox_max[2]], dtype=np.float32)
    corner_points[6, :] = np.array([bbox_max[0], bbox_max[1], bbox_max[2]], dtype=np.float32)
    corner_points[7, :] = np.array([bbox_max[0], bbox_min[1], bbox_max[2]], dtype=np.float32)
    return corner_points


def get_bounding_box_from_voxels(shape, limit=1.0):
    non_zero_voxels = np.where(shape)
    x_limits = limit*np.array([np.min(non_zero_voxels[0]), np.max(non_zero_voxels[0])])/(np.size(shape,0))
    y_limits = limit*np.array([np.min(non_zero_voxels[1]), np.max(non_zero_voxels[1])])/(np.size(shape,1))
    z_limits = limit*np.array([np.min(non_zero_voxels[2]), np.max(non_zero_voxels[2])])/(np.size(shape,2)-1)

    bbox_min = np.array([x_limits[0], y_limits[0], z_limits[0]]) - 0.5
    bbox_max = np.array([x_limits[1], y_limits[1], z_limits[1]]) - 0.5

    return (bbox_min, bbox_max)

def codify_room_data(house_data, world2cam, obj_loader, use_shape_code=False, max_object_classes=10):
    '''
    Coded form of objects.

    Args:
        house_data: nested dictionary corresponding to a single room
        world2cam: 4 X 4 transformation matrix
        obj_loader: pre-loading class to facilitate fast object lookup
    Returns:
        codes: list of (shape, transformation, scale, rot, trans) tuples
    '''
    n_obj = len(house_data['levels'][0]['nodes']) - 1
    codes = []
    object3dWorld = []
    object3dCam = []
    amodal_bboxes = []
    cam2world = invertTransformation(world2cam)
    object_classes = []
    object3dImageCoordinates = []
    cam2voxs = []
    vox2cams = []
    intrinsic_matrix = cam_intrinsic()
    for nx in range(1, n_obj + 1):
        obj_node = house_data['levels'][0]['nodes'][nx]
        model_name = obj_node['modelId']
        # pdb.set_trace()
        if (use_shape_code):
            shape = obj_loader.lookup(model_name, 'code')
        else:
            shape = obj_loader.lookup(model_name, 'voxels')

        vox_shift = trans_transform(np.ones((3, 1)) * 0.5)
        ## Fix for the two incorrect chairs. 
        if model_name in ['142', '106'] and False:
            if not (obj_node.has_key('isMirrored') and obj_node['isMirrored'] == 1):
                vox_shift = np.matmul(vox_shift, np.diag([-1, 1, 1, 1]))
        else:
            if (obj_node.has_key('isMirrored') and obj_node['isMirrored'] == 1):
                vox_shift = np.matmul(vox_shift, np.diag([-1, 1, 1, 1]))

        # print(model_name)
        # print(scale_transform(obj_loader.lookup(model_name, 'scale')))
        bbox3d_voxels = get_bounding_box_from_voxels(shape)
        bbox3d_voxels_corners = convert_bbox_max_min_to_corners(bbox3d_voxels[0], bbox3d_voxels[1]).astype(np.float32)
        vox2obj = np.matmul(
            np.matmul(
                trans_transform(obj_loader.lookup(model_name, 'translation')),
                scale_transform(obj_loader.lookup(model_name, 'scale')[0, 0])
            ), vox_shift)
        ## Rotate the object about Y-axis by 180 degrees.

        quat_vals = []
        syms = get_sym_angles_from_axis(obj_loader.lookup(model_name, 'sym'))
        cams2vox_object = []
        vox2cams_object = []
        for sym_angles in syms:
            theta, phi, psi = sym_angles
            sym_rot = euler_to_rotation_matrix(theta, phi, psi)
            obj2world = np.array(obj_node['transform'], copy=True).reshape(4, 4).transpose()
            obj2world = np.matmul(obj2world, sym_rot)
            obj2cam = np.matmul(world2cam, obj2world)
            vox2cam = np.matmul(obj2cam, vox2obj)
            rot_val, scale_val = np.linalg.qr(vox2cam[0:3, 0:3])
            scale_val = np.array([scale_val[0, 0], scale_val[1, 1], scale_val[2, 2]])
            for d in range(3):
                if scale_val[d] < 0:
                    scale_val[d] = -scale_val[d]
                    rot_val[:, d] *= -1
            trans_val = np.array(vox2cam[0:3, 3], copy=True)
            rot_val = np.pad(rot_val, (0, 1), 'constant')
            rot_val[3, 3] = 1
            
            new_vox2cam = rot_val.copy()
            new_vox2cam[0:3,3] = trans_val
            cams2vox_object.append(invertTransformation(new_vox2cam))
            vox2cams_object.append(new_vox2cam)

            quat_val = transformations.quaternion_from_matrix(rot_val, isprecise=True)
            quat_vals.append(quat_val)

        quat_val = np.stack(quat_vals).copy()

        transform_cam2vox =np.stack(cams2vox_object).astype(np.float32)
        transform_vox2cam = np.stack(vox2cams_object).astype(np.float32)

        cam2voxs.append((np.stack(cams2vox_object)).astype(np.float32))
        vox2cams.append((np.stack(vox2cams_object)).astype(np.float32))


        object_classes.append(obj_node['class'])

        bbox3d = np.array([obj_node['bbox']['min'], obj_node['bbox']['max']], dtype=np.float32)  ## min , max
        bbox3d_corner_points_world = convert_bbox_max_min_to_corners(bbox3d[0], bbox3d[1]).astype(np.float32)
        bbox3d_corner_points_cam = transform_coordinates(world2cam, bbox3d_corner_points_world).astype(np.float32)
        object3dCam.append(bbox3d_corner_points_cam)
        voxel_corners_cam = transform_coordinates(vox2cam, bbox3d_voxels_corners).astype(np.float32)
        object3d_bbox_image = transform_to_image_coordinates(voxel_corners_cam, intrinsic_matrix)
        object3dImageCoordinates.append(object3d_bbox_image)
        # pdb.set_trace()
        amodal_bbox = convert_2D_corners_bbox_max_min(object3d_bbox_image)
        amodal_bboxes.append(amodal_bbox)
        # object3dCam.append(bbox3d_corner_points_world)
        object3dWorld.append(bbox3d_corner_points_world)

        code = {'shape' : shape.astype(np.float32),
            'vox2cam' : vox2cam,
            'scale' : scale_val,
            'quat' : quat_val,
            'trans' :trans_val,
            'class' : np.array([obj_node['class']]),
            'amodal_bbox' : amodal_bbox,
            'transform_cam2vox' : transform_cam2vox ,
            'transform_vox2cam' : transform_vox2cam }
        codes.append(code)

    extra_codes = (world2cam.copy(), cam2world, object3dCam, object3dWorld,
     object_classes, amodal_bboxes, cam2voxs, vox2cams)
    assert len(codes) == len(cam2voxs), 'code and cam2vox length do not match'
    return codes, extra_codes


def copy_code(code):
    copy = {k : np.copy(c) for k,c in code.items()}
    return copy



def extract_proposal_codes(
        codes_gt, bboxes_gt, bboxes_proposals, max_proposals,
        add_gt_boxes=True, pos_thresh=0.7, neg_thresh=0.3, pos_ratio=0.25,
        only_pos_proposals=False):
    # initialize counters and arrays
    ctr, n_pos, n_neg = 0, 0, 0
    codes = []
    labels = np.zeros((max_proposals))
    bboxes = np.zeros((max_proposals, 4))

    # Add gt boxes, compute positive and negative inds
    all_inds = np.array(range(bboxes_proposals.shape[0]))
    if len(codes_gt) > 0:
        if add_gt_boxes:
            for gx in range(len(bboxes_gt)):
                if gx < max_proposals:
                    codes.append(copy_code(codes_gt[gx]))
                    bboxes[ctr, :] = np.copy(bboxes_gt[gx, :])
                    labels[ctr] = 1
                    ctr += 1

        # Compute positive and negative indices
        ious = bbox_utils.bbox_overlaps(bboxes_gt.astype(np.float), bboxes_proposals.astype(np.float))
        max_ious = np.amax(ious, axis=0)
        gt_inds = np.argmax(ious, axis=0)
        pos_inds = all_inds[max_ious >= pos_thresh]
        neg_inds = all_inds[max_ious < neg_thresh]
    else:
        pos_inds = np.array([])
        neg_inds = all_inds

    # Add positive proposals
    pos_perm = np.random.permutation(pos_inds.size)
    while (n_pos < pos_inds.size) and (ctr < pos_ratio * max_proposals):
        px = pos_inds[pos_perm[n_pos]]
        gx = gt_inds[px]
        codes.append(copy_code(codes_gt[gx]))
        bboxes[ctr, :] = np.copy(bboxes_proposals[px, :])
        labels[ctr] = 1
        ctr += 1
        n_pos += 1

    # Add negative proposals
    if not only_pos_proposals:
        neg_perm = np.random.permutation(neg_inds.size)
        while (n_neg < neg_inds.size) and (ctr < max_proposals):
            px = neg_inds[neg_perm[n_neg]]
            bboxes[ctr, :] = np.copy(bboxes_proposals[px, :])
            labels[ctr] = 0
            ctr += 1
            n_neg += 1

    bboxes = bboxes[0:ctr, :]
    labels = labels[0:ctr]
    # print(len(codes_gt), n_pos, n_neg)
    return codes, bboxes, labels


# ---------- Object Loader ---------#
# ----------------------------------#
class ObjectLoader:
    '''Pre-loading class to facilitate object lookup'''

    def __init__(self, object_dir):
        self._object_dir = object_dir
        object_names = [f for f in os.listdir(object_dir)]
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
                    self._object_dir, obj_id, obj_id + '.mat'))
        return copy.copy(self._curr_obj_data[field])

    def preload(self):
        # with open('symmetry2.pkl') as f:
        with open(symmetry_file) as f:
            preloaded_syms = pickle.load(f)
        for ox in range(len(self._object_names)):
            obj_id = self._object_names[ox]
            obj_data = sio.loadmat(osp.join(
                self._object_dir, obj_id, obj_id + '.mat'))
            obj_data_new = {}
            # pdb.set_trace()
            obj_data_new['scale'] = obj_data['scale'].copy()
            obj_data_new['translation'] = obj_data['translation'].copy()
            obj_data_new['voxels']  = obj_data['voxels'].copy()
            # for k in obj_data.keys():
            #     obj_data_new[k] = obj_data[k].clone()
            # pdb.set_trace()
            self._preloaded_data[obj_id] = obj_data_new
            
            if obj_id in preloaded_syms.keys():
                self._preloaded_data[obj_id]['sym'] = preloaded_syms[obj_id]
            else:
                self._preloaded_data[obj_id]['sym'] = ''
        return


def read_csv(csv_file):
    with open(csv_file) as f:
            reader = csv.DictReader(f)
            metadata = []
            # pdb.set_trace()
            for row in reader:
                metadata.append(row)
    return copy.copy(metadata)

class MetaLoader:
    '''Pre-loading class for object metadata'''

    def __init__(self, csv_file):
        self._metadata = read_csv(csv_file)
        self._id_dict = {}
        for mx in range(len(self._metadata)):
            model_id = copy.copy(self._metadata[mx]['model_id'])
            self._id_dict[model_id] = mx

    def lookup(self, obj_id, field='nyuv2_40class'):
        if obj_id not in self._id_dict:
            return None
        mx = self._id_dict[obj_id]
        return self._metadata[mx][field]

# class MetaLoader:
#     '''Pre-loading class for object metadata'''

#     def __init__(self, csv_file):
#         self.modelId2class = {}
#         with open(csv_file) as f:
#             for line in f:
#                 line  = line.strip().split(',')
#                 self.modelId2class[line[0]] = line[1]


#     def lookup(self, obj_id, field='nyuv2_40class'):
#         if obj_id not in self.modelId2class.keys():
#             return None
#         return self.modelId2class[obj_id]



class ObjectRetriever:
    '''Class for nearest neighbor lookup'''

    def __init__(self, obj_loader, encoder, gpu_id=0, downsampler=None, meta_loader=None):
        self.enc_shapes = []
        self.orig_shapes = []
        for model_name in obj_loader._object_names:
            if (meta_loader is not None):
                object_class = meta_loader.lookup(model_name, field='nyuv2_40class')
                if object_class not in valid_object_classes:
                    continue

            shape = obj_loader.lookup(model_name, 'voxels').astype(np.float32)
            shape = torch.from_numpy(shape).unsqueeze(0).unsqueeze(0)
            shape_var = torch.autograd.Variable(shape.cuda(device=gpu_id), requires_grad=False)
            if downsampler is not None:
                shape_var = downsampler.forward(shape_var)
            shape = shape_var.data.cpu()[0]
            self.orig_shapes.append(shape)

            shape_enc = encoder.forward(shape_var)
            shape_enc = shape_enc.data.cpu()[0]
            self.enc_shapes.append(shape_enc)

        self.enc_shapes = torch.stack(self.enc_shapes)
        self.nz_shape = self.enc_shapes.shape[1]

    def retrieve(self, code):
        code = code.view([1, self.nz_shape])
        dists = (self.enc_shapes - code).norm(p=2, dim=1)
        _, inds = torch.min(dists, 0)
        return self.orig_shapes[inds[0]].clone()


# -------- Train/Val splits --------#
# ----------------------------------#
def get_split(save_dir, house_names=None, train_ratio=0.75, val_ratio=0.1):
    ''' Loads saved splits if they exist. Otherwise creates a new split.

    Args:
        save_dir: Absolute path of where splits should be saved
    '''
    split_file = osp.join(save_dir, 'suncg_split.pkl')
    if os.path.isfile(split_file):
        return pickle.load(open(split_file, 'rb'))
    else:
        list.sort(house_names)
        house_names = np.ndarray.tolist(np.random.permutation(house_names))
        n_houses = len(house_names)
        n_train = int(n_houses * train_ratio)
        n_val = int(n_houses * val_ratio)
        splits = {
            'train': house_names[0:n_train],
            'val': house_names[n_train:(n_train + n_val)],
            'test': house_names[(n_train + n_val):]
        }
        pickle.dump(splits, open(split_file, 'wb'))
        return splits


def batchify(feature, bIndices):
    def uncollate_codes(code_tensors, batch_size, batch_inds):
        '''
        Assumes batch inds are 0 indexed, increasing
        '''
        start_ind = 0
        codes = []
        for b in range(batch_size):
            nelems = torch.eq(batch_inds, b).sum()
            if nelems > 0:
                codes_b = code_tensors[start_ind:(start_ind + nelems)]
                start_ind += nelems
                codes.append(codes_b)
        return codes

    features = uncollate_codes(feature, int(bIndices.max()) + 1, bIndices)
    return features


def bboxes_to_rois2(bboxes):
    ''' Concatenate boxes and associate batch index.
    Accepts batches with no bboxes.
    '''
    # rois = torch.cat(bboxes, 0).type(torch.FloatTensor)
    # if rois.numel() == 0:
    #     return torch.zeros(0, 5).type(torch.FloatTensor)
    # batch_inds = torch.ones((rois.size(0), 1)).type(torch.FloatTensor)
    
    ctr = 0
    rois = []
    batch_inds = []
    for bx, boxes in enumerate(bboxes):
        nb = 0
        if len(boxes) > 0:
            nb = boxes.size(0)
            batch_inds.append(torch.ones(nb, 1).mul_(bx).type(torch.FloatTensor))
            rois.append(boxes)

    rois = torch.cat(rois).type(torch.FloatTensor)
    batch_inds = torch.cat(batch_inds, 0)
    rois = torch.cat([batch_inds, rois], 1)

    return rois.type(torch.FloatTensor)

# ---------- Torch Utils -----------#
# ----------------------------------#
def bboxes_to_rois(bboxes):
    ''' Concatenate boxes and associate batch index.
    '''
    rois = torch.cat(bboxes, 0).type(torch.FloatTensor)
    if rois.numel() == 0:
        return torch.zeros(0, 5).type(torch.FloatTensor)
    batch_inds = torch.ones((rois.size(0), 1)).type(torch.FloatTensor)
    ctr = 0
    for bx, boxes in enumerate(bboxes):
        nb = 0
        if boxes.numel() > 0:
            nb = boxes.size(0)
            batch_inds[ctr:(ctr + nb)] *= bx
            ctr += nb
    rois = torch.cat([batch_inds, rois], 1)
    return rois.type(torch.FloatTensor)


def normalize_quat(quat):
    return quat / (1E-6 + quat.norm(p=2, dim=1).unsqueeze(1).expand(quat.size()))


def collate_codes_instance(codes_b):
    ''' [(shape_i, tmat_i, scale_i, quat_i, trans_i)] => (shapes, scales, quats, trans)
    '''
    if len(codes_b) == 0:
        return None

    codes_out_b = {}
    select_keys = ['scale', 'transform_cam2vox', 'amodal_bbox',
                   'transform_vox2cam', 'shape', 'trans',
                   'quat', 'class']
    for skey in select_keys:
        if skey in ['quat', 'transform_cam2vox', 'transform_vox2cam']:
            codes_out_b[skey] = [code[skey].type(torch.FloatTensor) for code in codes_b]
        else:
            codes_out_b[skey] = torch.stack([code[skey] for code in codes_b], dim=0)
    return codes_out_b


def uncollate_codes_instance(code_tensors_b):
    '''
    (shapes, scales, quats, trans) => [(shape_i, tmat_i, scale_i, quat_i, trans_i)]
    '''
    codes_b = []
    if code_tensors_b['shape'].numel() == 0:
        return codes_b
    for cx in range(code_tensors_b['shape'].size(0)):
        code = {}
        for k,t in code_tensors_b.items():
            if type(t[cx]) == list:
                code[k] = [temp.squeeze().numpy() for temp in t[cx]]
            else:
                code[k] = t[cx].squeeze().numpy()
        codes_b.append(code)
    return codes_b


def collate_codes(codes):
    codes_instance = []
    codes_out = {}
    codes_quat_out = []
    for b in range(len(codes)):
        codes_b = collate_codes_instance(codes[b])
        if codes_b is not None:
            codes_instance.append(codes_b)

    keys = codes_instance[0].keys()
    for pkey in keys:
        if pkey in ['quat', 'transform_cam2vox', 'transform_vox2cam']:
            codes_out[pkey] = [code[pkey] for code in codes_instance]
            codes_out[pkey] = [j for i in codes_out[pkey] for j in i]
        else:
            codes_out[pkey] = torch.cat([code[pkey] for code in codes_instance]).type(torch.FloatTensor)

    return codes_out


def uncollate_codes(code_tensors, batch_size, batch_inds):
    '''
    Assumes batch inds are 0 indexed, increasing
    '''
    start_ind = 0
    codes = []
    keys = code_tensors.keys()

    for key in keys:
        if type(code_tensors[key]) == list:
            code_tensors[key] = [t.data.cpu() for t in code_tensors[key]]
        else:
            code_tensors[key] = code_tensors[key].data.cpu()


    for b in range(batch_size):
        codes_b = []
        nelems = torch.eq(batch_inds, b).sum()
        if nelems > 0:
            code_tensors_b = {}
            for key, item in code_tensors.items():
                code_tensors_b[key]= item[start_ind:(start_ind + nelems)]
            codes_b = uncollate_codes_instance(code_tensors_b)
            start_ind += nelems
        codes.append(codes_b)
    return codes


def uncollate_codes_instance_variable(code_tensors_b):
    codes_b = []
    if code_tensors_b[0].numel() == 0:
        return codes_b
    for cx in range(code_tensors_b[0].size(0)):
        code = []
        for t in code_tensors_b:
            code.append(t[cx])
        codes_b.append(tuple(code))
    return codes_b


def uncollate_codes_variable(code_tensors, batch_size, batch_inds):
    '''
    Assumes batch inds are 0 indexed, increasing
    '''
    start_ind = 0
    codes = []

    for b in range(batch_size):
        codes_b = []
        nelems = torch.eq(batch_inds, b).sum()
        if nelems > 0:
            code_tensors_b = []
            for t in code_tensors:
                code_tensors_b.append(t[start_ind:(start_ind + nelems)])
            codes_b = uncollate_codes_instance_variable(code_tensors_b)
            start_ind += nelems
        codes.append(codes_b)
    return codes

def convert_codes_list_to_old_format(codes_list):
    new_codes_list = []
    ## 0 --> shape
    ## 1 --> scale
    ## 2 --> quat
    ## 3 --> trans
    for code in codes_list:
        new_code = ()
        new_code  += (code['shape'], code['scale'], )
        if code['quat'].ndim > 1:  ## to handle symmetry stuff.
            new_code += (code['quat'][0], )
        else:
            new_code += (code['quat']  ,)
        new_code += (code['trans'] ,)
        new_codes_list.append(new_code)
    return new_codes_list


def bins_to_real_values(values, bins):
    min_val = torch.min(bins)
    max_val = torch.max(bins)
    num_bins = bins.numel()
    interval = (max_val - min_val) / (num_bins - 1)
    values = interval * values + min_val
    return values


def real_values_to_bins(values, bins):
    min_val = torch.min(bins)
    max_val = torch.max(bins)
    num_bins = bins.numel()
    interval = (max_val - min_val) / (num_bins - 1)

    min_input_value = torch.min(values)
    max_input_value = torch.max(values)
    if min_input_value < min_val:
        print('Min {}'.format(min_input_value))
    if max_input_value > max_val:
        print('Max {}'.format(max_input_value))

    values_clipped = torch.clamp(values, min_val, max_val)
    bin_values = torch.round((values_clipped - min_val) / interval)
    max_val = torch.max(bin_values)
    if max_val >= num_bins or max_val < 0:
        print('some values are out of range to classify')
        pdb.set_trace()

    return bin_values


def quats_to_bininds(quats, medoids):
    '''
    Finds the closest bin for each quaternion.
    
    Args:
        quats: N X 4 tensor
        medoids: n_bins X 4  tensor
    Returns:
        bin_inds: N tensor with values in [0, n_bins-1]
    '''
    medoids = medoids.transpose(1, 0)
    prod = torch.matmul(quats, medoids).abs()
    _, bin_inds = torch.max(prod, 1)
    return bin_inds

def directions_to_bininds(quats, medoids):
    '''
    Finds the closest bin for each quaternion.
    
    Args:
        quats: N X 4 tensor
        medoids: n_bins X 4  tensor
    Returns:
        bin_inds: N tensor with values in [0, n_bins-1]
    '''
    medoids = medoids.transpose(1, 0)
    prod = torch.matmul(quats, medoids)
    _, bin_inds = torch.max(prod, 1)
    return bin_inds

def bininds_to_directions(bin_inds, medoids):
    '''
    Select thee corresponding quaternion.
    
    Args:
        bin_inds: N tensor with values in [0, n_bins-1]
        medoids: n_bins X 4  tensor
    Returns:
        quats: N X 4 tensor
    '''
    n_bins = medoids.size(0)
    n_quats = bin_inds.size(0)
    bin_inds = bin_inds.view(-1, 1)
    inds_one_hot = torch.zeros(n_quats, n_bins)
    inds_one_hot.scatter_(1, bin_inds, 1)
    return torch.matmul(inds_one_hot, medoids)

def recursively_long_tensor_to_variable_cuda(data):
    if type(data) == list:
        return [recursively_long_tensor_to_variable_cuda(x) for x in data]
    elif type(data) == torch.LongTensor:
        return Variable(data.cuda())
    else:
        assert False, 'Ilegal data type {}'.format(type(data))
    return




def bininds_to_quats(bin_inds, medoids):
    '''
    Select thee corresponding quaternion.
    
    Args:
        bin_inds: N tensor with values in [0, n_bins-1]
        medoids: n_bins X 4  tensor
    Returns:
        quats: N X 4 tensor
    '''
    n_bins = medoids.size(0)
    n_quats = bin_inds.size(0)
    bin_inds = bin_inds.view(-1, 1)
    inds_one_hot = torch.zeros(n_quats, n_bins)
    inds_one_hot.scatter_(1, bin_inds, 1)
    return torch.matmul(inds_one_hot, medoids)

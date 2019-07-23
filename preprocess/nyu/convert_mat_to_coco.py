'''
Converts the mat data dumped by using Saurabh Gupta's code to per image instance files and into a more readable format
'''
import os.path as osp
import argparse
import scipy.io as sio
import cPickle as pkl
import pdb
import numpy as np
import os
import json
import scipy.misc

parser = argparse.ArgumentParser(description='Parse arguments.')
parser.add_argument('--min', type=int, help='min id')
parser.add_argument('--max', type=int, default=0, help='max id')
parser.add_argument('--matfile', type=str, default='all')
parser.add_argument('--split_name', type=str, default='val')
parser.add_argument('--onecategory', type=bool, default=False)
args = parser.parse_args()

single_category=args.onecategory
nyudir = osp.join('/nfs.yoda/imisra/nileshk/', 'nyud2')
split_file = osp.join(nyudir,'splits/','nyu_split.pkl')
matfile = osp.join(nyudir,'matfiles', args.matfile)
pose_data = sio.loadmat(matfile, struct_as_record=False, squeeze_me=True)['data']
outdata_dir = osp.join(nyudir, 'annotations')
with open(split_file) as f:
    splits = pkl.load(f)


def get_object_categories():
    if single_category:
        object_class2index = {'object' : 1}
    else:
        object_class2index = {'bed' : 1, 'sofa' :2, 'table' :3, 
        'chair':4 , 'desk':5,} ## Television is not a classs.
    return object_class2index

def default_annotation():
    annotation = {}
    annotation['info'] = {'description' : 'NYUv2 in Coco format'}
    annotation['licenses'] = {}
    annotation['images'] = []
    annotation['annotations'] = []
    annotation['categories'] = []
    return annotation

def create_instance_json(data, splits, split_name):
    json_file = osp.join(outdata_dir, 'instances_1class_{}2017.json'.format(split_name))
    annotations = default_annotation()
    object_class2index = get_object_categories()
    for obj_class in object_class2index:
        annotations['categories'].append({'supercategory' : 'furniture',
         'id' : object_class2index[obj_class],
         'name' : obj_class})

    img_list = []
    if 'train' in split_name:
        img_list.extend(splits['train'])
    if 'val' in split_name:
        img_list.extend(splits['val'])
    if 'small' in split_name:
        img_list = img_list[0:10]
    datalen = len(pose_data)
    for ix in range(0, datalen):
        if isinstance(pose_data[ix], np.ndarray) and len(pose_data[ix]) == 0:
            continue;
        else:
            img_data = pose_data[ix]
            cls_data = pose_data[ix].cls
        imName = img_data.imName
        if "{}.png".format(imName) not in img_list:
            continue
        img_id = int(imName.split('_')[1])
        ann_img = {}
        ann_img['file_name'] = "{}.png".format(imName)
        ann_img['id'] = img_id
        image = scipy.misc.imread(osp.join(nyudir, 'images', '{}.png'.format(imName)))
        ann_img['height'] = image.shape[0]
        ann_img['width'] = image.shape[1]
        annotations['images'].append(ann_img)


        boxInfo = img_data.boxInfo
        if type(boxInfo) is not np.ndarray:
            boxInfo = np.array([boxInfo])

        objects_per_image = len(boxInfo)
        object_array = []
        out_data = {}

        if type(cls_data) != np.ndarray:
            cls_data = [cls_data]


        for ox in range(objects_per_image):
            ann_obj = {}
            object_id = 1000*img_id + ox 
            ann_obj['id'] = object_id
            bbox = boxInfo[ox].bbox.tolist()
            ann_obj['bbox'] = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
            try:
                if single_category:
                    ann_obj['category_id'] = 1
                else:
                    ann_obj['category_id'] = object_class2index[cls_data[ox]]
                # if cls_data[ox] == 'television':
                #     pdb.set_trace()
            except KeyError:
                pdb.set_trace()
            ann_obj['iscrowd'] = 0
            ann_obj['image_id'] = img_id
            ann_obj['segmentation'] = [[bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]]
            ann_obj['area'] = (bbox[2] -bbox[0]) * (bbox[3] - bbox[1])
            annotations['annotations'].append(ann_obj)
    with open(json_file, 'w') as outfile:
        json.dump(annotations, outfile)
    return

if not osp.exists(outdata_dir):
    os.makedirs(outdata_dir)

create_instance_json(pose_data, splits, args.split_name)
# datalen = len(pose_data)
# for ix in range(0, datalen):
#     print(ix)
#     if isinstance(pose_data[ix], np.ndarray) and len(pose_data[ix]) == 0:
#         continue;
#     else:
#         img_data = pose_data[ix]
#         cls_data = pose_data[ix].cls
  
#     filename = img_data.imName
#     boxInfo = img_data.boxInfo
#     if type(boxInfo) is not np.ndarray:
#         boxInfo = np.array([boxInfo])
#     objects_per_image = len(boxInfo)
#     object_array = []
#     out_data = {}
#     for ox in range(objects_per_image):
#         object_props = {}
#         object_props['basemodel'] = boxInfo[ox].basemodel
#         object_props['pose_top_view'] = np.array(boxInfo[ox].pose_top_view, copy=True)
#         object_props['scale'] = np.array(boxInfo[ox].scale, copy=True)
#         object_props['trans'] = np.array(boxInfo[ox].trans, copy=True)
#         object_props['bbox'] = np.array(boxInfo[ox].bbox, copy=True)
#         object_props['derekInd'] = boxInfo[ox].derekInd
#         object_props['pose_full'] = np.array(boxInfo[ox].pose_full, copy=True)
#         object_props['cls'] = cls_data[ox]
#         object_array.append(object_props)

#     out_data['objects'] = object_array
#     out_data['imName'] = img_data.imName
#     out_data['roomR'] = np.array(img_data.roomR, copy=True)
#     out_data['roomOrigin'] = np.array(img_data.roomOrigin, copy=True)
#     out_data['cameraR'] = np.array(img_data.camera.R, copy=True)
#     out_data['cameraOrigin'] = np.array([img_data.camera.Origin.x, img_data.camera.Origin.y, img_data.camera.Origin.z])
#     out_data['cameraK'] = np.array(img_data.camera.K, copy=True)
#     with open(osp.join(outdata_dir, "{}.pkl".format(filename)), 'wb') as f:
#         pkl.dump(out_data, f)
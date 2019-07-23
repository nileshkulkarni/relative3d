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
import pdb

parser = argparse.ArgumentParser(description='Parse arguments.')
parser.add_argument('--min', type=int, help='min id')
parser.add_argument('--max', type=int, default=0, help='max id')
parser.add_argument('--matfile', type=str, default='all')
args = parser.parse_args()

nyudir = osp.join('/nfs.yoda/imisra/nileshk/', 'nyud2')
matfile = osp.join(nyudir,'matfiles', args.matfile)
pose_data = sio.loadmat(matfile, struct_as_record=False, squeeze_me=True)['data']
outdata_dir = osp.join(nyudir, 'img_data')

if not osp.exists(outdata_dir):
    os.makedirs(outdata_dir)

datalen = len(pose_data)

for ix in range(0, datalen):
    print(ix)
    if isinstance(pose_data[ix], np.ndarray) and len(pose_data[ix]) == 0:
        continue;
    else:
        img_data = pose_data[ix]
        cls_data = pose_data[ix].cls
  
    filename = img_data.imName
    boxInfo = img_data.boxInfo
    if type(boxInfo) is not np.ndarray:
        boxInfo = np.array([boxInfo])
    objects_per_image = len(boxInfo)
    object_array = []
    out_data = {}

    if type(cls_data) != np.ndarray:
        cls_data = [cls_data]

    for ox in range(objects_per_image):
        object_props = {}
        object_props['basemodel'] = boxInfo[ox].basemodel
        object_props['pose_top_view'] = np.array(boxInfo[ox].pose_top_view, copy=True)
        object_props['scale'] = np.array(boxInfo[ox].scale, copy=True)
        object_props['trans'] = np.array(boxInfo[ox].trans, copy=True)
        object_props['bbox'] = np.array(boxInfo[ox].bbox, copy=True)
        object_props['derekInd'] = boxInfo[ox].derekInd
        object_props['pose_full'] = np.array(boxInfo[ox].pose_full, copy=True)
        object_props['cls'] = cls_data[ox]
        object_array.append(object_props)

    out_data['objects'] = object_array
    out_data['imName'] = img_data.imName
    out_data['roomR'] = np.array(img_data.roomR, copy=True)
    out_data['roomOrigin'] = np.array(img_data.roomOrigin, copy=True)
    out_data['cameraR'] = np.array(img_data.camera.R, copy=True)
    out_data['cameraOrigin'] = np.array([img_data.camera.Origin.x, img_data.camera.Origin.y, img_data.camera.Origin.z])
    out_data['cameraK'] = np.array(img_data.camera.K, copy=True)
    with open(osp.join(outdata_dir, "{}.pkl".format(filename)), 'wb') as f:
        pkl.dump(out_data, f)








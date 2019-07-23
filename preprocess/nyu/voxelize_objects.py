
'''
Converts the mat data mesh for object to vox
'''

import os.path as osp
import argparse
import scipy.io as sio
import pdb
import numpy as np
import os
import sys
#sys.path.append('/home/nileshk/3d/external/binvox')

code_dir=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../')) 

nyu_dir = '/nfs.yoda/imisra/nileshk/nyud2/'
binvox_dir = osp.join(code_dir, 'external','binvox')
binvox_exec_file = osp.join(binvox_dir, 'binvox')

import sys
sys.path.insert(0, osp.join(code_dir ,'external/binvox/'))
import binvox_rw


def convert_mat_to_obj(mat_file, obj_file):
    object_mat = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)

    all_faces = np.zeros((0,3))
    all_vertices = np.zeros((0,3))
    for comp in object_mat['comp']:
        f = comp.faces.reshape(-1, 3).astype(np.float32)
        v = comp.vertices
        f = f + len(all_vertices)
        all_vertices = np.concatenate([all_vertices, v])
        all_faces = np.concatenate([all_faces, f])

    all_faces = all_faces.astype(np.int)


    with open(obj_file, 'w') as fout:
        for vert in all_vertices:
            fout.write('v {}, {}, {}\n'.format(vert[0], vert[1], vert[2]))
        for f in all_faces:
            fout.write('f {}, {}, {}\n'.format(f[0], f[1], f[2]))
    return





grid_size = 64
parser = argparse.ArgumentParser(description='Parse arguments.')
parser.add_argument('--min', type=int, help='min id')
parser.add_argument('--max', type=int, default=0, help='max id')
parser.add_argument('--matfile', type=str, default='all')
args = parser.parse_args()
dc1 = 'find {} -name "*.binvox" -type f -delete'.format(osp.join(nyu_dir,'object_obj'))
dc2 = 'find {} -name "*.mat" -type f -delete'.format(osp.join(nyu_dir,'object_obj'))
os.system(dc1)
os.system(dc2)
object_ids = [name.replace(".mat","") for name in os.listdir(osp.join(nyu_dir, 'object'))]

n_objects = len(object_ids)
obj_dir = osp.join(nyu_dir, 'object_obj')
if not osp.exists(obj_dir):
    os.makedirs(obj_dir)
# n_objects = 2
for ix in range(n_objects):
    obj_id = object_ids[ix]
    print(obj_id)
    obj_file = osp.join(nyu_dir, 'object_obj', obj_id + ".obj")
    mat_file = osp.join(nyu_dir, 'object', obj_id + ".mat") 
    convert_mat_to_obj(mat_file, obj_file) 
    binvox_file_interior = osp.join(obj_dir, obj_id + '.binvox')
    binvox_file_surface = osp.join(obj_dir, obj_id + '_1.binvox')

    cmd_interior = '{} -cb -d {} {}'.format(binvox_exec_file, grid_size, osp.join(obj_dir, obj_id + '.obj'))
    cmd_surface = '{} -cb -e -d {} {}'.format(binvox_exec_file, grid_size, osp.join(obj_dir, obj_id + '.obj'))
    os.system(cmd_interior)
    os.system(cmd_surface)
    with open(binvox_file_interior, 'rb') as f0:
        with open(binvox_file_surface, 'rb') as f1:
            vox_read_interior = binvox_rw.read_as_3d_array(f0)
            vox_read_surface = binvox_rw.read_as_3d_array(f1)

            #need to add translation corresponding to voxel centering
            shape_vox = vox_read_interior.data.astype(np.bool) + vox_read_surface.data.astype(np.bool)
            if(np.max(shape_vox) > 0):
                Xs, Ys, Zs = np.where(shape_vox)
                trans_centre = np.array([1.0*np.min(Xs)/(np.size(shape_vox,0)), 1.0*np.min(Ys)/(np.size(shape_vox,1)), 1.0*np.min(Zs)/(np.size(shape_vox,2)-1)] )
                translate = vox_read_surface.translate - trans_centre*vox_read_surface.scale
                sio.savemat(osp.join(obj_dir, obj_id + '.mat'), {'voxels' : shape_vox, 'scale' : vox_read_surface.scale, 'translation' : translate})



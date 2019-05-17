from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import glob
import operator
import json
import pdb
'''
## Needs the both directories to have undergone an evaluation.
python -m factored3d.benchmark.suncg.suncg_compare_results --base_dir factored3d/cachedir/evaluation/box3d_base/val/box3d_base_small_rel_loc/ --improved_dir factored3d/cachedir/evaluation/box3d_base/val/box3d_base_small_rel_loc_all/
'''
flags.DEFINE_string('id', 'default', 'Plot string')

flags.DEFINE_string('base_dir', None, 'Baseline results')
flags.DEFINE_string('improved_dir', None, 'Improved results')

flags.DEFINE_string('key', 'trans', 'results key')
FLAGS = flags.FLAGS

def read_json(json_file):
    with open(json_file) as f:
        bench = json.load(f)
    return bench['bench']


def get_median_error(values):
    return np.median(np.array(values))

def get_mean_error(values):
    return np.mean(np.array(values))


def compare_bench(base_bench, improved_bench, key):
    # base_trans = base_bench['trans']
    # improved_trans = improved_bench['trans']
    # pdb.set_trace()
    base_trans = base_bench[key]
    improved_trans = improved_bench[key]
    ## sum the error in the image and see if the error has decreased
    base_median = get_mean_error(base_trans)
    imporved_median = get_mean_error(improved_trans)
    try:
        image_name = base_bench['image_name']
    except KeyError:
        house_name = base_bench['house_name']
        view_id = base_bench['view_id']
        image_name = "{}_{}".format(house_name, view_id)
        improved_bench['image_name'] = "{}_{}".format(improved_bench['house_name'], improved_bench['view_id'])
    assert image_name == improved_bench['image_name'], 'incorrect bench comparisons'

    return image_name, base_median, imporved_median

def main(_):
    key = FLAGS.key
    base_dir  =  FLAGS.base_dir
    improved_dir = FLAGS.improved_dir
    better_than_baseline = {}  ## maintain house_name _view_id, error_value
    worse_than_baseline = {}
    all_names = {}
    ## read the json files from base and improved and compare corresponding results
    thresh = 0.1
    if key == 'rot':
        thresh = 10
    if key == 'scales':
        thresh = 0.01
    json_files =  glob.glob(osp.join(base_dir, 'eval_result_*.json'))
    for jsn_file in json_files:
        jsn_file = osp.basename(jsn_file)
        base_json = osp.join(base_dir, jsn_file)
        improved_json = osp.join(improved_dir, jsn_file)
        image_name, median_err_base, median_err_improved = compare_bench(read_json(base_json), read_json(improved_json), key)
        all_names['{}'.format(image_name)] = 1
        if (median_err_base - median_err_improved) > thresh:
            better_than_baseline['{}'.format(image_name)] = median_err_base - median_err_improved
        elif (median_err_improved - median_err_base) > thresh:
            # print('{}_{} {}'.format(house_name, view_id, median_err_improved - median_err_base))
            worse_than_baseline['{}'.format(image_name)] = median_err_improved - median_err_base


    ## sort by errors and dump to eval_results for the improved dir.
    sorted_better = sorted(better_than_baseline.items(), key=operator.itemgetter(1), reverse=True)
    sorted_worse = sorted(worse_than_baseline.items(), key=operator.itemgetter(1), reverse=True)

    with open(osp.join(improved_dir,'better_{}.txt'.format(key)), 'w') as f:
        for s in sorted_better[0:50]:
            f.write('{}\n'.format(s[0]))

    with open(osp.join(improved_dir,'worse_{}.txt'.format(key)), 'w') as f:
        for s in sorted_worse[0:50]:
            f.write('{}\n'.format(s[0]))
    
    pdb.set_trace()
    with open(osp.join(improved_dir,'all.txt'.format(key)), 'w') as f:
        for s in all_names.keys():
            f.write('{}\n'.format(s))
    

if __name__ == '__main__':
    app.run(main)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags

flags.DEFINE_integer('roi_size', 4, 'RoI feat spatial size.')
flags.DEFINE_integer('nz_shape', 20, 'Number of latent feat dimension for shape prediction')
flags.DEFINE_integer('nz_feat', 300, 'RoI encoded feature size')
flags.DEFINE_boolean('use_context', True, 'Should we use bbox + full image features')
flags.DEFINE_boolean('pred_voxels', True, 'Predict voxels, or code instead')
flags.DEFINE_boolean('classify_rot', True, 'Classify rotation, or regress quaternion instead')

flags.DEFINE_integer('nz_rot', 24, 'Number of outputs for rot prediction. Value overriden in code.')
flags.DEFINE_boolean('use_spatial_map', False, 'Add spatial map after resnet')
flags.DEFINE_boolean('var_gmm_rot', False, 'predict Variance for  GMM abs rotation')
flags.DEFINE_boolean('gmm_dir', False, 'Use GMM for abs rotation')

flags.DEFINE_boolean('classify_dir', True, 'Classify direction, or regress direction instead')
flags.DEFINE_integer('nz_rel_dir', 24, 'Number of outputs for relative direction prediction')
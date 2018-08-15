#!/usr/bin/python3

import os
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils

load_fn = data_utils.load_cls_train_val_BU3D
map_fn = None
save_ply_fn = None

num_class = 6

sample_num = 3200

#cannot be set to 1 if training
batch_size = 12

num_epochs = 512

step_val = 2 * 648/batch_size

learning_rate_base = 0.002
decay_steps = 40*step_val
decay_rate = 0.5
learning_rate_min = 1e-6

weight_decay = 0.0

jitter = 0.0
jitter_val = 0.0

rotation_range = [0, 0, 0, 'u']
rotation_range_val = [0, 0, 0, 'u']
order = 'rxyz'

scaling_range = [0, 0, 0, 'g']
scaling_range_val = [0, 0, 0, 'u']

x = 2

# K, D, P, C
# knn, dilatation, points, channels
xconv_params = [(8, 1, -1, 16 * x),
                (16, 2, 1600, 32 * x),
                (24, 2, 800, 64 * x),
                (36, 3, 400, 128 * x),
                (36, 3, 200, 256 * x)]

# C, dropout_rate
fc_params = [(128 * x, 0.0), (64 * x, 0.0)]

with_fps = False

optimizer = 'adam'
epsilon = 1e-2

data_dim = 10
use_extra_features = True
with_normal_feature = False
with_X_transformation = True
sorting_method = None

keep_remainder = True

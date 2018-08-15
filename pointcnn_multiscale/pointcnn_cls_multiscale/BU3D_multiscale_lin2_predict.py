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

sample_num = 5000

#cannot be set to 1 if training
batch_size = 1

num_epochs = 512

n_train_samples = 576
step_val = 2 * n_train_samples / batch_size

learning_rate_base = 0.002
decay_steps = 40*step_val
decay_rate = 0.5
learning_rate_min = 1e-6

weight_decay = 1e-6

jitter = 0.0
jitter_val = 0.0

rotation_range = [0, 0, 0, 'u']
rotation_range_val = [0, 0, 0, 'u']
order = 'rxyz'

scaling_range = [0, 0, 0, 'g']
scaling_range_val = [0, 0, 0, 'u']

x = 2

# K, D, P, C
# knn, dilation, points, channels
#xconv_params = [[(12, 1, -1, 5 * x),(12, 2, -1, 5 * x),(12, 3, -1, 5 * x)],
#                [(18, 2, 2400, 10 * x),(18, 3, 2400, 10 * x),(18, 4, 2400, 10 * x)],
#                [(20, 3, 800, 20 * x),(20, 4, 800, 20 * x),(20, 5, 800, 20 * x)],
#                [(25, 2, 200, 40 * x),(25, 4, 200, 40 * x),(25, 6, 200, 40 * x)]]

# original
#xconv_params = [(8, 1, -1, 16 * x),
#                (16, 2, 2400, 32 * x),
#                (24, 2, 800, 64 * x),
#                (36, 3, 200, 128 * x)]

xconv_params = [[(8, 1, -1, 8 * x), (8, 2, -1, 8 * x), (8, 3, -1, 8 * x)],
                [(16, 2, 2400, 32 * x)],
                [(24, 2, 800, 64 * x)],
                [(36, 3, 200, 128 * x)]]

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

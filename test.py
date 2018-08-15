import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import tensorflow as tf
import pointfly as pf

def main():
    num_class = 6

    pts_fts = tf.placeholder(tf.float32, [2,1,7])
    labels = tf.placeholder(tf.int32, [2,])
 
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool, name='is_training')

    features_hd = pts_fts#pf.dense(pts_fts, 16, 'features_hd', is_training)
    #fc = pf.dense(features_hd, 128, 'fc', is_training, with_bn=True)
    dense = tf.layers.dense(features_hd, units=16, activation=tf.nn.elu,
                            kernel_initializer=tf.glorot_uniform_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))
    fc = tf.layers.batch_normalization(dense, momentum=0.9, training=is_training,
                                         beta_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                         gamma_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))
    logits = pf.dense(fc, num_class, 'logits', is_training, with_bn=False, activation=None)
    probs = tf.nn.softmax(logits, name='probs')

    labels_2d = tf.expand_dims(labels, axis=-1, name='labels_2d')
    labels_tile = tf.tile(labels_2d, (1, tf.shape(probs)[1]), name='labels_tile')

    #TODO
    labels_tile = labels
    logits = tf.reduce_mean(logits, axis=1)
   
    #TODO
    loss_op = tf.losses.sparse_softmax_cross_entropy(labels=labels_tile, logits=logits)
    #loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=labels_tile, logits=logits)

    lr_exp_op = tf.train.exponential_decay(0.001, global_step, 1000,
                                           0.5, staircase=True)
    lr_clip_op = tf.maximum(lr_exp_op, 1e-6)
    reg_loss = 0.0 * tf.losses.get_regularization_loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op, epsilon=1e-2)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_op + reg_loss, global_step=global_step)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init_op)
        pts_fts_v = [np.array([[[2,3,5,1,6,4,8]], [[2,6,7,2,4,7,4]]]).astype(np.float32), \
                     np.array([[[2,6,7,2,4,7,4]], [[2,3,5,1,6,4,8]]]).astype(np.float32)]
        labels_v = [np.array([0,3]).astype(np.int32), np.array([3,0]).astype(np.int32)]
        for idx in range(10000):
            out_val = \
                sess.run([dense, fc, train_op, loss_op],
                         feed_dict={
                             pts_fts: pts_fts_v[idx%2],
                             labels: labels_v[idx%2],
                             is_training: True,
                         })
            print('dense')
            print(out_val[0])
            print('fc')
            print(out_val[1])
            print(out_val[-1])


if __name__ == '__main__':
    main()

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
import os
import time
 
path = './BU3D_cls'
log_fn = [os.path.join(path, fd, 'log.txt') for fd in os.listdir(path) if fd.startswith('pointcnn_cls')]
log_fn.sort()
log_size_old = [os.path.getsize(fn) for fn in log_fn]
time.sleep(1)
log_size_new = [os.path.getsize(fn) for fn in log_fn]

for fi,fn in enumerate(log_fn):
    if log_size_old[fi] != log_size_new[fi]:
        continue
    fid = open(fn, 'r')
    lines = fid.readlines()
    fid.close()
    
    train_loss = []
    valid_loss = []
    lid = 0
    epoch_train_loss = []
    idle_st = 0
    train_st = 1
    state = idle_st
    while lid < len(lines):
        lsp = re.split('\s+', lines[lid].strip())
        lid += 1
        tag = lsp[1].split('[')[-1][0]
        if tag == 'T':
            if state == idle_st:
                epoch_train_loss = []
                state = train_st
            epoch_train_loss.append(eval(lsp[4]))
        elif tag == 'V' and state == train_st:
            train_loss.append(np.array(epoch_train_loss).mean())
            valid_loss.append(eval(lsp[4]))
            state = idle_st
    
    fig = plt.figure()
    x = [i for i in range(len(train_loss))]
    plot(x,train_loss,'-b')
    plot(x,valid_loss,'-g')
     
    save_path = os.path.join(os.path.split(fn)[0], 'loss.jpg')
    savefig(save_path)
    print('Save loss.jpg in %s' % save_path)

import os
import os.path as osp
import sys


sys.path.append(os.path.abspath("."))

class_list = ['dig', 'knock', 'shake', 'background', 'water', 'walk']
path = 'data/das6/train'

with open('data/das6/filelist/opendas_train.txt', "w") as f:
    for i in range(len(class_list)):
        cla = class_list[i]
        print('class:', cla)
        files = os.listdir(osp.join(path, cla))
        files.sort()
        for file in files:
            f.write('{:} {:}\n'.format(osp.join(path, cla, file), i))

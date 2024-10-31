import random
import numpy as np
import sys
import os


sys.path.append(os.path.abspath("."))

random.seed(2024)

class_list = ['dig', 'knock', 'shake', 'walk', 'background', 'water']
path = 'data/das6/train'

n_share = 3
n_source_private = 0
n_source = n_share + n_source_private
source_list = class_list[:n_source]
target_list = class_list[:n_share] + class_list[n_source:]

print(source_list)
print(len(source_list))
print(target_list)
print(len(target_list))

labels = []
with open('data/das6/filelist/train.txt', "r") as f:
    for line in f.readlines():
        _, label = line.split()
        label = int(label)
        labels.append(label)
print('length of the train dataset:',len(labels))

labels = np.array(labels)
labeled_idx = []
val_idx = []
unlabeled_idx = []

# percent of labeled and validation data
label_per_class = 24
val_per_class = 0

for i in range(n_source):
    idx = np.where(labels == i)[0]
    num_class = len(idx)
    print(num_class)
    idx = np.random.choice(idx, label_per_class + val_per_class, False)
    labeled_idx.extend(idx[:label_per_class])
    val_idx.extend(idx[label_per_class:])

labeled_idx = np.array(labeled_idx)
np.random.shuffle(labeled_idx)

unlabeled_idx = np.array(range(len(labels)))
unlabeled_idx = [idx for idx in unlabeled_idx if idx not in labeled_idx]
unlabeled_idx = [idx for idx in unlabeled_idx if idx not in val_idx]

print('有标签数据的长度:',len(labeled_idx))
print('验证数据的长度', len(val_idx))
print('无标签数据的长度', len(unlabeled_idx))

with open('data/das6/filelist/train.txt', "r") as f0:
    with open(f'data/das6/filelist/train_labeled_{int(n_share*label_per_class)}.txt', "w") as f1:
        with open('data/das6/filelist/val.txt', "w") as f2:
            with open(f'data/das6/filelist/train_unlabeled_{int(n_share*label_per_class)}.txt', "w") as f3:
                for i, line in enumerate(f0.readlines()):
                    path, label = line.split()
                    label = int(label)
                    if i in labeled_idx:
                        assert label < n_source, "something wrong!"
                        f1.write(line)
                    elif i in val_idx:
                        assert label < n_source, "something wrong!"
                        f2.write(line)
                    elif i in unlabeled_idx:
                        f3.write(line)
                    else:
                        raise NotImplementedError

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_roc(unk_all, label_all, num_known):
    Y_test = np.zeros(unk_all.shape[0])
    unk_pos = np.where(label_all >= num_known)[0]
    Y_test[unk_pos] = 1
    return roc_auc_score(Y_test, unk_all)


def h_score_compute(label_all, pred_class, class_list):
    per_class_num = np.zeros((len(class_list)))
    per_class_correct = np.zeros((len(class_list))).astype(np.float32)
    for i, t in enumerate(class_list):
        t_ind = np.where(label_all == t)
        correct_ind = np.where(pred_class[t_ind[0]] == t)
        per_class_correct[i] += float(len(correct_ind[0]))
        per_class_num[i] += float(len(t_ind[0]))
    open_class = len(class_list)
    per_class_acc = per_class_correct / per_class_num
    known_acc = per_class_acc[:open_class - 1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    return h_score, known_acc, unknown

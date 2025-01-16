import numpy as np
from collections import deque
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.algorithms.hooks import DistAlignQueueHook, PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool, compute_roc, h_score_compute
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from semilearn.core.utils import plot_cm

class OODMemoryQueue:
    def __init__(self, max_size, score_type):
        """
        OOD 样本队列，存储 MSP 最小的样本
        """
        self.queue = deque(maxlen=max_size)
        self.score_type = score_type

    def enqueue(self, images, logits, k):
        """
        根据 MSP 选择最小的 K 个样本并加入队列

        参数:
        - images: 当前批次的图像样本 (torch.Tensor, shape: [batch_size, C, H, W])
        - logits: 当前批次的 logits (torch.Tensor, shape: [batch_size, num_classes])
        - k: 要加入队列的样本数量 (int)
        """
        # 计算每个样本的 MSP（排除 OOD 类）
        probs = torch.softmax(logits,
                              dim=1)[:, :-1]  # [batch_size, num_classes-1]

        if self.score_type == 'msp':
            msp_scores = probs.max(dim=1).values  # [batch_size]
            # 找出 MSP 最小的 K 个样本的索引
            _, smallest_indices = torch.topk(msp_scores, k=k, largest=False)
        elif self.score_type == 'energy':
            enery_scores = -torch.logsumexp(logits[:, :-1], dim=1)
            _, smallest_indices = torch.topk(enery_scores, k=k, largest=True)
        else:
            raise NotImplementedError

        # 根据索引选择对应的样本
        selected_images = images[smallest_indices]

        # 将选中的样本加入队列
        self.queue.extend(selected_images)

    def get_samples(self, num_samples):
        """
        从队列中随机获取指定数量的样本
        """
        if len(self.queue) < num_samples:
            return []  # 如果队列中样本不足，返回空
        return [
            self.queue[i]
            for i in torch.randint(0, len(self.queue), (num_samples, ))
        ]


class ScoMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # iomatch specified arguments
        self.score_type = 'energy'
        self.use_rot = args.use_rot
        self.Km = 1
        self.Nm = 48
        self.ood_queue = OODMemoryQueue(self.Nm, self.score_type)
        self.id_cutoff = 0.95
        self.ood_cutoff_min = 0.8
        self.warm_epochs = 5

    def set_model(self):
        model = self.net_builder(num_classes=self.num_classes + 1,
                                 pretrained=self.args.use_pretrain,
                                 pretrained_path=self.args.pretrain_path)
        return model

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model = self.net_builder(num_classes=self.num_classes + 1)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):

        num_lb = y_lb.shape[0]
        is_warmup = self.epoch < self.warm_epochs

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        outputs = self.model(inputs)
        logits_x_lb = outputs['logits'][:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)

        # update memory queue
        self.ood_queue.enqueue(x_ulb_w, logits_x_ulb_w, self.Km)

        with self.amp_cm():
            if is_warmup:
                loss_sup_id = F.cross_entropy(logits_x_lb, y_lb)
                loss_total = loss_sup_id

                # 日志记录
                tb_dict = {
                    'train/sup_id_loss': loss_sup_id.item(),
                    'train/sup_ood_loss': 0,
                    'train/unsup_open_loss': 0,
                    'train/unsup_close_loss': 0,
                    'train/total_loss': loss_total.item(),
                }
            else:
                # ########################
                # compute sup loss
                # ########################
                loss_sup_id = ce_loss(logits_x_lb, y_lb, reduction='mean')

                if num_lb > self.Nm:
                    ood_samples = self.ood_queue.get_samples(self.Nm)
                else:
                    ood_samples = self.ood_queue.get_samples(num_lb)

                # print(len(ood_samples))

                if len(ood_samples) > 0:
                    x_ood = torch.stack(ood_samples).cuda(self.gpu)
                    y_ood = torch.full((len(ood_samples), ),
                                       self.num_classes,
                                       dtype=torch.long).cuda(self.gpu)
                    logits_ood = self.model(x_ood)['logits']
                    loss_sup_ood = ce_loss(logits_ood, y_ood, reduction='mean')
                else:
                    loss_sup_ood = torch.tensor(0).to(self.gpu)

                # ########################
                # compute self-training loss
                # ########################
                probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=1)
                confidence, pseudo_labels = probs_x_ulb_w.max(dim=1)
                # with torch.no_grad():
                #     logits_x_ulb_w_tea = self.ema_model(x_ulb_w)['logits']
                #     probs_x_ulb_w = torch.softmax(logits_x_ulb_w_tea, dim=1)
                #     confidence, pseudo_labels = probs_x_ulb_w.max(dim=1)

                # update dynamic threshold
                id_selected = (pseudo_labels < self.num_classes) & (
                    confidence > self.id_cutoff)
                ood_selected = (pseudo_labels == self.num_classes) & (
                    confidence > self.id_cutoff)
                num_id_selected = id_selected.sum().item()
                num_ood_selected = ood_selected.sum().item()

                ood_cutoff = self.id_cutoff
                if num_id_selected > 0:
                    ood_cutoff = (num_ood_selected /
                                  num_id_selected) * self.id_cutoff
                    ood_cutoff = max(self.ood_cutoff_min,
                                     min(ood_cutoff, self.id_cutoff))

                # generate mask
                id_mask = (pseudo_labels < self.num_classes) & (confidence >
                                                                self.id_cutoff)
                ood_mask = (pseudo_labels == self.num_classes) & (confidence >
                                                                  ood_cutoff)
                # id_mask = (pseudo_labels < self.num_classes)
                # ood_mask = (pseudo_labels == self.num_classes)
                mask = id_mask | ood_mask

                # open-set self-training
                logits_x_ulb = torch.cat([logits_x_ulb_w, logits_x_ulb_s],
                                         dim=0)
                pseudo_labels_all = pseudo_labels.repeat(2)
                mask_all = mask.repeat(2)

                # logits_x_ulb = logits_x_ulb_w
                # pseudo_labels_all = pseudo_labels
                # mask_all = mask

                loss_u_open = consistency_loss(logits_x_ulb,
                                               pseudo_labels_all,
                                               'ce',
                                               mask=mask_all)

                # close-set self-training
                logits_id = logits_x_ulb_s[:, :self.num_classes][id_mask]
                pseudo_labels_id = pseudo_labels[id_mask]
                if logits_id.size(0) > 0:
                    loss_u_close = F.cross_entropy(logits_id,
                                                   pseudo_labels_id,
                                                   reduction="mean")
                else:
                    loss_u_close = torch.tensor(0.0).to(self.gpu)

                loss_total = loss_sup_id + loss_sup_ood + loss_u_open + loss_u_close

                tb_dict = {
                    'train/sup_id_loss': loss_sup_id.item(),
                    'train/sup_ood_loss': loss_sup_ood.item(),
                    'train/unsup_open_loss': loss_u_open.item(),
                    'train/unsup_close_loss': loss_u_close.item(),
                    'train/total_loss': loss_total.item(),
                    'train/id_mask_ratio': id_mask.float().mean().item(),
                    'train/ood_mask_ratio': ood_mask.float().mean().item(),
                    'train/mask_ratio': mask.float().mean().item()
                }

        self.call_hook("param_update", "ParamUpdateHook", loss=loss_total)

        return tb_dict

    # @staticmethod
    # def get_argument():
    #     return [
    #         SSL_Argument('--use_rot', str2bool, False),
    #         SSL_Argument('--warm_epochs', int, 1),
    #         SSL_Argument('--Km', int, 1),
    #         SSL_Argument('--Nm', int, 64),
    #         SSL_Argument('--id_cutoff', float, 0.95),
    #         SSL_Argument('--ood_cutoff_min', float, 0.75),
    #     ]

    def evaluate_open(self):
        """
        open-set evaluation function 
        """
        self.model.eval()
        # self.ema.apply_shadow()

        full_loader = self.loader_dict['test']['full']
        total_num = 0.0
        y_true_list = []
        y_pred_closed_list = []
        y_pred_open_list = []
        unk_score_list = []

        class_list = [i for i in range(self.num_classes + 1)]
        print(f"class_list: {class_list}")
        results = {}

        with torch.no_grad():
            for data in full_loader:
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                logits = self.model(x)['logits']
                pred_close = logits[:, :-1].max(1)[1]
                pred_open = logits.max(1)[1]

                softmax_output = F.softmax(logits, dim=1)
                unk_score = softmax_output[:, -1]

                y_true_list.extend(y.cpu().tolist())
                y_pred_closed_list.extend(pred_close.cpu().tolist())
                y_pred_open_list.extend(pred_open.cpu().tolist())
                unk_score_list.extend(unk_score.cpu().tolist())

        y_true = np.array(y_true_list)

        closed_mask = y_true < self.num_classes
        open_mask = y_true >= self.num_classes
        y_true[open_mask] = self.num_classes

        y_pred_closed = np.array(y_pred_closed_list)
        y_pred_open = np.array(y_pred_open_list)
        unk_score_all = np.array(unk_score_list)

        # Closed Accuracy on Closed Test Data
        y_true_closed = y_true[closed_mask]
        y_pred_closed = y_pred_closed[closed_mask]

        closed_acc = accuracy_score(y_true_closed, y_pred_closed)
        closed_precision = precision_score(y_true_closed,
                                           y_pred_closed,
                                           average='macro')
        closed_recall = recall_score(y_true_closed,
                                     y_pred_closed,
                                     average='macro')
        closed_F1 = f1_score(y_true_closed, y_pred_closed, average='macro')
        closed_cfmat = confusion_matrix(y_true_closed,
                                        y_pred_closed,
                                        normalize='true')

        results['c_acc'] = closed_acc
        results['c_precision'] = closed_precision
        results['c_recall'] = closed_recall
        results['c_f1'] = closed_F1
        results['c_cfmat'] = closed_cfmat

        # Open Accuracy on Full Test Data
        open_acc = accuracy_score(y_true, y_pred_open)
        open_precision = precision_score(y_true, y_pred_open, average='macro')
        open_recall = recall_score(y_true, y_pred_open, average='macro')
        open_f1 = f1_score(y_true, y_pred_open, average='macro')
        open_cfmat = confusion_matrix(y_true, y_pred_open, normalize='true')
        auroc = compute_roc(unk_score_all,
                            y_true,
                            num_known=int(self.num_classes))
        h_score, known_acc, unknown_acc = h_score_compute(
            y_true, y_pred_open, class_list)

        results['o_acc'] = open_acc
        results['o_precision'] = open_precision
        results['o_recall'] = open_recall
        results['o_f1'] = open_f1
        results['o_cfmat'] = open_cfmat
        results['o_auroc'] = auroc
        results['o_hscore'] = h_score
        results['o_knownacc'] = known_acc
        results['o_unknownacc'] = unknown_acc

        # self.ema.restore()
        self.model.train()

        return results


    def test_final(self):
        """
        open-set evaluation function 
        """
        self.model.eval()
        # self.ema.apply_shadow()

        full_loader = self.loader_dict['test']['full']
        total_num = 0.0
        y_true_list = []
        y_pred_closed_list = []
        y_pred_open_list = []
        unk_score_list = []

        class_list = [i for i in range(self.num_classes + 1)]
        print(f"class_list: {class_list}")
        results = {}

        with torch.no_grad():
            for data in full_loader:
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                logits = self.model(x)['logits']
                pred_close = logits[:, :-1].max(1)[1]
                pred_open = logits.max(1)[1]

                softmax_output = F.softmax(logits, dim=1)
                unk_score = softmax_output[:, -1]

                y_true_list.extend(y.cpu().tolist())
                y_pred_closed_list.extend(pred_close.cpu().tolist())
                y_pred_open_list.extend(pred_open.cpu().tolist())
                unk_score_list.extend(unk_score.cpu().tolist())

        y_true = np.array(y_true_list)

        closed_mask = y_true < self.num_classes
        open_mask = y_true >= self.num_classes
        y_true[open_mask] = self.num_classes

        y_pred_closed = np.array(y_pred_closed_list)
        y_pred_open = np.array(y_pred_open_list)
        unk_score_all = np.array(unk_score_list)

        # Closed Accuracy on Closed Test Data
        y_true_closed = y_true[closed_mask]
        y_pred_closed = y_pred_closed[closed_mask]

        closed_acc = accuracy_score(y_true_closed, y_pred_closed)
        closed_precision = precision_score(y_true_closed,
                                           y_pred_closed,
                                           average='macro')
        closed_recall = recall_score(y_true_closed,
                                     y_pred_closed,
                                     average='macro')
        closed_F1 = f1_score(y_true_closed, y_pred_closed, average='macro')
        closed_cfmat = confusion_matrix(y_true_closed,
                                        y_pred_closed)
        closed_cfmat_path = osp.join(self.save_dir, 'close_cm')
        plot_cm(closed_cfmat, save_path=closed_cfmat_path)

        results['c_acc'] = closed_acc
        results['c_precision'] = closed_precision
        results['c_recall'] = closed_recall
        results['c_f1'] = closed_F1
        results['c_cfmat'] = closed_cfmat

        # Open Accuracy on Full Test Data
        open_acc = accuracy_score(y_true, y_pred_open)
        open_precision = precision_score(y_true, y_pred_open, average='macro')
        open_recall = recall_score(y_true, y_pred_open, average='macro')
        open_f1 = f1_score(y_true, y_pred_open, average='macro')
        open_cfmat = confusion_matrix(y_true, y_pred_open)
        open_cfmat_path = osp.join(self.save_dir, 'open_cm')
        plot_cm(open_cfmat, save_path=open_cfmat_path)
        auroc = compute_roc(unk_score_all,
                            y_true,
                            num_known=int(self.num_classes))
        h_score, known_acc, unknown_acc = h_score_compute(
            y_true, y_pred_open, class_list)

        results['o_acc'] = open_acc
        results['o_precision'] = open_precision
        results['o_recall'] = open_recall
        results['o_f1'] = open_f1
        results['o_cfmat'] = open_cfmat
        results['o_auroc'] = auroc
        results['o_hscore'] = h_score
        results['o_knownacc'] = known_acc
        results['o_unknownacc'] = unknown_acc

        # self.ema.restore()
        self.model.train()

        return results
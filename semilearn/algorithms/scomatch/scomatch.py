import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.algorithms.hooks import DistAlignQueueHook, PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool, compute_roc, h_score_compute
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class OODMemoryQueue:
    def __init__(self, max_size):
        """
        OOD 样本队列，存储 MSP 最小的样本
        """
        self.queue = deque(maxlen=max_size)

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
        msp_scores = probs.max(dim=1).values  # [batch_size]

        # 找出 MSP 最小的 K 个样本的索引
        _, smallest_indices = torch.topk(msp_scores, k=k, largest=False)

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
        self.use_rot = args.use_rot
        self.Km = 1
        self.Nm = 64
        self.ood_queue = OODMemoryQueue(self.Nm)
        self.id_cutoff = 0.95
        self.ood_cutoff_min = 0.75

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

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
        num_ulb = x_ulb_w.shape[0]

        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][
                    num_lb:].chunk(2)
            else:
                raise NotImplementedError

            # ########################
            # compute sup loss
            # ########################
            loss_sup_id = ce_loss(logits_x_lb, y_lb, reduction='mean')
            ood_samples = self.ood_queue.get_samples(num_lb)
            if ood_samples:
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

            # update dynamic threshold
            id_selected = (pseudo_labels <
                           self.num_classes) & (confidence > self.id_cutoff)
            ood_selected = (pseudo_labels == self.num_classes) & (
                confidence > self.id_cutoff)
            num_id_selected = id_selected.sum().item()
            num_ood_selected = ood_selected.sum().item()
            ood_cutoff = (num_ood_selected / num_id_selected) * self.id_cutoff
            ood_cutoff = max(self.ood_threshod_min, min(ood_cutoff, self.id_cutoff))

            # generate mask
            id_mask = (pseudo_labels < self.num_classes) & (confidence >
                                                            self.id_cutoff)
            ood_mask = (pseudo_labels == self.num_classes) & (confidence >
                                                              ood_cutoff)
            mask = id_mask | ood_mask

            # open-set self-training
            logits_x_ulb = torch.cat([logits_x_ulb_w, logits_x_ulb_s], dim=0)
            pseudo_labels_all = pseudo_labels.reapeat(2)
            mask_all = mask.repeat(2)

            loss_u_open = consistency_loss(logits_x_ulb,
                                           pseudo_labels_all,
                                           'ce',
                                           mask=mask_all)

            # close-set self-training
            logits_id_s = logits_x_ulb_s[:, :self.num_classes]
            loss_u_close = consistency_loss(logits_id_s,
                                            pseudo_labels,
                                            'ce',
                                            mask=id_mask)

            loss_total = loss_sup_id + loss_sup_ood + loss_u_open + loss_u_close

            # update memory queue
            self.ood_queue.enqueue(x_ulb_w, logits_x_ulb_w, self.Km)

        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {
            'train/sup_id_loss': loss_sup_id.item(),
            'train/sup_ood_loss': loss_sup_ood.item(),
            'train/unsup_open_loss': loss_u_open.item(),
            'train/unsup_close_loss': loss_u_close.item(),
            'train/total_loss': loss_total.item(),
        }

        return tb_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--use_rot', str2bool, False),
            SSL_Argument('--Km', int, 1),
            SSL_Argument('--Nm', int, 64),
            SSL_Argument('--id_cutoff', float, 0.95),
            SSL_Argument('--ood_cutoff_min', float, 0.75),
        ]

    def evaluate_open(self):
        """
        open-set evaluation function 
        """
        self.model.eval()
        self.ema.apply_shadow()

        full_loader = self.loader_dict['test']['full']

        total_num = 0.0
        y_true_list = []
        p_list = []
        pred_p_list = []
        pred_hat_q_list = []
        pred_q_list = []
        pred_hat_p_list = []

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

                y_true_list.extend(y.cpu().tolist())

                num_batch = y.shape[0]
                total_num += num_batch

                outputs = self.model(x)
                logits = outputs['logits']
                logits_mb = outputs['logits_mb']
                logits_open = outputs['logits_open']

                # predictions p of closed-set classifier
                p = F.softmax(logits, 1)
                pred_p = p.data.max(1)[1]
                pred_p_list.extend(pred_p.cpu().tolist())

                # predictions hat_q from (closed-set + multi-binary) classifiers
                r = F.softmax(logits_mb.view(logits_mb.size(0), 2, -1), 1)
                tmp_range = torch.arange(0, logits_mb.size(0)).long().cuda()
                hat_q = torch.zeros((num_batch, self.num_classes + 1)).cuda()
                o_neg = r[tmp_range, 0, :]
                o_pos = r[tmp_range, 1, :]
                hat_q[:, :self.num_classes] = p * o_pos
                hat_q[:, self.num_classes] = torch.sum(p * o_neg, 1)
                pred_hat_q = hat_q.data.max(1)[1]
                pred_hat_q_list.extend(pred_hat_q.cpu().tolist())

                # predictions q of open-set classifier
                q = F.softmax(logits_open, 1)
                pred_q = q.data.max(1)[1]
                pred_q_list.extend(pred_q.cpu().tolist())

                # prediction hat_p of open-set classifier
                hat_p = q[:, :self.num_classes] / q[:, :self.num_classes].sum(
                    1).unsqueeze(1)
                pred_hat_p = hat_p.data.max(1)[1]
                pred_hat_p_list.extend(pred_hat_p.cpu().tolist())

                unk_score = q[:, -1]
                unk_score_list.extend(unk_score.cpu().tolist())

        y_true = np.array(y_true_list)
        closed_mask = y_true < self.num_classes
        open_mask = y_true >= self.num_classes
        y_true[open_mask] = self.num_classes

        pred_p = np.array(pred_p_list)
        pred_hat_p = np.array(pred_hat_p_list)
        pred_q = np.array(pred_q_list)
        pred_hat_q = np.array(pred_hat_q_list)

        unk_score_all = np.array(unk_score_list)

        # closed accuracy of p / hat_p on closed test data
        y_true_closed = y_true[closed_mask]
        y_pred_closed = pred_p[closed_mask]

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

        # open accuracy of q / hat_q on full test data
        y_pred_open = pred_q
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

        self.ema.restore()
        self.model.train()

        return results

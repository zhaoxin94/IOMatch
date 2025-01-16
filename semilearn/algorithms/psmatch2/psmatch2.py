import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.algorithms.hooks import DistAlignQueueHook, PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool, compute_roc, h_score_compute
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.mixture import GaussianMixture

CE = nn.CrossEntropyLoss(reduction='none')


class OODMemoryQueue:
    def __init__(self, max_size, score_type):
        """
        OOD 样本队列，存储 MSP 最小的样本
        """
        self.queue = deque(maxlen=max_size)
        self.score_type = score_type

    def enqueue(self, images, w_unknown, k):

        w_unknown = w_unknown.flatten()

        _, indices = torch.topk(w_unknown, k=k, largest=True)

        # 根据索引选择对应的样本
        selected_images = images[indices]

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


class PSMatch2(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # iomatch specified arguments
        self.score_type = 'energy'
        self.use_rot = args.use_rot
        self.Km = 1
        self.Nm = 32
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

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            self.print_fn(f"-------{self.epoch}----------")

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            prob_unknown, _, _ = self.run_separation()
            w_unknown = torch.from_numpy(prob_unknown)
            # w_known = torch.from_numpy(prob_known)
            self.w_unknown = w_unknown.view(-1, 1).cuda(self.gpu)
            # self.w_known = w_known.view(-1, 1).cuda(self.gpu)

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.tb_dict = self.train_step(
                    **self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def run_separation(self):
        self.model.eval()
        all_score = []
        all_index = []

        with torch.no_grad():
            for batch in self.loader_dict['ulb_eval']:
                x_ulb_w = batch['x_ulb_w']
                idx_ulb = batch['idx_ulb']

                outputs = self.model(x_ulb_w)
                logits = outputs['logits']
                outputs = F.softmax(logits, dim=1)

                if self.score_type == 'msp':
                    score = outputs[:, :-1].max(1)[0]
                elif self.score_type == 'energy':
                    score = -torch.logsumexp(logits[:, :-1], dim=1)
                else:
                    raise NotImplementedError

                all_score.append(score)
                all_index.append(idx_ulb)

        all_score = torch.cat(all_score, dim=0)
        all_score = (all_score - all_score.min()) / (all_score.max() -
                                                     all_score.min())
        all_score = all_score.cpu()

        all_index = torch.cat(all_index, dim=0)

        print(all_index)

        assert len(all_index) == self.args.ulb_dest_len, "wrong dataset length"

        # fit a two-component GMM to the entropy
        input_score = all_score.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2,
                              max_iter=10,
                              tol=1e-2,
                              reg_covar=5e-4)

        gmm.fit(input_score)
        prob = gmm.predict_proba(input_score)
        if self.score_type in ['ent', 'energy']:
            prob_unknown = prob[:, gmm.means_.argmax()]
            prob_known = prob[:, gmm.means_.argmin()]
        elif self.score_type in ['msp', 'mls']:
            prob_unknown = prob[:, gmm.means_.argmin()]
            prob_known = prob[:, gmm.means_.argmax()]
        else:
            raise NotImplementedError

        self.model.train()

        return prob_unknown, prob_known, score

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, idx_ulb):

        num_lb = y_lb.shape[0]
        is_warmup = self.epoch < self.warm_epochs

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        outputs = self.model(inputs)
        logits_x_lb = outputs['logits'][:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)

        w_unknown = self.w_unknown[idx_ulb]
        w_known = 1.0 - w_unknown
        # print(w_unknown.shape)

        # update memory queue
        self.ood_queue.enqueue(x_ulb_w, w_unknown, self.Km)

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
                _, pseudo_labels = probs_x_ulb_w.max(dim=1)

                # generate mask
                id_mask = (pseudo_labels < self.num_classes)
                ood_mask = (pseudo_labels == self.num_classes)

                # open-set self-training
                logits_x_ulb = torch.cat([logits_x_ulb_w, logits_x_ulb_s],
                                         dim=0)
                pseudo_labels_all = pseudo_labels.repeat(2)
                id_mask_all = id_mask.repeat(2)
                ood_mask_all = ood_mask.repeat(2)
                w_known_all = w_known.repeat(2, 1)
                w_unknown_all = w_unknown.repeat(2, 1)

                logits_id = logits_x_ulb[id_mask_all]
                logits_ood = logits_x_ulb[ood_mask_all]
                w_known_id = w_known_all[id_mask_all]
                w_unknown_ood = w_unknown_all[ood_mask_all]
                pseudo_labels_id = pseudo_labels_all[id_mask_all]
                pseudo_labels_ood = pseudo_labels_all[ood_mask_all]

                if logits_id.size(0) > 0:
                    loss_u_open_id = (F.cross_entropy(
                        logits_id, pseudo_labels_id, reduction='none') *
                                      w_known_id).mean()
                else:
                    loss_u_open_id = torch.tensor(0.0).to(self.gpu)

                if logits_ood.size(0) > 0:
                    loss_u_open_ood = (F.cross_entropy(
                        logits_ood, pseudo_labels_ood, reduction='none') *
                                       w_unknown_ood).mean()
                else:
                    loss_u_open_ood = torch.tensor(0.0).to(self.gpu)

                loss_u_open = loss_u_open_id + loss_u_open_ood

                # close-set self-training
                logits_close_id = logits_x_ulb_s[:, :self.num_classes][id_mask]
                pseudo_labels_id = pseudo_labels[id_mask]
                w_known_id = w_known[id_mask]
                if logits_close_id.size(0) > 0:
                    loss_u_close = (F.cross_entropy(
                        logits_close_id, pseudo_labels_id, reduction="none") *
                                    w_known_id).mean()
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
                }

        self.call_hook("param_update", "ParamUpdateHook", loss=loss_total)

        return tb_dict

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

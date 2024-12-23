import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.algorithms.hooks import DistAlignQueueHook, PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool, compute_roc, h_score_compute
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .utils import mb_sup_loss


class IOMatchNet(nn.Module):
    def __init__(self, base, num_classes, proj_size=128, use_rot=False):
        super(IOMatchNet, self).__init__()
        self.backbone = base
        self.feat_planes = base.num_features
        self.use_rot = use_rot

        self.mlp_proj = nn.Sequential(*[
            nn.Linear(self.feat_planes, self.feat_planes),
            nn.ReLU(inplace=False),
            nn.Linear(self.feat_planes, proj_size)
        ])

        self.mb_classifiers = nn.Linear(proj_size, num_classes * 2, bias=False)
        self.openset_classifier = nn.Linear(proj_size, num_classes + 1)

        if self.use_rot:
            self.rot_classifier = nn.Linear(self.feat_planes, 4, bias=False)
            nn.init.xavier_normal_(self.rot_classifier.weight.data)

        # initialize the added two classifiers
        nn.init.xavier_normal_(self.mb_classifiers.weight.data)
        nn.init.xavier_normal_(self.openset_classifier.weight.data)
        self.openset_classifier.bias.data.zero_()

    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        feat_proj = self.mlp_proj(feat)
        logits_open = self.openset_classifier(feat_proj)  # (k+1)-way logits
        logits_mb = self.mb_classifiers(feat_proj)  # shape: [bsz, 2K]

        return_dict = {
            'feat': feat,
            'feat_proj': feat_proj,
            'logits': logits,
            'logits_open': logits_open,
            'logits_mb': logits_mb
        }
        if self.use_rot:
            logits_rot = self.rot_classifier(feat)
            return_dict['logits_rot'] = logits_rot

        return return_dict

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


class IOMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # iomatch specified arguments
        self.dist_align = args.dist_align
        self.use_rot = args.use_rot
        self.p_cutoff = args.p_cutoff
        self.q_cutoff = args.q_cutoff
        self.lambda_mb = args.mb_loss_ratio
        self.lambda_op = args.op_loss_ratio

    def set_hooks(self):
        self.register_hook(
            DistAlignQueueHook(num_classes=self.num_classes,
                               queue_length=self.args.da_len,
                               p_target_type='uniform'), "DistAlignHook")
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def set_model(self):
        model = super().set_model()  # backbone
        model = IOMatchNet(model,
                           num_classes=self.num_classes,
                           use_rot=self.args.use_rot)
        return model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = IOMatchNet(ema_model,
                               num_classes=self.num_classes,
                               use_rot=self.args.use_rot)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]

        # print(f'-----number of labeled data per batch:{num_lb}')

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_mb_x_lb = outputs['logits_mb'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][
                    num_lb:].chunk(2)
                logits_open_x_ulb_w, logits_open_x_ulb_s = outputs[
                    'logits_open'][num_lb:].chunk(2)
                logits_mb_x_ulb_w, _ = outputs['logits_mb'][num_lb:].chunk(2)
            else:
                raise ValueError("Bad configuration: use_cat should be True!")

            # supervised losses
            sup_closed_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
            sup_mb_loss = self.lambda_mb * mb_sup_loss(logits_mb_x_lb, y_lb)
            sup_loss = sup_closed_loss + sup_mb_loss

            if self.use_rot:
                x_ulb_r = torch.cat([
                    torch.rot90(x_ulb_w[:num_lb], i, [2, 3]) for i in range(4)
                ],
                                    dim=0)
                y_ulb_r = torch.cat([
                    torch.empty(x_ulb_w[:num_lb].size(0)).fill_(i).long()
                    for i in range(4)
                ],
                                    dim=0).cuda(self.gpu)
                self.bn_controller.freeze_bn(self.model)
                logits_rot = self.model(x_ulb_r)['logits_rot']
                self.bn_controller.unfreeze_bn(self.model)
                rot_loss = ce_loss(logits_rot, y_ulb_r, reduction='mean')
            else:
                rot_loss = torch.tensor(0).to(self.gpu)

            # generator closed-set and open-set targets (pseudo-labels)
            with torch.no_grad():
                p = F.softmax(logits_x_ulb_w, dim=-1)
                targets_p = p.detach()
                if self.dist_align:
                    targets_p = self.call_hook("dist_align",
                                               "DistAlignHook",
                                               probs_x_ulb=targets_p)
                logits_mb = logits_mb_x_ulb_w.view(num_ulb, 2, -1)
                r = F.softmax(logits_mb, 1)
                tmp_range = torch.arange(0, num_ulb).long().cuda(self.gpu)
                out_scores = torch.sum(targets_p * r[tmp_range, 0, :], 1)
                in_mask = (out_scores < 0.5)

                o_neg = r[tmp_range, 0, :]
                o_pos = r[tmp_range, 1, :]
                q = torch.zeros((num_ulb, self.num_classes + 1)).cuda(self.gpu)
                q[:, :self.num_classes] = targets_p * o_pos
                q[:, self.num_classes] = torch.sum(targets_p * o_neg, 1)
                targets_q = q.detach()

            p_mask = self.call_hook("masking",
                                    "MaskingHook",
                                    cutoff=self.p_cutoff,
                                    logits_x_ulb=targets_p,
                                    softmax_x_ulb=False)
            q_mask = self.call_hook("masking",
                                    "MaskingHook",
                                    cutoff=self.q_cutoff,
                                    logits_x_ulb=targets_q,
                                    softmax_x_ulb=False)

            ui_loss = consistency_loss(logits_x_ulb_s,
                                       targets_p,
                                       'ce',
                                       mask=in_mask * p_mask)
            op_loss = consistency_loss(logits_open_x_ulb_s,
                                       targets_q,
                                       'ce',
                                       mask=q_mask)

            if self.epoch == 0:
                op_loss *= 0.0

            unsup_loss = self.lambda_u * ui_loss
            op_loss = self.lambda_op * op_loss

            total_loss = sup_loss + unsup_loss + op_loss + rot_loss

        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {
            'train/s_loss': sup_closed_loss.item(),
            'train/mb_loss': sup_mb_loss.item(),
            'train/ui_loss': ui_loss.item(),
            'train/op_loss': op_loss.item(),
            'train/rot_loss': rot_loss.item(),
            'train/sup_loss': sup_loss.item(),
            'train/unsup_loss': unsup_loss.item(),
            'train/total_loss': total_loss.item(),
            'train/selected_ratio': (in_mask * p_mask).float().mean().item(),
            'train/in_mask_ratio': in_mask.float().mean().item(),
            'train/p_mask_ratio': p_mask.float().mean().item(),
            'train/q_mask_ratio': q_mask.float().mean().item()
        }

        return tb_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_model_ptr'] = self.hooks_dict[
            'DistAlignHook'].p_model_ptr.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(
            self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_model_ptr = checkpoint[
            'p_model_ptr'].cuda(self.args.gpu)
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--use_rot', str2bool, False),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--q_cutoff', float, 0.5),
            SSL_Argument('--da_len', int, 128),
            SSL_Argument('--mb_loss_ratio', float, 1.0),
            SSL_Argument('--op_loss_ratio', float, 1.0),
        ]

    def evaluate_open(self):
        """
        open-set evaluation function 
        """
        self.model.eval()
        # self.ema.apply_shadow()

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

        # self.ema.restore()
        self.model.train()

        return results

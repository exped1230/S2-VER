import pickle

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import os
from train_utils import AverageMeter

from .main_utils import Get_Scalar
from train_utils import ce_loss, wd_loss, EMA, Bn_Controller
import json

from sklearn.metrics import *
from copy import deepcopy
from train_utils import ce_loss
import contextlib


class S2_VER:
    # def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, \
    #              hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, \
                 hard_label=True, t_fn=None, p_fn=None, it=0, tb_log=None, args=None, logger=None):

        super(S2_VER, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        self.model = net_builder(num_classes=num_classes)
        self.ema_model = None

        # self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.lst = [[] for i in range(10)]
        self.abs_lst = [[] for i in range(10)]
        self.clsacc = [[] for i in range(10)]
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.prototype = torch.zeros(self.num_classes, args.low_dim)
        self.prototype = self.prototype.to(args.gpu)
        self.prototype_num = torch.zeros(self.num_classes)
        self.relation_matrix = torch.zeros(self.num_classes, self.num_classes).fill_(1/self.num_classes)
        self.relation_matrix = self.relation_matrix.to(args.gpu)

        self.polarity = torch.Tensor([1, 0, 1, 1, 0, 1, 0, 0]).to(args.gpu)

        self.bn_controller = Bn_Controller()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args, epoch, best_eval_acc, logger=None):

        ngpus_per_node = torch.cuda.device_count()

        # EMA Init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        
        sup_losses = AverageMeter()
        unsup_losses = AverageMeter()
        contrast_losses = AverageMeter()
        total_losses = AverageMeter()
        mask_ratios = AverageMeter()
        distribution_losses = AverageMeter()
        lr_last = 0
        batch_data_time = AverageMeter()
        batch_model_time = AverageMeter()

        pseudo_true_ratios = AverageMeter()

        start_batch.record()

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        iter_num = 0

        for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s0, x_ulb_s1, y_ulb) in zip(self.loader_dict['train_lb'],
                                                                    self.loader_dict['train_ulb']):

            iter_num += 1
            end_batch.record()
            torch.cuda.synchronize()
            batch_data_time.update(start_batch.elapsed_time(end_batch) / 1000)
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s0.shape[0] and num_ulb == x_ulb_s1.shape[0]

            x_lb, x_ulb_w, x_ulb_s0, x_ulb_s1 = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s0.cuda(args.gpu), x_ulb_s1.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)

            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s0, x_ulb_s1))

            # hyper-params for update
            T = self.t_fn(self.it)
            p_cutoff = self.p_fn(self.it)

            # inference and calculate sup/unsup losses
            with amp_cm():
                logits, features = self.model(inputs)
                logits_x_lb = logits[:num_lb]
                # logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                logits_x_ulb_w, logits_x_ulb_s0, logits_x_ulb_s1 = torch.split(logits[num_lb:], num_ulb)

                features_lb = features[:num_lb]
                features_ulb_w, features_ulb_s0, features_ulb_s1 = torch.split(features[num_lb:], num_ulb)

                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

                with torch.no_grad():
                    logits_x_ulb_w = logits_x_ulb_w.detach()
                    features_lb = features_lb.detach()
                    features_ulb_w = features_ulb_w.detach()

                    lb_probs = torch.softmax(logits_x_lb, dim=1)
                    _, lb_guess = torch.max(lb_probs, dim=1)

                    for cur_class in range(self.num_classes):
                        class_mask = (y_lb == cur_class) & (lb_guess == y_lb)
                        if class_mask.sum() == 0:
                            continue
                        cur_class_feature = features_lb[class_mask].sum(0) / class_mask.sum()
                        self.prototype[cur_class] = 0.9 * self.prototype[cur_class] + 0.1 * cur_class_feature
                    if args.update_m == 'L2':
                        L2_dis = torch.norm(self.prototype[:, None] - self.prototype, dim=2, p=2) / 0.5
                        new_relation_matrix = torch.exp(-L2_dis)
                    elif args.update_m == 'L1':
                        L1_dis = torch.sum(self.prototype[:, None] - self.prototype, dim=2) / 0.5
                        new_relation_matrix = torch.exp(-L1_dis)
                    elif args.update_m == 'cos':
                        prototype_tmp = torch.norm(self.prototype, p=2, dim=1, keepdim=True).expand_as(self.prototype) + 1e-12
                        prototype_tmp = self.prototype / prototype_tmp
                        cos_sim = torch.mm(prototype_tmp, prototype_tmp.T) / 0.5
                        new_relation_matrix = torch.exp(cos_sim)

                    new_relation_matrix = new_relation_matrix / torch.sum(new_relation_matrix, dim=1, keepdim=True).expand_as(new_relation_matrix)
                    self.relation_matrix = 0.9 * self.relation_matrix + 0.1 * new_relation_matrix

                    pseudo_LDL = torch.exp(torch.mm(features_ulb_w, self.prototype.t()) / T)
                    pseudo_LDL = pseudo_LDL / pseudo_LDL.sum(1, keepdim=True)
                    pseudo_LDL = torch.mm(pseudo_LDL, self.relation_matrix)

                    ulb_probs = torch.softmax(logits_x_ulb_w, dim=1)

                    ulb_LDL = (1-args.ldl_ratio) * ulb_probs + args.ldl_ratio * pseudo_LDL
                    
                    scores, lbs_u_guess = torch.max(ulb_probs, dim=1)

                    threshold = p_cutoff
                    if args.dynamic_th > 0:
                        ambiguity = -ulb_LDL * torch.log(ulb_LDL+1e-7)
                        ambiguity = ambiguity.sum(1) + 1
                        sign =  torch.tensor(range(8)).repeat(num_ulb, 1).to(args.gpu)
                        polarity_mask = lbs_u_guess.unsqueeze(1) == sign
                        polarity = self.polarity.repeat(num_ulb, 1)[polarity_mask]
                        confident = ulb_LDL * (self.polarity.repeat(num_ulb, 1) == polarity.unsqueeze(1))
                        confident = confident.sum(1)
                        threshold = 1 / (ambiguity * confident)

                        threshold[threshold>1] = 1
                        threshold = args.dynamic_th + (1-args.dynamic_th) * threshold
                        threshold[threshold>0.95] = 0.95

                    mask = scores.ge(threshold)

                    y_ulb = y_ulb.cuda(args.gpu)
                    pseudo_true_ratios.update(((lbs_u_guess == y_ulb) * mask).sum() / (mask.sum()+1e-7))

                    Q = torch.mm(ulb_LDL, ulb_LDL.t())
                    Q.fill_diagonal_(1)
                    pos_mask = (Q > args.noise_th).float()
                    Q = Q * pos_mask
                    Q = Q / Q.sum(1, keepdim=True)

                sim = torch.exp(torch.mm(features_ulb_s0, features_ulb_s1.t()) / T)
                sim_probs = sim / sim.sum(1, keepdim=True)

                loss_contrast = F.kl_div(torch.log(sim_probs), Q, reduction='batchmean')

                unsup_loss = F.cross_entropy(logits_x_ulb_s0, lbs_u_guess, reduction='none') * mask
                unsup_loss = unsup_loss.mean()

                if args.dis_ce:
                    unsup_dis_loss = ce_loss(logits_x_ulb_s1, ulb_LDL, use_hard_labels=False)
                else:
                    unsup_dis_loss = F.kl_div(F.log_softmax(logits_x_ulb_s1, dim=1), ulb_LDL, reduction='none').sum(1)
                unsup_dis_loss = (unsup_dis_loss * mask).mean()

                total_loss = sup_loss + self.lambda_u * unsup_loss + args.lam_c * loss_contrast + args.lam_d * unsup_dis_loss
                # total_loss = sup_loss + self.lambda_u * unsup_loss

            sup_losses.update(sup_loss.cpu().detach())
            unsup_losses.update(unsup_loss.cpu().detach())
            contrast_losses.update(loss_contrast.cpu().detach())
            # contrast_losses.update(0)
            distribution_losses.update(unsup_dis_loss.cpu().detach())
            # distribution_losses.update(0)
            total_losses.update(total_loss.cpu().detach())
            mask_ratios.update(mask.float().mean().cpu().detach())
            
            lr_last = self.optimizer.param_groups[0]['lr']
            
            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()
            batch_model_time.update(start_run.elapsed_time(end_run) / 1000)

            # self.it += 1
            start_batch.record()

        self.print_fn(self.relation_matrix)

        self.print_fn("Epoch {}/{} train: data time: {}, model time: {}, last lr: {}, labeled loss: {}, unlabeled loss: {}, distribution loss: {} contrastive loss: {}, total_loss: {}, mask ratio: {}, pseudo label correct ratio: {}".
                      format(epoch, args.epoch, batch_data_time.avg, batch_model_time.avg, lr_last, sup_losses.avg, unsup_losses.avg, distribution_losses.avg, contrast_losses.avg, total_losses.avg, mask_ratios.avg, pseudo_true_ratios.avg))

        eval_dict = self.evaluate(args=args)
        best_eval_acc = max(best_eval_acc, eval_dict['eval/top-1-acc'])
        self.print_fn("Epoch {}/{} test: test loss: {}, top-1 acc: {}, top-5 acc: {}, best top-1 acc: {}".format(
            epoch, args.epoch, eval_dict['eval/loss'], eval_dict['eval/top-1-acc'], eval_dict['eval/top-5-acc'], best_eval_acc
        ))
        
        save_path = os.path.join(args.save_dir, args.save_name)

        if eval_dict['eval/top-1-acc'] == best_eval_acc:
            self.save_model('model_best.pth', save_path)
        return eval_dict['eval/top-1-acc']

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits, _ = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().detach().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().detach().tolist())
            total_loss += loss.cpu().detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5}

    def save_model(self, save_name, save_path):
        # if self.it < 1000000:
        #     return
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model},
                   save_filename)
        if self.num_classes == 10:
            tb_path = os.path.join(save_path, 'tensorboard')
            if not os.path.exists(tb_path):
                os.makedirs(tb_path, exist_ok=True)
            with open(os.path.join(save_path, 'tensorboard', 'lst_fix.pkl'), 'wb') as f:
                pickle.dump(self.lst, f)
            with open(os.path.join(save_path, 'tensorboard', 'abs_lst.pkl'), 'wb') as h:
                pickle.dump(self.abs_lst, h)
            with open(os.path.join(save_path, 'tensorboard', 'clsacc.pkl'), 'wb') as g:
                pickle.dump(self.clsacc, g)
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn('model loaded')

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


if __name__ == "__main__":
    pass

# import needed library
import os
import logging
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import net_builder, get_logger, count_parameters, over_write_args_from_file
from train_utils import TBLog, get_optimizer, get_cosine_schedule_with_warmup
from models.main.main import S2_VER
from datasets.ssl_dataset import SSL_Dataset, ImageNetLoader, Emotion_SSL_Dataset
from datasets.data_utils import get_data_loader

'''
FI数据集与cifar10的切换: net部分，调整stride；数据集部分，SSL/EmotionSSL
'''


def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args.gpu, args)


# def main_worker(gpu, ngpus_per_node, args):
def main_worker(gpu, args):
    '''
    main_worker is conducted on each GPU.
    '''

    global best_acc1
    args.gpu = gpu

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")

    logger.info(f"  Task = {args.dataset}@{args.num_labels}")

    # SET CoMatch: class CoMatch in models.comatch
    args.bn_momentum = 1.0 - 0.999
    
    _net_builder = net_builder('ResNet50', False, None, is_remix=False, dim=args.low_dim, proj=True)

    model = S2_VER(_net_builder,
                     args.num_classes,
                     args.ema_m,
                     args.T,
                     args.p_cutoff,
                     args.ulb_loss_ratio,
                     args.hard_label,
                     tb_log=tb_log,
                     args=args,
                     logger=logger)

    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')

    # SET Optimizer & LR Scheduler
    ## construct SGD and cosine lr scheduler
    optimizer = get_optimizer(model.model, args.optim, args.lr, args.momentum, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                args.num_train_iter*args.epoch,
                                                num_warmup_steps=args.num_train_iter * 0)
    ## set SGD and cosine lr on CoMatch 
    model.set_optimizer(optimizer, scheduler)

    # SET Devices for (Distributed) DataParallel
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.model = model.model.cuda(args.gpu)

    else:
        model.model = torch.nn.DataParallel(model.model).cuda()

    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {args}")

    cudnn.benchmark = True

    # Construct Dataset & DataLoader
    if args.dataset != "imagenet":

        train_dset = Emotion_SSL_Dataset(args, alg='comatch', name=args.dataset, train=True,
                                num_classes=args.num_classes, data_dir=args.train_data_dir)
        lb_dset, ulb_dset = train_dset.get_ssl_dset(args.num_labels)

        _eval_dset = Emotion_SSL_Dataset(args, alg='comatch', name=args.dataset, train=False,
                                num_classes=args.num_classes, data_dir=args.test_data_dir)

        eval_dset = _eval_dset.get_dset()
    else:
        image_loader = ImageNetLoader(root_path=args.data_dir, num_labels=args.num_labels,
                                      num_class=args.num_classes)
        lb_dset = image_loader.get_lb_train_data()
        ulb_dset = image_loader.get_ulb_train_data()
        eval_dset = image_loader.get_lb_test_data()
    
    print(len(lb_dset), len(ulb_dset), len(eval_dset))
                            
    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}

    loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                              args.batch_size,
                                              data_sampler=args.train_sampler,
                                              num_iters=args.num_train_iter,
                                              num_workers=args.num_workers)

    loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                               args.batch_size * args.uratio,
                                               data_sampler=args.train_sampler,
                                               num_iters=args.num_train_iter,
                                               num_workers=args.num_workers)

    loader_dict['eval'] = get_data_loader(dset_dict['eval'],
                                          args.eval_batch_size,
                                          num_workers=args.num_workers,
                                          drop_last=False)
    
    print(len(loader_dict['train_lb']), len(loader_dict['train_ulb']), len(loader_dict['eval']))

    ## set DataLoader on CoMatch
    model.set_data_loader(loader_dict)
    model.set_dset(ulb_dset)
    # If args.resume, load checkpoints from args.load_path
    if args.resume:
        model.load_model(args.load_path)

    # START TRAINING of CoMatch
    trainer = model.train
    best_eval_acc = 0
    for epoch in range(args.epoch):
        eval_acc = trainer(args, epoch, best_eval_acc, logger=logger)
        best_eval_acc = max(eval_acc, best_eval_acc)

    # logging.warning(f"GPU {args.rank} training is FINISHED")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='main')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true')

    '''
    Training Configuration of main
    '''

    parser.add_argument('--epoch', type=int, default=128)
    parser.add_argument('--num_train_iter', type=int, default=1024,
                        help='total number of training iterations')
    parser.add_argument('-nl', '--num_labels', type=int, default=800)
    parser.add_argument('-bsz', '--batch_size', type=int, default=16)
    parser.add_argument('--uratio', type=int, default=4,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=512,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')

    ''' Comatch parameters '''

    parser.add_argument('--hard_label', type=str2bool, default=True)
    # comatch 默认温度是0.2
    parser.add_argument('--T', type=float, default=0.2)
    parser.add_argument('--p_cutoff', type=float, default=0.95, help='pseudo label threshold')
    parser.add_argument('--noise_th', default=0.3, type=float, help='graph noise threshold')
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    parser.add_argument('--ldl_ratio', type=float, default=0.2)
    parser.add_argument('--low_dim', type=int, default=64)
    parser.add_argument('--lam_c', type=float, default=3, help='coefficient of contrastive loss')
    parser.add_argument('--lam_d', type=float, default=3, help='coefficient of distribution loss')
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--dynamic_th', type=float, default=0.7)
    parser.add_argument('--dis_ce', action='store_true')
    parser.add_argument('--update_m', type=str, default='L2')


    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip', type=float, default=0)
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='Resnet50')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--train_data_dir', type=str, default='')
    parser.add_argument('--test_data_dir', type=str, default='')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('-nc', '--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)

    '''
    multi-GPUs & Distrbitued Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    # config file
    parser.add_argument('--c', type=str, default='')

    args = parser.parse_args()
    main(args)

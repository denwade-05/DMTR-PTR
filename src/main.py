import os
import sys
import argparse
import warnings
import random
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from trainer import Trainer
import dataloader_down4

parser = argparse.ArgumentParser(description='DMTR-PTR training / evaluation')

parser.add_argument('-e', '--experiment', required=True, type=str,
                    help='Experiment name [must be unique]')

# Model configuration
parser.add_argument('-m', '--model_name', required=True, type=str,
                    help="Model: vit | vit_dmtr_ptr (see vit/model_dmtr_ptr.py)")
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--shuffle', action='store_false',  # default true
                    help='shuffle training data')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
parser.add_argument('--train_patience', type=int, default=50,
                    help='Patience for early stopping')
parser.add_argument('-l', '--loss', type=str,
                    help='Loss function to use [genexp|mse]')

# Hyperparameter configuration
parser.add_argument('--h-patch', default=4, type=int, metavar='N',
                    help='square patch size, e.g. 16 (16 x 16)')
parser.add_argument('--h-dim', default=256, type=int, metavar='N',
                    help='model dimension d, e.g., n x d')
parser.add_argument('--h-depth', default=6, type=int, metavar='N',
                    help='number of transformer blocks')
parser.add_argument('--h-heads', default=12, type=int, metavar='N',
                    help='number of heads in each block')
parser.add_argument('--h-mlp-dim', default=512, type=int, metavar='N',
                    help='FFN hidden size for all tokens (shared FFN)')
parser.add_argument('--use-early-bypass', action='store_true',
                    help='Enable PTR schedule (default 2-1-1 for depth=4)')
parser.add_argument('--super-bypass-layer', default=3, type=int, metavar='N',
                    help='1-based layer index to start super-token bypass')
parser.add_argument('--enable-base-bypass', action='store_true',
                    help='Enable partial base-token bypass from base-bypass-layer')
parser.add_argument('--base-bypass-layer', default=4, type=int, metavar='N',
                    help='1-based layer index to start base-token bypass')
parser.add_argument('--base-bg-threshold', default=0.9, type=float,
                    help='Background probability threshold for base bypass')
parser.add_argument('--base-keep-min-ratio', default=0.2, type=float,
                    help='Minimum fraction of base tokens kept in attention')
parser.add_argument('--bg-aux-weight', default=0.0, type=float,
                    help='BCE weight for background aux head; 0 disables')
parser.add_argument('--bg-label-thresh', default=0.0, type=float,
                    help='Patch max <= thresh labels a base token as background')
parser.add_argument('--h-dim-head', default=64, type=int, metavar='N',
                    help='inner dimension of q,k,v b, e.g. n x d -> n x b -> n x d')

# Distributed configuration
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:30002', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# Directory / dummy-data configuration
parser.add_argument('--data-name', type=str, default='random_down4',
                    help='Dataset name (for logging); default is synthetic random tensors')
parser.add_argument('--dummy-train-samples', type=int, default=32,
                    help='Number of synthetic training samples')
parser.add_argument('--dummy-test-samples', type=int, default=8,
                    help='Number of synthetic validation/test samples')
parser.add_argument('--data-dir', type=str, default='../outputs',
                    help='Directory for optional prediction dumps (--save)')
parser.add_argument('--ckpt-dir', type=str, default='../ckpt',
                    help='Directory to save / load model checkpoints')
parser.add_argument('--ret-datetime', action='store_false')


# Training configuration
parser.add_argument('--cuda', action='store_true',  # default false
                    help='use CUDA')
parser.add_argument('--resume', action='store_true',  # default false
                    help='resume from last checkpoint')
parser.add_argument('--best', action='store_false',  # default true
                    help='Load best model or most recent for testing')

# Testing configuration
parser.add_argument('--test', action='store_true',  # default false
                    help='Test the model on the test set')
parser.add_argument('--only-metrics', action='store_true',
                    help='Load best ckpt and run eval_metrics on test split (no training, no saving predictions)')
parser.add_argument('--metrics-dbz-scale', type=float, default=60.0,
                    help='Scale factor for CSI/HSS (×pred, ×target before thresholding); use 60 when data are dBZ/60 in [0,1]')
parser.add_argument('--metrics-ssim-data-range', type=float, default=1.0,
                    help='skimage SSIM data_range (1.0 for normalized [0,1] targets)')
parser.add_argument('--metrics-raw-dbz', action='store_true',
                    help='Inputs are already 0–60 dBZ: no CSI scale, SSIM data_range=60, LPIPS uses x/30-1 mapping')
parser.add_argument('--save', action='store_true',  # default false
                    help='Save the results of the test set')
parser.add_argument('--save-label', action='store_true',  # default false
                    help='Save the labels')

# PolicyNet configuration
parser.add_argument('--policynet-path', type=str, default=None,
                    help='Frozen PolicyNet checkpoint (required unless --disable-merge)')
parser.add_argument('--bg-merge-ratio', type=float, default=0.5,
                    help='Global merge ratio over coarse grid (low PolicyNet score)')
parser.add_argument('--split-ratio', type=float, default=0.05,
                    help='Top fraction of base patches to split into subpatches')
parser.add_argument('--merge-size', type=int, default=2,
                    help='Merge size on patch grid (only 2 supported)')
parser.add_argument('--disable-merge', action='store_true',
                    help='Disable merge/split; keep all base patches')


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.distributed and args.gpu is not None:
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs of the current node.
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int(
            (args.workers + ngpus_per_node - 1) / ngpus_per_node)
    train_loader, train_sampler, \
        val_loader, val_sampler = dataloader_down4.get_dataloader(args)
    print(
        f"=> Finished loading data: {args.data_name} "
        f"for {'training' if not args.test else 'testing'} "
        f"with {len(train_loader.dataset) if not args.test else len(val_loader.dataset)} samples "
        f"X.shape={val_loader.dataset.xshape}, T.shape={val_loader.dataset.tshape}."
    )

    trainer = Trainer(args, val_loader.dataset.xshape,
                      val_loader.dataset.tshape)

    trainer.summary()

    if args.only_metrics:
        is_main = (not args.distributed) or (args.rank in (-1, 0))
        if is_main:
            trainer.eval_metrics(val_loader)
        return

    if not args.test:
        trainer.train(train_loader, train_sampler, val_loader)
        is_main = (not args.distributed) or (args.rank in (-1, 0))
        if is_main:
            print("=> Post-training evaluation on test split (no prediction saving)")
            trainer.eval(val_loader, test=True, save=False)
            trainer.eval_metrics(val_loader)
    else:
        trainer.eval(val_loader, args.test, args.save)


def main(args):
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print("distributed:",args.distributed)
    print(args.world_size, args.multiprocessing_distributed)

    if torch.cuda.is_available():
        args.device = 'cuda' if args.cuda else 'cpu'
        args.ngpus_per_node = torch.cuda.device_count()
        print(f'{args.ngpus_per_node} GPU(s) available')
    else:
        args.ngpus_per_node = 1
        args.device = 'cpu'
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node,
                 args=(args.ngpus_per_node, args))
    else:
        main_worker(args.gpu, args.ngpus_per_node, args)


if __name__ == '__main__':
    """
    See ../scripts/train_main.sh and ../README.md for the paper main configuration.
    """
    args = parser.parse_args()
    if args.only_metrics:
        args.test = True
    print(args)

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main(args=args)

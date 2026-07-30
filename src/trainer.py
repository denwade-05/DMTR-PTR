import os
import time
import shutil
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from common import MSEGenexp
from utils import AverageMeter, Summary
import argparse
from metrics_local import ScoresAllLocal

class Trainer:

    def __init__(self, args, xshape, tshape):

        if args.model_name == 'vit':
            from vit.model import ViT
            self.model = ViT(image_size=xshape[1:], patch_size=args.h_patch,
                             in_chans=xshape[0], out_chans=tshape[0],
                             dim=args.h_dim, depth=args.h_depth, heads=args.h_heads,
                             mlp_dim=args.h_mlp_dim, dim_head=args.h_dim_head)
        elif args.model_name == 'vit_dmtr_ptr':
            from vit.model_dmtr_ptr import ViTDMTRPTR
            self.model = ViTDMTRPTR(
                image_size=xshape[1:], patch_size=args.h_patch,
                in_chans=xshape[0], out_chans=tshape[0],
                dim=args.h_dim, depth=args.h_depth, heads=args.h_heads,
                mlp_dim=args.h_mlp_dim,
                dim_head=args.h_dim_head,
                policynet_path=args.policynet_path,
                bg_merge_ratio=args.bg_merge_ratio,
                split_ratio=args.split_ratio,
                merge_size=args.merge_size,
                disable_merge=args.disable_merge,
                use_early_bypass=getattr(args, 'use_early_bypass', False),
                super_bypass_layer=getattr(args, 'super_bypass_layer', 3),
                base_bypass_layer=getattr(args, 'base_bypass_layer', 4),
                enable_base_bypass=getattr(args, 'enable_base_bypass', False),
                base_bg_threshold=getattr(args, 'base_bg_threshold', 0.9),
                base_keep_min_ratio=getattr(args, 'base_keep_min_ratio', 0.2),
            )
        else:
            raise NotImplementedError

        self._xshape = xshape
        self.h_patch = int(args.h_patch)

        self.resume = args.resume
        self.best = args.best
        self.experiment = args.experiment
        self.model_name = args.model_name
        self.ckpt_dir = args.ckpt_dir
        self.data_dir = args.data_dir

        self.multiprocessing_distributed = args.multiprocessing_distributed
        self.ngpus_per_node = args.ngpus_per_node
        self.rank = args.rank
        self.distributed = args.distributed
        self.device = args.device
        self.gpu = args.gpu

        self._set_devices()
        self.device = torch.device(args.device)  # redefine this

        # training params
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.shuffle = args.shuffle
        self.seed = args.seed
        self.bg_aux_weight = float(getattr(args, 'bg_aux_weight', 0.0))
        self.bg_label_thresh = float(getattr(args, 'bg_label_thresh', 0.0))

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr
        )

        # loss 增加mae        
        if args.loss == 'genexp':
            print("=> Using MSEGenexp Loss Function")
            self.loss_fn = MSEGenexp(weight=(1.0, 5.0, 4.0))
        elif args.loss == 'mae':
            print("=> Using MAE (L1) Loss Function")
            self.loss_fn = nn.L1Loss()
        elif args.loss == 'mse':
            print("=> Using MSE Loss Function")
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {args.loss}. "
                            f"Supported: ['mse', 'mae', 'genexp']")

        self.loss_fn = self.loss_fn.to(self.device)

        # bookkeeping
        self.start_epoch = 0
        self.new_epoch_counter = 0
        self.best_val_metric = float('inf')
        self.train_patience = args.train_patience

        raw = bool(getattr(args, "metrics_raw_dbz", False))
        self.metrics_cls_scale = (
            1.0 if raw else float(getattr(args, "metrics_dbz_scale", 60.0))
        )
        self.metrics_ssim_dr = (
            60.0 if raw else self.metrics_cls_scale
        )
        self.metrics_normalized_01 = not raw

    def summary(self):
        # print(self.model)
        print(
            f'=> Trainable Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        print(
            f'[DEBUG] {self.device=} {next(self.model.parameters()).is_cuda=}')
        if self.model_name == 'vit_dmtr_ptr' and self.rank in (-1, 0):
            self._print_dmtr_token_preview()

    def _unwrap_base_module(self):
        m = self.model
        if isinstance(m, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            m = m.module
        return m

    def _print_dmtr_token_preview(self):
        from vit.model_dmtr_ptr import ViTDMTRPTR

        m = self._unwrap_base_module()
        if isinstance(m, ViTDMTRPTR):
            core = m
        elif hasattr(m, 'backbone') and isinstance(m.backbone, ViTDMTRPTR):
            core = m.backbone
        else:
            return
        c, h, w = self._xshape
        p = self.h_patch
        base_cells = (h // p) * (w // p)
        was_training = core.training
        core.eval()
        try:
            with torch.no_grad():
                dummy = torch.randn(1, c, h, w, device=self.device, dtype=torch.float32)
                _ = core(dummy)
        finally:
            if was_training:
                core.train()
        n1, n2, n0 = core._last_token_breakdown
        nt = n1 + n2 + n0
        pe = 'on' if core.use_early_bypass else 'off'
        bp = 'on' if getattr(core, "enable_base_bypass", False) else 'off'
        print(
            f'=> [ViTDMTRPTR] token stats (dummy forward; varies with PolicyNet score):'
            f' base_cells={base_cells} | scale1={n1} scale2(merge)={n2} scale0(sub)={n0}'
            f' | transformer length N={nt} (N/base={nt / max(base_cells, 1):.3f})'
            f' | super_bypass_layer={getattr(core, "super_bypass_layer", "?")}'
            f' base_bypass_layer={getattr(core, "base_bypass_layer", "?")} use={pe}'
            f' base_bypass={bp} bypassed={getattr(core, "_last_base_bypassed", 0)}'
        )

    def _compute_bg_aux_loss(self, target: torch.Tensor, logits=None, coords=None):
        if self.bg_aux_weight <= 0:
            return None
        if logits is None or coords is None:
            m = self._unwrap_base_module()
            logits = getattr(m, "_last_base_bg_logits", None)
            coords = getattr(m, "_last_base_bg_coords", None)
        if logits is None or coords is None or logits.numel() == 0:
            return None
        bsz, n1 = logits.shape
        p = self.h_patch
        t = target
        b, c, h, w = t.shape
        if b != bsz:
            return None
        if h % p != 0 or w % p != 0:
            return None
        patch_max = t.view(b, c, h // p, p, w // p, p).amax(dim=(1, 3, 5))
        gh = coords[:, :, 0].clamp(0, patch_max.shape[1] - 1)
        gw = coords[:, :, 1].clamp(0, patch_max.shape[2] - 1)
        gt = patch_max[torch.arange(b, device=t.device).unsqueeze(1), gh, gw]
        bg_label = (gt <= self.bg_label_thresh).to(logits.dtype)
        return nn.functional.binary_cross_entropy_with_logits(logits, bg_label)

    def _set_devices(self):
        if self.distributed and self.device == 'cuda':
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                self.model.cuda(self.gpu)
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.gpu])
                print(
                    f'=> DistributedDataParallel initialization on GPU:{self.gpu}')
            else:
                self.model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model)
                print('=> DistributedDataParallel initialization on GPU(s)')
        elif self.gpu is not None and self.device == 'cuda':
            torch.cuda.set_device(self.gpu)
            self.model = self.model.cuda(self.gpu)
            self.device = torch.device(f'cuda:{self.gpu}')
            print('=> Standard initialization on GPU')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.model = self.model.to(self.device)
        elif self.device == 'cuda':
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = torch.nn.DataParallel(self.model).cuda()
            print('=> DataParallel initialization on GPU(s)')

    def _train_one_epoch(self, train_loader, epoch):
        batch_time = AverageMeter(Summary.NONE)
        data_time = AverageMeter(Summary.NONE)
        losses = AverageMeter(Summary.NONE)
        aux_losses = AverageMeter(Summary.NONE)
        metrics = AverageMeter(Summary.AVERAGE)

        self.model.train()
        end = time.time()
        with tqdm(total=len(train_loader) * self.batch_size, position=0, leave=True) as pbar:
        # with tqdm(total=len(train_loader) * self.batch_size, position=0, leave=True, disable=True) as pbar:
            for i, (X, T, datetime_str) in enumerate(train_loader):
                # print(X.shape, T.shape, datetime_str)
                data_time.update(time.time() - end)

                X = X.to(self.device, non_blocking=True)
                T = T.to(self.device, non_blocking=True)

                # forward
                if self.model_name == 'vit_dmtr_ptr':
                    Y, bg_logits, bg_coords = self.model(X, return_aux=True)
                else:
                    Y = self.model(X)
                    bg_logits, bg_coords = None, None
                main_loss = self.loss_fn(Y, T)
                aux_loss = self._compute_bg_aux_loss(T, logits=bg_logits, coords=bg_coords)
                loss = main_loss if aux_loss is None else main_loss + self.bg_aux_weight * aux_loss
                
                # update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # metrics
                rmse = torch.sqrt(torch.mean((Y - T)**2))

                losses.update(loss.item(), X.size()[0])
                if aux_loss is not None:
                    aux_losses.update(aux_loss.item(), X.size()[0])
                metrics.update(rmse.item(), X.size()[0])
                batch_time.update(time.time() - end)
                end = time.time()

                pbar.set_description(
                    (
                        f"Epoch: {epoch + 1} [{i + 1}/{self.n_train_batches}] "
                        f"Time {batch_time.val:.5f} ({batch_time.avg:.5f}) "
                        f"Data {data_time.val:.5f} ({data_time.avg:.5f}) "
                        f"Loss {losses.val:.5f} ({losses.avg:.5f}) "
                        f"Aux {aux_losses.val:.5f} ({aux_losses.avg:.5f}) "
                        f"Met {metrics.val:.5f} ({metrics.avg:.5f})"
                    )
                )
                pbar.update(X.shape[0])

            return losses.avg, metrics.avg

    def train(self, train_loader, train_sampler, val_loader):

        # load the most recent checkpoint
        if self.resume:
            self._load_checkpoint(best=False)

        self.n_train_batches = len(train_loader)

        for epoch in range(self.start_epoch, self.epochs):
            if self.distributed:  # shuffle training data
                train_sampler.set_epoch(epoch)

            train_loss, train_metric = self._train_one_epoch(
                train_loader, epoch)
            val_loss, val_metric = self.eval(val_loader)

            # min < float(inf), max > 0
            is_best = val_metric < self.best_val_metric
            msg1 = "train loss: {:.5f} - train met: {:.5f} "
            msg2 = "- val loss: {:.5f} - val met: {:.5f}"
            if is_best:
                self.new_epoch_counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(
                msg.format(
                    train_loss, train_metric, val_loss, val_metric
                )
            )
            if not is_best:
                self.new_epoch_counter += 1
            if self.new_epoch_counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return self.best_val_metric

            # check for improvement
            self.best_val_metric = min(val_metric, self.best_val_metric)
            if not self.multiprocessing_distributed or (self.multiprocessing_distributed
                                                        and self.rank % self.ngpus_per_node == 0):
                self._save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_state": self.model.state_dict(),
                        "optim_state": self.optimizer.state_dict(),
                        "best_val_metric": self.best_val_metric,
                    },
                    is_best,
                )
        print('=> Finished Training.')

        return self.best_val_metric

    @torch.no_grad()
    def eval(self, val_loader, test=False, save=False):

        if test:
            self._load_checkpoint(best=self.best)

        losses = AverageMeter(Summary.NONE)
        aux_losses = AverageMeter(Summary.NONE)
        metrics = AverageMeter(Summary.AVERAGE)

        if save:
            f = os.path.join(self.data_dir, 'out',
                             f'{self.experiment}-{self.model_name}')
            f_label = os.path.join(self.data_dir, 'out','label')
            os.makedirs(f, exist_ok=True)
            os.makedirs(f_label, exist_ok=True)
            k = 0

        self.model.eval()
        for i, (X, T, datetime_str) in enumerate(val_loader):
            X, T = X.to(self.device), T.to(self.device)
            if self.model_name == 'vit_dmtr_ptr':
                Y, bg_logits, bg_coords = self.model(X, return_aux=True)
            else:
                Y = self.model(X)
                bg_logits, bg_coords = None, None
            main_loss = self.loss_fn(Y, T)
            aux_loss = self._compute_bg_aux_loss(T, logits=bg_logits, coords=bg_coords)
            loss = main_loss if aux_loss is None else main_loss + self.bg_aux_weight * aux_loss

            # metrics
            rmse = torch.sqrt(torch.mean((Y - T)**2))

            losses.update(loss.item(), X.size()[0])
            if aux_loss is not None:
                aux_losses.update(aux_loss.item(), X.size()[0])
            metrics.update(rmse.item(), X.size()[0])

            if save:
                for j in range(Y.shape[0]):
                    # np.save(os.path.join(f, f'test_predictions_{k:06d}.npy'),
                    #         Y[j].detach().cpu().numpy())
                    # np.save(os.path.join(f_label, f'label_{k:06d}.npy'),
                    #         T[j].detach().cpu().numpy())
                    np.save(os.path.join(f, f'{datetime_str[j]}.npy'),
                            Y[j].detach().cpu().numpy())
                    np.save(os.path.join(f_label, f'{datetime_str[j]}.npy'),
                            T[j].detach().cpu().numpy())
                    k += 1

        if self.distributed:
            losses.all_reduce()
            metrics.all_reduce()

        if test:
            print(
                f"[*] test loss: {losses.avg:.5f} - "
                f"test met: {metrics.avg:.5f}"
            )

        return losses.avg, metrics.avg

    @torch.no_grad()
    def eval_metrics(self, val_loader):
        """Run full metrics once and save as JSON by default."""
        if self.best:
            self._load_checkpoint(best=True)

        scorer = ScoresAllLocal(
            cls_dbz_scale=self.metrics_cls_scale,
            ssim_data_range=self.metrics_ssim_dr,
            normalized_01=self.metrics_normalized_01,
        )
        self.model.eval()
        for X, T, _ in val_loader:
            X, T = X.to(self.device), T.to(self.device)
            if self.model_name == 'vit_dmtr_ptr':
                Y, _, _ = self.model(X, return_aux=True)
            else:
                Y = self.model(X)
            scorer.update(Y[:, 0], T[:, 0])

        scores = scorer.get_scores()
        metrics_dir = os.path.abspath(os.path.join(self.ckpt_dir, "..", "results_metrics"))
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, f"{self.experiment}_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        return scores
    

    def _save_checkpoint(self, state, is_best):
        """Save a checkpoint; also copy to *_model_best.pth.tar when improved."""
        pre = self.experiment + '_'
        filename = pre + self.model_name + "_ckpt.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        torch.save(state, ckpt_path)
        if is_best:
            filename = pre + self.model_name + "_model_best.pth.tar"
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    def _load_checkpoint(self, best=False, backbone=None):
        """Load the latest or best checkpoint for resume / evaluation."""
        if backbone is not None:
            filename = backbone
            best = True if 'best' in backbone else False
        else:
            pre = self.experiment + '_'
            if best:
                filename = pre + self.model_name + "_model_best.pth.tar"
            else:
                filename = pre + self.model_name + "_ckpt.pth.tar"

        ckpt_path = os.path.join(self.ckpt_dir, filename)

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                "[!] No checkpoint found at '{}'".format(ckpt_path)
            )
        print("[*] Loading model from {}".format(ckpt_path))

        if self.gpu is None:
            map_location = None if torch.cuda.is_available() else torch.device('cpu')
            ckpt = torch.load(ckpt_path, map_location=map_location)
        elif torch.cuda.is_available():
            loc = 'cuda:{}'.format(self.gpu)
            ckpt = torch.load(ckpt_path, map_location=loc)
        else:
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

        if backbone is None:
            self.start_epoch = ckpt["epoch"]
            self.best_val_metric = ckpt["best_val_metric"]
            self.optimizer.load_state_dict(ckpt["optim_state"])

        model_state = ckpt["model_state"]
        try:
            self.model.load_state_dict(model_state)
        except RuntimeError:
            has_module_prefix = any(k.startswith("module.") for k in model_state.keys())
            if has_module_prefix:
                model_state = {k[len("module."):]: v for k, v in model_state.items()}
            else:
                model_state = {f"module.{k}": v for k, v in model_state.items()}
            self.model.load_state_dict(model_state)

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best metric of {:.5f}".format(
                    filename, ckpt["epoch"], ckpt["best_val_metric"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"]))
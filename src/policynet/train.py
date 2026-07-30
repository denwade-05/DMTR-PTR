import argparse
import math
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from policynet.config import Config
from policynet.net import PolicyNet
from policynet.data_srvit_patch3 import TokenSharingPolicyDatasetSrvitDown4Patch3
from policynet.data_srvit_patch4 import TokenSharingPolicyDatasetSrvitDown4Patch4


def _setup_distributed(backend: str = "nccl"):
    """torchrun 注入 RANK / LOCAL_RANK / WORLD_SIZE；单卡则不初始化进程组。"""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


def train(args):
    cfg = Config()
    if args.num_iterations:
        cfg.num_iterations = args.num_iterations
    if args.batch_size_train:
        cfg.batch_size_train = args.batch_size_train
    if args.lr:
        cfg.lr = args.lr
    cfg.dataset = args.dataset
    cfg.optimizer = args.optimizer

    distributed, rank, local_rank, world_size = _setup_distributed(cfg.ddp_backend)
    is_main = rank == 0

    if is_main:
        print(args)

    if args.exp_name:
        logdir = os.path.join(cfg.logdir, args.exp_name)
    else:
        logdir = os.path.join(cfg.logdir, "experiment")

    if is_main:
        if not os.path.exists(logdir):
            os.makedirs(logdir, exist_ok=False)

    if distributed:
        dist.barrier()

    txt_write = open(os.path.join(logdir, "log.txt"), "w") if is_main else None
    writer = SummaryWriter(log_dir=logdir) if is_main else None

    if cfg.dataset == "srvit_down4_patch4":
        dataset = TokenSharingPolicyDatasetSrvitDown4Patch4
        policy_patch = 4
    elif cfg.dataset == "srvit_down4_patch3":
        dataset = TokenSharingPolicyDatasetSrvitDown4Patch3
        policy_patch = 3
    else:
        raise NotImplementedError(f"Unsupported dataset: {cfg.dataset}")

    train_dataset = dataset(data_mode="train", ret_datetime=True)
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        if distributed
        else None
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size_train,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    eval_dataset = dataset(data_mode="test", ret_datetime=True)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size_test,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    # 仅 rank0 做 train split 验证：全量顺序遍历，避免 DDP train_loader 只含 1/world 数据
    train_eval_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size_test,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    model = PolicyNet(
        patch_size=policy_patch,
        input_hw=(192, 384),
        block_out_channels=cfg.block_out_channels,
        mse_genexp_weight=cfg.mse_genexp_weight,
    )
    model.train()

    device = torch.device(f"cuda:{local_rank}" if cfg.enable_cuda else "cpu")
    if cfg.enable_cuda:
        model = model.to(device)
        if distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = model.to(device)

    if cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.lr_momentum,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError("Only SGD, Adam or AdamW optimizers implemented")

    thresh_conv = cfg.conv_patch_t_threshold

    def evaluate(dataloader, split_name="test"):
        """MSEGenexp、MAE；强对流 recall@Topρ%；bottom50% 0dBZ。仅在 rank0 调用。"""
        _unwrap(model).eval()
        sum_loss = 0.0
        sum_mae = 0.0
        sum_mae_hi = 0.0
        n_hi = 0
        n_batches = 0
        n_elems = 0
        chunks_score = []
        chunks_target = []

        with torch.no_grad():
            for images, targets, _image_ids in dataloader:
                if targets.ndim == 4 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)

                if cfg.enable_cuda:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                out = _unwrap(model)(images, targets)
                loss = out["loss"]
                score = out["score"].squeeze(1)

                sum_loss += loss.item()
                sum_mae += F.l1_loss(score, targets, reduction="sum").item()

                hi = targets > 0.01
                if hi.any():
                    sum_mae_hi += (score[hi] - targets[hi]).abs().sum().item()
                    n_hi += hi.sum().item()

                n_batches += 1
                n_elems += targets.numel()

                chunks_score.append(score.detach().float().cpu().reshape(-1))
                chunks_target.append(targets.detach().float().cpu().reshape(-1))

        mean_loss = sum_loss / max(n_batches, 1)
        mean_mae = sum_mae / max(n_elems, 1)
        mean_mae_hi = sum_mae_hi / max(n_hi, 1) if n_hi > 0 else float("nan")

        all_s = torch.cat(chunks_score)
        all_t = torch.cat(chunks_target)
        n = all_s.numel()
        k_bot = max(1, int(math.floor(0.50 * n)))
        _, idx_bot = torch.topk(all_s, k=k_bot, largest=False)
        bot50_frac_zero = (all_t[idx_bot] <= 0.0).float().mean().item()

        conv_mask = all_t >= thresh_conv
        n_conv = int(conv_mask.sum().item())
        conv_patch_frac = n_conv / max(n, 1)
        conv_idx = torch.nonzero(conv_mask, as_tuple=False).squeeze(-1)

        conv_recall_at_top: dict = {}
        for rho in cfg.eval_recall_top_fracs:
            k_top = max(1, int(math.ceil(float(rho) * n)))
            _, idx_top = torch.topk(all_s, k=k_top, largest=True)
            if n_conv == 0:
                conv_recall_at_top[rho] = float("nan")
            else:
                top_mask = torch.zeros(n, dtype=torch.bool)
                top_mask[idx_top] = True
                hit = top_mask[conv_idx]
                conv_recall_at_top[rho] = (hit.float().sum() / n_conv).item()

        print(f"{split_name} MSEGenexp: {mean_loss:.6f}")
        print(f"{split_name} MAE (all): {mean_mae:.6f}")
        print(f"{split_name} MAE (target>0.01): {mean_mae_hi:.6f}")
        print(
            f"{split_name} conv_patch_frac (t≥{thresh_conv:.4f}): "
            f"{100.0 * conv_patch_frac:.3f}% (n_conv={n_conv})"
        )
        for rho in sorted(conv_recall_at_top.keys()):
            r = conv_recall_at_top[rho]
            pct_label = int(round(rho * 100))
            msg = (
                f"{split_name} conv_recall@Top{pct_label}% "
                f"(frac of conv patches in global top-{pct_label}% by score)"
            )
            if r != r:
                print(f"{msg}: nan (no conv patches)")
            else:
                print(f"{msg}: {100.0 * r:.2f}%")
        print(
            f"{split_name} bottom50%_pred patches with 0dBZ (t==0): "
            f"{100.0 * bot50_frac_zero:.2f}%"
        )

        if txt_write:
            txt_write.write(f"{split_name} MSEGenexp: {mean_loss:.6f}\n")
            txt_write.write(f"{split_name} MAE (all): {mean_mae:.6f}\n")
            txt_write.write(f"{split_name} MAE (target>0.01): {mean_mae_hi:.6f}\n")
            txt_write.write(f"{split_name} conv_patch_frac: {conv_patch_frac:.6f}\n")
            txt_write.write(f"{split_name} conv_patch_count: {n_conv}\n")
            for rho in sorted(conv_recall_at_top.keys()):
                pct_label = int(round(rho * 100))
                txt_write.write(
                    f"{split_name} conv_recall_top{pct_label}pct: "
                    f"{conv_recall_at_top[rho]:.6f}\n"
                )
            txt_write.write(f"{split_name} bottom50pct_0dBZ_frac: {bot50_frac_zero:.6f}\n")
            txt_write.flush()

        _unwrap(model).train()
        return (
            mean_loss,
            mean_mae,
            mean_mae_hi,
            conv_recall_at_top,
            conv_patch_frac,
            bot50_frac_zero,
        )

    running_loss = 0.0
    running_loss_tb = 0.0
    i = 0
    epoch = 0

    best_test_loss = float("inf")

    if is_main:
        mode = f"{world_size}-GPU DDP" if distributed else "single-GPU"
        print(f"Starting training (PolicyNet regression + MSEGenexp, {mode})...")
        if distributed:
            print(
                f"Per-GPU batch={cfg.batch_size_train}, "
                f"global batch ≈ {cfg.batch_size_train * world_size}"
            )

    while i < cfg.num_iterations:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch += 1

        for imgs, targets, _ids in train_dataloader:
            if i >= cfg.num_iterations:
                break

            if targets.ndim == 4 and targets.shape[1] == 1:
                targets = targets.squeeze(1)

            if cfg.enable_cuda:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(imgs, targets)
            loss = out["loss"]

            running_loss += loss.item()
            running_loss_tb += loss.item()

            loss.backward()
            optimizer.step()

            if is_main and i % cfg.log_iterations == 0:
                loss_avg = running_loss if i == 0 else running_loss / cfg.log_iterations
                print(f"Iteration {i} - Loss: {loss_avg:.5f}")
                if txt_write:
                    txt_write.write(f"Iteration {i} - Loss: {loss_avg:.5f}\n")
                    txt_write.flush()
                running_loss = 0.0

            if is_main and i % cfg.summary_iterations == 0:
                loss_summary = (
                    running_loss_tb if i == 0 else running_loss_tb / cfg.summary_iterations
                )
                writer.add_scalar("Loss/train", loss_summary, i)
                running_loss_tb = 0.0

            if i % cfg.eval_iterations == 0 and i != 0:
                if distributed:
                    dist.barrier()
                if is_main:
                    print("Evaluating...")
                    (
                        test_loss,
                        test_mae,
                        test_mae_hi,
                        test_conv_rec,
                        test_conv_frac,
                        test_bot50,
                    ) = evaluate(eval_dataloader, split_name="test")
                    writer.add_scalar("Loss/test_MSEGenexp", test_loss, i)
                    writer.add_scalar("MAE/test_all", test_mae, i)
                    writer.add_scalar("Eval/test_conv_patch_frac", test_conv_frac, i)
                    for rho, rec in sorted(test_conv_rec.items()):
                        pct = int(round(rho * 100))
                        tag = f"Eval/test_conv_recall_top{pct}pct"
                        writer.add_scalar(tag, rec, i)
                    writer.add_scalar("Eval/test_bottom50pct_0dBZ_frac", test_bot50, i)
                    if not math.isnan(test_mae_hi):
                        writer.add_scalar("MAE/test_echo", test_mae_hi, i)

                    (
                        tr_loss,
                        tr_mae,
                        tr_mae_hi,
                        tr_conv_rec,
                        tr_conv_frac,
                        tr_bot50,
                    ) = evaluate(train_eval_dataloader, split_name="train")
                    writer.add_scalar("Loss/train_eval_MSEGenexp", tr_loss, i)
                    writer.add_scalar("MAE/train_all", tr_mae, i)
                    writer.add_scalar("Eval/train_conv_patch_frac", tr_conv_frac, i)
                    for rho, rec in sorted(tr_conv_rec.items()):
                        pct = int(round(rho * 100))
                        writer.add_scalar(f"Eval/train_conv_recall_top{pct}pct", rec, i)
                    writer.add_scalar("Eval/train_bottom50pct_0dBZ_frac", tr_bot50, i)

                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        best_path = os.path.join(logdir, "model_best_msegenexp.pth")
                        torch.save(_unwrap(model).state_dict(), best_path)
                        print(
                            f"New best test MSEGenexp: {best_test_loss:.6f}, saved to {best_path}"
                        )
                        if txt_write:
                            txt_write.write(
                                f"New best test MSEGenexp: {best_test_loss:.6f}\n"
                            )
                            txt_write.flush()

                    print("Continuing training...")
                if distributed:
                    dist.barrier()

            i += 1

    if is_main:
        print("Finished training.")

    if distributed:
        dist.barrier()

    if is_main:
        save_path = os.path.join(logdir, "model_last.pth")
        torch.save(_unwrap(model).state_dict(), save_path)
        print(f"Saved trained model as {save_path}.")
        if txt_write:
            txt_write.write(f"Training finished. Last model saved to {save_path}\n")
            txt_write.flush()
            txt_write.close()
        writer.close()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, help="number of iterations")
    parser.add_argument("--batch_size_train", type=int, help="training batch size")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument("--optimizer", type=str, help="optimizer", default="AdamW")
    parser.add_argument(
        "--dataset",
        type=str,
        help="srvit_down4_patch3 or srvit_down4_patch4",
        default="srvit_down4_patch4",
    )

    args = parser.parse_args()

    if args.num_iterations:
        print("Num iterations:", args.num_iterations)
    if args.batch_size_train:
        print("Training batch size:", args.batch_size_train)
    if args.lr:
        print("Learning rate:", args.lr)
    if args.exp_name:
        print("Experiment name:", args.exp_name)

    train(args)
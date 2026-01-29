# -*- coding: latin-1 -*-

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from sklearn.metrics import roc_auc_score

# PyTorch built-in AMP
from torch.cuda.amp import GradScaler, autocast


def parse_option():
    # Create an argument parser for the Swin Transformer training/evaluation script.
    # `add_help=False` is used because help handling is often customized elsewhere
    # (e.g., when integrating with config systems or distributed launchers).
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script',
        add_help=False
    )

    # Path to the YAML configuration file.
    # This is the core configuration entry point of the project.
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        metavar="FILE",
        help='path to config file',
    )

    # Optional command-line overrides for configuration values.
    # These are typically key-value pairs that overwrite fields defined in the YAML file.
    # Example: --opts TRAIN.EPOCHS 300 MODEL.SWIN.DEPTHS [2,2,18,2]
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # -------------------------------------------------------------------------
    # Easy / frequently-used configuration overrides
    # These arguments allow modifying common settings without editing the YAML.
    # -------------------------------------------------------------------------

    # Batch size per single GPU (important in distributed training, since effective batch size = batch_size * world_size).
    parser.add_argument(
        '--batch-size',
        type=int,
        help="batch size for single GPU"
    )

    # Root path to the dataset.
    parser.add_argument(
        '--data-path',
        type=str,
        help='path to dataset'
    )

    # Whether the dataset is stored as a zip file instead of a directory.
    # Used to reduce filesystem overhead on some clusters.
    parser.add_argument(
        '--zip',
        action='store_true',
        help='use zipped dataset instead of folder dataset'
    )

    # Dataset caching strategy:
    # - 'no'   : do not cache data
    # - 'full' : cache the entire dataset in memory
    # - 'part' : cache only a shard of the dataset per process
    # The 'part' option is typically used for large-scale distributed training.
    parser.add_argument(
        '--cache-mode',
        type=str,
        default='part',
        choices=['no', 'full', 'part'],
        help='no: no cache, '
             'full: cache all data, '
             'part: sharding the dataset into nonoverlapping pieces and only cache one piece'
    )

    # Path to a checkpoint file for resuming training.
    # This usually restores model weights, optimizer state, and scheduler state.
    parser.add_argument(
        '--resume',
        help='resume from checkpoint'
    )

    # Number of gradient accumulation steps.
    # Allows simulating a larger batch size by accumulating gradients
    # across multiple forward/backward passes before an optimizer step.
    parser.add_argument(
        '--accumulation-steps',
        type=int,
        help="gradient accumulation steps"
    )

    # Whether to use gradient checkpointing.
    # This trades compute for memory by recomputing activations during backward pass,
    # which is especially useful for large Swin models.
    parser.add_argument(
        '--use-checkpoint',
        action='store_true',
        help="whether to use gradient checkpointing to save memory"
    )

    # Automatic Mixed Precision (AMP) optimization level.
    # O0: full FP32 (no AMP)
    # O1/O2: different levels of mixed precision for performance/memory trade-offs.
    parser.add_argument(
        '--amp-opt-level',
        type=str,
        default='O1',
        choices=['O0', 'O1', 'O2'],
        help='mixed precision opt level, if O0, no amp is used'
    )

    # Root output directory.
    # Final outputs are typically organized as: <output>/<model_name>/<experiment_tag>/
    parser.add_argument(
        '--output',
        default='output',
        type=str,
        metavar='PATH',
        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)'
    )

    # Experiment tag used to differentiate multiple runs with the same model/config.
    parser.add_argument(
        '--tag',
        help='tag of experiment'
    )

    # If set, the script runs evaluation only (no training loop).
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Perform evaluation only'
    )

    # If set, the script measures model throughput (images/sec),
    # usually without full training or evaluation.
    parser.add_argument(
        '--throughput',
        action='store_true',
        help='Test throughput only'
    )

    # -------------------------------------------------------------------------
    # Distributed training arguments
    # -------------------------------------------------------------------------

    # Local rank of the process in DistributedDataParallel (DDP).
    # This is typically set automatically by torch.distributed.launch or torchrun.
    parser.add_argument(
        "--local_rank",
        type=int,
        required=True,
        help='local rank for DistributedDataParallel'
    )

    # -------------------------------------------------------------------------
    # Dataset-specific arguments (NIH Chest X-ray customization)
    # These are not part of the original generic Swin codebase but an adaptation of the framework to the CheXray-14 dataset.
    # -------------------------------------------------------------------------

    # Path to the training image dataset.
    parser.add_argument(
        "--trainset",
        type=str,
        required=True,
        help='path to train dataset'
    )

    # Path to the validation image dataset.
    parser.add_argument(
        "--validset",
        type=str,
        required=True,
        help='path to validation dataset'
    )

    # Path to the test image dataset.
    parser.add_argument(
        "--testset",
        type=str,
        required=True,
        help='path to test dataset'
    )

    # CSV files containing labels/metadata for each split.
    # These are commonly used in medical datasets where labels are stored separately.
    parser.add_argument(
        "--train_csv_path",
        type=str,
        required=True,
        help='path to train csv file'
    )
    parser.add_argument(
        "--valid_csv_path",
        type=str,
        required=True,
        help='path to validation csv file'
    )
    parser.add_argument(
        "--test_csv_path",
        type=str,
        required=True,
        help='path to test csv file'
    )

    # Number of MLP heads (or layers) at the end of the network.
    # This is a task-specific modification, typically used for multi-label or hierarchical classification.
    parser.add_argument(
        "--num_mlp_heads",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help='number of mlp layers at end of network'
    )

    # -------------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------------

    # Parse known arguments and keep unparsed ones.
    # This allows compatibility with external launchers or additional arguments not explicitly defined here.
    args, unparsed = parser.parse_known_args()

    # Build the full configuration object by:
    # - loading the YAML file
    # - applying command-line overrides
    # - resolving defaults and derived parameters
    config = get_config(args)

    # Return both raw command-line arguments and the processed configuration object.
    return args, config


def main(config):
    # ---------------------------------------------------------------------
    # Build datasets and dataloaders
    # ---------------------------------------------------------------------
    # This function constructs:
    # - training / validation / test datasets
    # - corresponding PyTorch DataLoaders
    # - optional mixup function for data augmentation
    dataset_train, dataset_val, dataset_test, \
    data_loader_train, data_loader_val, data_loader_test, \
    mixup_fn = build_loader(config)

    # ---------------------------------------------------------------------
    # Model construction
    # ---------------------------------------------------------------------
    # Log the model type and name as defined in the configuration.
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    # Instantiate the model (e.g., Swin Transformer variants).
    model = build_model(config)

    # Move model parameters and buffers to GPU.
    model.cuda()

    # Print the full model architecture (rank 0 usually handles logging).
    logger.info(str(model))

    # ---------------------------------------------------------------------
    # Optimizer and Distributed Data Parallel (DDP) setup
    # ---------------------------------------------------------------------
    # Build optimizer (e.g., AdamW) based on config and model parameters.
    optimizer = build_optimizer(config, model)

    # Wrap the model with DistributedDataParallel for multi-GPU training.
    # - device_ids ensures each process uses the correct GPU.
    # - broadcast_buffers=False avoids synchronizing non-parameter buffers (e.g., running stats), which can reduce overhead.
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[config.LOCAL_RANK],
        broadcast_buffers=False
    )

    # Keep a reference to the underlying model without the DDP wrapper.
    # This is required for:
    # - checkpoint saving/loading
    # - FLOPs computation
    model_without_ddp = model.module

    # ---------------------------------------------------------------------
    # Model statistics
    # ---------------------------------------------------------------------
    # Count the number of trainable parameters.
    n_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f"number of params: {n_parameters}")

    # If the model implements a `flops()` method, compute theoretical FLOPs.
    # This is common in vision models for reporting efficiency.
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    # ---------------------------------------------------------------------
    # Learning rate scheduler
    # ---------------------------------------------------------------------
    # Build the learning rate scheduler.
    # The length of the training dataloader is often used to define step-based schedulers (e.g., per-iteration updates).
    lr_scheduler = build_scheduler(
        config,
        optimizer,
        len(data_loader_train)
    )

    # ---------------------------------------------------------------------
    # Automatic Mixed Precision (AMP)
    # ---------------------------------------------------------------------
    # GradScaler dynamically scales the loss to avoid underflow when using mixed precision training.
    scaler = GradScaler(enabled=(config.AMP_OPT_LEVEL != "O0"))

    # ---------------------------------------------------------------------
    # Loss function selection
    # ---------------------------------------------------------------------
    # The choice of criterion depends on the data augmentation and label smoothing strategy defined in the configuration.
    if config.AUG.MIXUP > 0.:
        # When mixup is enabled, labels are soft targets.
        # SoftTargetCrossEntropy handles probabilistic labels correctly.
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        # Label smoothing regularizes the classifier by preventing overconfident predictions.
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING
        )
    else:
        # Standard cross-entropy loss for classification.
        criterion = torch.nn.CrossEntropyLoss()

    # Track the best (maximum) accuracy achieved so far.
    max_accuracy = 0.0

    # ---------------------------------------------------------------------
    # Automatic resume from checkpoint
    # ---------------------------------------------------------------------
    # If enabled, automatically search for the latest checkpoint in the output directory and resume training from it.
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from "
                    f"{config.MODEL.RESUME} to {resume_file}"
                )
            # Temporarily unfreeze the config to modify it.
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume'
            )

    # ---------------------------------------------------------------------
    # Resume from a specified checkpoint
    # ---------------------------------------------------------------------
    if config.MODEL.RESUME:
        # Load model weights, optimizer state, scheduler state, and AMP scaler (if enabled).
        max_accuracy = load_checkpoint(
            config,
            model_without_ddp,
            optimizer,
            lr_scheduler,
            logger,
            scaler=scaler
        )

        # Run validation on the validation set after resuming.
        acc1, acc5, loss = validate(
            config,
            data_loader_val,
            model,
            is_validation=True
        )
        logger.info(
            f"Mean Accuracy of the network on the "
            f"{len(dataset_val)} validation images: {acc1:.2f}%"
        )
        logger.info(
            f"Mean Loss of the network on the "
            f"{len(dataset_val)} validation images: {loss:.5f}"
        )

        # Run evaluation on the test set.
        acc1, acc5, loss = validate(
            config,
            data_loader_test,
            model,
            is_validation=False
        )
        logger.info(
            f"Mean Accuracy of the network on the "
            f"{len(dataset_test)} test images: {acc1:.2f}%"
        )
        logger.info(
            f"Mean Loss of the network on the "
            f"{len(dataset_test)} test images: {loss:.5f}"
        )

        # If evaluation-only mode is enabled, exit early.
        if config.EVAL_MODE:
            return

    # ---------------------------------------------------------------------
    # Throughput benchmarking mode
    # ---------------------------------------------------------------------
    # Measures inference throughput (images/second) without training.
    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        throughput(data_loader_test, model, logger)
        return

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    logger.info("Start training")
    start_time = time.time()

    # Iterate over epochs.
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        # Set epoch for the distributed sampler to ensure proper shuffling across processes.
        data_loader_train.sampler.set_epoch(epoch)

        # Train the model for one epoch.
        train_one_epoch(
            config,
            model,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            mixup_fn,
            lr_scheduler,
            scaler
        )

        # Save checkpoints periodically and at the final epoch.
        # Only rank 0 handles checkpoint saving to avoid conflicts.
        if dist.get_rank() == 0 and (
            epoch % config.SAVE_FREQ == 0 or
            epoch == (config.TRAIN.EPOCHS - 1)
        ):
            save_checkpoint(
                config,
                epoch,
                model_without_ddp,
                max_accuracy,
                optimizer,
                lr_scheduler,
                logger,
                scaler=scaler
            )

        # -----------------------------------------------------------------
        # Validation and testing after each epoch
        # -----------------------------------------------------------------
        acc1, acc5, loss = validate(
            config,
            data_loader_val,
            model,
            is_validation=True
        )
        logger.info(
            f"Mean Accuracy of the network on the "
            f"{len(dataset_val)} validation images: {acc1:.2f}%"
        )
        logger.info(
            f"Mean Loss of the network on the "
            f"{len(dataset_val)} validation images: {loss:.5f}"
        )

        acc1, acc5, loss = validate(
            config,
            data_loader_test,
            model,
            is_validation=False
        )
        logger.info(
            f"Mean Accuracy of the network on the "
            f"{len(dataset_test)} test images: {acc1:.2f}%"
        )
        logger.info(
            f"Mean Loss of the network on the "
            f"{len(dataset_test)} test images: {loss:.5f}"
        )

        # Update and report the best test accuracy achieved so far.
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Test Max mean accuracy: {max_accuracy:.2f}%')

    # ---------------------------------------------------------------------
    # Training time reporting
    # ---------------------------------------------------------------------
    total_time = time.time() - start_time
    total_time_str = str(
        datetime.timedelta(seconds=int(total_time))
    )
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
    scaler
):
    # ---------------------------------------------------------------------
    # Set model to training mode
    # ---------------------------------------------------------------------
    # Enables behaviors such as dropout and stochastic depth.
    model.train()

    # Reset gradients before starting the epoch.
    optimizer.zero_grad()

    # Number of iterations (mini-batches) in one epoch.
    num_steps = len(data_loader)

    # Meters for tracking statistics over the epoch.
    batch_time = AverageMeter()   # time per iteration
    loss_meter = AverageMeter()    # training loss
    norm_meter = AverageMeter()    # gradient norm

    start = time.time()
    end = time.time()

    # ---------------------------------------------------------------------
    # Main training loop over mini-batches
    # ---------------------------------------------------------------------
    for idx, (samples, targets) in enumerate(data_loader):

        # Move input images to GPU.
        # non_blocking=True allows asynchronous transfers when using pinned memory.
        samples = samples.cuda(non_blocking=True)

        # Move targets to GPU.
        # Note: targets is a list, which suggests multi-head or multi-task supervision.
        for i in range(len(targets)):
            targets[i] = targets[i].cuda(non_blocking=True)

        # -----------------------------------------------------------------
        # Mixup augmentation (if enabled)
        # -----------------------------------------------------------------
        if mixup_fn is not None:
            # Mixup modifies both samples and targets, producing soft labels.
            # The TODO indicates potential refinement in handling multiple targets.
            samples, targets = mixup_fn(samples, targets)

        # -----------------------------------------------------------------
        # Forward pass with Automatic Mixed Precision (AMP)
        # -----------------------------------------------------------------
        # autocast enables FP16 computation where safe, reducing memory usage and increasing throughput on supported hardware.
        with autocast(enabled=(config.AMP_OPT_LEVEL != "O0")):
            outputs = model(samples)

        # -----------------------------------------------------------------
        # Gradient accumulation logic
        # -----------------------------------------------------------------
        # When ACCUMULATION_STEPS > 1, gradients are accumulated across multiple forward/backward passes before an optimizer step.
        if config.TRAIN.ACCUMULATION_STEPS > 1:

            # Compute loss for each output/target pair.
            # This implies the model returns multiple outputs (e.g., multiple classification heads).
            loss = criterion(outputs[0], targets[0])
            for i in range(1, len(targets)):
                loss += criterion(outputs[i], targets[i])

            # Normalize loss to keep gradient scale consistent.
            loss = loss / config.TRAIN.ACCUMULATION_STEPS

            # Backward pass with gradient scaling.
            scaler.scale(loss).backward()

            # Perform optimizer step only every ACCUMULATION_STEPS iterations.
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:

                # Unscale gradients before inspecting or modifying them.
                scaler.unscale_(optimizer)

                # Optional gradient clipping to stabilize training.
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.TRAIN.CLIP_GRAD
                    )
                else:
                    # Compute gradient norm without clipping.
                    grad_norm = get_grad_norm(model.parameters())

                # Update model parameters.
                scaler.step(optimizer)
                scaler.update()

                # Clear gradients for the next accumulation window.
                optimizer.zero_grad()

                # Update learning rate scheduler at iteration level.
                lr_scheduler.step_update(epoch * num_steps + idx)

        else:
            # -----------------------------------------------------------------
            # Standard training (no gradient accumulation)
            # -----------------------------------------------------------------

            # Compute loss across all outputs.
            loss = criterion(outputs[0], targets[0])
            for i in range(1, len(targets)):
                loss += criterion(outputs[i], targets[i])

            # Reset gradients.
            optimizer.zero_grad()

            # Backward pass with AMP scaling.
            scaler.scale(loss).backward()

            # Unscale gradients before clipping or norm computation.
            scaler.unscale_(optimizer)

            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.TRAIN.CLIP_GRAD
                )
            else:
                grad_norm = get_grad_norm(model.parameters())

            # Optimizer step and scaler update.
            scaler.step(optimizer)
            scaler.update()

            # Update learning rate scheduler per iteration.
            lr_scheduler.step_update(epoch * num_steps + idx)

        # -----------------------------------------------------------------
        # Synchronization and metric updates
        # -----------------------------------------------------------------
        # Ensure all CUDA operations are completed before timing/logging.
        torch.cuda.synchronize()

        # Update meters:
        # - loss is averaged using the batch size of the first target.
        loss_meter.update(loss.item(), targets[0].size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        # -----------------------------------------------------------------
        # Periodic logging
        # -----------------------------------------------------------------
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = (
                torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            )
            etas = batch_time.avg * (num_steps - idx)

            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} '
                f'lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB'
            )

    # ---------------------------------------------------------------------
    # End-of-epoch logging
    # ---------------------------------------------------------------------
    lr = optimizer.param_groups[0]['lr']
    memory_used = (
        torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    )
    logger.info(
        f'Train: [{epoch}/{config.TRAIN.EPOCHS}]\t'
        f'lr {lr:.6f}\t'
        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
        f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
        f'mem {memory_used:.0f}MB'
    )

    # Report total epoch duration.
    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes "
        f"{datetime.timedelta(seconds=int(epoch_time))}"
    )


@torch.no_grad()
def validate(config, data_loader, model, is_validation):
    # ---------------------------------------------------------------------
    # Evaluation mode selector
    # ---------------------------------------------------------------------
    # Used only for logging purposes.
    valid_or_test = "Validation" if is_validation else "Test"

    # Loss function used during evaluation.
    # Unlike training, this is always standard CrossEntropyLoss.
    criterion = torch.nn.CrossEntropyLoss()

    # Switch model to evaluation mode:
    # - disables dropout
    # - uses running statistics for normalization layers
    model.eval()

    # ---------------------------------------------------------------------
    # Metric containers (one per class / output head)
    # ---------------------------------------------------------------------
    # The hard-coded value 14 indicates:
    # - 14 independent classification heads, or
    # - 14 classes evaluated separately (common in NIH Chest X-ray tasks).
    batch_time = [AverageMeter() for _ in range(14)]
    loss_meter = [AverageMeter() for _ in range(14)]
    acc1_meter = [AverageMeter() for _ in range(14)]
    acc5_meter = [AverageMeter() for _ in range(14)]

    # Lists used to compute mean metrics across all classes.
    acc1s = []
    acc5s = []
    losses = []
    aucs = []

    end = time.time()

    # ---------------------------------------------------------------------
    # Buffers for AUC computation
    # ---------------------------------------------------------------------
    # all_preds[i]  : predicted probabilities for class i
    # all_label[i]  : ground-truth labels for class i
    # Data is accumulated across all batches.
    all_preds = [[] for _ in range(14)]
    all_label = [[] for _ in range(14)]

    # ---------------------------------------------------------------------
    # Main evaluation loop
    # ---------------------------------------------------------------------
    for idx, (images, target) in enumerate(data_loader):

        # Move images to GPU.
        images = images.cuda(non_blocking=True)

        # Move all targets to GPU.
        # target is a list, consistent with multi-head outputs.
        for i in range(len(target)):
            target[i] = target[i].cuda(non_blocking=True)

        # -----------------------------------------------------------------
        # Forward pass (no gradient computation)
        # -----------------------------------------------------------------
        output = model(images)

        # -----------------------------------------------------------------
        # Per-class / per-head evaluation
        # -----------------------------------------------------------------
        for i in range(len(target)):

            # Compute classification loss for this head/class.
            loss = criterion(output[i], target[i])

            # Compute top-1 accuracy.
            # accuracy(...) returns a list or tuple, hence the conversion.
            acc1 = accuracy(output[i], target[i], topk=(1,))
            acc1 = torch.Tensor(acc1).to(device='cuda')

            # Reduce metrics across all distributed processes.
            acc1 = reduce_tensor(acc1)
            loss = reduce_tensor(loss)

            # Update metric trackers.
            loss_meter[i].update(loss.item(), target[i].size(0))
            acc1_meter[i].update(acc1.item(), target[i].size(0))

            # -----------------------------------------------------------------
            # AUC computation preparation
            # -----------------------------------------------------------------
            # Convert logits to probabilities using softmax.
            preds = F.softmax(output[i], dim=1)

            # Accumulate predictions and labels across batches.
            # Data is stored on CPU as NumPy arrays for sklearn compatibility.
            if len(all_preds[i]) == 0:
                all_preds[i].append(preds.detach().cpu().numpy())
                all_label[i].append(target[i].detach().cpu().numpy())
            else:
                all_preds[i][0] = np.append(
                    all_preds[i][0],
                    preds.detach().cpu().numpy(),
                    axis=0
                )
                all_label[i][0] = np.append(
                    all_label[i][0],
                    target[i].detach().cpu().numpy(),
                    axis=0
                )

            # -----------------------------------------------------------------
            # Timing and logging
            # -----------------------------------------------------------------
            batch_time[i].update(time.time() - end)
            end = time.time()

            if idx % config.PRINT_FREQ == 0:
                memory_used = (
                    torch.cuda.max_memory_allocated() /
                    (1024.0 * 1024.0)
                )
                logger.info(
                    f'{valid_or_test}: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time[i].val:.3f} ({batch_time[i].avg:.3f})\t'
                    f'Loss {loss_meter[i].val:.4f} ({loss_meter[i].avg:.4f})\t'
                    f'Acc@1 {acc1_meter[i].val:.3f} ({acc1_meter[i].avg:.3f})\t'
                    #f'Acc@5 {acc5_meter[i].val:.3f} ({acc5_meter[i].avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB\t'
                    f'Class {i}'
                )

    # ---------------------------------------------------------------------
    # Final metric computation per class
    # ---------------------------------------------------------------------
    for i in range(14):

        # Extract accumulated predictions and labels.
        all_preds[i], all_label[i] = all_preds[i][0], all_label[i][0]

        # Compute AUC score.
        # Uses One-vs-Rest strategy for multi-class classification.
        # For binary classification, [:, 1] corresponds to the positive class.
        auc = roc_auc_score(
            all_label[i],
            all_preds[i][:, 1],
            multi_class='ovr'
        )

        logger.info(
            f' * Acc@1 {acc1_meter[i].avg:.3f}\t'
            f'Acc@5 {acc5_meter[i].avg:.3f}\t'
            f'{valid_or_test} AUC {auc:.5f}\t'
            f'Class {i}'
        )

        # Store metrics for global averaging.
        acc1s.append(acc1_meter[i].avg)
        acc5s.append(acc5_meter[i].avg)
        losses.append(loss_meter[i].avg)
        aucs.append(auc)

    # ---------------------------------------------------------------------
    # Mean metrics across all classes
    # ---------------------------------------------------------------------
    from statistics import mean
    logger.info(f'{valid_or_test} MEAN AUC: {mean(aucs):.5f}')

    # Return mean metrics for higher-level logging/checkpointing.
    return mean(acc1s), mean(acc5s), mean(losses)


@torch.no_grad()
def throughput(data_loader, model, logger):
    # Disable gradient computation globally for this function.
    # This reduces memory usage and avoids unnecessary autograd overhead, which is critical when benchmarking pure inference throughput.
    
    # Set the model to evaluation mode.
    # This disables training-specific layers such as Dropout and switches BatchNorm to use running statistics.
    model.eval()
    
    for idx, (images, _) in enumerate(data_loader):
        # Iterate over the data loader.
        # Only the first batch is used (due to the early return at the end).
        # Labels are ignored since throughput concerns only forward passes.

        # Move the input batch to GPU.
        # non_blocking=True allows asynchronous data transfer when the DataLoader uses pinned memory.
        images = images.cuda(non_blocking=True)

        # Extract batch size to compute throughput in samples/sec.
        batch_size = images.shape[0]
        
        # Warm-up phase:
        # Run several forward passes before actual timing.
        # This allows CUDA kernels to be initialized and cached, avoiding underestimation due to startup overhead.        
        for i in range(50):
            model(images)
        
        # Ensure that all queued CUDA operations are completed.
        # Without synchronization, timing measurements would be inaccurate because CUDA execution is asynchronous.        
        torch.cuda.synchronize()

        # Log the evaluation protocol:
        # throughput will be averaged over 30 forward passes.
        logger.info(f"throughput averaged with 30 times")

        # Start wall-clock timing (CPU-side).
        tic1 = time.time()
        
        # Perform 30 forward passes to measure steady-state inference speed.
        for i in range(30):
            model(images)
        
        # Synchronize again to ensure all GPU computations are finished
        # before stopping the timer.
        torch.cuda.synchronize()
        
        # End timing.
        tic2 = time.time()
        
        # Compute throughput:
        #   (number of forward passes × batch size) / elapsed time
        # Result is expressed in samples per second.
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}"
        )
        
        # Exit after processing a single batch.
        # This function is intended as a quick benchmark,
        # not a full pass over the dataset.
        return


if __name__ == '__main__':
    # Entry point of the training script.
    # This block is executed once per process in a distributed setup.

    # Parse command-line arguments and load the full configuration object.
    # The first return value is typically the args namespace, which is ignored here.
    _, config = parse_option()
    
    # Check whether the script is launched in a distributed environment (e.g., via torchrun or torch.distributed.launch).
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:        
        # rank: global process index across all nodes
        # world_size: total number of processes participating in training
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    
    # Fallback for non-distributed execution.
    else:
        # rank = -1 indicates single-process training.    
        rank = -1
        world_size = -1

    # Bind the current process to a specific GPU.
    # LOCAL_RANK is typically provided by torchrun and corresponds to the GPU index on the current node.
    torch.cuda.set_device(config.LOCAL_RANK)

    # Initialize the distributed process group.
    # - NCCL is the recommended backend for multi-GPU training on NVIDIA hardware.
    # - init_method='env://' reads rendezvous information from environment variables.
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Synchronize all processes.
    # Ensures that distributed initialization is completed before proceeding.
    torch.distributed.barrier()

    # Use a rank-dependent random seed.
    # This guarantees different random streams across processes while preserving global reproducibility.
    seed = config.SEED + dist.get_rank()

    # Seed PyTorch and NumPy RNGs.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Enable CuDNN auto-tuner.
    # This allows CuDNN to select the fastest convolution algorithms for the current input shapes, 
    # improving performance at the cost of determinism.
    cudnn.benchmark = True

    # ------------------------------------------------------------------
    # Learning rate scaling based on total batch size
    # ------------------------------------------------------------------

    # Linearly scale the learning rate according to the global batch size.
    # Reference batch size is 512, following common large-scale training heuristics.
    # Note: This heuristic may not be optimal for all setups.
    linear_scaled_lr = (
        config.TRAIN.BASE_LR
        * config.DATA.BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )

    linear_scaled_warmup_lr = (
        config.TRAIN.WARMUP_LR
        * config.DATA.BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )

    linear_scaled_min_lr = (
        config.TRAIN.MIN_LR
        * config.DATA.BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )

    # Gradient accumulation also requires learning rate scaling.
    # Accumulating gradients effectively increases the batch size
    # by ACCUMULATION_STEPS.
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr *= config.TRAIN.ACCUMULATION_STEPS

    # Temporarily unlock the configuration object to allow modification.
    config.defrost()

    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr

    # Re-freeze the configuration to prevent accidental changes.
    config.freeze()

    # ------------------------------------------------------------------
    # Logging and configuration persistence
    # ------------------------------------------------------------------

    # Create the output directory if it does not exist.
    # All logs, checkpoints, and configs will be stored here.
    os.makedirs(config.OUTPUT, exist_ok=True)

    # Initialize a distributed-aware logger.
    # Typically, only rank 0 writes full logs to disk.
    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=dist.get_rank(),
        name=f"{config.MODEL.NAME}"
    )

    if dist.get_rank() == 0:
        # Only the main process saves the full configuration to disk to avoid race conditions and duplicated files.
        
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        
        logger.info(f"Full config saved to {path}")

    # Print the full configuration (usually only visible from rank 0).
    logger.info(config.dump())

    # ------------------------------------------------------------------
    # Launch training / evaluation
    # ------------------------------------------------------------------

    # Call the main training loop with the fully initialized configuration.
    main(config)

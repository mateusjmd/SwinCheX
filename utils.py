# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist

def load_checkpoint(config, model, optimizer, lr_scheduler, logger, scaler=None):
    # Inform that the training is being resumed from a given checkpoint path or URL
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")

    # Load checkpoint either from a remote URL (e.g., official pretrained Swin weights)
    # or from a local file path
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME,
            map_location='cpu',     # always load to CPU first for safety/portability
            check_hash=True         # ensures integrity of downloaded weights
        )
    else:
        checkpoint = torch.load(
            config.MODEL.RESUME,
            map_location='cpu'
        )

    # Current model state dictionary (architecture already instantiated)
    sd = model.state_dict()

    # Manually copy parameters from the checkpoint into the current model
    for key, value in checkpoint['model'].items():

        # IMPORTANT:
        # The classification head is intentionally excluded.
        # This is the standard behavior when:
        #   - loading a pretrained backbone
        #   - but training on a different downstream task
        #     (e.g., different number of classes)
        #
        # If you want to *continue training* from an exact checkpoint
        # (same task, same head), this condition should be removed.
        if key != 'head.weight' and key != 'head.bias':
            sd[key] = value

    # Load the updated state dict into the model
    model.load_state_dict(sd)

    # ---- Alternative approaches (commented out) ----
    # These lines illustrate another common pattern:
    #   - explicitly reinitializing the classification head
    #   - loading the rest with strict=False
    #
    # checkpoint['model']['head.weight'] = torch.zeros(2, model.num_features)
    # checkpoint['model']['head.bias'] = torch.zeros(2)
    # msg = model.load_state_dict(checkpoint['model'], strict=False)
    # logger.info(msg)

    logger.info("Pretrain model loaded successfully!")

    # Default value for max validation accuracy (used for model selection)
    max_accuracy = 0.0

    # Only restore optimizer / scheduler / scaler state if:
    #   - we are NOT in pure evaluation mode
    #   - and the checkpoint corresponds to an actual training run
    if (
        not config.EVAL_MODE and
        'optimizer' in checkpoint and
        'lr_scheduler' in checkpoint and
        'epoch' in checkpoint
    ):
        # Restore optimizer internal state (momentum, adaptive moments, etc.)
        optimizer.load_state_dict(checkpoint['optimizer'])

        # Restore LR scheduler state (important for cosine decay / warmup continuity)
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        # Update the starting epoch so training resumes correctly
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()

        # Restore AMP GradScaler state if mixed precision is enabled
        if 'scaler' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])

        logger.info(
            f"=> loaded successfully '{config.MODEL.RESUME}' "
            f"(epoch {checkpoint['epoch']})"
        )

        # Restore best validation accuracy seen so far (for checkpointing logic)
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    # Explicitly free checkpoint memory
    del checkpoint
    torch.cuda.empty_cache()

    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, scaler=None):
    # Dictionary that stores the full training state required to resume training exactly
    save_state = {
        'model': model.state_dict(),                 # Model parameters and buffers
        'optimizer': optimizer.state_dict(),         # Optimizer internal state (e.g., momentum, Adam moments)
        'lr_scheduler': lr_scheduler.state_dict(),   # Learning rate scheduler state
        'max_accuracy': max_accuracy,                # Best accuracy observed so far
        'epoch': epoch,                              # Current epoch index
        'config': config                             # Full experiment configuration for reproducibility
    }

    # Save AMP GradScaler state to allow exact resumption of mixed-precision training
    if scaler is not None:
        save_state['scaler'] = scaler.state_dict()

    # Define checkpoint file path (one checkpoint per epoch)
    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")

    # Serialize checkpoint to disk
    torch.save(save_state, save_path)

    # Log successful save
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    # If a single tensor is provided, wrap it into a list for uniform processing
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter out parameters that do not have gradients (e.g., frozen layers)
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    # Convert norm type to float to support non-integer norms if needed
    norm_type = float(norm_type)

    total_norm = 0

    # Iterate over all parameters and accumulate their gradient norms
    for p in parameters:
        # Compute the norm of the gradient tensor for the current parameter
        param_norm = p.grad.data.norm(norm_type)

        # Accumulate the powered norm values (needed for global norm computation)
        total_norm += param_norm.item() ** norm_type

    # Take the norm_type-th root to obtain the global gradient norm
    total_norm = total_norm ** (1. / norm_type)

    # Return the total gradient norm (useful for logging or gradient clipping diagnostics)
    return total_norm


def auto_resume_helper(output_dir):
    # List all files in the output directory
    checkpoints = os.listdir(output_dir)

    # Filter only checkpoint files with '.pth' extension
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]

    # Log all discovered checkpoints (useful for debugging auto-resume behavior)
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")

    if len(checkpoints) > 0:
        # Select the most recently modified checkpoint file
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints],
            key=os.path.getmtime
        )

        # Log which checkpoint will be used for resuming
        print(f"The latest checkpoint founded: {latest_checkpoint}")

        resume_file = latest_checkpoint
    else:
        # No checkpoint found: training will start from scratch
        resume_file = None

    # Return path to the checkpoint to resume from (or None if not available)
    return resume_file


def reduce_tensor(tensor):
    # Create a clone of the input tensor to avoid modifying it in-place
    rt = tensor.clone()

    # Perform an all-reduce operation across all processes,
    # summing the tensor values from each distributed worker
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)

    # Average the reduced tensor by dividing by the total number of processes
    # This ensures that metrics (e.g., loss, accuracy) are globally consistent
    rt /= dist.get_world_size()

    # Return the globally averaged tensor
    return rt

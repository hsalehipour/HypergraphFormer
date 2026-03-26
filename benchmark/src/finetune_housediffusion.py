#!/usr/bin/env python3
"""
Fine-tune House-Diffusion on Hypergrph1K Dataset.

This script loads the converted Hypergrph1K Dataset and fine-tunes the House-Diffusion
diffusion model on it. It wraps the house_diffusion training infrastructure with
a user-friendly interface similar to the HouseGAN++ fine-tuning script.
"""

import argparse
import os
from pathlib import Path

# Note: house_diffusion must be installed first (pip install -e /path/to/house_diffusion)
# This is done automatically by setup_housediffusion.sh
from house_diffusion import dist_util, logger
from house_diffusion.rplanhg_datasets import load_rplanhg_data
from house_diffusion.resample import create_named_schedule_sampler
from house_diffusion.script_util import create_model_and_diffusion
from house_diffusion.train_util import TrainLoop


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune House-Diffusion on Hypergrph1K Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic fine-tuning with default settings (uses NPZ files in processed_rplan/)
  python finetune_housediffusion.py

  # Fine-tune with custom parameters
  python finetune_housediffusion.py \\
    --checkpoint ../../house_diffusion/checkpoints/pretrained_model250000.pt \\
    --exp_name ft_housediffusion \\
    --batch_size 16 \\
    --lr 1e-4 \\
    --target_set 8

  # Resume training from a checkpoint
  python finetune_housediffusion.py \\
    --resume ./checkpoints/ft_housediffusion_model050000.pt \\
    --target_set 8

Note:
  - Training data is loaded from house_diffusion/processed_rplan/ directory
  - Run setup_housediffusion.sh first to convert hypergraph-based pickle files to NPZ format
  - House-Diffusion uses step-based training (not epoch-based)
  - Since --max_steps is not coded in House-Diffusion, training runs indefinitely 
    until you stop it (Ctrl+C)
  - Checkpoints are saved every --save_interval steps (default: 5000)
"""
    )
    
    # Data parameters
    # Note: House-Diffusion loads data from processed_rplan/rplan_{set_name}_{target_set}.npz
    # The data must be converted to NPZ format first (see convert_pickle_to_npz.py)
    parser.add_argument("--target_set", type=int, default=None,
                        help="filter by number of rooms (None = no filter, use all samples)")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, 
                        default="../../house_diffusion/checkpoints/pretrained_model250000.pt",
                        help="path to pretrained diffusion model checkpoint")
    parser.add_argument("--resume", type=str, default=None,
                        help="path to checkpoint to resume training from (overrides --checkpoint)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate (default: 1e-3)")
    parser.add_argument("--analog_bit", action="store_true",
                        help="use binary bit representation instead of continuous")
    
    # Diffusion parameters
    parser.add_argument("--schedule_sampler", type=str, default="uniform",
                        choices=["uniform", "loss-second-moment"],
                        help="schedule sampler type (default: uniform)")
    parser.add_argument("--diffusion_steps", type=int, default=1000,
                        help="number of diffusion steps (default: 1000)")
    parser.add_argument("--noise_schedule", type=str, default="linear",
                        help="noise schedule type (default: linear)")
    
    # Model architecture (Transformer)
    parser.add_argument("--model_channels", type=int, default=512,
                        help="base channel count for the model (default: 512, must match pretrained checkpoint)")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of attention heads (default: 4, not used in current implementation)")
    
    # Output parameters
    parser.add_argument("--exp_name", type=str, default='hypergraph_diffusion',
                        help="experiment name for outputs")
    parser.add_argument("--log_interval", type=int, default=500,
                        help="interval for logging (default: 500)")
    parser.add_argument("--save_interval", type=int, default=5000,
                        help="interval for saving checkpoints (default: 5000)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="maximum number of training steps (default: None = train indefinitely)")
    
    # Advanced options
    parser.add_argument("--use_fp16", action="store_true",
                        help="use mixed precision training (fp16)")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="weight decay for optimizer (default: 0.05)")
    parser.add_argument("--ema_rate", type=str, default="0.9999",
                        help="comma-separated EMA rates (default: 0.9999)")
    
    parser.add_argument("--use_tensorboard", action="store_true",
                        help="enable TensorBoard logging (requires tensorflow)")
    
    return parser.parse_args()


def validate_npz_files(target_set):
    """Verify that the required NPZ files exist in processed_rplan/."""
    npz_dir = Path("./processed_rplan")
    target_set_str = "all" if target_set is None else str(target_set)
    train_file = npz_dir / f"rplan_train_{target_set_str}.npz"
    cndist_file = npz_dir / f"rplan_train_{target_set_str}_cndist.npz"
    
    if not train_file.exists():
        print(f"✗ Error: Training NPZ file not found at {train_file}")
        print(f"  Please run setup_housediffusion.sh first to convert the dataset")
        sys.exit(1)
    
    if not cndist_file.exists():
        print(f"✗ Error: Corner distribution file not found at {cndist_file}")
        print(f"  Please run setup_housediffusion.sh first to convert the dataset")
        sys.exit(1)


def print_header(opt):
    """Print configuration header."""
    target_set_str = "all" if opt.target_set is None else str(opt.target_set)
    print("=" * 80)
    print("HOUSE-DIFFUSION FINE-TUNING ON Hypergrph1K Dataset")
    print("=" * 80)
    print(f"Training data:       processed_rplan/rplan_train_{target_set_str}.npz")
    if opt.target_set is None:
        print(f"Target set (rooms):  None (using all samples)")
    else:
        print(f"Target set (rooms):  {opt.target_set}")
    print(f"Experiment name:     {opt.exp_name}")
    print(f"Batch size:          {opt.batch_size}")
    print(f"Learning rate:       {opt.lr}")
    print(f"Diffusion steps:     {opt.diffusion_steps}")
    print(f"Noise schedule:      {opt.noise_schedule}")
    print(f"Schedule sampler:    {opt.schedule_sampler}")
    print(f"Model channels:      {opt.model_channels}")
    print(f"Attention heads:     {opt.num_heads}")
    print(f"Use FP16:            {opt.use_fp16}")
    print(f"Analog bit mode:     {opt.analog_bit}")
    
    if opt.resume:
        print(f"Resume checkpoint:   {opt.resume}")
    else:
        print(f"Pretrained model:    {opt.checkpoint}")
    
    print(f"Log interval:        {opt.log_interval}")
    print(f"Save interval:       {opt.save_interval}")
    print("=" * 80)
    print()



def main():
    opt = parse_args()
    
    # Print configuration
    print_header(opt)
    
    # Validate NPZ files exist
    validate_npz_files(opt.target_set)
    
    # Setup distributed training (single GPU by default)
    dist_util.setup_dist()
    
    # Configure logger (house-diffusion's built-in logger)
    log_dir = f"./runs/{opt.exp_name}"
    
    # Add TensorBoard support if requested
    format_strs = None
    if opt.use_tensorboard:
        format_strs = ['log', 'csv', 'tensorboard']
        logger.log("TensorBoard logging enabled")
    
    logger.configure(dir=log_dir, format_strs=format_strs)
    
    logger.log(f"Logging to: {log_dir}")
    logger.log(f"Experiment: {opt.exp_name}")
    
    # Create model and diffusion
    logger.log("Creating model and diffusion...")
    
    # Build model configuration matching house_diffusion's expected parameters
    # Based on script_util.py:update_arg_parser() for RPLAN dataset
    num_coords = 16 if opt.analog_bit else 2
    input_channels = num_coords + (2*8 if not opt.analog_bit else 0)  # RPLAN format
    condition_channels = 89  # RPLAN room conditions
    out_channels = num_coords
    
    model_config = {
        'input_channels': input_channels,
        'condition_channels': condition_channels,
        'num_channels': opt.model_channels,
        'out_channels': out_channels,
        'dataset': 'hypergraph_rplan',
        'use_checkpoint': False,
        'use_unet': False,
        'analog_bit': opt.analog_bit,
        'diffusion_steps': opt.diffusion_steps,
        'noise_schedule': opt.noise_schedule,
        'learn_sigma': False,
        'timestep_respacing': '',
        'use_kl': False,
        'predict_xstart': False,
        'rescale_timesteps': False,
        'rescale_learned_sigmas': False,
        'target_set': opt.target_set,
        'set_name': 'train',
    }
    
    logger.log(f"Model configuration:")
    logger.log(f"  input_channels: {input_channels}")
    logger.log(f"  condition_channels: {condition_channels}")
    logger.log(f"  num_channels: {opt.model_channels}")
    logger.log(f"  out_channels: {out_channels}")
    logger.log(f"  analog_bit: {opt.analog_bit}")
    
    model, diffusion = create_model_and_diffusion(**model_config)
    model.to(dist_util.dev())
    
    # Load pretrained weights or resume
    if opt.resume:
        logger.log(f"Resuming from checkpoint: {opt.resume}")
        resume_checkpoint = opt.resume
    elif os.path.exists(opt.checkpoint):
        logger.log(f"Loading pretrained checkpoint: {opt.checkpoint}")
        resume_checkpoint = opt.checkpoint
    else:
        logger.log(f"⚠️  Warning: No checkpoint found, training from scratch")
        resume_checkpoint = ""
    
    # Create schedule sampler
    schedule_sampler = create_named_schedule_sampler(opt.schedule_sampler, diffusion)
    
    # Load data
    target_set_str = "all" if opt.target_set is None else str(opt.target_set)
    logger.log("Creating data loader...")
    logger.log(f"Loading NPZ data: processed_rplan/rplan_train_{target_set_str}.npz")
    
    # House-Diffusion loads data from processed_rplan/ directory
    # NPZ files must be created first using convert_pickle_to_npz.py
    data = load_rplanhg_data(
        batch_size=opt.batch_size,
        analog_bit=opt.analog_bit,
        target_set=opt.target_set,
        set_name='train',
    )
    
    # Get dataset size for informational logging
    import numpy as np
    target_set_str = "all" if opt.target_set is None else str(opt.target_set)
    npz_path = f"./processed_rplan/rplan_train_{target_set_str}.npz"
    try:
        npz_data = np.load(npz_path)
        dataset_size = npz_data['houses'].shape[0]
        steps_per_epoch = dataset_size / opt.batch_size
        
        logger.log("=" * 80)
        logger.log("📊 Training Information:")
        logger.log(f"   Dataset size: {dataset_size} samples")
        logger.log(f"   Batch size: {opt.batch_size}")
        logger.log(f"   Steps per epoch: ~{steps_per_epoch:.1f}")
        logger.log(f"")
        logger.log(f"   Approximate epoch equivalents:")
        logger.log(f"     100 steps   ≈ {100/steps_per_epoch:.1f} epochs")
        logger.log(f"     250 steps   ≈ {250/steps_per_epoch:.1f} epochs")
        logger.log(f"     500 steps   ≈ {500/steps_per_epoch:.1f} epochs")
        logger.log(f"     1000 steps  ≈ {1000/steps_per_epoch:.1f} epochs")
        logger.log(f"")
        if opt.max_steps:
            logger.log(f"   ⏱️  Max steps: {opt.max_steps} (~{opt.max_steps/steps_per_epoch:.1f} total epochs from start)")
        logger.log(f"   Save interval: {opt.save_interval} steps")
        logger.log(f"   Log interval: {opt.log_interval} steps")
        logger.log(f"")
        logger.log(f"   💡 Recommendation: For finetuning on small datasets,")
        logger.log(f"      train for 5-10 epochs only to avoid catastrophic forgetting")
        logger.log("=" * 80)
    except Exception as e:
        logger.log(f"⚠️  Could not load dataset size: {e}")
    
    # Determine stopping point
    # If max_steps is provided, use lr_anneal_steps as a stopping condition
    # (training stops when self.step + self.resume_step >= lr_anneal_steps)
    if opt.max_steps:
        lr_anneal_steps = opt.max_steps
        logger.log(f"⚠️  Training will stop after {opt.max_steps} total steps")
    else:
        lr_anneal_steps = 0  # Train indefinitely
        logger.log("⚠️  Training will run indefinitely (use Ctrl+C to stop)")
    
    logger.log("")
    logger.log("Starting training...")
    logger.log("=" * 80)
    
    # Create training loop
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=opt.batch_size,
        microbatch=-1,  # -1 disables microbatches
        lr=opt.lr,
        ema_rate=opt.ema_rate,
        log_interval=opt.log_interval,
        save_interval=opt.save_interval,
        resume_checkpoint=resume_checkpoint,
        use_fp16=opt.use_fp16,
        fp16_scale_growth=1e-3,
        schedule_sampler=schedule_sampler,
        weight_decay=opt.weight_decay,
        lr_anneal_steps=lr_anneal_steps,  # Use as stopping condition
        analog_bit=opt.analog_bit,
    ).run_loop()
    
    logger.log("=" * 80)
    logger.log(f"✅ Training complete!")
    logger.log(f"📁 Checkpoints saved in: ./checkpoints/")
    logger.log(f"📊 Logs saved in: {log_dir}")
    logger.log(f"   To view: tensorboard --logdir={log_dir}")
    logger.log("=" * 80)


if __name__ == "__main__":
    main()

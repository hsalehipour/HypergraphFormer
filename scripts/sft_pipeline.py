#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) Pipeline for Hypergraph Generation.

Fine-tunes a QWEN model on prepared training data using LoRA (Low-Rank
Adaptation) for efficient parameter-efficient fine-tuning.

Formerly ``train_qwen.py``.

IMPORTANT: This script runs in offline mode (HF_HUB_OFFLINE=1).
Make sure the model is cached locally before running:
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')"

Models are saved to: checkpoints/qwen_hypergraph_filtered/epochs{N}_lora_r{R}_alpha{ALPHA}/

Usage:
    # Basic training (single GPU, default: 3 epochs, r=64, alpha=128)
    python scripts/sft_pipeline.py

    # Multi-GPU training with Accelerate (recommended for 8 GPUs)
    accelerate launch --num_processes=8 scripts/sft_pipeline.py --multi-gpu

    # With custom epochs and LoRA settings
    python scripts/sft_pipeline.py --epochs 5 --lora-r 128 --lora-alpha 256

    # Custom output directory
    python scripts/sft_pipeline.py --output-dir my_custom_path/
"""

import os
from pathlib import Path
import argparse
import logging
import torch

# Use offline mode to avoid network issues with HuggingFace Hub
# Model must be cached locally first
os.environ['HF_HUB_OFFLINE'] = '1'
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_chat_messages(example, tokenizer):
    """
    Format chat messages into tokenized input for training.
    
    Args:
        example: Dictionary with 'messages' field
        tokenizer: Tokenizer instance
        
    Returns:
        dict: Tokenized input with 'input_ids', 'attention_mask', 'labels'
    """
    # Apply chat template
    text = tokenizer.apply_chat_template(
        example['messages'],
        tokenize=False,
        add_generation_prompt=False
    )
    
    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=6144,  # Covers 99.2% of data (only 8 samples truncated)
        padding=False,
        return_tensors=None
    )
    
    # For causal LM, labels are the same as input_ids (copy as list)
    tokenized['labels'] = list(tokenized['input_ids'])
    
    return tokenized


def load_training_data(data_dir):
    """
    Load training and validation datasets.
    
    Args:
        data_dir: Directory containing train.jsonl and validation.jsonl
        
    Returns:
        DatasetDict with train and validation splits
    """
    data_dir = Path(data_dir)
    
    dataset = load_dataset('json', data_files={
        'train': str(data_dir / 'train.jsonl'),
        'validation': str(data_dir / 'validation.jsonl')
    })
    
    logger.info(f"Loaded datasets:")
    logger.info(f"  Train: {len(dataset['train'])} samples")
    logger.info(f"  Validation: {len(dataset['validation'])} samples")
    
    return dataset


def setup_model_and_tokenizer(model_name, use_lora=True, lora_r=16, lora_alpha=32, multi_gpu=False):
    """
    Load model and tokenizer, optionally with LoRA.
    
    Args:
        model_name: Model name or path
        use_lora: Whether to use LoRA for efficient fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        multi_gpu: Whether training in multi-GPU mode
        
    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Loading tokenizer from cache: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"❌ Failed to load tokenizer. Model may not be cached locally.")
        logger.error(f"   Run this first to download: python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('{model_name}')\"")
        raise
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model from cache: {model_name}")
    
    # Load model in float16 for memory efficiency
    logger.info("Loading model in float16...")
    
    if multi_gpu:
        # For multi-GPU/DDP: Load on CPU first, let Accelerate/Trainer handle device placement
        logger.info("Multi-GPU mode: Loading model on CPU, Accelerate will distribute...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        logger.info("Model loaded on CPU, will be distributed by Accelerate")
    else:
        # For single-GPU: Load directly to GPU (CUDA_VISIBLE_DEVICES already set to target GPU)
        logger.info("Single-GPU mode: Loading directly to GPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",  # Will use the single visible GPU
            trust_remote_code=True
        )
        logger.info(f"Model loaded successfully on GPU")
    
    # Enable gradient checkpointing to save memory (if not already enabled)
    if not model.is_gradient_checkpointing:
        logger.info("Enabling gradient checkpointing for memory efficiency...")
        model.gradient_checkpointing_enable()
    
    if use_lora:
        logger.info("Setting up LoRA for efficient fine-tuning...")
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune QWEN for hypergraph generation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-4B-Instruct-2507',
                       help='Base model to fine-tune (default: Qwen/Qwen3-4B-Instruct-2507)')
    parser.add_argument('--data-dir', type=str, default='dataset/hypergraph/sft',
                       help='Directory with training data')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for fine-tuned model (default: checkpoints/qwen_hypergraph/lora_r{R}_alpha{ALPHA})')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=8,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size per device (default: 1)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                       help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='Learning rate (default: 2e-4)')
    parser.add_argument('--warmup-steps', type=int, default=100,
                       help='Number of warmup steps (default: 100)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='Max gradient norm for clipping (default: 1.0)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    
    # LoRA arguments
    parser.add_argument('--use-lora', action='store_true', default=True,
                       help='Use LoRA for efficient fine-tuning (default: True)')
    parser.add_argument('--no-lora', dest='use_lora', action='store_false',
                       help='Disable LoRA (full fine-tuning)')
    parser.add_argument('--lora-r', type=int, default=64,
                       help='LoRA rank (default: 64, higher = more parameters/capacity)')
    parser.add_argument('--lora-alpha', type=int, default=128,
                       help='LoRA alpha (default: 128, typically 2x rank)')
    
    # Other arguments
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID for single-GPU training (default: 0)')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use multi-GPU training with DistributedDataParallel (launch with accelerate)')
    parser.add_argument('--save-steps', type=int, default=100,
                       help='Save checkpoint every N steps (default: 100)')
    parser.add_argument('--eval-steps', type=int, default=100,
                       help='Evaluate every N steps (default: 100)')
    parser.add_argument('--logging-steps', type=int, default=10,
                       help='Log every N steps (default: 10)')
    
    args = parser.parse_args()
    
    # Set default output directory with dataset, epochs, and LoRA parameters in the name
    if args.output_dir is None:
        if args.use_lora:
            lora_suffix = f"epochs{args.epochs}_lora_r{args.lora_r}_alpha{args.lora_alpha}"
        else:
            lora_suffix = f"epochs{args.epochs}_full_finetune"
        args.output_dir = f"checkpoints/qwen_rplan_filtered/{lora_suffix}"
        logger.info(f"Using default output directory: {args.output_dir}")
    
    # Set CUDA_VISIBLE_DEVICES for single-GPU mode only
    # For multi-GPU, let accelerate handle device assignment
    if not args.multi_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        logger.info(f"Single-GPU mode: Setting CUDA_VISIBLE_DEVICES={args.gpu}")
    else:
        logger.info("Multi-GPU mode: Device assignment handled by Accelerate")
    
    logger.info("="*80)
    logger.info("🚀 QWEN Fine-tuning for Hypergraph Generation")
    logger.info("="*80)
    logger.info("📡 Running in OFFLINE mode (using cached models)")
    logger.info(f"Base model: {args.model}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Training mode: {'Multi-GPU (DDP)' if args.multi_gpu else f'Single-GPU (GPU {args.gpu})'}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"LoRA: {'Enabled' if args.use_lora else 'Disabled'}")
    if args.use_lora:
        logger.info(f"  - Rank: {args.lora_r}")
        logger.info(f"  - Alpha: {args.lora_alpha}")
    logger.info("="*80)
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! Training on CPU will be very slow.")
        return 1
    
    # Display GPU information
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.error("❌ No CUDA devices available")
        return 1
    
    if args.multi_gpu:
        logger.info(f"🎮 Multi-GPU Training: {num_gpus} GPU(s) available")
        for i in range(num_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")
    else:
        # Single GPU mode - after setting CUDA_VISIBLE_DEVICES, it's always device 0
        logger.info(f"🎮 Single-GPU Training on GPU {args.gpu}: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load data
    logger.info("\n📂 Loading training data...")
    dataset = load_training_data(args.data_dir)
    
    # Load model and tokenizer
    logger.info("\n🧠 Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(
        args.model,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        multi_gpu=args.multi_gpu
    )
    
    # Tokenize datasets
    logger.info("\n🔤 Tokenizing datasets...")
    tokenized_dataset = dataset.map(
        lambda x: format_chat_messages(x, tokenizer),
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing"
    )
    
    # Data collator - handles padding dynamically
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8  # For better GPU utilization
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,  # Disabled to avoid OOM during checkpoint loading in multi-GPU
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # Mixed precision training
        gradient_checkpointing=True,  # Save memory by recomputing activations
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        save_total_limit=50,  # Keep only last 10 checkpoints
        # DDP-specific settings for better multi-GPU stability
        ddp_find_unused_parameters=False,  # Set to True if you get DDP errors with LoRA
        ddp_backend="nccl" if args.multi_gpu and torch.cuda.is_available() else None,
    )
    
    # Initialize trainer
    logger.info("\n🏋️  Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
    )
    
    # Train from scratch
    logger.info("\n🚀 Starting training from scratch...")
    logger.info("="*80)
    
    train_result = trainer.train()
    
    # Save final model
    logger.info("\n💾 Saving final model...")
    trainer.save_model()
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Skip final evaluation to avoid confusion (would evaluate last checkpoint, not best)
    # Best model metrics are already logged during training at each eval_steps
    
    logger.info("\n" + "="*80)
    logger.info("✅ Training complete!")
    logger.info("="*80)
    logger.info(f"📁 Final checkpoint saved to: {args.output_dir}")
    logger.info(f"📊 Logs saved to: {args.output_dir}/logs")
    logger.info(f"\nTraining metrics:")
    logger.info(f"  Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    
    # Log best checkpoint info
    if hasattr(trainer.state, 'best_model_checkpoint') and trainer.state.best_model_checkpoint:
        logger.info(f"\n🏆 Best checkpoint (use this for inference):")
        logger.info(f"   Path: {trainer.state.best_model_checkpoint}")
        logger.info(f"   Best eval loss: {trainer.state.best_metric:.4f}")
    
    logger.info("="*80)
    
    logger.info("\n💡 To use the fine-tuned model:")
    logger.info(f"   python scripts/fewshot_generator.py --backend qwen --model {args.output_dir} --from-dataset")
    logger.info(f"\n💡 To resume training from a checkpoint:")
    logger.info(f"   python scripts/sft_pipeline.py --output-dir {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())


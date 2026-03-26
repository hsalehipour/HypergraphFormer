#!/usr/bin/env python3
"""
Benchmark QWEN Model Inference Timing

This script tests the inference timing of QWEN models across different
input token lengths. It measures:
- Time to encode tokens
- Time to generate output
- Total inference time
- Tokens per second

Usage:
    python utils/benchmark_qwen_inference.py --model Qwen/Qwen2.5-0.5B-Instruct
    python utils/benchmark_qwen_inference.py --model Qwen/Qwen2.5-7B-Instruct --device cuda
    python utils/benchmark_qwen_inference.py --model path/to/local/model --token-lengths 128,256,512,1024
"""

import argparse
import time
import json
from pathlib import Path
from typing import List, Dict
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name: str, device: str = "auto"):
    """
    Load QWEN model and tokenizer.
    
    Args:
        model_name: Model name or path (e.g., "Qwen/Qwen2.5-7B-Instruct")
        device: Device to load model on ("cpu", "cuda", "auto")
        
    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device in ["cuda", "auto"] and torch.cuda.is_available() else torch.float32,
        device_map=device if device == "auto" else None,
        trust_remote_code=True
    )
    
    # Move to device if not using device_map
    if device != "auto":
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        model = model.to(device)
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Device: {next(model.parameters()).device}")
    logger.info(f"Model dtype: {next(model.parameters()).dtype}")
    
    return model, tokenizer


def generate_prompt(token_length: int, tokenizer) -> str:
    """
    Generate a prompt of approximately the specified token length.
    
    Args:
        token_length: Target number of tokens
        tokenizer: Tokenizer to use for counting tokens
        
    Returns:
        str: Generated prompt
    """
    # Base template
    base_text = """You are an expert AI assistant. Please analyze the following information carefully and provide detailed insights.

Context: """
    
    # Filler text to reach desired token length
    filler = "The process of analyzing complex data structures involves understanding relationships, patterns, and hierarchical organizations. " * 100
    
    # Combine and truncate to desired length
    full_text = base_text + filler
    tokens = tokenizer.encode(full_text)
    
    # Truncate to desired length
    if len(tokens) > token_length:
        tokens = tokens[:token_length]
    else:
        # Add more filler if needed
        while len(tokens) < token_length:
            additional = "Additional context information. "
            additional_tokens = tokenizer.encode(additional, add_special_tokens=False)
            tokens.extend(additional_tokens)
        tokens = tokens[:token_length]
    
    return tokenizer.decode(tokens)


def benchmark_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Benchmark a single inference run.
    
    Args:
        model: The model to benchmark
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum new tokens to generate
        device: Device the model is on
        
    Returns:
        dict: Timing statistics
    """
    # Encode prompt
    start_encode = time.time()
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to device
    if device != "auto":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    input_length = inputs["input_ids"].shape[1]
    encode_time = time.time() - start_encode
    
    # Generate
    start_generate = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for fair comparison
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    # Ensure generation is complete (synchronize for CUDA)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    generate_time = time.time() - start_generate
    
    # Decode output
    start_decode = time.time()
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decode_time = time.time() - start_decode
    
    output_length = outputs.shape[1]
    new_tokens = output_length - input_length
    
    total_time = encode_time + generate_time + decode_time
    tokens_per_second = new_tokens / generate_time if generate_time > 0 else 0
    
    return {
        "input_tokens": input_length,
        "output_tokens": new_tokens,
        "encode_time": encode_time,
        "generate_time": generate_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "tokens_per_second": tokens_per_second,
        "output_text": output_text[:100] + "..." if len(output_text) > 100 else output_text
    }


def run_benchmark(
    model,
    tokenizer,
    token_lengths: List[int],
    num_runs: int = 3,
    max_new_tokens_list: List[int] = [50],
    warmup_runs: int = 2,
    device: str = "cuda"
) -> pd.DataFrame:
    """
    Run benchmark across multiple token lengths and generation lengths.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer
        token_lengths: List of input token lengths to test
        num_runs: Number of runs per length (for averaging)
        max_new_tokens_list: List of max new tokens to generate
        warmup_runs: Number of warmup runs before benchmarking
        device: Device model is on
        
    Returns:
        pd.DataFrame: Results dataframe
    """
    logger.info("="*80)
    logger.info("🚀 Starting Benchmark")
    logger.info("="*80)
    logger.info(f"Token lengths: {token_lengths}")
    logger.info(f"Max new tokens: {max_new_tokens_list}")
    logger.info(f"Runs per combination: {num_runs}")
    logger.info(f"Total tests: {len(token_lengths) * len(max_new_tokens_list) * num_runs}")
    logger.info(f"Warmup runs: {warmup_runs}")
    logger.info("="*80)
    
    # Warmup
    if warmup_runs > 0:
        logger.info("\n🔥 Warming up model...")
        warmup_prompt = generate_prompt(128, tokenizer)
        for i in range(warmup_runs):
            _ = benchmark_inference(model, tokenizer, warmup_prompt, max_new_tokens_list[0], device)
            logger.info(f"  Warmup run {i+1}/{warmup_runs} complete")
    
    results = []
    
    for max_new_tokens in max_new_tokens_list:
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 Testing with max_new_tokens = {max_new_tokens}")
        logger.info(f"{'='*80}")
        
        for token_length in token_lengths:
            logger.info(f"\n📊 Input token length: {token_length}")
            prompt = generate_prompt(token_length, tokenizer)
            
            run_results = []
            for run in range(num_runs):
                result = benchmark_inference(model, tokenizer, prompt, max_new_tokens, device)
                run_results.append(result)
                logger.info(f"  Run {run+1}/{num_runs}: ✓ ({result['total_time']:.3f}s, {result['tokens_per_second']:.1f} tok/s)")
            
            # Average results
            avg_result = {
                "token_length": token_length,
                "input_tokens": run_results[0]["input_tokens"],
                "max_new_tokens": max_new_tokens,
                "output_tokens": sum(r["output_tokens"] for r in run_results) / num_runs,
                "encode_time": sum(r["encode_time"] for r in run_results) / num_runs,
                "generate_time": sum(r["generate_time"] for r in run_results) / num_runs,
                "decode_time": sum(r["decode_time"] for r in run_results) / num_runs,
                "total_time": sum(r["total_time"] for r in run_results) / num_runs,
                "tokens_per_second": sum(r["tokens_per_second"] for r in run_results) / num_runs,
            }
            results.append(avg_result)
    
    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame, output_dir: Path, model_name: str):
    """
    Create visualization plots for benchmark results.
    
    Args:
        df: Results dataframe
        output_dir: Directory to save plots
        model_name: Model name for titles
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we have multiple max_new_tokens values
    unique_max_tokens = df['max_new_tokens'].unique()
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#95a5a6']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'QWEN Model Inference Benchmark: {model_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Total time vs input tokens (grouped by max_new_tokens)
    ax1 = axes[0, 0]
    for idx, max_tokens in enumerate(unique_max_tokens):
        subset = df[df['max_new_tokens'] == max_tokens]
        color = colors[idx % len(colors)]
        ax1.plot(subset['input_tokens'], subset['total_time'], 'o-', 
                linewidth=2, markersize=8, color=color, label=f'{max_tokens} new tokens')
    ax1.set_xlabel('Input Tokens', fontsize=12)
    ax1.set_ylabel('Total Time (seconds)', fontsize=12)
    ax1.set_title('Total Inference Time vs Input Length', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if len(unique_max_tokens) > 1:
        ax1.legend()
    
    # Plot 2: Tokens per second (grouped by max_new_tokens)
    ax2 = axes[0, 1]
    for idx, max_tokens in enumerate(unique_max_tokens):
        subset = df[df['max_new_tokens'] == max_tokens]
        color = colors[idx % len(colors)]
        ax2.plot(subset['input_tokens'], subset['tokens_per_second'], 'o-',
                linewidth=2, markersize=8, color=color, label=f'{max_tokens} new tokens')
    ax2.set_xlabel('Input Tokens', fontsize=12)
    ax2.set_ylabel('Tokens/Second', fontsize=12)
    ax2.set_title('Generation Speed', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if len(unique_max_tokens) > 1:
        ax2.legend()
    
    # Plot 3: Time breakdown (stacked bar for first max_new_tokens)
    ax3 = axes[1, 0]
    # Use data from first max_new_tokens value
    subset = df[df['max_new_tokens'] == unique_max_tokens[0]]
    x = range(len(subset))
    width = 0.6
    ax3.bar(x, subset['encode_time'], width, label='Encode', color='#e74c3c')
    ax3.bar(x, subset['generate_time'], width, bottom=subset['encode_time'], 
            label='Generate', color='#f39c12')
    ax3.bar(x, subset['decode_time'], width, 
            bottom=subset['encode_time'] + subset['generate_time'], 
            label='Decode', color='#9b59b6')
    ax3.set_xlabel('Input Token Length', fontsize=12)
    ax3.set_ylabel('Time (seconds)', fontsize=12)
    title = f'Time Breakdown (max_new_tokens={unique_max_tokens[0]})'
    ax3.set_title(title, fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{int(t)}" for t in subset['input_tokens']], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Generate time vs input tokens (grouped by max_new_tokens)
    ax4 = axes[1, 1]
    for idx, max_tokens in enumerate(unique_max_tokens):
        subset = df[df['max_new_tokens'] == max_tokens]
        color = colors[idx % len(colors)]
        ax4.plot(subset['input_tokens'], subset['generate_time'], 'o-',
                linewidth=2, markersize=8, color=color, label=f'{max_tokens} new tokens')
    ax4.set_xlabel('Input Tokens', fontsize=12)
    ax4.set_ylabel('Generation Time (seconds)', fontsize=12)
    ax4.set_title('Generation Time vs Input Length', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    if len(unique_max_tokens) > 1:
        ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'benchmark_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Plot saved to: {plot_path}")
    
    plt.close()


def save_results(df: pd.DataFrame, output_dir: Path, model_name: str, args: argparse.Namespace):
    """
    Save benchmark results to files.
    
    Args:
        df: Results dataframe
        output_dir: Output directory
        model_name: Model name
        args: Command line arguments
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_dir / 'benchmark_results.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"💾 CSV saved to: {csv_path}")
    
    # Save as JSON with metadata
    results_dict = {
        "model": model_name,
        "device": args.device,
        "max_new_tokens_list": args.max_new_tokens,
        "num_runs": args.num_runs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": df.to_dict(orient='records')
    }
    
    json_path = output_dir / 'benchmark_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"💾 JSON saved to: {json_path}")
    
    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("📈 BENCHMARK RESULTS SUMMARY")
    logger.info("="*80)
    print("\n" + df.to_string(index=False))
    logger.info("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark QWEN model inference timing across different token lengths"
    )
    parser.add_argument('--model', type=str,
                       default='Qwen/Qwen3-4B-Instruct-2507',
                       help='Model name or path (default: Qwen/Qwen3-4B-Instruct-2507)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to run on (default: auto)')
    parser.add_argument('--token-lengths', type=str,
                       default='64,128,256,512,1024,2048',
                       help='Comma-separated list of token lengths to test (default: 64,128,256,512,1024,2048)')
    parser.add_argument('--max-new-tokens', type=str, default='50',
                       help='Comma-separated list of max new tokens to generate (default: 50)')
    parser.add_argument('--num-runs', type=int, default=3,
                       help='Number of runs per combination (default: 3)')
    parser.add_argument('--warmup-runs', type=int, default=2,
                       help='Number of warmup runs (default: 2)')
    parser.add_argument('--output-dir', type=str,
                       default='benchmark_results',
                       help='Output directory for results (default: benchmark_results)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Parse token lengths
    token_lengths = [int(x.strip()) for x in args.token_lengths.split(',')]
    token_lengths.sort()
    
    # Parse max new tokens
    max_new_tokens_list = [int(x.strip()) for x in args.max_new_tokens.split(',')]
    max_new_tokens_list.sort()
    
    logger.info("="*80)
    logger.info("🤖 QWEN MODEL INFERENCE BENCHMARK")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Token lengths: {token_lengths}")
    logger.info(f"Max new tokens: {max_new_tokens_list}")
    logger.info(f"Runs per combination: {args.num_runs}")
    logger.info(f"Total tests: {len(token_lengths) * len(max_new_tokens_list) * args.num_runs}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*80)
    
    try:
        # Load model
        model, tokenizer = load_model_and_tokenizer(args.model, args.device)
        
        # Run benchmark
        results_df = run_benchmark(
            model=model,
            tokenizer=tokenizer,
            token_lengths=token_lengths,
            num_runs=args.num_runs,
            max_new_tokens_list=max_new_tokens_list,
            warmup_runs=args.warmup_runs,
            device=args.device
        )
        
        # Save results
        output_dir = Path(args.output_dir)
        # Store parsed values back to args for save_results
        args.max_new_tokens = max_new_tokens_list
        save_results(results_df, output_dir, args.model, args)
        
        # Plot results
        if not args.no_plot:
            plot_results(results_df, output_dir, args.model)
        
        logger.info("\n✅ Benchmark complete!")
        logger.info(f"📁 Results saved to: {output_dir.absolute()}")
        
    except Exception as e:
        logger.error(f"❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


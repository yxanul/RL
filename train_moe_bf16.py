"""
Enhanced training script with BF16 support for MoE Model
Optimized for modern GPUs (A100, H100, RTX 4090, etc.)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import argparse
from tqdm import tqdm
import time
import json
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import wandb

from tutel import system, net
from tutel import moe as tutel_moe
from moe_model import ExperimentalMoEModel, MoEConfig, moe_stats_enable, moe_stats_drain
from dataset import OptimizedDataLoader


def setup_wandb(args, model_config, parallel_env):
    """Initialize W&B with comprehensive MoE tracking"""
    if parallel_env.global_rank == 0:  # Only rank 0 logs to W&B
        wandb.init(
            project="moe-training",
            name=f"moe_{args.num_experts}e_{args.num_experts_per_token}t_{args.precision}",
            config={
                # Model config
                "model": {
                    "hidden_dim": model_config.hidden_dim,
                    "num_layers": model_config.num_layers,
                    "num_attention_heads": model_config.num_attention_heads,
                    "num_experts": model_config.num_experts,
                    "num_experts_per_token": model_config.num_experts_per_token,
                    "vocab_size": model_config.vocab_size,
                    "expert_hidden_dim": model_config.expert_hidden_dim,
                    "num_key_value_heads": model_config.num_key_value_heads,
                },
                # Training config
                "training": {
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "warmup_steps": args.warmup_steps,
                    "max_steps": args.max_steps,
                    "precision": args.precision,
                },
                # Hardware
                "hardware": {
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                    "num_gpus": torch.cuda.device_count(),
                }
            },
            tags=[f"moe-{args.num_experts}", args.precision, "tutel"],
        )
        return True
    return False


def setup_distributed():
    """Initialize distributed training environment"""
    parallel_env = system.init_data_model_parallel(
        backend='nccl' if torch.cuda.is_available() else 'gloo'
    )
    return parallel_env


def check_bf16_support():
    """Check if the current device supports BF16"""
    if not torch.cuda.is_available():
        return False

    # Check compute capability (7.0+ for BF16)
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)

    # BF16 requires compute capability 8.0+ (A100, RTX 3090, etc.)
    # or 9.0+ (H100)
    supports_bf16 = capability[0] >= 8

    if supports_bf16:
        print(f"Device supports BF16 (Compute Capability: {capability[0]}.{capability[1]})")
    else:
        print(f"Device does not support BF16 (Compute Capability: {capability[0]}.{capability[1]})")
        print("Will fall back to FP16 for mixed precision")

    return supports_bf16


def create_model(config, device, parallel_env):
    """Create and initialize the MoE model"""
    model = ExperimentalMoEModel(config).to(device)

    # Print model info only on rank 0
    if parallel_env.global_rank == 0:
        print(f"\nModel created on {device}")
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Total parameters: {param_count:.2f}M")

    return model


def create_optimizer(model, args):
    """Create optimizer with parameter groups"""
    # Separate parameters
    expert_params = []
    non_expert_params = []

    for name, param in model.named_parameters():
        if hasattr(param, 'skip_allreduce'):
            expert_params.append(param)
        else:
            non_expert_params.append(param)

    # Create parameter groups with different learning rates
    param_groups = [
        {'params': non_expert_params, 'lr': args.learning_rate},
        {'params': expert_params, 'lr': args.learning_rate * args.expert_lr_scale}
    ]

    # AdamW optimizer with weight decay and fused kernels for better performance
    optimizer = optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
        fused=torch.cuda.is_available()  # Use fused optimizer on GPU for better performance
    )

    return optimizer


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate schedule with linear warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_gradient_norm(model):
    """Compute total gradient norm across all parameters"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_model_tflops(batch_size, seq_len, config, time_elapsed, is_training=True):
    """Compute TFLOPS for the model - corrected calculation"""
    # Reference: https://arxiv.org/pdf/2104.04473.pdf (Efficient Large-Scale Language Model Training)
    # For each transformer layer:
    # - Attention: 2 * batch * seq_len * seq_len * hidden_dim (for QKV projections)
    # - Attention computation: 2 * batch * num_heads * seq_len * seq_len * (hidden_dim/num_heads)
    # - FFN: 8 * batch * seq_len * hidden_dim * ffn_dim (two linear layers)

    # Simplified calculation for transformer blocks
    # Attention block (including QKV, attention, output projection)
    attn_flops_per_layer = 2 * batch_size * seq_len * (3 * config.hidden_dim * config.hidden_dim +  # QKV
                                                        2 * seq_len * config.hidden_dim +  # Attention
                                                        config.hidden_dim * config.hidden_dim)  # Output proj

    # MoE FFN (only active experts count)
    # For SwiGLU: 3 matrices (gate, up, down) so it's 3x the standard FFN
    active_experts = config.num_experts_per_token
    ffn_flops_per_layer = 3 * 2 * batch_size * seq_len * config.hidden_dim * config.expert_hidden_dim * active_experts

    # Gate computation for routing
    gate_flops_per_layer = 2 * batch_size * seq_len * config.hidden_dim * config.num_experts

    # Total per layer
    flops_per_layer = attn_flops_per_layer + ffn_flops_per_layer + gate_flops_per_layer

    # Total for all layers
    total_flops = flops_per_layer * config.num_layers

    # Training multiplier (forward + backward + gradient computation)
    if is_training:
        total_flops *= 3

    # Convert to TFLOPS/s
    tflops = total_flops / (time_elapsed * 1e12)

    return tflops


def get_gpu_memory_info():
    """Get detailed GPU memory information"""
    if not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    # Memory efficiency ratio
    efficiency = (allocated / reserved * 100) if reserved > 0 else 0

    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated,
        'efficiency_percent': efficiency
    }


def extract_moe_metrics(model, outputs=None):
    """Extract MoE-specific metrics for monitoring

    Since Tutel doesn't expose detailed routing statistics by default,
    we extract what we can from auxiliary losses and gate parameters.
    """
    metrics = {}

    # Track auxiliary losses from each layer
    aux_losses = []
    num_moe_layers = 0

    # Find MoE layers and extract available statistics
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'moe_layer'):
            moe_layer = layer.moe_layer
            num_moe_layers += 1

            # Get auxiliary loss (load balancing loss)
            if hasattr(moe_layer, 'l_aux') and moe_layer.l_aux is not None:
                aux_loss_val = moe_layer.l_aux.item() if hasattr(moe_layer.l_aux, 'item') else float(moe_layer.l_aux)
                metrics[f'layer_{i}/aux_loss'] = aux_loss_val
                aux_losses.append(aux_loss_val)

            # Try to get gate weight statistics (routing parameters)
            if hasattr(moe_layer, 'gates'):
                gates = moe_layer.gates
                if gates is not None:
                    # Gate linear layer exists
                    if hasattr(gates, 'weight'):
                        gate_weights = gates.weight.data
                        metrics[f'layer_{i}/gate_weight_mean'] = gate_weights.mean().item()
                        metrics[f'layer_{i}/gate_weight_std'] = gate_weights.std().item()
                        metrics[f'layer_{i}/gate_weight_norm'] = gate_weights.norm().item()

            # Check for any exposed routing scores or counts
            if hasattr(moe_layer, 'gates_s'):
                # Sometimes Tutel stores softmax scores here
                gates_s = moe_layer.gates_s
                if gates_s is not None and gates_s.numel() > 0:
                    # Calculate entropy of routing decisions
                    routing_probs = F.softmax(gates_s, dim=-1)
                    entropy = -(routing_probs * (routing_probs + 1e-10).log()).sum(dim=-1).mean()
                    metrics[f'layer_{i}/routing_entropy'] = entropy.item()

            # Check for load balancing statistics if available
            if hasattr(moe_layer, 'expert_count'):
                expert_count = moe_layer.expert_count
                if expert_count is not None:
                    # Normalize to get load distribution
                    total_count = expert_count.sum()
                    if total_count > 0:
                        expert_load = expert_count.float() / total_count
                        metrics[f'layer_{i}/load_balance_variance'] = expert_load.var().item()
                        metrics[f'layer_{i}/max_expert_load'] = expert_load.max().item()
                        metrics[f'layer_{i}/min_expert_load'] = expert_load.min().item()

    # Calculate overall metrics
    if aux_losses:
        metrics['overall/mean_aux_loss'] = sum(aux_losses) / len(aux_losses)
        metrics['overall/max_aux_loss'] = max(aux_losses)
        metrics['overall/min_aux_loss'] = min(aux_losses)

    # Add the total auxiliary loss from model outputs if available
    if outputs and 'aux_loss' in outputs:
        metrics['overall/total_aux_loss'] = outputs['aux_loss'].item() if hasattr(outputs['aux_loss'], 'item') else float(outputs['aux_loss'])

    metrics['overall/num_moe_layers'] = num_moe_layers

    return metrics


def train_step(model, batch, optimizer, scaler, parallel_env, args):
    """Single training step with mixed precision and MoE metrics"""
    model.train()
    step_start_time = time.time()

    # Move batch to device
    input_ids = batch['input_ids']
    labels = batch['labels']

    # Configure autocast based on precision type
    if args.precision == 'bf16':
        autocast_dtype = torch.bfloat16
    elif args.precision == 'fp16':
        autocast_dtype = torch.float16
    else:  # fp32
        autocast_dtype = None

    # Mixed precision forward pass
    if autocast_dtype:
        with autocast(enabled=True, dtype=autocast_dtype):
            outputs = model(input_ids, None, labels)
            loss = outputs['loss']
            aux_loss = outputs['aux_loss']

            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
    else:
        outputs = model(input_ids, None, labels)
        loss = outputs['loss']
        aux_loss = outputs['aux_loss']
        loss = loss / args.gradient_accumulation_steps

    # Extract MoE metrics before backward pass
    moe_metrics = extract_moe_metrics(model, outputs)

    # Backward pass
    if args.precision == 'fp16':
        # FP16 requires GradScaler
        scaler.scale(loss).backward()
    else:
        # BF16 and FP32 don't need GradScaler
        loss.backward()

    # All-reduce gradients for non-expert parameters
    if parallel_env.global_size > 1:
        for param in model.parameters():
            if not hasattr(param, 'skip_allreduce') and param.grad is not None:
                param.grad = net.simple_all_reduce(param.grad) / parallel_env.global_size

    # Compute gradient norm before clipping (for monitoring)
    grad_norm = compute_gradient_norm(model)

    step_time = time.time() - step_start_time

    return loss.item() * args.gradient_accumulation_steps, aux_loss.item(), grad_norm, step_time, moe_metrics


@torch.no_grad()
def evaluate(model, dataloader, device, num_eval_steps=50):
    """Evaluate model perplexity"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(dataloader):
        if i >= num_eval_steps:
            break

        input_ids = batch['input_ids']
        labels = batch['labels']

        outputs = model(input_ids, None, labels)
        loss = outputs['loss']

        total_loss += loss.item() * input_ids.size(0)
        total_tokens += input_ids.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return avg_loss, perplexity.item()


def main():
    parser = argparse.ArgumentParser(description='Train MoE Language Model with BF16 Support')

    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=768, help='Model hidden dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num_experts', type=int, default=16, help='Total number of experts')
    parser.add_argument('--num_experts_per_token', type=int, default=2, help='Experts per token (top-k)')

    # Precision arguments
    parser.add_argument('--precision', type=str, default='bf16',
                       choices=['fp32', 'fp16', 'bf16'],
                       help='Training precision (fp32/fp16/bf16)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--expert_lr_scale', type=float, default=1.0, help='LR scale for expert parameters')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum training steps')
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--save_steps', type=int, default=1000, help='Checkpoint save interval')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length')

    # Data arguments
    parser.add_argument('--dataset_name', type=str, default='HuggingFaceFW/fineweb-edu', help='Dataset name')
    parser.add_argument('--num_workers', type=int, default=None, help='DataLoader workers')
    parser.add_argument('--log_interval', type=int, default=50, help='Steps between detailed logs')

    # System arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--profile', action='store_true', help='Enable PyTorch profiler')
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='moe-training', help='W&B project name')

    args = parser.parse_args()

    # Setup distributed training
    parallel_env = setup_distributed()
    device = parallel_env.local_device

    # Check BF16 support and adjust precision if needed
    if args.precision == 'bf16' and torch.cuda.is_available():
        if not check_bf16_support():
            print("BF16 not supported, falling back to FP16")
            args.precision = 'fp16'
    elif args.precision in ['bf16', 'fp16'] and not torch.cuda.is_available():
        print("No CUDA available, falling back to FP32")
        args.precision = 'fp32'

    # Print training configuration
    if parallel_env.global_rank == 0:
        print("\n" + "="*60)
        print("Training Configuration")
        print("="*60)
        print(f"Precision: {args.precision.upper()}")
        print(f"Batch size: {args.batch_size} x {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Max steps: {args.max_steps}")
        print("="*60 + "\n")

    # Set default dtype based on precision
    if args.precision == 'bf16':
        torch.set_default_dtype(torch.bfloat16)
    elif args.precision == 'fp16':
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(torch.float32)

    # Create output directory
    if parallel_env.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        # Save training config
        with open(os.path.join(args.output_dir, 'training_config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    # Create model configuration
    config = MoEConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        num_experts_per_token=args.num_experts_per_token,
        max_seq_length=args.max_length
    )

    # Create model
    model = create_model(config, device, parallel_env)

    # Create optimizer
    optimizer = create_optimizer(model, args)

    # Initialize W&B if enabled
    is_main_rank = parallel_env.global_rank == 0
    if args.use_wandb and is_main_rank:
        setup_wandb(args, config, parallel_env)

    # Create learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )

    # Create GradScaler only for FP16
    if args.precision == 'fp16':
        scaler = GradScaler()
        print("Using GradScaler for FP16 training")
    else:
        scaler = None
        if args.precision == 'bf16':
            print("Using native BF16 (no GradScaler needed)")

    # Create data loader
    if parallel_env.global_rank == 0:
        print("\nInitializing dataset...")

    dataloader = OptimizedDataLoader(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        force_cpu=not torch.cuda.is_available(),
        verbose=(parallel_env.global_rank == 0)
    )

    # Training info
    if parallel_env.global_rank == 0:
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        print(f"Precision: {args.precision.upper()}")
        print(f"Device: {device}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e6:.2f}M")

        # Calculate theoretical peak TFLOPS for the GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            # Peak TFLOPS for BF16/FP16 Tensor Cores (not FP32!)
            gpu_peak_tflops = {
                'A100': 312,      # A100 80GB - BF16 tensor cores
                'A6000': 155,     # RTX A6000 - FP16 tensor cores
                'V100': 125,      # V100 - FP16 tensor cores
                '4090': 330,      # RTX 4090 - ~330 TFLOPS BF16 tensor cores (not 82.6 FP32!)
                '4080': 195,      # RTX 4080 - ~195 TFLOPS BF16 tensor cores
                '3090': 142,      # RTX 3090 - FP16 tensor cores (no BF16 support)
                'H100': 989,      # H100 80GB - BF16 tensor cores
                'A40': 150,       # A40 - FP16 tensor cores
                'L40': 362,       # L40 - FP16 tensor cores
            }

            peak_tflops = 100  # Default estimate
            for gpu_key, tflops_val in gpu_peak_tflops.items():
                if gpu_key in gpu_name:
                    peak_tflops = tflops_val
                    break

            print(f"GPU: {gpu_name}")
            print(f"Theoretical Peak TFLOPS: ~{peak_tflops}")

        print("="*60 + "\n")

    # Initialize metrics tracking
    loss_window = deque(maxlen=100)
    aux_loss_window = deque(maxlen=100)
    grad_norm_window = deque(maxlen=100)
    tflops_window = deque(maxlen=50)
    tokens_processed = 0
    training_start_time = time.time()
    last_log_time = time.time()
    step_times = deque(maxlen=100)

    # Model configuration for TFLOPS calculation
    model_config = config

    # Optional profiling
    profiler = None
    if args.profile and torch.cuda.is_available() and parallel_env.global_rank == 0:
        print("PyTorch Profiler enabled - will profile steps 100-110")
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=99, warmup=1, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(args.output_dir, 'profiler')),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()

    # Training loop
    model.train()
    optimizer.zero_grad()
    global_step = 0

    pbar = tqdm(total=args.max_steps, desc="Training", disable=(parallel_env.global_rank != 0), unit="step")

    for batch_idx, batch in enumerate(dataloader):
        if global_step >= args.max_steps:
            break

        # Training step
        loss, aux_loss, grad_norm, step_time, moe_metrics = train_step(
            model, batch, optimizer, scaler, parallel_env, args
        )

        # Update metrics
        loss_window.append(loss)
        aux_loss_window.append(aux_loss)
        grad_norm_window.append(grad_norm)
        step_times.append(step_time)

        # Gradient accumulation
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # Optimizer step
            if args.precision == 'fp16':
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Step profiler if enabled
            if profiler is not None:
                profiler.step()

            # Update token count
            tokens_processed += args.batch_size * args.gradient_accumulation_steps * args.max_length

            # Calculate metrics
            avg_loss = np.mean(loss_window) if loss_window else loss
            avg_aux_loss = np.mean(aux_loss_window) if aux_loss_window else aux_loss
            avg_grad_norm = np.mean(grad_norm_window) if grad_norm_window else grad_norm
            perplexity = np.exp(min(avg_loss, 20))  # Cap to avoid overflow

            # Calculate TFLOPS (using actual step time, not per-sample time)
            # Note: step_time is for single batch, but we accumulate gradients
            # So effective computation is batch_size * gradient_accumulation_steps
            effective_batch = args.batch_size * args.gradient_accumulation_steps

            # Use the average time per gradient accumulation cycle
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 and step_times:
                # Time for full gradient accumulation cycle
                cycle_time = sum(list(step_times)[-args.gradient_accumulation_steps:])

                tflops = compute_model_tflops(
                    batch_size=effective_batch,
                    seq_len=args.max_length,
                    config=model_config,
                    time_elapsed=cycle_time,
                    is_training=True
                )
                tflops_window.append(tflops)
                avg_tflops = np.mean(tflops_window) if tflops_window else tflops
            else:
                avg_tflops = np.mean(tflops_window) if tflops_window else 0

            # Calculate tokens/sec
            current_time = time.time()
            elapsed_time = current_time - training_start_time
            tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0

            # Calculate ETA
            steps_remaining = args.max_steps - global_step
            if global_step > 0:
                time_per_step = elapsed_time / global_step
                eta_seconds = time_per_step * steps_remaining
                eta_str = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta_str = "--:--:--"

            # Update progress bar with comprehensive metrics
            if parallel_env.global_rank == 0:
                # Get memory info
                mem_info = get_gpu_memory_info()

                # Simple progress bar (minimal terminal output)
                pbar_dict = {
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{perplexity:.2f}',
                }
                pbar.set_postfix(pbar_dict)
                pbar.update(1)

                # Enable MoE tracking every 50 steps for detailed routing info
                capture_moe_routing = (global_step % 50 == 0)
                if capture_moe_routing:
                    moe_stats_enable(True)

                # Log to W&B instead of terminal
                if args.use_wandb and global_step % 10 == 0:  # Log to W&B every 10 steps
                    wandb_metrics = {
                        # Core metrics
                        'train/loss': avg_loss,
                        'train/perplexity': perplexity,
                        'train/aux_loss': avg_aux_loss,
                        'train/grad_norm': avg_grad_norm,
                        'train/learning_rate': scheduler.get_last_lr()[0],

                        # Performance metrics
                        'performance/tokens_per_sec': tokens_per_sec,
                        'performance/tflops': avg_tflops,
                        'performance/step_time_ms': np.mean(step_times) * 1000 if step_times else 0,
                        'performance/tokens_processed': tokens_processed,

                        # GPU metrics
                        'gpu/memory_allocated_gb': mem_info['allocated_gb'],
                        'gpu/memory_reserved_gb': mem_info['reserved_gb'],
                        'gpu/memory_efficiency': mem_info['efficiency_percent'],
                    }

                    # Add MoE-specific metrics
                    for key, value in moe_metrics.items():
                        wandb_metrics[f'moe/{key}'] = value

                    # Calculate GPU utilization if possible
                    if torch.cuda.is_available() and avg_tflops > 0:
                        gpu_name = torch.cuda.get_device_name()
                        for gpu_key in ['4090', '3090', 'A100', 'V100', 'H100', 'A40', 'L40', '4080']:
                            if gpu_key in gpu_name:
                                peak = gpu_peak_tflops.get(gpu_key, 100)
                                utilization = (avg_tflops / peak) * 100
                                wandb_metrics['gpu/compute_utilization'] = utilization
                                wandb_metrics['gpu/peak_tflops'] = peak
                                break

                    # Add detailed routing stats if we captured them
                    if capture_moe_routing:
                        layer_stats = moe_stats_drain(reset=True)
                        if layer_stats:
                            for li, st in enumerate(layer_stats):
                                tok = st["tokens"].numpy()
                                total = tok.sum()
                                if total > 0:
                                    # Per-expert token counts
                                    for expert_id in range(len(tok)):
                                        wandb_metrics[f'routing/layer_{li}/expert_{expert_id}_tokens'] = int(tok[expert_id])

                                    # Load distribution metrics
                                    load_pct = (tok / max(total, 1e-9)) * 100.0
                                    wandb_metrics[f'routing/layer_{li}/load_std'] = float(load_pct.std())
                                    wandb_metrics[f'routing/layer_{li}/load_cv'] = float(load_pct.std() / (load_pct.mean() + 1e-9))
                                    wandb_metrics[f'routing/layer_{li}/max_load'] = float(load_pct.max())
                                    wandb_metrics[f'routing/layer_{li}/min_load'] = float(load_pct.min())

                                    # Find dead experts (< 1% of traffic)
                                    dead_experts = (load_pct < 1.0).sum()
                                    wandb_metrics[f'routing/layer_{li}/dead_experts'] = int(dead_experts)

                                    # Find dominant experts (> 20% of traffic)
                                    dominant_experts = (load_pct > 20.0).sum()
                                    wandb_metrics[f'routing/layer_{li}/dominant_experts'] = int(dominant_experts)

                        moe_stats_enable(False)  # Disable until next capture

                    wandb.log(wandb_metrics, step=global_step)

                # Minimal terminal logging (only every 100 steps)
                if global_step % 100 == 0:
                    # Minimal console output
                    print(f"Step {global_step}/{args.max_steps} | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | Tokens/s: {tokens_per_sec:.0f} | TFLOPS: {avg_tflops:.1f}")

                    # If we have routing stats, print a summary
                    if capture_moe_routing:
                        layer_stats = moe_stats_drain(reset=False)  # Don't reset, already drained above
                        if layer_stats:
                            print("\nMoE Routing Summary (tokens per expert):")
                            for li, st in enumerate(layer_stats):
                                tok = st["tokens"].numpy()
                                total = tok.sum()
                                if total == 0:
                                    continue
                                load_pct = (tok / max(total, 1e-9)) * 100.0
                                cv = load_pct.std() / (load_pct.mean() + 1e-9)
                                top3 = tok.argsort()[-3:][::-1]
                                bot3 = tok.argsort()[:3]
                                dead = (load_pct < 1.0).sum()
                                print(f"  L{li:02d} | CV: {cv:.3f} | Dead: {dead} | Top-3: {list(zip(top3, load_pct[top3].round(1)))} | Bot-3: {list(zip(bot3, load_pct[bot3].round(1)))}")
                            print()

                    last_log_time = current_time

    pbar.close()

    if parallel_env.global_rank == 0:
        total_time = time.time() - training_start_time
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Final step: {global_step}")
        print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
        print(f"Total tokens processed: {tokens_processed:,}")
        print(f"Average tokens/sec: {tokens_processed/total_time:.0f}")

        if loss_window:
            final_loss = np.mean(list(loss_window)[-10:])  # Last 10 losses
            final_ppl = np.exp(min(final_loss, 20))
            print(f"Final loss (last 10 steps): {final_loss:.4f}")
            print(f"Final perplexity: {final_ppl:.2f}")

        print("="*70)

        # Finish W&B run
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
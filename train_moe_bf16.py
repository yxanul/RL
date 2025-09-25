"""
Enhanced training script with BF16 support for MoE Model
Optimized for modern GPUs (A100, H100, RTX 4090, etc.)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import argparse
from tqdm import tqdm
import time
import json
from datetime import datetime

from tutel import system, net
from moe_model import ExperimentalMoEModel, MoEConfig
from dataset import OptimizedDataLoader


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

    # AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay
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


def train_step(model, batch, optimizer, scaler, parallel_env, args):
    """Single training step with mixed precision"""
    model.train()

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

    return loss.item() * args.gradient_accumulation_steps, aux_loss.item()


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
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
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

    # System arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')

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
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print("="*60 + "\n")

    # Training loop
    model.train()
    optimizer.zero_grad()
    global_step = 0

    pbar = tqdm(total=args.max_steps, desc="Training", disable=(parallel_env.global_rank != 0))

    for batch_idx, batch in enumerate(dataloader):
        if global_step >= args.max_steps:
            break

        # Training step
        loss, aux_loss = train_step(
            model, batch, optimizer, scaler, parallel_env, args
        )

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

            # Update progress bar
            if parallel_env.global_rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'aux': f'{aux_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                pbar.update(1)

    pbar.close()

    if parallel_env.global_rank == 0:
        print("\nTraining completed!")
        print(f"Final step: {global_step}")


if __name__ == "__main__":
    main()
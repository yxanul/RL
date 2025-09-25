"""
Training Script for Experimental MoE Model
Integrates optimized dataset loading with Tutel MoE training
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
    # Initialize Tutel's parallel environment
    parallel_env = system.init_data_model_parallel(
        backend='nccl' if torch.cuda.is_available() else 'gloo'
    )
    return parallel_env


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
    # Note: We don't use attention_mask since we have no padding (drop_last=True)
    # and use causal masking in the model

    # Mixed precision forward pass
    with autocast(enabled=args.use_amp):
        outputs = model(input_ids, None, labels)  # Pass None for attention_mask
        loss = outputs['loss']
        aux_loss = outputs['aux_loss']

        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps

    # Backward pass
    if args.use_amp:
        scaler.scale(loss).backward()
    else:
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
        # Don't use attention_mask with causal masking

        outputs = model(input_ids, None, labels)  # Pass None for attention_mask
        loss = outputs['loss']

        total_loss += loss.item() * input_ids.size(0)
        total_tokens += input_ids.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return avg_loss, perplexity.item()


def save_checkpoint(model, optimizer, scheduler, epoch, step, args, parallel_env):
    """Save training checkpoint"""
    if parallel_env.global_rank != 0:
        return

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'args': vars(args),
        'config': vars(model.config)
    }

    checkpoint_path = os.path.join(
        args.output_dir,
        f'checkpoint_epoch{epoch}_step{step}.pt'
    )
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train MoE Language Model')

    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=1536, help='Model hidden dimension')
    parser.add_argument('--num_layers', type=int, default=16, help='Number of transformer layers')
    parser.add_argument('--num_experts', type=int, default=16, help='Total number of experts')
    parser.add_argument('--num_experts_per_token', type=int, default=2, help='Experts per token (top-k)')

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
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')

    args = parser.parse_args()

    # Setup distributed training
    parallel_env = setup_distributed()
    device = parallel_env.local_device

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

    # Mixed precision scaler
    scaler = GradScaler(enabled=args.use_amp)

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['step']
        print(f"Resumed from checkpoint: epoch {start_epoch}, step {global_step}")

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

    # Training loop
    if parallel_env.global_rank == 0:
        print("\nStarting training...")
        print(f"Total steps: {args.max_steps}")
        print(f"Warmup steps: {args.warmup_steps}")
        print(f"Batch size: {args.batch_size} x {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps}")

    # Initialize metrics
    train_losses = []
    aux_losses = []
    step_times = []

    # Training
    model.train()
    optimizer.zero_grad()

    pbar = tqdm(total=args.max_steps, desc="Training", disable=(parallel_env.global_rank != 0))

    for batch_idx, batch in enumerate(dataloader):
        if global_step >= args.max_steps:
            break

        start_time = time.time()

        # Training step
        loss, aux_loss = train_step(
            model, batch, optimizer, scaler, parallel_env, args
        )

        train_losses.append(loss)
        aux_losses.append(aux_loss)

        # Gradient accumulation
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # Optimizer step
            if args.use_amp:
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

            # Log metrics
            step_time = time.time() - start_time
            step_times.append(step_time)

            if parallel_env.global_rank == 0:
                avg_loss = sum(train_losses[-10:]) / min(10, len(train_losses))
                avg_aux_loss = sum(aux_losses[-10:]) / min(10, len(aux_losses))
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'aux_loss': f'{avg_aux_loss:.6f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    'step_time': f'{step_time:.2f}s'
                })
                pbar.update(1)

            # Evaluation
            if global_step % args.eval_steps == 0:
                eval_loss, perplexity = evaluate(model, dataloader, device)
                if parallel_env.global_rank == 0:
                    print(f"\n[Step {global_step}] Eval Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
                model.train()

            # Save checkpoint
            if global_step % args.save_steps == 0:
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch=global_step // len(dataloader),
                    step=global_step,
                    args=args,
                    parallel_env=parallel_env
                )

    pbar.close()

    # Final evaluation
    if parallel_env.global_rank == 0:
        print("\nTraining completed!")
        eval_loss, perplexity = evaluate(model, dataloader, device, num_eval_steps=100)
        print(f"Final Eval Loss: {eval_loss:.4f}")
        print(f"Final Perplexity: {perplexity:.2f}")

        # Save final model
        save_checkpoint(
            model, optimizer, scheduler,
            epoch='final',
            step=global_step,
            args=args,
            parallel_env=parallel_env
        )


if __name__ == "__main__":
    main()
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from typing import Optional, Iterator, Dict
import multiprocessing as mp
import platform
import os

class StreamingTextDataset(IterableDataset):
    """
    Optimized streaming dataset for large-scale text pretraining.
    Implements efficient batching and tokenization with minimal overhead.
    """
    
    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        dataset_config: str = "default",
        tokenizer_name: str = "mistralai/Mistral-7B-v0.1",  # or your preferred tokenizer
        max_length: int = 2048,
        buffer_size: int = 10000,  # Buffer for shuffling
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.seed = seed
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # CRITICAL FIX: Initialize dataset ONCE here, not in __iter__
        # This prevents rate limit errors from multiple workers making API calls
        print(f"Initializing dataset {dataset_name} (one-time initialization)...")
        self._base_dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    
    def __iter__(self) -> Iterator[dict]:
        # Get worker info for proper sharding in multi-worker setup
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single worker
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        # FIXED: Use the pre-initialized dataset instead of creating new ones
        # This avoids hitting HuggingFace rate limits (1000 requests/5min)
        if num_workers > 1:
            # For multiple workers, use skip for efficient sharding
            dataset_shard = self._base_dataset.skip(worker_id)
        else:
            dataset_shard = self._base_dataset

        # Shuffle with worker-specific seed
        dataset_shard = dataset_shard.shuffle(seed=self.seed + worker_id, buffer_size=self.buffer_size)
        dataset_iter = iter(dataset_shard)
        
        # Process items with stride equal to number of workers
        token_buffer = []
        items_processed = 0

        for item in dataset_iter:
            # Only process every num_workers-th item for this worker
            if num_workers > 1 and items_processed % num_workers != 0:
                items_processed += 1
                continue
            items_processed += 1
            
            # Extract text (adjust field name based on your dataset)
            text = item.get('text', '') or item.get('content', '')
            
            if text:
                # Tokenize
                tokens = self.tokenizer(
                    text,
                    truncation=False,
                    padding=False,
                    return_attention_mask=False,
                )['input_ids']
                
                token_buffer.extend(tokens)
                
                # Yield complete chunks
                while len(token_buffer) >= self.max_length:
                    chunk = token_buffer[:self.max_length]
                    token_buffer = token_buffer[self.max_length:]
                    
                    # Convert to tensors
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    
                    # Create labels (shifted input_ids for next-token prediction)
                    labels = input_ids.clone()
                    
                    yield {
                        'input_ids': input_ids,
                        'labels': labels,
                        'attention_mask': torch.ones_like(input_ids)
                    }


def collate_fn(batch):
    """
    Custom collate function for efficient batching.
    Pre-allocates tensors for better memory usage.
    """
    if not batch:
        return {}
    
    batch_size = len(batch)
    max_length = batch[0]['input_ids'].shape[0]
    
    # Pre-allocate tensors with pinned memory only if CUDA is available
    has_cuda = torch.cuda.is_available()
    input_ids = torch.zeros((batch_size, max_length), dtype=torch.long, pin_memory=has_cuda)
    labels = torch.zeros((batch_size, max_length), dtype=torch.long, pin_memory=has_cuda)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long, pin_memory=has_cuda)
    
    # Fill tensors
    for i, item in enumerate(batch):
        input_ids[i] = item['input_ids']
        labels[i] = item['labels']
        attention_mask[i] = item['attention_mask']
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }


class OptimizedDataLoader:
    """
    Production-ready dataloader with automatic GPU/CPU optimization.
    Automatically detects environment and optimizes accordingly.
    """

    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        dataset_config: str = "default",
        batch_size: int = 8,
        max_length: int = 2048,
        num_workers: int = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        force_cpu: bool = False,  # Force CPU mode for testing
        verbose: bool = False,  # Enable detailed logging
    ):
        self.verbose = verbose
        self.force_cpu = force_cpu

        # Detect environment
        self.platform = platform.system()
        self.has_cuda = torch.cuda.is_available() and not force_cpu
        self.cpu_count = mp.cpu_count()

        # Auto-configure workers based on environment
        if num_workers is None:
            if self.platform == 'Windows':
                # Windows: Use workers only if explicitly running on GPU
                num_workers = 4 if self.has_cuda else 0
            else:
                # Linux/Mac: Optimal for GPU training environments
                if self.has_cuda:
                    num_workers = min(self.cpu_count, 8)  # GPU training: use multiple workers
                else:
                    num_workers = min(self.cpu_count // 2, 4)  # CPU testing: fewer workers

        self.num_workers = num_workers

        # Set device before printing config
        self.device = torch.device('cuda' if self.has_cuda else 'cpu')

        if self.verbose:
            self._print_config()

        self.dataset = StreamingTextDataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            max_length=max_length,
        )

        # Configure DataLoader with environment-specific optimizations
        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': self.num_workers,
            'collate_fn': collate_fn,
            'drop_last': True,  # Consistent batch sizes for training
        }

        # GPU optimizations
        if self.has_cuda:
            dataloader_kwargs['pin_memory'] = True  # Fast CPU→GPU transfer
            if self.num_workers > 0:
                dataloader_kwargs['prefetch_factor'] = prefetch_factor
                dataloader_kwargs['persistent_workers'] = persistent_workers
        else:
            # CPU mode: disable GPU-specific optimizations
            dataloader_kwargs['pin_memory'] = False
            if self.num_workers > 0:
                dataloader_kwargs['prefetch_factor'] = prefetch_factor
                dataloader_kwargs['persistent_workers'] = False  # Save memory on CPU

        self.dataloader = DataLoader(self.dataset, **dataloader_kwargs)

        # Pre-warm the dataloader
        self._iterator = None

    def _print_config(self):
        """Print detailed configuration for debugging."""
        print("\n" + "="*60)
        print("DataLoader Configuration")
        print("="*60)
        print(f"Platform: {self.platform}")
        print(f"CUDA Available: {self.has_cuda}")
        print(f"Device: {self.device}")
        print(f"CPU Count: {self.cpu_count}")
        print(f"Workers: {self.num_workers}")
        print(f"Pin Memory: {self.has_cuda}")
        print(f"Force CPU: {self.force_cpu}")
        print("="*60 + "\n")
    
    def __iter__(self):
        """
        Optimized iteration with non-blocking GPU transfers.
        Handles both GPU and CPU environments gracefully.
        """
        for batch in self.dataloader:
            if not batch:  # Skip empty batches
                continue

            # Transfer to device with non-blocking for GPU
            if self.has_cuda:
                batch_device = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch.items()
                }
            else:
                # CPU mode: regular transfer
                batch_device = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                }
            yield batch_device
    
    def __len__(self):
        return float('inf')  # Streaming dataset has no fixed length


# Minimal training example
def train_step(model, batch, optimizer, scaler=None):
    """
    Single training step with mixed precision support.
    """
    if scaler is not None:
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
    return loss.item()


def test_dataloader_performance(force_cpu: bool = False, num_batches: int = 100):
    """
    Test function to benchmark dataloader performance.
    Can be used for both CPU testing and GPU training verification.
    """
    import time
    from tqdm import tqdm

    print("\n" + "="*60)
    print("DataLoader Performance Test")
    print("="*60)

    # Initialize optimized dataloader with automatic configuration
    try:
        dataloader = OptimizedDataLoader(
            dataset_name="HuggingFaceFW/fineweb-edu",
            dataset_config="default",
            batch_size=16,
            max_length=2048,
            num_workers=None,  # Auto-detect optimal value
            prefetch_factor=2,
            persistent_workers=True,
            force_cpu=force_cpu,  # Can force CPU mode for testing
            verbose=True,  # Show configuration
        )
    except Exception as e:
        print(f"\n❌ Error initializing dataloader: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nStarting performance benchmark...")
    print(f"Testing {num_batches} batches...\n")

    # Pre-warm: First batch is always slower
    try:
        iterator = iter(dataloader)
        print("Loading first batch (pre-warming)...")
        first_batch = next(iterator)
        print(f"✓ First batch loaded successfully!")
        print(f"  Batch shapes: input_ids={first_batch['input_ids'].shape}")
        print(f"  Device: {first_batch['input_ids'].device}")
    except StopIteration:
        print("❌ Dataset appears to be empty")
        return
    except Exception as e:
        print(f"\n❌ Error loading first batch: {e}")
        import traceback
        traceback.print_exc()
        return

    # Benchmark
    start_time = time.time()
    batch_times = []

    try:
        for i, batch in enumerate(tqdm(iterator, total=num_batches-1, desc="Loading batches")):
            if i >= num_batches - 1:
                break

            batch_start = time.time()

            # Simulate model work
            if torch.cuda.is_available() and not force_cpu:
                # GPU: simulate forward pass
                dummy_compute = torch.mm(
                    batch['input_ids'].float(),
                    torch.randn(2048, 768, device=batch['input_ids'].device)
                )
                torch.cuda.synchronize()  # Wait for GPU
            else:
                # CPU: lighter computation
                dummy_compute = batch['input_ids'].sum()

            batch_times.append(time.time() - batch_start)

    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

    # Results
    elapsed = time.time() - start_time
    actual_batches = min(len(batch_times) + 1, num_batches)  # +1 for first batch

    print(f"\n" + "="*60)
    print("Benchmark Results")
    print("="*60)
    print(f"Total batches processed: {actual_batches}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average time per batch: {elapsed/actual_batches*1000:.2f}ms")
    print(f"Throughput: {actual_batches * 16 / elapsed:.2f} samples/sec")

    if batch_times:
        print(f"\nBatch timing statistics (excluding first batch):")
        print(f"  Min: {min(batch_times)*1000:.2f}ms")
        print(f"  Max: {max(batch_times)*1000:.2f}ms")
        print(f"  Mean: {np.mean(batch_times)*1000:.2f}ms")
        print(f"  Std: {np.std(batch_times)*1000:.2f}ms")

    # Memory stats
    if torch.cuda.is_available() and not force_cpu:
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test optimized DataLoader")
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode even if GPU is available')
    parser.add_argument('--batches', type=int, default=100, help='Number of batches to test')
    args = parser.parse_args()

    # Run performance test
    test_dataloader_performance(force_cpu=args.cpu, num_batches=args.batches)
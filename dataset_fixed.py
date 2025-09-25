"""
Optimized streaming dataset implementation that avoids HuggingFace rate limits
"""

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
    Streaming dataset that efficiently loads data without hitting API rate limits.
    Key optimizations:
    - Single dataset initialization shared across workers
    - Proper worker sharding without redundant API calls
    - Local caching support
    """

    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        dataset_config: str = "default",
        tokenizer_name: str = "gpt2",
        max_length: int = 2048,
        seed: int = 42,
        buffer_size: int = 10_000,
        cache_dir: Optional[str] = None,
        num_proc: int = 1,  # For tokenization parallelism
        streaming: bool = True,  # Can disable for small datasets
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.max_length = max_length
        self.seed = seed
        self.buffer_size = buffer_size
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/datasets")
        self.num_proc = num_proc
        self.streaming = streaming

        # Initialize tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset ONCE at initialization, not in __iter__
        # This prevents multiple API calls from different workers
        if self.streaming:
            # For streaming, we'll use a single shared dataset
            self._base_dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split="train",
                streaming=True,
                cache_dir=self.cache_dir,
                trust_remote_code=True,  # Required for some datasets
            )
        else:
            # For non-streaming, load the full dataset once
            self._base_dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split="train",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

    def __iter__(self) -> Iterator[Dict]:
        """
        Iterate through the dataset with proper worker sharding
        """
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single process data loading
            worker_id = 0
            num_workers = 1
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # For streaming datasets, use skip/take for efficient sharding
        if self.streaming:
            # Each worker skips to their portion and takes every num_workers-th item
            # This is more efficient than the previous approach
            dataset_shard = self._base_dataset.skip(worker_id)

            # Create iterator for this worker's shard
            dataset_iter = iter(dataset_shard)

            # Process with proper striding
            token_buffer = []
            items_processed = 0

            for item in dataset_iter:
                # Skip items not meant for this worker (stride pattern)
                if items_processed % num_workers != 0:
                    items_processed += 1
                    continue

                items_processed += 1

                # Extract text from the dataset
                text = self._extract_text(item)
                if not text:
                    continue

                # Tokenize the text
                tokens = self.tokenizer(
                    text,
                    truncation=False,
                    padding=False,
                    return_attention_mask=False,
                )['input_ids']

                # Add to buffer
                token_buffer.extend(tokens)

                # Yield complete sequences
                while len(token_buffer) >= self.max_length:
                    sequence = token_buffer[:self.max_length]
                    token_buffer = token_buffer[self.max_length:]

                    yield self._prepare_sample(sequence)
        else:
            # For non-streaming, use index-based sharding
            total_size = len(self._base_dataset)
            per_worker = total_size // num_workers
            start_idx = worker_id * per_worker
            end_idx = start_idx + per_worker if worker_id < num_workers - 1 else total_size

            token_buffer = []

            for idx in range(start_idx, end_idx):
                item = self._base_dataset[idx]
                text = self._extract_text(item)

                if not text:
                    continue

                tokens = self.tokenizer(
                    text,
                    truncation=False,
                    padding=False,
                    return_attention_mask=False,
                )['input_ids']

                token_buffer.extend(tokens)

                while len(token_buffer) >= self.max_length:
                    sequence = token_buffer[:self.max_length]
                    token_buffer = token_buffer[self.max_length:]

                    yield self._prepare_sample(sequence)

    def _extract_text(self, item: dict) -> str:
        """Extract text from dataset item"""
        # Common field names in text datasets
        for field in ['text', 'content', 'document', 'passage']:
            if field in item:
                return item[field]

        # If no known field, try to get the first string field
        for value in item.values():
            if isinstance(value, str):
                return value

        return ""

    def _prepare_sample(self, tokens: list) -> dict:
        """Prepare a sample for training"""
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        # Create attention mask (all ones since no padding)
        attention_mask = torch.ones_like(input_ids)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }


def fast_collate_fn(batch: list, max_length: int = 2048) -> dict:
    """
    Fast collation with pre-allocated tensors
    """
    batch_size = len(batch)

    # Pre-allocate tensors
    input_ids = torch.zeros((batch_size, max_length - 1), dtype=torch.long)
    labels = torch.zeros((batch_size, max_length - 1), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length - 1), dtype=torch.long)

    for i, sample in enumerate(batch):
        seq_len = sample['input_ids'].size(0)
        input_ids[i, :seq_len] = sample['input_ids']
        labels[i, :seq_len] = sample['labels']
        attention_mask[i, :seq_len] = sample['attention_mask']

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    }


class OptimizedDataLoader:
    """
    Optimized DataLoader that avoids rate limits and maximizes throughput
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
        force_cpu: bool = False,
        verbose: bool = False,
        cache_dir: Optional[str] = None,
        use_fast_tokenizer: bool = True,
    ):
        self.verbose = verbose
        self.force_cpu = force_cpu
        self.batch_size = batch_size
        self.max_length = max_length

        # Detect environment
        self.platform = platform.system()
        self.has_cuda = torch.cuda.is_available() and not force_cpu
        self.cpu_count = mp.cpu_count()

        # Configure workers based on environment
        if num_workers is None:
            if self.platform == 'Windows':
                # Windows: Conservative worker count
                num_workers = 0 if not self.has_cuda else 2
            else:
                # Linux/Mac: Optimal for training
                if self.has_cuda:
                    num_workers = min(self.cpu_count, 8)
                else:
                    num_workers = min(self.cpu_count // 2, 4)

        self.num_workers = num_workers

        if self.verbose:
            self._print_config()

        # Create the dataset with optimizations
        self.dataset = StreamingTextDataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            tokenizer_name="gpt2",
            max_length=max_length,
            cache_dir=cache_dir,
            streaming=True,  # Always use streaming for large datasets
        )

        # Configure DataLoader with environment-specific optimizations
        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': self.num_workers,
            'collate_fn': lambda x: fast_collate_fn(x, max_length),
            'drop_last': True,
        }

        # GPU optimizations
        if self.has_cuda:
            dataloader_kwargs['pin_memory'] = True
            if self.num_workers > 0:
                dataloader_kwargs['prefetch_factor'] = prefetch_factor
                dataloader_kwargs['persistent_workers'] = persistent_workers
        else:
            dataloader_kwargs['pin_memory'] = False
            if self.num_workers > 0:
                dataloader_kwargs['prefetch_factor'] = prefetch_factor
                dataloader_kwargs['persistent_workers'] = False

        self.dataloader = DataLoader(self.dataset, **dataloader_kwargs)

        # Device for data transfer
        self.device = torch.device('cuda' if self.has_cuda else 'cpu')

    def _print_config(self):
        """Print configuration details"""
        print("\n" + "="*60)
        print("DataLoader Configuration")
        print("="*60)
        print(f"Platform: {self.platform}")
        print(f"CUDA Available: {self.has_cuda}")
        print(f"CPU Count: {self.cpu_count}")
        print(f"Workers: {self.num_workers}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Max Length: {self.max_length}")
        print(f"Pin Memory: {self.has_cuda}")
        print("="*60 + "\n")

    def __iter__(self):
        """Iterate with optimized data transfer"""
        for batch in self.dataloader:
            if not batch:
                continue

            # Transfer to device
            if self.has_cuda:
                batch_device = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch.items()
                }
            else:
                batch_device = batch

            yield batch_device

    def __len__(self):
        """Length is undefined for streaming datasets"""
        return float('inf')  # Streaming dataset has infinite length


# Alternative: Download dataset locally to avoid API calls entirely
def download_dataset_locally(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    cache_dir: str = "./data",
    subset_size: Optional[int] = None
):
    """
    Download dataset locally once to avoid repeated API calls
    This is the BEST solution for avoiding rate limits
    """
    print(f"Downloading {dataset_name} to {cache_dir}...")

    # Download with a subset for testing
    if subset_size:
        dataset = load_dataset(
            dataset_name,
            split=f"train[:{subset_size}]",
            cache_dir=cache_dir,
            keep_in_memory=False,
        )
    else:
        dataset = load_dataset(
            dataset_name,
            split="train",
            cache_dir=cache_dir,
            keep_in_memory=False,
        )

    # Save to disk in Arrow format (efficient)
    save_path = os.path.join(cache_dir, dataset_name.replace("/", "_"))
    dataset.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}")
    print(f"You can now load it with: datasets.load_from_disk('{save_path}')")

    return save_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Download dataset locally')
    parser.add_argument('--test', action='store_true', help='Test the dataloader')
    parser.add_argument('--subset', type=int, default=None, help='Download subset size')
    args = parser.parse_args()

    if args.download:
        # Download dataset to avoid rate limits
        path = download_dataset_locally(subset_size=args.subset)
        print(f"\nDataset downloaded to: {path}")
        print("Use this path in your training script to avoid API calls")

    if args.test:
        # Test the optimized dataloader
        print("\nTesting optimized dataloader...")
        dataloader = OptimizedDataLoader(
            batch_size=4,
            max_length=512,
            verbose=True,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        )

        # Test a few batches
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
            print(f"Batch {i}: input_ids shape = {batch['input_ids'].shape}")

        print("\nDataLoader test successful!")
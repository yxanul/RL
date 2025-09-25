"""
Experimental MoE Language Model with Modern Optimizations
Architecture: ~800M parameters with 16 experts
Features: RMSNorm, GQA, SwiGLU, SDPA, RoPE, Tutel MoE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from tutel import moe as tutel_moe
from tutel import system
import logging

# --- MoE Inspector (module-level) -------------------------------------------
class _MoEInspector:
    """Tracks per-expert token routing statistics across all layers"""
    def __init__(self):
        self.enabled = False
        self.num_layers = None
        self.num_experts = None
        self._per_layer = None  # list of dicts

    def init_if_needed(self, num_layers, num_experts):
        if self.num_layers != num_layers or self.num_experts != num_experts or self._per_layer is None:
            self.num_layers, self.num_experts = num_layers, num_experts
            self._per_layer = [
                {
                    "tokens": torch.zeros(num_experts, dtype=torch.long),
                    "capacity": torch.zeros(num_experts, dtype=torch.long),
                    "dropped": torch.zeros(num_experts, dtype=torch.long),
                }
                for _ in range(num_layers)
            ]

    def enable(self, flag: bool):
        self.enabled = bool(flag)

    @torch.no_grad()
    def record(self, layer_idx: int, counts: torch.Tensor, capacity: torch.Tensor = None, dropped: torch.Tensor = None):
        if not self.enabled or self._per_layer is None:
            return
        d = self._per_layer[layer_idx]
        d["tokens"][:counts.numel()] += counts.cpu()
        if capacity is not None:
            d["capacity"][:capacity.numel()] += capacity.cpu()
        if dropped is not None:
            d["dropped"][:dropped.numel()] += dropped.cpu()

    def drain(self, reset=True):
        payload = self._per_layer
        if reset and payload is not None:
            self._per_layer = [
                {k: v.clone().zero_() for k, v in layer.items()}
                for layer in payload
            ]
        return payload

_MOE_INSPECTOR = _MoEInspector()

def moe_stats_enable(flag: bool):
    """Enable or disable MoE statistics collection"""
    _MOE_INSPECTOR.enable(flag)

def moe_stats_drain(reset=True):
    """Get collected statistics and optionally reset"""
    return _MOE_INSPECTOR.drain(reset)

# Model Configuration
class MoEConfig:
    """Configuration for our experimental MoE model"""
    # Model dimensions
    vocab_size: int = 32768  # Mistral-7B-Instruct-v0.3 tokenizer size
    hidden_dim: int = 768   # Model dimension (reduced for <1B params)
    num_layers: int = 12    # Number of transformer layers

    # Attention configuration (GQA)
    num_attention_heads: int = 12  # Query heads
    num_key_value_heads: int = 3   # KV heads (4:1 ratio for GQA)
    head_dim: int = 64  # Dimension per head

    # MoE configuration
    num_experts: int = 16  # Total experts
    num_experts_per_token: int = 2  # Top-k routing
    expert_hidden_dim: int = 1536  # Expert FFN dimension (2x model dim)

    # Training configuration
    max_seq_length: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout_rate: float = 0.1

    # Capacity factor for load balancing
    capacity_factor: float = 1.25

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def total_params_estimate(self):
        """Rough parameter count estimate"""
        # Embeddings
        embed_params = self.vocab_size * self.hidden_dim * 2  # input + output

        # Attention layers (with GQA)
        qkv_params = self.num_layers * self.hidden_dim * (
            self.num_attention_heads * self.head_dim +  # Q
            2 * self.num_key_value_heads * self.head_dim  # K + V
        )
        out_proj_params = self.num_layers * self.hidden_dim * self.hidden_dim

        # MoE layers (only count top-k active)
        gate_params = self.num_layers * self.hidden_dim * self.num_experts
        expert_params = self.num_layers * self.num_experts * (
            3 * self.hidden_dim * self.expert_hidden_dim  # SwiGLU: W1, W2, W3
        )

        # Layer norms
        norm_params = self.num_layers * 2 * self.hidden_dim

        total = embed_params + qkv_params + out_proj_params + gate_params + expert_params + norm_params
        return total


class RMSNorm(nn.Module):
    """RMSNorm: Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return x * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0, device: str = 'cpu'):
    """Precompute RoPE frequency tensor for faster inference"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(positions, freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    """Apply rotary position embeddings to Q and K"""
    # q shape: [batch, seq_len, num_heads, head_dim]
    # k shape: [batch, seq_len, num_kv_heads, head_dim]
    # freqs shape: [seq_len, head_dim//2]

    batch_size, seq_len, num_q_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape

    # Reshape q and k to separate pairs of dimensions
    q_reshape = q.reshape(batch_size, seq_len, num_q_heads, head_dim // 2, 2)
    k_reshape = k.reshape(batch_size, seq_len, num_kv_heads, head_dim // 2, 2)

    # Create rotation matrices
    # Expand freqs for batch and heads (handle different num heads for q and k)
    freqs_cos = freqs_cos[:seq_len, :head_dim // 2].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
    freqs_sin = freqs_sin[:seq_len, :head_dim // 2].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]

    # Apply rotation using complex number formula
    # (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
    q_rot = torch.stack([
        q_reshape[..., 0] * freqs_cos - q_reshape[..., 1] * freqs_sin,
        q_reshape[..., 0] * freqs_sin + q_reshape[..., 1] * freqs_cos
    ], dim=-1)

    k_rot = torch.stack([
        k_reshape[..., 0] * freqs_cos - k_reshape[..., 1] * freqs_sin,
        k_reshape[..., 0] * freqs_sin + k_reshape[..., 1] * freqs_cos
    ], dim=-1)

    # Reshape back
    q_embed = q_rot.flatten(-2)
    k_embed = k_rot.flatten(-2)

    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with single GEMM optimization"""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_heads_per_kv = self.num_q_heads // self.num_kv_heads

        # Single GEMM for QKV projection
        self.qkv_proj = nn.Linear(
            config.hidden_dim,
            (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False
        )

        # Output projection
        self.out_proj = nn.Linear(
            self.num_q_heads * self.head_dim,
            config.hidden_dim,
            bias=False
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)

        # Precompute RoPE frequencies
        self.register_buffer(
            "rope_cos",
            precompute_rope_freqs(self.head_dim, config.max_seq_length, config.rope_theta)[0],
            persistent=False
        )
        self.register_buffer(
            "rope_sin",
            precompute_rope_freqs(self.head_dim, config.max_seq_length, config.rope_theta)[1],
            persistent=False
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Single GEMM for QKV
        qkv = self.qkv_proj(x)

        # Split into Q, K, V
        q_size = self.num_q_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE to Q and K tensors (before expanding K for GQA)
        q, k = apply_rotary_emb(q, k, self.rope_cos, self.rope_sin)

        # Expand K and V for GQA (repeat for each Q head group) after RoPE
        k = k.repeat_interleave(self.q_heads_per_kv, dim=2)
        v = v.repeat_interleave(self.q_heads_per_kv, dim=2)

        # Transpose for attention: [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch's native SDPA (Scaled Dot-Product Attention)
        # This automatically uses Flash Attention when available
        # Note: We use is_causal=True for autoregressive LM, no need for attention_mask
        # The attention_mask from dataloader is for padding, which we don't use (drop_last=True)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # Don't pass mask - we use is_causal=True instead
            dropout_p=self.config.dropout_rate if self.training else 0.0,
            is_causal=True  # Causal mask for autoregressive LM
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)

        return self.dropout(output)


class SwiGLUExpert(nn.Module):
    """Custom SwiGLU expert for Tutel MoE with token tracking"""
    def __init__(self, **custom_options):
        super().__init__()
        # Parse custom options from Tutel
        for key in custom_options:
            if key in ('model_dim', 'num_experts_per_device', 'sharded_count'):
                setattr(self, key, custom_options[key])
            elif key == 'hidden_size_per_expert':
                self.hidden_dim = custom_options[key]
            elif key == 'layer_idx':
                self.layer_idx = int(custom_options[key])
            else:
                # Log but don't fail on unknown options
                pass

        # SwiGLU uses 3 matrices: gate, up, and down
        # Combined upward projection for gate and value
        self.W_gate_up = nn.Parameter(
            torch.empty(self.num_experts_per_device, self.model_dim, 2 * self.hidden_dim)
        )
        # Downward projection
        self.W_down = nn.Parameter(
            torch.empty(self.num_experts_per_device, self.hidden_dim, self.model_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights"""
        with torch.no_grad():
            # Small initialization for stability
            nn.init.normal_(self.W_gate_up, mean=0.0, std=0.02)
            nn.init.normal_(self.W_down, mean=0.0, std=0.02)

    def forward(self, x, ctx):
        """Forward pass for SwiGLU expert with token tracking"""
        # --- Track tokens actually processed per expert ----------------------
        if _MOE_INSPECTOR.enabled:
            # Debug: Print ctx attributes on first call
            layer_idx = getattr(self, 'layer_idx', 0)
            if layer_idx == 0 and not hasattr(self, '_debug_printed'):
                self._debug_printed = True
                print(f"\n[DEBUG] Tutel ctx attributes: {list(vars(ctx).keys())}")
                for attr in vars(ctx):
                    val = getattr(ctx, attr)
                    if torch.is_tensor(val):
                        print(f"  {attr}: tensor with shape {val.shape}, dtype {val.dtype}")
                    else:
                        print(f"  {attr}: {type(val).__name__} = {val}")

            # Prefer ctx-provided counts if available (names vary by Tutel version)
            counts = None
            cap = getattr(ctx, 'expert_capacity', None)
            drop = getattr(ctx, 'dropped_per_expert', None)

            # Check more possible field names
            for candidate in ('tokens_per_expert', 'exp_counts', 'expert_tokens', 'counts',
                            'expert_count', 'indices', 'locations', 'bins'):
                if hasattr(ctx, candidate):
                    val = getattr(ctx, candidate)
                    if torch.is_tensor(val):
                        counts = val
                        break

            if counts is None:
                # Fallback: count non-padding tokens
                # x: [E, T, D] -> valid token if row has any nonzero
                # But be careful - sometimes x is already normalized/small values
                valid = (x.abs().sum(dim=-1) > 1e-10)
                counts = valid.sum(dim=1)  # [E]

            _MOE_INSPECTOR.record(layer_idx, counts, cap, drop)
        # ------------------------------------------------------------------------

        if ctx.sharded_count > 1:
            raise NotImplementedError("Sharding not supported in this expert")

        # x shape: [num_experts, tokens_per_expert, model_dim]
        # Apply gate and up projection
        gate_up = torch.matmul(x, self.W_gate_up)  # [experts, tokens, 2*hidden]
        gate, up = gate_up.chunk(2, dim=-1)  # Each: [experts, tokens, hidden]

        # SwiGLU activation
        hidden = F.silu(gate) * up  # [experts, tokens, hidden]

        # Down projection
        output = torch.matmul(hidden, self.W_down)  # [experts, tokens, model_dim]

        return output


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE FFN"""
    def __init__(self, config: MoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Pre-normalization layers
        self.attn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

        # Grouped Query Attention
        self.attention = GroupedQueryAttention(config)

        # MoE layer with custom SwiGLU experts
        self.moe_layer = tutel_moe.moe_layer(
            gate_type={
                'type': 'top',
                'k': config.num_experts_per_token,
                'capacity_factor': config.capacity_factor,
                'fp32_gate': True  # Use FP32 for gating stability
            },
            experts={
                'type': 'custom',
                'module': SwiGLUExpert,
                'num_experts_per_device': config.num_experts,
                'hidden_size_per_expert': config.expert_hidden_dim,
                'model_dim': config.hidden_dim,
                'layer_idx': layer_idx,  # Pass layer index for tracking
            },
            model_dim=config.hidden_dim,
            scan_expert_func=lambda name, param: setattr(param, 'skip_allreduce', True),
            seeds=(42, layer_idx, 42),  # Deterministic initialization per layer
        )

        # Dropout for residual connections
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize MoE inspector
        _MOE_INSPECTOR.init_if_needed(config.num_layers, config.num_experts)

        # Attach gate hook for pre-capacity statistics
        self._attach_gate_hook(layer_idx)

    def _attach_gate_hook(self, layer_idx):
        """Attach hook to gate module for pre-capacity routing statistics"""
        # Try to find the internal gate submodule
        gate_mod = None
        for name, mod in self.moe_layer.named_modules():
            cname = mod.__class__.__name__.lower()
            if 'gate' in name.lower() or cname.endswith('gate'):
                gate_mod = mod
                break

        if gate_mod is None:
            return  # Gate module not found in this Tutel version

        def gate_hook(mod, inputs, output):
            """Hook to capture gate outputs (indices/scores)"""
            if not _MOE_INSPECTOR.enabled:
                return

            # Different Tutel versions return different outputs
            tensors = []
            if isinstance(output, (tuple, list)):
                tensors = [t for t in output if torch.is_tensor(t)]
            elif torch.is_tensor(output):
                tensors = [output]

            # Look for [tokens, k] integer tensor = expert indices
            for t in tensors:
                if t.dtype in (torch.int32, torch.int64) and t.dim() == 2:
                    idx = t  # [tokens, k]
                    num_exp = _MOE_INSPECTOR.num_experts or idx.max().item() + 1
                    # Count pre-capacity routing decisions
                    counts = torch.bincount(idx.reshape(-1), minlength=num_exp)
                    # Store as "capacity" to distinguish from post-capacity "tokens"
                    _MOE_INSPECTOR.record(layer_idx, torch.zeros_like(counts), capacity=counts)
                    break

        gate_mod.register_forward_hook(gate_hook)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with pre-norm
        # Note: attention_mask is ignored since we use causal masking
        attn_input = self.attn_norm(x)
        attn_output = self.attention(attn_input, None)  # Don't pass mask to attention
        x = x + self.dropout(attn_output)

        # MoE FFN with pre-norm
        ffn_input = self.ffn_norm(x)
        ffn_output = self.moe_layer(ffn_input)
        x = x + self.dropout(ffn_output)

        # Return both output and auxiliary loss from MoE
        # Ensure aux_loss is always a tensor on the same device
        if hasattr(self.moe_layer, 'l_aux'):
            aux_loss = self.moe_layer.l_aux
        else:
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return x, aux_loss


class ExperimentalMoEModel(nn.Module):
    """Complete MoE Language Model"""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer blocks with MoE
        self.layers = nn.ModuleList([
            MoETransformerBlock(config, i) for i in range(config.num_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

        # Language model head (share weights with embeddings)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Print model statistics
        self._print_model_stats()

    def _init_weights(self, module):
        """Initialize weights with scaled normal distribution"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _print_model_stats(self):
        """Print model parameter statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count MoE expert params separately
        expert_params = 0
        for layer in self.layers:
            if hasattr(layer.moe_layer, 'get_parameter_iterator'):
                expert_params += sum(
                    p.numel() for _, p in
                    layer.moe_layer.get_parameter_iterator(param_type='local_experts')
                )

        print(f"\n{'='*60}")
        print(f"Model Statistics:")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
        print(f"Expert parameters: {expert_params / 1e6:.2f}M")
        print(f"Active params per token: {(total_params - expert_params + expert_params * self.config.num_experts_per_token / self.config.num_experts) / 1e6:.2f}M")
        print(f"{'='*60}\n")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Process through transformer blocks
        # Initialize aux loss as tensor to avoid runtime errors
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, aux_loss = layer(x, attention_mask)
            total_aux_loss = total_aux_loss + aux_loss

        # Final normalization
        x = self.final_norm(x)

        # Language model logits
        logits = self.lm_head(x)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

            # Always add auxiliary loss for load balancing (as tensor operation)
            loss = loss + 0.01 * total_aux_loss  # Small weight for aux loss

        return {
            'loss': loss,
            'logits': logits,
            'aux_loss': total_aux_loss
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.95
    ) -> torch.Tensor:
        """Simple generation method for testing"""
        self.eval()

        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self(input_ids)
            logits = outputs['logits'][:, -1, :] / temperature

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop if EOS token
            if next_token.item() == 2:  # Mistral EOS token
                break

        return input_ids


if __name__ == "__main__":
    # Test the model
    config = MoEConfig()
    model = ExperimentalMoEModel(config)

    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids, labels=labels)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Auxiliary Loss: {outputs['aux_loss'].item():.6f}")
    print(f"Logits shape: {outputs['logits'].shape}")
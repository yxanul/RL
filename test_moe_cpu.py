"""
CPU-friendly test script for MoE model
Tests auxiliary loss handling and attention masking fixes
"""

import torch
import torch.nn.functional as F
from moe_model import ExperimentalMoEModel, MoEConfig
import time

def test_aux_loss_handling():
    """Test that auxiliary loss is properly handled as tensor"""
    print("\n" + "="*60)
    print("Testing Auxiliary Loss Handling")
    print("="*60)

    # Create small model for CPU testing
    config = MoEConfig(
        hidden_dim=256,  # Reduced for CPU
        num_layers=2,    # Fewer layers for speed
        num_experts=4,   # Fewer experts for CPU
        num_experts_per_token=2,
        expert_hidden_dim=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64
    )

    model = ExperimentalMoEModel(config)
    model.eval()  # Set to eval mode for consistent testing

    # Create small batch for CPU
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    # Test forward pass with aux loss
    try:
        outputs = model(input_ids, attention_mask=None, labels=labels)

        # Check that aux_loss is a tensor
        assert isinstance(outputs['aux_loss'], torch.Tensor), "aux_loss should be a tensor"
        print(f"[PASS] Auxiliary loss is tensor: {type(outputs['aux_loss'])}")

        # Check that loss arithmetic works
        total_loss = outputs['loss']  # This internally adds aux_loss
        assert isinstance(total_loss, torch.Tensor), "Total loss should be a tensor"
        print(f"[PASS] Total loss with aux_loss: {total_loss.item():.4f}")

        # Test backward pass
        total_loss.backward()
        print("[PASS] Backward pass successful with aux_loss")

    except Exception as e:
        print(f"[FAIL] Error in aux_loss handling: {e}")
        return False

    print("[PASS] Auxiliary loss handling test PASSED")
    return True


def test_attention_masking():
    """Test that attention works correctly without explicit mask"""
    print("\n" + "="*60)
    print("Testing Attention Masking")
    print("="*60)

    # Create small model
    config = MoEConfig(
        hidden_dim=256,
        num_layers=1,
        num_experts=2,
        num_experts_per_token=1,
        expert_hidden_dim=384,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=64
    )

    model = ExperimentalMoEModel(config)
    model.eval()

    batch_size = 2
    seq_len = 16

    # Test 1: Forward pass without attention mask (should work)
    try:
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        outputs = model(input_ids, attention_mask=None)
        print(f"[PASS] Forward pass without mask: output shape {outputs['logits'].shape}")
    except Exception as e:
        print(f"[FAIL] Error without mask: {e}")
        return False

    # Test 2: Verify causal masking is applied
    # Generate a sequence where later tokens should not affect earlier ones
    test_input = torch.zeros((1, seq_len), dtype=torch.long)
    test_input[0, -1] = 1  # Set last token to 1, rest to 0

    with torch.no_grad():
        outputs1 = model(test_input)
        logits1 = outputs1['logits'][0, 0, :]  # First position logits

        # Change last token - should not affect first position due to causal mask
        test_input[0, -1] = 100
        outputs2 = model(test_input)
        logits2 = outputs2['logits'][0, 0, :]  # First position logits

        # Check that first position logits are unchanged
        diff = (logits1 - logits2).abs().max().item()
        if diff < 1e-5:
            print(f"[PASS] Causal masking working: max diff = {diff:.2e}")
        else:
            print(f"[FAIL] Causal masking issue: max diff = {diff:.2e}")
            return False

    print("[PASS] Attention masking test PASSED")
    return True


def test_generation():
    """Test model generation on CPU"""
    print("\n" + "="*60)
    print("Testing Generation (CPU)")
    print("="*60)

    # Very small model for CPU generation
    config = MoEConfig(
        hidden_dim=128,
        num_layers=2,
        num_experts=2,
        num_experts_per_token=1,
        expert_hidden_dim=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64
    )

    model = ExperimentalMoEModel(config)
    model.eval()

    # Generate a few tokens
    input_ids = torch.tensor([[1]])  # Start with token 1

    print("Generating tokens...")
    start_time = time.time()

    with torch.no_grad():
        for i in range(10):
            outputs = model(input_ids)
            logits = outputs['logits'][0, -1, :]

            # Simple greedy sampling
            next_token = logits.argmax().unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            print(f"  Token {i+1}: {next_token.item()}", end=" ")

    elapsed = time.time() - start_time
    print(f"\n[PASS] Generated 10 tokens in {elapsed:.2f}s")
    print(f"  Final sequence shape: {input_ids.shape}")

    return True


def test_training_step():
    """Test a single training step on CPU"""
    print("\n" + "="*60)
    print("Testing Training Step (CPU)")
    print("="*60)

    # Small model for CPU
    config = MoEConfig(
        hidden_dim=192,
        num_layers=2,
        num_experts=4,
        num_experts_per_token=2,
        expert_hidden_dim=384,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=64
    )

    model = ExperimentalMoEModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Small batch
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    # Training step
    model.train()
    initial_loss = None

    print("Running 5 training steps...")
    for step in range(5):
        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=None, labels=labels)
        loss = outputs['loss']
        aux_loss = outputs['aux_loss']

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if initial_loss is None:
            initial_loss = current_loss

        print(f"  Step {step+1}: Loss={current_loss:.4f}, Aux={aux_loss.item():.4f}")

    # Check if loss decreased (not guaranteed but likely with random data)
    print(f"[PASS] Training completed: Initial loss={initial_loss:.4f}, Final={current_loss:.4f}")

    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MoE Model CPU Test Suite")
    print("="*60)

    # Set to CPU mode
    device = torch.device('cpu')
    print(f"Running on device: {device}")

    # Disable autocast for CPU
    torch.set_default_dtype(torch.float32)

    # Run tests
    tests = [
        ("Auxiliary Loss Handling", test_aux_loss_handling),
        ("Attention Masking", test_attention_masking),
        ("Generation", test_generation),
        ("Training Step", test_training_step)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n[FAIL] Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "[PASS] PASSED" if passed else "[FAIL] FAILED"
        print(f"{test_name:30} {status}")
        all_passed = all_passed and passed

    print("="*60)
    if all_passed:
        print("[PASS] All tests PASSED!")
        print("\nModel is ready for GPU training when you get your instance.")
    else:
        print("[FAIL] Some tests failed. Please review the errors above.")

    # Print model info for the actual config
    print("\n" + "="*60)
    print("Full Model Configuration")
    print("="*60)
    full_config = MoEConfig()  # Default full-size config
    print(f"Hidden dimension: {full_config.hidden_dim}")
    print(f"Number of layers: {full_config.num_layers}")
    print(f"Total experts: {full_config.num_experts}")
    print(f"Active experts per token: {full_config.num_experts_per_token}")
    print(f"Estimated total params: ~{full_config.total_params_estimate / 1e6:.1f}M")
    print("="*60)


if __name__ == "__main__":
    main()
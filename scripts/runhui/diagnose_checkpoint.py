#!/usr/bin/env python3
"""
Checkpoint Diagnostic Tool

This script checks if a checkpoint is corrupted and suggests recovery options.
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path


def check_checkpoint_files(checkpoint_dir):
    """Check if all checkpoint files exist and are readable."""
    print(f"\n{'='*60}")
    print(f"Checking checkpoint: {checkpoint_dir}")
    print(f"{'='*60}\n")

    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory does not exist: {checkpoint_dir}")
        return False

    # Check for config files
    config_file = checkpoint_path / "config.json"
    if config_file.exists():
        print(f"✓ config.json exists ({config_file.stat().st_size:,} bytes)")
        try:
            with open(config_file) as f:
                config = json.load(f)
            print(f"  Model type: {config.get('model_type', 'unknown')}")
            print(f"  Hidden size: {config.get('hidden_size', 'unknown')}")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not read config.json: {e}")
    else:
        print(f"❌ config.json missing")

    # Check for model files
    print("\nModel checkpoint files:")

    # Check for index file (sharded checkpoint)
    index_file = checkpoint_path / "pytorch_model.bin.index.json"
    if index_file.exists():
        print(f"✓ pytorch_model.bin.index.json exists (SHARDED CHECKPOINT)")
        try:
            with open(index_file) as f:
                index_data = json.load(f)

            # Get shard file names
            weight_map = index_data.get('weight_map', {})
            shard_files = sorted(set(weight_map.values()))

            print(f"  Expected shards: {len(shard_files)}")

            all_shards_exist = True
            corrupted_shards = []

            for i, shard_file in enumerate(shard_files, 1):
                shard_path = checkpoint_path / shard_file
                if shard_path.exists():
                    size_mb = shard_path.stat().st_size / (1024**2)
                    print(f"  ✓ Shard {i}/{len(shard_files)}: {shard_file} ({size_mb:.2f} MB)")

                    # Try to load shard to check if it's corrupted
                    try:
                        with torch.no_grad():
                            shard_data = torch.load(shard_path, map_location='cpu')
                            num_tensors = len(shard_data)
                            print(f"    Contains {num_tensors} tensors")
                            del shard_data  # Free memory
                    except Exception as e:
                        print(f"    ❌ CORRUPTED: Cannot load shard: {e}")
                        corrupted_shards.append(shard_file)
                        all_shards_exist = False
                else:
                    print(f"  ❌ Shard {i}/{len(shard_files)}: {shard_file} MISSING")
                    all_shards_exist = False

            if corrupted_shards:
                print(f"\n❌ CHECKPOINT IS CORRUPTED")
                print(f"   Corrupted shards: {len(corrupted_shards)}")
                for shard in corrupted_shards:
                    print(f"   - {shard}")
                return False

            if not all_shards_exist:
                print(f"\n❌ CHECKPOINT IS INCOMPLETE")
                print(f"   Some shard files are missing")
                return False

            print(f"\n✓ All shards exist and are loadable")

        except Exception as e:
            print(f"  ❌ Error reading index file: {e}")
            return False

    # Check for single model file
    model_file = checkpoint_path / "pytorch_model.bin"
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024**2)
        print(f"✓ pytorch_model.bin exists ({size_mb:.2f} MB)")

        # Try to load it
        try:
            print("  Attempting to load checkpoint...")
            with torch.no_grad():
                state_dict = torch.load(model_file, map_location='cpu')
                num_params = len(state_dict)
                print(f"  ✓ Successfully loaded {num_params} parameters")
                del state_dict
        except Exception as e:
            print(f"  ❌ CORRUPTED: Cannot load checkpoint: {e}")
            return False

    if not index_file.exists() and not model_file.exists():
        print("❌ No model checkpoint files found")
        return False

    return True


def suggest_recovery(checkpoint_dir):
    """Suggest recovery options based on checkpoint status."""
    print(f"\n{'='*60}")
    print("Recovery Options")
    print(f"{'='*60}\n")

    checkpoint_path = Path(checkpoint_dir)
    parent_dir = checkpoint_path.parent

    # Look for other checkpoints in the same directory
    print("Looking for other checkpoints in the same output directory...")
    checkpoints = sorted(parent_dir.glob("checkpoint-*"))

    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoint(s):\n")
        for cp in checkpoints:
            is_current = cp == checkpoint_path
            marker = "← CURRENT (CORRUPTED)" if is_current else ""
            print(f"  {'✗' if is_current else '✓'} {cp.name} {marker}")

        print("\nRECOMMENDED ACTIONS:")
        print("1. Use an earlier checkpoint that's not corrupted")
        print("   Example:")

        # Find the most recent checkpoint that's not the current one
        for cp in reversed(checkpoints):
            if cp != checkpoint_path:
                print(f"   --model-name {cp}")
                break

        print("\n2. Or re-run training from an earlier checkpoint:")
        print("   - Resume from the last good checkpoint")
        print("   - Use smaller batch size to prevent OOM during saving")
    else:
        print("No other checkpoints found in this directory.\n")
        print("RECOMMENDED ACTIONS:")
        print("1. Re-run training from scratch")
        print("2. Use batch_size=1 and gradient_accumulation to compensate")
        print("3. Enable better memory management during checkpoint saving")

    print("\nPREVENTION FOR FUTURE TRAINING:")
    print("1. Set save_total_limit=3 to keep multiple checkpoints")
    print("2. Use batch_size=1 with higher gradient_accumulation")
    print("   (batch_size=2 caused OOM during checkpoint save)")
    print("3. Add GPU cache clearing before saves (already in llava_trainer.py)")
    print("4. Monitor GPU memory during training with nvidia-smi")


def test_loading(checkpoint_dir):
    """Attempt to load the checkpoint and report detailed errors."""
    print(f"\n{'='*60}")
    print("Testing Checkpoint Loading")
    print(f"{'='*60}\n")

    print("Attempting to load with different methods...\n")

    # Method 1: Standard loading
    print("Method 1: Standard loading with low_cpu_mem_usage=True")
    try:
        from llava import LlavaLlamaForCausalLM

        model = LlavaLlamaForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map='cpu'  # Load to CPU to avoid GPU OOM
        )
        print("✓ SUCCESS: Model loaded successfully!\n")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        del model
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}\n")

    # Method 2: device_map='auto'
    print("Method 2: Loading with device_map='auto'")
    try:
        from llava import LlavaLlamaForCausalLM

        model = LlavaLlamaForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.float16,
            device_map='auto',
            low_cpu_mem_usage=True
        )
        print("✓ SUCCESS: Model loaded successfully!\n")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        del model
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}\n")

    # Method 3: Load state dict directly
    print("Method 3: Loading state_dict directly")
    try:
        checkpoint_path = Path(checkpoint_dir)
        model_file = checkpoint_path / "pytorch_model.bin"

        if model_file.exists():
            state_dict = torch.load(model_file, map_location='cpu')
            print(f"✓ SUCCESS: State dict loaded with {len(state_dict)} parameters\n")
            del state_dict
            return True
        else:
            print("✗ No single pytorch_model.bin file found\n")

    except Exception as e:
        print(f"✗ FAILED: {e}\n")

    print("❌ All loading methods failed - checkpoint is likely corrupted")
    return False


def main():
    parser = argparse.ArgumentParser(description="Diagnose checkpoint corruption")
    parser.add_argument("checkpoint_dir", type=str, help="Path to checkpoint directory")
    parser.add_argument("--test-load", action="store_true", help="Attempt to load the checkpoint")
    args = parser.parse_args()

    # Check files
    files_ok = check_checkpoint_files(args.checkpoint_dir)

    # Test loading if requested
    if args.test_load:
        load_ok = test_loading(args.checkpoint_dir)
    else:
        load_ok = None

    # Suggest recovery
    if not files_ok or (load_ok is False):
        suggest_recovery(args.checkpoint_dir)
        sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print("✓ Checkpoint appears to be valid")
        print(f"{'='*60}\n")
        if load_ok is None:
            print("Run with --test-load to verify loading works")
        sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple test to verify Mirage MPK works
Based on the original qwen3_mirage.py record_cuda_graph method
"""

import os
import sys
import torch

# Set MIRAGE_HOME
MIRAGE_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.environ["MIRAGE_HOME"] = MIRAGE_HOME

from qwen3_mirage_example import MirageQwen3Model


def simple_test():
    print("=" * 80)
    print("Simple Mirage MPK Test")
    print("=" * 80)

    # Initialize model
    print("\n[1] Initializing model...")
    model = MirageQwen3Model(
        max_num_batched_tokens=8,
        max_num_batched_requests=4,
        page_size=4096,
        max_num_pages=16,
        max_seq_length=512,
        out_put_dir=".",
    )
    print("✓ Model initialized")

    # Create prompts
    print("\n[2] Creating prompts...")
    prompt = "What is artificial intelligence?"
    model.create_prompts(prompt)
    print(f"✓ Prompt created: {model.prompt_lengths[0].item()} tokens")

    # Create kernel
    print("\n[3] Creating Mirage kernel...")
    model.create_mirage_kernel()
    print("✓ Kernel created")

    # Build MPK
    print("\n[4] Building MPK...")
    model.build_mpk()
    print("✓ MPK built and compiled")

    # Run inference (original record_cuda_graph logic)
    print("\n[5] Running inference...")
    print("  Note: Mirage MPK generates all tokens in a single call")

    model.starter.record()
    model.mpk()
    model.ender.record()
    torch.cuda.synchronize()

    run_time = model.starter.elapsed_time(model.ender)

    print(f"✓ Inference completed in {run_time:.2f} ms")

    # Decode and print results
    print("\n[6] Decoding results...")
    print("-" * 80)

    for r in range(model.total_num_requests):
        generated_ids = model.tokens[r, : model.step[r].item() + 1]
        response = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Request {r}:")
        print(response)
        print()

    print("-" * 80)

    # Summary
    prompt_len = model.prompt_lengths[0].item()
    gen_len = model.step[0].item() + 1 - prompt_len
    per_token_latency = run_time / (model.step[0].item() + 1)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Prompt length: {prompt_len} tokens")
    print(f"Generated length: {gen_len} tokens")
    print(f"Total tokens: {model.step[0].item() + 1}")
    print(f"Total time: {run_time:.2f} ms")
    print(f"Per-token latency: {per_token_latency:.2f} ms/token")
    print(f"Throughput: {1000.0 / per_token_latency:.2f} tokens/sec")
    print("=" * 80)


if __name__ == "__main__":
    try:
        simple_test()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

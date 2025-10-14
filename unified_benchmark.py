#!/usr/bin/env python3
"""
统一的 Mirage MPK vs vLLM 性能对比脚本
可以同时测试两个 backend 或单独测试其中一个
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional
import torch

# 设置 MIRAGE_HOME
MIRAGE_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.environ["MIRAGE_HOME"] = MIRAGE_HOME


class BenchmarkResult:
    """存储 benchmark 结果"""
    def __init__(self, backend: str):
        self.backend = backend
        self.load_time = 0.0
        self.inference_time = 0.0
        self.total_tokens = 0
        self.num_prompts = 0
        self.prompts_tested = []
        self.generated_texts = []
        self.per_prompt_times = []

    @property
    def throughput(self) -> float:
        """Tokens per second"""
        return self.total_tokens / self.inference_time if self.inference_time > 0 else 0

    @property
    def avg_latency(self) -> float:
        """Average time per prompt (seconds)"""
        return self.inference_time / self.num_prompts if self.num_prompts > 0 else 0

    @property
    def avg_tokens_per_prompt(self) -> float:
        """Average tokens generated per prompt"""
        return self.total_tokens / self.num_prompts if self.num_prompts > 0 else 0


def test_mirage_mpk(prompts: List[str], max_tokens: int = 100) -> BenchmarkResult:
    """测试 Mirage MPK backend (支持多个 prompts)"""
    print("\n" + "="*80)
    print("Testing Mirage MPK Backend")
    print("="*80 + "\n")

    result = BenchmarkResult("Mirage MPK")

    try:
        from qwen3_mirage_example import MirageQwen3Model
    except ImportError as e:
        print(f"✗ Failed to import MirageQwen3Model: {e}")
        return result

    # 1. Load model
    print("[1/4] Loading model...")
    start_time = time.time()

    model = MirageQwen3Model(
        max_num_batched_tokens=8,
        max_num_batched_requests=max(4, len(prompts)),  # 根据 prompt 数量动态调整
        page_size=4096,
        max_num_pages=16,
        max_seq_length=512,
        out_put_dir=".",
    )

    result.load_time = time.time() - start_time
    print(f"✓ Model loaded in {result.load_time:.2f}s\n")

    # 2. Initialize kernel
    print("[2/4] Initializing Mirage kernel...")
    kernel_start = time.time()

    model.create_prompts(prompts)  # 支持多个 prompts
    model.create_mirage_kernel()
    model.build_mpk()

    kernel_time = time.time() - kernel_start
    print(f"✓ Kernel initialized and compiled in {kernel_time:.2f}s\n")

    # 3. Run inference
    print("[3/4] Running inference...")
    inference_start = time.time()

    model.starter.record()
    model.mpk()
    model.ender.record()
    torch.cuda.synchronize()

    result.inference_time = time.time() - inference_start
    print(f"✓ Inference completed in {result.inference_time*1000:.2f} ms\n")

    # 4. Decode results
    print("[4/4] Decoding results...")
    all_generated = []
    total_generated_tokens = 0

    for r in range(len(prompts)):
        generated_ids = model.tokens[r, : model.step[r].item() + 1]
        generated_text = model.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 提取生成的部分
        try:
            if "assistant" in generated_text:
                generated_part = generated_text.split("assistant", 1)[1].strip()
            else:
                generated_part = generated_text
        except:
            generated_part = generated_text

        all_generated.append(generated_part)

        # 统计 tokens
        prompt_length = model.prompt_lengths[r].item()
        total_length = model.step[r].item() + 1
        num_generated = total_length - prompt_length
        total_generated_tokens += num_generated

    result.total_tokens = total_generated_tokens
    result.num_prompts = len(prompts)
    result.prompts_tested = prompts
    result.generated_texts = all_generated

    print("-" * 80)
    for i, (prompt, generated) in enumerate(zip(prompts, all_generated)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Generated: {generated[:200]}..." if len(generated) > 200 else f"Generated: {generated}")
    print("-" * 80)

    return result


def test_vllm(prompts: List[str], max_tokens: int = 100) -> BenchmarkResult:
    """测试 vLLM backend (支持多个 prompts)"""
    print("\n" + "="*80)
    print("Testing vLLM Backend")
    print("="*80 + "\n")

    result = BenchmarkResult("vLLM")

    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        print(f"✗ Failed to import vLLM: {e}")
        print("  Please install vLLM: pip install vllm")
        return result

    # 1. Load model
    print("[1/3] Loading model...")
    start_time = time.time()

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        enforce_eager=True,  # Disable CUDA graphs for fair comparison
    )

    result.load_time = time.time() - start_time
    print(f"✓ Model loaded in {result.load_time:.2f}s\n")

    # 2. Prepare prompts with chat template
    print(f"[2/3] Preparing {len(prompts)} prompts...")
    tokenizer = llm.get_tokenizer()

    formatted_prompts = []
    for prompt in prompts:
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)

    print(f"✓ {len(formatted_prompts)} prompts prepared\n")

    # 3. Run inference
    print("[3/3] Running inference...")

    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy for reproducibility
        max_tokens=max_tokens,
    )

    start_time = time.time()
    outputs = llm.generate(formatted_prompts, sampling_params)
    result.inference_time = time.time() - start_time

    # Process results
    all_generated = []
    total_tokens = 0

    for output in outputs:
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        all_generated.append(generated_text)
        total_tokens += num_tokens

    result.total_tokens = total_tokens
    result.num_prompts = len(prompts)
    result.prompts_tested = prompts
    result.generated_texts = all_generated

    print(f"✓ Inference completed in {result.inference_time*1000:.2f} ms\n")
    print("-" * 80)
    for i, (prompt, generated) in enumerate(zip(prompts, all_generated)):
        num_tokens = len(outputs[i].outputs[0].token_ids)
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Generated ({num_tokens} tokens): {generated[:200]}..." if len(generated) > 200 else f"Generated: {generated}")
    print("-" * 80)

    return result


def print_comparison(mirage_result: Optional[BenchmarkResult],
                    vllm_result: Optional[BenchmarkResult]):
    """打印性能对比表格"""

    print("\n" + "="*100)
    print(" " * 35 + "PERFORMANCE COMPARISON")
    print("="*100)

    if not mirage_result and not vllm_result:
        print("No results to compare!")
        return

    # 表头
    print(f"{'Metric':<40} {'Mirage MPK':<25} {'vLLM':<25} {'Winner':<10}")
    print("-"*100)

    def format_val(val, unit=""):
        if val is None or val == 0:
            return "N/A"
        return f"{val:.2f}{unit}"

    # 如果只有一个结果
    if not mirage_result:
        mirage_result = BenchmarkResult("Mirage MPK")
    if not vllm_result:
        vllm_result = BenchmarkResult("vLLM")

    # Model loading time
    m_load = format_val(mirage_result.load_time, "s")
    v_load = format_val(vllm_result.load_time, "s")
    winner_load = ""
    if mirage_result.load_time > 0 and vllm_result.load_time > 0:
        winner_load = "Mirage" if mirage_result.load_time < vllm_result.load_time else "vLLM"
    print(f"{'Model Loading Time':<40} {m_load:<25} {v_load:<25} {winner_load:<10}")

    # Number of prompts
    m_prompts = mirage_result.num_prompts if mirage_result.num_prompts > 0 else "N/A"
    v_prompts = vllm_result.num_prompts if vllm_result.num_prompts > 0 else "N/A"
    print(f"{'Number of Prompts':<40} {str(m_prompts):<25} {str(v_prompts):<25} {'=':<10}")

    # Total tokens
    m_tokens = mirage_result.total_tokens if mirage_result.total_tokens > 0 else "N/A"
    v_tokens = vllm_result.total_tokens if vllm_result.total_tokens > 0 else "N/A"
    print(f"{'Total Tokens Generated':<40} {str(m_tokens):<25} {str(v_tokens):<25} {'=':<10}")

    # Inference time
    m_time = format_val(mirage_result.inference_time, "s")
    v_time = format_val(vllm_result.inference_time, "s")
    winner_time = ""
    if mirage_result.inference_time > 0 and vllm_result.inference_time > 0:
        winner_time = "Mirage" if mirage_result.inference_time < vllm_result.inference_time else "vLLM"
    print(f"{'Total Inference Time':<40} {m_time:<25} {v_time:<25} {winner_time:<10}")

    # Throughput
    m_thr = format_val(mirage_result.throughput, " tok/s")
    v_thr = format_val(vllm_result.throughput, " tok/s")
    winner_thr = ""
    if mirage_result.throughput > 0 and vllm_result.throughput > 0:
        winner_thr = "Mirage" if mirage_result.throughput > vllm_result.throughput else "vLLM"
    print(f"{'Throughput':<40} {m_thr:<25} {v_thr:<25} {winner_thr:<10}")

    # Average latency per prompt
    m_lat = format_val(mirage_result.avg_latency, "s")
    v_lat = format_val(vllm_result.avg_latency, "s")
    winner_lat = ""
    if mirage_result.avg_latency > 0 and vllm_result.avg_latency > 0:
        winner_lat = "Mirage" if mirage_result.avg_latency < vllm_result.avg_latency else "vLLM"
    print(f"{'Avg Latency per Prompt':<40} {m_lat:<25} {v_lat:<25} {winner_lat:<10}")

    # Average tokens per prompt
    m_avg_tok = format_val(mirage_result.avg_tokens_per_prompt, " tok")
    v_avg_tok = format_val(vllm_result.avg_tokens_per_prompt, " tok")
    print(f"{'Avg Tokens per Prompt':<40} {m_avg_tok:<25} {v_avg_tok:<25} {'=':<10}")

    print("-"*100)

    # Speedup analysis
    if mirage_result.throughput > 0 and vllm_result.throughput > 0:
        print(f"\n{'SPEEDUP ANALYSIS':<40}")
        print("-"*100)

        speedup = mirage_result.throughput / vllm_result.throughput
        if speedup > 1:
            print(f"{'Mirage is FASTER by':<40} {speedup:.2f}x")
        else:
            print(f"{'vLLM is FASTER by':<40} {1.0/speedup:.2f}x")

        if mirage_result.avg_latency > 0 and vllm_result.avg_latency > 0:
            latency_speedup = vllm_result.avg_latency / mirage_result.avg_latency
            if latency_speedup > 1:
                print(f"{'Mirage has LOWER latency by':<40} {latency_speedup:.2f}x")
            else:
                print(f"{'vLLM has LOWER latency by':<40} {1.0/latency_speedup:.2f}x")

    print("="*100)


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark: Mirage MPK vs vLLM for Qwen3-0.6B"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["mirage", "vllm", "both"],
        default="both",
        help="Which backend to test (default: both)"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["What is artificial intelligence?"],
        help="Prompts to test (space-separated)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )

    args = parser.parse_args()

    print("\n" + "="*100)
    print(" " * 30 + "UNIFIED BENCHMARK: Qwen3-0.6B")
    print("="*100)
    print(f"Backend: {args.backend}")
    print(f"Number of prompts: {len(args.prompts)}")
    print(f"Prompts: {args.prompts}")
    print(f"Max tokens: {args.max_tokens}")
    print("="*100)

    mirage_result = None
    vllm_result = None

    # Test Mirage
    if args.backend in ["mirage", "both"]:
        try:
            mirage_result = test_mirage_mpk(args.prompts, args.max_tokens)
        except Exception as e:
            print(f"\n✗ Mirage test failed: {e}")
            import traceback
            traceback.print_exc()

    # Test vLLM
    if args.backend in ["vllm", "both"]:
        # 如果测试了 Mirage，清理 GPU 内存
        if mirage_result:
            print("\nCleaning GPU memory before vLLM test...")
            torch.cuda.empty_cache()
            time.sleep(2)

        try:
            vllm_result = test_vllm(args.prompts, args.max_tokens)
        except Exception as e:
            print(f"\n✗ vLLM test failed: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison
    print_comparison(mirage_result, vllm_result)

    # Save results to JSON
    try:
        import json
        results = {
            "mirage": {
                "load_time": mirage_result.load_time if mirage_result else None,
                "inference_time": mirage_result.inference_time if mirage_result else None,
                "total_tokens": mirage_result.total_tokens if mirage_result else None,
                "throughput": mirage_result.throughput if mirage_result else None,
                "avg_latency": mirage_result.avg_latency if mirage_result else None,
            } if mirage_result else None,
            "vllm": {
                "load_time": vllm_result.load_time if vllm_result else None,
                "inference_time": vllm_result.inference_time if vllm_result else None,
                "total_tokens": vllm_result.total_tokens if vllm_result else None,
                "throughput": vllm_result.throughput if vllm_result else None,
                "avg_latency": vllm_result.avg_latency if vllm_result else None,
            } if vllm_result else None,
            "prompts": args.prompts,
            "max_tokens": args.max_tokens,
        }

        with open("unified_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to unified_benchmark_results.json")
    except Exception as e:
        print(f"\n⚠ Failed to save results: {e}")

    print("\n" + "="*100)
    print("Benchmark completed!")
    print("="*100 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

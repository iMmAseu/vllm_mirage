from models.my_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os
import mirage as mi

def grid_for_rmsnorm_linear_layer(size):
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64
    
def max_factor_leq_n(m: int, n: int) -> int:
    max_factor = 1
    i = 1
    while i * i <= m:
        if m % i == 0:
            if i <= n:
                max_factor = max(max_factor, i)
            if m // i <= n:
                max_factor = max(max_factor, m // i)
        i += 1
    return max_factor

class MirageQwen3Model:
    def __init__(
        self,
        max_num_batched_tokens: int = 8,
        max_num_batched_requests: int = 4,
        page_size: int = 4096,
        max_num_pages: int = 16,
        max_seq_length: int = 256,
        out_put_dir: str = "",
    ):
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_batched_requests = max_num_batched_requests
        self.page_size = page_size
        self.max_num_pages = max_num_pages
        self.max_seq_length = max_seq_length
        self.spec_decode = None
        self.ngram_size = 0
        self.spec_length = 0
        self.model_name = "Qwen/Qwen3-0.6B"
        self.output_dir = out_put_dir
        self.trace_name = ""

        # MPI/Distributed settings
        self.world_size = 1
        self.rank = 0

        # Total number of requests (same as max_num_batched_requests for simplicity)
        self.total_num_requests = self.max_num_batched_requests

        torch.set_default_dtype(torch.bfloat16)
        torch.cuda.set_device(self.rank)

        self.model = Qwen3ForCausalLM.from_pretrained(
            self.model_name,
            world_size=self.world_size,
            max_num_pages=self.max_num_pages,
            page_size=self.page_size
        ).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokens = torch.full((self.max_num_batched_requests, self.max_seq_length), 0, dtype=torch.long, device="cuda")

    def create_prompts(self, prompts):
        """
        Accept either a single prompt (str) or multiple prompts (list of str).
        Each prompt will be processed independently.
        """
        # Handle single prompt case
        if isinstance(prompts, str):
            prompts = [prompts]

        # Limit to max_num_batched_requests
        if len(prompts) > self.max_num_batched_requests:
            print(f"Warning: {len(prompts)} prompts provided, but max_num_batched_requests={self.max_num_batched_requests}. Using first {self.max_num_batched_requests} prompts.")
            prompts = prompts[:self.max_num_batched_requests]

        # Prepare all prompts with chat template
        texts = []
        for prompt in prompts:
            messages = [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

        # Tokenize all prompts at once
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)

        # Initialize prompt_lengths tensor
        self.prompt_lengths = torch.zeros((self.total_num_requests,), dtype=torch.int, device="cuda")

        # Fill tokens for each request
        for r in range(len(prompts)):
            # Get actual length (excluding padding)
            actual_length = (model_inputs.input_ids[r] != self.tokenizer.pad_token_id).sum().item()
            self.prompt_lengths[r] = actual_length

            # Copy tokens
            for i in range(model_inputs.input_ids.shape[-1]):
                self.tokens[r, i] = model_inputs.input_ids[r, i]

        # For remaining slots (if prompts < max_num_batched_requests), replicate last prompt
        if len(prompts) < self.total_num_requests:
            last_prompt_length = self.prompt_lengths[len(prompts) - 1]
            for r in range(len(prompts), self.total_num_requests):
                self.prompt_lengths[r] = last_prompt_length
                for i in range(model_inputs.input_ids.shape[-1]):
                    self.tokens[r, i] = model_inputs.input_ids[len(prompts) - 1, i]

    def create_mirage_kernel(self):
        positions = torch.arange(32768).unsqueeze(0).to(self.model.device)
        self.position_embeddings = self.model.model.rotary_emb(positions)
        self.input_tokens = torch.full((self.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
        self.output_tokens = torch.full((self.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
        self.prev_pos = 0

        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True)

        self.step = torch.full((self.total_num_requests, ), 0, dtype=torch.int32, device="cuda")
        self.num_new_tokens = torch.full((self.total_num_requests, ), 1, dtype=torch.int32, device="cuda")

        self.hidden_size = self.model.config.hidden_size
        self.intermediate_size = self.model.config.intermediate_size

        self.lm_head_weight = torch.cat(
            (
                self.model.lm_head.weight,
                torch.full(
                    (153600 - self.model.config.vocab_size, self.hidden_size), 0, device="cuda"
                ),
            ),
            0,
        )

        assert self.lm_head_weight.stride()[0] == self.hidden_size
        self.vocab_size = 153600
        self.num_q_heads = self.model.config.num_attention_heads
        self.num_kv_heads = self.model.config.num_key_value_heads
        self.num_local_q_heads = self.num_q_heads // self.world_size
        self.num_local_kv_heads = self.num_kv_heads // self.world_size
        self.head_dim = self.model.config.head_dim
        self.fused_outdim_1 = (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim
        self.fused_outdim_2 = 2 * self.intermediate_size

        self.profiler_tensor = None

        self.spec_decode_config = mi.speculative.spec_decode_class(
            self.spec_decode,
            ngram_size=self.ngram_size,
            spec_length=self.spec_length,
        )

        self.num_workers, self.num_schedulers = mi.get_configurations_from_gpu(self.rank)
        self.qo_indptr_buffer = torch.empty(
            self.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        self.paged_kv_indptr_buffer = torch.empty(
            self.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        self.paged_kv_indices_buffer = torch.empty(
            self.max_num_pages, dtype=torch.int32, device="cuda")
        self.paged_kv_last_page_len_buffer = torch.empty(
            self.max_num_batched_requests, dtype=torch.int32, device="cuda")
        self.mpk = mi.PersistentKernel(
            mode="offline",
            world_size=self.world_size,
            mpi_rank=self.rank,
            num_workers=self.num_workers,
            num_local_schedulers=self.num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=self.max_seq_length,
            max_num_batched_requests=self.max_num_batched_requests,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_pages=self.max_num_pages,
            page_size=self.page_size,
            eos_token_id=self.model.config.eos_token_id,
            meta_tensors={
                "step": self.step,
                "tokens": self.tokens,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "num_new_tokens": self.num_new_tokens,
                "prompt_lengths": self.prompt_lengths,
                "qo_indptr_buffer": self.qo_indptr_buffer,
                "paged_kv_indptr_buffer": self.paged_kv_indptr_buffer,
                "paged_kv_indices_buffer": self.paged_kv_indices_buffer,
                "paged_kv_last_page_len_buffer": self.paged_kv_last_page_len_buffer,
            },
            profiler_tensor=self.profiler_tensor,
            trace_name=self.trace_name,
            spec_decode_config=self.spec_decode_config,
        )

        if self.spec_decode_config and self.spec_decode_config.method == "promptlookup":
            self.all_tokens = self.mpk.attach_input(torch_tensor=self.tokens, name="all_tokens")
            self.num_tokens_extend = self.spec_decode_config.spec_length + 1
        else:
            self.num_tokens_extend = 1
        
    def build_mpk(self):
        x = self.mpk.attach_input(torch_tensor=self.input_tokens, name="input_token")
        cos_pos_embed = self.mpk.attach_input(
            torch_tensor=self.position_embeddings[0][0, :4096, :],
            name="cos_position_embedding",
        )
        sin_pos_embed = self.mpk.attach_input(
            torch_tensor=self.position_embeddings[1][0, :4096, :],
            name="sin_position_embedding",
        )
        y = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=mi.bfloat16,
            name="embed_out",
            io_category="cuda_tensor",
        )
        rmsnorm_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=mi.bfloat16,
            name="rmsnorm_out",
            io_category="cuda_tensor",
        )
        attn_in = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.fused_outdim_1 // self.world_size), # [6, 6144]
            dtype=mi.bfloat16,
            name="attn_in",
            io_category="cuda_tensor",
        )
        attn_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.num_local_q_heads * self.head_dim),
            dtype=mi.bfloat16,
            name="attn_out",
            io_category="cuda_tensor",
        )
        attn_proj_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=mi.bfloat16,
            name="attn_proj_out",
            io_category="cuda_tensor",
        )
        mlp_mid = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.fused_outdim_2 // self.world_size),
            dtype=mi.bfloat16,
            name="mlp_mid",
            io_category="cuda_tensor",
        )
        silu_mul_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.intermediate_size // self.world_size),
            dtype=mi.bfloat16,
            name="silu_mul_out",
            io_category="cuda_tensor",
        )
        mlp_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=mi.bfloat16,
            name="mlp_out",
            io_category="nvshmem_tensor" if self.world_size > 1 else "cuda_tensor",
        )
        # mlp_final = self.mpk.new_tensor(
        #     dims=(self.max_num_batched_tokens, self.hidden_size),
        #     dtype=mi.bfloat16,
        #     name="mlp_final",
        #     io_category="nvshmem_tensor" if self.world_size > 1 else "cuda_tensor",
        # )
        argmax_in = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.vocab_size),
            dtype=mi.bfloat16,
            name="argmax_in",
            io_category="cuda_tensor",
        )
        argmax_part_value = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.num_workers),
            dtype=mi.bfloat16,
            name="argmax_part_value",
            io_category="cuda_tensor",
        )
        argmax_part_index = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.num_workers),
            dtype=mi.int64,
            name="argmax_part_index",
            io_category="cuda_tensor",
        )
        argmax_out = self.mpk.attach_input(torch_tensor=self.output_tokens, name="output_token")

        # no spec_decode this case

        w = self.mpk.attach_input(
            torch_tensor=self.model.model.embed_tokens.weight, name="embed_tokens"
        )
        
        self.mpk.embed_layer(
            input=x, 
            weight=w, 
            output=y, 
            # grid_dim=(max_factor_leq_n(hidden_size, 96 // args.max_num_batched_tokens), total_tokens_per_iter, 1), 
            grid_dim=(1, 1, 1), 
            block_dim=(128, 1, 1),
            input_source=1,
        )
        
        # get embed output
        x = y
        for i, layer in enumerate(self.model.model.layers):
            # add rmsnorm + linear
            w_norm = self.mpk.attach_input(
                torch_tensor=layer.input_layernorm.weight,
                name=f"layer_{i}_input_layernorm",
            )
            w_q = self.mpk.attach_input(
                torch_tensor=layer.self_attn.q_proj.weight, name=f"layer_{i}_q_proj"
            )
            w_k = self.mpk.attach_input(
                torch_tensor=layer.self_attn.k_proj.weight, name=f"layer_{i}_k_proj"
            )
            w_v = self.mpk.attach_input(
                torch_tensor=layer.self_attn.v_proj.weight, name=f"layer_{i}_v_proj"
            )
            w_qkv = self.mpk.shuffle_tensors(
                inputs=[w_q, w_k, w_v],
                shuffled_dim=0,
                num_groups=self.model.config.num_key_value_heads // self.world_size,
                name=f"layer_{i}_qkv_proj",
            )
            self.mpk.rmsnorm_layer(
                input=x,
                weight=w_norm,
                output=rmsnorm_out,
                grid_dim=(self.mpk.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mpk.linear_layer(
                input=rmsnorm_out,
                weight=w_qkv,
                output=attn_in,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0)), 1, 1),
                block_dim=(128, 1, 1),
            )
            w_q_norm = self.mpk.attach_input(
                torch_tensor=layer.self_attn.q_norm.weight, name=f"layer_{i}_q_norm"
            )
            w_k_norm = self.mpk.attach_input(
                torch_tensor=layer.self_attn.k_norm.weight, name=f"layer_{i}_k_norm"
            )
            k_cache = self.mpk.attach_input(
                torch_tensor=self.model.model.kv_cache[0][i], name=f"layer_{i}_k_cache"
            )
            v_cache = self.mpk.attach_input(
                torch_tensor=self.model.model.kv_cache[1][i], name=f"layer_{i}_v_cache"
            )
            self.mpk.paged_attention_layer(
                    input=attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    output=attn_out,
                    grid_dim=(self.mpk.max_num_batched_requests, self.num_local_kv_heads, 1),
                    block_dim=(128, 1, 1),
                )
            w = self.mpk.attach_input(
                torch_tensor=layer.self_attn.o_proj.weight, name=f"layer_{i}_o_proj"
            )
            self.mpk.linear_with_residual_layer(
                input=attn_out,
                weight=w,
                residual=x,
                output=attn_proj_out,
                grid_dim=(self.hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            # reset residual input as x
            x = attn_proj_out
            # add rmsnorm_linear layer
            w_norm = self.mpk.attach_input(
                torch_tensor=layer.post_attention_layernorm.weight,
                name=f"layer_{i}_post_attn_layernorm",
            )
            w_gate_proj = self.mpk.attach_input(
                torch_tensor=layer.mlp.gate_proj.weight, name=f"layer_{i}_gate_proj"
            )
            w_up_proj = self.mpk.attach_input(
                torch_tensor=layer.mlp.up_proj.weight, name=f"layer_{i}_up_proj"
            )
            rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(w_gate_proj.dim(0) + w_up_proj.dim(0))
            w_gatedup = self.mpk.shuffle_tensors(
                inputs=[w_gate_proj, w_up_proj],
                shuffled_dim=0,
                num_groups=rmsnorm_num_tasks//2,
                name=f"layer_{i}_gatedup_proj",
            )
            self.mpk.rmsnorm_layer(
                input=x,
                weight=w_norm,
                output=rmsnorm_out,
                grid_dim=(self.mpk.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mpk.linear_layer(
                input=rmsnorm_out,
                weight=w_gatedup,
                output=mlp_mid,
                grid_dim=(rmsnorm_num_tasks, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mpk.silu_mul_layer(
                input=mlp_mid,
                output=silu_mul_out,
                grid_dim=(rmsnorm_num_tasks//2, 1, 1),
                block_dim=(128, 1, 1),
            )
            # add silu_mul_linear layer
            w = self.mpk.attach_input(
                torch_tensor=layer.mlp.down_proj.weight, name=f"layer_{i}_down_proj"
            )
            self.mpk.linear_with_residual_layer(
                input=silu_mul_out,
                weight=w,
                residual=x,
                output=mlp_out,
                grid_dim=(self.hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            # reset residual input as x
            x = mlp_out
        
        w_norm = self.mpk.attach_input(
            torch_tensor=self.model.model.norm.weight, name="model_norm_weight"
        )
        w_proj = self.mpk.attach_input(torch_tensor=self.lm_head_weight, name="lm_head")
        self.mpk.rmsnorm_layer(
            input=x,
            weight=w_norm,
            output=rmsnorm_out,
            grid_dim=(self.mpk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.mpk.linear_layer(
            input=rmsnorm_out,
            weight=w_proj,
            output=argmax_in,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_proj.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )
        argmax_partial_grid_dim = (self.mpk.num_workers, 1, 1)
        argmax_reduce_grid_dim = (1, 1, 1)

        self.mpk.argmax_partial_layer(
            input=argmax_in,
            output=(argmax_part_value, argmax_part_index),
            grid_dim=argmax_partial_grid_dim,
            block_dim=(128, 1, 1),
        )
        self.mpk.argmax_reduce_layer(
            input=(argmax_part_value, argmax_part_index),
            output=argmax_out,
            grid_dim=argmax_reduce_grid_dim,
            block_dim=(128, 1, 1),
        )

        results = self.mpk.kn_graph.generate_task_graph(num_gpus=self.world_size, my_gpu_id=self.rank)
        with open(f"task_graph_{self.rank}.json", "w") as f:
            f.write(results["json_file"])
        with open(f"kernel_{self.rank}.cu", "w") as f:
            f.write(results["cuda_code"])

        self.mpk.compile(output_dir=self.output_dir)
    
    def record_cuda_graph(self):
        stream = torch.cuda.Stream()
        warmup = 0
        output_len = 512

        self.starter.record()
        self.mpk()
        self.ender.record()
        torch.cuda.synchronize()
        run_time = self.starter.elapsed_time(self.ender)

        print("tokens.shape = ", self.tokens.shape)
        for r in range(self.total_num_requests):
            generated_ids = self.tokens[r, : self.step[r] + 1]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(response)

        print("Prompt length {}, generate length {}, per-token latency (both prefill and decode): {} ms".format(
              self.prompt_lengths[0], self.step[0] + 1 - self.prompt_lengths[0], run_time / (self.step[0] + 1)
            )
        )









        
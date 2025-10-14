"""Mirage MPK wrapper for Qwen3 model."""

import os
import torch
from typing import Optional, List, Tuple

import mirage as mi


def grid_for_rmsnorm_linear_layer(size):
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64
    return 1

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


class MirageQwen3Wrapper:
    """
    Wrapper class that integrates Mirage MPK with vLLM for Qwen3 inference.

    This class handles:
    - Mirage kernel initialization and compilation
    - KV-cache format adaptation between vLLM and Mirage
    - Meta tensors management
    - Forward pass execution
    """

    def __init__(
        self,
        config,
        cache_config,
        quant_config=None,
        prefix: str = "",
    ):
        self.config = config
        self.cache_config = cache_config
        self.quant_config = quant_config

        # Model parameters
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.intermediate_size = config.intermediate_size

        # Mirage MPK parameters
        self.max_num_batched_tokens = 8
        self.max_num_batched_requests = 4
        self.page_size = cache_config.block_size if cache_config else 4096
        self.max_num_pages = 16
        self.max_seq_length = 256

        # Distributed settings
        self.world_size = 1
        self.rank = 0

        # Initialize flag
        self.initialized = False
        self.mpk = None

        # Load the original Qwen3 model for weights
        from models.my_qwen3 import Qwen3ForCausalLM
        torch.set_default_dtype(torch.bfloat16)
        print(f"[Mirage] Loading Qwen3 model weights...")
        self.model = Qwen3ForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            world_size=self.world_size,
            max_num_pages=self.max_num_pages,
            page_size=self.page_size
        ).to("cuda")

        # Meta tensors
        self.step = None
        self.tokens = None
        self.input_tokens = None
        self.output_tokens = None
        self.num_new_tokens = None
        self.prompt_lengths = None

        # KV-cache buffers
        self.qo_indptr_buffer = None
        self.paged_kv_indptr_buffer = None
        self.paged_kv_indices_buffer = None
        self.paged_kv_last_page_len_buffer = None

        # Position embeddings
        self.position_embeddings = None

        # get nedded hidden results
        self.last_hidden_buffer = torch.empty(
            (self.max_num_batched_tokens, self.hidden_size),
            dtype=torch.bfloat16, device="cuda"
        )

        print(f"[Mirage] MirageQwen3Wrapper initialized")

    def _initialize_mpk(self, batch_size: int = 4):
        """Initialize Mirage Persistent Kernel."""
        if self.initialized:
            return

        print(f"[Mirage] Initializing MPK with batch_size={batch_size}...")

        # Adjust batch size
        self.max_num_batched_requests = max(4, batch_size)
        self.total_num_requests = self.max_num_batched_requests

        # Allocate meta tensors
        self.step = torch.zeros((self.max_num_batched_requests,), dtype=torch.int32, device="cuda")
        self.tokens = torch.zeros((self.max_num_batched_requests, self.max_seq_length), dtype=torch.long, device="cuda")
        self.input_tokens = torch.zeros((self.max_num_batched_tokens, 1), dtype=torch.long, device="cuda")
        self.output_tokens = torch.zeros((self.max_num_batched_tokens, 1), dtype=torch.long, device="cuda")
        self.num_new_tokens = torch.ones((self.max_num_batched_requests,), dtype=torch.int32, device="cuda")
        self.prompt_lengths = torch.zeros((self.max_num_batched_requests,), dtype=torch.int32, device="cuda")

        # Allocate KV-cache buffers
        self.qo_indptr_buffer = torch.zeros(self.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        self.paged_kv_indptr_buffer = torch.zeros(self.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        self.paged_kv_indices_buffer = torch.zeros(self.max_num_pages, dtype=torch.int32, device="cuda")
        self.paged_kv_last_page_len_buffer = torch.zeros(self.max_num_batched_requests, dtype=torch.int32, device="cuda")

        # Position embeddings
        positions = torch.arange(32768).unsqueeze(0).to("cuda")
        self.position_embeddings = self.model.model.rotary_emb(positions)

        # Prepare LM head weight (pad vocab_size to 153600)
        self.lm_head_weight = torch.cat(
            (
                self.model.lm_head.weight,
                torch.zeros(
                    (153600 - self.config.vocab_size, self.hidden_size), device="cuda"
                ),
            ),
            0,
        )
        assert self.lm_head_weight.stride()[0] == self.hidden_size
        self.padded_vocab_size = 153600

        # Get worker and scheduler configuration
        num_workers, num_schedulers = mi.get_configurations_from_gpu(self.rank)
        self.num_workers = num_workers
        self.num_schedulers = num_schedulers

        # Fused dimensions
        self.num_local_q_heads = self.num_heads // self.world_size
        self.num_local_kv_heads = self.num_kv_heads // self.world_size
        self.fused_outdim_1 = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.fused_outdim_2 = 2 * self.intermediate_size

        # Create MPK
        self.mpk = mi.PersistentKernel(
            mode="offline",
            world_size=self.world_size,
            mpi_rank=self.rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=self.max_seq_length,
            max_num_batched_requests=self.max_num_batched_requests,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_pages=self.max_num_pages,
            page_size=self.page_size,
            eos_token_id=self.config.eos_token_id,
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
            profiler_tensor=None,
            trace_name="",
            spec_decode_config=None,
        )

        print(f"[Mirage] Building MPK graph...")
        self._build_mpk_graph()

        # Check if kernel is already compiled
        # no need for out put
        # kernel_path = "/root/mirage-vllm/mirage_implements/mirage_output/kernel_0.so"
        # if os.path.exists(kernel_path):
        #     print(f"[Mirage] Pre-compiled kernel found at {kernel_path}")
        # else:
        #     print(f"[Mirage] Compiling kernel (this may take 1-2 minutes)...")
        #     output_dir = "/root/mirage-vllm/mirage_implements/mirage_output"
        #     os.makedirs(output_dir, exist_ok=True)
        #     self.mpk.compile(output_dir=output_dir)
        #     print(f"[Mirage] Kernel compiled and saved to {kernel_path}")

        self.initialized = True
        print(f"[Mirage] MPK initialization completed")

    def _build_mpk_graph(self):
        """Build the complete Mirage computation graph (based on qwen3_mirage.py)."""
        # Input tensors
        x = self.mpk.attach_input(torch_tensor=self.input_tokens, name="input_token")
        cos_pos_embed = self.mpk.attach_input(
            torch_tensor=self.position_embeddings[0][0, :4096, :],
            name="cos_position_embedding",
        )
        sin_pos_embed = self.mpk.attach_input(
            torch_tensor=self.position_embeddings[1][0, :4096, :],
            name="sin_position_embedding",
        )

        # Intermediate tensors
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
            dims=(self.max_num_batched_tokens, self.fused_outdim_1 // self.world_size),
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
            io_category="cuda_tensor",
        )
        argmax_in = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.padded_vocab_size),
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

        # Embedding layer
        w = self.mpk.attach_input(
            torch_tensor=self.model.model.embed_tokens.weight, name="embed_tokens"
        )
        self.mpk.embed_layer(
            input=x,
            weight=w,
            output=y,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
            input_source=1,
        )
        x = y

        # Transformer layers
        for i, layer in enumerate(self.model.model.layers):
            # RMSNorm + QKV Projection
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
                num_groups=self.config.num_key_value_heads // self.world_size,
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

            # Attention
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

            # Attention output projection
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
            x = attn_proj_out

            # MLP
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
            x = mlp_out

        # Final layer norm + LM head
        w_norm = self.mpk.attach_input(
            torch_tensor=self.model.model.norm.weight, name="model_norm_weight"
        )
        final_hidden_out = self.mpk.attach_input(
            torch_tensor=self.last_hidden_buffer,
            name="final_hidden_states",
        )
        self.mpk.rmsnorm_layer(
            input=x,
            weight=w_norm,
            output=final_hidden_out,
            grid_dim=(self.mpk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )

        # self.mpk.linear_layer(
        #     input=rmsnorm_out,
        #     weight=w_proj,
        #     output=argmax_in,
        #     grid_dim=(grid_for_rmsnorm_linear_layer(w_proj.dim(0)), 1, 1),
        #     block_dim=(128, 1, 1),
        # )
        # # Argmax
        # argmax_partial_grid_dim = (self.mpk.num_workers, 1, 1)
        # argmax_reduce_grid_dim = (1, 1, 1)
        # self.mpk.argmax_partial_layer(
        #     input=argmax_in,
        #     output=(argmax_part_value, argmax_part_index),
        #     grid_dim=argmax_partial_grid_dim,
        #     block_dim=(128, 1, 1),
        # )
        # self.mpk.argmax_reduce_layer(
        #     input=(argmax_part_value, argmax_part_index),
        #     output=argmax_out,
        #     grid_dim=argmax_reduce_grid_dim,
        #     block_dim=(128, 1, 1),
        # )

        # Generate task graph and kernel code
        # for intergration no need for output
        # results = self.mpk.kn_graph.generate_task_graph(num_gpus=self.world_size, my_gpu_id=self.rank)
        # output_dir = "/root/mirage-vllm/mirage_implements/mirage_output"
        # os.makedirs(output_dir, exist_ok=True)
        # with open(f"{output_dir}/task_graph_{self.rank}.json", "w") as f:
        #     f.write(results["json_file"])
        # with open(f"{output_dir}/kernel_{self.rank}.cu", "w") as f:
        #     f.write(results["cuda_code"])

        print(f"[Mirage] MPK graph built successfully")

    def _adapt_kv_cache(self, kv_caches, attn_metadata):
        """
        Adapt vLLM's KV-cache format to Mirage's paged format.

        vLLM format:
        - kv_caches: List of (key_cache, value_cache) per layer
        - attn_metadata contains block_tables, slot_mapping, etc.

        Mirage format:
        - qo_indptr_buffer: Pointer array for query/output indices
        - paged_kv_indptr_buffer: Pointer array for KV page indices
        - paged_kv_indices_buffer: Block/page indices
        - paged_kv_last_page_len_buffer: Last page length per sequence
        """

        # Reset buffers before filling them
        self.qo_indptr_buffer.zero_()
        self.paged_kv_indptr_buffer.zero_()
        self.paged_kv_indices_buffer.fill_(-1)
        self.paged_kv_last_page_len_buffer.zero_()

        if attn_metadata is None:
            return

        # Prefer decode metadata since MPK is used during decoding.
        metadata = None
        if hasattr(attn_metadata, "decode_metadata"):
            metadata = attn_metadata.decode_metadata
            if metadata is not None and metadata.num_decode_tokens > 0:
                pass
            else:
                metadata = None

        if metadata is None and hasattr(attn_metadata, "prefill_metadata"):
            metadata = attn_metadata.prefill_metadata
        if metadata is None:
            metadata = attn_metadata

        if metadata is None:
            return

        # Obtain per-sequence lengths.
        seq_lens_tensor = getattr(metadata, "seq_lens_tensor", None)
        if seq_lens_tensor is None:
            context_lens_tensor = getattr(metadata, "context_lens_tensor", None)
            if context_lens_tensor is not None:
                seq_lens_tensor = context_lens_tensor
                if getattr(metadata, "num_decode_tokens", 0) > 0:
                    seq_lens_tensor = seq_lens_tensor + 1
            elif getattr(metadata, "query_start_loc", None) is not None:
                query_locs = metadata.query_start_loc
                if query_locs.numel() > 1:
                    seq_lens_tensor = query_locs[1:] - query_locs[:-1]
        if seq_lens_tensor is None:
            raise RuntimeError(
                "Unable to derive sequence lengths from attention metadata to adapt KV cache."
            )

        seq_lens_tensor = seq_lens_tensor.to(device=self.qo_indptr_buffer.device,
                                             dtype=torch.int32)
        if seq_lens_tensor.numel() == 0:
            return

        num_reqs = min(seq_lens_tensor.shape[0], self.max_num_batched_requests)
        seq_lens = seq_lens_tensor[:num_reqs]

        # Build qo_indptr (token prefix sums) from metadata when available.
        qo_indptr_src = getattr(metadata, "query_start_loc", None)
        if qo_indptr_src is not None and qo_indptr_src.numel() >= num_reqs + 1:
            qo_indptr = qo_indptr_src[:num_reqs + 1].to(
                device=self.qo_indptr_buffer.device,
                dtype=self.qo_indptr_buffer.dtype,
            )
        else:
            ones = torch.ones_like(seq_lens, dtype=self.qo_indptr_buffer.dtype)
            qo_indptr = torch.zeros(num_reqs + 1,
                                    dtype=self.qo_indptr_buffer.dtype,
                                    device=self.qo_indptr_buffer.device)
            qo_indptr[1:] = torch.cumsum(ones, dim=0)
        self.qo_indptr_buffer[:num_reqs + 1].copy_(qo_indptr)
        if num_reqs + 1 < self.qo_indptr_buffer.shape[0]:
            self.qo_indptr_buffer[num_reqs + 1:].fill_(int(qo_indptr[-1]))

        block_tables = getattr(metadata, "block_tables", None)
        if block_tables is None or block_tables.numel() == 0:
            # Without block tables we cannot map to kv pages; nothing more to do.
            return

        block_tables = block_tables.to(device=self.paged_kv_indices_buffer.device,
                                       dtype=torch.int32)
        if block_tables.dim() != 2:
            raise RuntimeError("Expected block_tables to be 2-D.")

        block_tables = block_tables[:num_reqs]
        page_size = self.page_size
        block_table_bounds = torch.div(seq_lens + page_size - 1,
                                       page_size,
                                       rounding_mode="floor")

        max_blocks_available = block_tables.size(1)
        block_table_bounds = torch.clamp(block_table_bounds, max=max_blocks_available)

        total_pages = int(block_table_bounds.sum().item())
        if total_pages > self.max_num_pages:
            raise RuntimeError(
                f"Mirage MPK configured for {self.max_num_pages} KV pages, "
                f"but received {total_pages} pages from vLLM."
            )

        if total_pages == 0:
            return

        mask = (
            torch.arange(max_blocks_available,
                         device=block_tables.device,
                         dtype=torch.int32).unsqueeze(0)
            < block_table_bounds.unsqueeze(1)
        )
        selected_pages = torch.masked_select(block_tables, mask)
        if selected_pages.numel() != total_pages:
            raise RuntimeError("Mismatch between page bounds and selected block table entries.")

        self.paged_kv_indices_buffer[:total_pages].copy_(selected_pages)
        if total_pages < self.max_num_pages:
            self.paged_kv_indices_buffer[total_pages:].fill_(-1)

        paged_kv_indptr = torch.zeros(num_reqs + 1,
                                      dtype=torch.int32,
                                      device=self.paged_kv_indptr_buffer.device)
        paged_kv_indptr[1:] = torch.cumsum(block_table_bounds.to(torch.int32), dim=0)
        self.paged_kv_indptr_buffer[:num_reqs + 1].copy_(paged_kv_indptr)
        if num_reqs + 1 < self.paged_kv_indptr_buffer.shape[0]:
            self.paged_kv_indptr_buffer[num_reqs + 1:].fill_(int(paged_kv_indptr[-1]))

        last_page_len = seq_lens % page_size
        last_page_len = torch.where(last_page_len == 0, torch.full_like(last_page_len, page_size),
                                    last_page_len)
        last_page_len = torch.where(block_table_bounds == 0,
                                    torch.zeros_like(last_page_len),
                                    last_page_len)
        self.paged_kv_last_page_len_buffer[:num_reqs].copy_(last_page_len.to(torch.int32))
        if num_reqs < self.paged_kv_last_page_len_buffer.shape[0]:
            self.paged_kv_last_page_len_buffer[num_reqs:].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass using Mirage MPK.

        Args:
            input_ids: [batch_size, seq_len] or [num_tokens]
            positions: Token positions
            kv_caches: vLLM's KV cache blocks
            attn_metadata: Attention metadata from vLLM

        Returns:
            hidden_states: [num_tokens, hidden_size]
        """
        # Lazy initialization
        if not self.initialized:
            batch_size = input_ids.shape[0] if input_ids.dim() == 2 else 1
            self._initialize_mpk(batch_size)

        # Adapt input_ids to Mirage format
        if input_ids.dim() == 1:
            # [num_tokens] -> fill into input_tokens
            num_tokens = min(input_ids.shape[0], self.max_num_batched_tokens)
            self.input_tokens[:num_tokens, 0] = input_ids[:num_tokens]
            # Also fill tokens buffer for first request
            self.tokens[0, :num_tokens] = input_ids[:num_tokens]
            self.prompt_lengths[0] = num_tokens
        else:
            # [batch, seq_len] -> fill into tokens
            batch_size = min(input_ids.shape[0], self.max_num_batched_requests)
            seq_len = min(input_ids.shape[1], self.max_seq_length)
            self.tokens[:batch_size, :seq_len] = input_ids[:batch_size, :seq_len]
            self.prompt_lengths[:batch_size] = seq_len
            # First token of each sequence goes to input_tokens
            self.input_tokens[:batch_size, 0] = input_ids[:batch_size, 0]

        # Adapt KV-cache
        if kv_caches is not None:
            self._adapt_kv_cache(kv_caches, attn_metadata)

        # Run Mirage MPK
        self.mpk()

        # Extract outputs
        # Hidden states are materialized in self.last_hidden_buffer by the final RMSNorm layer.

        num_tokens = input_ids.shape[0] if input_ids.dim() == 1 else input_ids.numel()
        num_tokens = min(num_tokens, self.max_num_batched_tokens)

        hidden_states = self.last_hidden_buffer[:num_tokens, :].clone()
        self.result_hidden_states = hidden_states
        print(f"[Mirage] Forward pass completed, returning hidden_states shape: {self.result_hidden_states.shape}")
        return self.result_hidden_states

    def load_weights(self, weights):
        """Load weights (already loaded in __init__)."""
        pass

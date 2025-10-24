#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <vector>

#include <cuda_runtime.h>

namespace py = pybind11;

namespace {

using MirageExecFn = void (*)(std::vector<void const *>, std::vector<void *>,
                              void *, cudaStream_t, void *);

// Forward declarations for the generated kernels (renamed via macros)
extern "C" void rope_execute_mugraph_batch1(std::vector<void const *>,
                                            std::vector<void *>, void *,
                                            cudaStream_t, void *);
extern "C" void rope_execute_mugraph_batch2(std::vector<void const *>,
                                            std::vector<void *>, void *,
                                            cudaStream_t, void *);
extern "C" void rope_execute_mugraph_batch4(std::vector<void const *>,
                                            std::vector<void *>, void *,
                                            cudaStream_t, void *);
extern "C" void rope_execute_mugraph_batch8(std::vector<void const *>,
                                            std::vector<void *>, void *,
                                            cudaStream_t, void *);

extern "C" void attention_execute_mugraph_batch1(std::vector<void const *>,
                                                 std::vector<void *>, void *,
                                                 cudaStream_t, void *);
extern "C" void attention_execute_mugraph_batch2(std::vector<void const *>,
                                                 std::vector<void *>, void *,
                                                 cudaStream_t, void *);
extern "C" void attention_execute_mugraph_batch4(std::vector<void const *>,
                                                 std::vector<void *>, void *,
                                                 cudaStream_t, void *);
extern "C" void attention_execute_mugraph_batch8(std::vector<void const *>,
                                                 std::vector<void *>, void *,
                                                 cudaStream_t, void *);

extern "C" void qkv_rmsnorm_execute_mugraph_batch1(std::vector<void const *>,
                                                   std::vector<void *>, void *,
                                                   cudaStream_t, void *);
extern "C" void qkv_rmsnorm_execute_mugraph_batch2(std::vector<void const *>,
                                                   std::vector<void *>, void *,
                                                   cudaStream_t, void *);
extern "C" void qkv_rmsnorm_execute_mugraph_batch4(std::vector<void const *>,
                                                   std::vector<void *>, void *,
                                                   cudaStream_t, void *);
extern "C" void qkv_rmsnorm_execute_mugraph_batch8(std::vector<void const *>,
                                                   std::vector<void *>, void *,
                                                   cudaStream_t, void *);

extern "C" void rmsnorm_only_execute_mugraph_batch1(std::vector<void const *>,
                                                    std::vector<void *>, void *,
                                                    cudaStream_t, void *);
extern "C" void rmsnorm_only_execute_mugraph_batch2(std::vector<void const *>,
                                                    std::vector<void *>, void *,
                                                    cudaStream_t, void *);
extern "C" void rmsnorm_only_execute_mugraph_batch4(std::vector<void const *>,
                                                    std::vector<void *>, void *,
                                                    cudaStream_t, void *);
extern "C" void rmsnorm_only_execute_mugraph_batch8(std::vector<void const *>,
                                                    std::vector<void *>, void *,
                                                    cudaStream_t, void *);

extern "C" void ffn_execute_mugraph_batch1(std::vector<void const *>,
                                           std::vector<void *>, void *,
                                           cudaStream_t, void *);
extern "C" void ffn_execute_mugraph_batch2(std::vector<void const *>,
                                           std::vector<void *>, void *,
                                           cudaStream_t, void *);
extern "C" void ffn_execute_mugraph_batch4(std::vector<void const *>,
                                           std::vector<void *>, void *,
                                           cudaStream_t, void *);
extern "C" void ffn_execute_mugraph_batch8(std::vector<void const *>,
                                           std::vector<void *>, void *,
                                           cudaStream_t, void *);

constexpr size_t kRopeWorkspace1 = 1441792 + 262144;
constexpr size_t kRopeWorkspace2 = 2883584 + 262144;
constexpr size_t kRopeWorkspace4 = 5767168 + 262144;
constexpr size_t kRopeWorkspace8 = 11534336 + 262144;

constexpr size_t kAttentionWorkspace1 = 2533376 + (1 << 20);
constexpr size_t kAttentionWorkspace2 = 5146624 + (1 << 20);
constexpr size_t kAttentionWorkspace4 = 10618880 + (1 << 20);
constexpr size_t kAttentionWorkspace8 = 22548480 + (1 << 20);

constexpr size_t kQkvWorkspace1 = 1050880 + 131072;
constexpr size_t kQkvWorkspace2 = 2101760 + 131072;
constexpr size_t kQkvWorkspace4 = 4203520 + 131072;
constexpr size_t kQkvWorkspace8 = 8407040 + 131072;

constexpr size_t kFfnWorkspace1 = 589824 + 131072;
constexpr size_t kFfnWorkspace2 = 1179648 + 131072;
constexpr size_t kFfnWorkspace4 = 2359296 + 131072;
constexpr size_t kFfnWorkspace8 = 4718592 + 131072;

inline MirageExecFn get_rope_exec(int64_t batch) {
  switch (batch) {
    case 1:
      return rope_execute_mugraph_batch1;
    case 2:
      return rope_execute_mugraph_batch2;
    case 4:
      return rope_execute_mugraph_batch4;
    case 8:
      return rope_execute_mugraph_batch8;
    default:
      return nullptr;
  }
}

inline size_t get_rope_workspace(int64_t batch) {
  switch (batch) {
    case 1:
      return kRopeWorkspace1;
    case 2:
      return kRopeWorkspace2;
    case 4:
      return kRopeWorkspace4;
    case 8:
      return kRopeWorkspace8;
    default:
      return 0;
  }
}

inline MirageExecFn get_attention_exec(int64_t batch) {
  switch (batch) {
    case 1:
      return attention_execute_mugraph_batch1;
    case 2:
      return attention_execute_mugraph_batch2;
    case 4:
      return attention_execute_mugraph_batch4;
    case 8:
      return attention_execute_mugraph_batch8;
    default:
      return nullptr;
  }
}

inline size_t get_attention_workspace(int64_t batch) {
  switch (batch) {
    case 1:
      return kAttentionWorkspace1;
    case 2:
      return kAttentionWorkspace2;
    case 4:
      return kAttentionWorkspace4;
    case 8:
      return kAttentionWorkspace8;
    default:
      return 0;
  }
}

inline MirageExecFn get_qkv_exec(int64_t batch) {
  switch (batch) {
    case 1:
      return qkv_rmsnorm_execute_mugraph_batch1;
    case 2:
      return qkv_rmsnorm_execute_mugraph_batch2;
    case 4:
      return qkv_rmsnorm_execute_mugraph_batch4;
    case 8:
      return qkv_rmsnorm_execute_mugraph_batch8;
    default:
      return nullptr;
  }
}

inline size_t get_qkv_workspace(int64_t batch) {
  switch (batch) {
    case 1:
      return kQkvWorkspace1;
    case 2:
      return kQkvWorkspace2;
    case 4:
      return kQkvWorkspace4;
    case 8:
      return kQkvWorkspace8;
    default:
      return 0;
  }
}

inline MirageExecFn get_rmsnorm_exec(int64_t batch) {
  switch (batch) {
    case 1:
      return rmsnorm_only_execute_mugraph_batch1;
    case 2:
      return rmsnorm_only_execute_mugraph_batch2;
    case 4:
      return rmsnorm_only_execute_mugraph_batch4;
    case 8:
      return rmsnorm_only_execute_mugraph_batch8;
    default:
      return nullptr;
  }
}

inline size_t get_rmsnorm_workspace(int64_t /*batch*/) { return 0; }

inline MirageExecFn get_ffn_exec(int64_t batch) {
  switch (batch) {
    case 1:
      return ffn_execute_mugraph_batch1;
    case 2:
      return ffn_execute_mugraph_batch2;
    case 4:
      return ffn_execute_mugraph_batch4;
    case 8:
      return ffn_execute_mugraph_batch8;
    default:
      return nullptr;
  }
}

inline size_t get_ffn_workspace(int64_t batch) {
  switch (batch) {
    case 1:
      return kFfnWorkspace1;
    case 2:
      return kFfnWorkspace2;
    case 4:
      return kFfnWorkspace4;
    case 8:
      return kFfnWorkspace8;
    default:
      return 0;
  }
}

template <typename TensorListT>
std::vector<at::Tensor> to_contiguous_cuda(const TensorListT &tensors,
                                           const char *kernel_name,
                                           bool allow_outputs = false) {
  std::vector<at::Tensor> storage;
  storage.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto &t = tensors[i];
    TORCH_CHECK(t.is_cuda(),
                kernel_name, ": tensor ", i, " must reside on CUDA device.");
    if (allow_outputs) {
      TORCH_CHECK(t.is_contiguous(),
                  kernel_name, ": output tensor ", i, " must be contiguous.");
      storage.push_back(t);
    } else {
      storage.push_back(t.contiguous());
    }
  }
  return storage;
}

void run_kernel_impl(const char *kernel_name,
                     int64_t batch,
                     const at::TensorList &inputs,
                     const at::TensorList &outputs,
                     int64_t expected_inputs,
                     int64_t expected_outputs,
                     MirageExecFn (*exec_getter)(int64_t),
                     size_t (*workspace_getter)(int64_t)) {
  TORCH_CHECK(batch == 1 || batch == 2 || batch == 4 || batch == 8,
              kernel_name, ": unsupported batch size ", batch,
              ". Supported: 1, 2, 4, 8.");
  if (expected_inputs >= 0) {
    TORCH_CHECK(inputs.size() == expected_inputs,
                kernel_name, ": expected ", expected_inputs,
                " inputs, got ", inputs.size(), ".");
  }
  if (expected_outputs >= 0) {
    TORCH_CHECK(outputs.size() == expected_outputs,
                kernel_name, ": expected ", expected_outputs,
                " outputs, got ", outputs.size(), ".");
  }

  auto exec = exec_getter(batch);
  TORCH_CHECK(exec, kernel_name, ": executor not available for batch ", batch);

  auto input_tensors = to_contiguous_cuda(inputs, kernel_name, /*allow_outputs=*/false);
  auto output_tensors =
      to_contiguous_cuda(outputs, kernel_name, /*allow_outputs=*/true);

  TORCH_CHECK(!input_tensors.empty(),
              kernel_name, ": input tensor list must be non-empty.");
  auto device = input_tensors.front().device();
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    TORCH_CHECK(input_tensors[i].device() == device,
                kernel_name, ": input tensor ", i,
                " is on a different device.");
  }
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    TORCH_CHECK(output_tensors[i].device() == device,
                kernel_name, ": output tensor ", i,
                " is on a different device.");
  }

  std::vector<void const *> input_ptrs;
  input_ptrs.reserve(input_tensors.size());
  for (const auto &t : input_tensors) {
    input_ptrs.push_back(t.data_ptr());
  }

  std::vector<void *> output_ptrs;
  output_ptrs.reserve(output_tensors.size());
  for (const auto &t : output_tensors) {
    output_ptrs.push_back(t.data_ptr());
  }

  size_t workspace_bytes = workspace_getter(batch);
  at::Tensor workspace_tensor;
  void *workspace_ptr = nullptr;
  if (workspace_bytes > 0) {
    auto options = input_tensors.front().options().dtype(at::kByte);
    workspace_tensor = at::empty(
        {static_cast<int64_t>(workspace_bytes)}, options);
    workspace_ptr = workspace_tensor.data_ptr();
  }

  auto stream = c10::cuda::getCurrentCUDAStream();
  exec(input_ptrs, output_ptrs, workspace_ptr, stream.stream(), nullptr);
}

void rope_run(int64_t batch,
              const at::TensorList &inputs,
              const at::TensorList &outputs) {
  run_kernel_impl("mirage_rope", batch, inputs, outputs, /*expected_inputs=*/10,
                  /*expected_outputs=*/4, get_rope_exec, get_rope_workspace);
}

void attention_run(int64_t batch,
                   const at::TensorList &inputs,
                   const at::TensorList &outputs) {
  run_kernel_impl("mirage_attention", batch, inputs, outputs,
                  /*expected_inputs=*/42,
                  /*expected_outputs=*/1, get_attention_exec,
                  get_attention_workspace);
}

void qkv_rmsnorm_run(int64_t batch,
                     const at::TensorList &inputs,
                     const at::TensorList &outputs) {
  run_kernel_impl("mirage_qkv_rmsnorm", batch, inputs, outputs,
                  /*expected_inputs=*/16, /*expected_outputs=*/1,
                  get_qkv_exec, get_qkv_workspace);
}

void rmsnorm_only_run(int64_t batch,
                      const at::TensorList &inputs,
                      const at::TensorList &outputs) {
  run_kernel_impl("mirage_rmsnorm_only", batch, inputs, outputs,
                  /*expected_inputs=*/2, /*expected_outputs=*/1,
                  get_rmsnorm_exec, get_rmsnorm_workspace);
}

void ffn_run(int64_t batch,
             const at::TensorList &inputs,
             const at::TensorList &outputs) {
  run_kernel_impl("mirage_ffn", batch, inputs, outputs,
                  /*expected_inputs=*/7, /*expected_outputs=*/1,
                  get_ffn_exec, get_ffn_workspace);
}

}  // namespace

TORCH_LIBRARY(mirage_ops, m) {
  m.def("rope_run(int batch, Tensor[] inputs, Tensor[] outputs) -> ()");
  m.def("attention_run(int batch, Tensor[] inputs, Tensor[] outputs) -> ()");
  m.def("qkv_rmsnorm_run(int batch, Tensor[] inputs, Tensor[] outputs) -> ()");
  m.def("rmsnorm_only_run(int batch, Tensor[] inputs, Tensor[] outputs) -> ()");
  m.def("ffn_run(int batch, Tensor[] inputs, Tensor[] outputs) -> ()");
}

TORCH_LIBRARY_IMPL(mirage_ops, CUDA, m) {
  m.impl("rope_run", rope_run);
  m.impl("attention_run", attention_run);
  m.impl("qkv_rmsnorm_run", qkv_rmsnorm_run);
  m.impl("rmsnorm_only_run", rmsnorm_only_run);
  m.impl("ffn_run", ffn_run);
}

// Use PYBIND11_MODULE to ensure proper initialization
// This creates the Python module and allows TORCH_LIBRARY to register
PYBIND11_MODULE(_mirage_ops, m) {
  m.doc() = "Mirage fused operators for vLLM";
  // TORCH_LIBRARY registrations happen automatically via static initializers
}

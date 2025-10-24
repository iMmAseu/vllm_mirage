#define execute_mugraph rmsnorm_only_execute_mugraph_batch8
#define PyInit___mirage_launcher rmsnorm_only_PyInit_batch8
#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_7(bfloat16_t* __restrict__ dtensor10000035_ptr, bfloat16_t const* __restrict__ dtensor10000033_ptr, bfloat16_t const* __restrict__ dtensor10000034_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  bfloat16_t *stensor20000207_ptr = (bfloat16_t*)(buf + 20864);
  bfloat16_t *stensor20000205_ptr = (bfloat16_t*)(buf + 20608);
  bfloat16_t *stensor20000206_ptr = (bfloat16_t*)(buf + 4224);
  bfloat16_t *stensor30000199_ptr = (bfloat16_t*)(buf + 14464);
  bfloat16_t *stensor20000199_ptr = (bfloat16_t*)(buf + 12416);
  bfloat16_t *stensor30000198_ptr = (bfloat16_t*)(buf + 8320);
  bfloat16_t *stensor20000198_ptr = (bfloat16_t*)(buf + 4224);
  bfloat16_t *stensor20000203_ptr = (bfloat16_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000033 -> stensor 20000198
  const bfloat16_t *dtensor10000033_tile_ptr = dtensor10000033_ptr ;
  using DTensor10000033TileLayout = Layout<Shape<Int<16>, Int<128>>, Stride<Int<1>, Int<1024>>>;
  using STensor20000198InputAtom = tb::InputChunkedAsyncCopy<bfloat16_t, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<128>>, Stride<Int<1>, Int<16>>>{})), DTensor10000033TileLayout, NUM_THREADS>;
  bfloat16_t *stensor20000198_async_copy_buf = stensor30000198_ptr;
  // Copy for G->S: dtensor 10000034 -> stensor 20000199
  const bfloat16_t *dtensor10000034_tile_ptr = dtensor10000034_ptr  + blockIdx.x*64*1;
  using DTensor10000034TileLayout = Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<1024>>>;
  using STensor20000199InputAtom = tb::InputChunkedAsyncCopy<bfloat16_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{})), DTensor10000034TileLayout, NUM_THREADS>;
  bfloat16_t *stensor20000199_async_copy_buf = stensor30000199_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000207 -> dtensor 10000035
  bfloat16_t *dtensor10000035_tile_ptr = dtensor10000035_ptr  + blockIdx.x*64*128;
  using DTensor10000035TileLayout = Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>;
  using STensor20000207OutputAtom = tb::OutputChunkedSyncCopy<bfloat16_t, DTensor10000035TileLayout, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<bfloat16_t, 2048, NUM_THREADS>::run(stensor20000203_ptr, thread_idx);
  
  
  using Matmul20000206LayoutA = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<128>>, Stride<Int<1>, Int<16>>>{}));
  using Matmul20000206LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000206LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000206LayoutAAligned = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<128>>, Stride<Int<1>, Int<16>>>{}));
  using Matmul20000206LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000206Kernel = tb::Matmul<bfloat16_t, SM80_16x8x16_F32BF16BF16F32_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20000206LayoutA, Matmul20000206LayoutB, Matmul20000206LayoutC, Matmul20000206LayoutAAligned, Matmul20000206LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20000206_accum = Matmul20000206Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20000199InputAtom::run(stensor20000199_async_copy_buf, dtensor10000034_tile_ptr, thread_idx);
    STensor20000198InputAtom::run(stensor20000198_async_copy_buf, dtensor10000033_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 64; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 64) {
        STensor20000199InputAtom::run(stensor20000199_ptr, dtensor10000034_tile_ptr + 16384*(for_idx+1), thread_idx);
        STensor20000198InputAtom::run(stensor20000198_ptr, dtensor10000033_tile_ptr + 16*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000199_ptr, stensor20000199_async_copy_buf);
      SWAP(stensor20000198_ptr, stensor20000198_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      Matmul20000206Kernel::run(matmul_20000206_accum, stensor20000198_ptr, stensor20000199_ptr, (char*)(buf+0), thread_idx);
    }
    {
      // OP type: tb_square_op
      using InLayout = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
      using OutLayout = Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>;
      using Kernel = tb::ElementUnaryKernel<bfloat16_t, tb::ElementUnaryOpType::SQUARE, OutLayout, InLayout, NUM_THREADS, tb::EpilogueMulScalar<bfloat16_t, tb::EpilogueStoreAccum<bfloat16_t>>>;
      const float scalars[] = {0.000977f, 0.0f};
      Kernel::run(stensor20000203_ptr, stensor20000198_ptr, thread_idx, 0.000000, scalars);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20000206Kernel::write_back_mma_rC(stensor20000206_ptr, matmul_20000206_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_1_op
    using InLayout = Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>;
    using OutLayout = Layout<Shape<Int<128>, Int<1>>, Stride<Int<1>, Int<128>>>;
    using Kernel = tb::ReductionKernel<bfloat16_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueSqrt<bfloat16_t, tb::EpilogueStore<bfloat16_t>>>;
    const float scalars[] = {0.000000f, 0.0f};
    Kernel::run(stensor20000205_ptr, stensor20000203_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
    using In1Layout = Layout<Shape<Int<128>, Int<1>>, Stride<Int<1>, Int<128>>>;
    using OutLayout = Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>;
    using Kernel = tb::ElementBinaryKernel<bfloat16_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<bfloat16_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000207_ptr, stensor20000206_ptr, stensor20000205_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000207OutputAtom::run(dtensor10000035_tile_ptr, stensor20000207_ptr, thread_idx);
  }
}


static void _init() {
}


static void _execute_mugraph(std::vector<void const *> input_tensors, std::vector<void*> output_tensors, void* buf, cudaStream_t stream, void * profiler_buffer){
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_customized_op
    bfloat16_t *dtensor10000035 = (bfloat16_t*)output_tensors.at(0);
    bfloat16_t *dtensor10000033 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000034 = (bfloat16_t*)input_tensors.at(1);
    dim3 grid_dim(16, 1, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 37248;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_7, cudaFuncAttributeMaxDynamicSharedMemorySize, 37248);
    custom_kernel_7<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000035, dtensor10000033, dtensor10000034);
  }
  {
    // OP type: kn_output_op
  }
}


#include <Python.h>
#include <cuda_runtime.h>

static PyObject *launch(PyObject *self, PyObject *args) {
  PyObject *input_list, *output_list, *py_buffer, *py_stream, *py_profiler_buffer;
  void *buffer;
  std::vector<void const *> input_tensors;
  std::vector<void*> output_tensors;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "OOOOO", &input_list, &output_list, &py_buffer, &py_stream, &py_profiler_buffer)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(input_list) || !PyList_Check(output_list)) {
    PyErr_SetString(PyExc_TypeError, "Both arg1 and arg2 must be lists.");
    return NULL;
  }

  Py_ssize_t input_size = PyList_Size(input_list);
  Py_ssize_t output_size = PyList_Size(output_list);

  for(Py_ssize_t i = 0; i < input_size; i++) {
    PyObject *item = PyList_GetItem(input_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (input) to void pointer", i);
      return NULL;
    }
    input_tensors.push_back(PyLong_AsVoidPtr(item));
  }

  for(Py_ssize_t i = 0; i < output_size; i++) {
    PyObject *item = PyList_GetItem(output_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (output) to void pointer", i);
      return NULL;
    }
    output_tensors.push_back(PyLong_AsVoidPtr(item));
  }

  buffer = PyLong_AsVoidPtr(py_buffer);
  profiler_buffer = PyLong_AsVoidPtr(py_profiler_buffer);
  cudaStream_t stream = (cudaStream_t)PyLong_AsVoidPtr(py_stream);
  execute_mugraph(input_tensors, output_tensors, buffer, stream, profiler_buffer);

  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__mirage_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods,
  nullptr,                  // m_slots     
  nullptr,                  // m_traverse  
  nullptr,                  // m_clear     
  nullptr,                  // m_free      
};

PyMODINIT_FUNC PyInit___mirage_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}

#undef execute_mugraph
#undef PyInit___mirage_launcher

#define execute_mugraph qkv_rmsnorm_execute_mugraph_batch8
#define PyInit___mirage_launcher qkv_rmsnorm_PyInit_batch8
#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;


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
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000429 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000430 = (bfloat16_t*)input_tensors.at(1);
    bfloat16_t *dtensor10000445 = (bfloat16_t*)((char*)buf + 0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000445,dtensor10000429,dtensor10000430, 128,4096,1024, 1024,1, 4096,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000445 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000431 = (bfloat16_t*)input_tensors.at(2);
    bfloat16_t *dtensor10000446 = (bfloat16_t*)((char*)buf + 1048576);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000446,dtensor10000445,dtensor10000431, 128,2048,4096, 1,128, 2048,1, 2048,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000445 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000432 = (bfloat16_t*)input_tensors.at(3);
    bfloat16_t *dtensor10000447 = (bfloat16_t*)((char*)buf + 1572864);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000447,dtensor10000445,dtensor10000432, 128,1024,4096, 1,128, 1024,1, 1024,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000445 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000433 = (bfloat16_t*)input_tensors.at(4);
    bfloat16_t *dtensor10000448 = (bfloat16_t*)((char*)buf + 1835008);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000448,dtensor10000445,dtensor10000433, 128,1024,4096, 1,128, 1024,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_square_op
    bfloat16_t *dtensor10000446 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10000449 = (bfloat16_t*)((char*)buf + 2097152);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQUARE, Layout<Shape<Int<2048>, Int<128>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<128>>, Stride<Int<1>, Int<2048>>>>;
    kernel::run(dtensor10000449, dtensor10000446);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000449 = (bfloat16_t*)((char*)buf + 2097152);
    bfloat16_t *dtensor10000437 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000450 = (bfloat16_t*)((char*)buf + 2621440);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000450,dtensor10000449,dtensor10000437, 128,16,2048, 2048,1, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000450 = (bfloat16_t*)((char*)buf + 2621440);
    bfloat16_t *dtensor10000441 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10000451 = (bfloat16_t*)((char*)buf + 2625536);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<128>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<128>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<128>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000451, dtensor10000450, dtensor10000441);
  }
  {
    // OP type: kn_sqrt_op
    bfloat16_t *dtensor10000451 = (bfloat16_t*)((char*)buf + 2625536);
    bfloat16_t *dtensor10000452 = (bfloat16_t*)((char*)buf + 2629632);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQRT, Layout<Shape<Int<16>, Int<128>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<128>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000452, dtensor10000451);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000452 = (bfloat16_t*)((char*)buf + 2629632);
    bfloat16_t *dtensor10000439 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000453 = (bfloat16_t*)((char*)buf + 2633728);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000453,dtensor10000452,dtensor10000439, 128,2048,16, 16,1, 2048,1, 2048,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000446 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10000453 = (bfloat16_t*)((char*)buf + 2633728);
    bfloat16_t *dtensor10000454 = (bfloat16_t*)((char*)buf + 3158016);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<2048>, Int<128>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<128>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<128>>, Stride<Int<1>, Int<2048>>>>;
    kernel::run(dtensor10000454, dtensor10000446, dtensor10000453);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000454 = (bfloat16_t*)((char*)buf + 3158016);
    bfloat16_t *dtensor10000443 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10000455 = (bfloat16_t*)((char*)buf + 3682304);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<2048>, Int<128>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<128>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<128>>, Stride<Int<1>, Int<2048>>>>;
    kernel::run(dtensor10000455, dtensor10000454, dtensor10000443);
  }
  {
    // OP type: kn_square_op
    bfloat16_t *dtensor10000447 = (bfloat16_t*)((char*)buf + 1572864);
    bfloat16_t *dtensor10000456 = (bfloat16_t*)((char*)buf + 4206592);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQUARE, Layout<Shape<Int<1024>, Int<128>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<128>>, Stride<Int<1>, Int<1024>>>>;
    kernel::run(dtensor10000456, dtensor10000447);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000456 = (bfloat16_t*)((char*)buf + 4206592);
    bfloat16_t *dtensor10000438 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000457 = (bfloat16_t*)((char*)buf + 4468736);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000457,dtensor10000456,dtensor10000438, 128,8,1024, 1024,1, 8,1, 8,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000457 = (bfloat16_t*)((char*)buf + 4468736);
    bfloat16_t *dtensor10000442 = (bfloat16_t*)input_tensors.at(13);
    bfloat16_t *dtensor10000458 = (bfloat16_t*)((char*)buf + 4470784);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<8>, Int<128>>, Stride<Int<1>, Int<8>>>, Layout<Shape<Int<8>, Int<128>>, Stride<Int<1>, Int<8>>>, Layout<Shape<Int<8>, Int<128>>, Stride<Int<1>, Int<8>>>>;
    kernel::run(dtensor10000458, dtensor10000457, dtensor10000442);
  }
  {
    // OP type: kn_sqrt_op
    bfloat16_t *dtensor10000458 = (bfloat16_t*)((char*)buf + 4470784);
    bfloat16_t *dtensor10000459 = (bfloat16_t*)((char*)buf + 4472832);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQRT, Layout<Shape<Int<8>, Int<128>>, Stride<Int<1>, Int<8>>>, Layout<Shape<Int<8>, Int<128>>, Stride<Int<1>, Int<8>>>>;
    kernel::run(dtensor10000459, dtensor10000458);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000459 = (bfloat16_t*)((char*)buf + 4472832);
    bfloat16_t *dtensor10000440 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000460 = (bfloat16_t*)((char*)buf + 4474880);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000460,dtensor10000459,dtensor10000440, 128,1024,8, 8,1, 1024,1, 1024,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000447 = (bfloat16_t*)((char*)buf + 1572864);
    bfloat16_t *dtensor10000460 = (bfloat16_t*)((char*)buf + 4474880);
    bfloat16_t *dtensor10000461 = (bfloat16_t*)((char*)buf + 4737024);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<1024>, Int<128>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<128>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<128>>, Stride<Int<1>, Int<1024>>>>;
    kernel::run(dtensor10000461, dtensor10000447, dtensor10000460);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000461 = (bfloat16_t*)((char*)buf + 4737024);
    bfloat16_t *dtensor10000444 = (bfloat16_t*)input_tensors.at(15);
    bfloat16_t *dtensor10000462 = (bfloat16_t*)((char*)buf + 4999168);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<1024>, Int<128>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<128>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<128>>, Stride<Int<1>, Int<1024>>>>;
    kernel::run(dtensor10000462, dtensor10000461, dtensor10000444);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000455 = (bfloat16_t*)((char*)buf + 3682304);
    bfloat16_t *dtensor10000434 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10000463 = (bfloat16_t*)((char*)buf + 5261312);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000463,dtensor10000455,dtensor10000434, 128,4096,2048, 2048,1, 4096,1, 4096,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000462 = (bfloat16_t*)((char*)buf + 4999168);
    bfloat16_t *dtensor10000435 = (bfloat16_t*)input_tensors.at(6);
    bfloat16_t *dtensor10000464 = (bfloat16_t*)((char*)buf + 6309888);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000464,dtensor10000462,dtensor10000435, 128,4096,1024, 1024,1, 4096,1, 4096,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000448 = (bfloat16_t*)((char*)buf + 1835008);
    bfloat16_t *dtensor10000436 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000465 = (bfloat16_t*)((char*)buf + 7358464);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000465,dtensor10000448,dtensor10000436, 128,4096,1024, 1,128, 4096,1, 4096,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000463 = (bfloat16_t*)((char*)buf + 5261312);
    bfloat16_t *dtensor10000464 = (bfloat16_t*)((char*)buf + 6309888);
    bfloat16_t *dtensor10000466 = (bfloat16_t*)((char*)buf + 8407040);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<4096>, Int<128>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<128>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<128>>, Stride<Int<1>, Int<4096>>>>;
    kernel::run(dtensor10000466, dtensor10000463, dtensor10000464);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000466 = (bfloat16_t*)((char*)buf + 8407040);
    bfloat16_t *dtensor10000465 = (bfloat16_t*)((char*)buf + 7358464);
    bfloat16_t *dtensor10000467 = (bfloat16_t*)output_tensors.at(0);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<4096>, Int<128>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<128>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<128>>, Stride<Int<1>, Int<4096>>>>;
    kernel::run(dtensor10000467, dtensor10000466, dtensor10000465);
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

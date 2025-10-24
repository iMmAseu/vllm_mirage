#define execute_mugraph qkv_rmsnorm_execute_mugraph_batch4
#define PyInit___mirage_launcher qkv_rmsnorm_PyInit_batch4
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
    bfloat16_t *dtensor10000312 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000313 = (bfloat16_t*)input_tensors.at(1);
    bfloat16_t *dtensor10000328 = (bfloat16_t*)((char*)buf + 0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000328,dtensor10000312,dtensor10000313, 64,4096,1024, 1024,1, 4096,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000328 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000314 = (bfloat16_t*)input_tensors.at(2);
    bfloat16_t *dtensor10000329 = (bfloat16_t*)((char*)buf + 524288);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000329,dtensor10000328,dtensor10000314, 64,2048,4096, 1,64, 2048,1, 2048,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000328 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000315 = (bfloat16_t*)input_tensors.at(3);
    bfloat16_t *dtensor10000330 = (bfloat16_t*)((char*)buf + 786432);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000330,dtensor10000328,dtensor10000315, 64,1024,4096, 1,64, 1024,1, 1024,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000328 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000316 = (bfloat16_t*)input_tensors.at(4);
    bfloat16_t *dtensor10000331 = (bfloat16_t*)((char*)buf + 917504);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000331,dtensor10000328,dtensor10000316, 64,1024,4096, 1,64, 1024,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_square_op
    bfloat16_t *dtensor10000329 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10000332 = (bfloat16_t*)((char*)buf + 1048576);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQUARE, Layout<Shape<Int<2048>, Int<64>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<64>>, Stride<Int<1>, Int<2048>>>>;
    kernel::run(dtensor10000332, dtensor10000329);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000332 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10000320 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000333 = (bfloat16_t*)((char*)buf + 1310720);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000333,dtensor10000332,dtensor10000320, 64,16,2048, 2048,1, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000333 = (bfloat16_t*)((char*)buf + 1310720);
    bfloat16_t *dtensor10000324 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10000334 = (bfloat16_t*)((char*)buf + 1312768);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<64>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<64>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<64>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000334, dtensor10000333, dtensor10000324);
  }
  {
    // OP type: kn_sqrt_op
    bfloat16_t *dtensor10000334 = (bfloat16_t*)((char*)buf + 1312768);
    bfloat16_t *dtensor10000335 = (bfloat16_t*)((char*)buf + 1314816);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQRT, Layout<Shape<Int<16>, Int<64>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<64>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000335, dtensor10000334);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000335 = (bfloat16_t*)((char*)buf + 1314816);
    bfloat16_t *dtensor10000322 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000336 = (bfloat16_t*)((char*)buf + 1316864);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000336,dtensor10000335,dtensor10000322, 64,2048,16, 16,1, 2048,1, 2048,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000329 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10000336 = (bfloat16_t*)((char*)buf + 1316864);
    bfloat16_t *dtensor10000337 = (bfloat16_t*)((char*)buf + 1579008);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<2048>, Int<64>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<64>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<64>>, Stride<Int<1>, Int<2048>>>>;
    kernel::run(dtensor10000337, dtensor10000329, dtensor10000336);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000337 = (bfloat16_t*)((char*)buf + 1579008);
    bfloat16_t *dtensor10000326 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10000338 = (bfloat16_t*)((char*)buf + 1841152);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<2048>, Int<64>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<64>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<64>>, Stride<Int<1>, Int<2048>>>>;
    kernel::run(dtensor10000338, dtensor10000337, dtensor10000326);
  }
  {
    // OP type: kn_square_op
    bfloat16_t *dtensor10000330 = (bfloat16_t*)((char*)buf + 786432);
    bfloat16_t *dtensor10000339 = (bfloat16_t*)((char*)buf + 2103296);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQUARE, Layout<Shape<Int<1024>, Int<64>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<64>>, Stride<Int<1>, Int<1024>>>>;
    kernel::run(dtensor10000339, dtensor10000330);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000339 = (bfloat16_t*)((char*)buf + 2103296);
    bfloat16_t *dtensor10000321 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000340 = (bfloat16_t*)((char*)buf + 2234368);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000340,dtensor10000339,dtensor10000321, 64,8,1024, 1024,1, 8,1, 8,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000340 = (bfloat16_t*)((char*)buf + 2234368);
    bfloat16_t *dtensor10000325 = (bfloat16_t*)input_tensors.at(13);
    bfloat16_t *dtensor10000341 = (bfloat16_t*)((char*)buf + 2235392);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<8>, Int<64>>, Stride<Int<1>, Int<8>>>, Layout<Shape<Int<8>, Int<64>>, Stride<Int<1>, Int<8>>>, Layout<Shape<Int<8>, Int<64>>, Stride<Int<1>, Int<8>>>>;
    kernel::run(dtensor10000341, dtensor10000340, dtensor10000325);
  }
  {
    // OP type: kn_sqrt_op
    bfloat16_t *dtensor10000341 = (bfloat16_t*)((char*)buf + 2235392);
    bfloat16_t *dtensor10000342 = (bfloat16_t*)((char*)buf + 2236416);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQRT, Layout<Shape<Int<8>, Int<64>>, Stride<Int<1>, Int<8>>>, Layout<Shape<Int<8>, Int<64>>, Stride<Int<1>, Int<8>>>>;
    kernel::run(dtensor10000342, dtensor10000341);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000342 = (bfloat16_t*)((char*)buf + 2236416);
    bfloat16_t *dtensor10000323 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000343 = (bfloat16_t*)((char*)buf + 2237440);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000343,dtensor10000342,dtensor10000323, 64,1024,8, 8,1, 1024,1, 1024,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000330 = (bfloat16_t*)((char*)buf + 786432);
    bfloat16_t *dtensor10000343 = (bfloat16_t*)((char*)buf + 2237440);
    bfloat16_t *dtensor10000344 = (bfloat16_t*)((char*)buf + 2368512);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<1024>, Int<64>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<64>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<64>>, Stride<Int<1>, Int<1024>>>>;
    kernel::run(dtensor10000344, dtensor10000330, dtensor10000343);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000344 = (bfloat16_t*)((char*)buf + 2368512);
    bfloat16_t *dtensor10000327 = (bfloat16_t*)input_tensors.at(15);
    bfloat16_t *dtensor10000345 = (bfloat16_t*)((char*)buf + 2499584);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<1024>, Int<64>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<64>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<64>>, Stride<Int<1>, Int<1024>>>>;
    kernel::run(dtensor10000345, dtensor10000344, dtensor10000327);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000338 = (bfloat16_t*)((char*)buf + 1841152);
    bfloat16_t *dtensor10000317 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10000346 = (bfloat16_t*)((char*)buf + 2630656);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000346,dtensor10000338,dtensor10000317, 64,4096,2048, 2048,1, 4096,1, 4096,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000345 = (bfloat16_t*)((char*)buf + 2499584);
    bfloat16_t *dtensor10000318 = (bfloat16_t*)input_tensors.at(6);
    bfloat16_t *dtensor10000347 = (bfloat16_t*)((char*)buf + 3154944);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000347,dtensor10000345,dtensor10000318, 64,4096,1024, 1024,1, 4096,1, 4096,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000331 = (bfloat16_t*)((char*)buf + 917504);
    bfloat16_t *dtensor10000319 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000348 = (bfloat16_t*)((char*)buf + 3679232);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000348,dtensor10000331,dtensor10000319, 64,4096,1024, 1,64, 4096,1, 4096,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000346 = (bfloat16_t*)((char*)buf + 2630656);
    bfloat16_t *dtensor10000347 = (bfloat16_t*)((char*)buf + 3154944);
    bfloat16_t *dtensor10000349 = (bfloat16_t*)((char*)buf + 4203520);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<4096>, Int<64>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<64>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<64>>, Stride<Int<1>, Int<4096>>>>;
    kernel::run(dtensor10000349, dtensor10000346, dtensor10000347);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000349 = (bfloat16_t*)((char*)buf + 4203520);
    bfloat16_t *dtensor10000348 = (bfloat16_t*)((char*)buf + 3679232);
    bfloat16_t *dtensor10000350 = (bfloat16_t*)output_tensors.at(0);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<4096>, Int<64>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<64>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<64>>, Stride<Int<1>, Int<4096>>>>;
    kernel::run(dtensor10000350, dtensor10000349, dtensor10000348);
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

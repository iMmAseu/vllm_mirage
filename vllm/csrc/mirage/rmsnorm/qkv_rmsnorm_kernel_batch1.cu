#define execute_mugraph qkv_rmsnorm_execute_mugraph_batch1
#define PyInit___mirage_launcher qkv_rmsnorm_PyInit_batch1
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
    bfloat16_t *dtensor10000078 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000079 = (bfloat16_t*)input_tensors.at(1);
    bfloat16_t *dtensor10000094 = (bfloat16_t*)((char*)buf + 0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000094,dtensor10000078,dtensor10000079, 16,4096,1024, 1024,1, 4096,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000094 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000080 = (bfloat16_t*)input_tensors.at(2);
    bfloat16_t *dtensor10000095 = (bfloat16_t*)((char*)buf + 131072);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000095,dtensor10000094,dtensor10000080, 16,2048,4096, 1,16, 2048,1, 2048,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000094 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000081 = (bfloat16_t*)input_tensors.at(3);
    bfloat16_t *dtensor10000096 = (bfloat16_t*)((char*)buf + 196608);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000096,dtensor10000094,dtensor10000081, 16,1024,4096, 1,16, 1024,1, 1024,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000094 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000082 = (bfloat16_t*)input_tensors.at(4);
    bfloat16_t *dtensor10000097 = (bfloat16_t*)((char*)buf + 229376);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000097,dtensor10000094,dtensor10000082, 16,1024,4096, 1,16, 1024,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_square_op
    bfloat16_t *dtensor10000095 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000098 = (bfloat16_t*)((char*)buf + 262144);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQUARE, Layout<Shape<Int<2048>, Int<16>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<16>>, Stride<Int<1>, Int<2048>>>>;
    kernel::run(dtensor10000098, dtensor10000095);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000098 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10000086 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000099 = (bfloat16_t*)((char*)buf + 327680);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000099,dtensor10000098,dtensor10000086, 16,16,2048, 2048,1, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000099 = (bfloat16_t*)((char*)buf + 327680);
    bfloat16_t *dtensor10000090 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10000100 = (bfloat16_t*)((char*)buf + 328192);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000100, dtensor10000099, dtensor10000090);
  }
  {
    // OP type: kn_sqrt_op
    bfloat16_t *dtensor10000100 = (bfloat16_t*)((char*)buf + 328192);
    bfloat16_t *dtensor10000101 = (bfloat16_t*)((char*)buf + 328704);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQRT, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000101, dtensor10000100);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000101 = (bfloat16_t*)((char*)buf + 328704);
    bfloat16_t *dtensor10000088 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000102 = (bfloat16_t*)((char*)buf + 329216);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000102,dtensor10000101,dtensor10000088, 16,2048,16, 16,1, 2048,1, 2048,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000095 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000102 = (bfloat16_t*)((char*)buf + 329216);
    bfloat16_t *dtensor10000103 = (bfloat16_t*)((char*)buf + 394752);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<2048>, Int<16>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<16>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<16>>, Stride<Int<1>, Int<2048>>>>;
    kernel::run(dtensor10000103, dtensor10000095, dtensor10000102);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000103 = (bfloat16_t*)((char*)buf + 394752);
    bfloat16_t *dtensor10000092 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10000104 = (bfloat16_t*)((char*)buf + 460288);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<2048>, Int<16>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<16>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<16>>, Stride<Int<1>, Int<2048>>>>;
    kernel::run(dtensor10000104, dtensor10000103, dtensor10000092);
  }
  {
    // OP type: kn_square_op
    bfloat16_t *dtensor10000096 = (bfloat16_t*)((char*)buf + 196608);
    bfloat16_t *dtensor10000105 = (bfloat16_t*)((char*)buf + 525824);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQUARE, Layout<Shape<Int<1024>, Int<16>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<16>>, Stride<Int<1>, Int<1024>>>>;
    kernel::run(dtensor10000105, dtensor10000096);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000105 = (bfloat16_t*)((char*)buf + 525824);
    bfloat16_t *dtensor10000087 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000106 = (bfloat16_t*)((char*)buf + 558592);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000106,dtensor10000105,dtensor10000087, 16,8,1024, 1024,1, 8,1, 8,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000106 = (bfloat16_t*)((char*)buf + 558592);
    bfloat16_t *dtensor10000091 = (bfloat16_t*)input_tensors.at(13);
    bfloat16_t *dtensor10000107 = (bfloat16_t*)((char*)buf + 558848);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<8>, Int<16>>, Stride<Int<1>, Int<8>>>, Layout<Shape<Int<8>, Int<16>>, Stride<Int<1>, Int<8>>>, Layout<Shape<Int<8>, Int<16>>, Stride<Int<1>, Int<8>>>>;
    kernel::run(dtensor10000107, dtensor10000106, dtensor10000091);
  }
  {
    // OP type: kn_sqrt_op
    bfloat16_t *dtensor10000107 = (bfloat16_t*)((char*)buf + 558848);
    bfloat16_t *dtensor10000108 = (bfloat16_t*)((char*)buf + 559104);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQRT, Layout<Shape<Int<8>, Int<16>>, Stride<Int<1>, Int<8>>>, Layout<Shape<Int<8>, Int<16>>, Stride<Int<1>, Int<8>>>>;
    kernel::run(dtensor10000108, dtensor10000107);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000108 = (bfloat16_t*)((char*)buf + 559104);
    bfloat16_t *dtensor10000089 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000109 = (bfloat16_t*)((char*)buf + 559360);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000109,dtensor10000108,dtensor10000089, 16,1024,8, 8,1, 1024,1, 1024,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000096 = (bfloat16_t*)((char*)buf + 196608);
    bfloat16_t *dtensor10000109 = (bfloat16_t*)((char*)buf + 559360);
    bfloat16_t *dtensor10000110 = (bfloat16_t*)((char*)buf + 592128);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<1024>, Int<16>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<16>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<16>>, Stride<Int<1>, Int<1024>>>>;
    kernel::run(dtensor10000110, dtensor10000096, dtensor10000109);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000110 = (bfloat16_t*)((char*)buf + 592128);
    bfloat16_t *dtensor10000093 = (bfloat16_t*)input_tensors.at(15);
    bfloat16_t *dtensor10000111 = (bfloat16_t*)((char*)buf + 624896);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<1024>, Int<16>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<16>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<16>>, Stride<Int<1>, Int<1024>>>>;
    kernel::run(dtensor10000111, dtensor10000110, dtensor10000093);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000104 = (bfloat16_t*)((char*)buf + 460288);
    bfloat16_t *dtensor10000083 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10000112 = (bfloat16_t*)((char*)buf + 657664);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000112,dtensor10000104,dtensor10000083, 16,4096,2048, 2048,1, 4096,1, 4096,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000111 = (bfloat16_t*)((char*)buf + 624896);
    bfloat16_t *dtensor10000084 = (bfloat16_t*)input_tensors.at(6);
    bfloat16_t *dtensor10000113 = (bfloat16_t*)((char*)buf + 788736);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000113,dtensor10000111,dtensor10000084, 16,4096,1024, 1024,1, 4096,1, 4096,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000097 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000085 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000114 = (bfloat16_t*)((char*)buf + 919808);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000114,dtensor10000097,dtensor10000085, 16,4096,1024, 1,16, 4096,1, 4096,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000112 = (bfloat16_t*)((char*)buf + 657664);
    bfloat16_t *dtensor10000113 = (bfloat16_t*)((char*)buf + 788736);
    bfloat16_t *dtensor10000115 = (bfloat16_t*)((char*)buf + 1050880);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<4096>, Int<16>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<16>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<16>>, Stride<Int<1>, Int<4096>>>>;
    kernel::run(dtensor10000115, dtensor10000112, dtensor10000113);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000115 = (bfloat16_t*)((char*)buf + 1050880);
    bfloat16_t *dtensor10000114 = (bfloat16_t*)((char*)buf + 919808);
    bfloat16_t *dtensor10000116 = (bfloat16_t*)output_tensors.at(0);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<4096>, Int<16>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<16>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<16>>, Stride<Int<1>, Int<4096>>>>;
    kernel::run(dtensor10000116, dtensor10000115, dtensor10000114);
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

#define execute_mugraph qkv_rmsnorm_execute_mugraph_batch2
#define PyInit___mirage_launcher qkv_rmsnorm_PyInit_batch2
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
    bfloat16_t *dtensor10000195 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000196 = (bfloat16_t*)input_tensors.at(1);
    bfloat16_t *dtensor10000211 = (bfloat16_t*)((char*)buf + 0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000211,dtensor10000195,dtensor10000196, 32,4096,1024, 1024,1, 4096,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000211 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000197 = (bfloat16_t*)input_tensors.at(2);
    bfloat16_t *dtensor10000212 = (bfloat16_t*)((char*)buf + 262144);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000212,dtensor10000211,dtensor10000197, 32,2048,4096, 1,32, 2048,1, 2048,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000211 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000198 = (bfloat16_t*)input_tensors.at(3);
    bfloat16_t *dtensor10000213 = (bfloat16_t*)((char*)buf + 393216);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000213,dtensor10000211,dtensor10000198, 32,1024,4096, 1,32, 1024,1, 1024,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000211 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000199 = (bfloat16_t*)input_tensors.at(4);
    bfloat16_t *dtensor10000214 = (bfloat16_t*)((char*)buf + 458752);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000214,dtensor10000211,dtensor10000199, 32,1024,4096, 1,32, 1024,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_square_op
    bfloat16_t *dtensor10000212 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10000215 = (bfloat16_t*)((char*)buf + 524288);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQUARE, Layout<Shape<Int<2048>, Int<32>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<32>>, Stride<Int<1>, Int<2048>>>>;
    kernel::run(dtensor10000215, dtensor10000212);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000215 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10000203 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000216 = (bfloat16_t*)((char*)buf + 655360);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000216,dtensor10000215,dtensor10000203, 32,16,2048, 2048,1, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000216 = (bfloat16_t*)((char*)buf + 655360);
    bfloat16_t *dtensor10000207 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10000217 = (bfloat16_t*)((char*)buf + 656384);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<32>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<32>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<32>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000217, dtensor10000216, dtensor10000207);
  }
  {
    // OP type: kn_sqrt_op
    bfloat16_t *dtensor10000217 = (bfloat16_t*)((char*)buf + 656384);
    bfloat16_t *dtensor10000218 = (bfloat16_t*)((char*)buf + 657408);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQRT, Layout<Shape<Int<16>, Int<32>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<32>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000218, dtensor10000217);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000218 = (bfloat16_t*)((char*)buf + 657408);
    bfloat16_t *dtensor10000205 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000219 = (bfloat16_t*)((char*)buf + 658432);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000219,dtensor10000218,dtensor10000205, 32,2048,16, 16,1, 2048,1, 2048,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000212 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10000219 = (bfloat16_t*)((char*)buf + 658432);
    bfloat16_t *dtensor10000220 = (bfloat16_t*)((char*)buf + 789504);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<2048>, Int<32>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<32>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<32>>, Stride<Int<1>, Int<2048>>>>;
    kernel::run(dtensor10000220, dtensor10000212, dtensor10000219);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000220 = (bfloat16_t*)((char*)buf + 789504);
    bfloat16_t *dtensor10000209 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10000221 = (bfloat16_t*)((char*)buf + 920576);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<2048>, Int<32>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<32>>, Stride<Int<1>, Int<2048>>>, Layout<Shape<Int<2048>, Int<32>>, Stride<Int<1>, Int<2048>>>>;
    kernel::run(dtensor10000221, dtensor10000220, dtensor10000209);
  }
  {
    // OP type: kn_square_op
    bfloat16_t *dtensor10000213 = (bfloat16_t*)((char*)buf + 393216);
    bfloat16_t *dtensor10000222 = (bfloat16_t*)((char*)buf + 1051648);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQUARE, Layout<Shape<Int<1024>, Int<32>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<32>>, Stride<Int<1>, Int<1024>>>>;
    kernel::run(dtensor10000222, dtensor10000213);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000222 = (bfloat16_t*)((char*)buf + 1051648);
    bfloat16_t *dtensor10000204 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000223 = (bfloat16_t*)((char*)buf + 1117184);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000223,dtensor10000222,dtensor10000204, 32,8,1024, 1024,1, 8,1, 8,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000223 = (bfloat16_t*)((char*)buf + 1117184);
    bfloat16_t *dtensor10000208 = (bfloat16_t*)input_tensors.at(13);
    bfloat16_t *dtensor10000224 = (bfloat16_t*)((char*)buf + 1117696);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<8>, Int<32>>, Stride<Int<1>, Int<8>>>, Layout<Shape<Int<8>, Int<32>>, Stride<Int<1>, Int<8>>>, Layout<Shape<Int<8>, Int<32>>, Stride<Int<1>, Int<8>>>>;
    kernel::run(dtensor10000224, dtensor10000223, dtensor10000208);
  }
  {
    // OP type: kn_sqrt_op
    bfloat16_t *dtensor10000224 = (bfloat16_t*)((char*)buf + 1117696);
    bfloat16_t *dtensor10000225 = (bfloat16_t*)((char*)buf + 1118208);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SQRT, Layout<Shape<Int<8>, Int<32>>, Stride<Int<1>, Int<8>>>, Layout<Shape<Int<8>, Int<32>>, Stride<Int<1>, Int<8>>>>;
    kernel::run(dtensor10000225, dtensor10000224);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000225 = (bfloat16_t*)((char*)buf + 1118208);
    bfloat16_t *dtensor10000206 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000226 = (bfloat16_t*)((char*)buf + 1118720);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000226,dtensor10000225,dtensor10000206, 32,1024,8, 8,1, 1024,1, 1024,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000213 = (bfloat16_t*)((char*)buf + 393216);
    bfloat16_t *dtensor10000226 = (bfloat16_t*)((char*)buf + 1118720);
    bfloat16_t *dtensor10000227 = (bfloat16_t*)((char*)buf + 1184256);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<1024>, Int<32>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<32>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<32>>, Stride<Int<1>, Int<1024>>>>;
    kernel::run(dtensor10000227, dtensor10000213, dtensor10000226);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000227 = (bfloat16_t*)((char*)buf + 1184256);
    bfloat16_t *dtensor10000210 = (bfloat16_t*)input_tensors.at(15);
    bfloat16_t *dtensor10000228 = (bfloat16_t*)((char*)buf + 1249792);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<1024>, Int<32>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<32>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<32>>, Stride<Int<1>, Int<1024>>>>;
    kernel::run(dtensor10000228, dtensor10000227, dtensor10000210);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000221 = (bfloat16_t*)((char*)buf + 920576);
    bfloat16_t *dtensor10000200 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10000229 = (bfloat16_t*)((char*)buf + 1315328);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000229,dtensor10000221,dtensor10000200, 32,4096,2048, 2048,1, 4096,1, 4096,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000228 = (bfloat16_t*)((char*)buf + 1249792);
    bfloat16_t *dtensor10000201 = (bfloat16_t*)input_tensors.at(6);
    bfloat16_t *dtensor10000230 = (bfloat16_t*)((char*)buf + 1577472);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000230,dtensor10000228,dtensor10000201, 32,4096,1024, 1024,1, 4096,1, 4096,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000214 = (bfloat16_t*)((char*)buf + 458752);
    bfloat16_t *dtensor10000202 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000231 = (bfloat16_t*)((char*)buf + 1839616);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000231,dtensor10000214,dtensor10000202, 32,4096,1024, 1,32, 4096,1, 4096,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000229 = (bfloat16_t*)((char*)buf + 1315328);
    bfloat16_t *dtensor10000230 = (bfloat16_t*)((char*)buf + 1577472);
    bfloat16_t *dtensor10000232 = (bfloat16_t*)((char*)buf + 2101760);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<4096>, Int<32>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<32>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<32>>, Stride<Int<1>, Int<4096>>>>;
    kernel::run(dtensor10000232, dtensor10000229, dtensor10000230);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000232 = (bfloat16_t*)((char*)buf + 2101760);
    bfloat16_t *dtensor10000231 = (bfloat16_t*)((char*)buf + 1839616);
    bfloat16_t *dtensor10000233 = (bfloat16_t*)output_tensors.at(0);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<4096>, Int<32>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<32>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<32>>, Stride<Int<1>, Int<4096>>>>;
    kernel::run(dtensor10000233, dtensor10000232, dtensor10000231);
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

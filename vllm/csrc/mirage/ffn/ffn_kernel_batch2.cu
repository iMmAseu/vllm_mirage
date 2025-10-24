#define execute_mugraph ffn_execute_mugraph_batch2
#define PyInit___mirage_launcher ffn_PyInit_batch2
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
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000075 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000076 = (bfloat16_t*)input_tensors.at(1);
    bfloat16_t *dtensor10000082 = (bfloat16_t*)((char*)buf + 0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000082,dtensor10000075,dtensor10000076, 32,3072,1024, 1024,1, 3072,1, 3072,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000082 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000079 = (bfloat16_t*)input_tensors.at(4);
    bfloat16_t *dtensor10000083 = (bfloat16_t*)((char*)buf + 196608);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<3072>, Int<32>>, Stride<Int<1>, Int<3072>>>, Layout<Shape<Int<3072>, Int<32>>, Stride<Int<1>, Int<3072>>>, Layout<Shape<Int<3072>, Int<32>>, Stride<Int<1>, Int<3072>>>>;
    kernel::run(dtensor10000083, dtensor10000082, dtensor10000079);
  }
  {
    // OP type: kn_silu_op
    bfloat16_t *dtensor10000083 = (bfloat16_t*)((char*)buf + 196608);
    bfloat16_t *dtensor10000084 = (bfloat16_t*)((char*)buf + 393216);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SILU, Layout<Shape<Int<3072>, Int<32>>, Stride<Int<1>, Int<3072>>>, Layout<Shape<Int<3072>, Int<32>>, Stride<Int<1>, Int<3072>>>>;
    kernel::run(dtensor10000084, dtensor10000083);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000075 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000077 = (bfloat16_t*)input_tensors.at(2);
    bfloat16_t *dtensor10000085 = (bfloat16_t*)((char*)buf + 589824);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000085,dtensor10000075,dtensor10000077, 32,3072,1024, 1024,1, 3072,1, 3072,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000085 = (bfloat16_t*)((char*)buf + 589824);
    bfloat16_t *dtensor10000080 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10000086 = (bfloat16_t*)((char*)buf + 786432);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<3072>, Int<32>>, Stride<Int<1>, Int<3072>>>, Layout<Shape<Int<3072>, Int<32>>, Stride<Int<1>, Int<3072>>>, Layout<Shape<Int<3072>, Int<32>>, Stride<Int<1>, Int<3072>>>>;
    kernel::run(dtensor10000086, dtensor10000085, dtensor10000080);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000084 = (bfloat16_t*)((char*)buf + 393216);
    bfloat16_t *dtensor10000086 = (bfloat16_t*)((char*)buf + 786432);
    bfloat16_t *dtensor10000087 = (bfloat16_t*)((char*)buf + 983040);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<3072>, Int<32>>, Stride<Int<1>, Int<3072>>>, Layout<Shape<Int<3072>, Int<32>>, Stride<Int<1>, Int<3072>>>, Layout<Shape<Int<3072>, Int<32>>, Stride<Int<1>, Int<3072>>>>;
    kernel::run(dtensor10000087, dtensor10000084, dtensor10000086);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000087 = (bfloat16_t*)((char*)buf + 983040);
    bfloat16_t *dtensor10000078 = (bfloat16_t*)input_tensors.at(3);
    bfloat16_t *dtensor10000088 = (bfloat16_t*)((char*)buf + 1179648);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000088,dtensor10000087,dtensor10000078, 32,1024,3072, 3072,1, 1024,1, 1024,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000088 = (bfloat16_t*)((char*)buf + 1179648);
    bfloat16_t *dtensor10000081 = (bfloat16_t*)input_tensors.at(6);
    bfloat16_t *dtensor10000089 = (bfloat16_t*)output_tensors.at(0);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<1024>, Int<32>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<32>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<32>>, Stride<Int<1>, Int<1024>>>>;
    kernel::run(dtensor10000089, dtensor10000088, dtensor10000081);
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

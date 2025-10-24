#define execute_mugraph ffn_execute_mugraph_batch4
#define PyInit___mirage_launcher ffn_PyInit_batch4
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
    bfloat16_t *dtensor10000120 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000121 = (bfloat16_t*)input_tensors.at(1);
    bfloat16_t *dtensor10000127 = (bfloat16_t*)((char*)buf + 0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000127,dtensor10000120,dtensor10000121, 64,3072,1024, 1024,1, 3072,1, 3072,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000127 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000124 = (bfloat16_t*)input_tensors.at(4);
    bfloat16_t *dtensor10000128 = (bfloat16_t*)((char*)buf + 393216);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<3072>, Int<64>>, Stride<Int<1>, Int<3072>>>, Layout<Shape<Int<3072>, Int<64>>, Stride<Int<1>, Int<3072>>>, Layout<Shape<Int<3072>, Int<64>>, Stride<Int<1>, Int<3072>>>>;
    kernel::run(dtensor10000128, dtensor10000127, dtensor10000124);
  }
  {
    // OP type: kn_silu_op
    bfloat16_t *dtensor10000128 = (bfloat16_t*)((char*)buf + 393216);
    bfloat16_t *dtensor10000129 = (bfloat16_t*)((char*)buf + 786432);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::SILU, Layout<Shape<Int<3072>, Int<64>>, Stride<Int<1>, Int<3072>>>, Layout<Shape<Int<3072>, Int<64>>, Stride<Int<1>, Int<3072>>>>;
    kernel::run(dtensor10000129, dtensor10000128);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000120 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000122 = (bfloat16_t*)input_tensors.at(2);
    bfloat16_t *dtensor10000130 = (bfloat16_t*)((char*)buf + 1179648);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000130,dtensor10000120,dtensor10000122, 64,3072,1024, 1024,1, 3072,1, 3072,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000130 = (bfloat16_t*)((char*)buf + 1179648);
    bfloat16_t *dtensor10000125 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10000131 = (bfloat16_t*)((char*)buf + 1572864);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<3072>, Int<64>>, Stride<Int<1>, Int<3072>>>, Layout<Shape<Int<3072>, Int<64>>, Stride<Int<1>, Int<3072>>>, Layout<Shape<Int<3072>, Int<64>>, Stride<Int<1>, Int<3072>>>>;
    kernel::run(dtensor10000131, dtensor10000130, dtensor10000125);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000129 = (bfloat16_t*)((char*)buf + 786432);
    bfloat16_t *dtensor10000131 = (bfloat16_t*)((char*)buf + 1572864);
    bfloat16_t *dtensor10000132 = (bfloat16_t*)((char*)buf + 1966080);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<3072>, Int<64>>, Stride<Int<1>, Int<3072>>>, Layout<Shape<Int<3072>, Int<64>>, Stride<Int<1>, Int<3072>>>, Layout<Shape<Int<3072>, Int<64>>, Stride<Int<1>, Int<3072>>>>;
    kernel::run(dtensor10000132, dtensor10000129, dtensor10000131);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000132 = (bfloat16_t*)((char*)buf + 1966080);
    bfloat16_t *dtensor10000123 = (bfloat16_t*)input_tensors.at(3);
    bfloat16_t *dtensor10000133 = (bfloat16_t*)((char*)buf + 2359296);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000133,dtensor10000132,dtensor10000123, 64,1024,3072, 3072,1, 1024,1, 1024,1, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000133 = (bfloat16_t*)((char*)buf + 2359296);
    bfloat16_t *dtensor10000126 = (bfloat16_t*)input_tensors.at(6);
    bfloat16_t *dtensor10000134 = (bfloat16_t*)output_tensors.at(0);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<1024>, Int<64>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<64>>, Stride<Int<1>, Int<1024>>>, Layout<Shape<Int<1024>, Int<64>>, Stride<Int<1>, Int<1024>>>>;
    kernel::run(dtensor10000134, dtensor10000133, dtensor10000126);
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

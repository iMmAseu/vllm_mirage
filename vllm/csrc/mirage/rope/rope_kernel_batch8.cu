#define execute_mugraph rope_execute_mugraph_batch8
#define PyInit___mirage_launcher rope_PyInit_mirage_rope_batch8
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
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000242 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000246 = (bfloat16_t*)input_tensors.at(4);
    bfloat16_t *dtensor10000252 = (bfloat16_t*)((char*)buf + 0);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>>;
    kernel::run(dtensor10000252, dtensor10000242, dtensor10000246);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000243 = (bfloat16_t*)input_tensors.at(1);
    bfloat16_t *dtensor10000248 = (bfloat16_t*)input_tensors.at(6);
    bfloat16_t *dtensor10000253 = (bfloat16_t*)((char*)buf + 2097152);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>>;
    kernel::run(dtensor10000253, dtensor10000243, dtensor10000248);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000252 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000253 = (bfloat16_t*)((char*)buf + 2097152);
    bfloat16_t *dtensor10000254 = (bfloat16_t*)output_tensors.at(0);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>>;
    kernel::run(dtensor10000254, dtensor10000252, dtensor10000253);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000243 = (bfloat16_t*)input_tensors.at(1);
    bfloat16_t *dtensor10000246 = (bfloat16_t*)input_tensors.at(4);
    bfloat16_t *dtensor10000255 = (bfloat16_t*)((char*)buf + 4194304);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>>;
    kernel::run(dtensor10000255, dtensor10000243, dtensor10000246);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000242 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000247 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10000256 = (bfloat16_t*)((char*)buf + 6291456);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>>;
    kernel::run(dtensor10000256, dtensor10000242, dtensor10000247);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000255 = (bfloat16_t*)((char*)buf + 4194304);
    bfloat16_t *dtensor10000256 = (bfloat16_t*)((char*)buf + 6291456);
    bfloat16_t *dtensor10000257 = (bfloat16_t*)output_tensors.at(1);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>>;
    kernel::run(dtensor10000257, dtensor10000255, dtensor10000256);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000244 = (bfloat16_t*)input_tensors.at(2);
    bfloat16_t *dtensor10000249 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000258 = (bfloat16_t*)((char*)buf + 8388608);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>>;
    kernel::run(dtensor10000258, dtensor10000244, dtensor10000249);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000245 = (bfloat16_t*)input_tensors.at(3);
    bfloat16_t *dtensor10000251 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000259 = (bfloat16_t*)((char*)buf + 9437184);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>>;
    kernel::run(dtensor10000259, dtensor10000245, dtensor10000251);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000258 = (bfloat16_t*)((char*)buf + 8388608);
    bfloat16_t *dtensor10000259 = (bfloat16_t*)((char*)buf + 9437184);
    bfloat16_t *dtensor10000260 = (bfloat16_t*)output_tensors.at(2);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>>;
    kernel::run(dtensor10000260, dtensor10000258, dtensor10000259);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000245 = (bfloat16_t*)input_tensors.at(3);
    bfloat16_t *dtensor10000249 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000261 = (bfloat16_t*)((char*)buf + 10485760);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>>;
    kernel::run(dtensor10000261, dtensor10000245, dtensor10000249);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000244 = (bfloat16_t*)input_tensors.at(2);
    bfloat16_t *dtensor10000250 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000262 = (bfloat16_t*)((char*)buf + 11534336);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>>;
    kernel::run(dtensor10000262, dtensor10000244, dtensor10000250);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000261 = (bfloat16_t*)((char*)buf + 10485760);
    bfloat16_t *dtensor10000262 = (bfloat16_t*)((char*)buf + 11534336);
    bfloat16_t *dtensor10000263 = (bfloat16_t*)output_tensors.at(3);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<8>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>>;
    kernel::run(dtensor10000263, dtensor10000261, dtensor10000262);
  }
  {
    // OP type: kn_output_op
  }
  {
    // OP type: kn_output_op
  }
  {
    // OP type: kn_output_op
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

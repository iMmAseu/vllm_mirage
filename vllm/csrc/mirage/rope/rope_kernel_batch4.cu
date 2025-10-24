#define execute_mugraph rope_execute_mugraph_batch4
#define PyInit___mirage_launcher rope_PyInit_mirage_rope_batch4
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
    bfloat16_t *dtensor10000176 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000180 = (bfloat16_t*)input_tensors.at(4);
    bfloat16_t *dtensor10000186 = (bfloat16_t*)((char*)buf + 0);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>>;
    kernel::run(dtensor10000186, dtensor10000176, dtensor10000180);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000177 = (bfloat16_t*)input_tensors.at(1);
    bfloat16_t *dtensor10000182 = (bfloat16_t*)input_tensors.at(6);
    bfloat16_t *dtensor10000187 = (bfloat16_t*)((char*)buf + 1048576);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>>;
    kernel::run(dtensor10000187, dtensor10000177, dtensor10000182);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000186 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000187 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10000188 = (bfloat16_t*)output_tensors.at(0);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>>;
    kernel::run(dtensor10000188, dtensor10000186, dtensor10000187);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000177 = (bfloat16_t*)input_tensors.at(1);
    bfloat16_t *dtensor10000180 = (bfloat16_t*)input_tensors.at(4);
    bfloat16_t *dtensor10000189 = (bfloat16_t*)((char*)buf + 2097152);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>>;
    kernel::run(dtensor10000189, dtensor10000177, dtensor10000180);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000176 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000181 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10000190 = (bfloat16_t*)((char*)buf + 3145728);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>>;
    kernel::run(dtensor10000190, dtensor10000176, dtensor10000181);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000189 = (bfloat16_t*)((char*)buf + 2097152);
    bfloat16_t *dtensor10000190 = (bfloat16_t*)((char*)buf + 3145728);
    bfloat16_t *dtensor10000191 = (bfloat16_t*)output_tensors.at(1);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>, Layout<Shape<Int<64>, Int<16>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<1024>, Int<131072>>>>;
    kernel::run(dtensor10000191, dtensor10000189, dtensor10000190);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000178 = (bfloat16_t*)input_tensors.at(2);
    bfloat16_t *dtensor10000183 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000192 = (bfloat16_t*)((char*)buf + 4194304);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>>;
    kernel::run(dtensor10000192, dtensor10000178, dtensor10000183);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000179 = (bfloat16_t*)input_tensors.at(3);
    bfloat16_t *dtensor10000185 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000193 = (bfloat16_t*)((char*)buf + 4718592);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>>;
    kernel::run(dtensor10000193, dtensor10000179, dtensor10000185);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000192 = (bfloat16_t*)((char*)buf + 4194304);
    bfloat16_t *dtensor10000193 = (bfloat16_t*)((char*)buf + 4718592);
    bfloat16_t *dtensor10000194 = (bfloat16_t*)output_tensors.at(2);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>>;
    kernel::run(dtensor10000194, dtensor10000192, dtensor10000193);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000179 = (bfloat16_t*)input_tensors.at(3);
    bfloat16_t *dtensor10000183 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000195 = (bfloat16_t*)((char*)buf + 5242880);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>>;
    kernel::run(dtensor10000195, dtensor10000179, dtensor10000183);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000178 = (bfloat16_t*)input_tensors.at(2);
    bfloat16_t *dtensor10000184 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000196 = (bfloat16_t*)((char*)buf + 5767168);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>>;
    kernel::run(dtensor10000196, dtensor10000178, dtensor10000184);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000195 = (bfloat16_t*)((char*)buf + 5242880);
    bfloat16_t *dtensor10000196 = (bfloat16_t*)((char*)buf + 5767168);
    bfloat16_t *dtensor10000197 = (bfloat16_t*)output_tensors.at(3);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>, Layout<Shape<Int<64>, Int<8>, Int<128>, Int<4>>, Stride<Int<1>, Int<64>, Int<512>, Int<65536>>>>;
    kernel::run(dtensor10000197, dtensor10000195, dtensor10000196);
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

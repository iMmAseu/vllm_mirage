#define execute_mugraph attention_execute_mugraph_batch4
#define PyInit___mirage_launcher attention_PyInit_mirage_batch4
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
    bfloat16_t *dtensor10002025 = (bfloat16_t*)input_tensors.at(1);
    bfloat16_t *dtensor10002029 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002066,dtensor10002025,dtensor10002029, 64,2048,1024, 1024,1, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002030 = (bfloat16_t*)input_tensors.at(6);
    bfloat16_t *dtensor10002026 = (bfloat16_t*)input_tensors.at(2);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002067,dtensor10002030,dtensor10002026, 2048,64,1024, 1024,1, 64,1, 1,2048, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002027 = (bfloat16_t*)input_tensors.at(3);
    bfloat16_t *dtensor10002029 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002068,dtensor10002027,dtensor10002029, 64,2048,1024, 1024,1, 2048,1, 2048,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002034 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10002069 = (bfloat16_t*)((char*)buf + 786432);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002069,dtensor10002024,dtensor10002034, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002034 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10002070 = (bfloat16_t*)((char*)buf + 802816);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002070,dtensor10002066,dtensor10002034, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002035 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002071 = (bfloat16_t*)((char*)buf + 819200);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002071,dtensor10002035,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002069 = (bfloat16_t*)((char*)buf + 786432);
    bfloat16_t *dtensor10002071 = (bfloat16_t*)((char*)buf + 819200);
    bfloat16_t *dtensor10002072 = (bfloat16_t*)((char*)buf + 835584);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002072,dtensor10002069,dtensor10002071, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002072 = (bfloat16_t*)((char*)buf + 835584);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002073 = (bfloat16_t*)((char*)buf + 843776);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002073, dtensor10002072, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002073 = (bfloat16_t*)((char*)buf + 843776);
    bfloat16_t *dtensor10002074 = (bfloat16_t*)((char*)buf + 851968);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002074, dtensor10002073);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002074 = (bfloat16_t*)((char*)buf + 851968);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002075 = (bfloat16_t*)((char*)buf + 860160);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002075,dtensor10002074,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002075 = (bfloat16_t*)((char*)buf + 860160);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002076 = (bfloat16_t*)((char*)buf + 860288);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002076,dtensor10002075,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002074 = (bfloat16_t*)((char*)buf + 851968);
    bfloat16_t *dtensor10002076 = (bfloat16_t*)((char*)buf + 860288);
    bfloat16_t *dtensor10002077 = (bfloat16_t*)((char*)buf + 868480);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002077, dtensor10002074, dtensor10002076);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002034 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10002078 = (bfloat16_t*)((char*)buf + 876672);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002078,dtensor10002068,dtensor10002034, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002077 = (bfloat16_t*)((char*)buf + 868480);
    bfloat16_t *dtensor10002078 = (bfloat16_t*)((char*)buf + 876672);
    bfloat16_t *dtensor10002079 = (bfloat16_t*)((char*)buf + 893056);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002079,dtensor10002077,dtensor10002078, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002079 = (bfloat16_t*)((char*)buf + 893056);
    bfloat16_t *dtensor10002035 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10002080 = (bfloat16_t*)((char*)buf + 909440);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002080,dtensor10002079,dtensor10002035, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002036 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10002081 = (bfloat16_t*)((char*)buf + 1171584);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002081,dtensor10002024,dtensor10002036, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002036 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10002082 = (bfloat16_t*)((char*)buf + 1187968);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002082,dtensor10002066,dtensor10002036, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002037 = (bfloat16_t*)input_tensors.at(13);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002083 = (bfloat16_t*)((char*)buf + 1204352);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002083,dtensor10002037,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002081 = (bfloat16_t*)((char*)buf + 1171584);
    bfloat16_t *dtensor10002083 = (bfloat16_t*)((char*)buf + 1204352);
    bfloat16_t *dtensor10002084 = (bfloat16_t*)((char*)buf + 1220736);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002084,dtensor10002081,dtensor10002083, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002084 = (bfloat16_t*)((char*)buf + 1220736);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002085 = (bfloat16_t*)((char*)buf + 1228928);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002085, dtensor10002084, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002085 = (bfloat16_t*)((char*)buf + 1228928);
    bfloat16_t *dtensor10002086 = (bfloat16_t*)((char*)buf + 1237120);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002086, dtensor10002085);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002086 = (bfloat16_t*)((char*)buf + 1237120);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002087 = (bfloat16_t*)((char*)buf + 1245312);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002087,dtensor10002086,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002087 = (bfloat16_t*)((char*)buf + 1245312);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002088 = (bfloat16_t*)((char*)buf + 1245440);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002088,dtensor10002087,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002086 = (bfloat16_t*)((char*)buf + 1237120);
    bfloat16_t *dtensor10002088 = (bfloat16_t*)((char*)buf + 1245440);
    bfloat16_t *dtensor10002089 = (bfloat16_t*)((char*)buf + 1253632);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002089, dtensor10002086, dtensor10002088);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002036 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10002090 = (bfloat16_t*)((char*)buf + 1261824);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002090,dtensor10002068,dtensor10002036, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002089 = (bfloat16_t*)((char*)buf + 1253632);
    bfloat16_t *dtensor10002090 = (bfloat16_t*)((char*)buf + 1261824);
    bfloat16_t *dtensor10002091 = (bfloat16_t*)((char*)buf + 1278208);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002091,dtensor10002089,dtensor10002090, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002091 = (bfloat16_t*)((char*)buf + 1278208);
    bfloat16_t *dtensor10002037 = (bfloat16_t*)input_tensors.at(13);
    bfloat16_t *dtensor10002092 = (bfloat16_t*)((char*)buf + 1294592);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002092,dtensor10002091,dtensor10002037, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002080 = (bfloat16_t*)((char*)buf + 909440);
    bfloat16_t *dtensor10002092 = (bfloat16_t*)((char*)buf + 1294592);
    bfloat16_t *dtensor10002093 = (bfloat16_t*)((char*)buf + 1556736);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002093, dtensor10002080, dtensor10002092);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002038 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10002094 = (bfloat16_t*)((char*)buf + 1818880);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002094,dtensor10002024,dtensor10002038, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002038 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10002095 = (bfloat16_t*)((char*)buf + 1835264);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002095,dtensor10002066,dtensor10002038, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002039 = (bfloat16_t*)input_tensors.at(15);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002096 = (bfloat16_t*)((char*)buf + 1851648);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002096,dtensor10002039,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002094 = (bfloat16_t*)((char*)buf + 1818880);
    bfloat16_t *dtensor10002096 = (bfloat16_t*)((char*)buf + 1851648);
    bfloat16_t *dtensor10002097 = (bfloat16_t*)((char*)buf + 1868032);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002097,dtensor10002094,dtensor10002096, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002097 = (bfloat16_t*)((char*)buf + 1868032);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002098 = (bfloat16_t*)((char*)buf + 1876224);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002098, dtensor10002097, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002098 = (bfloat16_t*)((char*)buf + 1876224);
    bfloat16_t *dtensor10002099 = (bfloat16_t*)((char*)buf + 1884416);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002099, dtensor10002098);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002099 = (bfloat16_t*)((char*)buf + 1884416);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002100 = (bfloat16_t*)((char*)buf + 1892608);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002100,dtensor10002099,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002100 = (bfloat16_t*)((char*)buf + 1892608);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002101 = (bfloat16_t*)((char*)buf + 1892736);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002101,dtensor10002100,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002099 = (bfloat16_t*)((char*)buf + 1884416);
    bfloat16_t *dtensor10002101 = (bfloat16_t*)((char*)buf + 1892736);
    bfloat16_t *dtensor10002102 = (bfloat16_t*)((char*)buf + 1900928);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002102, dtensor10002099, dtensor10002101);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002038 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10002103 = (bfloat16_t*)((char*)buf + 1909120);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002103,dtensor10002068,dtensor10002038, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002102 = (bfloat16_t*)((char*)buf + 1900928);
    bfloat16_t *dtensor10002103 = (bfloat16_t*)((char*)buf + 1909120);
    bfloat16_t *dtensor10002104 = (bfloat16_t*)((char*)buf + 1925504);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002104,dtensor10002102,dtensor10002103, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002104 = (bfloat16_t*)((char*)buf + 1925504);
    bfloat16_t *dtensor10002039 = (bfloat16_t*)input_tensors.at(15);
    bfloat16_t *dtensor10002105 = (bfloat16_t*)((char*)buf + 1941888);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002105,dtensor10002104,dtensor10002039, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002093 = (bfloat16_t*)((char*)buf + 1556736);
    bfloat16_t *dtensor10002105 = (bfloat16_t*)((char*)buf + 1941888);
    bfloat16_t *dtensor10002106 = (bfloat16_t*)((char*)buf + 2204032);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002106, dtensor10002093, dtensor10002105);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002040 = (bfloat16_t*)input_tensors.at(16);
    bfloat16_t *dtensor10002107 = (bfloat16_t*)((char*)buf + 2466176);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002107,dtensor10002024,dtensor10002040, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002040 = (bfloat16_t*)input_tensors.at(16);
    bfloat16_t *dtensor10002108 = (bfloat16_t*)((char*)buf + 2482560);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002108,dtensor10002066,dtensor10002040, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002041 = (bfloat16_t*)input_tensors.at(17);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002109 = (bfloat16_t*)((char*)buf + 2498944);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002109,dtensor10002041,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002107 = (bfloat16_t*)((char*)buf + 2466176);
    bfloat16_t *dtensor10002109 = (bfloat16_t*)((char*)buf + 2498944);
    bfloat16_t *dtensor10002110 = (bfloat16_t*)((char*)buf + 2515328);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002110,dtensor10002107,dtensor10002109, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002110 = (bfloat16_t*)((char*)buf + 2515328);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002111 = (bfloat16_t*)((char*)buf + 2523520);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002111, dtensor10002110, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002111 = (bfloat16_t*)((char*)buf + 2523520);
    bfloat16_t *dtensor10002112 = (bfloat16_t*)((char*)buf + 2531712);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002112, dtensor10002111);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002112 = (bfloat16_t*)((char*)buf + 2531712);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002113 = (bfloat16_t*)((char*)buf + 2539904);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002113,dtensor10002112,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002113 = (bfloat16_t*)((char*)buf + 2539904);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002114 = (bfloat16_t*)((char*)buf + 2540032);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002114,dtensor10002113,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002112 = (bfloat16_t*)((char*)buf + 2531712);
    bfloat16_t *dtensor10002114 = (bfloat16_t*)((char*)buf + 2540032);
    bfloat16_t *dtensor10002115 = (bfloat16_t*)((char*)buf + 2548224);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002115, dtensor10002112, dtensor10002114);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002040 = (bfloat16_t*)input_tensors.at(16);
    bfloat16_t *dtensor10002116 = (bfloat16_t*)((char*)buf + 2556416);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002116,dtensor10002068,dtensor10002040, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002115 = (bfloat16_t*)((char*)buf + 2548224);
    bfloat16_t *dtensor10002116 = (bfloat16_t*)((char*)buf + 2556416);
    bfloat16_t *dtensor10002117 = (bfloat16_t*)((char*)buf + 2572800);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002117,dtensor10002115,dtensor10002116, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002117 = (bfloat16_t*)((char*)buf + 2572800);
    bfloat16_t *dtensor10002041 = (bfloat16_t*)input_tensors.at(17);
    bfloat16_t *dtensor10002118 = (bfloat16_t*)((char*)buf + 2589184);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002118,dtensor10002117,dtensor10002041, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002106 = (bfloat16_t*)((char*)buf + 2204032);
    bfloat16_t *dtensor10002118 = (bfloat16_t*)((char*)buf + 2589184);
    bfloat16_t *dtensor10002119 = (bfloat16_t*)((char*)buf + 2851328);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002119, dtensor10002106, dtensor10002118);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002042 = (bfloat16_t*)input_tensors.at(18);
    bfloat16_t *dtensor10002120 = (bfloat16_t*)((char*)buf + 3113472);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002120,dtensor10002024,dtensor10002042, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002042 = (bfloat16_t*)input_tensors.at(18);
    bfloat16_t *dtensor10002121 = (bfloat16_t*)((char*)buf + 3129856);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002121,dtensor10002066,dtensor10002042, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002043 = (bfloat16_t*)input_tensors.at(19);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002122 = (bfloat16_t*)((char*)buf + 3146240);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002122,dtensor10002043,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002120 = (bfloat16_t*)((char*)buf + 3113472);
    bfloat16_t *dtensor10002122 = (bfloat16_t*)((char*)buf + 3146240);
    bfloat16_t *dtensor10002123 = (bfloat16_t*)((char*)buf + 3162624);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002123,dtensor10002120,dtensor10002122, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002123 = (bfloat16_t*)((char*)buf + 3162624);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002124 = (bfloat16_t*)((char*)buf + 3170816);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002124, dtensor10002123, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002124 = (bfloat16_t*)((char*)buf + 3170816);
    bfloat16_t *dtensor10002125 = (bfloat16_t*)((char*)buf + 3179008);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002125, dtensor10002124);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002125 = (bfloat16_t*)((char*)buf + 3179008);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002126 = (bfloat16_t*)((char*)buf + 3187200);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002126,dtensor10002125,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002126 = (bfloat16_t*)((char*)buf + 3187200);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002127 = (bfloat16_t*)((char*)buf + 3187328);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002127,dtensor10002126,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002125 = (bfloat16_t*)((char*)buf + 3179008);
    bfloat16_t *dtensor10002127 = (bfloat16_t*)((char*)buf + 3187328);
    bfloat16_t *dtensor10002128 = (bfloat16_t*)((char*)buf + 3195520);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002128, dtensor10002125, dtensor10002127);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002042 = (bfloat16_t*)input_tensors.at(18);
    bfloat16_t *dtensor10002129 = (bfloat16_t*)((char*)buf + 3203712);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002129,dtensor10002068,dtensor10002042, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002128 = (bfloat16_t*)((char*)buf + 3195520);
    bfloat16_t *dtensor10002129 = (bfloat16_t*)((char*)buf + 3203712);
    bfloat16_t *dtensor10002130 = (bfloat16_t*)((char*)buf + 3220096);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002130,dtensor10002128,dtensor10002129, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002130 = (bfloat16_t*)((char*)buf + 3220096);
    bfloat16_t *dtensor10002043 = (bfloat16_t*)input_tensors.at(19);
    bfloat16_t *dtensor10002131 = (bfloat16_t*)((char*)buf + 3236480);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002131,dtensor10002130,dtensor10002043, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002119 = (bfloat16_t*)((char*)buf + 2851328);
    bfloat16_t *dtensor10002131 = (bfloat16_t*)((char*)buf + 3236480);
    bfloat16_t *dtensor10002132 = (bfloat16_t*)((char*)buf + 3498624);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002132, dtensor10002119, dtensor10002131);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002044 = (bfloat16_t*)input_tensors.at(20);
    bfloat16_t *dtensor10002133 = (bfloat16_t*)((char*)buf + 3760768);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002133,dtensor10002024,dtensor10002044, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002044 = (bfloat16_t*)input_tensors.at(20);
    bfloat16_t *dtensor10002134 = (bfloat16_t*)((char*)buf + 3777152);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002134,dtensor10002066,dtensor10002044, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002045 = (bfloat16_t*)input_tensors.at(21);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002135 = (bfloat16_t*)((char*)buf + 3793536);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002135,dtensor10002045,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002133 = (bfloat16_t*)((char*)buf + 3760768);
    bfloat16_t *dtensor10002135 = (bfloat16_t*)((char*)buf + 3793536);
    bfloat16_t *dtensor10002136 = (bfloat16_t*)((char*)buf + 3809920);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002136,dtensor10002133,dtensor10002135, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002136 = (bfloat16_t*)((char*)buf + 3809920);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002137 = (bfloat16_t*)((char*)buf + 3818112);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002137, dtensor10002136, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002137 = (bfloat16_t*)((char*)buf + 3818112);
    bfloat16_t *dtensor10002138 = (bfloat16_t*)((char*)buf + 3826304);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002138, dtensor10002137);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002138 = (bfloat16_t*)((char*)buf + 3826304);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002139 = (bfloat16_t*)((char*)buf + 3834496);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002139,dtensor10002138,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002139 = (bfloat16_t*)((char*)buf + 3834496);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002140 = (bfloat16_t*)((char*)buf + 3834624);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002140,dtensor10002139,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002138 = (bfloat16_t*)((char*)buf + 3826304);
    bfloat16_t *dtensor10002140 = (bfloat16_t*)((char*)buf + 3834624);
    bfloat16_t *dtensor10002141 = (bfloat16_t*)((char*)buf + 3842816);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002141, dtensor10002138, dtensor10002140);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002044 = (bfloat16_t*)input_tensors.at(20);
    bfloat16_t *dtensor10002142 = (bfloat16_t*)((char*)buf + 3851008);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002142,dtensor10002068,dtensor10002044, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002141 = (bfloat16_t*)((char*)buf + 3842816);
    bfloat16_t *dtensor10002142 = (bfloat16_t*)((char*)buf + 3851008);
    bfloat16_t *dtensor10002143 = (bfloat16_t*)((char*)buf + 3867392);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002143,dtensor10002141,dtensor10002142, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002143 = (bfloat16_t*)((char*)buf + 3867392);
    bfloat16_t *dtensor10002045 = (bfloat16_t*)input_tensors.at(21);
    bfloat16_t *dtensor10002144 = (bfloat16_t*)((char*)buf + 3883776);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002144,dtensor10002143,dtensor10002045, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002132 = (bfloat16_t*)((char*)buf + 3498624);
    bfloat16_t *dtensor10002144 = (bfloat16_t*)((char*)buf + 3883776);
    bfloat16_t *dtensor10002145 = (bfloat16_t*)((char*)buf + 4145920);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002145, dtensor10002132, dtensor10002144);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002046 = (bfloat16_t*)input_tensors.at(22);
    bfloat16_t *dtensor10002146 = (bfloat16_t*)((char*)buf + 4408064);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002146,dtensor10002024,dtensor10002046, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002046 = (bfloat16_t*)input_tensors.at(22);
    bfloat16_t *dtensor10002147 = (bfloat16_t*)((char*)buf + 4424448);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002147,dtensor10002066,dtensor10002046, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002047 = (bfloat16_t*)input_tensors.at(23);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002148 = (bfloat16_t*)((char*)buf + 4440832);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002148,dtensor10002047,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002146 = (bfloat16_t*)((char*)buf + 4408064);
    bfloat16_t *dtensor10002148 = (bfloat16_t*)((char*)buf + 4440832);
    bfloat16_t *dtensor10002149 = (bfloat16_t*)((char*)buf + 4457216);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002149,dtensor10002146,dtensor10002148, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002149 = (bfloat16_t*)((char*)buf + 4457216);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002150 = (bfloat16_t*)((char*)buf + 4465408);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002150, dtensor10002149, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002150 = (bfloat16_t*)((char*)buf + 4465408);
    bfloat16_t *dtensor10002151 = (bfloat16_t*)((char*)buf + 4473600);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002151, dtensor10002150);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002151 = (bfloat16_t*)((char*)buf + 4473600);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002152 = (bfloat16_t*)((char*)buf + 4481792);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002152,dtensor10002151,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002152 = (bfloat16_t*)((char*)buf + 4481792);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002153 = (bfloat16_t*)((char*)buf + 4481920);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002153,dtensor10002152,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002151 = (bfloat16_t*)((char*)buf + 4473600);
    bfloat16_t *dtensor10002153 = (bfloat16_t*)((char*)buf + 4481920);
    bfloat16_t *dtensor10002154 = (bfloat16_t*)((char*)buf + 4490112);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002154, dtensor10002151, dtensor10002153);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002046 = (bfloat16_t*)input_tensors.at(22);
    bfloat16_t *dtensor10002155 = (bfloat16_t*)((char*)buf + 4498304);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002155,dtensor10002068,dtensor10002046, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002154 = (bfloat16_t*)((char*)buf + 4490112);
    bfloat16_t *dtensor10002155 = (bfloat16_t*)((char*)buf + 4498304);
    bfloat16_t *dtensor10002156 = (bfloat16_t*)((char*)buf + 4514688);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002156,dtensor10002154,dtensor10002155, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002156 = (bfloat16_t*)((char*)buf + 4514688);
    bfloat16_t *dtensor10002047 = (bfloat16_t*)input_tensors.at(23);
    bfloat16_t *dtensor10002157 = (bfloat16_t*)((char*)buf + 4531072);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002157,dtensor10002156,dtensor10002047, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002145 = (bfloat16_t*)((char*)buf + 4145920);
    bfloat16_t *dtensor10002157 = (bfloat16_t*)((char*)buf + 4531072);
    bfloat16_t *dtensor10002158 = (bfloat16_t*)((char*)buf + 4793216);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002158, dtensor10002145, dtensor10002157);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002048 = (bfloat16_t*)input_tensors.at(24);
    bfloat16_t *dtensor10002159 = (bfloat16_t*)((char*)buf + 5055360);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002159,dtensor10002024,dtensor10002048, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002048 = (bfloat16_t*)input_tensors.at(24);
    bfloat16_t *dtensor10002160 = (bfloat16_t*)((char*)buf + 5071744);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002160,dtensor10002066,dtensor10002048, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002049 = (bfloat16_t*)input_tensors.at(25);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002161 = (bfloat16_t*)((char*)buf + 5088128);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002161,dtensor10002049,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002159 = (bfloat16_t*)((char*)buf + 5055360);
    bfloat16_t *dtensor10002161 = (bfloat16_t*)((char*)buf + 5088128);
    bfloat16_t *dtensor10002162 = (bfloat16_t*)((char*)buf + 5104512);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002162,dtensor10002159,dtensor10002161, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002162 = (bfloat16_t*)((char*)buf + 5104512);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002163 = (bfloat16_t*)((char*)buf + 5112704);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002163, dtensor10002162, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002163 = (bfloat16_t*)((char*)buf + 5112704);
    bfloat16_t *dtensor10002164 = (bfloat16_t*)((char*)buf + 5120896);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002164, dtensor10002163);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002164 = (bfloat16_t*)((char*)buf + 5120896);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002165 = (bfloat16_t*)((char*)buf + 5129088);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002165,dtensor10002164,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002165 = (bfloat16_t*)((char*)buf + 5129088);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002166 = (bfloat16_t*)((char*)buf + 5129216);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002166,dtensor10002165,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002164 = (bfloat16_t*)((char*)buf + 5120896);
    bfloat16_t *dtensor10002166 = (bfloat16_t*)((char*)buf + 5129216);
    bfloat16_t *dtensor10002167 = (bfloat16_t*)((char*)buf + 5137408);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002167, dtensor10002164, dtensor10002166);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002048 = (bfloat16_t*)input_tensors.at(24);
    bfloat16_t *dtensor10002168 = (bfloat16_t*)((char*)buf + 5145600);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002168,dtensor10002068,dtensor10002048, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002167 = (bfloat16_t*)((char*)buf + 5137408);
    bfloat16_t *dtensor10002168 = (bfloat16_t*)((char*)buf + 5145600);
    bfloat16_t *dtensor10002169 = (bfloat16_t*)((char*)buf + 5161984);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002169,dtensor10002167,dtensor10002168, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002169 = (bfloat16_t*)((char*)buf + 5161984);
    bfloat16_t *dtensor10002049 = (bfloat16_t*)input_tensors.at(25);
    bfloat16_t *dtensor10002170 = (bfloat16_t*)((char*)buf + 5178368);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002170,dtensor10002169,dtensor10002049, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002158 = (bfloat16_t*)((char*)buf + 4793216);
    bfloat16_t *dtensor10002170 = (bfloat16_t*)((char*)buf + 5178368);
    bfloat16_t *dtensor10002171 = (bfloat16_t*)((char*)buf + 5440512);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002171, dtensor10002158, dtensor10002170);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002050 = (bfloat16_t*)input_tensors.at(26);
    bfloat16_t *dtensor10002172 = (bfloat16_t*)((char*)buf + 5702656);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002172,dtensor10002024,dtensor10002050, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002050 = (bfloat16_t*)input_tensors.at(26);
    bfloat16_t *dtensor10002173 = (bfloat16_t*)((char*)buf + 5719040);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002173,dtensor10002066,dtensor10002050, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002051 = (bfloat16_t*)input_tensors.at(27);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002174 = (bfloat16_t*)((char*)buf + 5735424);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002174,dtensor10002051,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002172 = (bfloat16_t*)((char*)buf + 5702656);
    bfloat16_t *dtensor10002174 = (bfloat16_t*)((char*)buf + 5735424);
    bfloat16_t *dtensor10002175 = (bfloat16_t*)((char*)buf + 5751808);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002175,dtensor10002172,dtensor10002174, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002175 = (bfloat16_t*)((char*)buf + 5751808);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002176 = (bfloat16_t*)((char*)buf + 5760000);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002176, dtensor10002175, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002176 = (bfloat16_t*)((char*)buf + 5760000);
    bfloat16_t *dtensor10002177 = (bfloat16_t*)((char*)buf + 5768192);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002177, dtensor10002176);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002177 = (bfloat16_t*)((char*)buf + 5768192);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002178 = (bfloat16_t*)((char*)buf + 5776384);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002178,dtensor10002177,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002178 = (bfloat16_t*)((char*)buf + 5776384);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002179 = (bfloat16_t*)((char*)buf + 5776512);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002179,dtensor10002178,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002177 = (bfloat16_t*)((char*)buf + 5768192);
    bfloat16_t *dtensor10002179 = (bfloat16_t*)((char*)buf + 5776512);
    bfloat16_t *dtensor10002180 = (bfloat16_t*)((char*)buf + 5784704);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002180, dtensor10002177, dtensor10002179);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002050 = (bfloat16_t*)input_tensors.at(26);
    bfloat16_t *dtensor10002181 = (bfloat16_t*)((char*)buf + 5792896);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002181,dtensor10002068,dtensor10002050, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002180 = (bfloat16_t*)((char*)buf + 5784704);
    bfloat16_t *dtensor10002181 = (bfloat16_t*)((char*)buf + 5792896);
    bfloat16_t *dtensor10002182 = (bfloat16_t*)((char*)buf + 5809280);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002182,dtensor10002180,dtensor10002181, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002182 = (bfloat16_t*)((char*)buf + 5809280);
    bfloat16_t *dtensor10002051 = (bfloat16_t*)input_tensors.at(27);
    bfloat16_t *dtensor10002183 = (bfloat16_t*)((char*)buf + 5825664);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002183,dtensor10002182,dtensor10002051, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002171 = (bfloat16_t*)((char*)buf + 5440512);
    bfloat16_t *dtensor10002183 = (bfloat16_t*)((char*)buf + 5825664);
    bfloat16_t *dtensor10002184 = (bfloat16_t*)((char*)buf + 6087808);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002184, dtensor10002171, dtensor10002183);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002052 = (bfloat16_t*)input_tensors.at(28);
    bfloat16_t *dtensor10002185 = (bfloat16_t*)((char*)buf + 6349952);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002185,dtensor10002024,dtensor10002052, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002052 = (bfloat16_t*)input_tensors.at(28);
    bfloat16_t *dtensor10002186 = (bfloat16_t*)((char*)buf + 6366336);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002186,dtensor10002066,dtensor10002052, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002053 = (bfloat16_t*)input_tensors.at(29);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002187 = (bfloat16_t*)((char*)buf + 6382720);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002187,dtensor10002053,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002185 = (bfloat16_t*)((char*)buf + 6349952);
    bfloat16_t *dtensor10002187 = (bfloat16_t*)((char*)buf + 6382720);
    bfloat16_t *dtensor10002188 = (bfloat16_t*)((char*)buf + 6399104);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002188,dtensor10002185,dtensor10002187, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002188 = (bfloat16_t*)((char*)buf + 6399104);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002189 = (bfloat16_t*)((char*)buf + 6407296);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002189, dtensor10002188, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002189 = (bfloat16_t*)((char*)buf + 6407296);
    bfloat16_t *dtensor10002190 = (bfloat16_t*)((char*)buf + 6415488);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002190, dtensor10002189);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002190 = (bfloat16_t*)((char*)buf + 6415488);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002191 = (bfloat16_t*)((char*)buf + 6423680);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002191,dtensor10002190,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002191 = (bfloat16_t*)((char*)buf + 6423680);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002192 = (bfloat16_t*)((char*)buf + 6423808);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002192,dtensor10002191,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002190 = (bfloat16_t*)((char*)buf + 6415488);
    bfloat16_t *dtensor10002192 = (bfloat16_t*)((char*)buf + 6423808);
    bfloat16_t *dtensor10002193 = (bfloat16_t*)((char*)buf + 6432000);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002193, dtensor10002190, dtensor10002192);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002052 = (bfloat16_t*)input_tensors.at(28);
    bfloat16_t *dtensor10002194 = (bfloat16_t*)((char*)buf + 6440192);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002194,dtensor10002068,dtensor10002052, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002193 = (bfloat16_t*)((char*)buf + 6432000);
    bfloat16_t *dtensor10002194 = (bfloat16_t*)((char*)buf + 6440192);
    bfloat16_t *dtensor10002195 = (bfloat16_t*)((char*)buf + 6456576);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002195,dtensor10002193,dtensor10002194, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002195 = (bfloat16_t*)((char*)buf + 6456576);
    bfloat16_t *dtensor10002053 = (bfloat16_t*)input_tensors.at(29);
    bfloat16_t *dtensor10002196 = (bfloat16_t*)((char*)buf + 6472960);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002196,dtensor10002195,dtensor10002053, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002184 = (bfloat16_t*)((char*)buf + 6087808);
    bfloat16_t *dtensor10002196 = (bfloat16_t*)((char*)buf + 6472960);
    bfloat16_t *dtensor10002197 = (bfloat16_t*)((char*)buf + 6735104);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002197, dtensor10002184, dtensor10002196);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002054 = (bfloat16_t*)input_tensors.at(30);
    bfloat16_t *dtensor10002198 = (bfloat16_t*)((char*)buf + 6997248);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002198,dtensor10002024,dtensor10002054, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002054 = (bfloat16_t*)input_tensors.at(30);
    bfloat16_t *dtensor10002199 = (bfloat16_t*)((char*)buf + 7013632);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002199,dtensor10002066,dtensor10002054, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002055 = (bfloat16_t*)input_tensors.at(31);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002200 = (bfloat16_t*)((char*)buf + 7030016);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002200,dtensor10002055,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002198 = (bfloat16_t*)((char*)buf + 6997248);
    bfloat16_t *dtensor10002200 = (bfloat16_t*)((char*)buf + 7030016);
    bfloat16_t *dtensor10002201 = (bfloat16_t*)((char*)buf + 7046400);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002201,dtensor10002198,dtensor10002200, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002201 = (bfloat16_t*)((char*)buf + 7046400);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002202 = (bfloat16_t*)((char*)buf + 7054592);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002202, dtensor10002201, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002202 = (bfloat16_t*)((char*)buf + 7054592);
    bfloat16_t *dtensor10002203 = (bfloat16_t*)((char*)buf + 7062784);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002203, dtensor10002202);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002203 = (bfloat16_t*)((char*)buf + 7062784);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002204 = (bfloat16_t*)((char*)buf + 7070976);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002204,dtensor10002203,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002204 = (bfloat16_t*)((char*)buf + 7070976);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002205 = (bfloat16_t*)((char*)buf + 7071104);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002205,dtensor10002204,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002203 = (bfloat16_t*)((char*)buf + 7062784);
    bfloat16_t *dtensor10002205 = (bfloat16_t*)((char*)buf + 7071104);
    bfloat16_t *dtensor10002206 = (bfloat16_t*)((char*)buf + 7079296);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002206, dtensor10002203, dtensor10002205);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002054 = (bfloat16_t*)input_tensors.at(30);
    bfloat16_t *dtensor10002207 = (bfloat16_t*)((char*)buf + 7087488);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002207,dtensor10002068,dtensor10002054, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002206 = (bfloat16_t*)((char*)buf + 7079296);
    bfloat16_t *dtensor10002207 = (bfloat16_t*)((char*)buf + 7087488);
    bfloat16_t *dtensor10002208 = (bfloat16_t*)((char*)buf + 7103872);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002208,dtensor10002206,dtensor10002207, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002208 = (bfloat16_t*)((char*)buf + 7103872);
    bfloat16_t *dtensor10002055 = (bfloat16_t*)input_tensors.at(31);
    bfloat16_t *dtensor10002209 = (bfloat16_t*)((char*)buf + 7120256);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002209,dtensor10002208,dtensor10002055, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002197 = (bfloat16_t*)((char*)buf + 6735104);
    bfloat16_t *dtensor10002209 = (bfloat16_t*)((char*)buf + 7120256);
    bfloat16_t *dtensor10002210 = (bfloat16_t*)((char*)buf + 7382400);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002210, dtensor10002197, dtensor10002209);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002056 = (bfloat16_t*)input_tensors.at(32);
    bfloat16_t *dtensor10002211 = (bfloat16_t*)((char*)buf + 7644544);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002211,dtensor10002024,dtensor10002056, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002056 = (bfloat16_t*)input_tensors.at(32);
    bfloat16_t *dtensor10002212 = (bfloat16_t*)((char*)buf + 7660928);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002212,dtensor10002066,dtensor10002056, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002057 = (bfloat16_t*)input_tensors.at(33);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002213 = (bfloat16_t*)((char*)buf + 7677312);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002213,dtensor10002057,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002211 = (bfloat16_t*)((char*)buf + 7644544);
    bfloat16_t *dtensor10002213 = (bfloat16_t*)((char*)buf + 7677312);
    bfloat16_t *dtensor10002214 = (bfloat16_t*)((char*)buf + 7693696);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002214,dtensor10002211,dtensor10002213, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002214 = (bfloat16_t*)((char*)buf + 7693696);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002215 = (bfloat16_t*)((char*)buf + 7701888);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002215, dtensor10002214, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002215 = (bfloat16_t*)((char*)buf + 7701888);
    bfloat16_t *dtensor10002216 = (bfloat16_t*)((char*)buf + 7710080);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002216, dtensor10002215);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002216 = (bfloat16_t*)((char*)buf + 7710080);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002217 = (bfloat16_t*)((char*)buf + 7718272);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002217,dtensor10002216,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002217 = (bfloat16_t*)((char*)buf + 7718272);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002218 = (bfloat16_t*)((char*)buf + 7718400);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002218,dtensor10002217,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002216 = (bfloat16_t*)((char*)buf + 7710080);
    bfloat16_t *dtensor10002218 = (bfloat16_t*)((char*)buf + 7718400);
    bfloat16_t *dtensor10002219 = (bfloat16_t*)((char*)buf + 7726592);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002219, dtensor10002216, dtensor10002218);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002056 = (bfloat16_t*)input_tensors.at(32);
    bfloat16_t *dtensor10002220 = (bfloat16_t*)((char*)buf + 7734784);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002220,dtensor10002068,dtensor10002056, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002219 = (bfloat16_t*)((char*)buf + 7726592);
    bfloat16_t *dtensor10002220 = (bfloat16_t*)((char*)buf + 7734784);
    bfloat16_t *dtensor10002221 = (bfloat16_t*)((char*)buf + 7751168);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002221,dtensor10002219,dtensor10002220, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002221 = (bfloat16_t*)((char*)buf + 7751168);
    bfloat16_t *dtensor10002057 = (bfloat16_t*)input_tensors.at(33);
    bfloat16_t *dtensor10002222 = (bfloat16_t*)((char*)buf + 7767552);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002222,dtensor10002221,dtensor10002057, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002210 = (bfloat16_t*)((char*)buf + 7382400);
    bfloat16_t *dtensor10002222 = (bfloat16_t*)((char*)buf + 7767552);
    bfloat16_t *dtensor10002223 = (bfloat16_t*)((char*)buf + 8029696);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002223, dtensor10002210, dtensor10002222);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002058 = (bfloat16_t*)input_tensors.at(34);
    bfloat16_t *dtensor10002224 = (bfloat16_t*)((char*)buf + 8291840);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002224,dtensor10002024,dtensor10002058, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002058 = (bfloat16_t*)input_tensors.at(34);
    bfloat16_t *dtensor10002225 = (bfloat16_t*)((char*)buf + 8308224);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002225,dtensor10002066,dtensor10002058, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002059 = (bfloat16_t*)input_tensors.at(35);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002226 = (bfloat16_t*)((char*)buf + 8324608);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002226,dtensor10002059,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002224 = (bfloat16_t*)((char*)buf + 8291840);
    bfloat16_t *dtensor10002226 = (bfloat16_t*)((char*)buf + 8324608);
    bfloat16_t *dtensor10002227 = (bfloat16_t*)((char*)buf + 8340992);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002227,dtensor10002224,dtensor10002226, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002227 = (bfloat16_t*)((char*)buf + 8340992);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002228 = (bfloat16_t*)((char*)buf + 8349184);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002228, dtensor10002227, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002228 = (bfloat16_t*)((char*)buf + 8349184);
    bfloat16_t *dtensor10002229 = (bfloat16_t*)((char*)buf + 8357376);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002229, dtensor10002228);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002229 = (bfloat16_t*)((char*)buf + 8357376);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002230 = (bfloat16_t*)((char*)buf + 8365568);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002230,dtensor10002229,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002230 = (bfloat16_t*)((char*)buf + 8365568);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002231 = (bfloat16_t*)((char*)buf + 8365696);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002231,dtensor10002230,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002229 = (bfloat16_t*)((char*)buf + 8357376);
    bfloat16_t *dtensor10002231 = (bfloat16_t*)((char*)buf + 8365696);
    bfloat16_t *dtensor10002232 = (bfloat16_t*)((char*)buf + 8373888);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002232, dtensor10002229, dtensor10002231);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002058 = (bfloat16_t*)input_tensors.at(34);
    bfloat16_t *dtensor10002233 = (bfloat16_t*)((char*)buf + 8382080);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002233,dtensor10002068,dtensor10002058, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002232 = (bfloat16_t*)((char*)buf + 8373888);
    bfloat16_t *dtensor10002233 = (bfloat16_t*)((char*)buf + 8382080);
    bfloat16_t *dtensor10002234 = (bfloat16_t*)((char*)buf + 8398464);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002234,dtensor10002232,dtensor10002233, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002234 = (bfloat16_t*)((char*)buf + 8398464);
    bfloat16_t *dtensor10002059 = (bfloat16_t*)input_tensors.at(35);
    bfloat16_t *dtensor10002235 = (bfloat16_t*)((char*)buf + 8414848);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002235,dtensor10002234,dtensor10002059, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002223 = (bfloat16_t*)((char*)buf + 8029696);
    bfloat16_t *dtensor10002235 = (bfloat16_t*)((char*)buf + 8414848);
    bfloat16_t *dtensor10002236 = (bfloat16_t*)((char*)buf + 8676992);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002236, dtensor10002223, dtensor10002235);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002060 = (bfloat16_t*)input_tensors.at(36);
    bfloat16_t *dtensor10002237 = (bfloat16_t*)((char*)buf + 8939136);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002237,dtensor10002024,dtensor10002060, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002060 = (bfloat16_t*)input_tensors.at(36);
    bfloat16_t *dtensor10002238 = (bfloat16_t*)((char*)buf + 8955520);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002238,dtensor10002066,dtensor10002060, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002061 = (bfloat16_t*)input_tensors.at(37);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002239 = (bfloat16_t*)((char*)buf + 8971904);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002239,dtensor10002061,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002237 = (bfloat16_t*)((char*)buf + 8939136);
    bfloat16_t *dtensor10002239 = (bfloat16_t*)((char*)buf + 8971904);
    bfloat16_t *dtensor10002240 = (bfloat16_t*)((char*)buf + 8988288);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002240,dtensor10002237,dtensor10002239, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002240 = (bfloat16_t*)((char*)buf + 8988288);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002241 = (bfloat16_t*)((char*)buf + 8996480);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002241, dtensor10002240, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002241 = (bfloat16_t*)((char*)buf + 8996480);
    bfloat16_t *dtensor10002242 = (bfloat16_t*)((char*)buf + 9004672);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002242, dtensor10002241);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002242 = (bfloat16_t*)((char*)buf + 9004672);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002243 = (bfloat16_t*)((char*)buf + 9012864);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002243,dtensor10002242,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002243 = (bfloat16_t*)((char*)buf + 9012864);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002244 = (bfloat16_t*)((char*)buf + 9012992);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002244,dtensor10002243,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002242 = (bfloat16_t*)((char*)buf + 9004672);
    bfloat16_t *dtensor10002244 = (bfloat16_t*)((char*)buf + 9012992);
    bfloat16_t *dtensor10002245 = (bfloat16_t*)((char*)buf + 9021184);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002245, dtensor10002242, dtensor10002244);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002060 = (bfloat16_t*)input_tensors.at(36);
    bfloat16_t *dtensor10002246 = (bfloat16_t*)((char*)buf + 9029376);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002246,dtensor10002068,dtensor10002060, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002245 = (bfloat16_t*)((char*)buf + 9021184);
    bfloat16_t *dtensor10002246 = (bfloat16_t*)((char*)buf + 9029376);
    bfloat16_t *dtensor10002247 = (bfloat16_t*)((char*)buf + 9045760);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002247,dtensor10002245,dtensor10002246, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002247 = (bfloat16_t*)((char*)buf + 9045760);
    bfloat16_t *dtensor10002061 = (bfloat16_t*)input_tensors.at(37);
    bfloat16_t *dtensor10002248 = (bfloat16_t*)((char*)buf + 9062144);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002248,dtensor10002247,dtensor10002061, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002236 = (bfloat16_t*)((char*)buf + 8676992);
    bfloat16_t *dtensor10002248 = (bfloat16_t*)((char*)buf + 9062144);
    bfloat16_t *dtensor10002249 = (bfloat16_t*)((char*)buf + 9324288);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002249, dtensor10002236, dtensor10002248);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002062 = (bfloat16_t*)input_tensors.at(38);
    bfloat16_t *dtensor10002250 = (bfloat16_t*)((char*)buf + 9586432);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002250,dtensor10002024,dtensor10002062, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002062 = (bfloat16_t*)input_tensors.at(38);
    bfloat16_t *dtensor10002251 = (bfloat16_t*)((char*)buf + 9602816);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002251,dtensor10002066,dtensor10002062, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002063 = (bfloat16_t*)input_tensors.at(39);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002252 = (bfloat16_t*)((char*)buf + 9619200);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002252,dtensor10002063,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002250 = (bfloat16_t*)((char*)buf + 9586432);
    bfloat16_t *dtensor10002252 = (bfloat16_t*)((char*)buf + 9619200);
    bfloat16_t *dtensor10002253 = (bfloat16_t*)((char*)buf + 9635584);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002253,dtensor10002250,dtensor10002252, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002253 = (bfloat16_t*)((char*)buf + 9635584);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002254 = (bfloat16_t*)((char*)buf + 9643776);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002254, dtensor10002253, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002254 = (bfloat16_t*)((char*)buf + 9643776);
    bfloat16_t *dtensor10002255 = (bfloat16_t*)((char*)buf + 9651968);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002255, dtensor10002254);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002255 = (bfloat16_t*)((char*)buf + 9651968);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002256 = (bfloat16_t*)((char*)buf + 9660160);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002256,dtensor10002255,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002256 = (bfloat16_t*)((char*)buf + 9660160);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002257 = (bfloat16_t*)((char*)buf + 9660288);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002257,dtensor10002256,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002255 = (bfloat16_t*)((char*)buf + 9651968);
    bfloat16_t *dtensor10002257 = (bfloat16_t*)((char*)buf + 9660288);
    bfloat16_t *dtensor10002258 = (bfloat16_t*)((char*)buf + 9668480);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002258, dtensor10002255, dtensor10002257);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002062 = (bfloat16_t*)input_tensors.at(38);
    bfloat16_t *dtensor10002259 = (bfloat16_t*)((char*)buf + 9676672);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002259,dtensor10002068,dtensor10002062, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002258 = (bfloat16_t*)((char*)buf + 9668480);
    bfloat16_t *dtensor10002259 = (bfloat16_t*)((char*)buf + 9676672);
    bfloat16_t *dtensor10002260 = (bfloat16_t*)((char*)buf + 9693056);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002260,dtensor10002258,dtensor10002259, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002260 = (bfloat16_t*)((char*)buf + 9693056);
    bfloat16_t *dtensor10002063 = (bfloat16_t*)input_tensors.at(39);
    bfloat16_t *dtensor10002261 = (bfloat16_t*)((char*)buf + 9709440);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002261,dtensor10002260,dtensor10002063, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002249 = (bfloat16_t*)((char*)buf + 9324288);
    bfloat16_t *dtensor10002261 = (bfloat16_t*)((char*)buf + 9709440);
    bfloat16_t *dtensor10002262 = (bfloat16_t*)((char*)buf + 9971584);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002262, dtensor10002249, dtensor10002261);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002024 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002064 = (bfloat16_t*)input_tensors.at(40);
    bfloat16_t *dtensor10002263 = (bfloat16_t*)((char*)buf + 10233728);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002263,dtensor10002024,dtensor10002064, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002066 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002064 = (bfloat16_t*)input_tensors.at(40);
    bfloat16_t *dtensor10002264 = (bfloat16_t*)((char*)buf + 10250112);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002264,dtensor10002066,dtensor10002064, 64,128,2048, 1,64, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002065 = (bfloat16_t*)input_tensors.at(41);
    bfloat16_t *dtensor10002067 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10002265 = (bfloat16_t*)((char*)buf + 10266496);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002265,dtensor10002065,dtensor10002067, 128,64,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002263 = (bfloat16_t*)((char*)buf + 10233728);
    bfloat16_t *dtensor10002265 = (bfloat16_t*)((char*)buf + 10266496);
    bfloat16_t *dtensor10002266 = (bfloat16_t*)((char*)buf + 10282880);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002266,dtensor10002263,dtensor10002265, 64,64,128, 1,64, 1,128, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002266 = (bfloat16_t*)((char*)buf + 10282880);
    bfloat16_t *dtensor10002031 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002267 = (bfloat16_t*)((char*)buf + 10291072);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002267, dtensor10002266, dtensor10002031);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002267 = (bfloat16_t*)((char*)buf + 10291072);
    bfloat16_t *dtensor10002268 = (bfloat16_t*)((char*)buf + 10299264);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002268, dtensor10002267);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002268 = (bfloat16_t*)((char*)buf + 10299264);
    bfloat16_t *dtensor10002032 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002269 = (bfloat16_t*)((char*)buf + 10307456);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002269,dtensor10002268,dtensor10002032, 64,1,64, 64,1, 1,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002269 = (bfloat16_t*)((char*)buf + 10307456);
    bfloat16_t *dtensor10002033 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002270 = (bfloat16_t*)((char*)buf + 10307584);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002270,dtensor10002269,dtensor10002033, 64,64,1, 1,64, 64,1, 64,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002268 = (bfloat16_t*)((char*)buf + 10299264);
    bfloat16_t *dtensor10002270 = (bfloat16_t*)((char*)buf + 10307584);
    bfloat16_t *dtensor10002271 = (bfloat16_t*)((char*)buf + 10315776);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002271, dtensor10002268, dtensor10002270);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002068 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002064 = (bfloat16_t*)input_tensors.at(40);
    bfloat16_t *dtensor10002272 = (bfloat16_t*)((char*)buf + 10323968);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002272,dtensor10002068,dtensor10002064, 64,128,2048, 2048,1, 128,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002271 = (bfloat16_t*)((char*)buf + 10315776);
    bfloat16_t *dtensor10002272 = (bfloat16_t*)((char*)buf + 10323968);
    bfloat16_t *dtensor10002273 = (bfloat16_t*)((char*)buf + 10340352);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002273,dtensor10002271,dtensor10002272, 64,128,64, 64,1, 1,64, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002273 = (bfloat16_t*)((char*)buf + 10340352);
    bfloat16_t *dtensor10002065 = (bfloat16_t*)input_tensors.at(41);
    bfloat16_t *dtensor10002274 = (bfloat16_t*)((char*)buf + 10356736);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002274,dtensor10002273,dtensor10002065, 64,2048,128, 1,64, 2048,1, 1,64, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002262 = (bfloat16_t*)((char*)buf + 9971584);
    bfloat16_t *dtensor10002274 = (bfloat16_t*)((char*)buf + 10356736);
    bfloat16_t *dtensor10002275 = (bfloat16_t*)((char*)buf + 10618880);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>, Layout<Shape<Int<64>, Int<2048>>, Stride<Int<1>, Int<64>>>>;
    kernel::run(dtensor10002275, dtensor10002262, dtensor10002274);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002275 = (bfloat16_t*)((char*)buf + 10618880);
    bfloat16_t *dtensor10002028 = (bfloat16_t*)input_tensors.at(4);
    bfloat16_t *dtensor10002276 = (bfloat16_t*)output_tensors.at(0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002276,dtensor10002275,dtensor10002028, 64,1024,2048, 1,64, 1024,1, 1,64, 1, 0,0,0);
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

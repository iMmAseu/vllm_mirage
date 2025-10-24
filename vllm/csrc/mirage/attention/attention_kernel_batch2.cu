#define execute_mugraph attention_execute_mugraph_batch2
#define PyInit___mirage_launcher attention_PyInit_mirage_batch2
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
    bfloat16_t *dtensor10001266 = (bfloat16_t*)input_tensors.at(1);
    bfloat16_t *dtensor10001270 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001307,dtensor10001266,dtensor10001270, 32,2048,1024, 1024,1, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001271 = (bfloat16_t*)input_tensors.at(6);
    bfloat16_t *dtensor10001267 = (bfloat16_t*)input_tensors.at(2);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001308,dtensor10001271,dtensor10001267, 2048,32,1024, 1024,1, 32,1, 1,2048, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001268 = (bfloat16_t*)input_tensors.at(3);
    bfloat16_t *dtensor10001270 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001309,dtensor10001268,dtensor10001270, 32,2048,1024, 1024,1, 2048,1, 2048,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001275 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10001310 = (bfloat16_t*)((char*)buf + 393216);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001310,dtensor10001265,dtensor10001275, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001275 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10001311 = (bfloat16_t*)((char*)buf + 401408);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001311,dtensor10001307,dtensor10001275, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001276 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001312 = (bfloat16_t*)((char*)buf + 409600);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001312,dtensor10001276,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001310 = (bfloat16_t*)((char*)buf + 393216);
    bfloat16_t *dtensor10001312 = (bfloat16_t*)((char*)buf + 409600);
    bfloat16_t *dtensor10001313 = (bfloat16_t*)((char*)buf + 417792);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001313,dtensor10001310,dtensor10001312, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001313 = (bfloat16_t*)((char*)buf + 417792);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001314 = (bfloat16_t*)((char*)buf + 419840);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001314, dtensor10001313, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001314 = (bfloat16_t*)((char*)buf + 419840);
    bfloat16_t *dtensor10001315 = (bfloat16_t*)((char*)buf + 421888);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001315, dtensor10001314);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001315 = (bfloat16_t*)((char*)buf + 421888);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001316 = (bfloat16_t*)((char*)buf + 423936);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001316,dtensor10001315,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001316 = (bfloat16_t*)((char*)buf + 423936);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001317 = (bfloat16_t*)((char*)buf + 424064);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001317,dtensor10001316,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001315 = (bfloat16_t*)((char*)buf + 421888);
    bfloat16_t *dtensor10001317 = (bfloat16_t*)((char*)buf + 424064);
    bfloat16_t *dtensor10001318 = (bfloat16_t*)((char*)buf + 426112);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001318, dtensor10001315, dtensor10001317);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001275 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10001319 = (bfloat16_t*)((char*)buf + 428160);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001319,dtensor10001309,dtensor10001275, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001318 = (bfloat16_t*)((char*)buf + 426112);
    bfloat16_t *dtensor10001319 = (bfloat16_t*)((char*)buf + 428160);
    bfloat16_t *dtensor10001320 = (bfloat16_t*)((char*)buf + 436352);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001320,dtensor10001318,dtensor10001319, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001320 = (bfloat16_t*)((char*)buf + 436352);
    bfloat16_t *dtensor10001276 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10001321 = (bfloat16_t*)((char*)buf + 444544);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001321,dtensor10001320,dtensor10001276, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001277 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10001322 = (bfloat16_t*)((char*)buf + 575616);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001322,dtensor10001265,dtensor10001277, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001277 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10001323 = (bfloat16_t*)((char*)buf + 583808);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001323,dtensor10001307,dtensor10001277, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001278 = (bfloat16_t*)input_tensors.at(13);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001324 = (bfloat16_t*)((char*)buf + 592000);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001324,dtensor10001278,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001322 = (bfloat16_t*)((char*)buf + 575616);
    bfloat16_t *dtensor10001324 = (bfloat16_t*)((char*)buf + 592000);
    bfloat16_t *dtensor10001325 = (bfloat16_t*)((char*)buf + 600192);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001325,dtensor10001322,dtensor10001324, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001325 = (bfloat16_t*)((char*)buf + 600192);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001326 = (bfloat16_t*)((char*)buf + 602240);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001326, dtensor10001325, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001326 = (bfloat16_t*)((char*)buf + 602240);
    bfloat16_t *dtensor10001327 = (bfloat16_t*)((char*)buf + 604288);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001327, dtensor10001326);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001327 = (bfloat16_t*)((char*)buf + 604288);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001328 = (bfloat16_t*)((char*)buf + 606336);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001328,dtensor10001327,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001328 = (bfloat16_t*)((char*)buf + 606336);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001329 = (bfloat16_t*)((char*)buf + 606464);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001329,dtensor10001328,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001327 = (bfloat16_t*)((char*)buf + 604288);
    bfloat16_t *dtensor10001329 = (bfloat16_t*)((char*)buf + 606464);
    bfloat16_t *dtensor10001330 = (bfloat16_t*)((char*)buf + 608512);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001330, dtensor10001327, dtensor10001329);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001277 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10001331 = (bfloat16_t*)((char*)buf + 610560);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001331,dtensor10001309,dtensor10001277, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001330 = (bfloat16_t*)((char*)buf + 608512);
    bfloat16_t *dtensor10001331 = (bfloat16_t*)((char*)buf + 610560);
    bfloat16_t *dtensor10001332 = (bfloat16_t*)((char*)buf + 618752);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001332,dtensor10001330,dtensor10001331, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001332 = (bfloat16_t*)((char*)buf + 618752);
    bfloat16_t *dtensor10001278 = (bfloat16_t*)input_tensors.at(13);
    bfloat16_t *dtensor10001333 = (bfloat16_t*)((char*)buf + 626944);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001333,dtensor10001332,dtensor10001278, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10001321 = (bfloat16_t*)((char*)buf + 444544);
    bfloat16_t *dtensor10001333 = (bfloat16_t*)((char*)buf + 626944);
    bfloat16_t *dtensor10001334 = (bfloat16_t*)((char*)buf + 758016);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001334, dtensor10001321, dtensor10001333);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001279 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10001335 = (bfloat16_t*)((char*)buf + 889088);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001335,dtensor10001265,dtensor10001279, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001279 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10001336 = (bfloat16_t*)((char*)buf + 897280);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001336,dtensor10001307,dtensor10001279, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001280 = (bfloat16_t*)input_tensors.at(15);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001337 = (bfloat16_t*)((char*)buf + 905472);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001337,dtensor10001280,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001335 = (bfloat16_t*)((char*)buf + 889088);
    bfloat16_t *dtensor10001337 = (bfloat16_t*)((char*)buf + 905472);
    bfloat16_t *dtensor10001338 = (bfloat16_t*)((char*)buf + 913664);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001338,dtensor10001335,dtensor10001337, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001338 = (bfloat16_t*)((char*)buf + 913664);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001339 = (bfloat16_t*)((char*)buf + 915712);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001339, dtensor10001338, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001339 = (bfloat16_t*)((char*)buf + 915712);
    bfloat16_t *dtensor10001340 = (bfloat16_t*)((char*)buf + 917760);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001340, dtensor10001339);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001340 = (bfloat16_t*)((char*)buf + 917760);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001341 = (bfloat16_t*)((char*)buf + 919808);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001341,dtensor10001340,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001341 = (bfloat16_t*)((char*)buf + 919808);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001342 = (bfloat16_t*)((char*)buf + 919936);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001342,dtensor10001341,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001340 = (bfloat16_t*)((char*)buf + 917760);
    bfloat16_t *dtensor10001342 = (bfloat16_t*)((char*)buf + 919936);
    bfloat16_t *dtensor10001343 = (bfloat16_t*)((char*)buf + 921984);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001343, dtensor10001340, dtensor10001342);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001279 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10001344 = (bfloat16_t*)((char*)buf + 924032);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001344,dtensor10001309,dtensor10001279, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001343 = (bfloat16_t*)((char*)buf + 921984);
    bfloat16_t *dtensor10001344 = (bfloat16_t*)((char*)buf + 924032);
    bfloat16_t *dtensor10001345 = (bfloat16_t*)((char*)buf + 932224);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001345,dtensor10001343,dtensor10001344, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001345 = (bfloat16_t*)((char*)buf + 932224);
    bfloat16_t *dtensor10001280 = (bfloat16_t*)input_tensors.at(15);
    bfloat16_t *dtensor10001346 = (bfloat16_t*)((char*)buf + 940416);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001346,dtensor10001345,dtensor10001280, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10001334 = (bfloat16_t*)((char*)buf + 758016);
    bfloat16_t *dtensor10001346 = (bfloat16_t*)((char*)buf + 940416);
    bfloat16_t *dtensor10001347 = (bfloat16_t*)((char*)buf + 1071488);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001347, dtensor10001334, dtensor10001346);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001281 = (bfloat16_t*)input_tensors.at(16);
    bfloat16_t *dtensor10001348 = (bfloat16_t*)((char*)buf + 1202560);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001348,dtensor10001265,dtensor10001281, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001281 = (bfloat16_t*)input_tensors.at(16);
    bfloat16_t *dtensor10001349 = (bfloat16_t*)((char*)buf + 1210752);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001349,dtensor10001307,dtensor10001281, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001282 = (bfloat16_t*)input_tensors.at(17);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001350 = (bfloat16_t*)((char*)buf + 1218944);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001350,dtensor10001282,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001348 = (bfloat16_t*)((char*)buf + 1202560);
    bfloat16_t *dtensor10001350 = (bfloat16_t*)((char*)buf + 1218944);
    bfloat16_t *dtensor10001351 = (bfloat16_t*)((char*)buf + 1227136);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001351,dtensor10001348,dtensor10001350, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001351 = (bfloat16_t*)((char*)buf + 1227136);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001352 = (bfloat16_t*)((char*)buf + 1229184);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001352, dtensor10001351, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001352 = (bfloat16_t*)((char*)buf + 1229184);
    bfloat16_t *dtensor10001353 = (bfloat16_t*)((char*)buf + 1231232);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001353, dtensor10001352);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001353 = (bfloat16_t*)((char*)buf + 1231232);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001354 = (bfloat16_t*)((char*)buf + 1233280);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001354,dtensor10001353,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001354 = (bfloat16_t*)((char*)buf + 1233280);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001355 = (bfloat16_t*)((char*)buf + 1233408);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001355,dtensor10001354,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001353 = (bfloat16_t*)((char*)buf + 1231232);
    bfloat16_t *dtensor10001355 = (bfloat16_t*)((char*)buf + 1233408);
    bfloat16_t *dtensor10001356 = (bfloat16_t*)((char*)buf + 1235456);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001356, dtensor10001353, dtensor10001355);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001281 = (bfloat16_t*)input_tensors.at(16);
    bfloat16_t *dtensor10001357 = (bfloat16_t*)((char*)buf + 1237504);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001357,dtensor10001309,dtensor10001281, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001356 = (bfloat16_t*)((char*)buf + 1235456);
    bfloat16_t *dtensor10001357 = (bfloat16_t*)((char*)buf + 1237504);
    bfloat16_t *dtensor10001358 = (bfloat16_t*)((char*)buf + 1245696);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001358,dtensor10001356,dtensor10001357, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001358 = (bfloat16_t*)((char*)buf + 1245696);
    bfloat16_t *dtensor10001282 = (bfloat16_t*)input_tensors.at(17);
    bfloat16_t *dtensor10001359 = (bfloat16_t*)((char*)buf + 1253888);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001359,dtensor10001358,dtensor10001282, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10001347 = (bfloat16_t*)((char*)buf + 1071488);
    bfloat16_t *dtensor10001359 = (bfloat16_t*)((char*)buf + 1253888);
    bfloat16_t *dtensor10001360 = (bfloat16_t*)((char*)buf + 1384960);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001360, dtensor10001347, dtensor10001359);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001283 = (bfloat16_t*)input_tensors.at(18);
    bfloat16_t *dtensor10001361 = (bfloat16_t*)((char*)buf + 1516032);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001361,dtensor10001265,dtensor10001283, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001283 = (bfloat16_t*)input_tensors.at(18);
    bfloat16_t *dtensor10001362 = (bfloat16_t*)((char*)buf + 1524224);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001362,dtensor10001307,dtensor10001283, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001284 = (bfloat16_t*)input_tensors.at(19);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001363 = (bfloat16_t*)((char*)buf + 1532416);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001363,dtensor10001284,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001361 = (bfloat16_t*)((char*)buf + 1516032);
    bfloat16_t *dtensor10001363 = (bfloat16_t*)((char*)buf + 1532416);
    bfloat16_t *dtensor10001364 = (bfloat16_t*)((char*)buf + 1540608);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001364,dtensor10001361,dtensor10001363, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001364 = (bfloat16_t*)((char*)buf + 1540608);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001365 = (bfloat16_t*)((char*)buf + 1542656);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001365, dtensor10001364, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001365 = (bfloat16_t*)((char*)buf + 1542656);
    bfloat16_t *dtensor10001366 = (bfloat16_t*)((char*)buf + 1544704);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001366, dtensor10001365);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001366 = (bfloat16_t*)((char*)buf + 1544704);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001367 = (bfloat16_t*)((char*)buf + 1546752);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001367,dtensor10001366,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001367 = (bfloat16_t*)((char*)buf + 1546752);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001368 = (bfloat16_t*)((char*)buf + 1546880);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001368,dtensor10001367,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001366 = (bfloat16_t*)((char*)buf + 1544704);
    bfloat16_t *dtensor10001368 = (bfloat16_t*)((char*)buf + 1546880);
    bfloat16_t *dtensor10001369 = (bfloat16_t*)((char*)buf + 1548928);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001369, dtensor10001366, dtensor10001368);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001283 = (bfloat16_t*)input_tensors.at(18);
    bfloat16_t *dtensor10001370 = (bfloat16_t*)((char*)buf + 1550976);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001370,dtensor10001309,dtensor10001283, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001369 = (bfloat16_t*)((char*)buf + 1548928);
    bfloat16_t *dtensor10001370 = (bfloat16_t*)((char*)buf + 1550976);
    bfloat16_t *dtensor10001371 = (bfloat16_t*)((char*)buf + 1559168);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001371,dtensor10001369,dtensor10001370, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001371 = (bfloat16_t*)((char*)buf + 1559168);
    bfloat16_t *dtensor10001284 = (bfloat16_t*)input_tensors.at(19);
    bfloat16_t *dtensor10001372 = (bfloat16_t*)((char*)buf + 1567360);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001372,dtensor10001371,dtensor10001284, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10001360 = (bfloat16_t*)((char*)buf + 1384960);
    bfloat16_t *dtensor10001372 = (bfloat16_t*)((char*)buf + 1567360);
    bfloat16_t *dtensor10001373 = (bfloat16_t*)((char*)buf + 1698432);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001373, dtensor10001360, dtensor10001372);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001285 = (bfloat16_t*)input_tensors.at(20);
    bfloat16_t *dtensor10001374 = (bfloat16_t*)((char*)buf + 1829504);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001374,dtensor10001265,dtensor10001285, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001285 = (bfloat16_t*)input_tensors.at(20);
    bfloat16_t *dtensor10001375 = (bfloat16_t*)((char*)buf + 1837696);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001375,dtensor10001307,dtensor10001285, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001286 = (bfloat16_t*)input_tensors.at(21);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001376 = (bfloat16_t*)((char*)buf + 1845888);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001376,dtensor10001286,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001374 = (bfloat16_t*)((char*)buf + 1829504);
    bfloat16_t *dtensor10001376 = (bfloat16_t*)((char*)buf + 1845888);
    bfloat16_t *dtensor10001377 = (bfloat16_t*)((char*)buf + 1854080);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001377,dtensor10001374,dtensor10001376, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001377 = (bfloat16_t*)((char*)buf + 1854080);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001378 = (bfloat16_t*)((char*)buf + 1856128);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001378, dtensor10001377, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001378 = (bfloat16_t*)((char*)buf + 1856128);
    bfloat16_t *dtensor10001379 = (bfloat16_t*)((char*)buf + 1858176);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001379, dtensor10001378);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001379 = (bfloat16_t*)((char*)buf + 1858176);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001380 = (bfloat16_t*)((char*)buf + 1860224);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001380,dtensor10001379,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001380 = (bfloat16_t*)((char*)buf + 1860224);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001381 = (bfloat16_t*)((char*)buf + 1860352);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001381,dtensor10001380,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001379 = (bfloat16_t*)((char*)buf + 1858176);
    bfloat16_t *dtensor10001381 = (bfloat16_t*)((char*)buf + 1860352);
    bfloat16_t *dtensor10001382 = (bfloat16_t*)((char*)buf + 1862400);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001382, dtensor10001379, dtensor10001381);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001285 = (bfloat16_t*)input_tensors.at(20);
    bfloat16_t *dtensor10001383 = (bfloat16_t*)((char*)buf + 1864448);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001383,dtensor10001309,dtensor10001285, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001382 = (bfloat16_t*)((char*)buf + 1862400);
    bfloat16_t *dtensor10001383 = (bfloat16_t*)((char*)buf + 1864448);
    bfloat16_t *dtensor10001384 = (bfloat16_t*)((char*)buf + 1872640);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001384,dtensor10001382,dtensor10001383, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001384 = (bfloat16_t*)((char*)buf + 1872640);
    bfloat16_t *dtensor10001286 = (bfloat16_t*)input_tensors.at(21);
    bfloat16_t *dtensor10001385 = (bfloat16_t*)((char*)buf + 1880832);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001385,dtensor10001384,dtensor10001286, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10001373 = (bfloat16_t*)((char*)buf + 1698432);
    bfloat16_t *dtensor10001385 = (bfloat16_t*)((char*)buf + 1880832);
    bfloat16_t *dtensor10001386 = (bfloat16_t*)((char*)buf + 2011904);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001386, dtensor10001373, dtensor10001385);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001287 = (bfloat16_t*)input_tensors.at(22);
    bfloat16_t *dtensor10001387 = (bfloat16_t*)((char*)buf + 2142976);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001387,dtensor10001265,dtensor10001287, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001287 = (bfloat16_t*)input_tensors.at(22);
    bfloat16_t *dtensor10001388 = (bfloat16_t*)((char*)buf + 2151168);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001388,dtensor10001307,dtensor10001287, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001288 = (bfloat16_t*)input_tensors.at(23);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001389 = (bfloat16_t*)((char*)buf + 2159360);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001389,dtensor10001288,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001387 = (bfloat16_t*)((char*)buf + 2142976);
    bfloat16_t *dtensor10001389 = (bfloat16_t*)((char*)buf + 2159360);
    bfloat16_t *dtensor10001390 = (bfloat16_t*)((char*)buf + 2167552);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001390,dtensor10001387,dtensor10001389, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001390 = (bfloat16_t*)((char*)buf + 2167552);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001391 = (bfloat16_t*)((char*)buf + 2169600);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001391, dtensor10001390, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001391 = (bfloat16_t*)((char*)buf + 2169600);
    bfloat16_t *dtensor10001392 = (bfloat16_t*)((char*)buf + 2171648);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001392, dtensor10001391);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001392 = (bfloat16_t*)((char*)buf + 2171648);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001393 = (bfloat16_t*)((char*)buf + 2173696);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001393,dtensor10001392,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001393 = (bfloat16_t*)((char*)buf + 2173696);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001394 = (bfloat16_t*)((char*)buf + 2173824);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001394,dtensor10001393,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001392 = (bfloat16_t*)((char*)buf + 2171648);
    bfloat16_t *dtensor10001394 = (bfloat16_t*)((char*)buf + 2173824);
    bfloat16_t *dtensor10001395 = (bfloat16_t*)((char*)buf + 2175872);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001395, dtensor10001392, dtensor10001394);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001287 = (bfloat16_t*)input_tensors.at(22);
    bfloat16_t *dtensor10001396 = (bfloat16_t*)((char*)buf + 2177920);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001396,dtensor10001309,dtensor10001287, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001395 = (bfloat16_t*)((char*)buf + 2175872);
    bfloat16_t *dtensor10001396 = (bfloat16_t*)((char*)buf + 2177920);
    bfloat16_t *dtensor10001397 = (bfloat16_t*)((char*)buf + 2186112);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001397,dtensor10001395,dtensor10001396, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001397 = (bfloat16_t*)((char*)buf + 2186112);
    bfloat16_t *dtensor10001288 = (bfloat16_t*)input_tensors.at(23);
    bfloat16_t *dtensor10001398 = (bfloat16_t*)((char*)buf + 2194304);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001398,dtensor10001397,dtensor10001288, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10001386 = (bfloat16_t*)((char*)buf + 2011904);
    bfloat16_t *dtensor10001398 = (bfloat16_t*)((char*)buf + 2194304);
    bfloat16_t *dtensor10001399 = (bfloat16_t*)((char*)buf + 2325376);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001399, dtensor10001386, dtensor10001398);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001289 = (bfloat16_t*)input_tensors.at(24);
    bfloat16_t *dtensor10001400 = (bfloat16_t*)((char*)buf + 2456448);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001400,dtensor10001265,dtensor10001289, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001289 = (bfloat16_t*)input_tensors.at(24);
    bfloat16_t *dtensor10001401 = (bfloat16_t*)((char*)buf + 2464640);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001401,dtensor10001307,dtensor10001289, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001290 = (bfloat16_t*)input_tensors.at(25);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001402 = (bfloat16_t*)((char*)buf + 2472832);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001402,dtensor10001290,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001400 = (bfloat16_t*)((char*)buf + 2456448);
    bfloat16_t *dtensor10001402 = (bfloat16_t*)((char*)buf + 2472832);
    bfloat16_t *dtensor10001403 = (bfloat16_t*)((char*)buf + 2481024);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001403,dtensor10001400,dtensor10001402, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001403 = (bfloat16_t*)((char*)buf + 2481024);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001404 = (bfloat16_t*)((char*)buf + 2483072);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001404, dtensor10001403, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001404 = (bfloat16_t*)((char*)buf + 2483072);
    bfloat16_t *dtensor10001405 = (bfloat16_t*)((char*)buf + 2485120);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001405, dtensor10001404);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001405 = (bfloat16_t*)((char*)buf + 2485120);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001406 = (bfloat16_t*)((char*)buf + 2487168);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001406,dtensor10001405,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001406 = (bfloat16_t*)((char*)buf + 2487168);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001407 = (bfloat16_t*)((char*)buf + 2487296);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001407,dtensor10001406,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001405 = (bfloat16_t*)((char*)buf + 2485120);
    bfloat16_t *dtensor10001407 = (bfloat16_t*)((char*)buf + 2487296);
    bfloat16_t *dtensor10001408 = (bfloat16_t*)((char*)buf + 2489344);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001408, dtensor10001405, dtensor10001407);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001289 = (bfloat16_t*)input_tensors.at(24);
    bfloat16_t *dtensor10001409 = (bfloat16_t*)((char*)buf + 2491392);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001409,dtensor10001309,dtensor10001289, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001408 = (bfloat16_t*)((char*)buf + 2489344);
    bfloat16_t *dtensor10001409 = (bfloat16_t*)((char*)buf + 2491392);
    bfloat16_t *dtensor10001410 = (bfloat16_t*)((char*)buf + 2499584);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001410,dtensor10001408,dtensor10001409, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001410 = (bfloat16_t*)((char*)buf + 2499584);
    bfloat16_t *dtensor10001290 = (bfloat16_t*)input_tensors.at(25);
    bfloat16_t *dtensor10001411 = (bfloat16_t*)((char*)buf + 2507776);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001411,dtensor10001410,dtensor10001290, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10001399 = (bfloat16_t*)((char*)buf + 2325376);
    bfloat16_t *dtensor10001411 = (bfloat16_t*)((char*)buf + 2507776);
    bfloat16_t *dtensor10001412 = (bfloat16_t*)((char*)buf + 2638848);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001412, dtensor10001399, dtensor10001411);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001291 = (bfloat16_t*)input_tensors.at(26);
    bfloat16_t *dtensor10001413 = (bfloat16_t*)((char*)buf + 2769920);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001413,dtensor10001265,dtensor10001291, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001291 = (bfloat16_t*)input_tensors.at(26);
    bfloat16_t *dtensor10001414 = (bfloat16_t*)((char*)buf + 2778112);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001414,dtensor10001307,dtensor10001291, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001292 = (bfloat16_t*)input_tensors.at(27);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001415 = (bfloat16_t*)((char*)buf + 2786304);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001415,dtensor10001292,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001413 = (bfloat16_t*)((char*)buf + 2769920);
    bfloat16_t *dtensor10001415 = (bfloat16_t*)((char*)buf + 2786304);
    bfloat16_t *dtensor10001416 = (bfloat16_t*)((char*)buf + 2794496);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001416,dtensor10001413,dtensor10001415, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001416 = (bfloat16_t*)((char*)buf + 2794496);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001417 = (bfloat16_t*)((char*)buf + 2796544);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001417, dtensor10001416, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001417 = (bfloat16_t*)((char*)buf + 2796544);
    bfloat16_t *dtensor10001418 = (bfloat16_t*)((char*)buf + 2798592);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001418, dtensor10001417);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001418 = (bfloat16_t*)((char*)buf + 2798592);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001419 = (bfloat16_t*)((char*)buf + 2800640);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001419,dtensor10001418,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001419 = (bfloat16_t*)((char*)buf + 2800640);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001420 = (bfloat16_t*)((char*)buf + 2800768);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001420,dtensor10001419,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001418 = (bfloat16_t*)((char*)buf + 2798592);
    bfloat16_t *dtensor10001420 = (bfloat16_t*)((char*)buf + 2800768);
    bfloat16_t *dtensor10001421 = (bfloat16_t*)((char*)buf + 2802816);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001421, dtensor10001418, dtensor10001420);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001291 = (bfloat16_t*)input_tensors.at(26);
    bfloat16_t *dtensor10001422 = (bfloat16_t*)((char*)buf + 2804864);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001422,dtensor10001309,dtensor10001291, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001421 = (bfloat16_t*)((char*)buf + 2802816);
    bfloat16_t *dtensor10001422 = (bfloat16_t*)((char*)buf + 2804864);
    bfloat16_t *dtensor10001423 = (bfloat16_t*)((char*)buf + 2813056);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001423,dtensor10001421,dtensor10001422, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001423 = (bfloat16_t*)((char*)buf + 2813056);
    bfloat16_t *dtensor10001292 = (bfloat16_t*)input_tensors.at(27);
    bfloat16_t *dtensor10001424 = (bfloat16_t*)((char*)buf + 2821248);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001424,dtensor10001423,dtensor10001292, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10001412 = (bfloat16_t*)((char*)buf + 2638848);
    bfloat16_t *dtensor10001424 = (bfloat16_t*)((char*)buf + 2821248);
    bfloat16_t *dtensor10001425 = (bfloat16_t*)((char*)buf + 2952320);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001425, dtensor10001412, dtensor10001424);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001293 = (bfloat16_t*)input_tensors.at(28);
    bfloat16_t *dtensor10001426 = (bfloat16_t*)((char*)buf + 3083392);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001426,dtensor10001265,dtensor10001293, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001293 = (bfloat16_t*)input_tensors.at(28);
    bfloat16_t *dtensor10001427 = (bfloat16_t*)((char*)buf + 3091584);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001427,dtensor10001307,dtensor10001293, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001294 = (bfloat16_t*)input_tensors.at(29);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001428 = (bfloat16_t*)((char*)buf + 3099776);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001428,dtensor10001294,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001426 = (bfloat16_t*)((char*)buf + 3083392);
    bfloat16_t *dtensor10001428 = (bfloat16_t*)((char*)buf + 3099776);
    bfloat16_t *dtensor10001429 = (bfloat16_t*)((char*)buf + 3107968);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001429,dtensor10001426,dtensor10001428, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001429 = (bfloat16_t*)((char*)buf + 3107968);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001430 = (bfloat16_t*)((char*)buf + 3110016);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001430, dtensor10001429, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001430 = (bfloat16_t*)((char*)buf + 3110016);
    bfloat16_t *dtensor10001431 = (bfloat16_t*)((char*)buf + 3112064);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001431, dtensor10001430);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001431 = (bfloat16_t*)((char*)buf + 3112064);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001432 = (bfloat16_t*)((char*)buf + 3114112);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001432,dtensor10001431,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001432 = (bfloat16_t*)((char*)buf + 3114112);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001433 = (bfloat16_t*)((char*)buf + 3114240);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001433,dtensor10001432,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001431 = (bfloat16_t*)((char*)buf + 3112064);
    bfloat16_t *dtensor10001433 = (bfloat16_t*)((char*)buf + 3114240);
    bfloat16_t *dtensor10001434 = (bfloat16_t*)((char*)buf + 3116288);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001434, dtensor10001431, dtensor10001433);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001293 = (bfloat16_t*)input_tensors.at(28);
    bfloat16_t *dtensor10001435 = (bfloat16_t*)((char*)buf + 3118336);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001435,dtensor10001309,dtensor10001293, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001434 = (bfloat16_t*)((char*)buf + 3116288);
    bfloat16_t *dtensor10001435 = (bfloat16_t*)((char*)buf + 3118336);
    bfloat16_t *dtensor10001436 = (bfloat16_t*)((char*)buf + 3126528);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001436,dtensor10001434,dtensor10001435, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001436 = (bfloat16_t*)((char*)buf + 3126528);
    bfloat16_t *dtensor10001294 = (bfloat16_t*)input_tensors.at(29);
    bfloat16_t *dtensor10001437 = (bfloat16_t*)((char*)buf + 3134720);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001437,dtensor10001436,dtensor10001294, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10001425 = (bfloat16_t*)((char*)buf + 2952320);
    bfloat16_t *dtensor10001437 = (bfloat16_t*)((char*)buf + 3134720);
    bfloat16_t *dtensor10001438 = (bfloat16_t*)((char*)buf + 3265792);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001438, dtensor10001425, dtensor10001437);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001295 = (bfloat16_t*)input_tensors.at(30);
    bfloat16_t *dtensor10001439 = (bfloat16_t*)((char*)buf + 3396864);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001439,dtensor10001265,dtensor10001295, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001295 = (bfloat16_t*)input_tensors.at(30);
    bfloat16_t *dtensor10001440 = (bfloat16_t*)((char*)buf + 3405056);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001440,dtensor10001307,dtensor10001295, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001296 = (bfloat16_t*)input_tensors.at(31);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001441 = (bfloat16_t*)((char*)buf + 3413248);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001441,dtensor10001296,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001439 = (bfloat16_t*)((char*)buf + 3396864);
    bfloat16_t *dtensor10001441 = (bfloat16_t*)((char*)buf + 3413248);
    bfloat16_t *dtensor10001442 = (bfloat16_t*)((char*)buf + 3421440);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001442,dtensor10001439,dtensor10001441, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001442 = (bfloat16_t*)((char*)buf + 3421440);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001443 = (bfloat16_t*)((char*)buf + 3423488);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001443, dtensor10001442, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001443 = (bfloat16_t*)((char*)buf + 3423488);
    bfloat16_t *dtensor10001444 = (bfloat16_t*)((char*)buf + 3425536);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001444, dtensor10001443);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001444 = (bfloat16_t*)((char*)buf + 3425536);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001445 = (bfloat16_t*)((char*)buf + 3427584);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001445,dtensor10001444,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001445 = (bfloat16_t*)((char*)buf + 3427584);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001446 = (bfloat16_t*)((char*)buf + 3427712);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001446,dtensor10001445,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001444 = (bfloat16_t*)((char*)buf + 3425536);
    bfloat16_t *dtensor10001446 = (bfloat16_t*)((char*)buf + 3427712);
    bfloat16_t *dtensor10001447 = (bfloat16_t*)((char*)buf + 3429760);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001447, dtensor10001444, dtensor10001446);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001295 = (bfloat16_t*)input_tensors.at(30);
    bfloat16_t *dtensor10001448 = (bfloat16_t*)((char*)buf + 3431808);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001448,dtensor10001309,dtensor10001295, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001447 = (bfloat16_t*)((char*)buf + 3429760);
    bfloat16_t *dtensor10001448 = (bfloat16_t*)((char*)buf + 3431808);
    bfloat16_t *dtensor10001449 = (bfloat16_t*)((char*)buf + 3440000);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001449,dtensor10001447,dtensor10001448, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001449 = (bfloat16_t*)((char*)buf + 3440000);
    bfloat16_t *dtensor10001296 = (bfloat16_t*)input_tensors.at(31);
    bfloat16_t *dtensor10001450 = (bfloat16_t*)((char*)buf + 3448192);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001450,dtensor10001449,dtensor10001296, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10001438 = (bfloat16_t*)((char*)buf + 3265792);
    bfloat16_t *dtensor10001450 = (bfloat16_t*)((char*)buf + 3448192);
    bfloat16_t *dtensor10001451 = (bfloat16_t*)((char*)buf + 3579264);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001451, dtensor10001438, dtensor10001450);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001297 = (bfloat16_t*)input_tensors.at(32);
    bfloat16_t *dtensor10001452 = (bfloat16_t*)((char*)buf + 3710336);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001452,dtensor10001265,dtensor10001297, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001297 = (bfloat16_t*)input_tensors.at(32);
    bfloat16_t *dtensor10001453 = (bfloat16_t*)((char*)buf + 3718528);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001453,dtensor10001307,dtensor10001297, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001298 = (bfloat16_t*)input_tensors.at(33);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001454 = (bfloat16_t*)((char*)buf + 3726720);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001454,dtensor10001298,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001452 = (bfloat16_t*)((char*)buf + 3710336);
    bfloat16_t *dtensor10001454 = (bfloat16_t*)((char*)buf + 3726720);
    bfloat16_t *dtensor10001455 = (bfloat16_t*)((char*)buf + 3734912);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001455,dtensor10001452,dtensor10001454, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001455 = (bfloat16_t*)((char*)buf + 3734912);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001456 = (bfloat16_t*)((char*)buf + 3736960);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001456, dtensor10001455, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001456 = (bfloat16_t*)((char*)buf + 3736960);
    bfloat16_t *dtensor10001457 = (bfloat16_t*)((char*)buf + 3739008);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001457, dtensor10001456);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001457 = (bfloat16_t*)((char*)buf + 3739008);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001458 = (bfloat16_t*)((char*)buf + 3741056);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001458,dtensor10001457,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001458 = (bfloat16_t*)((char*)buf + 3741056);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001459 = (bfloat16_t*)((char*)buf + 3741184);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001459,dtensor10001458,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001457 = (bfloat16_t*)((char*)buf + 3739008);
    bfloat16_t *dtensor10001459 = (bfloat16_t*)((char*)buf + 3741184);
    bfloat16_t *dtensor10001460 = (bfloat16_t*)((char*)buf + 3743232);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001460, dtensor10001457, dtensor10001459);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001297 = (bfloat16_t*)input_tensors.at(32);
    bfloat16_t *dtensor10001461 = (bfloat16_t*)((char*)buf + 3745280);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001461,dtensor10001309,dtensor10001297, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001460 = (bfloat16_t*)((char*)buf + 3743232);
    bfloat16_t *dtensor10001461 = (bfloat16_t*)((char*)buf + 3745280);
    bfloat16_t *dtensor10001462 = (bfloat16_t*)((char*)buf + 3753472);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001462,dtensor10001460,dtensor10001461, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001462 = (bfloat16_t*)((char*)buf + 3753472);
    bfloat16_t *dtensor10001298 = (bfloat16_t*)input_tensors.at(33);
    bfloat16_t *dtensor10001463 = (bfloat16_t*)((char*)buf + 3761664);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001463,dtensor10001462,dtensor10001298, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10001451 = (bfloat16_t*)((char*)buf + 3579264);
    bfloat16_t *dtensor10001463 = (bfloat16_t*)((char*)buf + 3761664);
    bfloat16_t *dtensor10001464 = (bfloat16_t*)((char*)buf + 3892736);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001464, dtensor10001451, dtensor10001463);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001299 = (bfloat16_t*)input_tensors.at(34);
    bfloat16_t *dtensor10001465 = (bfloat16_t*)((char*)buf + 4023808);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001465,dtensor10001265,dtensor10001299, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001299 = (bfloat16_t*)input_tensors.at(34);
    bfloat16_t *dtensor10001466 = (bfloat16_t*)((char*)buf + 4032000);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001466,dtensor10001307,dtensor10001299, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001300 = (bfloat16_t*)input_tensors.at(35);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001467 = (bfloat16_t*)((char*)buf + 4040192);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001467,dtensor10001300,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001465 = (bfloat16_t*)((char*)buf + 4023808);
    bfloat16_t *dtensor10001467 = (bfloat16_t*)((char*)buf + 4040192);
    bfloat16_t *dtensor10001468 = (bfloat16_t*)((char*)buf + 4048384);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001468,dtensor10001465,dtensor10001467, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001468 = (bfloat16_t*)((char*)buf + 4048384);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001469 = (bfloat16_t*)((char*)buf + 4050432);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001469, dtensor10001468, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001469 = (bfloat16_t*)((char*)buf + 4050432);
    bfloat16_t *dtensor10001470 = (bfloat16_t*)((char*)buf + 4052480);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001470, dtensor10001469);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001470 = (bfloat16_t*)((char*)buf + 4052480);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001471 = (bfloat16_t*)((char*)buf + 4054528);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001471,dtensor10001470,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001471 = (bfloat16_t*)((char*)buf + 4054528);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001472 = (bfloat16_t*)((char*)buf + 4054656);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001472,dtensor10001471,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001470 = (bfloat16_t*)((char*)buf + 4052480);
    bfloat16_t *dtensor10001472 = (bfloat16_t*)((char*)buf + 4054656);
    bfloat16_t *dtensor10001473 = (bfloat16_t*)((char*)buf + 4056704);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001473, dtensor10001470, dtensor10001472);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001299 = (bfloat16_t*)input_tensors.at(34);
    bfloat16_t *dtensor10001474 = (bfloat16_t*)((char*)buf + 4058752);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001474,dtensor10001309,dtensor10001299, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001473 = (bfloat16_t*)((char*)buf + 4056704);
    bfloat16_t *dtensor10001474 = (bfloat16_t*)((char*)buf + 4058752);
    bfloat16_t *dtensor10001475 = (bfloat16_t*)((char*)buf + 4066944);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001475,dtensor10001473,dtensor10001474, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001475 = (bfloat16_t*)((char*)buf + 4066944);
    bfloat16_t *dtensor10001300 = (bfloat16_t*)input_tensors.at(35);
    bfloat16_t *dtensor10001476 = (bfloat16_t*)((char*)buf + 4075136);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001476,dtensor10001475,dtensor10001300, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10001464 = (bfloat16_t*)((char*)buf + 3892736);
    bfloat16_t *dtensor10001476 = (bfloat16_t*)((char*)buf + 4075136);
    bfloat16_t *dtensor10001477 = (bfloat16_t*)((char*)buf + 4206208);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001477, dtensor10001464, dtensor10001476);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001301 = (bfloat16_t*)input_tensors.at(36);
    bfloat16_t *dtensor10001478 = (bfloat16_t*)((char*)buf + 4337280);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001478,dtensor10001265,dtensor10001301, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001301 = (bfloat16_t*)input_tensors.at(36);
    bfloat16_t *dtensor10001479 = (bfloat16_t*)((char*)buf + 4345472);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001479,dtensor10001307,dtensor10001301, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001302 = (bfloat16_t*)input_tensors.at(37);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001480 = (bfloat16_t*)((char*)buf + 4353664);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001480,dtensor10001302,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001478 = (bfloat16_t*)((char*)buf + 4337280);
    bfloat16_t *dtensor10001480 = (bfloat16_t*)((char*)buf + 4353664);
    bfloat16_t *dtensor10001481 = (bfloat16_t*)((char*)buf + 4361856);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001481,dtensor10001478,dtensor10001480, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001481 = (bfloat16_t*)((char*)buf + 4361856);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001482 = (bfloat16_t*)((char*)buf + 4363904);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001482, dtensor10001481, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001482 = (bfloat16_t*)((char*)buf + 4363904);
    bfloat16_t *dtensor10001483 = (bfloat16_t*)((char*)buf + 4365952);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001483, dtensor10001482);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001483 = (bfloat16_t*)((char*)buf + 4365952);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001484 = (bfloat16_t*)((char*)buf + 4368000);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001484,dtensor10001483,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001484 = (bfloat16_t*)((char*)buf + 4368000);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001485 = (bfloat16_t*)((char*)buf + 4368128);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001485,dtensor10001484,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001483 = (bfloat16_t*)((char*)buf + 4365952);
    bfloat16_t *dtensor10001485 = (bfloat16_t*)((char*)buf + 4368128);
    bfloat16_t *dtensor10001486 = (bfloat16_t*)((char*)buf + 4370176);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001486, dtensor10001483, dtensor10001485);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001301 = (bfloat16_t*)input_tensors.at(36);
    bfloat16_t *dtensor10001487 = (bfloat16_t*)((char*)buf + 4372224);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001487,dtensor10001309,dtensor10001301, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001486 = (bfloat16_t*)((char*)buf + 4370176);
    bfloat16_t *dtensor10001487 = (bfloat16_t*)((char*)buf + 4372224);
    bfloat16_t *dtensor10001488 = (bfloat16_t*)((char*)buf + 4380416);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001488,dtensor10001486,dtensor10001487, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001488 = (bfloat16_t*)((char*)buf + 4380416);
    bfloat16_t *dtensor10001302 = (bfloat16_t*)input_tensors.at(37);
    bfloat16_t *dtensor10001489 = (bfloat16_t*)((char*)buf + 4388608);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001489,dtensor10001488,dtensor10001302, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10001477 = (bfloat16_t*)((char*)buf + 4206208);
    bfloat16_t *dtensor10001489 = (bfloat16_t*)((char*)buf + 4388608);
    bfloat16_t *dtensor10001490 = (bfloat16_t*)((char*)buf + 4519680);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001490, dtensor10001477, dtensor10001489);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001303 = (bfloat16_t*)input_tensors.at(38);
    bfloat16_t *dtensor10001491 = (bfloat16_t*)((char*)buf + 4650752);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001491,dtensor10001265,dtensor10001303, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001303 = (bfloat16_t*)input_tensors.at(38);
    bfloat16_t *dtensor10001492 = (bfloat16_t*)((char*)buf + 4658944);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001492,dtensor10001307,dtensor10001303, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001304 = (bfloat16_t*)input_tensors.at(39);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001493 = (bfloat16_t*)((char*)buf + 4667136);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001493,dtensor10001304,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001491 = (bfloat16_t*)((char*)buf + 4650752);
    bfloat16_t *dtensor10001493 = (bfloat16_t*)((char*)buf + 4667136);
    bfloat16_t *dtensor10001494 = (bfloat16_t*)((char*)buf + 4675328);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001494,dtensor10001491,dtensor10001493, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001494 = (bfloat16_t*)((char*)buf + 4675328);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001495 = (bfloat16_t*)((char*)buf + 4677376);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001495, dtensor10001494, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001495 = (bfloat16_t*)((char*)buf + 4677376);
    bfloat16_t *dtensor10001496 = (bfloat16_t*)((char*)buf + 4679424);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001496, dtensor10001495);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001496 = (bfloat16_t*)((char*)buf + 4679424);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001497 = (bfloat16_t*)((char*)buf + 4681472);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001497,dtensor10001496,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001497 = (bfloat16_t*)((char*)buf + 4681472);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001498 = (bfloat16_t*)((char*)buf + 4681600);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001498,dtensor10001497,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001496 = (bfloat16_t*)((char*)buf + 4679424);
    bfloat16_t *dtensor10001498 = (bfloat16_t*)((char*)buf + 4681600);
    bfloat16_t *dtensor10001499 = (bfloat16_t*)((char*)buf + 4683648);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001499, dtensor10001496, dtensor10001498);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001303 = (bfloat16_t*)input_tensors.at(38);
    bfloat16_t *dtensor10001500 = (bfloat16_t*)((char*)buf + 4685696);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001500,dtensor10001309,dtensor10001303, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001499 = (bfloat16_t*)((char*)buf + 4683648);
    bfloat16_t *dtensor10001500 = (bfloat16_t*)((char*)buf + 4685696);
    bfloat16_t *dtensor10001501 = (bfloat16_t*)((char*)buf + 4693888);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001501,dtensor10001499,dtensor10001500, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001501 = (bfloat16_t*)((char*)buf + 4693888);
    bfloat16_t *dtensor10001304 = (bfloat16_t*)input_tensors.at(39);
    bfloat16_t *dtensor10001502 = (bfloat16_t*)((char*)buf + 4702080);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001502,dtensor10001501,dtensor10001304, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10001490 = (bfloat16_t*)((char*)buf + 4519680);
    bfloat16_t *dtensor10001502 = (bfloat16_t*)((char*)buf + 4702080);
    bfloat16_t *dtensor10001503 = (bfloat16_t*)((char*)buf + 4833152);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001503, dtensor10001490, dtensor10001502);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001265 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10001305 = (bfloat16_t*)input_tensors.at(40);
    bfloat16_t *dtensor10001504 = (bfloat16_t*)((char*)buf + 4964224);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001504,dtensor10001265,dtensor10001305, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001307 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10001305 = (bfloat16_t*)input_tensors.at(40);
    bfloat16_t *dtensor10001505 = (bfloat16_t*)((char*)buf + 4972416);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001505,dtensor10001307,dtensor10001305, 32,128,2048, 1,32, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001306 = (bfloat16_t*)input_tensors.at(41);
    bfloat16_t *dtensor10001308 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10001506 = (bfloat16_t*)((char*)buf + 4980608);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001506,dtensor10001306,dtensor10001308, 128,32,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001504 = (bfloat16_t*)((char*)buf + 4964224);
    bfloat16_t *dtensor10001506 = (bfloat16_t*)((char*)buf + 4980608);
    bfloat16_t *dtensor10001507 = (bfloat16_t*)((char*)buf + 4988800);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001507,dtensor10001504,dtensor10001506, 32,32,128, 1,32, 1,128, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10001507 = (bfloat16_t*)((char*)buf + 4988800);
    bfloat16_t *dtensor10001272 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10001508 = (bfloat16_t*)((char*)buf + 4990848);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001508, dtensor10001507, dtensor10001272);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10001508 = (bfloat16_t*)((char*)buf + 4990848);
    bfloat16_t *dtensor10001509 = (bfloat16_t*)((char*)buf + 4992896);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001509, dtensor10001508);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001509 = (bfloat16_t*)((char*)buf + 4992896);
    bfloat16_t *dtensor10001273 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10001510 = (bfloat16_t*)((char*)buf + 4994944);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001510,dtensor10001509,dtensor10001273, 32,1,32, 32,1, 1,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001510 = (bfloat16_t*)((char*)buf + 4994944);
    bfloat16_t *dtensor10001274 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10001511 = (bfloat16_t*)((char*)buf + 4995072);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001511,dtensor10001510,dtensor10001274, 32,32,1, 1,32, 32,1, 32,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10001509 = (bfloat16_t*)((char*)buf + 4992896);
    bfloat16_t *dtensor10001511 = (bfloat16_t*)((char*)buf + 4995072);
    bfloat16_t *dtensor10001512 = (bfloat16_t*)((char*)buf + 4997120);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<32>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001512, dtensor10001509, dtensor10001511);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001309 = (bfloat16_t*)((char*)buf + 262144);
    bfloat16_t *dtensor10001305 = (bfloat16_t*)input_tensors.at(40);
    bfloat16_t *dtensor10001513 = (bfloat16_t*)((char*)buf + 4999168);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001513,dtensor10001309,dtensor10001305, 32,128,2048, 2048,1, 128,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001512 = (bfloat16_t*)((char*)buf + 4997120);
    bfloat16_t *dtensor10001513 = (bfloat16_t*)((char*)buf + 4999168);
    bfloat16_t *dtensor10001514 = (bfloat16_t*)((char*)buf + 5007360);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001514,dtensor10001512,dtensor10001513, 32,128,32, 32,1, 1,32, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001514 = (bfloat16_t*)((char*)buf + 5007360);
    bfloat16_t *dtensor10001306 = (bfloat16_t*)input_tensors.at(41);
    bfloat16_t *dtensor10001515 = (bfloat16_t*)((char*)buf + 5015552);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001515,dtensor10001514,dtensor10001306, 32,2048,128, 1,32, 2048,1, 1,32, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10001503 = (bfloat16_t*)((char*)buf + 4833152);
    bfloat16_t *dtensor10001515 = (bfloat16_t*)((char*)buf + 5015552);
    bfloat16_t *dtensor10001516 = (bfloat16_t*)((char*)buf + 5146624);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>, Layout<Shape<Int<32>, Int<2048>>, Stride<Int<1>, Int<32>>>>;
    kernel::run(dtensor10001516, dtensor10001503, dtensor10001515);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10001516 = (bfloat16_t*)((char*)buf + 5146624);
    bfloat16_t *dtensor10001269 = (bfloat16_t*)input_tensors.at(4);
    bfloat16_t *dtensor10001517 = (bfloat16_t*)output_tensors.at(0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10001517,dtensor10001516,dtensor10001269, 32,1024,2048, 1,32, 1024,1, 1,32, 1, 0,0,0);
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

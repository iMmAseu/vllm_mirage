#define execute_mugraph attention_execute_mugraph_batch1
#define PyInit___mirage_launcher attention_PyInit_mirage_batch1
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
    bfloat16_t *dtensor10000507 = (bfloat16_t*)input_tensors.at(1);
    bfloat16_t *dtensor10000511 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000548,dtensor10000507,dtensor10000511, 16,2048,1024, 1024,1, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000512 = (bfloat16_t*)input_tensors.at(6);
    bfloat16_t *dtensor10000508 = (bfloat16_t*)input_tensors.at(2);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000549,dtensor10000512,dtensor10000508, 2048,16,1024, 1024,1, 16,1, 1,2048, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000509 = (bfloat16_t*)input_tensors.at(3);
    bfloat16_t *dtensor10000511 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000550,dtensor10000509,dtensor10000511, 16,2048,1024, 1024,1, 2048,1, 2048,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000516 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000551 = (bfloat16_t*)((char*)buf + 196608);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000551,dtensor10000506,dtensor10000516, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000516 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000552 = (bfloat16_t*)((char*)buf + 200704);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000552,dtensor10000548,dtensor10000516, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000517 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000553 = (bfloat16_t*)((char*)buf + 204800);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000553,dtensor10000517,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000551 = (bfloat16_t*)((char*)buf + 196608);
    bfloat16_t *dtensor10000553 = (bfloat16_t*)((char*)buf + 204800);
    bfloat16_t *dtensor10000554 = (bfloat16_t*)((char*)buf + 208896);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000554,dtensor10000551,dtensor10000553, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000554 = (bfloat16_t*)((char*)buf + 208896);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000555 = (bfloat16_t*)((char*)buf + 209408);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000555, dtensor10000554, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000555 = (bfloat16_t*)((char*)buf + 209408);
    bfloat16_t *dtensor10000556 = (bfloat16_t*)((char*)buf + 209920);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000556, dtensor10000555);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000556 = (bfloat16_t*)((char*)buf + 209920);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000557 = (bfloat16_t*)((char*)buf + 210432);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000557,dtensor10000556,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000557 = (bfloat16_t*)((char*)buf + 210432);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000558 = (bfloat16_t*)((char*)buf + 210560);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000558,dtensor10000557,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000556 = (bfloat16_t*)((char*)buf + 209920);
    bfloat16_t *dtensor10000558 = (bfloat16_t*)((char*)buf + 210560);
    bfloat16_t *dtensor10000559 = (bfloat16_t*)((char*)buf + 211072);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000559, dtensor10000556, dtensor10000558);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000516 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000560 = (bfloat16_t*)((char*)buf + 211584);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000560,dtensor10000550,dtensor10000516, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000559 = (bfloat16_t*)((char*)buf + 211072);
    bfloat16_t *dtensor10000560 = (bfloat16_t*)((char*)buf + 211584);
    bfloat16_t *dtensor10000561 = (bfloat16_t*)((char*)buf + 215680);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000561,dtensor10000559,dtensor10000560, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000561 = (bfloat16_t*)((char*)buf + 215680);
    bfloat16_t *dtensor10000517 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 219776);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000562,dtensor10000561,dtensor10000517, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000518 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10000563 = (bfloat16_t*)((char*)buf + 285312);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000563,dtensor10000506,dtensor10000518, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000518 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10000564 = (bfloat16_t*)((char*)buf + 289408);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000564,dtensor10000548,dtensor10000518, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000519 = (bfloat16_t*)input_tensors.at(13);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000565 = (bfloat16_t*)((char*)buf + 293504);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000565,dtensor10000519,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000563 = (bfloat16_t*)((char*)buf + 285312);
    bfloat16_t *dtensor10000565 = (bfloat16_t*)((char*)buf + 293504);
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 297600);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000566,dtensor10000563,dtensor10000565, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 297600);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 298112);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000567, dtensor10000566, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 298112);
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 298624);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000568, dtensor10000567);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 298624);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000569 = (bfloat16_t*)((char*)buf + 299136);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000569,dtensor10000568,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000569 = (bfloat16_t*)((char*)buf + 299136);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000570 = (bfloat16_t*)((char*)buf + 299264);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000570,dtensor10000569,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 298624);
    bfloat16_t *dtensor10000570 = (bfloat16_t*)((char*)buf + 299264);
    bfloat16_t *dtensor10000571 = (bfloat16_t*)((char*)buf + 299776);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000571, dtensor10000568, dtensor10000570);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000518 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10000572 = (bfloat16_t*)((char*)buf + 300288);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000572,dtensor10000550,dtensor10000518, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000571 = (bfloat16_t*)((char*)buf + 299776);
    bfloat16_t *dtensor10000572 = (bfloat16_t*)((char*)buf + 300288);
    bfloat16_t *dtensor10000573 = (bfloat16_t*)((char*)buf + 304384);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000573,dtensor10000571,dtensor10000572, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000573 = (bfloat16_t*)((char*)buf + 304384);
    bfloat16_t *dtensor10000519 = (bfloat16_t*)input_tensors.at(13);
    bfloat16_t *dtensor10000574 = (bfloat16_t*)((char*)buf + 308480);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000574,dtensor10000573,dtensor10000519, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 219776);
    bfloat16_t *dtensor10000574 = (bfloat16_t*)((char*)buf + 308480);
    bfloat16_t *dtensor10000575 = (bfloat16_t*)((char*)buf + 374016);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000575, dtensor10000562, dtensor10000574);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000520 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10000576 = (bfloat16_t*)((char*)buf + 439552);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000576,dtensor10000506,dtensor10000520, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000520 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10000577 = (bfloat16_t*)((char*)buf + 443648);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000577,dtensor10000548,dtensor10000520, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000521 = (bfloat16_t*)input_tensors.at(15);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000578 = (bfloat16_t*)((char*)buf + 447744);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000578,dtensor10000521,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000576 = (bfloat16_t*)((char*)buf + 439552);
    bfloat16_t *dtensor10000578 = (bfloat16_t*)((char*)buf + 447744);
    bfloat16_t *dtensor10000579 = (bfloat16_t*)((char*)buf + 451840);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000579,dtensor10000576,dtensor10000578, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000579 = (bfloat16_t*)((char*)buf + 451840);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000580 = (bfloat16_t*)((char*)buf + 452352);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000580, dtensor10000579, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000580 = (bfloat16_t*)((char*)buf + 452352);
    bfloat16_t *dtensor10000581 = (bfloat16_t*)((char*)buf + 452864);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000581, dtensor10000580);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000581 = (bfloat16_t*)((char*)buf + 452864);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000582 = (bfloat16_t*)((char*)buf + 453376);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000582,dtensor10000581,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000582 = (bfloat16_t*)((char*)buf + 453376);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000583 = (bfloat16_t*)((char*)buf + 453504);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000583,dtensor10000582,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000581 = (bfloat16_t*)((char*)buf + 452864);
    bfloat16_t *dtensor10000583 = (bfloat16_t*)((char*)buf + 453504);
    bfloat16_t *dtensor10000584 = (bfloat16_t*)((char*)buf + 454016);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000584, dtensor10000581, dtensor10000583);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000520 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10000585 = (bfloat16_t*)((char*)buf + 454528);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000585,dtensor10000550,dtensor10000520, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000584 = (bfloat16_t*)((char*)buf + 454016);
    bfloat16_t *dtensor10000585 = (bfloat16_t*)((char*)buf + 454528);
    bfloat16_t *dtensor10000586 = (bfloat16_t*)((char*)buf + 458624);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000586,dtensor10000584,dtensor10000585, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000586 = (bfloat16_t*)((char*)buf + 458624);
    bfloat16_t *dtensor10000521 = (bfloat16_t*)input_tensors.at(15);
    bfloat16_t *dtensor10000587 = (bfloat16_t*)((char*)buf + 462720);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000587,dtensor10000586,dtensor10000521, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000575 = (bfloat16_t*)((char*)buf + 374016);
    bfloat16_t *dtensor10000587 = (bfloat16_t*)((char*)buf + 462720);
    bfloat16_t *dtensor10000588 = (bfloat16_t*)((char*)buf + 528256);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000588, dtensor10000575, dtensor10000587);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000522 = (bfloat16_t*)input_tensors.at(16);
    bfloat16_t *dtensor10000589 = (bfloat16_t*)((char*)buf + 593792);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000589,dtensor10000506,dtensor10000522, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000522 = (bfloat16_t*)input_tensors.at(16);
    bfloat16_t *dtensor10000590 = (bfloat16_t*)((char*)buf + 597888);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000590,dtensor10000548,dtensor10000522, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000523 = (bfloat16_t*)input_tensors.at(17);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000591 = (bfloat16_t*)((char*)buf + 601984);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000591,dtensor10000523,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000589 = (bfloat16_t*)((char*)buf + 593792);
    bfloat16_t *dtensor10000591 = (bfloat16_t*)((char*)buf + 601984);
    bfloat16_t *dtensor10000592 = (bfloat16_t*)((char*)buf + 606080);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000592,dtensor10000589,dtensor10000591, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000592 = (bfloat16_t*)((char*)buf + 606080);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000593 = (bfloat16_t*)((char*)buf + 606592);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000593, dtensor10000592, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000593 = (bfloat16_t*)((char*)buf + 606592);
    bfloat16_t *dtensor10000594 = (bfloat16_t*)((char*)buf + 607104);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000594, dtensor10000593);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000594 = (bfloat16_t*)((char*)buf + 607104);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000595 = (bfloat16_t*)((char*)buf + 607616);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000595,dtensor10000594,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000595 = (bfloat16_t*)((char*)buf + 607616);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000596 = (bfloat16_t*)((char*)buf + 607744);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000596,dtensor10000595,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000594 = (bfloat16_t*)((char*)buf + 607104);
    bfloat16_t *dtensor10000596 = (bfloat16_t*)((char*)buf + 607744);
    bfloat16_t *dtensor10000597 = (bfloat16_t*)((char*)buf + 608256);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000597, dtensor10000594, dtensor10000596);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000522 = (bfloat16_t*)input_tensors.at(16);
    bfloat16_t *dtensor10000598 = (bfloat16_t*)((char*)buf + 608768);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000598,dtensor10000550,dtensor10000522, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000597 = (bfloat16_t*)((char*)buf + 608256);
    bfloat16_t *dtensor10000598 = (bfloat16_t*)((char*)buf + 608768);
    bfloat16_t *dtensor10000599 = (bfloat16_t*)((char*)buf + 612864);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000599,dtensor10000597,dtensor10000598, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000599 = (bfloat16_t*)((char*)buf + 612864);
    bfloat16_t *dtensor10000523 = (bfloat16_t*)input_tensors.at(17);
    bfloat16_t *dtensor10000600 = (bfloat16_t*)((char*)buf + 616960);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000600,dtensor10000599,dtensor10000523, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000588 = (bfloat16_t*)((char*)buf + 528256);
    bfloat16_t *dtensor10000600 = (bfloat16_t*)((char*)buf + 616960);
    bfloat16_t *dtensor10000601 = (bfloat16_t*)((char*)buf + 682496);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000601, dtensor10000588, dtensor10000600);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000524 = (bfloat16_t*)input_tensors.at(18);
    bfloat16_t *dtensor10000602 = (bfloat16_t*)((char*)buf + 748032);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000602,dtensor10000506,dtensor10000524, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000524 = (bfloat16_t*)input_tensors.at(18);
    bfloat16_t *dtensor10000603 = (bfloat16_t*)((char*)buf + 752128);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000603,dtensor10000548,dtensor10000524, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000525 = (bfloat16_t*)input_tensors.at(19);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000604 = (bfloat16_t*)((char*)buf + 756224);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000604,dtensor10000525,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000602 = (bfloat16_t*)((char*)buf + 748032);
    bfloat16_t *dtensor10000604 = (bfloat16_t*)((char*)buf + 756224);
    bfloat16_t *dtensor10000605 = (bfloat16_t*)((char*)buf + 760320);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000605,dtensor10000602,dtensor10000604, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000605 = (bfloat16_t*)((char*)buf + 760320);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000606 = (bfloat16_t*)((char*)buf + 760832);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000606, dtensor10000605, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000606 = (bfloat16_t*)((char*)buf + 760832);
    bfloat16_t *dtensor10000607 = (bfloat16_t*)((char*)buf + 761344);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000607, dtensor10000606);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000607 = (bfloat16_t*)((char*)buf + 761344);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000608 = (bfloat16_t*)((char*)buf + 761856);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000608,dtensor10000607,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000608 = (bfloat16_t*)((char*)buf + 761856);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000609 = (bfloat16_t*)((char*)buf + 761984);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000609,dtensor10000608,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000607 = (bfloat16_t*)((char*)buf + 761344);
    bfloat16_t *dtensor10000609 = (bfloat16_t*)((char*)buf + 761984);
    bfloat16_t *dtensor10000610 = (bfloat16_t*)((char*)buf + 762496);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000610, dtensor10000607, dtensor10000609);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000524 = (bfloat16_t*)input_tensors.at(18);
    bfloat16_t *dtensor10000611 = (bfloat16_t*)((char*)buf + 763008);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000611,dtensor10000550,dtensor10000524, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000610 = (bfloat16_t*)((char*)buf + 762496);
    bfloat16_t *dtensor10000611 = (bfloat16_t*)((char*)buf + 763008);
    bfloat16_t *dtensor10000612 = (bfloat16_t*)((char*)buf + 767104);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000612,dtensor10000610,dtensor10000611, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000612 = (bfloat16_t*)((char*)buf + 767104);
    bfloat16_t *dtensor10000525 = (bfloat16_t*)input_tensors.at(19);
    bfloat16_t *dtensor10000613 = (bfloat16_t*)((char*)buf + 771200);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000613,dtensor10000612,dtensor10000525, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000601 = (bfloat16_t*)((char*)buf + 682496);
    bfloat16_t *dtensor10000613 = (bfloat16_t*)((char*)buf + 771200);
    bfloat16_t *dtensor10000614 = (bfloat16_t*)((char*)buf + 836736);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000614, dtensor10000601, dtensor10000613);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000526 = (bfloat16_t*)input_tensors.at(20);
    bfloat16_t *dtensor10000615 = (bfloat16_t*)((char*)buf + 902272);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000615,dtensor10000506,dtensor10000526, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000526 = (bfloat16_t*)input_tensors.at(20);
    bfloat16_t *dtensor10000616 = (bfloat16_t*)((char*)buf + 906368);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000616,dtensor10000548,dtensor10000526, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(21);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000617 = (bfloat16_t*)((char*)buf + 910464);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000617,dtensor10000527,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000615 = (bfloat16_t*)((char*)buf + 902272);
    bfloat16_t *dtensor10000617 = (bfloat16_t*)((char*)buf + 910464);
    bfloat16_t *dtensor10000618 = (bfloat16_t*)((char*)buf + 914560);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000618,dtensor10000615,dtensor10000617, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000618 = (bfloat16_t*)((char*)buf + 914560);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000619 = (bfloat16_t*)((char*)buf + 915072);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000619, dtensor10000618, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000619 = (bfloat16_t*)((char*)buf + 915072);
    bfloat16_t *dtensor10000620 = (bfloat16_t*)((char*)buf + 915584);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000620, dtensor10000619);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000620 = (bfloat16_t*)((char*)buf + 915584);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000621 = (bfloat16_t*)((char*)buf + 916096);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000621,dtensor10000620,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000621 = (bfloat16_t*)((char*)buf + 916096);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000622 = (bfloat16_t*)((char*)buf + 916224);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000622,dtensor10000621,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000620 = (bfloat16_t*)((char*)buf + 915584);
    bfloat16_t *dtensor10000622 = (bfloat16_t*)((char*)buf + 916224);
    bfloat16_t *dtensor10000623 = (bfloat16_t*)((char*)buf + 916736);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000623, dtensor10000620, dtensor10000622);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000526 = (bfloat16_t*)input_tensors.at(20);
    bfloat16_t *dtensor10000624 = (bfloat16_t*)((char*)buf + 917248);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000624,dtensor10000550,dtensor10000526, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000623 = (bfloat16_t*)((char*)buf + 916736);
    bfloat16_t *dtensor10000624 = (bfloat16_t*)((char*)buf + 917248);
    bfloat16_t *dtensor10000625 = (bfloat16_t*)((char*)buf + 921344);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000625,dtensor10000623,dtensor10000624, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000625 = (bfloat16_t*)((char*)buf + 921344);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(21);
    bfloat16_t *dtensor10000626 = (bfloat16_t*)((char*)buf + 925440);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000626,dtensor10000625,dtensor10000527, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000614 = (bfloat16_t*)((char*)buf + 836736);
    bfloat16_t *dtensor10000626 = (bfloat16_t*)((char*)buf + 925440);
    bfloat16_t *dtensor10000627 = (bfloat16_t*)((char*)buf + 990976);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000627, dtensor10000614, dtensor10000626);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(22);
    bfloat16_t *dtensor10000628 = (bfloat16_t*)((char*)buf + 1056512);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000628,dtensor10000506,dtensor10000528, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(22);
    bfloat16_t *dtensor10000629 = (bfloat16_t*)((char*)buf + 1060608);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000629,dtensor10000548,dtensor10000528, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(23);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000630 = (bfloat16_t*)((char*)buf + 1064704);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000630,dtensor10000529,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000628 = (bfloat16_t*)((char*)buf + 1056512);
    bfloat16_t *dtensor10000630 = (bfloat16_t*)((char*)buf + 1064704);
    bfloat16_t *dtensor10000631 = (bfloat16_t*)((char*)buf + 1068800);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000631,dtensor10000628,dtensor10000630, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000631 = (bfloat16_t*)((char*)buf + 1068800);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000632 = (bfloat16_t*)((char*)buf + 1069312);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000632, dtensor10000631, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000632 = (bfloat16_t*)((char*)buf + 1069312);
    bfloat16_t *dtensor10000633 = (bfloat16_t*)((char*)buf + 1069824);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000633, dtensor10000632);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000633 = (bfloat16_t*)((char*)buf + 1069824);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000634 = (bfloat16_t*)((char*)buf + 1070336);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000634,dtensor10000633,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000634 = (bfloat16_t*)((char*)buf + 1070336);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000635 = (bfloat16_t*)((char*)buf + 1070464);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000635,dtensor10000634,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000633 = (bfloat16_t*)((char*)buf + 1069824);
    bfloat16_t *dtensor10000635 = (bfloat16_t*)((char*)buf + 1070464);
    bfloat16_t *dtensor10000636 = (bfloat16_t*)((char*)buf + 1070976);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000636, dtensor10000633, dtensor10000635);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(22);
    bfloat16_t *dtensor10000637 = (bfloat16_t*)((char*)buf + 1071488);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000637,dtensor10000550,dtensor10000528, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000636 = (bfloat16_t*)((char*)buf + 1070976);
    bfloat16_t *dtensor10000637 = (bfloat16_t*)((char*)buf + 1071488);
    bfloat16_t *dtensor10000638 = (bfloat16_t*)((char*)buf + 1075584);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000638,dtensor10000636,dtensor10000637, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000638 = (bfloat16_t*)((char*)buf + 1075584);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(23);
    bfloat16_t *dtensor10000639 = (bfloat16_t*)((char*)buf + 1079680);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000639,dtensor10000638,dtensor10000529, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000627 = (bfloat16_t*)((char*)buf + 990976);
    bfloat16_t *dtensor10000639 = (bfloat16_t*)((char*)buf + 1079680);
    bfloat16_t *dtensor10000640 = (bfloat16_t*)((char*)buf + 1145216);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000640, dtensor10000627, dtensor10000639);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000530 = (bfloat16_t*)input_tensors.at(24);
    bfloat16_t *dtensor10000641 = (bfloat16_t*)((char*)buf + 1210752);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000641,dtensor10000506,dtensor10000530, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000530 = (bfloat16_t*)input_tensors.at(24);
    bfloat16_t *dtensor10000642 = (bfloat16_t*)((char*)buf + 1214848);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000642,dtensor10000548,dtensor10000530, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000531 = (bfloat16_t*)input_tensors.at(25);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000643 = (bfloat16_t*)((char*)buf + 1218944);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000643,dtensor10000531,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000641 = (bfloat16_t*)((char*)buf + 1210752);
    bfloat16_t *dtensor10000643 = (bfloat16_t*)((char*)buf + 1218944);
    bfloat16_t *dtensor10000644 = (bfloat16_t*)((char*)buf + 1223040);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000644,dtensor10000641,dtensor10000643, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000644 = (bfloat16_t*)((char*)buf + 1223040);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000645 = (bfloat16_t*)((char*)buf + 1223552);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000645, dtensor10000644, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000645 = (bfloat16_t*)((char*)buf + 1223552);
    bfloat16_t *dtensor10000646 = (bfloat16_t*)((char*)buf + 1224064);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000646, dtensor10000645);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000646 = (bfloat16_t*)((char*)buf + 1224064);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000647 = (bfloat16_t*)((char*)buf + 1224576);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000647,dtensor10000646,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000647 = (bfloat16_t*)((char*)buf + 1224576);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000648 = (bfloat16_t*)((char*)buf + 1224704);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000648,dtensor10000647,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000646 = (bfloat16_t*)((char*)buf + 1224064);
    bfloat16_t *dtensor10000648 = (bfloat16_t*)((char*)buf + 1224704);
    bfloat16_t *dtensor10000649 = (bfloat16_t*)((char*)buf + 1225216);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000649, dtensor10000646, dtensor10000648);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000530 = (bfloat16_t*)input_tensors.at(24);
    bfloat16_t *dtensor10000650 = (bfloat16_t*)((char*)buf + 1225728);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000650,dtensor10000550,dtensor10000530, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000649 = (bfloat16_t*)((char*)buf + 1225216);
    bfloat16_t *dtensor10000650 = (bfloat16_t*)((char*)buf + 1225728);
    bfloat16_t *dtensor10000651 = (bfloat16_t*)((char*)buf + 1229824);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000651,dtensor10000649,dtensor10000650, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000651 = (bfloat16_t*)((char*)buf + 1229824);
    bfloat16_t *dtensor10000531 = (bfloat16_t*)input_tensors.at(25);
    bfloat16_t *dtensor10000652 = (bfloat16_t*)((char*)buf + 1233920);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000652,dtensor10000651,dtensor10000531, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000640 = (bfloat16_t*)((char*)buf + 1145216);
    bfloat16_t *dtensor10000652 = (bfloat16_t*)((char*)buf + 1233920);
    bfloat16_t *dtensor10000653 = (bfloat16_t*)((char*)buf + 1299456);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000653, dtensor10000640, dtensor10000652);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000532 = (bfloat16_t*)input_tensors.at(26);
    bfloat16_t *dtensor10000654 = (bfloat16_t*)((char*)buf + 1364992);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000654,dtensor10000506,dtensor10000532, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000532 = (bfloat16_t*)input_tensors.at(26);
    bfloat16_t *dtensor10000655 = (bfloat16_t*)((char*)buf + 1369088);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000655,dtensor10000548,dtensor10000532, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000533 = (bfloat16_t*)input_tensors.at(27);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000656 = (bfloat16_t*)((char*)buf + 1373184);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000656,dtensor10000533,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000654 = (bfloat16_t*)((char*)buf + 1364992);
    bfloat16_t *dtensor10000656 = (bfloat16_t*)((char*)buf + 1373184);
    bfloat16_t *dtensor10000657 = (bfloat16_t*)((char*)buf + 1377280);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000657,dtensor10000654,dtensor10000656, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000657 = (bfloat16_t*)((char*)buf + 1377280);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000658 = (bfloat16_t*)((char*)buf + 1377792);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000658, dtensor10000657, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000658 = (bfloat16_t*)((char*)buf + 1377792);
    bfloat16_t *dtensor10000659 = (bfloat16_t*)((char*)buf + 1378304);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000659, dtensor10000658);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000659 = (bfloat16_t*)((char*)buf + 1378304);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000660 = (bfloat16_t*)((char*)buf + 1378816);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000660,dtensor10000659,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000660 = (bfloat16_t*)((char*)buf + 1378816);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000661 = (bfloat16_t*)((char*)buf + 1378944);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000661,dtensor10000660,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000659 = (bfloat16_t*)((char*)buf + 1378304);
    bfloat16_t *dtensor10000661 = (bfloat16_t*)((char*)buf + 1378944);
    bfloat16_t *dtensor10000662 = (bfloat16_t*)((char*)buf + 1379456);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000662, dtensor10000659, dtensor10000661);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000532 = (bfloat16_t*)input_tensors.at(26);
    bfloat16_t *dtensor10000663 = (bfloat16_t*)((char*)buf + 1379968);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000663,dtensor10000550,dtensor10000532, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000662 = (bfloat16_t*)((char*)buf + 1379456);
    bfloat16_t *dtensor10000663 = (bfloat16_t*)((char*)buf + 1379968);
    bfloat16_t *dtensor10000664 = (bfloat16_t*)((char*)buf + 1384064);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000664,dtensor10000662,dtensor10000663, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000664 = (bfloat16_t*)((char*)buf + 1384064);
    bfloat16_t *dtensor10000533 = (bfloat16_t*)input_tensors.at(27);
    bfloat16_t *dtensor10000665 = (bfloat16_t*)((char*)buf + 1388160);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000665,dtensor10000664,dtensor10000533, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000653 = (bfloat16_t*)((char*)buf + 1299456);
    bfloat16_t *dtensor10000665 = (bfloat16_t*)((char*)buf + 1388160);
    bfloat16_t *dtensor10000666 = (bfloat16_t*)((char*)buf + 1453696);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000666, dtensor10000653, dtensor10000665);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000534 = (bfloat16_t*)input_tensors.at(28);
    bfloat16_t *dtensor10000667 = (bfloat16_t*)((char*)buf + 1519232);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000667,dtensor10000506,dtensor10000534, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000534 = (bfloat16_t*)input_tensors.at(28);
    bfloat16_t *dtensor10000668 = (bfloat16_t*)((char*)buf + 1523328);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000668,dtensor10000548,dtensor10000534, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000535 = (bfloat16_t*)input_tensors.at(29);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000669 = (bfloat16_t*)((char*)buf + 1527424);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000669,dtensor10000535,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000667 = (bfloat16_t*)((char*)buf + 1519232);
    bfloat16_t *dtensor10000669 = (bfloat16_t*)((char*)buf + 1527424);
    bfloat16_t *dtensor10000670 = (bfloat16_t*)((char*)buf + 1531520);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000670,dtensor10000667,dtensor10000669, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000670 = (bfloat16_t*)((char*)buf + 1531520);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000671 = (bfloat16_t*)((char*)buf + 1532032);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000671, dtensor10000670, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000671 = (bfloat16_t*)((char*)buf + 1532032);
    bfloat16_t *dtensor10000672 = (bfloat16_t*)((char*)buf + 1532544);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000672, dtensor10000671);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000672 = (bfloat16_t*)((char*)buf + 1532544);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000673 = (bfloat16_t*)((char*)buf + 1533056);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000673,dtensor10000672,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000673 = (bfloat16_t*)((char*)buf + 1533056);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000674 = (bfloat16_t*)((char*)buf + 1533184);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000674,dtensor10000673,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000672 = (bfloat16_t*)((char*)buf + 1532544);
    bfloat16_t *dtensor10000674 = (bfloat16_t*)((char*)buf + 1533184);
    bfloat16_t *dtensor10000675 = (bfloat16_t*)((char*)buf + 1533696);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000675, dtensor10000672, dtensor10000674);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000534 = (bfloat16_t*)input_tensors.at(28);
    bfloat16_t *dtensor10000676 = (bfloat16_t*)((char*)buf + 1534208);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000676,dtensor10000550,dtensor10000534, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000675 = (bfloat16_t*)((char*)buf + 1533696);
    bfloat16_t *dtensor10000676 = (bfloat16_t*)((char*)buf + 1534208);
    bfloat16_t *dtensor10000677 = (bfloat16_t*)((char*)buf + 1538304);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000677,dtensor10000675,dtensor10000676, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000677 = (bfloat16_t*)((char*)buf + 1538304);
    bfloat16_t *dtensor10000535 = (bfloat16_t*)input_tensors.at(29);
    bfloat16_t *dtensor10000678 = (bfloat16_t*)((char*)buf + 1542400);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000678,dtensor10000677,dtensor10000535, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000666 = (bfloat16_t*)((char*)buf + 1453696);
    bfloat16_t *dtensor10000678 = (bfloat16_t*)((char*)buf + 1542400);
    bfloat16_t *dtensor10000679 = (bfloat16_t*)((char*)buf + 1607936);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000679, dtensor10000666, dtensor10000678);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000536 = (bfloat16_t*)input_tensors.at(30);
    bfloat16_t *dtensor10000680 = (bfloat16_t*)((char*)buf + 1673472);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000680,dtensor10000506,dtensor10000536, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000536 = (bfloat16_t*)input_tensors.at(30);
    bfloat16_t *dtensor10000681 = (bfloat16_t*)((char*)buf + 1677568);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000681,dtensor10000548,dtensor10000536, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000537 = (bfloat16_t*)input_tensors.at(31);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000682 = (bfloat16_t*)((char*)buf + 1681664);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000682,dtensor10000537,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000680 = (bfloat16_t*)((char*)buf + 1673472);
    bfloat16_t *dtensor10000682 = (bfloat16_t*)((char*)buf + 1681664);
    bfloat16_t *dtensor10000683 = (bfloat16_t*)((char*)buf + 1685760);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000683,dtensor10000680,dtensor10000682, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000683 = (bfloat16_t*)((char*)buf + 1685760);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000684 = (bfloat16_t*)((char*)buf + 1686272);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000684, dtensor10000683, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000684 = (bfloat16_t*)((char*)buf + 1686272);
    bfloat16_t *dtensor10000685 = (bfloat16_t*)((char*)buf + 1686784);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000685, dtensor10000684);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000685 = (bfloat16_t*)((char*)buf + 1686784);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000686 = (bfloat16_t*)((char*)buf + 1687296);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000686,dtensor10000685,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000686 = (bfloat16_t*)((char*)buf + 1687296);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000687 = (bfloat16_t*)((char*)buf + 1687424);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000687,dtensor10000686,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000685 = (bfloat16_t*)((char*)buf + 1686784);
    bfloat16_t *dtensor10000687 = (bfloat16_t*)((char*)buf + 1687424);
    bfloat16_t *dtensor10000688 = (bfloat16_t*)((char*)buf + 1687936);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000688, dtensor10000685, dtensor10000687);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000536 = (bfloat16_t*)input_tensors.at(30);
    bfloat16_t *dtensor10000689 = (bfloat16_t*)((char*)buf + 1688448);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000689,dtensor10000550,dtensor10000536, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000688 = (bfloat16_t*)((char*)buf + 1687936);
    bfloat16_t *dtensor10000689 = (bfloat16_t*)((char*)buf + 1688448);
    bfloat16_t *dtensor10000690 = (bfloat16_t*)((char*)buf + 1692544);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000690,dtensor10000688,dtensor10000689, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000690 = (bfloat16_t*)((char*)buf + 1692544);
    bfloat16_t *dtensor10000537 = (bfloat16_t*)input_tensors.at(31);
    bfloat16_t *dtensor10000691 = (bfloat16_t*)((char*)buf + 1696640);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000691,dtensor10000690,dtensor10000537, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000679 = (bfloat16_t*)((char*)buf + 1607936);
    bfloat16_t *dtensor10000691 = (bfloat16_t*)((char*)buf + 1696640);
    bfloat16_t *dtensor10000692 = (bfloat16_t*)((char*)buf + 1762176);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000692, dtensor10000679, dtensor10000691);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000538 = (bfloat16_t*)input_tensors.at(32);
    bfloat16_t *dtensor10000693 = (bfloat16_t*)((char*)buf + 1827712);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000693,dtensor10000506,dtensor10000538, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000538 = (bfloat16_t*)input_tensors.at(32);
    bfloat16_t *dtensor10000694 = (bfloat16_t*)((char*)buf + 1831808);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000694,dtensor10000548,dtensor10000538, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000539 = (bfloat16_t*)input_tensors.at(33);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000695 = (bfloat16_t*)((char*)buf + 1835904);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000695,dtensor10000539,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000693 = (bfloat16_t*)((char*)buf + 1827712);
    bfloat16_t *dtensor10000695 = (bfloat16_t*)((char*)buf + 1835904);
    bfloat16_t *dtensor10000696 = (bfloat16_t*)((char*)buf + 1840000);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000696,dtensor10000693,dtensor10000695, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000696 = (bfloat16_t*)((char*)buf + 1840000);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000697 = (bfloat16_t*)((char*)buf + 1840512);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000697, dtensor10000696, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000697 = (bfloat16_t*)((char*)buf + 1840512);
    bfloat16_t *dtensor10000698 = (bfloat16_t*)((char*)buf + 1841024);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000698, dtensor10000697);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000698 = (bfloat16_t*)((char*)buf + 1841024);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000699 = (bfloat16_t*)((char*)buf + 1841536);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000699,dtensor10000698,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000699 = (bfloat16_t*)((char*)buf + 1841536);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000700 = (bfloat16_t*)((char*)buf + 1841664);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000700,dtensor10000699,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000698 = (bfloat16_t*)((char*)buf + 1841024);
    bfloat16_t *dtensor10000700 = (bfloat16_t*)((char*)buf + 1841664);
    bfloat16_t *dtensor10000701 = (bfloat16_t*)((char*)buf + 1842176);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000701, dtensor10000698, dtensor10000700);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000538 = (bfloat16_t*)input_tensors.at(32);
    bfloat16_t *dtensor10000702 = (bfloat16_t*)((char*)buf + 1842688);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000702,dtensor10000550,dtensor10000538, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000701 = (bfloat16_t*)((char*)buf + 1842176);
    bfloat16_t *dtensor10000702 = (bfloat16_t*)((char*)buf + 1842688);
    bfloat16_t *dtensor10000703 = (bfloat16_t*)((char*)buf + 1846784);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000703,dtensor10000701,dtensor10000702, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000703 = (bfloat16_t*)((char*)buf + 1846784);
    bfloat16_t *dtensor10000539 = (bfloat16_t*)input_tensors.at(33);
    bfloat16_t *dtensor10000704 = (bfloat16_t*)((char*)buf + 1850880);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000704,dtensor10000703,dtensor10000539, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000692 = (bfloat16_t*)((char*)buf + 1762176);
    bfloat16_t *dtensor10000704 = (bfloat16_t*)((char*)buf + 1850880);
    bfloat16_t *dtensor10000705 = (bfloat16_t*)((char*)buf + 1916416);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000705, dtensor10000692, dtensor10000704);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000540 = (bfloat16_t*)input_tensors.at(34);
    bfloat16_t *dtensor10000706 = (bfloat16_t*)((char*)buf + 1981952);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000706,dtensor10000506,dtensor10000540, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000540 = (bfloat16_t*)input_tensors.at(34);
    bfloat16_t *dtensor10000707 = (bfloat16_t*)((char*)buf + 1986048);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000707,dtensor10000548,dtensor10000540, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000541 = (bfloat16_t*)input_tensors.at(35);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000708 = (bfloat16_t*)((char*)buf + 1990144);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000708,dtensor10000541,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000706 = (bfloat16_t*)((char*)buf + 1981952);
    bfloat16_t *dtensor10000708 = (bfloat16_t*)((char*)buf + 1990144);
    bfloat16_t *dtensor10000709 = (bfloat16_t*)((char*)buf + 1994240);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000709,dtensor10000706,dtensor10000708, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000709 = (bfloat16_t*)((char*)buf + 1994240);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000710 = (bfloat16_t*)((char*)buf + 1994752);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000710, dtensor10000709, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000710 = (bfloat16_t*)((char*)buf + 1994752);
    bfloat16_t *dtensor10000711 = (bfloat16_t*)((char*)buf + 1995264);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000711, dtensor10000710);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000711 = (bfloat16_t*)((char*)buf + 1995264);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000712 = (bfloat16_t*)((char*)buf + 1995776);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000712,dtensor10000711,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000712 = (bfloat16_t*)((char*)buf + 1995776);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000713 = (bfloat16_t*)((char*)buf + 1995904);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000713,dtensor10000712,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000711 = (bfloat16_t*)((char*)buf + 1995264);
    bfloat16_t *dtensor10000713 = (bfloat16_t*)((char*)buf + 1995904);
    bfloat16_t *dtensor10000714 = (bfloat16_t*)((char*)buf + 1996416);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000714, dtensor10000711, dtensor10000713);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000540 = (bfloat16_t*)input_tensors.at(34);
    bfloat16_t *dtensor10000715 = (bfloat16_t*)((char*)buf + 1996928);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000715,dtensor10000550,dtensor10000540, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000714 = (bfloat16_t*)((char*)buf + 1996416);
    bfloat16_t *dtensor10000715 = (bfloat16_t*)((char*)buf + 1996928);
    bfloat16_t *dtensor10000716 = (bfloat16_t*)((char*)buf + 2001024);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000716,dtensor10000714,dtensor10000715, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000716 = (bfloat16_t*)((char*)buf + 2001024);
    bfloat16_t *dtensor10000541 = (bfloat16_t*)input_tensors.at(35);
    bfloat16_t *dtensor10000717 = (bfloat16_t*)((char*)buf + 2005120);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000717,dtensor10000716,dtensor10000541, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000705 = (bfloat16_t*)((char*)buf + 1916416);
    bfloat16_t *dtensor10000717 = (bfloat16_t*)((char*)buf + 2005120);
    bfloat16_t *dtensor10000718 = (bfloat16_t*)((char*)buf + 2070656);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000718, dtensor10000705, dtensor10000717);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000542 = (bfloat16_t*)input_tensors.at(36);
    bfloat16_t *dtensor10000719 = (bfloat16_t*)((char*)buf + 2136192);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000719,dtensor10000506,dtensor10000542, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000542 = (bfloat16_t*)input_tensors.at(36);
    bfloat16_t *dtensor10000720 = (bfloat16_t*)((char*)buf + 2140288);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000720,dtensor10000548,dtensor10000542, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000543 = (bfloat16_t*)input_tensors.at(37);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000721 = (bfloat16_t*)((char*)buf + 2144384);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000721,dtensor10000543,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000719 = (bfloat16_t*)((char*)buf + 2136192);
    bfloat16_t *dtensor10000721 = (bfloat16_t*)((char*)buf + 2144384);
    bfloat16_t *dtensor10000722 = (bfloat16_t*)((char*)buf + 2148480);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000722,dtensor10000719,dtensor10000721, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000722 = (bfloat16_t*)((char*)buf + 2148480);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000723 = (bfloat16_t*)((char*)buf + 2148992);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000723, dtensor10000722, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000723 = (bfloat16_t*)((char*)buf + 2148992);
    bfloat16_t *dtensor10000724 = (bfloat16_t*)((char*)buf + 2149504);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000724, dtensor10000723);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000724 = (bfloat16_t*)((char*)buf + 2149504);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000725 = (bfloat16_t*)((char*)buf + 2150016);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000725,dtensor10000724,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000725 = (bfloat16_t*)((char*)buf + 2150016);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000726 = (bfloat16_t*)((char*)buf + 2150144);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000726,dtensor10000725,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000724 = (bfloat16_t*)((char*)buf + 2149504);
    bfloat16_t *dtensor10000726 = (bfloat16_t*)((char*)buf + 2150144);
    bfloat16_t *dtensor10000727 = (bfloat16_t*)((char*)buf + 2150656);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000727, dtensor10000724, dtensor10000726);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000542 = (bfloat16_t*)input_tensors.at(36);
    bfloat16_t *dtensor10000728 = (bfloat16_t*)((char*)buf + 2151168);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000728,dtensor10000550,dtensor10000542, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000727 = (bfloat16_t*)((char*)buf + 2150656);
    bfloat16_t *dtensor10000728 = (bfloat16_t*)((char*)buf + 2151168);
    bfloat16_t *dtensor10000729 = (bfloat16_t*)((char*)buf + 2155264);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000729,dtensor10000727,dtensor10000728, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000729 = (bfloat16_t*)((char*)buf + 2155264);
    bfloat16_t *dtensor10000543 = (bfloat16_t*)input_tensors.at(37);
    bfloat16_t *dtensor10000730 = (bfloat16_t*)((char*)buf + 2159360);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000730,dtensor10000729,dtensor10000543, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000718 = (bfloat16_t*)((char*)buf + 2070656);
    bfloat16_t *dtensor10000730 = (bfloat16_t*)((char*)buf + 2159360);
    bfloat16_t *dtensor10000731 = (bfloat16_t*)((char*)buf + 2224896);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000731, dtensor10000718, dtensor10000730);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000544 = (bfloat16_t*)input_tensors.at(38);
    bfloat16_t *dtensor10000732 = (bfloat16_t*)((char*)buf + 2290432);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000732,dtensor10000506,dtensor10000544, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000544 = (bfloat16_t*)input_tensors.at(38);
    bfloat16_t *dtensor10000733 = (bfloat16_t*)((char*)buf + 2294528);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000733,dtensor10000548,dtensor10000544, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000545 = (bfloat16_t*)input_tensors.at(39);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000734 = (bfloat16_t*)((char*)buf + 2298624);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000734,dtensor10000545,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000732 = (bfloat16_t*)((char*)buf + 2290432);
    bfloat16_t *dtensor10000734 = (bfloat16_t*)((char*)buf + 2298624);
    bfloat16_t *dtensor10000735 = (bfloat16_t*)((char*)buf + 2302720);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000735,dtensor10000732,dtensor10000734, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000735 = (bfloat16_t*)((char*)buf + 2302720);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000736 = (bfloat16_t*)((char*)buf + 2303232);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000736, dtensor10000735, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000736 = (bfloat16_t*)((char*)buf + 2303232);
    bfloat16_t *dtensor10000737 = (bfloat16_t*)((char*)buf + 2303744);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000737, dtensor10000736);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000737 = (bfloat16_t*)((char*)buf + 2303744);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000738 = (bfloat16_t*)((char*)buf + 2304256);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000738,dtensor10000737,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000738 = (bfloat16_t*)((char*)buf + 2304256);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000739 = (bfloat16_t*)((char*)buf + 2304384);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000739,dtensor10000738,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000737 = (bfloat16_t*)((char*)buf + 2303744);
    bfloat16_t *dtensor10000739 = (bfloat16_t*)((char*)buf + 2304384);
    bfloat16_t *dtensor10000740 = (bfloat16_t*)((char*)buf + 2304896);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000740, dtensor10000737, dtensor10000739);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000544 = (bfloat16_t*)input_tensors.at(38);
    bfloat16_t *dtensor10000741 = (bfloat16_t*)((char*)buf + 2305408);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000741,dtensor10000550,dtensor10000544, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000740 = (bfloat16_t*)((char*)buf + 2304896);
    bfloat16_t *dtensor10000741 = (bfloat16_t*)((char*)buf + 2305408);
    bfloat16_t *dtensor10000742 = (bfloat16_t*)((char*)buf + 2309504);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000742,dtensor10000740,dtensor10000741, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000742 = (bfloat16_t*)((char*)buf + 2309504);
    bfloat16_t *dtensor10000545 = (bfloat16_t*)input_tensors.at(39);
    bfloat16_t *dtensor10000743 = (bfloat16_t*)((char*)buf + 2313600);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000743,dtensor10000742,dtensor10000545, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000731 = (bfloat16_t*)((char*)buf + 2224896);
    bfloat16_t *dtensor10000743 = (bfloat16_t*)((char*)buf + 2313600);
    bfloat16_t *dtensor10000744 = (bfloat16_t*)((char*)buf + 2379136);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000744, dtensor10000731, dtensor10000743);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000506 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000546 = (bfloat16_t*)input_tensors.at(40);
    bfloat16_t *dtensor10000745 = (bfloat16_t*)((char*)buf + 2444672);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000745,dtensor10000506,dtensor10000546, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000548 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000546 = (bfloat16_t*)input_tensors.at(40);
    bfloat16_t *dtensor10000746 = (bfloat16_t*)((char*)buf + 2448768);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000746,dtensor10000548,dtensor10000546, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000547 = (bfloat16_t*)input_tensors.at(41);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000747 = (bfloat16_t*)((char*)buf + 2452864);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000747,dtensor10000547,dtensor10000549, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000745 = (bfloat16_t*)((char*)buf + 2444672);
    bfloat16_t *dtensor10000747 = (bfloat16_t*)((char*)buf + 2452864);
    bfloat16_t *dtensor10000748 = (bfloat16_t*)((char*)buf + 2456960);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000748,dtensor10000745,dtensor10000747, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000748 = (bfloat16_t*)((char*)buf + 2456960);
    bfloat16_t *dtensor10000513 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000749 = (bfloat16_t*)((char*)buf + 2457472);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000749, dtensor10000748, dtensor10000513);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000749 = (bfloat16_t*)((char*)buf + 2457472);
    bfloat16_t *dtensor10000750 = (bfloat16_t*)((char*)buf + 2457984);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000750, dtensor10000749);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000750 = (bfloat16_t*)((char*)buf + 2457984);
    bfloat16_t *dtensor10000514 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000751 = (bfloat16_t*)((char*)buf + 2458496);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000751,dtensor10000750,dtensor10000514, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000751 = (bfloat16_t*)((char*)buf + 2458496);
    bfloat16_t *dtensor10000515 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000752 = (bfloat16_t*)((char*)buf + 2458624);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000752,dtensor10000751,dtensor10000515, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000750 = (bfloat16_t*)((char*)buf + 2457984);
    bfloat16_t *dtensor10000752 = (bfloat16_t*)((char*)buf + 2458624);
    bfloat16_t *dtensor10000753 = (bfloat16_t*)((char*)buf + 2459136);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000753, dtensor10000750, dtensor10000752);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000550 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000546 = (bfloat16_t*)input_tensors.at(40);
    bfloat16_t *dtensor10000754 = (bfloat16_t*)((char*)buf + 2459648);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000754,dtensor10000550,dtensor10000546, 16,128,2048, 2048,1, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000753 = (bfloat16_t*)((char*)buf + 2459136);
    bfloat16_t *dtensor10000754 = (bfloat16_t*)((char*)buf + 2459648);
    bfloat16_t *dtensor10000755 = (bfloat16_t*)((char*)buf + 2463744);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000755,dtensor10000753,dtensor10000754, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000755 = (bfloat16_t*)((char*)buf + 2463744);
    bfloat16_t *dtensor10000547 = (bfloat16_t*)input_tensors.at(41);
    bfloat16_t *dtensor10000756 = (bfloat16_t*)((char*)buf + 2467840);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000756,dtensor10000755,dtensor10000547, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000744 = (bfloat16_t*)((char*)buf + 2379136);
    bfloat16_t *dtensor10000756 = (bfloat16_t*)((char*)buf + 2467840);
    bfloat16_t *dtensor10000757 = (bfloat16_t*)((char*)buf + 2533376);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000757, dtensor10000744, dtensor10000756);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000757 = (bfloat16_t*)((char*)buf + 2533376);
    bfloat16_t *dtensor10000510 = (bfloat16_t*)input_tensors.at(4);
    bfloat16_t *dtensor10000758 = (bfloat16_t*)output_tensors.at(0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000758,dtensor10000757,dtensor10000510, 16,1024,2048, 1,16, 1024,1, 1,16, 1, 0,0,0);
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

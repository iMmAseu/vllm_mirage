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
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000518 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000520 = (bfloat16_t*)input_tensors.at(2);
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000562,dtensor10000518,dtensor10000520, 16,2048,1024, 1024,1, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000518 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000521 = (bfloat16_t*)input_tensors.at(3);
    bfloat16_t *dtensor10000563 = (bfloat16_t*)((char*)buf + 65536);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000563,dtensor10000518,dtensor10000521, 16,1024,1024, 1024,1, 1024,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000518 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000522 = (bfloat16_t*)input_tensors.at(4);
    bfloat16_t *dtensor10000564 = (bfloat16_t*)((char*)buf + 98304);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000564,dtensor10000518,dtensor10000522, 16,1024,1024, 1024,1, 1024,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000524 = (bfloat16_t*)input_tensors.at(6);
    bfloat16_t *dtensor10000519 = (bfloat16_t*)input_tensors.at(1);
    bfloat16_t *dtensor10000565 = (bfloat16_t*)((char*)buf + 131072);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000565,dtensor10000524,dtensor10000519, 1024,16,1024, 1024,1, 16,1, 1,1024, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000563 = (bfloat16_t*)((char*)buf + 65536);
    bfloat16_t *dtensor10000525 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000566,dtensor10000563,dtensor10000525, 16,2048,1024, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000526 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10000565 = (bfloat16_t*)((char*)buf + 131072);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000567,dtensor10000526,dtensor10000565, 2048,16,1024, 1024,1, 1,1024, 1,2048, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000564 = (bfloat16_t*)((char*)buf + 98304);
    bfloat16_t *dtensor10000525 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000568,dtensor10000564,dtensor10000525, 16,2048,1024, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000530 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10000569 = (bfloat16_t*)((char*)buf + 360448);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000569,dtensor10000562,dtensor10000530, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000530 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10000570 = (bfloat16_t*)((char*)buf + 364544);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000570,dtensor10000566,dtensor10000530, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000531 = (bfloat16_t*)input_tensors.at(13);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000571 = (bfloat16_t*)((char*)buf + 368640);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000571,dtensor10000531,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000569 = (bfloat16_t*)((char*)buf + 360448);
    bfloat16_t *dtensor10000571 = (bfloat16_t*)((char*)buf + 368640);
    bfloat16_t *dtensor10000572 = (bfloat16_t*)((char*)buf + 372736);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000572,dtensor10000569,dtensor10000571, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000572 = (bfloat16_t*)((char*)buf + 372736);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000573 = (bfloat16_t*)((char*)buf + 373248);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000573, dtensor10000572, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000573 = (bfloat16_t*)((char*)buf + 373248);
    bfloat16_t *dtensor10000574 = (bfloat16_t*)((char*)buf + 373760);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000574, dtensor10000573);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000574 = (bfloat16_t*)((char*)buf + 373760);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000575 = (bfloat16_t*)((char*)buf + 374272);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000575,dtensor10000574,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000575 = (bfloat16_t*)((char*)buf + 374272);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000576 = (bfloat16_t*)((char*)buf + 374400);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000576,dtensor10000575,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000574 = (bfloat16_t*)((char*)buf + 373760);
    bfloat16_t *dtensor10000576 = (bfloat16_t*)((char*)buf + 374400);
    bfloat16_t *dtensor10000577 = (bfloat16_t*)((char*)buf + 374912);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000577, dtensor10000574, dtensor10000576);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000530 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10000578 = (bfloat16_t*)((char*)buf + 375424);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000578,dtensor10000568,dtensor10000530, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000577 = (bfloat16_t*)((char*)buf + 374912);
    bfloat16_t *dtensor10000578 = (bfloat16_t*)((char*)buf + 375424);
    bfloat16_t *dtensor10000579 = (bfloat16_t*)((char*)buf + 379520);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000579,dtensor10000577,dtensor10000578, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000579 = (bfloat16_t*)((char*)buf + 379520);
    bfloat16_t *dtensor10000531 = (bfloat16_t*)input_tensors.at(13);
    bfloat16_t *dtensor10000580 = (bfloat16_t*)((char*)buf + 383616);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000580,dtensor10000579,dtensor10000531, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000532 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10000581 = (bfloat16_t*)((char*)buf + 449152);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000581,dtensor10000562,dtensor10000532, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000532 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10000582 = (bfloat16_t*)((char*)buf + 453248);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000582,dtensor10000566,dtensor10000532, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000533 = (bfloat16_t*)input_tensors.at(15);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000583 = (bfloat16_t*)((char*)buf + 457344);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000583,dtensor10000533,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000581 = (bfloat16_t*)((char*)buf + 449152);
    bfloat16_t *dtensor10000583 = (bfloat16_t*)((char*)buf + 457344);
    bfloat16_t *dtensor10000584 = (bfloat16_t*)((char*)buf + 461440);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000584,dtensor10000581,dtensor10000583, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000584 = (bfloat16_t*)((char*)buf + 461440);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000585 = (bfloat16_t*)((char*)buf + 461952);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000585, dtensor10000584, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000585 = (bfloat16_t*)((char*)buf + 461952);
    bfloat16_t *dtensor10000586 = (bfloat16_t*)((char*)buf + 462464);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000586, dtensor10000585);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000586 = (bfloat16_t*)((char*)buf + 462464);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000587 = (bfloat16_t*)((char*)buf + 462976);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000587,dtensor10000586,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000587 = (bfloat16_t*)((char*)buf + 462976);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000588 = (bfloat16_t*)((char*)buf + 463104);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000588,dtensor10000587,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000586 = (bfloat16_t*)((char*)buf + 462464);
    bfloat16_t *dtensor10000588 = (bfloat16_t*)((char*)buf + 463104);
    bfloat16_t *dtensor10000589 = (bfloat16_t*)((char*)buf + 463616);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000589, dtensor10000586, dtensor10000588);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000532 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10000590 = (bfloat16_t*)((char*)buf + 464128);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000590,dtensor10000568,dtensor10000532, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000589 = (bfloat16_t*)((char*)buf + 463616);
    bfloat16_t *dtensor10000590 = (bfloat16_t*)((char*)buf + 464128);
    bfloat16_t *dtensor10000591 = (bfloat16_t*)((char*)buf + 468224);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000591,dtensor10000589,dtensor10000590, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000591 = (bfloat16_t*)((char*)buf + 468224);
    bfloat16_t *dtensor10000533 = (bfloat16_t*)input_tensors.at(15);
    bfloat16_t *dtensor10000592 = (bfloat16_t*)((char*)buf + 472320);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000592,dtensor10000591,dtensor10000533, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000580 = (bfloat16_t*)((char*)buf + 383616);
    bfloat16_t *dtensor10000592 = (bfloat16_t*)((char*)buf + 472320);
    bfloat16_t *dtensor10000593 = (bfloat16_t*)((char*)buf + 537856);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000593, dtensor10000580, dtensor10000592);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000534 = (bfloat16_t*)input_tensors.at(16);
    bfloat16_t *dtensor10000594 = (bfloat16_t*)((char*)buf + 603392);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000594,dtensor10000562,dtensor10000534, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000534 = (bfloat16_t*)input_tensors.at(16);
    bfloat16_t *dtensor10000595 = (bfloat16_t*)((char*)buf + 607488);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000595,dtensor10000566,dtensor10000534, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000535 = (bfloat16_t*)input_tensors.at(17);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000596 = (bfloat16_t*)((char*)buf + 611584);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000596,dtensor10000535,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000594 = (bfloat16_t*)((char*)buf + 603392);
    bfloat16_t *dtensor10000596 = (bfloat16_t*)((char*)buf + 611584);
    bfloat16_t *dtensor10000597 = (bfloat16_t*)((char*)buf + 615680);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000597,dtensor10000594,dtensor10000596, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000597 = (bfloat16_t*)((char*)buf + 615680);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000598 = (bfloat16_t*)((char*)buf + 616192);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000598, dtensor10000597, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000598 = (bfloat16_t*)((char*)buf + 616192);
    bfloat16_t *dtensor10000599 = (bfloat16_t*)((char*)buf + 616704);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000599, dtensor10000598);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000599 = (bfloat16_t*)((char*)buf + 616704);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000600 = (bfloat16_t*)((char*)buf + 617216);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000600,dtensor10000599,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000600 = (bfloat16_t*)((char*)buf + 617216);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000601 = (bfloat16_t*)((char*)buf + 617344);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000601,dtensor10000600,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000599 = (bfloat16_t*)((char*)buf + 616704);
    bfloat16_t *dtensor10000601 = (bfloat16_t*)((char*)buf + 617344);
    bfloat16_t *dtensor10000602 = (bfloat16_t*)((char*)buf + 617856);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000602, dtensor10000599, dtensor10000601);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000534 = (bfloat16_t*)input_tensors.at(16);
    bfloat16_t *dtensor10000603 = (bfloat16_t*)((char*)buf + 618368);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000603,dtensor10000568,dtensor10000534, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000602 = (bfloat16_t*)((char*)buf + 617856);
    bfloat16_t *dtensor10000603 = (bfloat16_t*)((char*)buf + 618368);
    bfloat16_t *dtensor10000604 = (bfloat16_t*)((char*)buf + 622464);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000604,dtensor10000602,dtensor10000603, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000604 = (bfloat16_t*)((char*)buf + 622464);
    bfloat16_t *dtensor10000535 = (bfloat16_t*)input_tensors.at(17);
    bfloat16_t *dtensor10000605 = (bfloat16_t*)((char*)buf + 626560);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000605,dtensor10000604,dtensor10000535, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000593 = (bfloat16_t*)((char*)buf + 537856);
    bfloat16_t *dtensor10000605 = (bfloat16_t*)((char*)buf + 626560);
    bfloat16_t *dtensor10000606 = (bfloat16_t*)((char*)buf + 692096);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000606, dtensor10000593, dtensor10000605);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000536 = (bfloat16_t*)input_tensors.at(18);
    bfloat16_t *dtensor10000607 = (bfloat16_t*)((char*)buf + 757632);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000607,dtensor10000562,dtensor10000536, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000536 = (bfloat16_t*)input_tensors.at(18);
    bfloat16_t *dtensor10000608 = (bfloat16_t*)((char*)buf + 761728);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000608,dtensor10000566,dtensor10000536, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000537 = (bfloat16_t*)input_tensors.at(19);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000609 = (bfloat16_t*)((char*)buf + 765824);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000609,dtensor10000537,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000607 = (bfloat16_t*)((char*)buf + 757632);
    bfloat16_t *dtensor10000609 = (bfloat16_t*)((char*)buf + 765824);
    bfloat16_t *dtensor10000610 = (bfloat16_t*)((char*)buf + 769920);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000610,dtensor10000607,dtensor10000609, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000610 = (bfloat16_t*)((char*)buf + 769920);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000611 = (bfloat16_t*)((char*)buf + 770432);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000611, dtensor10000610, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000611 = (bfloat16_t*)((char*)buf + 770432);
    bfloat16_t *dtensor10000612 = (bfloat16_t*)((char*)buf + 770944);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000612, dtensor10000611);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000612 = (bfloat16_t*)((char*)buf + 770944);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000613 = (bfloat16_t*)((char*)buf + 771456);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000613,dtensor10000612,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000613 = (bfloat16_t*)((char*)buf + 771456);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000614 = (bfloat16_t*)((char*)buf + 771584);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000614,dtensor10000613,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000612 = (bfloat16_t*)((char*)buf + 770944);
    bfloat16_t *dtensor10000614 = (bfloat16_t*)((char*)buf + 771584);
    bfloat16_t *dtensor10000615 = (bfloat16_t*)((char*)buf + 772096);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000615, dtensor10000612, dtensor10000614);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000536 = (bfloat16_t*)input_tensors.at(18);
    bfloat16_t *dtensor10000616 = (bfloat16_t*)((char*)buf + 772608);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000616,dtensor10000568,dtensor10000536, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000615 = (bfloat16_t*)((char*)buf + 772096);
    bfloat16_t *dtensor10000616 = (bfloat16_t*)((char*)buf + 772608);
    bfloat16_t *dtensor10000617 = (bfloat16_t*)((char*)buf + 776704);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000617,dtensor10000615,dtensor10000616, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000617 = (bfloat16_t*)((char*)buf + 776704);
    bfloat16_t *dtensor10000537 = (bfloat16_t*)input_tensors.at(19);
    bfloat16_t *dtensor10000618 = (bfloat16_t*)((char*)buf + 780800);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000618,dtensor10000617,dtensor10000537, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000606 = (bfloat16_t*)((char*)buf + 692096);
    bfloat16_t *dtensor10000618 = (bfloat16_t*)((char*)buf + 780800);
    bfloat16_t *dtensor10000619 = (bfloat16_t*)((char*)buf + 846336);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000619, dtensor10000606, dtensor10000618);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000538 = (bfloat16_t*)input_tensors.at(20);
    bfloat16_t *dtensor10000620 = (bfloat16_t*)((char*)buf + 911872);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000620,dtensor10000562,dtensor10000538, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000538 = (bfloat16_t*)input_tensors.at(20);
    bfloat16_t *dtensor10000621 = (bfloat16_t*)((char*)buf + 915968);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000621,dtensor10000566,dtensor10000538, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000539 = (bfloat16_t*)input_tensors.at(21);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000622 = (bfloat16_t*)((char*)buf + 920064);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000622,dtensor10000539,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000620 = (bfloat16_t*)((char*)buf + 911872);
    bfloat16_t *dtensor10000622 = (bfloat16_t*)((char*)buf + 920064);
    bfloat16_t *dtensor10000623 = (bfloat16_t*)((char*)buf + 924160);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000623,dtensor10000620,dtensor10000622, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000623 = (bfloat16_t*)((char*)buf + 924160);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000624 = (bfloat16_t*)((char*)buf + 924672);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000624, dtensor10000623, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000624 = (bfloat16_t*)((char*)buf + 924672);
    bfloat16_t *dtensor10000625 = (bfloat16_t*)((char*)buf + 925184);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000625, dtensor10000624);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000625 = (bfloat16_t*)((char*)buf + 925184);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000626 = (bfloat16_t*)((char*)buf + 925696);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000626,dtensor10000625,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000626 = (bfloat16_t*)((char*)buf + 925696);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000627 = (bfloat16_t*)((char*)buf + 925824);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000627,dtensor10000626,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000625 = (bfloat16_t*)((char*)buf + 925184);
    bfloat16_t *dtensor10000627 = (bfloat16_t*)((char*)buf + 925824);
    bfloat16_t *dtensor10000628 = (bfloat16_t*)((char*)buf + 926336);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000628, dtensor10000625, dtensor10000627);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000538 = (bfloat16_t*)input_tensors.at(20);
    bfloat16_t *dtensor10000629 = (bfloat16_t*)((char*)buf + 926848);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000629,dtensor10000568,dtensor10000538, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000628 = (bfloat16_t*)((char*)buf + 926336);
    bfloat16_t *dtensor10000629 = (bfloat16_t*)((char*)buf + 926848);
    bfloat16_t *dtensor10000630 = (bfloat16_t*)((char*)buf + 930944);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000630,dtensor10000628,dtensor10000629, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000630 = (bfloat16_t*)((char*)buf + 930944);
    bfloat16_t *dtensor10000539 = (bfloat16_t*)input_tensors.at(21);
    bfloat16_t *dtensor10000631 = (bfloat16_t*)((char*)buf + 935040);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000631,dtensor10000630,dtensor10000539, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000619 = (bfloat16_t*)((char*)buf + 846336);
    bfloat16_t *dtensor10000631 = (bfloat16_t*)((char*)buf + 935040);
    bfloat16_t *dtensor10000632 = (bfloat16_t*)((char*)buf + 1000576);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000632, dtensor10000619, dtensor10000631);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000540 = (bfloat16_t*)input_tensors.at(22);
    bfloat16_t *dtensor10000633 = (bfloat16_t*)((char*)buf + 1066112);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000633,dtensor10000562,dtensor10000540, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000540 = (bfloat16_t*)input_tensors.at(22);
    bfloat16_t *dtensor10000634 = (bfloat16_t*)((char*)buf + 1070208);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000634,dtensor10000566,dtensor10000540, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000541 = (bfloat16_t*)input_tensors.at(23);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000635 = (bfloat16_t*)((char*)buf + 1074304);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000635,dtensor10000541,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000633 = (bfloat16_t*)((char*)buf + 1066112);
    bfloat16_t *dtensor10000635 = (bfloat16_t*)((char*)buf + 1074304);
    bfloat16_t *dtensor10000636 = (bfloat16_t*)((char*)buf + 1078400);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000636,dtensor10000633,dtensor10000635, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000636 = (bfloat16_t*)((char*)buf + 1078400);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000637 = (bfloat16_t*)((char*)buf + 1078912);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000637, dtensor10000636, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000637 = (bfloat16_t*)((char*)buf + 1078912);
    bfloat16_t *dtensor10000638 = (bfloat16_t*)((char*)buf + 1079424);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000638, dtensor10000637);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000638 = (bfloat16_t*)((char*)buf + 1079424);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000639 = (bfloat16_t*)((char*)buf + 1079936);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000639,dtensor10000638,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000639 = (bfloat16_t*)((char*)buf + 1079936);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000640 = (bfloat16_t*)((char*)buf + 1080064);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000640,dtensor10000639,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000638 = (bfloat16_t*)((char*)buf + 1079424);
    bfloat16_t *dtensor10000640 = (bfloat16_t*)((char*)buf + 1080064);
    bfloat16_t *dtensor10000641 = (bfloat16_t*)((char*)buf + 1080576);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000641, dtensor10000638, dtensor10000640);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000540 = (bfloat16_t*)input_tensors.at(22);
    bfloat16_t *dtensor10000642 = (bfloat16_t*)((char*)buf + 1081088);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000642,dtensor10000568,dtensor10000540, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000641 = (bfloat16_t*)((char*)buf + 1080576);
    bfloat16_t *dtensor10000642 = (bfloat16_t*)((char*)buf + 1081088);
    bfloat16_t *dtensor10000643 = (bfloat16_t*)((char*)buf + 1085184);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000643,dtensor10000641,dtensor10000642, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000643 = (bfloat16_t*)((char*)buf + 1085184);
    bfloat16_t *dtensor10000541 = (bfloat16_t*)input_tensors.at(23);
    bfloat16_t *dtensor10000644 = (bfloat16_t*)((char*)buf + 1089280);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000644,dtensor10000643,dtensor10000541, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000632 = (bfloat16_t*)((char*)buf + 1000576);
    bfloat16_t *dtensor10000644 = (bfloat16_t*)((char*)buf + 1089280);
    bfloat16_t *dtensor10000645 = (bfloat16_t*)((char*)buf + 1154816);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000645, dtensor10000632, dtensor10000644);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000542 = (bfloat16_t*)input_tensors.at(24);
    bfloat16_t *dtensor10000646 = (bfloat16_t*)((char*)buf + 1220352);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000646,dtensor10000562,dtensor10000542, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000542 = (bfloat16_t*)input_tensors.at(24);
    bfloat16_t *dtensor10000647 = (bfloat16_t*)((char*)buf + 1224448);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000647,dtensor10000566,dtensor10000542, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000543 = (bfloat16_t*)input_tensors.at(25);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000648 = (bfloat16_t*)((char*)buf + 1228544);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000648,dtensor10000543,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000646 = (bfloat16_t*)((char*)buf + 1220352);
    bfloat16_t *dtensor10000648 = (bfloat16_t*)((char*)buf + 1228544);
    bfloat16_t *dtensor10000649 = (bfloat16_t*)((char*)buf + 1232640);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000649,dtensor10000646,dtensor10000648, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000649 = (bfloat16_t*)((char*)buf + 1232640);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000650 = (bfloat16_t*)((char*)buf + 1233152);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000650, dtensor10000649, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000650 = (bfloat16_t*)((char*)buf + 1233152);
    bfloat16_t *dtensor10000651 = (bfloat16_t*)((char*)buf + 1233664);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000651, dtensor10000650);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000651 = (bfloat16_t*)((char*)buf + 1233664);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000652 = (bfloat16_t*)((char*)buf + 1234176);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000652,dtensor10000651,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000652 = (bfloat16_t*)((char*)buf + 1234176);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000653 = (bfloat16_t*)((char*)buf + 1234304);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000653,dtensor10000652,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000651 = (bfloat16_t*)((char*)buf + 1233664);
    bfloat16_t *dtensor10000653 = (bfloat16_t*)((char*)buf + 1234304);
    bfloat16_t *dtensor10000654 = (bfloat16_t*)((char*)buf + 1234816);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000654, dtensor10000651, dtensor10000653);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000542 = (bfloat16_t*)input_tensors.at(24);
    bfloat16_t *dtensor10000655 = (bfloat16_t*)((char*)buf + 1235328);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000655,dtensor10000568,dtensor10000542, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000654 = (bfloat16_t*)((char*)buf + 1234816);
    bfloat16_t *dtensor10000655 = (bfloat16_t*)((char*)buf + 1235328);
    bfloat16_t *dtensor10000656 = (bfloat16_t*)((char*)buf + 1239424);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000656,dtensor10000654,dtensor10000655, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000656 = (bfloat16_t*)((char*)buf + 1239424);
    bfloat16_t *dtensor10000543 = (bfloat16_t*)input_tensors.at(25);
    bfloat16_t *dtensor10000657 = (bfloat16_t*)((char*)buf + 1243520);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000657,dtensor10000656,dtensor10000543, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000645 = (bfloat16_t*)((char*)buf + 1154816);
    bfloat16_t *dtensor10000657 = (bfloat16_t*)((char*)buf + 1243520);
    bfloat16_t *dtensor10000658 = (bfloat16_t*)((char*)buf + 1309056);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000658, dtensor10000645, dtensor10000657);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000544 = (bfloat16_t*)input_tensors.at(26);
    bfloat16_t *dtensor10000659 = (bfloat16_t*)((char*)buf + 1374592);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000659,dtensor10000562,dtensor10000544, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000544 = (bfloat16_t*)input_tensors.at(26);
    bfloat16_t *dtensor10000660 = (bfloat16_t*)((char*)buf + 1378688);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000660,dtensor10000566,dtensor10000544, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000545 = (bfloat16_t*)input_tensors.at(27);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000661 = (bfloat16_t*)((char*)buf + 1382784);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000661,dtensor10000545,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000659 = (bfloat16_t*)((char*)buf + 1374592);
    bfloat16_t *dtensor10000661 = (bfloat16_t*)((char*)buf + 1382784);
    bfloat16_t *dtensor10000662 = (bfloat16_t*)((char*)buf + 1386880);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000662,dtensor10000659,dtensor10000661, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000662 = (bfloat16_t*)((char*)buf + 1386880);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000663 = (bfloat16_t*)((char*)buf + 1387392);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000663, dtensor10000662, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000663 = (bfloat16_t*)((char*)buf + 1387392);
    bfloat16_t *dtensor10000664 = (bfloat16_t*)((char*)buf + 1387904);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000664, dtensor10000663);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000664 = (bfloat16_t*)((char*)buf + 1387904);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000665 = (bfloat16_t*)((char*)buf + 1388416);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000665,dtensor10000664,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000665 = (bfloat16_t*)((char*)buf + 1388416);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000666 = (bfloat16_t*)((char*)buf + 1388544);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000666,dtensor10000665,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000664 = (bfloat16_t*)((char*)buf + 1387904);
    bfloat16_t *dtensor10000666 = (bfloat16_t*)((char*)buf + 1388544);
    bfloat16_t *dtensor10000667 = (bfloat16_t*)((char*)buf + 1389056);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000667, dtensor10000664, dtensor10000666);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000544 = (bfloat16_t*)input_tensors.at(26);
    bfloat16_t *dtensor10000668 = (bfloat16_t*)((char*)buf + 1389568);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000668,dtensor10000568,dtensor10000544, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000667 = (bfloat16_t*)((char*)buf + 1389056);
    bfloat16_t *dtensor10000668 = (bfloat16_t*)((char*)buf + 1389568);
    bfloat16_t *dtensor10000669 = (bfloat16_t*)((char*)buf + 1393664);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000669,dtensor10000667,dtensor10000668, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000669 = (bfloat16_t*)((char*)buf + 1393664);
    bfloat16_t *dtensor10000545 = (bfloat16_t*)input_tensors.at(27);
    bfloat16_t *dtensor10000670 = (bfloat16_t*)((char*)buf + 1397760);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000670,dtensor10000669,dtensor10000545, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000658 = (bfloat16_t*)((char*)buf + 1309056);
    bfloat16_t *dtensor10000670 = (bfloat16_t*)((char*)buf + 1397760);
    bfloat16_t *dtensor10000671 = (bfloat16_t*)((char*)buf + 1463296);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000671, dtensor10000658, dtensor10000670);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000546 = (bfloat16_t*)input_tensors.at(28);
    bfloat16_t *dtensor10000672 = (bfloat16_t*)((char*)buf + 1528832);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000672,dtensor10000562,dtensor10000546, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000546 = (bfloat16_t*)input_tensors.at(28);
    bfloat16_t *dtensor10000673 = (bfloat16_t*)((char*)buf + 1532928);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000673,dtensor10000566,dtensor10000546, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000547 = (bfloat16_t*)input_tensors.at(29);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000674 = (bfloat16_t*)((char*)buf + 1537024);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000674,dtensor10000547,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000672 = (bfloat16_t*)((char*)buf + 1528832);
    bfloat16_t *dtensor10000674 = (bfloat16_t*)((char*)buf + 1537024);
    bfloat16_t *dtensor10000675 = (bfloat16_t*)((char*)buf + 1541120);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000675,dtensor10000672,dtensor10000674, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000675 = (bfloat16_t*)((char*)buf + 1541120);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000676 = (bfloat16_t*)((char*)buf + 1541632);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000676, dtensor10000675, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000676 = (bfloat16_t*)((char*)buf + 1541632);
    bfloat16_t *dtensor10000677 = (bfloat16_t*)((char*)buf + 1542144);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000677, dtensor10000676);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000677 = (bfloat16_t*)((char*)buf + 1542144);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000678 = (bfloat16_t*)((char*)buf + 1542656);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000678,dtensor10000677,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000678 = (bfloat16_t*)((char*)buf + 1542656);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000679 = (bfloat16_t*)((char*)buf + 1542784);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000679,dtensor10000678,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000677 = (bfloat16_t*)((char*)buf + 1542144);
    bfloat16_t *dtensor10000679 = (bfloat16_t*)((char*)buf + 1542784);
    bfloat16_t *dtensor10000680 = (bfloat16_t*)((char*)buf + 1543296);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000680, dtensor10000677, dtensor10000679);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000546 = (bfloat16_t*)input_tensors.at(28);
    bfloat16_t *dtensor10000681 = (bfloat16_t*)((char*)buf + 1543808);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000681,dtensor10000568,dtensor10000546, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000680 = (bfloat16_t*)((char*)buf + 1543296);
    bfloat16_t *dtensor10000681 = (bfloat16_t*)((char*)buf + 1543808);
    bfloat16_t *dtensor10000682 = (bfloat16_t*)((char*)buf + 1547904);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000682,dtensor10000680,dtensor10000681, 16,128,16, 16,1, 1,16, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000682 = (bfloat16_t*)((char*)buf + 1547904);
    bfloat16_t *dtensor10000547 = (bfloat16_t*)input_tensors.at(29);
    bfloat16_t *dtensor10000683 = (bfloat16_t*)((char*)buf + 1552000);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000683,dtensor10000682,dtensor10000547, 16,2048,128, 128,1, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000671 = (bfloat16_t*)((char*)buf + 1463296);
    bfloat16_t *dtensor10000683 = (bfloat16_t*)((char*)buf + 1552000);
    bfloat16_t *dtensor10000684 = (bfloat16_t*)((char*)buf + 1617536);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000684, dtensor10000671, dtensor10000683);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000548 = (bfloat16_t*)input_tensors.at(30);
    bfloat16_t *dtensor10000685 = (bfloat16_t*)((char*)buf + 1683072);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000685,dtensor10000562,dtensor10000548, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000548 = (bfloat16_t*)input_tensors.at(30);
    bfloat16_t *dtensor10000686 = (bfloat16_t*)((char*)buf + 1687168);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000686,dtensor10000566,dtensor10000548, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000549 = (bfloat16_t*)input_tensors.at(31);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000687 = (bfloat16_t*)((char*)buf + 1691264);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000687,dtensor10000549,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000685 = (bfloat16_t*)((char*)buf + 1683072);
    bfloat16_t *dtensor10000687 = (bfloat16_t*)((char*)buf + 1691264);
    bfloat16_t *dtensor10000688 = (bfloat16_t*)((char*)buf + 1695360);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000688,dtensor10000685,dtensor10000687, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000688 = (bfloat16_t*)((char*)buf + 1695360);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000689 = (bfloat16_t*)((char*)buf + 1695872);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000689, dtensor10000688, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000689 = (bfloat16_t*)((char*)buf + 1695872);
    bfloat16_t *dtensor10000690 = (bfloat16_t*)((char*)buf + 1696384);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000690, dtensor10000689);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000690 = (bfloat16_t*)((char*)buf + 1696384);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000691 = (bfloat16_t*)((char*)buf + 1696896);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000691,dtensor10000690,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000691 = (bfloat16_t*)((char*)buf + 1696896);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000692 = (bfloat16_t*)((char*)buf + 1697024);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000692,dtensor10000691,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000690 = (bfloat16_t*)((char*)buf + 1696384);
    bfloat16_t *dtensor10000692 = (bfloat16_t*)((char*)buf + 1697024);
    bfloat16_t *dtensor10000693 = (bfloat16_t*)((char*)buf + 1697536);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000693, dtensor10000690, dtensor10000692);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000548 = (bfloat16_t*)input_tensors.at(30);
    bfloat16_t *dtensor10000694 = (bfloat16_t*)((char*)buf + 1698048);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000694,dtensor10000568,dtensor10000548, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000693 = (bfloat16_t*)((char*)buf + 1697536);
    bfloat16_t *dtensor10000694 = (bfloat16_t*)((char*)buf + 1698048);
    bfloat16_t *dtensor10000695 = (bfloat16_t*)((char*)buf + 1702144);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000695,dtensor10000693,dtensor10000694, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000695 = (bfloat16_t*)((char*)buf + 1702144);
    bfloat16_t *dtensor10000549 = (bfloat16_t*)input_tensors.at(31);
    bfloat16_t *dtensor10000696 = (bfloat16_t*)((char*)buf + 1706240);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000696,dtensor10000695,dtensor10000549, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000684 = (bfloat16_t*)((char*)buf + 1617536);
    bfloat16_t *dtensor10000696 = (bfloat16_t*)((char*)buf + 1706240);
    bfloat16_t *dtensor10000697 = (bfloat16_t*)((char*)buf + 1771776);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000697, dtensor10000684, dtensor10000696);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000550 = (bfloat16_t*)input_tensors.at(32);
    bfloat16_t *dtensor10000698 = (bfloat16_t*)((char*)buf + 1837312);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000698,dtensor10000562,dtensor10000550, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000550 = (bfloat16_t*)input_tensors.at(32);
    bfloat16_t *dtensor10000699 = (bfloat16_t*)((char*)buf + 1841408);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000699,dtensor10000566,dtensor10000550, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000551 = (bfloat16_t*)input_tensors.at(33);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000700 = (bfloat16_t*)((char*)buf + 1845504);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000700,dtensor10000551,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000698 = (bfloat16_t*)((char*)buf + 1837312);
    bfloat16_t *dtensor10000700 = (bfloat16_t*)((char*)buf + 1845504);
    bfloat16_t *dtensor10000701 = (bfloat16_t*)((char*)buf + 1849600);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000701,dtensor10000698,dtensor10000700, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000701 = (bfloat16_t*)((char*)buf + 1849600);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000702 = (bfloat16_t*)((char*)buf + 1850112);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000702, dtensor10000701, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000702 = (bfloat16_t*)((char*)buf + 1850112);
    bfloat16_t *dtensor10000703 = (bfloat16_t*)((char*)buf + 1850624);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000703, dtensor10000702);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000703 = (bfloat16_t*)((char*)buf + 1850624);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000704 = (bfloat16_t*)((char*)buf + 1851136);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000704,dtensor10000703,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000704 = (bfloat16_t*)((char*)buf + 1851136);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000705 = (bfloat16_t*)((char*)buf + 1851264);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000705,dtensor10000704,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000703 = (bfloat16_t*)((char*)buf + 1850624);
    bfloat16_t *dtensor10000705 = (bfloat16_t*)((char*)buf + 1851264);
    bfloat16_t *dtensor10000706 = (bfloat16_t*)((char*)buf + 1851776);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000706, dtensor10000703, dtensor10000705);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000550 = (bfloat16_t*)input_tensors.at(32);
    bfloat16_t *dtensor10000707 = (bfloat16_t*)((char*)buf + 1852288);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000707,dtensor10000568,dtensor10000550, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000706 = (bfloat16_t*)((char*)buf + 1851776);
    bfloat16_t *dtensor10000707 = (bfloat16_t*)((char*)buf + 1852288);
    bfloat16_t *dtensor10000708 = (bfloat16_t*)((char*)buf + 1856384);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000708,dtensor10000706,dtensor10000707, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000708 = (bfloat16_t*)((char*)buf + 1856384);
    bfloat16_t *dtensor10000551 = (bfloat16_t*)input_tensors.at(33);
    bfloat16_t *dtensor10000709 = (bfloat16_t*)((char*)buf + 1860480);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000709,dtensor10000708,dtensor10000551, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000697 = (bfloat16_t*)((char*)buf + 1771776);
    bfloat16_t *dtensor10000709 = (bfloat16_t*)((char*)buf + 1860480);
    bfloat16_t *dtensor10000710 = (bfloat16_t*)((char*)buf + 1926016);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000710, dtensor10000697, dtensor10000709);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000552 = (bfloat16_t*)input_tensors.at(34);
    bfloat16_t *dtensor10000711 = (bfloat16_t*)((char*)buf + 1991552);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000711,dtensor10000562,dtensor10000552, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000552 = (bfloat16_t*)input_tensors.at(34);
    bfloat16_t *dtensor10000712 = (bfloat16_t*)((char*)buf + 1995648);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000712,dtensor10000566,dtensor10000552, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000553 = (bfloat16_t*)input_tensors.at(35);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000713 = (bfloat16_t*)((char*)buf + 1999744);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000713,dtensor10000553,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000711 = (bfloat16_t*)((char*)buf + 1991552);
    bfloat16_t *dtensor10000713 = (bfloat16_t*)((char*)buf + 1999744);
    bfloat16_t *dtensor10000714 = (bfloat16_t*)((char*)buf + 2003840);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000714,dtensor10000711,dtensor10000713, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000714 = (bfloat16_t*)((char*)buf + 2003840);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000715 = (bfloat16_t*)((char*)buf + 2004352);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000715, dtensor10000714, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000715 = (bfloat16_t*)((char*)buf + 2004352);
    bfloat16_t *dtensor10000716 = (bfloat16_t*)((char*)buf + 2004864);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000716, dtensor10000715);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000716 = (bfloat16_t*)((char*)buf + 2004864);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000717 = (bfloat16_t*)((char*)buf + 2005376);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000717,dtensor10000716,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000717 = (bfloat16_t*)((char*)buf + 2005376);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000718 = (bfloat16_t*)((char*)buf + 2005504);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000718,dtensor10000717,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000716 = (bfloat16_t*)((char*)buf + 2004864);
    bfloat16_t *dtensor10000718 = (bfloat16_t*)((char*)buf + 2005504);
    bfloat16_t *dtensor10000719 = (bfloat16_t*)((char*)buf + 2006016);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000719, dtensor10000716, dtensor10000718);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000552 = (bfloat16_t*)input_tensors.at(34);
    bfloat16_t *dtensor10000720 = (bfloat16_t*)((char*)buf + 2006528);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000720,dtensor10000568,dtensor10000552, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000719 = (bfloat16_t*)((char*)buf + 2006016);
    bfloat16_t *dtensor10000720 = (bfloat16_t*)((char*)buf + 2006528);
    bfloat16_t *dtensor10000721 = (bfloat16_t*)((char*)buf + 2010624);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000721,dtensor10000719,dtensor10000720, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000721 = (bfloat16_t*)((char*)buf + 2010624);
    bfloat16_t *dtensor10000553 = (bfloat16_t*)input_tensors.at(35);
    bfloat16_t *dtensor10000722 = (bfloat16_t*)((char*)buf + 2014720);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000722,dtensor10000721,dtensor10000553, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000710 = (bfloat16_t*)((char*)buf + 1926016);
    bfloat16_t *dtensor10000722 = (bfloat16_t*)((char*)buf + 2014720);
    bfloat16_t *dtensor10000723 = (bfloat16_t*)((char*)buf + 2080256);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000723, dtensor10000710, dtensor10000722);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000554 = (bfloat16_t*)input_tensors.at(36);
    bfloat16_t *dtensor10000724 = (bfloat16_t*)((char*)buf + 2145792);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000724,dtensor10000562,dtensor10000554, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000554 = (bfloat16_t*)input_tensors.at(36);
    bfloat16_t *dtensor10000725 = (bfloat16_t*)((char*)buf + 2149888);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000725,dtensor10000566,dtensor10000554, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000555 = (bfloat16_t*)input_tensors.at(37);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000726 = (bfloat16_t*)((char*)buf + 2153984);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000726,dtensor10000555,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000724 = (bfloat16_t*)((char*)buf + 2145792);
    bfloat16_t *dtensor10000726 = (bfloat16_t*)((char*)buf + 2153984);
    bfloat16_t *dtensor10000727 = (bfloat16_t*)((char*)buf + 2158080);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000727,dtensor10000724,dtensor10000726, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000727 = (bfloat16_t*)((char*)buf + 2158080);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000728 = (bfloat16_t*)((char*)buf + 2158592);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000728, dtensor10000727, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000728 = (bfloat16_t*)((char*)buf + 2158592);
    bfloat16_t *dtensor10000729 = (bfloat16_t*)((char*)buf + 2159104);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000729, dtensor10000728);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000729 = (bfloat16_t*)((char*)buf + 2159104);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000730 = (bfloat16_t*)((char*)buf + 2159616);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000730,dtensor10000729,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000730 = (bfloat16_t*)((char*)buf + 2159616);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000731 = (bfloat16_t*)((char*)buf + 2159744);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000731,dtensor10000730,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000729 = (bfloat16_t*)((char*)buf + 2159104);
    bfloat16_t *dtensor10000731 = (bfloat16_t*)((char*)buf + 2159744);
    bfloat16_t *dtensor10000732 = (bfloat16_t*)((char*)buf + 2160256);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000732, dtensor10000729, dtensor10000731);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000554 = (bfloat16_t*)input_tensors.at(36);
    bfloat16_t *dtensor10000733 = (bfloat16_t*)((char*)buf + 2160768);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000733,dtensor10000568,dtensor10000554, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000732 = (bfloat16_t*)((char*)buf + 2160256);
    bfloat16_t *dtensor10000733 = (bfloat16_t*)((char*)buf + 2160768);
    bfloat16_t *dtensor10000734 = (bfloat16_t*)((char*)buf + 2164864);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000734,dtensor10000732,dtensor10000733, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000734 = (bfloat16_t*)((char*)buf + 2164864);
    bfloat16_t *dtensor10000555 = (bfloat16_t*)input_tensors.at(37);
    bfloat16_t *dtensor10000735 = (bfloat16_t*)((char*)buf + 2168960);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000735,dtensor10000734,dtensor10000555, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000723 = (bfloat16_t*)((char*)buf + 2080256);
    bfloat16_t *dtensor10000735 = (bfloat16_t*)((char*)buf + 2168960);
    bfloat16_t *dtensor10000736 = (bfloat16_t*)((char*)buf + 2234496);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000736, dtensor10000723, dtensor10000735);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000556 = (bfloat16_t*)input_tensors.at(38);
    bfloat16_t *dtensor10000737 = (bfloat16_t*)((char*)buf + 2300032);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000737,dtensor10000562,dtensor10000556, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000556 = (bfloat16_t*)input_tensors.at(38);
    bfloat16_t *dtensor10000738 = (bfloat16_t*)((char*)buf + 2304128);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000738,dtensor10000566,dtensor10000556, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000557 = (bfloat16_t*)input_tensors.at(39);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000739 = (bfloat16_t*)((char*)buf + 2308224);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000739,dtensor10000557,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000737 = (bfloat16_t*)((char*)buf + 2300032);
    bfloat16_t *dtensor10000739 = (bfloat16_t*)((char*)buf + 2308224);
    bfloat16_t *dtensor10000740 = (bfloat16_t*)((char*)buf + 2312320);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000740,dtensor10000737,dtensor10000739, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000740 = (bfloat16_t*)((char*)buf + 2312320);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000741 = (bfloat16_t*)((char*)buf + 2312832);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000741, dtensor10000740, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000741 = (bfloat16_t*)((char*)buf + 2312832);
    bfloat16_t *dtensor10000742 = (bfloat16_t*)((char*)buf + 2313344);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000742, dtensor10000741);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000742 = (bfloat16_t*)((char*)buf + 2313344);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000743 = (bfloat16_t*)((char*)buf + 2313856);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000743,dtensor10000742,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000743 = (bfloat16_t*)((char*)buf + 2313856);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000744 = (bfloat16_t*)((char*)buf + 2313984);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000744,dtensor10000743,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000742 = (bfloat16_t*)((char*)buf + 2313344);
    bfloat16_t *dtensor10000744 = (bfloat16_t*)((char*)buf + 2313984);
    bfloat16_t *dtensor10000745 = (bfloat16_t*)((char*)buf + 2314496);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000745, dtensor10000742, dtensor10000744);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000556 = (bfloat16_t*)input_tensors.at(38);
    bfloat16_t *dtensor10000746 = (bfloat16_t*)((char*)buf + 2315008);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000746,dtensor10000568,dtensor10000556, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000745 = (bfloat16_t*)((char*)buf + 2314496);
    bfloat16_t *dtensor10000746 = (bfloat16_t*)((char*)buf + 2315008);
    bfloat16_t *dtensor10000747 = (bfloat16_t*)((char*)buf + 2319104);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000747,dtensor10000745,dtensor10000746, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000747 = (bfloat16_t*)((char*)buf + 2319104);
    bfloat16_t *dtensor10000557 = (bfloat16_t*)input_tensors.at(39);
    bfloat16_t *dtensor10000748 = (bfloat16_t*)((char*)buf + 2323200);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000748,dtensor10000747,dtensor10000557, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000736 = (bfloat16_t*)((char*)buf + 2234496);
    bfloat16_t *dtensor10000748 = (bfloat16_t*)((char*)buf + 2323200);
    bfloat16_t *dtensor10000749 = (bfloat16_t*)((char*)buf + 2388736);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000749, dtensor10000736, dtensor10000748);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000558 = (bfloat16_t*)input_tensors.at(40);
    bfloat16_t *dtensor10000750 = (bfloat16_t*)((char*)buf + 2454272);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000750,dtensor10000562,dtensor10000558, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000558 = (bfloat16_t*)input_tensors.at(40);
    bfloat16_t *dtensor10000751 = (bfloat16_t*)((char*)buf + 2458368);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000751,dtensor10000566,dtensor10000558, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000559 = (bfloat16_t*)input_tensors.at(41);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000752 = (bfloat16_t*)((char*)buf + 2462464);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000752,dtensor10000559,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000750 = (bfloat16_t*)((char*)buf + 2454272);
    bfloat16_t *dtensor10000752 = (bfloat16_t*)((char*)buf + 2462464);
    bfloat16_t *dtensor10000753 = (bfloat16_t*)((char*)buf + 2466560);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000753,dtensor10000750,dtensor10000752, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000753 = (bfloat16_t*)((char*)buf + 2466560);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000754 = (bfloat16_t*)((char*)buf + 2467072);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000754, dtensor10000753, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000754 = (bfloat16_t*)((char*)buf + 2467072);
    bfloat16_t *dtensor10000755 = (bfloat16_t*)((char*)buf + 2467584);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000755, dtensor10000754);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000755 = (bfloat16_t*)((char*)buf + 2467584);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000756 = (bfloat16_t*)((char*)buf + 2468096);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000756,dtensor10000755,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000756 = (bfloat16_t*)((char*)buf + 2468096);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000757 = (bfloat16_t*)((char*)buf + 2468224);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000757,dtensor10000756,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000755 = (bfloat16_t*)((char*)buf + 2467584);
    bfloat16_t *dtensor10000757 = (bfloat16_t*)((char*)buf + 2468224);
    bfloat16_t *dtensor10000758 = (bfloat16_t*)((char*)buf + 2468736);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000758, dtensor10000755, dtensor10000757);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000558 = (bfloat16_t*)input_tensors.at(40);
    bfloat16_t *dtensor10000759 = (bfloat16_t*)((char*)buf + 2469248);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000759,dtensor10000568,dtensor10000558, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000758 = (bfloat16_t*)((char*)buf + 2468736);
    bfloat16_t *dtensor10000759 = (bfloat16_t*)((char*)buf + 2469248);
    bfloat16_t *dtensor10000760 = (bfloat16_t*)((char*)buf + 2473344);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000760,dtensor10000758,dtensor10000759, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000760 = (bfloat16_t*)((char*)buf + 2473344);
    bfloat16_t *dtensor10000559 = (bfloat16_t*)input_tensors.at(41);
    bfloat16_t *dtensor10000761 = (bfloat16_t*)((char*)buf + 2477440);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000761,dtensor10000760,dtensor10000559, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000749 = (bfloat16_t*)((char*)buf + 2388736);
    bfloat16_t *dtensor10000761 = (bfloat16_t*)((char*)buf + 2477440);
    bfloat16_t *dtensor10000762 = (bfloat16_t*)((char*)buf + 2542976);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000762, dtensor10000749, dtensor10000761);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000562 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10000560 = (bfloat16_t*)input_tensors.at(42);
    bfloat16_t *dtensor10000763 = (bfloat16_t*)((char*)buf + 2608512);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000763,dtensor10000562,dtensor10000560, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000566 = (bfloat16_t*)((char*)buf + 163840);
    bfloat16_t *dtensor10000560 = (bfloat16_t*)input_tensors.at(42);
    bfloat16_t *dtensor10000764 = (bfloat16_t*)((char*)buf + 2612608);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000764,dtensor10000566,dtensor10000560, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000561 = (bfloat16_t*)input_tensors.at(43);
    bfloat16_t *dtensor10000567 = (bfloat16_t*)((char*)buf + 229376);
    bfloat16_t *dtensor10000765 = (bfloat16_t*)((char*)buf + 2616704);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000765,dtensor10000561,dtensor10000567, 128,16,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000763 = (bfloat16_t*)((char*)buf + 2608512);
    bfloat16_t *dtensor10000765 = (bfloat16_t*)((char*)buf + 2616704);
    bfloat16_t *dtensor10000766 = (bfloat16_t*)((char*)buf + 2620800);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000766,dtensor10000763,dtensor10000765, 16,16,128, 1,16, 1,128, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10000766 = (bfloat16_t*)((char*)buf + 2620800);
    bfloat16_t *dtensor10000527 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10000767 = (bfloat16_t*)((char*)buf + 2621312);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000767, dtensor10000766, dtensor10000527);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10000767 = (bfloat16_t*)((char*)buf + 2621312);
    bfloat16_t *dtensor10000768 = (bfloat16_t*)((char*)buf + 2621824);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000768, dtensor10000767);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000768 = (bfloat16_t*)((char*)buf + 2621824);
    bfloat16_t *dtensor10000528 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10000769 = (bfloat16_t*)((char*)buf + 2622336);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000769,dtensor10000768,dtensor10000528, 16,1,16, 16,1, 1,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000769 = (bfloat16_t*)((char*)buf + 2622336);
    bfloat16_t *dtensor10000529 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10000770 = (bfloat16_t*)((char*)buf + 2622464);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000770,dtensor10000769,dtensor10000529, 16,16,1, 1,16, 16,1, 16,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10000768 = (bfloat16_t*)((char*)buf + 2621824);
    bfloat16_t *dtensor10000770 = (bfloat16_t*)((char*)buf + 2622464);
    bfloat16_t *dtensor10000771 = (bfloat16_t*)((char*)buf + 2622976);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<16>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000771, dtensor10000768, dtensor10000770);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000568 = (bfloat16_t*)((char*)buf + 294912);
    bfloat16_t *dtensor10000560 = (bfloat16_t*)input_tensors.at(42);
    bfloat16_t *dtensor10000772 = (bfloat16_t*)((char*)buf + 2623488);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000772,dtensor10000568,dtensor10000560, 16,128,2048, 1,16, 128,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000771 = (bfloat16_t*)((char*)buf + 2622976);
    bfloat16_t *dtensor10000772 = (bfloat16_t*)((char*)buf + 2623488);
    bfloat16_t *dtensor10000773 = (bfloat16_t*)((char*)buf + 2627584);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000773,dtensor10000771,dtensor10000772, 16,128,16, 16,1, 1,16, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000773 = (bfloat16_t*)((char*)buf + 2627584);
    bfloat16_t *dtensor10000561 = (bfloat16_t*)input_tensors.at(43);
    bfloat16_t *dtensor10000774 = (bfloat16_t*)((char*)buf + 2631680);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000774,dtensor10000773,dtensor10000561, 16,2048,128, 1,16, 2048,1, 1,16, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10000762 = (bfloat16_t*)((char*)buf + 2542976);
    bfloat16_t *dtensor10000774 = (bfloat16_t*)((char*)buf + 2631680);
    bfloat16_t *dtensor10000775 = (bfloat16_t*)((char*)buf + 2697216);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>, Layout<Shape<Int<16>, Int<2048>>, Stride<Int<1>, Int<16>>>>;
    kernel::run(dtensor10000775, dtensor10000762, dtensor10000774);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10000775 = (bfloat16_t*)((char*)buf + 2697216);
    bfloat16_t *dtensor10000523 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10000776 = (bfloat16_t*)output_tensors.at(0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10000776,dtensor10000775,dtensor10000523, 16,1024,2048, 1,16, 1024,1, 1,16, 1, 0,0,0);
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

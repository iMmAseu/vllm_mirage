#define execute_mugraph attention_execute_mugraph_batch8
#define PyInit___mirage_launcher attention_PyInit_mirage_batch8
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
    bfloat16_t *dtensor10002784 = (bfloat16_t*)input_tensors.at(1);
    bfloat16_t *dtensor10002788 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002825,dtensor10002784,dtensor10002788, 128,2048,1024, 1024,1, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002789 = (bfloat16_t*)input_tensors.at(6);
    bfloat16_t *dtensor10002785 = (bfloat16_t*)input_tensors.at(2);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002826,dtensor10002789,dtensor10002785, 2048,128,1024, 1024,1, 128,1, 1,2048, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002786 = (bfloat16_t*)input_tensors.at(3);
    bfloat16_t *dtensor10002788 = (bfloat16_t*)input_tensors.at(5);
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002827,dtensor10002786,dtensor10002788, 128,2048,1024, 1024,1, 2048,1, 2048,1, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002793 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10002828 = (bfloat16_t*)((char*)buf + 1572864);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002828,dtensor10002783,dtensor10002793, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002793 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10002829 = (bfloat16_t*)((char*)buf + 1605632);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002829,dtensor10002825,dtensor10002793, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002794 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002830 = (bfloat16_t*)((char*)buf + 1638400);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002830,dtensor10002794,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002828 = (bfloat16_t*)((char*)buf + 1572864);
    bfloat16_t *dtensor10002830 = (bfloat16_t*)((char*)buf + 1638400);
    bfloat16_t *dtensor10002831 = (bfloat16_t*)((char*)buf + 1671168);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002831,dtensor10002828,dtensor10002830, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002831 = (bfloat16_t*)((char*)buf + 1671168);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002832 = (bfloat16_t*)((char*)buf + 1703936);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002832, dtensor10002831, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002832 = (bfloat16_t*)((char*)buf + 1703936);
    bfloat16_t *dtensor10002833 = (bfloat16_t*)((char*)buf + 1736704);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002833, dtensor10002832);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002833 = (bfloat16_t*)((char*)buf + 1736704);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002834 = (bfloat16_t*)((char*)buf + 1769472);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002834,dtensor10002833,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002834 = (bfloat16_t*)((char*)buf + 1769472);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002835 = (bfloat16_t*)((char*)buf + 1769728);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002835,dtensor10002834,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002833 = (bfloat16_t*)((char*)buf + 1736704);
    bfloat16_t *dtensor10002835 = (bfloat16_t*)((char*)buf + 1769728);
    bfloat16_t *dtensor10002836 = (bfloat16_t*)((char*)buf + 1802496);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002836, dtensor10002833, dtensor10002835);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002793 = (bfloat16_t*)input_tensors.at(10);
    bfloat16_t *dtensor10002837 = (bfloat16_t*)((char*)buf + 1835264);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002837,dtensor10002827,dtensor10002793, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002836 = (bfloat16_t*)((char*)buf + 1802496);
    bfloat16_t *dtensor10002837 = (bfloat16_t*)((char*)buf + 1835264);
    bfloat16_t *dtensor10002838 = (bfloat16_t*)((char*)buf + 1868032);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002838,dtensor10002836,dtensor10002837, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002838 = (bfloat16_t*)((char*)buf + 1868032);
    bfloat16_t *dtensor10002794 = (bfloat16_t*)input_tensors.at(11);
    bfloat16_t *dtensor10002839 = (bfloat16_t*)((char*)buf + 1900800);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002839,dtensor10002838,dtensor10002794, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002795 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10002840 = (bfloat16_t*)((char*)buf + 2425088);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002840,dtensor10002783,dtensor10002795, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002795 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10002841 = (bfloat16_t*)((char*)buf + 2457856);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002841,dtensor10002825,dtensor10002795, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002796 = (bfloat16_t*)input_tensors.at(13);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002842 = (bfloat16_t*)((char*)buf + 2490624);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002842,dtensor10002796,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002840 = (bfloat16_t*)((char*)buf + 2425088);
    bfloat16_t *dtensor10002842 = (bfloat16_t*)((char*)buf + 2490624);
    bfloat16_t *dtensor10002843 = (bfloat16_t*)((char*)buf + 2523392);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002843,dtensor10002840,dtensor10002842, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002843 = (bfloat16_t*)((char*)buf + 2523392);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002844 = (bfloat16_t*)((char*)buf + 2556160);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002844, dtensor10002843, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002844 = (bfloat16_t*)((char*)buf + 2556160);
    bfloat16_t *dtensor10002845 = (bfloat16_t*)((char*)buf + 2588928);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002845, dtensor10002844);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002845 = (bfloat16_t*)((char*)buf + 2588928);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002846 = (bfloat16_t*)((char*)buf + 2621696);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002846,dtensor10002845,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002846 = (bfloat16_t*)((char*)buf + 2621696);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002847 = (bfloat16_t*)((char*)buf + 2621952);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002847,dtensor10002846,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002845 = (bfloat16_t*)((char*)buf + 2588928);
    bfloat16_t *dtensor10002847 = (bfloat16_t*)((char*)buf + 2621952);
    bfloat16_t *dtensor10002848 = (bfloat16_t*)((char*)buf + 2654720);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002848, dtensor10002845, dtensor10002847);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002795 = (bfloat16_t*)input_tensors.at(12);
    bfloat16_t *dtensor10002849 = (bfloat16_t*)((char*)buf + 2687488);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002849,dtensor10002827,dtensor10002795, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002848 = (bfloat16_t*)((char*)buf + 2654720);
    bfloat16_t *dtensor10002849 = (bfloat16_t*)((char*)buf + 2687488);
    bfloat16_t *dtensor10002850 = (bfloat16_t*)((char*)buf + 2720256);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002850,dtensor10002848,dtensor10002849, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002850 = (bfloat16_t*)((char*)buf + 2720256);
    bfloat16_t *dtensor10002796 = (bfloat16_t*)input_tensors.at(13);
    bfloat16_t *dtensor10002851 = (bfloat16_t*)((char*)buf + 2753024);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002851,dtensor10002850,dtensor10002796, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002839 = (bfloat16_t*)((char*)buf + 1900800);
    bfloat16_t *dtensor10002851 = (bfloat16_t*)((char*)buf + 2753024);
    bfloat16_t *dtensor10002852 = (bfloat16_t*)((char*)buf + 3277312);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002852, dtensor10002839, dtensor10002851);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002797 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10002853 = (bfloat16_t*)((char*)buf + 3801600);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002853,dtensor10002783,dtensor10002797, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002797 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10002854 = (bfloat16_t*)((char*)buf + 3834368);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002854,dtensor10002825,dtensor10002797, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002798 = (bfloat16_t*)input_tensors.at(15);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002855 = (bfloat16_t*)((char*)buf + 3867136);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002855,dtensor10002798,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002853 = (bfloat16_t*)((char*)buf + 3801600);
    bfloat16_t *dtensor10002855 = (bfloat16_t*)((char*)buf + 3867136);
    bfloat16_t *dtensor10002856 = (bfloat16_t*)((char*)buf + 3899904);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002856,dtensor10002853,dtensor10002855, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002856 = (bfloat16_t*)((char*)buf + 3899904);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002857 = (bfloat16_t*)((char*)buf + 3932672);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002857, dtensor10002856, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002857 = (bfloat16_t*)((char*)buf + 3932672);
    bfloat16_t *dtensor10002858 = (bfloat16_t*)((char*)buf + 3965440);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002858, dtensor10002857);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002858 = (bfloat16_t*)((char*)buf + 3965440);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002859 = (bfloat16_t*)((char*)buf + 3998208);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002859,dtensor10002858,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002859 = (bfloat16_t*)((char*)buf + 3998208);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002860 = (bfloat16_t*)((char*)buf + 3998464);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002860,dtensor10002859,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002858 = (bfloat16_t*)((char*)buf + 3965440);
    bfloat16_t *dtensor10002860 = (bfloat16_t*)((char*)buf + 3998464);
    bfloat16_t *dtensor10002861 = (bfloat16_t*)((char*)buf + 4031232);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002861, dtensor10002858, dtensor10002860);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002797 = (bfloat16_t*)input_tensors.at(14);
    bfloat16_t *dtensor10002862 = (bfloat16_t*)((char*)buf + 4064000);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002862,dtensor10002827,dtensor10002797, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002861 = (bfloat16_t*)((char*)buf + 4031232);
    bfloat16_t *dtensor10002862 = (bfloat16_t*)((char*)buf + 4064000);
    bfloat16_t *dtensor10002863 = (bfloat16_t*)((char*)buf + 4096768);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002863,dtensor10002861,dtensor10002862, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002863 = (bfloat16_t*)((char*)buf + 4096768);
    bfloat16_t *dtensor10002798 = (bfloat16_t*)input_tensors.at(15);
    bfloat16_t *dtensor10002864 = (bfloat16_t*)((char*)buf + 4129536);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002864,dtensor10002863,dtensor10002798, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002852 = (bfloat16_t*)((char*)buf + 3277312);
    bfloat16_t *dtensor10002864 = (bfloat16_t*)((char*)buf + 4129536);
    bfloat16_t *dtensor10002865 = (bfloat16_t*)((char*)buf + 4653824);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002865, dtensor10002852, dtensor10002864);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002799 = (bfloat16_t*)input_tensors.at(16);
    bfloat16_t *dtensor10002866 = (bfloat16_t*)((char*)buf + 5178112);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002866,dtensor10002783,dtensor10002799, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002799 = (bfloat16_t*)input_tensors.at(16);
    bfloat16_t *dtensor10002867 = (bfloat16_t*)((char*)buf + 5210880);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002867,dtensor10002825,dtensor10002799, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002800 = (bfloat16_t*)input_tensors.at(17);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002868 = (bfloat16_t*)((char*)buf + 5243648);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002868,dtensor10002800,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002866 = (bfloat16_t*)((char*)buf + 5178112);
    bfloat16_t *dtensor10002868 = (bfloat16_t*)((char*)buf + 5243648);
    bfloat16_t *dtensor10002869 = (bfloat16_t*)((char*)buf + 5276416);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002869,dtensor10002866,dtensor10002868, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002869 = (bfloat16_t*)((char*)buf + 5276416);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002870 = (bfloat16_t*)((char*)buf + 5309184);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002870, dtensor10002869, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002870 = (bfloat16_t*)((char*)buf + 5309184);
    bfloat16_t *dtensor10002871 = (bfloat16_t*)((char*)buf + 5341952);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002871, dtensor10002870);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002871 = (bfloat16_t*)((char*)buf + 5341952);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002872 = (bfloat16_t*)((char*)buf + 5374720);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002872,dtensor10002871,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002872 = (bfloat16_t*)((char*)buf + 5374720);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002873 = (bfloat16_t*)((char*)buf + 5374976);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002873,dtensor10002872,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002871 = (bfloat16_t*)((char*)buf + 5341952);
    bfloat16_t *dtensor10002873 = (bfloat16_t*)((char*)buf + 5374976);
    bfloat16_t *dtensor10002874 = (bfloat16_t*)((char*)buf + 5407744);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002874, dtensor10002871, dtensor10002873);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002799 = (bfloat16_t*)input_tensors.at(16);
    bfloat16_t *dtensor10002875 = (bfloat16_t*)((char*)buf + 5440512);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002875,dtensor10002827,dtensor10002799, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002874 = (bfloat16_t*)((char*)buf + 5407744);
    bfloat16_t *dtensor10002875 = (bfloat16_t*)((char*)buf + 5440512);
    bfloat16_t *dtensor10002876 = (bfloat16_t*)((char*)buf + 5473280);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002876,dtensor10002874,dtensor10002875, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002876 = (bfloat16_t*)((char*)buf + 5473280);
    bfloat16_t *dtensor10002800 = (bfloat16_t*)input_tensors.at(17);
    bfloat16_t *dtensor10002877 = (bfloat16_t*)((char*)buf + 5506048);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002877,dtensor10002876,dtensor10002800, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002865 = (bfloat16_t*)((char*)buf + 4653824);
    bfloat16_t *dtensor10002877 = (bfloat16_t*)((char*)buf + 5506048);
    bfloat16_t *dtensor10002878 = (bfloat16_t*)((char*)buf + 6030336);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002878, dtensor10002865, dtensor10002877);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002801 = (bfloat16_t*)input_tensors.at(18);
    bfloat16_t *dtensor10002879 = (bfloat16_t*)((char*)buf + 6554624);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002879,dtensor10002783,dtensor10002801, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002801 = (bfloat16_t*)input_tensors.at(18);
    bfloat16_t *dtensor10002880 = (bfloat16_t*)((char*)buf + 6587392);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002880,dtensor10002825,dtensor10002801, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002802 = (bfloat16_t*)input_tensors.at(19);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002881 = (bfloat16_t*)((char*)buf + 6620160);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002881,dtensor10002802,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002879 = (bfloat16_t*)((char*)buf + 6554624);
    bfloat16_t *dtensor10002881 = (bfloat16_t*)((char*)buf + 6620160);
    bfloat16_t *dtensor10002882 = (bfloat16_t*)((char*)buf + 6652928);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002882,dtensor10002879,dtensor10002881, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002882 = (bfloat16_t*)((char*)buf + 6652928);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002883 = (bfloat16_t*)((char*)buf + 6685696);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002883, dtensor10002882, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002883 = (bfloat16_t*)((char*)buf + 6685696);
    bfloat16_t *dtensor10002884 = (bfloat16_t*)((char*)buf + 6718464);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002884, dtensor10002883);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002884 = (bfloat16_t*)((char*)buf + 6718464);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002885 = (bfloat16_t*)((char*)buf + 6751232);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002885,dtensor10002884,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002885 = (bfloat16_t*)((char*)buf + 6751232);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002886 = (bfloat16_t*)((char*)buf + 6751488);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002886,dtensor10002885,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002884 = (bfloat16_t*)((char*)buf + 6718464);
    bfloat16_t *dtensor10002886 = (bfloat16_t*)((char*)buf + 6751488);
    bfloat16_t *dtensor10002887 = (bfloat16_t*)((char*)buf + 6784256);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002887, dtensor10002884, dtensor10002886);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002801 = (bfloat16_t*)input_tensors.at(18);
    bfloat16_t *dtensor10002888 = (bfloat16_t*)((char*)buf + 6817024);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002888,dtensor10002827,dtensor10002801, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002887 = (bfloat16_t*)((char*)buf + 6784256);
    bfloat16_t *dtensor10002888 = (bfloat16_t*)((char*)buf + 6817024);
    bfloat16_t *dtensor10002889 = (bfloat16_t*)((char*)buf + 6849792);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002889,dtensor10002887,dtensor10002888, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002889 = (bfloat16_t*)((char*)buf + 6849792);
    bfloat16_t *dtensor10002802 = (bfloat16_t*)input_tensors.at(19);
    bfloat16_t *dtensor10002890 = (bfloat16_t*)((char*)buf + 6882560);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002890,dtensor10002889,dtensor10002802, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002878 = (bfloat16_t*)((char*)buf + 6030336);
    bfloat16_t *dtensor10002890 = (bfloat16_t*)((char*)buf + 6882560);
    bfloat16_t *dtensor10002891 = (bfloat16_t*)((char*)buf + 7406848);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002891, dtensor10002878, dtensor10002890);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002803 = (bfloat16_t*)input_tensors.at(20);
    bfloat16_t *dtensor10002892 = (bfloat16_t*)((char*)buf + 7931136);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002892,dtensor10002783,dtensor10002803, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002803 = (bfloat16_t*)input_tensors.at(20);
    bfloat16_t *dtensor10002893 = (bfloat16_t*)((char*)buf + 7963904);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002893,dtensor10002825,dtensor10002803, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002804 = (bfloat16_t*)input_tensors.at(21);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002894 = (bfloat16_t*)((char*)buf + 7996672);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002894,dtensor10002804,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002892 = (bfloat16_t*)((char*)buf + 7931136);
    bfloat16_t *dtensor10002894 = (bfloat16_t*)((char*)buf + 7996672);
    bfloat16_t *dtensor10002895 = (bfloat16_t*)((char*)buf + 8029440);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002895,dtensor10002892,dtensor10002894, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002895 = (bfloat16_t*)((char*)buf + 8029440);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002896 = (bfloat16_t*)((char*)buf + 8062208);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002896, dtensor10002895, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002896 = (bfloat16_t*)((char*)buf + 8062208);
    bfloat16_t *dtensor10002897 = (bfloat16_t*)((char*)buf + 8094976);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002897, dtensor10002896);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002897 = (bfloat16_t*)((char*)buf + 8094976);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002898 = (bfloat16_t*)((char*)buf + 8127744);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002898,dtensor10002897,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002898 = (bfloat16_t*)((char*)buf + 8127744);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002899 = (bfloat16_t*)((char*)buf + 8128000);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002899,dtensor10002898,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002897 = (bfloat16_t*)((char*)buf + 8094976);
    bfloat16_t *dtensor10002899 = (bfloat16_t*)((char*)buf + 8128000);
    bfloat16_t *dtensor10002900 = (bfloat16_t*)((char*)buf + 8160768);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002900, dtensor10002897, dtensor10002899);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002803 = (bfloat16_t*)input_tensors.at(20);
    bfloat16_t *dtensor10002901 = (bfloat16_t*)((char*)buf + 8193536);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002901,dtensor10002827,dtensor10002803, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002900 = (bfloat16_t*)((char*)buf + 8160768);
    bfloat16_t *dtensor10002901 = (bfloat16_t*)((char*)buf + 8193536);
    bfloat16_t *dtensor10002902 = (bfloat16_t*)((char*)buf + 8226304);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002902,dtensor10002900,dtensor10002901, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002902 = (bfloat16_t*)((char*)buf + 8226304);
    bfloat16_t *dtensor10002804 = (bfloat16_t*)input_tensors.at(21);
    bfloat16_t *dtensor10002903 = (bfloat16_t*)((char*)buf + 8259072);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002903,dtensor10002902,dtensor10002804, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002891 = (bfloat16_t*)((char*)buf + 7406848);
    bfloat16_t *dtensor10002903 = (bfloat16_t*)((char*)buf + 8259072);
    bfloat16_t *dtensor10002904 = (bfloat16_t*)((char*)buf + 8783360);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002904, dtensor10002891, dtensor10002903);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002805 = (bfloat16_t*)input_tensors.at(22);
    bfloat16_t *dtensor10002905 = (bfloat16_t*)((char*)buf + 9307648);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002905,dtensor10002783,dtensor10002805, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002805 = (bfloat16_t*)input_tensors.at(22);
    bfloat16_t *dtensor10002906 = (bfloat16_t*)((char*)buf + 9340416);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002906,dtensor10002825,dtensor10002805, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002806 = (bfloat16_t*)input_tensors.at(23);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002907 = (bfloat16_t*)((char*)buf + 9373184);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002907,dtensor10002806,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002905 = (bfloat16_t*)((char*)buf + 9307648);
    bfloat16_t *dtensor10002907 = (bfloat16_t*)((char*)buf + 9373184);
    bfloat16_t *dtensor10002908 = (bfloat16_t*)((char*)buf + 9405952);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002908,dtensor10002905,dtensor10002907, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002908 = (bfloat16_t*)((char*)buf + 9405952);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002909 = (bfloat16_t*)((char*)buf + 9438720);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002909, dtensor10002908, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002909 = (bfloat16_t*)((char*)buf + 9438720);
    bfloat16_t *dtensor10002910 = (bfloat16_t*)((char*)buf + 9471488);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002910, dtensor10002909);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002910 = (bfloat16_t*)((char*)buf + 9471488);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002911 = (bfloat16_t*)((char*)buf + 9504256);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002911,dtensor10002910,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002911 = (bfloat16_t*)((char*)buf + 9504256);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002912 = (bfloat16_t*)((char*)buf + 9504512);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002912,dtensor10002911,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002910 = (bfloat16_t*)((char*)buf + 9471488);
    bfloat16_t *dtensor10002912 = (bfloat16_t*)((char*)buf + 9504512);
    bfloat16_t *dtensor10002913 = (bfloat16_t*)((char*)buf + 9537280);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002913, dtensor10002910, dtensor10002912);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002805 = (bfloat16_t*)input_tensors.at(22);
    bfloat16_t *dtensor10002914 = (bfloat16_t*)((char*)buf + 9570048);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002914,dtensor10002827,dtensor10002805, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002913 = (bfloat16_t*)((char*)buf + 9537280);
    bfloat16_t *dtensor10002914 = (bfloat16_t*)((char*)buf + 9570048);
    bfloat16_t *dtensor10002915 = (bfloat16_t*)((char*)buf + 9602816);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002915,dtensor10002913,dtensor10002914, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002915 = (bfloat16_t*)((char*)buf + 9602816);
    bfloat16_t *dtensor10002806 = (bfloat16_t*)input_tensors.at(23);
    bfloat16_t *dtensor10002916 = (bfloat16_t*)((char*)buf + 9635584);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002916,dtensor10002915,dtensor10002806, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002904 = (bfloat16_t*)((char*)buf + 8783360);
    bfloat16_t *dtensor10002916 = (bfloat16_t*)((char*)buf + 9635584);
    bfloat16_t *dtensor10002917 = (bfloat16_t*)((char*)buf + 10159872);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002917, dtensor10002904, dtensor10002916);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002807 = (bfloat16_t*)input_tensors.at(24);
    bfloat16_t *dtensor10002918 = (bfloat16_t*)((char*)buf + 10684160);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002918,dtensor10002783,dtensor10002807, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002807 = (bfloat16_t*)input_tensors.at(24);
    bfloat16_t *dtensor10002919 = (bfloat16_t*)((char*)buf + 10716928);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002919,dtensor10002825,dtensor10002807, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002808 = (bfloat16_t*)input_tensors.at(25);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002920 = (bfloat16_t*)((char*)buf + 10749696);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002920,dtensor10002808,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002918 = (bfloat16_t*)((char*)buf + 10684160);
    bfloat16_t *dtensor10002920 = (bfloat16_t*)((char*)buf + 10749696);
    bfloat16_t *dtensor10002921 = (bfloat16_t*)((char*)buf + 10782464);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002921,dtensor10002918,dtensor10002920, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002921 = (bfloat16_t*)((char*)buf + 10782464);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002922 = (bfloat16_t*)((char*)buf + 10815232);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002922, dtensor10002921, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002922 = (bfloat16_t*)((char*)buf + 10815232);
    bfloat16_t *dtensor10002923 = (bfloat16_t*)((char*)buf + 10848000);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002923, dtensor10002922);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002923 = (bfloat16_t*)((char*)buf + 10848000);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002924 = (bfloat16_t*)((char*)buf + 10880768);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002924,dtensor10002923,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002924 = (bfloat16_t*)((char*)buf + 10880768);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002925 = (bfloat16_t*)((char*)buf + 10881024);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002925,dtensor10002924,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002923 = (bfloat16_t*)((char*)buf + 10848000);
    bfloat16_t *dtensor10002925 = (bfloat16_t*)((char*)buf + 10881024);
    bfloat16_t *dtensor10002926 = (bfloat16_t*)((char*)buf + 10913792);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002926, dtensor10002923, dtensor10002925);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002807 = (bfloat16_t*)input_tensors.at(24);
    bfloat16_t *dtensor10002927 = (bfloat16_t*)((char*)buf + 10946560);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002927,dtensor10002827,dtensor10002807, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002926 = (bfloat16_t*)((char*)buf + 10913792);
    bfloat16_t *dtensor10002927 = (bfloat16_t*)((char*)buf + 10946560);
    bfloat16_t *dtensor10002928 = (bfloat16_t*)((char*)buf + 10979328);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002928,dtensor10002926,dtensor10002927, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002928 = (bfloat16_t*)((char*)buf + 10979328);
    bfloat16_t *dtensor10002808 = (bfloat16_t*)input_tensors.at(25);
    bfloat16_t *dtensor10002929 = (bfloat16_t*)((char*)buf + 11012096);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002929,dtensor10002928,dtensor10002808, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002917 = (bfloat16_t*)((char*)buf + 10159872);
    bfloat16_t *dtensor10002929 = (bfloat16_t*)((char*)buf + 11012096);
    bfloat16_t *dtensor10002930 = (bfloat16_t*)((char*)buf + 11536384);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002930, dtensor10002917, dtensor10002929);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002809 = (bfloat16_t*)input_tensors.at(26);
    bfloat16_t *dtensor10002931 = (bfloat16_t*)((char*)buf + 12060672);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002931,dtensor10002783,dtensor10002809, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002809 = (bfloat16_t*)input_tensors.at(26);
    bfloat16_t *dtensor10002932 = (bfloat16_t*)((char*)buf + 12093440);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002932,dtensor10002825,dtensor10002809, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002810 = (bfloat16_t*)input_tensors.at(27);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002933 = (bfloat16_t*)((char*)buf + 12126208);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002933,dtensor10002810,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002931 = (bfloat16_t*)((char*)buf + 12060672);
    bfloat16_t *dtensor10002933 = (bfloat16_t*)((char*)buf + 12126208);
    bfloat16_t *dtensor10002934 = (bfloat16_t*)((char*)buf + 12158976);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002934,dtensor10002931,dtensor10002933, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002934 = (bfloat16_t*)((char*)buf + 12158976);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002935 = (bfloat16_t*)((char*)buf + 12191744);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002935, dtensor10002934, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002935 = (bfloat16_t*)((char*)buf + 12191744);
    bfloat16_t *dtensor10002936 = (bfloat16_t*)((char*)buf + 12224512);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002936, dtensor10002935);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002936 = (bfloat16_t*)((char*)buf + 12224512);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002937 = (bfloat16_t*)((char*)buf + 12257280);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002937,dtensor10002936,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002937 = (bfloat16_t*)((char*)buf + 12257280);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002938 = (bfloat16_t*)((char*)buf + 12257536);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002938,dtensor10002937,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002936 = (bfloat16_t*)((char*)buf + 12224512);
    bfloat16_t *dtensor10002938 = (bfloat16_t*)((char*)buf + 12257536);
    bfloat16_t *dtensor10002939 = (bfloat16_t*)((char*)buf + 12290304);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002939, dtensor10002936, dtensor10002938);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002809 = (bfloat16_t*)input_tensors.at(26);
    bfloat16_t *dtensor10002940 = (bfloat16_t*)((char*)buf + 12323072);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002940,dtensor10002827,dtensor10002809, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002939 = (bfloat16_t*)((char*)buf + 12290304);
    bfloat16_t *dtensor10002940 = (bfloat16_t*)((char*)buf + 12323072);
    bfloat16_t *dtensor10002941 = (bfloat16_t*)((char*)buf + 12355840);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002941,dtensor10002939,dtensor10002940, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002941 = (bfloat16_t*)((char*)buf + 12355840);
    bfloat16_t *dtensor10002810 = (bfloat16_t*)input_tensors.at(27);
    bfloat16_t *dtensor10002942 = (bfloat16_t*)((char*)buf + 12388608);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002942,dtensor10002941,dtensor10002810, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002930 = (bfloat16_t*)((char*)buf + 11536384);
    bfloat16_t *dtensor10002942 = (bfloat16_t*)((char*)buf + 12388608);
    bfloat16_t *dtensor10002943 = (bfloat16_t*)((char*)buf + 12912896);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002943, dtensor10002930, dtensor10002942);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002811 = (bfloat16_t*)input_tensors.at(28);
    bfloat16_t *dtensor10002944 = (bfloat16_t*)((char*)buf + 13437184);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002944,dtensor10002783,dtensor10002811, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002811 = (bfloat16_t*)input_tensors.at(28);
    bfloat16_t *dtensor10002945 = (bfloat16_t*)((char*)buf + 13469952);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002945,dtensor10002825,dtensor10002811, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002812 = (bfloat16_t*)input_tensors.at(29);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002946 = (bfloat16_t*)((char*)buf + 13502720);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002946,dtensor10002812,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002944 = (bfloat16_t*)((char*)buf + 13437184);
    bfloat16_t *dtensor10002946 = (bfloat16_t*)((char*)buf + 13502720);
    bfloat16_t *dtensor10002947 = (bfloat16_t*)((char*)buf + 13535488);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002947,dtensor10002944,dtensor10002946, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002947 = (bfloat16_t*)((char*)buf + 13535488);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002948 = (bfloat16_t*)((char*)buf + 13568256);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002948, dtensor10002947, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002948 = (bfloat16_t*)((char*)buf + 13568256);
    bfloat16_t *dtensor10002949 = (bfloat16_t*)((char*)buf + 13601024);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002949, dtensor10002948);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002949 = (bfloat16_t*)((char*)buf + 13601024);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002950 = (bfloat16_t*)((char*)buf + 13633792);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002950,dtensor10002949,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002950 = (bfloat16_t*)((char*)buf + 13633792);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002951 = (bfloat16_t*)((char*)buf + 13634048);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002951,dtensor10002950,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002949 = (bfloat16_t*)((char*)buf + 13601024);
    bfloat16_t *dtensor10002951 = (bfloat16_t*)((char*)buf + 13634048);
    bfloat16_t *dtensor10002952 = (bfloat16_t*)((char*)buf + 13666816);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002952, dtensor10002949, dtensor10002951);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002811 = (bfloat16_t*)input_tensors.at(28);
    bfloat16_t *dtensor10002953 = (bfloat16_t*)((char*)buf + 13699584);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002953,dtensor10002827,dtensor10002811, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002952 = (bfloat16_t*)((char*)buf + 13666816);
    bfloat16_t *dtensor10002953 = (bfloat16_t*)((char*)buf + 13699584);
    bfloat16_t *dtensor10002954 = (bfloat16_t*)((char*)buf + 13732352);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002954,dtensor10002952,dtensor10002953, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002954 = (bfloat16_t*)((char*)buf + 13732352);
    bfloat16_t *dtensor10002812 = (bfloat16_t*)input_tensors.at(29);
    bfloat16_t *dtensor10002955 = (bfloat16_t*)((char*)buf + 13765120);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002955,dtensor10002954,dtensor10002812, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002943 = (bfloat16_t*)((char*)buf + 12912896);
    bfloat16_t *dtensor10002955 = (bfloat16_t*)((char*)buf + 13765120);
    bfloat16_t *dtensor10002956 = (bfloat16_t*)((char*)buf + 14289408);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002956, dtensor10002943, dtensor10002955);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002813 = (bfloat16_t*)input_tensors.at(30);
    bfloat16_t *dtensor10002957 = (bfloat16_t*)((char*)buf + 14813696);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002957,dtensor10002783,dtensor10002813, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002813 = (bfloat16_t*)input_tensors.at(30);
    bfloat16_t *dtensor10002958 = (bfloat16_t*)((char*)buf + 14846464);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002958,dtensor10002825,dtensor10002813, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002814 = (bfloat16_t*)input_tensors.at(31);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002959 = (bfloat16_t*)((char*)buf + 14879232);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002959,dtensor10002814,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002957 = (bfloat16_t*)((char*)buf + 14813696);
    bfloat16_t *dtensor10002959 = (bfloat16_t*)((char*)buf + 14879232);
    bfloat16_t *dtensor10002960 = (bfloat16_t*)((char*)buf + 14912000);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002960,dtensor10002957,dtensor10002959, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002960 = (bfloat16_t*)((char*)buf + 14912000);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002961 = (bfloat16_t*)((char*)buf + 14944768);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002961, dtensor10002960, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002961 = (bfloat16_t*)((char*)buf + 14944768);
    bfloat16_t *dtensor10002962 = (bfloat16_t*)((char*)buf + 14977536);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002962, dtensor10002961);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002962 = (bfloat16_t*)((char*)buf + 14977536);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002963 = (bfloat16_t*)((char*)buf + 15010304);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002963,dtensor10002962,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002963 = (bfloat16_t*)((char*)buf + 15010304);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002964 = (bfloat16_t*)((char*)buf + 15010560);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002964,dtensor10002963,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002962 = (bfloat16_t*)((char*)buf + 14977536);
    bfloat16_t *dtensor10002964 = (bfloat16_t*)((char*)buf + 15010560);
    bfloat16_t *dtensor10002965 = (bfloat16_t*)((char*)buf + 15043328);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002965, dtensor10002962, dtensor10002964);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002813 = (bfloat16_t*)input_tensors.at(30);
    bfloat16_t *dtensor10002966 = (bfloat16_t*)((char*)buf + 15076096);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002966,dtensor10002827,dtensor10002813, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002965 = (bfloat16_t*)((char*)buf + 15043328);
    bfloat16_t *dtensor10002966 = (bfloat16_t*)((char*)buf + 15076096);
    bfloat16_t *dtensor10002967 = (bfloat16_t*)((char*)buf + 15108864);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002967,dtensor10002965,dtensor10002966, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002967 = (bfloat16_t*)((char*)buf + 15108864);
    bfloat16_t *dtensor10002814 = (bfloat16_t*)input_tensors.at(31);
    bfloat16_t *dtensor10002968 = (bfloat16_t*)((char*)buf + 15141632);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002968,dtensor10002967,dtensor10002814, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002956 = (bfloat16_t*)((char*)buf + 14289408);
    bfloat16_t *dtensor10002968 = (bfloat16_t*)((char*)buf + 15141632);
    bfloat16_t *dtensor10002969 = (bfloat16_t*)((char*)buf + 15665920);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002969, dtensor10002956, dtensor10002968);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002815 = (bfloat16_t*)input_tensors.at(32);
    bfloat16_t *dtensor10002970 = (bfloat16_t*)((char*)buf + 16190208);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002970,dtensor10002783,dtensor10002815, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002815 = (bfloat16_t*)input_tensors.at(32);
    bfloat16_t *dtensor10002971 = (bfloat16_t*)((char*)buf + 16222976);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002971,dtensor10002825,dtensor10002815, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002816 = (bfloat16_t*)input_tensors.at(33);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002972 = (bfloat16_t*)((char*)buf + 16255744);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002972,dtensor10002816,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002970 = (bfloat16_t*)((char*)buf + 16190208);
    bfloat16_t *dtensor10002972 = (bfloat16_t*)((char*)buf + 16255744);
    bfloat16_t *dtensor10002973 = (bfloat16_t*)((char*)buf + 16288512);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002973,dtensor10002970,dtensor10002972, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002973 = (bfloat16_t*)((char*)buf + 16288512);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002974 = (bfloat16_t*)((char*)buf + 16321280);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002974, dtensor10002973, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002974 = (bfloat16_t*)((char*)buf + 16321280);
    bfloat16_t *dtensor10002975 = (bfloat16_t*)((char*)buf + 16354048);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002975, dtensor10002974);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002975 = (bfloat16_t*)((char*)buf + 16354048);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002976 = (bfloat16_t*)((char*)buf + 16386816);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002976,dtensor10002975,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002976 = (bfloat16_t*)((char*)buf + 16386816);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002977 = (bfloat16_t*)((char*)buf + 16387072);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002977,dtensor10002976,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002975 = (bfloat16_t*)((char*)buf + 16354048);
    bfloat16_t *dtensor10002977 = (bfloat16_t*)((char*)buf + 16387072);
    bfloat16_t *dtensor10002978 = (bfloat16_t*)((char*)buf + 16419840);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002978, dtensor10002975, dtensor10002977);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002815 = (bfloat16_t*)input_tensors.at(32);
    bfloat16_t *dtensor10002979 = (bfloat16_t*)((char*)buf + 16452608);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002979,dtensor10002827,dtensor10002815, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002978 = (bfloat16_t*)((char*)buf + 16419840);
    bfloat16_t *dtensor10002979 = (bfloat16_t*)((char*)buf + 16452608);
    bfloat16_t *dtensor10002980 = (bfloat16_t*)((char*)buf + 16485376);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002980,dtensor10002978,dtensor10002979, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002980 = (bfloat16_t*)((char*)buf + 16485376);
    bfloat16_t *dtensor10002816 = (bfloat16_t*)input_tensors.at(33);
    bfloat16_t *dtensor10002981 = (bfloat16_t*)((char*)buf + 16518144);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002981,dtensor10002980,dtensor10002816, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002969 = (bfloat16_t*)((char*)buf + 15665920);
    bfloat16_t *dtensor10002981 = (bfloat16_t*)((char*)buf + 16518144);
    bfloat16_t *dtensor10002982 = (bfloat16_t*)((char*)buf + 17042432);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002982, dtensor10002969, dtensor10002981);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002817 = (bfloat16_t*)input_tensors.at(34);
    bfloat16_t *dtensor10002983 = (bfloat16_t*)((char*)buf + 17566720);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002983,dtensor10002783,dtensor10002817, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002817 = (bfloat16_t*)input_tensors.at(34);
    bfloat16_t *dtensor10002984 = (bfloat16_t*)((char*)buf + 17599488);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002984,dtensor10002825,dtensor10002817, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002818 = (bfloat16_t*)input_tensors.at(35);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002985 = (bfloat16_t*)((char*)buf + 17632256);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002985,dtensor10002818,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002983 = (bfloat16_t*)((char*)buf + 17566720);
    bfloat16_t *dtensor10002985 = (bfloat16_t*)((char*)buf + 17632256);
    bfloat16_t *dtensor10002986 = (bfloat16_t*)((char*)buf + 17665024);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002986,dtensor10002983,dtensor10002985, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002986 = (bfloat16_t*)((char*)buf + 17665024);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10002987 = (bfloat16_t*)((char*)buf + 17697792);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002987, dtensor10002986, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10002987 = (bfloat16_t*)((char*)buf + 17697792);
    bfloat16_t *dtensor10002988 = (bfloat16_t*)((char*)buf + 17730560);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002988, dtensor10002987);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002988 = (bfloat16_t*)((char*)buf + 17730560);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10002989 = (bfloat16_t*)((char*)buf + 17763328);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002989,dtensor10002988,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002989 = (bfloat16_t*)((char*)buf + 17763328);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10002990 = (bfloat16_t*)((char*)buf + 17763584);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002990,dtensor10002989,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10002988 = (bfloat16_t*)((char*)buf + 17730560);
    bfloat16_t *dtensor10002990 = (bfloat16_t*)((char*)buf + 17763584);
    bfloat16_t *dtensor10002991 = (bfloat16_t*)((char*)buf + 17796352);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002991, dtensor10002988, dtensor10002990);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002817 = (bfloat16_t*)input_tensors.at(34);
    bfloat16_t *dtensor10002992 = (bfloat16_t*)((char*)buf + 17829120);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002992,dtensor10002827,dtensor10002817, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002991 = (bfloat16_t*)((char*)buf + 17796352);
    bfloat16_t *dtensor10002992 = (bfloat16_t*)((char*)buf + 17829120);
    bfloat16_t *dtensor10002993 = (bfloat16_t*)((char*)buf + 17861888);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002993,dtensor10002991,dtensor10002992, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002993 = (bfloat16_t*)((char*)buf + 17861888);
    bfloat16_t *dtensor10002818 = (bfloat16_t*)input_tensors.at(35);
    bfloat16_t *dtensor10002994 = (bfloat16_t*)((char*)buf + 17894656);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002994,dtensor10002993,dtensor10002818, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002982 = (bfloat16_t*)((char*)buf + 17042432);
    bfloat16_t *dtensor10002994 = (bfloat16_t*)((char*)buf + 17894656);
    bfloat16_t *dtensor10002995 = (bfloat16_t*)((char*)buf + 18418944);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10002995, dtensor10002982, dtensor10002994);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002819 = (bfloat16_t*)input_tensors.at(36);
    bfloat16_t *dtensor10002996 = (bfloat16_t*)((char*)buf + 18943232);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002996,dtensor10002783,dtensor10002819, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002819 = (bfloat16_t*)input_tensors.at(36);
    bfloat16_t *dtensor10002997 = (bfloat16_t*)((char*)buf + 18976000);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002997,dtensor10002825,dtensor10002819, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002820 = (bfloat16_t*)input_tensors.at(37);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10002998 = (bfloat16_t*)((char*)buf + 19008768);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002998,dtensor10002820,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002996 = (bfloat16_t*)((char*)buf + 18943232);
    bfloat16_t *dtensor10002998 = (bfloat16_t*)((char*)buf + 19008768);
    bfloat16_t *dtensor10002999 = (bfloat16_t*)((char*)buf + 19041536);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10002999,dtensor10002996,dtensor10002998, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10002999 = (bfloat16_t*)((char*)buf + 19041536);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10003000 = (bfloat16_t*)((char*)buf + 19074304);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10003000, dtensor10002999, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10003000 = (bfloat16_t*)((char*)buf + 19074304);
    bfloat16_t *dtensor10003001 = (bfloat16_t*)((char*)buf + 19107072);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10003001, dtensor10003000);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10003001 = (bfloat16_t*)((char*)buf + 19107072);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10003002 = (bfloat16_t*)((char*)buf + 19139840);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003002,dtensor10003001,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10003002 = (bfloat16_t*)((char*)buf + 19139840);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10003003 = (bfloat16_t*)((char*)buf + 19140096);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003003,dtensor10003002,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10003001 = (bfloat16_t*)((char*)buf + 19107072);
    bfloat16_t *dtensor10003003 = (bfloat16_t*)((char*)buf + 19140096);
    bfloat16_t *dtensor10003004 = (bfloat16_t*)((char*)buf + 19172864);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10003004, dtensor10003001, dtensor10003003);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002819 = (bfloat16_t*)input_tensors.at(36);
    bfloat16_t *dtensor10003005 = (bfloat16_t*)((char*)buf + 19205632);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003005,dtensor10002827,dtensor10002819, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10003004 = (bfloat16_t*)((char*)buf + 19172864);
    bfloat16_t *dtensor10003005 = (bfloat16_t*)((char*)buf + 19205632);
    bfloat16_t *dtensor10003006 = (bfloat16_t*)((char*)buf + 19238400);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003006,dtensor10003004,dtensor10003005, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10003006 = (bfloat16_t*)((char*)buf + 19238400);
    bfloat16_t *dtensor10002820 = (bfloat16_t*)input_tensors.at(37);
    bfloat16_t *dtensor10003007 = (bfloat16_t*)((char*)buf + 19271168);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003007,dtensor10003006,dtensor10002820, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10002995 = (bfloat16_t*)((char*)buf + 18418944);
    bfloat16_t *dtensor10003007 = (bfloat16_t*)((char*)buf + 19271168);
    bfloat16_t *dtensor10003008 = (bfloat16_t*)((char*)buf + 19795456);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10003008, dtensor10002995, dtensor10003007);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002821 = (bfloat16_t*)input_tensors.at(38);
    bfloat16_t *dtensor10003009 = (bfloat16_t*)((char*)buf + 20319744);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003009,dtensor10002783,dtensor10002821, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002821 = (bfloat16_t*)input_tensors.at(38);
    bfloat16_t *dtensor10003010 = (bfloat16_t*)((char*)buf + 20352512);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003010,dtensor10002825,dtensor10002821, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002822 = (bfloat16_t*)input_tensors.at(39);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10003011 = (bfloat16_t*)((char*)buf + 20385280);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003011,dtensor10002822,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10003009 = (bfloat16_t*)((char*)buf + 20319744);
    bfloat16_t *dtensor10003011 = (bfloat16_t*)((char*)buf + 20385280);
    bfloat16_t *dtensor10003012 = (bfloat16_t*)((char*)buf + 20418048);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003012,dtensor10003009,dtensor10003011, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10003012 = (bfloat16_t*)((char*)buf + 20418048);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10003013 = (bfloat16_t*)((char*)buf + 20450816);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10003013, dtensor10003012, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10003013 = (bfloat16_t*)((char*)buf + 20450816);
    bfloat16_t *dtensor10003014 = (bfloat16_t*)((char*)buf + 20483584);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10003014, dtensor10003013);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10003014 = (bfloat16_t*)((char*)buf + 20483584);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10003015 = (bfloat16_t*)((char*)buf + 20516352);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003015,dtensor10003014,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10003015 = (bfloat16_t*)((char*)buf + 20516352);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10003016 = (bfloat16_t*)((char*)buf + 20516608);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003016,dtensor10003015,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10003014 = (bfloat16_t*)((char*)buf + 20483584);
    bfloat16_t *dtensor10003016 = (bfloat16_t*)((char*)buf + 20516608);
    bfloat16_t *dtensor10003017 = (bfloat16_t*)((char*)buf + 20549376);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10003017, dtensor10003014, dtensor10003016);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002821 = (bfloat16_t*)input_tensors.at(38);
    bfloat16_t *dtensor10003018 = (bfloat16_t*)((char*)buf + 20582144);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003018,dtensor10002827,dtensor10002821, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10003017 = (bfloat16_t*)((char*)buf + 20549376);
    bfloat16_t *dtensor10003018 = (bfloat16_t*)((char*)buf + 20582144);
    bfloat16_t *dtensor10003019 = (bfloat16_t*)((char*)buf + 20614912);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003019,dtensor10003017,dtensor10003018, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10003019 = (bfloat16_t*)((char*)buf + 20614912);
    bfloat16_t *dtensor10002822 = (bfloat16_t*)input_tensors.at(39);
    bfloat16_t *dtensor10003020 = (bfloat16_t*)((char*)buf + 20647680);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003020,dtensor10003019,dtensor10002822, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10003008 = (bfloat16_t*)((char*)buf + 19795456);
    bfloat16_t *dtensor10003020 = (bfloat16_t*)((char*)buf + 20647680);
    bfloat16_t *dtensor10003021 = (bfloat16_t*)((char*)buf + 21171968);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10003021, dtensor10003008, dtensor10003020);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002783 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10002823 = (bfloat16_t*)input_tensors.at(40);
    bfloat16_t *dtensor10003022 = (bfloat16_t*)((char*)buf + 21696256);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003022,dtensor10002783,dtensor10002823, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002825 = (bfloat16_t*)((char*)buf + 0);
    bfloat16_t *dtensor10002823 = (bfloat16_t*)input_tensors.at(40);
    bfloat16_t *dtensor10003023 = (bfloat16_t*)((char*)buf + 21729024);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003023,dtensor10002825,dtensor10002823, 128,128,2048, 1,128, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002824 = (bfloat16_t*)input_tensors.at(41);
    bfloat16_t *dtensor10002826 = (bfloat16_t*)((char*)buf + 524288);
    bfloat16_t *dtensor10003024 = (bfloat16_t*)((char*)buf + 21761792);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003024,dtensor10002824,dtensor10002826, 128,128,2048, 2048,1, 1,2048, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10003022 = (bfloat16_t*)((char*)buf + 21696256);
    bfloat16_t *dtensor10003024 = (bfloat16_t*)((char*)buf + 21761792);
    bfloat16_t *dtensor10003025 = (bfloat16_t*)((char*)buf + 21794560);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003025,dtensor10003022,dtensor10003024, 128,128,128, 1,128, 1,128, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_mul_op
    bfloat16_t *dtensor10003025 = (bfloat16_t*)((char*)buf + 21794560);
    bfloat16_t *dtensor10002790 = (bfloat16_t*)input_tensors.at(7);
    bfloat16_t *dtensor10003026 = (bfloat16_t*)((char*)buf + 21827328);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::MUL, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10003026, dtensor10003025, dtensor10002790);
  }
  {
    // OP type: kn_exp_op
    bfloat16_t *dtensor10003026 = (bfloat16_t*)((char*)buf + 21827328);
    bfloat16_t *dtensor10003027 = (bfloat16_t*)((char*)buf + 21860096);
    using kernel = kn::ElementUnaryKernel<bfloat16_t, kn::ElementUnaryOpType::EXP, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10003027, dtensor10003026);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10003027 = (bfloat16_t*)((char*)buf + 21860096);
    bfloat16_t *dtensor10002791 = (bfloat16_t*)input_tensors.at(8);
    bfloat16_t *dtensor10003028 = (bfloat16_t*)((char*)buf + 21892864);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003028,dtensor10003027,dtensor10002791, 128,1,128, 128,1, 1,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10003028 = (bfloat16_t*)((char*)buf + 21892864);
    bfloat16_t *dtensor10002792 = (bfloat16_t*)input_tensors.at(9);
    bfloat16_t *dtensor10003029 = (bfloat16_t*)((char*)buf + 21893120);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003029,dtensor10003028,dtensor10002792, 128,128,1, 1,128, 128,1, 128,1, 1, 0,0,0);
  }
  {
    // OP type: kn_div_op
    bfloat16_t *dtensor10003027 = (bfloat16_t*)((char*)buf + 21860096);
    bfloat16_t *dtensor10003029 = (bfloat16_t*)((char*)buf + 21893120);
    bfloat16_t *dtensor10003030 = (bfloat16_t*)((char*)buf + 21925888);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::DIV, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<128>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10003030, dtensor10003027, dtensor10003029);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10002827 = (bfloat16_t*)((char*)buf + 1048576);
    bfloat16_t *dtensor10002823 = (bfloat16_t*)input_tensors.at(40);
    bfloat16_t *dtensor10003031 = (bfloat16_t*)((char*)buf + 21958656);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003031,dtensor10002827,dtensor10002823, 128,128,2048, 2048,1, 128,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10003030 = (bfloat16_t*)((char*)buf + 21925888);
    bfloat16_t *dtensor10003031 = (bfloat16_t*)((char*)buf + 21958656);
    bfloat16_t *dtensor10003032 = (bfloat16_t*)((char*)buf + 21991424);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003032,dtensor10003030,dtensor10003031, 128,128,128, 128,1, 1,128, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10003032 = (bfloat16_t*)((char*)buf + 21991424);
    bfloat16_t *dtensor10002824 = (bfloat16_t*)input_tensors.at(41);
    bfloat16_t *dtensor10003033 = (bfloat16_t*)((char*)buf + 22024192);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003033,dtensor10003032,dtensor10002824, 128,2048,128, 1,128, 2048,1, 1,128, 1, 0,0,0);
  }
  {
    // OP type: kn_add_op
    bfloat16_t *dtensor10003021 = (bfloat16_t*)((char*)buf + 21171968);
    bfloat16_t *dtensor10003033 = (bfloat16_t*)((char*)buf + 22024192);
    bfloat16_t *dtensor10003034 = (bfloat16_t*)((char*)buf + 22548480);
    using kernel = kn::ElementBinaryKernel<bfloat16_t, kn::ElementBinaryOpType::ADD, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>, Layout<Shape<Int<128>, Int<2048>>, Stride<Int<1>, Int<128>>>>;
    kernel::run(dtensor10003034, dtensor10003021, dtensor10003033);
  }
  {
    // OP type: kn_matmul_op
    bfloat16_t *dtensor10003034 = (bfloat16_t*)((char*)buf + 22548480);
    bfloat16_t *dtensor10002787 = (bfloat16_t*)input_tensors.at(4);
    bfloat16_t *dtensor10003035 = (bfloat16_t*)output_tensors.at(0);
    kn::gemm<CUBLAS_COMPUTE_32F>(dtensor10003035,dtensor10003034,dtensor10002787, 128,1024,2048, 1,128, 1024,1, 1,128, 1, 0,0,0);
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

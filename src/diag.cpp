
#include "canticle/common.h"
#include "canticle/kernels.h"
#include "canticle/quant.h"
#include <random>
using namespace canticle;
int diag_run() {
  int N=64, K=64;
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> i4(-8,7);
  std::uniform_real_distribution<float> xd(-1.f,1.f);
  Int4Matrix W; W.alloc(N,K); W.scale=0.0137f;
  for(int n=0;n<N;n++) for(int k=0;k<K;k++) W.set(n,k,i4(rng));
  std::vector<uint16_t> hx(K);
  for(int k=0;k<K;k++) hx[k]=f32_to_f16(xd(rng));

  std::vector<float> ref(N), got(N);
  gemv_int4_reference(W.W.data(), (const __half*)hx.data(), ref.data(), N, K, W.scale);

  GpuBuf dW,dx,dy; dW.alloc(W.W.size()*4); dx.alloc(K*2); dy.alloc(N*4);
  CUDA_CHECK(cudaMemcpy(dW.ptr,W.W.data(),W.W.size()*4,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dx.ptr,hx.data(),K*2,cudaMemcpyHostToDevice));
  cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
  launch_gemv_int4_naive((const uint32_t*)dW.ptr,(const __half*)dx.ptr,(float*)dy.ptr,N,K,W.scale,st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  CUDA_CHECK(cudaMemcpy(got.data(),dy.ptr,N*4,cudaMemcpyDeviceToHost));

  printf("idx   ref            gpu            ratio\n");
  for(int n=0;n<6;n++) printf("%3d  %12.6f  %12.6f  %10.6f\n", n, ref[n], got[n], got[n]/(ref[n]+1e-30f));

  // manual first-row recompute
  printf("\nrow0 manual:\n");
  double a=0; int shown=0;
  for(int k=0;k<K;k++){ double w=W.get(0,k); double x=f16_to_f32(hx[k]); a+=w*x;
    if(shown<5 && w!=0){ printf("  k=%d w=%+.0f x=%.6f term=%.6f\n",k,w,x,w*x); shown++; } }
  printf("  raw_acc=%.6f scaled=%.6f  ref[0]=%.6f\n", a, a*W.scale, ref[0]);
  printf("  W.W.size()=%zu (expect %d)  hx.size()=%zu\n", W.W.size(), N*(K/8), hx.size());
  return 0;
}

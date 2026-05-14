#include <cuda_runtime.h>

__forceinline__ __device__ void fma_f32x2(
    float &d0, float &d1,
    float a0, float a1,
    float b0, float b1,
    float c0, float c1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)
    asm volatile(
        "{\n\t"
        ".reg .b64 ra, rb, rc, rd;\n\t"
        "mov.b64 ra, {%2, %3};\n\t"
        "mov.b64 rb, {%4, %5};\n\t"
        "mov.b64 rc, {%6, %7};\n\t"
        "fma.rn.f32x2 rd, ra, rb, rc;\n\t"
        "mov.b64 {%0, %1}, rd;\n\t"
        "}\n"
        : "=f"(d0), "=f"(d1)
        : "f"(a0), "f"(a1), "f"(b0), "f"(b1), "f"(c0), "f"(c1));
#else
    d0 = fmaf(a0, b0, c0);
    d1 = fmaf(a1, b1, c1);
#endif
}

__global__ void k(float *out, const float *a, const float *b, const float *c) {
    int i = threadIdx.x * 2;
    float d0, d1;
    fma_f32x2(d0, d1, a[i], a[i+1], b[i], b[i+1], c[i], c[i+1]);
    out[i] = d0;
    out[i+1] = d1;
}

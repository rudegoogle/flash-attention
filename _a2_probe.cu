#include <cuda_runtime.h>

__global__ void probe(float *out, const float *a, const float *b, const float *c) {
    int i = threadIdx.x * 2;
    float a0=a[i], a1=a[i+1], b0=b[i], b1=b[i+1], c0=c[i], c1=c[i+1];
    float d0, d1, e0, e1, f0, f1;

    // mul.f32x2
    asm volatile(
        "{\n\t.reg .b64 ra,rb,rd;\n\t"
        "mov.b64 ra,{%2,%3};\n\t"
        "mov.b64 rb,{%4,%5};\n\t"
        "mul.f32x2 rd,ra,rb;\n\t"
        "mov.b64 {%0,%1},rd;\n\t}\n"
        : "=f"(d0),"=f"(d1) : "f"(a0),"f"(a1),"f"(b0),"f"(b1));

    // add.f32x2
    asm volatile(
        "{\n\t.reg .b64 ra,rb,rd;\n\t"
        "mov.b64 ra,{%2,%3};\n\t"
        "mov.b64 rb,{%4,%5};\n\t"
        "add.f32x2 rd,ra,rb;\n\t"
        "mov.b64 {%0,%1},rd;\n\t}\n"
        : "=f"(e0),"=f"(e1) : "f"(d0),"f"(d1),"f"(c0),"f"(c1));

    // fma.rn.f32x2
    asm volatile(
        "{\n\t.reg .b64 ra,rb,rc,rd;\n\t"
        "mov.b64 ra,{%2,%3};\n\t"
        "mov.b64 rb,{%4,%5};\n\t"
        "mov.b64 rc,{%6,%7};\n\t"
        "fma.rn.f32x2 rd,ra,rb,rc;\n\t"
        "mov.b64 {%0,%1},rd;\n\t}\n"
        : "=f"(f0),"=f"(f1) : "f"(a0),"f"(a1),"f"(b0),"f"(b1),"f"(c0),"f"(c1));

    out[i]   = d0 + e0 + f0;
    out[i+1] = d1 + e1 + f1;
}

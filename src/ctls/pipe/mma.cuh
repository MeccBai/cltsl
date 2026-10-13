
#include <cuda_fp16.h>
#include <stdint.h>

#define packed __attribute__((packed)) 

namespace cuda::mma
{

    struct alignas(4) FP16Fragment
    {
        __half a[4];
        __half b[2];
        __half c[4];
    };

    static_assert(sizeof(FP16Fragment) == 20);
    static_assert(alignof(FP16Fragment) == 4);

    __device__ __forceinline__ void tensor_core(const FP16Fragment &frag, __half *target)
    {
        uint32_t a[2], b, c[2], d[2];

        __builtin_memcpy(a, frag.a, sizeof(a));
        __builtin_memcpy(&b, frag.b, sizeof(b));
        __builtin_memcpy(c, frag.c, sizeof(c));

        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
            "{%0, %1}, "
            "{%2, %3}, "
            "{%4}, "
            "{%5, %6};"
            : "=r"(d[0]), "=r"(d[1])
            : "r"(a[0]), "r"(a[1]),
              "r"(b),
              "r"(c[0]), "r"(c[1]));

        __builtin_memcpy(target, d, sizeof(d));
    }

}
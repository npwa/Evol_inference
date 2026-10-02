// Quantized dot-product / GEMV kernels for the two weight formats the genome selects, in ggml's block
// layouts: Q8_0 (32 x int8 + fp16 scale) and Q4_0 (32 x 4-bit + fp16 scale), against Q8_0 activations.
//
//   y[m] = sum_b  d_w[m,b] * d_x[b] * sum_{i<32} w[m,b,i] * x[b,i]
//
// Three implementations of the same function:
//   *_ref    scalar C++, portable (builds and runs on x86): the oracle.
//   *_sdot   NEON + SDOT (Armv8.2 dotprod), the decode (GEMV) workhorse.
//   *_smmla  NEON + SMMLA (Armv8.6 i8mm), 2 weight rows x 2 activation vectors per instruction (prefill/GEMM).
// All of them accumulate the per-block integer sum exactly and then add `float(sumi) * (dw * dx)` to a
// float accumulator in block order, so results are BIT-IDENTICAL across implementations provided the
// compiler does not contract multiply-adds (build with -ffp-contract=off).
#pragma once
#include <cstddef>
#include <cstdint>

namespace qdot {

constexpr int QK = 32;

struct block_q8_0 { uint16_t d; int8_t qs[QK]; };        // d = IEEE fp16 bits
struct block_q4_0 { uint16_t d; uint8_t qs[QK / 2]; };   // byte j: low nibble = element j, high nibble = element j+16; value = nibble - 8

float fp16_to_fp32(uint16_t h);       // software conversion, exact for all inputs
uint16_t fp32_to_fp16(float f);       // round-to-nearest-even

// y[m] = dot(W row m, x) for m in [0, M); `nb` blocks per row (K = nb * 32); W row-major, nb blocks per row.
void gemv_q8_0_ref(const block_q8_0* W, const block_q8_0* x, float* y, int M, int nb);
void gemv_q4_0_ref(const block_q4_0* W, const block_q8_0* x, float* y, int M, int nb);

// Quantize a float row (K multiple of 32) to Q8_0 / Q4_0 the way ggml does (used by tests/bench).
void quantize_row_q8_0(const float* src, block_q8_0* dst, int K);
void quantize_row_q4_0(const float* src, block_q4_0* dst, int K);

#if defined(__aarch64__)
bool have_sdot();
bool have_i8mm();
void gemv_q8_0_sdot(const block_q8_0* W, const block_q8_0* x, float* y, int M, int nb);
void gemv_q4_0_sdot(const block_q4_0* W, const block_q8_0* x, float* y, int M, int nb);
// GEMM tile: Y[2][M] = W (M rows) . X[2 activation vectors]; M even. y0/y1 receive the two outputs.
void gemm2_q8_0_smmla(const block_q8_0* W, const block_q8_0* x0, const block_q8_0* x1,
                      float* y0, float* y1, int M, int nb);
#endif

}  // namespace qdot

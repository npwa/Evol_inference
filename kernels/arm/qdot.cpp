#include "qdot.h"

#include <cmath>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#include <sys/auxv.h>
#endif

namespace qdot {

// ---------------------------------------------------------------- fp16 helpers (portable, exact)

float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff, bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else {  // subnormal: normalise
            int e = -1;
            do { e++; man <<= 1; } while (!(man & 0x400));
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((man & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f; std::memcpy(&f, &bits, 4); return f;
}

uint16_t fp32_to_fp16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t half = man >> shift, rem = man & ((1u << shift) - 1), mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1))) half++;
        return (uint16_t)(sign | half);
    }
    uint32_t half = (uint32_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1))) half++;
    return (uint16_t)half;
}

// ---------------------------------------------------------------- quantization (ggml's reference)

void quantize_row_q8_0(const float* src, block_q8_0* dst, int K) {
    for (int b = 0; b < K / QK; b++) {
        float amax = 0.f;
        for (int i = 0; i < QK; i++) amax = std::fmax(amax, std::fabs(src[b * QK + i]));
        const float d = amax / 127.f, id = d ? 1.f / d : 0.f;
        dst[b].d = fp32_to_fp16(d);
        for (int i = 0; i < QK; i++) dst[b].qs[i] = (int8_t)std::lround(src[b * QK + i] * id);
    }
}

void quantize_row_q4_0(const float* src, block_q4_0* dst, int K) {
    for (int b = 0; b < K / QK; b++) {
        float amax = 0.f, mx = 0.f;
        for (int i = 0; i < QK; i++) {
            const float v = src[b * QK + i];
            if (std::fabs(v) > amax) { amax = std::fabs(v); mx = v; }
        }
        const float d = mx / -8.f, id = d ? 1.f / d : 0.f;
        dst[b].d = fp32_to_fp16(d);
        for (int j = 0; j < QK / 2; j++) {
            const float x0 = src[b * QK + j] * id, x1 = src[b * QK + QK / 2 + j] * id;
            const uint8_t q0 = (uint8_t)std::fmin(15.f, (float)(int8_t)(x0 + 8.5f));
            const uint8_t q1 = (uint8_t)std::fmin(15.f, (float)(int8_t)(x1 + 8.5f));
            dst[b].qs[j] = q0 | (q1 << 4);
        }
    }
}

// ---------------------------------------------------------------- scalar reference (the oracle)

void gemv_q8_0_ref(const block_q8_0* W, const block_q8_0* x, float* y, int M, int nb) {
    for (int m = 0; m < M; m++) {
        float sumf = 0.f;
        for (int b = 0; b < nb; b++) {
            const block_q8_0& w = W[(size_t)m * nb + b];
            int32_t sumi = 0;
            for (int i = 0; i < QK; i++) sumi += (int32_t)w.qs[i] * (int32_t)x[b].qs[i];
            sumf += (float)sumi * (fp16_to_fp32(w.d) * fp16_to_fp32(x[b].d));
        }
        y[m] = sumf;
    }
}

void gemv_q4_0_ref(const block_q4_0* W, const block_q8_0* x, float* y, int M, int nb) {
    for (int m = 0; m < M; m++) {
        float sumf = 0.f;
        for (int b = 0; b < nb; b++) {
            const block_q4_0& w = W[(size_t)m * nb + b];
            int32_t sumi = 0;
            for (int j = 0; j < QK / 2; j++) {
                const int v0 = (w.qs[j] & 0x0f) - 8, v1 = (w.qs[j] >> 4) - 8;
                sumi += v0 * (int32_t)x[b].qs[j] + v1 * (int32_t)x[b].qs[j + QK / 2];
            }
            sumf += (float)sumi * (fp16_to_fp32(w.d) * fp16_to_fp32(x[b].d));
        }
        y[m] = sumf;
    }
}

#if defined(__aarch64__)
// ---------------------------------------------------------------- NEON kernels
// Compiled with per-function target attributes so a single binary can carry all paths and the
// caller picks one at run time (have_sdot / have_i8mm), instead of SIGILL on an older CPU.

bool have_sdot() { return (getauxval(AT_HWCAP) & (1UL << 20)) != 0; }       // HWCAP_ASIMDDP
bool have_i8mm() { return (getauxval(AT_HWCAP2) & (1UL << 13)) != 0; }      // HWCAP2_I8MM

__attribute__((target("arch=armv8.2-a+dotprod")))
void gemv_q8_0_sdot(const block_q8_0* W, const block_q8_0* x, float* y, int M, int nb) {
    for (int m = 0; m < M; m++) {
        float sumf = 0.f;
        for (int b = 0; b < nb; b++) {
            const block_q8_0& w = W[(size_t)m * nb + b];
            int32x4_t acc = vdupq_n_s32(0);
            acc = vdotq_s32(acc, vld1q_s8(w.qs), vld1q_s8(x[b].qs));
            acc = vdotq_s32(acc, vld1q_s8(w.qs + 16), vld1q_s8(x[b].qs + 16));
            const int32_t sumi = vaddvq_s32(acc);
            sumf += (float)sumi * (fp16_to_fp32(w.d) * fp16_to_fp32(x[b].d));
        }
        y[m] = sumf;
    }
}

__attribute__((target("arch=armv8.2-a+dotprod")))
void gemv_q4_0_sdot(const block_q4_0* W, const block_q8_0* x, float* y, int M, int nb) {
    const uint8x16_t mask = vdupq_n_u8(0x0f);
    const int8x16_t eight = vdupq_n_s8(8);
    for (int m = 0; m < M; m++) {
        float sumf = 0.f;
        for (int b = 0; b < nb; b++) {
            const block_q4_0& w = W[(size_t)m * nb + b];
            const uint8x16_t packed = vld1q_u8(w.qs);
            const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed, mask)), eight);   // elements 0..15
            const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), eight);    // elements 16..31
            int32x4_t acc = vdupq_n_s32(0);
            acc = vdotq_s32(acc, lo, vld1q_s8(x[b].qs));
            acc = vdotq_s32(acc, hi, vld1q_s8(x[b].qs + 16));
            const int32_t sumi = vaddvq_s32(acc);
            sumf += (float)sumi * (fp16_to_fp32(w.d) * fp16_to_fp32(x[b].d));
        }
        y[m] = sumf;
    }
}

// SMMLA: C(2x2, int32) += A(2x8, int8) . B(2x8, int8)^T. Rows of A = two weight rows, rows of B = two
// activation vectors, so each instruction yields 4 dot products over 8 elements.
__attribute__((target("arch=armv8.6-a+i8mm")))
void gemm2_q8_0_smmla(const block_q8_0* W, const block_q8_0* x0, const block_q8_0* x1,
                      float* y0, float* y1, int M, int nb) {
    for (int m = 0; m < M; m += 2) {
        float s00 = 0.f, s01 = 0.f, s10 = 0.f, s11 = 0.f;  // [weight row][activation vector]
        for (int b = 0; b < nb; b++) {
            const block_q8_0& wa = W[(size_t)m * nb + b];
            const block_q8_0& wb = W[(size_t)(m + 1) * nb + b];
            int32x4_t acc = vdupq_n_s32(0);
            for (int k = 0; k < QK; k += 8) {
                const int8x16_t a = vcombine_s8(vld1_s8(wa.qs + k), vld1_s8(wb.qs + k));     // rows: wa, wb
                const int8x16_t bb = vcombine_s8(vld1_s8(x0[b].qs + k), vld1_s8(x1[b].qs + k));  // rows: x0, x1
                acc = vmmlaq_s32(acc, a, bb);
            }
            // lanes: [wa.x0, wa.x1, wb.x0, wb.x1]
            const float dwa = fp16_to_fp32(wa.d), dwb = fp16_to_fp32(wb.d);
            const float dx0 = fp16_to_fp32(x0[b].d), dx1 = fp16_to_fp32(x1[b].d);
            s00 += (float)vgetq_lane_s32(acc, 0) * (dwa * dx0);
            s01 += (float)vgetq_lane_s32(acc, 1) * (dwa * dx1);
            s10 += (float)vgetq_lane_s32(acc, 2) * (dwb * dx0);
            s11 += (float)vgetq_lane_s32(acc, 3) * (dwb * dx1);
        }
        y0[m] = s00; y1[m] = s01; y0[m + 1] = s10; y1[m + 1] = s11;
    }
}
#endif

}  // namespace qdot

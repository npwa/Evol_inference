// Correctness test for qdot: portable reference vs an independent double-precision oracle (runs
// anywhere), and on aarch64 the NEON kernels vs the reference, BIT-IDENTICAL. Exit code 0 = all pass.
#include "qdot.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace qdot;
static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static std::mt19937 rng(12345);
static int rint_(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }
static uint16_t rand_scale() { return fp32_to_fp16(std::uniform_real_distribution<float>(0.0005f, 0.5f)(rng)); }

static void fill_q8(std::vector<block_q8_0>& v, int mode) {
    for (auto& b : v) {
        b.d = rand_scale();
        for (int i = 0; i < QK; i++)
            b.qs[i] = mode == 0 ? (int8_t)rint_(-128, 127) : mode == 1 ? (int8_t)-128 : (int8_t)127;
    }
}
static void fill_q4(std::vector<block_q4_0>& v, int mode) {
    for (auto& b : v) {
        b.d = rand_scale();
        for (int j = 0; j < QK / 2; j++)
            b.qs[j] = mode == 0 ? (uint8_t)rint_(0, 255) : mode == 1 ? 0x00 : 0xff;
    }
}

// independent oracle: integer sums via a different loop shape, float scales/accumulation in double
// Returns the exact-ish value and (via `absum`) the sum of absolute block terms, which bounds the
// float accumulation error of any summation order: |err| <= nb * 2^-24 * absum (standard gamma_n bound).
static double oracle_q8(const block_q8_0* w, const block_q8_0* x, int nb, double* absum) {
    double acc = 0; *absum = 0;
    for (int b = 0; b < nb; b++) {
        long s = 0;
        for (int i = 0; i < QK; i++) s += (long)w[b].qs[i] * x[b].qs[i];
        const double t = (double)s * fp16_to_fp32(w[b].d) * (double)fp16_to_fp32(x[b].d);
        acc += t; *absum += std::fabs(t);
    }
    return acc;
}
static double oracle_q4(const block_q4_0* w, const block_q8_0* x, int nb, double* absum) {
    double acc = 0; *absum = 0;
    for (int b = 0; b < nb; b++) {
        long s = 0;
        for (int i = 0; i < QK; i++) {
            const int nib = i < 16 ? (w[b].qs[i] & 15) : (w[b].qs[i - 16] >> 4);
            s += (long)(nib - 8) * x[b].qs[i];
        }
        const double t = (double)s * fp16_to_fp32(w[b].d) * (double)fp16_to_fp32(x[b].d);
        acc += t; *absum += std::fabs(t);
    }
    return acc;
}

[[maybe_unused]] static bool same_bits(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }

int main() {
    // ---- fp16 conversion: exhaustive round trip
    for (uint32_t h = 0; h < 65536; h++) {
        const float f = fp16_to_fp32((uint16_t)h);
        if (std::isnan(f)) continue;
        CHECK(fp32_to_fp16(f) == h, "fp16 roundtrip %04x", h);
    }
#if defined(__aarch64__)
    for (uint32_t h = 0; h < 65536; h++) {  // against the hardware conversion
        __fp16 hw; uint16_t hb = (uint16_t)h; std::memcpy(&hw, &hb, 2);
        const float f = (float)hw;
        if (std::isnan(f)) continue;
        CHECK(same_bits(fp16_to_fp32(hb), f), "fp16->fp32 differs from hardware for %04x", h);
    }
    std::printf("aarch64: sdot=%d i8mm=%d\n", (int)have_sdot(), (int)have_i8mm());
#endif

    // ---- quantizers produce valid blocks and small reconstruction error
    {
        std::vector<float> row(128);
        for (auto& v : row) v = std::normal_distribution<float>(0.f, 1.f)(rng);
        std::vector<block_q8_0> q8(4); std::vector<block_q4_0> q4(4);
        quantize_row_q8_0(row.data(), q8.data(), 128); quantize_row_q4_0(row.data(), q4.data(), 128);
        double e8 = 0, e4 = 0, en = 0;
        for (int b = 0; b < 4; b++) for (int i = 0; i < QK; i++) {
            const double d8 = fp16_to_fp32(q8[b].d), d4 = fp16_to_fp32(q4[b].d);
            const int nib = i < 16 ? (q4[b].qs[i] & 15) : (q4[b].qs[i - 16] >> 4);
            const double x = row[b * QK + i];
            e8 += std::pow(d8 * q8[b].qs[i] - x, 2); e4 += std::pow(d4 * (nib - 8) - x, 2); en += x * x;
        }
        CHECK(e8 / en < 1e-4, "q8_0 quantization error too large: %g", e8 / en);
        CHECK(e4 / en < 2e-2 && e4 > e8, "q4_0 quantization error unexpected: %g (q8 %g)", e4 / en, e8 / en);
    }

    // ---- reference vs oracle, and NEON vs reference (bit-exact), over shapes and data modes
    const int Ms[] = {1, 2, 3, 8, 17, 64}, NBs[] = {1, 2, 5, 16, 33, 128};
    int cases = 0;
    for (int mode = 0; mode < 3; mode++)
    for (int M : Ms) for (int nb : NBs) {
        std::vector<block_q8_0> W8((size_t)M * nb), x(nb), x1(nb);
        std::vector<block_q4_0> W4((size_t)M * nb);
        fill_q8(W8, mode); fill_q4(W4, mode); fill_q8(x, mode); fill_q8(x1, 0);
        std::vector<float> r8(M), r4(M);
        gemv_q8_0_ref(W8.data(), x.data(), r8.data(), M, nb);
        gemv_q4_0_ref(W4.data(), x.data(), r4.data(), M, nb);
        for (int m = 0; m < M; m++) {
            double a8, a4;
            const double o8 = oracle_q8(&W8[(size_t)m * nb], x.data(), nb, &a8), o4 = oracle_q4(&W4[(size_t)m * nb], x.data(), nb, &a4);
            const double eps = 5.97e-8;  // 2^-24
            CHECK(std::fabs(r8[m] - o8) <= nb * eps * a8 + 1e-30, "ref q8 vs oracle M=%d nb=%d m=%d: %g vs %g", M, nb, m, r8[m], o8);
            CHECK(std::fabs(r4[m] - o4) <= nb * eps * a4 + 1e-30, "ref q4 vs oracle M=%d nb=%d m=%d: %g vs %g", M, nb, m, r4[m], o4);
        }
#if defined(__aarch64__)
        if (have_sdot()) {
            std::vector<float> s8(M), s4(M);
            gemv_q8_0_sdot(W8.data(), x.data(), s8.data(), M, nb);
            gemv_q4_0_sdot(W4.data(), x.data(), s4.data(), M, nb);
            for (int m = 0; m < M; m++) {
                CHECK(same_bits(s8[m], r8[m]), "sdot q8 not bit-exact M=%d nb=%d m=%d mode=%d: %a vs %a", M, nb, m, mode, s8[m], r8[m]);
                CHECK(same_bits(s4[m], r4[m]), "sdot q4 not bit-exact M=%d nb=%d m=%d mode=%d: %a vs %a", M, nb, m, mode, s4[m], r4[m]);
            }
        }
        if (have_i8mm() && M % 2 == 0) {
            std::vector<float> y0(M), y1(M), e0(M), e1(M);
            gemm2_q8_0_smmla(W8.data(), x.data(), x1.data(), y0.data(), y1.data(), M, nb);
            gemv_q8_0_ref(W8.data(), x.data(), e0.data(), M, nb);
            gemv_q8_0_ref(W8.data(), x1.data(), e1.data(), M, nb);
            for (int m = 0; m < M; m++) {
                CHECK(same_bits(y0[m], e0[m]), "smmla x0 not bit-exact M=%d nb=%d m=%d: %a vs %a", M, nb, m, y0[m], e0[m]);
                CHECK(same_bits(y1[m], e1[m]), "smmla x1 not bit-exact M=%d nb=%d m=%d: %a vs %a", M, nb, m, y1[m], e1[m]);
            }
        }
#endif
        cases++;
    }
    std::printf("%s: %d shape/mode cases, %d failures\n", failures ? "FAIL" : "PASS", cases, failures);
    return failures ? 1 : 0;
}

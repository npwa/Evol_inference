// Micro-benchmark: effective weight bandwidth (GB/s) of each qdot kernel on an M x K GEMV (decode
// shape). On real Arm hardware (Graviton / M4) compare against STREAM bandwidth to place the kernel on
// the roofline; under QEMU the numbers are meaningless (correctness only). Usage: qdot_bench [M] [K] [reps]
#include "qdot.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace qdot;
using clk = std::chrono::steady_clock;

template <class F> static double best_seconds(F&& f, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        const auto t0 = clk::now(); f();
        best = std::min(best, std::chrono::duration<double>(clk::now() - t0).count());
    }
    return best;
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? std::atoi(argv[1]) : 4096, K = argc > 2 ? std::atoi(argv[2]) : 4096;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 20, nb = K / QK;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> wf((size_t)M * K), xf(K);
    for (auto& v : wf) v = nd(rng);
    for (auto& v : xf) v = nd(rng);
    std::vector<block_q8_0> W8((size_t)M * nb), x(nb);
    std::vector<block_q4_0> W4((size_t)M * nb);
    for (int m = 0; m < M; m++) { quantize_row_q8_0(&wf[(size_t)m * K], &W8[(size_t)m * nb], K); quantize_row_q4_0(&wf[(size_t)m * K], &W4[(size_t)m * nb], K); }
    quantize_row_q8_0(xf.data(), x.data(), K);
    std::vector<float> y(M);
    const double b8 = (double)M * nb * sizeof(block_q8_0), b4 = (double)M * nb * sizeof(block_q4_0);
    auto report = [&](const char* name, double bytes, double s) {
        std::printf("{\"kernel\":\"%s\",\"M\":%d,\"K\":%d,\"gb_per_s\":%.3f,\"ms\":%.4f}\n", name, M, K, bytes / s / 1e9, s * 1e3);
    };
    report("q8_0_ref", b8, best_seconds([&] { gemv_q8_0_ref(W8.data(), x.data(), y.data(), M, nb); }, reps));
    report("q4_0_ref", b4, best_seconds([&] { gemv_q4_0_ref(W4.data(), x.data(), y.data(), M, nb); }, reps));
#if defined(__aarch64__)
    if (have_sdot()) {
        report("q8_0_sdot", b8, best_seconds([&] { gemv_q8_0_sdot(W8.data(), x.data(), y.data(), M, nb); }, reps));
        report("q4_0_sdot", b4, best_seconds([&] { gemv_q4_0_sdot(W4.data(), x.data(), y.data(), M, nb); }, reps));
    }
    if (have_i8mm()) {
        std::vector<float> y2(M);
        report("q8_0_smmla_2vec", b8, best_seconds([&] { gemm2_q8_0_smmla(W8.data(), x.data(), x.data(), y.data(), y2.data(), M, nb); }, reps) / 2);  // per activation vector
    }
#endif
    return 0;
}

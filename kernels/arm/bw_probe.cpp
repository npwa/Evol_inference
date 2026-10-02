// Memory read-bandwidth probe for the roofline: sums a large array with N threads and reports GB/s.
// The decode GEMV is bandwidth-bound, so qdot_bench's GB/s should be read against this number.
// Usage: bw_probe [GiB per thread-set (default 1)] [max_threads (default hardware_concurrency)]
// Prints one JSON object per thread count (1, 2, 4, ... max).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static uint64_t sum_range(const uint64_t* p, size_t n) {
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;  // independent accumulators so the loads, not the adds, limit speed
    for (size_t i = 0; i + 4 <= n; i += 4) { a0 += p[i]; a1 += p[i + 1]; a2 += p[i + 2]; a3 += p[i + 3]; }
    return a0 + a1 + a2 + a3;
}

int main(int argc, char** argv) {
    const double gib = argc > 1 ? std::atof(argv[1]) : 1.0;
    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned maxt = argc > 2 ? (unsigned)std::atoi(argv[2]) : (hw ? hw : 1);
    const size_t n = (size_t)(gib * (1ull << 30)) / sizeof(uint64_t);
    std::vector<uint64_t> buf(n);
    for (size_t i = 0; i < n; i++) buf[i] = i * 2654435761u;  // touch every page
    volatile uint64_t sink = 0;
    for (unsigned t = 1; t <= maxt; t = t < maxt && t * 2 > maxt ? maxt : t * 2) {
        double best = 1e30;
        for (int rep = 0; rep < 5; rep++) {
            std::vector<std::thread> th; std::vector<uint64_t> part(t);
            const auto t0 = std::chrono::steady_clock::now();
            for (unsigned k = 0; k < t; k++)
                th.emplace_back([&, k] { const size_t lo = n / t * k, hi = k + 1 == t ? n : n / t * (k + 1); part[k] = sum_range(&buf[lo], hi - lo); });
            for (auto& x : th) x.join();
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            for (auto v : part) sink += v;
        }
        std::printf("{\"threads\":%u,\"gb_per_s\":%.2f}\n", t, (double)n * 8 / best / 1e9);
        if (t == maxt) break;
    }
    return sink == 1 ? 1 : 0;
}

#include "lowrank_q4_kernel.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace gguf_compress_v3;

static uint16_t f32_to_f16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    const int e = int((u >> 23) & 0xff) - 127 + 15;
    const uint32_t f = u & 0x7fffffu;
    if (e <= 0) return uint16_t(s);
    if (e >= 31) return uint16_t(s | 0x7c00u);
    return uint16_t(s | (uint32_t(e) << 10) | (f >> 13));
}

static void set_q4(std::vector<int8_t> & q, int row, int cols, int col, int value) {
    const size_t stride = (cols + 1) / 2;
    const size_t idx = size_t(row) * stride + (col >> 1);
    auto * p = reinterpret_cast<uint8_t *>(q.data());
    const uint8_t nibble = uint8_t(value + 8) & 15;
    if (col & 1) p[idx] = (p[idx] & 15) | (nibble << 4);
    else p[idx] = (p[idx] & 240) | nibble;
}

int main(int argc, char ** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int rank = argc > 2 ? std::atoi(argv[2]) : 16;
    const int batch = argc > 3 ? std::atoi(argv[3]) : 8;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 10;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1, 1);
    std::uniform_int_distribution<int> qdist(-7, 7);

    std::vector<float> x(size_t(batch) * n), y(size_t(batch) * n);
    std::vector<float> dense_w(size_t(n) * n), y_dense(size_t(batch) * n);
    for (auto & v : x) v = dist(rng);

    std::vector<int8_t> a(size_t(n) * ((rank + 1) / 2), 0);
    std::vector<int8_t> b(size_t(rank) * ((n + 1) / 2), 0);
    std::vector<uint16_t> a_scales(n, f32_to_f16(0.05f));
    std::vector<uint16_t> b_scales(rank, f32_to_f16(0.05f));

    for (int o = 0; o < n; ++o) for (int r = 0; r < rank; ++r) set_q4(a, o, rank, r, qdist(rng));
    for (int r = 0; r < rank; ++r) for (int i = 0; i < n; ++i) set_q4(b, r, n, i, qdist(rng));

    // Dense W is built only to obtain a same-math scalar reference baseline.
    for (int o = 0; o < n; ++o) {
        for (int i = 0; i < n; ++i) {
            float sum = 0;
            for (int r = 0; r < rank; ++r) {
                sum += float(q4_at(a.data() + size_t(o) * ((rank + 1) / 2), r)) * 0.05f
                     * float(q4_at(b.data() + size_t(r) * ((n + 1) / 2), i)) * 0.05f;
            }
            dense_w[size_t(o) * n + i] = sum;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int z = 0; z < iters; ++z) {
        lowrank_q4_fused(x.data(), batch, n, n, rank, a.data(), a_scales.data(), b.data(), b_scales.data(), y.data());
    }
    const auto t1 = std::chrono::steady_clock::now();

    const auto t2 = std::chrono::steady_clock::now();
    for (int z = 0; z < iters; ++z) {
        for (int bb = 0; bb < batch; ++bb) {
            for (int o = 0; o < n; ++o) {
                float sum = 0;
                for (int i = 0; i < n; ++i) sum += x[size_t(bb) * n + i] * dense_w[size_t(o) * n + i];
                y_dense[size_t(bb) * n + o] = sum;
            }
        }
    }
    const auto t3 = std::chrono::steady_clock::now();

    const double lowrank_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    const double dense_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / iters;
    double max_abs_error = 0;
    for (size_t i = 0; i < y.size(); ++i) max_abs_error = std::max(max_abs_error, double(std::abs(y[i] - y_dense[i])));

    std::cout << "{\"size\":" << n
              << ",\"rank\":" << rank
              << ",\"batch\":" << batch
              << ",\"lowrank_ms\":" << lowrank_ms
              << ",\"dense_ms\":" << dense_ms
              << ",\"speedup\":" << (dense_ms / lowrank_ms)
              << ",\"max_abs_error\":" << max_abs_error << "}\n";
    return 0;
}

//===- wka_tb.cpp - Hand-written omega-K HLS testbench -------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <vector>

#include "corner_turn.h"
#include "fft_core.h"
#include "generated/wka_luts.h"
#include "wka_top.h"

struct ddr_buffers_t {
    bus_t *in;
    bus_t *out;
    bus_t *tmp;
    size_t total_samples;
    size_t total_words;
    size_t bytes;
};

static uint64_t load_sample(const bus_t *memory, size_t sample_index) {
    size_t word_index = sample_index / BUS_LANES;
    int lane = static_cast<int>(sample_index % BUS_LANES);
    return static_cast<uint64_t>(memory[word_index].range((lane + 1) * WKA_COMPLEX_SAMPLE_BITS - 1,
                                                          lane * WKA_COMPLEX_SAMPLE_BITS));
}

static float load_real_sample(const bus_t *memory, size_t sample_index) {
    size_t word_index = sample_index / REAL_BUS_LANES;
    int lane = static_cast<int>(sample_index % REAL_BUS_LANES);
    uint32_t bits = static_cast<uint32_t>(
        memory[word_index].range((lane + 1) * WKA_IO_SCALAR_BITS - 1,
                                 lane * WKA_IO_SCALAR_BITS));
    return unpack_real(bits);
}

static void store_sample(bus_t *memory, size_t sample_index, uint64_t value) {
    size_t word_index = sample_index / BUS_LANES;
    int lane = static_cast<int>(sample_index % BUS_LANES);
    bus_t word = memory[word_index];
    word.range((lane + 1) * WKA_COMPLEX_SAMPLE_BITS - 1, lane * WKA_COMPLEX_SAMPLE_BITS) = value;
    memory[word_index] = word;
}

static void release_ddr_buffers(ddr_buffers_t &buffers) {
    delete[] buffers.in;
    delete[] buffers.out;
    delete[] buffers.tmp;
    buffers.in = nullptr;
    buffers.out = nullptr;
    buffers.tmp = nullptr;
}

static int allocate_ddr_buffers(ddr_buffers_t &buffers) {
    buffers.total_samples = static_cast<size_t>(N) * N;
    buffers.total_words = buffers.total_samples / BUS_LANES;
    buffers.bytes = buffers.total_words * sizeof(bus_t);
    buffers.in = new (std::nothrow) bus_t[buffers.total_words];
    buffers.out = new (std::nothrow) bus_t[buffers.total_words];
    buffers.tmp = new (std::nothrow) bus_t[buffers.total_words];

    if (!buffers.in || !buffers.out || !buffers.tmp) {
        std::cerr << "Memory allocation failed!" << std::endl;
        release_ddr_buffers(buffers);
        return -1;
    }

    memset(buffers.out, 0, buffers.bytes);
    memset(buffers.tmp, 0, buffers.bytes);
    return 0;
}

static void run_wka_kernel(ddr_buffers_t &buffers) {
    std::cout << "Running WKA kernel..." << std::endl;
    wka_sar_top(buffers.in, buffers.out, buffers.tmp);
    std::cout << "Kernel done." << std::endl;
}

static float compute_percentile_from_sorted(const std::vector<float> &values, float p) {
    if (values.empty()) {
        return 0.0f;
    }

    const double rank = static_cast<double>(p) * 0.01 * (values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = static_cast<size_t>(std::ceil(rank));

    const float vlo = values[lo];
    const float vhi = values[hi];
    const float w = static_cast<float>(rank - lo);
    return vlo + (vhi - vlo) * w;
}

static int write_grayscale_bmp(const char *filename, const unsigned char *pixels, int w, int h) {
    int row_stride = (w + 3) & ~3;
    int pixel_bytes = row_stride * h;
    int offset = 14 + 40 + 1024;
    int file_size = offset + pixel_bytes;

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        std::cerr << "Cannot write " << filename << std::endl;
        return -1;
    }

    // BMP file header
    unsigned char bfh[14];
    memset(bfh, 0, sizeof(bfh));
    bfh[0] = 'B';
    bfh[1] = 'M';
    bfh[2] = file_size & 0xFF;
    bfh[3] = (file_size >> 8) & 0xFF;
    bfh[4] = (file_size >> 16) & 0xFF;
    bfh[5] = (file_size >> 24) & 0xFF;
    bfh[10] = offset & 0xFF;
    bfh[11] = (offset >> 8) & 0xFF;
    bfh[12] = (offset >> 16) & 0xFF;
    bfh[13] = (offset >> 24) & 0xFF;
    fwrite(bfh, 1, 14, fp);

    // BITMAPINFOHEADER
    unsigned char bih[40];
    memset(bih, 0, sizeof(bih));
    bih[0] = 40;
    bih[4] = w & 0xFF;
    bih[5] = (w >> 8) & 0xFF;
    bih[6] = (w >> 16) & 0xFF;
    bih[7] = (w >> 24) & 0xFF;
    bih[8] = h & 0xFF;
    bih[9] = (h >> 8) & 0xFF;
    bih[10] = (h >> 16) & 0xFF;
    bih[11] = (h >> 24) & 0xFF;
    bih[12] = 1;
    bih[14] = 8;
    bih[20] = pixel_bytes & 0xFF;
    bih[21] = (pixel_bytes >> 8) & 0xFF;
    bih[22] = (pixel_bytes >> 16) & 0xFF;
    bih[23] = (pixel_bytes >> 24) & 0xFF;
    fwrite(bih, 1, 40, fp);

    // 8-bit grayscale palette
    for (int i = 0; i < 256; i++) {
        unsigned char gray = static_cast<unsigned char>(i);
        unsigned char e[4] = {gray, gray, gray, 0};
        fwrite(e, 1, 4, fp);
    }

    // BMP stores rows from bottom to top.
    unsigned char pad[4] = {};
    int pad_bytes = row_stride - w;
    for (int y = h - 1; y >= 0; y--) {
        fwrite(&pixels[y * w], 1, w, fp);
        if (pad_bytes > 0)
            fwrite(pad, 1, pad_bytes, fp);
    }

    int status = ferror(fp) ? -1 : 0;
    if (fclose(fp) != 0) {
        status = -1;
    }
    return status;
}

#if WKA_SIZE_PROFILE != WKA_SIZE_PRODUCTION

static void fill_small_cosim_input(bus_t *ddr_in) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float phase =
                0.03125f * static_cast<float>(i % 17) + 0.0625f * static_cast<float>(j % 29);
            float amp = 0.25f + 0.05f * static_cast<float>((i + j) % 11);
            float real = amp * std::cos(phase);
            float imag = amp * std::sin(phase);
            store_sample(ddr_in, static_cast<size_t>(i) * N + j, pack_data(data_t(real, imag)));
        }
    }
}

static int run_reduced_unit_tests() {
    ddr_buffers_t buffers = {};
    if (allocate_ddr_buffers(buffers) != 0) {
        return -1;
    }

    const size_t sample_count = static_cast<size_t>(N) * N;
    std::vector<float> expected_r(sample_count);
    std::vector<float> expected_i(sample_count);
    fill_small_cosim_input(buffers.in);
    for (size_t idx = 0; idx < sample_count; idx++) {
        data_t value = unpack_data(load_sample(buffers.in, idx));
        expected_r[idx] = value.real();
        expected_i[idx] = value.imag();
    }

    run_row_transform(buffers.in, buffers.out, WKA_ROW_FORWARD);
    run_row_transform(buffers.out, buffers.tmp, WKA_ROW_INVERSE);
    double ref_energy = 0.0;
    double error_energy = 0.0;
    for (size_t idx = 0; idx < sample_count; idx++) {
        data_t value = unpack_data(load_sample(buffers.tmp, idx));
        double dr = static_cast<double>(value.real()) - expected_r[idx];
        double di = static_cast<double>(value.imag()) - expected_i[idx];
        ref_energy += static_cast<double>(expected_r[idx]) * expected_r[idx] +
                      static_cast<double>(expected_i[idx]) * expected_i[idx];
        error_energy += dr * dr + di * di;
    }
    double fft_nrmse = std::sqrt(error_energy / std::max(ref_energy, 1.0e-30));
    std::cout << "Unit FFT round-trip NRMSE = " << fft_nrmse << std::endl;
    if (!(fft_nrmse <= 1.0e-3)) {
        std::cerr << "FFT round-trip regression failed." << std::endl;
        release_ddr_buffers(buffers);
        return -1;
    }

    for (size_t idx = 0; idx < sample_count; idx++) {
        float code = static_cast<float>(idx) / static_cast<float>(sample_count);
        store_sample(buffers.in, idx, pack_data(data_t(code, -code)));
    }
    corner_turn(buffers.in, buffers.out, false);
    corner_turn(buffers.out, buffers.tmp, false);
    for (size_t idx = 0; idx < sample_count; idx++) {
        data_t value = unpack_data(load_sample(buffers.tmp, idx));
        float code = static_cast<float>(idx) / static_cast<float>(sample_count);
        if (std::fabs(value.real() - code) > 1.0e-7f || std::fabs(value.imag() + code) > 1.0e-7f) {
            std::cerr << "Corner-turn round-trip mismatch at " << idx << std::endl;
            release_ddr_buffers(buffers);
            return -1;
        }
    }

    if (std::fabs(WKA_RANGE_WINDOW_ROM[0]) > 1.0e-7f ||
        std::fabs(WKA_RANGE_WINDOW_ROM[N - 1]) > 1.0e-7f ||
        std::fabs(WKA_AZIMUTH_WINDOW_ROM[0]) > 1.0e-7f ||
        std::fabs(WKA_AZIMUTH_WINDOW_ROM[N - 1]) > 1.0e-7f) {
        std::cerr << "Window ROM endpoints are not zero." << std::endl;
        release_ddr_buffers(buffers);
        return -1;
    }
    for (int column = 0; column < N; column++) {
        float range_mirror = WKA_RANGE_WINDOW_ROM[N - 1 - column];
        float azimuth_mirror = WKA_AZIMUTH_WINDOW_ROM[N - 1 - column];
        if (WKA_RANGE_WINDOW_ROM[column] < 0.0f ||
            WKA_RANGE_WINDOW_ROM[column] > 1.0f ||
            std::fabs(WKA_RANGE_WINDOW_ROM[column] - range_mirror) > 1.0e-6f ||
            WKA_AZIMUTH_WINDOW_ROM[column] < 0.0f ||
            WKA_AZIMUTH_WINDOW_ROM[column] > 1.0f ||
            std::fabs(WKA_AZIMUTH_WINDOW_ROM[column] - azimuth_mirror) > 1.0e-6f) {
            std::cerr << "Window ROM regression failed at " << column << std::endl;
            release_ddr_buffers(buffers);
            return -1;
        }
    }

    std::cout << "Reduced unit tests passed." << std::endl;
    release_ddr_buffers(buffers);
    return 0;
}

static int validate_small_cosim_output(const bus_t *ddr_out) {
    double sum_mag = 0.0;
    float max_mag = 0.0f;
    int nonzero_count = 0;

    const size_t sample_count = static_cast<size_t>(N) * N;
    for (size_t idx = 0; idx < sample_count; idx++) {
        float mag = load_real_sample(ddr_out, idx);
        if (!std::isfinite(mag)) {
            std::cerr << "Non-finite output detected at sample " << idx << std::endl;
            return -1;
        }
        sum_mag += mag;
        if (mag > max_mag) {
            max_mag = mag;
        }
        if (mag > WKA_TB_NONZERO_MAG_EPS) {
            nonzero_count++;
        }
    }

    std::cout << "Small cosim summary:" << std::endl;
    std::cout << "  sum_mag     = " << sum_mag << std::endl;
    std::cout << "  max_mag     = " << max_mag << std::endl;
    std::cout << "  nonzero_cnt = " << nonzero_count << std::endl;

    if (nonzero_count == 0) {
        std::cerr << "All outputs are zero in small cosim." << std::endl;
        return -1;
    }
    return 0;
}

static int dump_reduced_output(const bus_t *ddr_out) {
    const char *output_path = std::getenv("WKA_TB_DUMP_PATH");
    if (!output_path || output_path[0] == '\0') {
        return 0;
    }
    FILE *output = std::fopen(output_path, "wb");
    if (!output) {
        std::cerr << "Cannot write reduced output: " << output_path << std::endl;
        return -1;
    }
    const size_t sample_count = static_cast<size_t>(N) * N;
    for (size_t idx = 0; idx < sample_count; idx++) {
        float value = load_real_sample(ddr_out, idx);
        if (std::fwrite(&value, sizeof(value), 1, output) != 1) {
            std::fclose(output);
            std::cerr << "Short write while dumping reduced output." << std::endl;
            return -1;
        }
    }
    std::fclose(output);
    std::cout << "Reduced output saved: " << output_path << std::endl;
    return 0;
}

static int compare_reduced_golden(const bus_t *ddr_out, const char *golden_path) {
    if (!golden_path || golden_path[0] == '\0') {
        return 0;
    }
    FILE *golden = std::fopen(golden_path, "rb");
    if (!golden) {
        std::cerr << "Cannot open reduced golden: " << golden_path << std::endl;
        return -1;
    }

    double reference_energy = 0.0;
    double candidate_energy = 0.0;
    double error_energy = 0.0;
    double correlation_sum = 0.0;
    const size_t sample_count = static_cast<size_t>(N) * N;
    for (size_t idx = 0; idx < sample_count; idx++) {
        float expected = 0.0f;
        if (std::fread(&expected, sizeof(expected), 1, golden) != 1) {
            std::fclose(golden);
            std::cerr << "Reduced golden is shorter than expected." << std::endl;
            return -1;
        }
        double candidate = load_real_sample(ddr_out, idx);
        double error = candidate - expected;
        reference_energy += static_cast<double>(expected) * expected;
        candidate_energy += candidate * candidate;
        error_energy += error * error;
        correlation_sum += static_cast<double>(expected) * candidate;
    }
    std::fclose(golden);

    double nrmse = std::sqrt(error_energy / std::max(reference_energy, 1.0e-30));
    double correlation = std::abs(correlation_sum) /
                         std::sqrt(std::max(reference_energy * candidate_energy, 1.0e-30));
    std::cout << "Golden comparison: NRMSE=" << nrmse << " correlation=" << correlation
              << std::endl;
    if (nrmse > 1.0e-3 || correlation < 0.99999) {
        std::cerr << "Reduced golden comparison failed." << std::endl;
        return -1;
    }
    return 0;
}

static int run_small_cosim_tb(const char *golden_path) {
    if (run_reduced_unit_tests() != 0) {
        return -1;
    }

    ddr_buffers_t buffers = {};
    if (allocate_ddr_buffers(buffers) != 0) {
        return -1;
    }

    std::cout << "Generating synthetic small-cosim input..." << std::endl;
    fill_small_cosim_input(buffers.in);
    std::cout << "Generated " << buffers.total_samples << " samples." << std::endl;

    run_wka_kernel(buffers);

    int status = validate_small_cosim_output(buffers.out);
    if (status == 0) {
        status = compare_reduced_golden(buffers.out, golden_path);
    }
    if (status == 0) {
        status = dump_reduced_output(buffers.out);
    }
    release_ddr_buffers(buffers);
    return status;
}

#else

static int load_full_input(bus_t *ddr_in, size_t total_samples, const char *input_path) {
    std::cout << "Loading binary input..." << std::endl;
    FILE *fin = fopen(input_path, "rb");
    if (!fin) {
        std::cerr << "Cannot open input file!" << std::endl;
        return -1;
    }

    size_t read_cnt = 0;
    for (size_t word_idx = 0; word_idx < total_samples / BUS_LANES; word_idx++) {
        uint64_t lanes[BUS_LANES];
        size_t got = fread(lanes, sizeof(uint64_t), BUS_LANES, fin);
        if (got != static_cast<size_t>(BUS_LANES)) {
            fclose(fin);
            std::cerr << "Input file size mismatch, expected " << total_samples << " samples."
                      << std::endl;
            return -1;
        }
        bus_t packed = 0;
        for (int lane = 0; lane < BUS_LANES; lane++) {
            packed.range((lane + 1) * WKA_COMPLEX_SAMPLE_BITS - 1, lane * WKA_COMPLEX_SAMPLE_BITS) =
                lanes[lane];
        }
        ddr_in[word_idx] = packed;
        read_cnt += BUS_LANES;
    }
    fclose(fin);
    std::cout << "Loaded " << read_cnt << " samples." << std::endl;
    if (read_cnt != total_samples) {
        std::cerr << "Input file size mismatch, expected " << total_samples << " samples."
                  << std::endl;
        return -1;
    }

    return 0;
}

static int generate_full_output(const bus_t *ddr_out, const char *output_path) {
    std::cout << "Generating SAR image..." << std::endl;

    const int CROP_W = VALID_COLS;
    const int CROP_H = N;
    const float INC_RAD = static_cast<float>(WKA_DISPLAY_INC_ANGLE_DEG) * PI / 180.0f;
    const float DX_GROUND_RANGE = (C0 / (2.0f * FS)) / std::sin(INC_RAD);
    const float PIXEL_ASPECT = (VR / PRF) / DX_GROUND_RANGE;
    const int IMG_W = CROP_W;
    const int IMG_H = static_cast<int>(static_cast<float>(CROP_H) * PIXEL_ASPECT + 0.5f);
    const int IMG_PIXELS = IMG_H * IMG_W;

    std::cout << "  pixel_aspect = " << PIXEL_ASPECT << "  output " << IMG_W << " x " << IMG_H
              << std::endl;

    std::vector<float> img_disp(static_cast<size_t>(CROP_H) * CROP_W, 0.0f);

    // Flip vertically and crop to the valid range samples.
    for (int oy = 0; oy < CROP_H; oy++) {
        const int src_row = (N - 1) - oy;
        const size_t row_base = static_cast<size_t>(src_row) * N;
        for (int ox = 0; ox < CROP_W; ox++) {
            img_disp[static_cast<size_t>(oy) * CROP_W + ox] =
                load_real_sample(ddr_out, row_base + ox);
        }
    }

    std::vector<float> percentile_buf = img_disp;
    std::sort(percentile_buf.begin(), percentile_buf.end());
    float vmin = compute_percentile_from_sorted(percentile_buf, 2.0f);
    float vmax = compute_percentile_from_sorted(percentile_buf, 99.0f);

    if (vmax <= vmin) {
        vmax = vmin + 1.0f;
    }

    std::cout << "Display range: [" << vmin << ", " << vmax << "]" << std::endl;

    std::vector<float> img_norm(static_cast<size_t>(CROP_H) * CROP_W, 0.0f);
    const float inv_range = 1.0f / (vmax - vmin + 1.0e-6f);
    for (size_t i = 0; i < img_disp.size(); i++) {
        float t = (img_disp[i] - vmin) * inv_range;
        if (t < 0.0f) {
            t = 0.0f;
        }
        if (t > 1.0f) {
            t = 1.0f;
        }
        img_norm[i] = t;
    }

    std::vector<unsigned char> pixels(static_cast<size_t>(IMG_PIXELS), 0);

    // Resample azimuth spacing for display.
    for (int oy = 0; oy < IMG_H; oy++) {
        float src_y = (PIXEL_ASPECT > 0.0f) ? (static_cast<float>(oy) / PIXEL_ASPECT) : 0.0f;
        int y0 = static_cast<int>(src_y);
        if (y0 < 0) {
            y0 = 0;
        }
        if (y0 >= CROP_H) {
            y0 = CROP_H - 1;
        }
        int y1 = (y0 + 1 < CROP_H) ? (y0 + 1) : y0;
        float wy = src_y - static_cast<float>(y0);
        if (wy < 0.0f) {
            wy = 0.0f;
        }
        if (wy > 1.0f) {
            wy = 1.0f;
        }

        for (int ox = 0; ox < IMG_W; ox++) {
            const float v0 = img_norm[static_cast<size_t>(y0) * CROP_W + ox];
            const float v1 = img_norm[static_cast<size_t>(y1) * CROP_W + ox];
            const float t = v0 + (v1 - v0) * wy;
            pixels[static_cast<size_t>(oy) * IMG_W + ox] =
                static_cast<unsigned char>(t * 255.0f + 0.5f);
        }
    }

    if (write_grayscale_bmp(output_path, pixels.data(), IMG_W, IMG_H) != 0) {
        return -1;
    }
    std::cout << "Image saved: " << output_path << " (" << IMG_W << " x " << IMG_H << ")"
              << std::endl;

    return 0;
}

static int run_full_tb(const char *input_path, const char *output_path) {
    ddr_buffers_t buffers = {};
    if (allocate_ddr_buffers(buffers) != 0) {
        return -1;
    }

    if (load_full_input(buffers.in, buffers.total_samples, input_path) != 0) {
        release_ddr_buffers(buffers);
        return -1;
    }

    run_wka_kernel(buffers);

    int status = generate_full_output(buffers.out, output_path);
    release_ddr_buffers(buffers);
    return status;
}

#endif

int main(int argc, char **argv) {
    std::cout << "=== SAR WKA HLS Testbench (" << N << " x " << N << ") ===" << std::endl;

#if WKA_SIZE_PROFILE != WKA_SIZE_PRODUCTION
    const char *golden_path = (argc > 1) ? argv[1] : nullptr;
    int status = run_small_cosim_tb(golden_path);
#else
    const char *input_path = (argc > 1) ? argv[1] : WKA_TB_INPUT_PATH;
    const char *output_path = (argc > 2) ? argv[2] : WKA_TB_OUTPUT_PATH;
    int status = run_full_tb(input_path, output_path);
#endif

    std::cout << "=== Testbench Complete ===" << std::endl;
    return status;
}

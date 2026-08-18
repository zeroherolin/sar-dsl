//===- fft_core.cpp - Hand-written radix-4 HLS engine --------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "fft_core.h"
#include "generated/wka_luts.h"
#include "stolt_interpolation.h"

static int radix4_digit_reverse(int x) {
#pragma HLS INLINE
    int result = 0;
    for (int digit = 0; digit < LOG2_N / 2; digit++) {
#pragma HLS UNROLL
        result = (result << 2) | (x & 3);
        x >>= 2;
    }
    return result;
}

static data_t complex_multiply(data_t value, acc_t wr, acc_t wi) {
#pragma HLS INLINE
    acc_t real = value.real();
    acc_t imag = value.imag();
    return data_t(real * wr - imag * wi, real * wi + imag * wr);
}

static void lookup_twiddle(int index, bool inverse, acc_t &wr, acc_t &wi) {
#pragma HLS INLINE
    bool upper_half = index >= N / 2;
    int rom_index = upper_half ? index - N / 2 : index;
    acc_t sign = upper_half ? -1.0f : 1.0f;
    wr = sign * WKA_TWIDDLE_COS_ROM[rom_index];
    acc_t forward_sin = sign * WKA_TWIDDLE_SIN_ROM[rom_index];
    wi = inverse ? -forward_sin : forward_sin;
}

static void load_transform_block(bus_t *__restrict in, data_t buffer[WKA_FFT_PAR_ROWS][N],
                                 int base_row, bool load_shift, bool apply_window) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = WKA_AZIMUTH_WINDOW_ROM cyclic factor = BUS_LANES dim = 1
#pragma HLS BIND_STORAGE variable = WKA_AZIMUTH_WINDOW_ROM type = rom_2p impl = bram

LOAD_ROWS:
    for (int row_lane = 0; row_lane < WKA_FFT_PAR_ROWS; row_lane++) {
        int row = base_row + row_lane;
        size_t row_base = (static_cast<size_t>(row) * N) / BUS_LANES;
    LOAD_WORDS:
        for (int word = 0; word < N / BUS_LANES; word++) {
#pragma HLS PIPELINE II = WKA_ROW_LOAD_STORE_II
            bus_t packed = in[row_base + word];
            for (int lane = 0; lane < BUS_LANES; lane++) {
#pragma HLS UNROLL
                int column = word * BUS_LANES + lane;
                int target =
                    load_shift ? ((column < N / 2) ? column + N / 2 : column - N / 2) : column;
                data_t value = unpack_data(static_cast<uint64_t>(packed.range(
                    (lane + 1) * WKA_COMPLEX_SAMPLE_BITS - 1, lane * WKA_COMPLEX_SAMPLE_BITS)));
                if (apply_window) {
                    acc_t coefficient = WKA_AZIMUTH_WINDOW_ROM[column];
                    value = data_t(value.real() * coefficient, value.imag() * coefficient);
                }
                buffer[row_lane][target] = value;
            }
        }
    }
}

static void store_transform_block(data_t buffer[WKA_FFT_PAR_ROWS][N], bus_t *__restrict out,
                                  int base_row, bool store_shift, bool inverse) {
#pragma HLS INLINE off

    const acc_t scale = inverse ? 1.0f / static_cast<float>(N) : 1.0f;
STORE_ROWS:
    for (int row_lane = 0; row_lane < WKA_FFT_PAR_ROWS; row_lane++) {
        int row = base_row + row_lane;
        size_t row_base = (static_cast<size_t>(row) * N) / BUS_LANES;
    STORE_WORDS:
        for (int word = 0; word < N / BUS_LANES; word++) {
#pragma HLS PIPELINE II = WKA_ROW_LOAD_STORE_II
            bus_t packed = 0;
            for (int lane = 0; lane < BUS_LANES; lane++) {
#pragma HLS UNROLL
                int column = word * BUS_LANES + lane;
                int source =
                    store_shift ? ((column < N / 2) ? column + N / 2 : column - N / 2) : column;
                data_t value = buffer[row_lane][source];
                if (inverse) {
                    value = data_t(value.real() * scale, value.imag() * scale);
                }
                packed.range((lane + 1) * WKA_COMPLEX_SAMPLE_BITS - 1,
                             lane * WKA_COMPLEX_SAMPLE_BITS) = pack_data(value);
            }
            out[row_base + word] = packed;
        }
    }
}

static void compute_fft_block(data_t buf_in[WKA_FFT_PAR_ROWS][N],
                              data_t buf_out[WKA_FFT_PAR_ROWS][N], bool inverse) {
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable = WKA_TWIDDLE_COS_ROM type = rom_np impl = bram
#pragma HLS BIND_STORAGE variable = WKA_TWIDDLE_SIN_ROM type = rom_np impl = bram

    // Radix-4 DIT shares each twiddle across all parallel rows.
RADIX4_DIGIT_REVERSE:
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II = WKA_FFT_BITREV_II
        int reversed = radix4_digit_reverse(i);
        for (int row_lane = 0; row_lane < WKA_FFT_PAR_ROWS; row_lane++) {
#pragma HLS UNROLL
            buf_out[row_lane][reversed] = buf_in[row_lane][i];
        }
    }

RADIX4_STAGES:
    for (int stage = 1; stage <= LOG2_N / 2; stage++) {
#pragma HLS LOOP_FLATTEN off
        int quarter = 1 << (2 * (stage - 1));
        int group_shift = 2 * stage;
        int twiddle_shift = LOG2_N - group_shift;
    RADIX4_BUTTERFLIES:
        for (int butterfly = 0; butterfly < N / 4; butterfly++) {
#pragma HLS PIPELINE II = WKA_FFT_BUTTERFLY_II
            // Butterflies within a stage own disjoint radix-4 groups.
#pragma HLS DEPENDENCE variable = buf_out type = inter false
            int k = butterfly & (quarter - 1);
            int group = (butterfly >> (group_shift - 2)) << group_shift;
            int index0 = group + k;
            int index1 = index0 + quarter;
            int index2 = index1 + quarter;
            int index3 = index2 + quarter;
            int twiddle1 = k << twiddle_shift;
            int twiddle2 = twiddle1 << 1;
            int twiddle3 = twiddle1 + twiddle2;
            acc_t w1r, w1i, w2r, w2i, w3r, w3i;
            lookup_twiddle(twiddle1, inverse, w1r, w1i);
            lookup_twiddle(twiddle2, inverse, w2r, w2i);
            lookup_twiddle(twiddle3, inverse, w3r, w3i);

            for (int row_lane = 0; row_lane < WKA_FFT_PAR_ROWS; row_lane++) {
#pragma HLS UNROLL
                data_t a = buf_out[row_lane][index0];
                data_t b = complex_multiply(buf_out[row_lane][index1], w1r, w1i);
                data_t c = complex_multiply(buf_out[row_lane][index2], w2r, w2i);
                data_t d = complex_multiply(buf_out[row_lane][index3], w3r, w3i);

                acc_t ar = a.real();
                acc_t ai = a.imag();
                acc_t br = b.real();
                acc_t bi = b.imag();
                acc_t cr = c.real();
                acc_t ci = c.imag();
                acc_t dr = d.real();
                acc_t di = d.imag();

                buf_out[row_lane][index0] = data_t(ar + br + cr + dr, ai + bi + ci + di);
                buf_out[row_lane][index2] = data_t(ar - br + cr - dr, ai - bi + ci - di);
                if (inverse) {
                    buf_out[row_lane][index1] = data_t(ar - bi - cr + di, ai + br - ci - dr);
                    buf_out[row_lane][index3] = data_t(ar + bi - cr - di, ai - br - ci + dr);
                } else {
                    buf_out[row_lane][index1] = data_t(ar + bi - cr - di, ai - br - ci + dr);
                    buf_out[row_lane][index3] = data_t(ar - bi - cr + di, ai + br - ci - dr);
                }
            }
        }
    }
}

static void run_row_transform_impl(bus_t *__restrict in, bus_t *__restrict out, bool inverse,
                                   bool load_shift, bool store_shift, bool apply_window,
                                   bool fused_stolt) {
#pragma HLS INLINE off

    static data_t buffer_in[WKA_FFT_PAR_ROWS][N];
    static data_t buffer_out[WKA_FFT_PAR_ROWS][N];
#pragma HLS AGGREGATE variable = buffer_in compact = auto
#pragma HLS AGGREGATE variable = buffer_out compact = auto
#pragma HLS ARRAY_PARTITION variable = buffer_in complete dim = 1
#pragma HLS ARRAY_PARTITION variable = buffer_out complete dim = 1
#pragma HLS ARRAY_PARTITION variable = buffer_in dim = 2 cyclic factor = WKA_FFT_BLOCK_PART_FACTOR
#pragma HLS ARRAY_PARTITION variable = buffer_out dim = 2 cyclic factor = WKA_FFT_BLOCK_PART_FACTOR
#pragma HLS BIND_STORAGE variable = buffer_in type = ram_t2p impl = WKA_FFT_BUF_STORAGE_IMPL
#pragma HLS BIND_STORAGE variable = buffer_out type = ram_t2p impl = WKA_FFT_BUF_STORAGE_IMPL

ROW_LOOP:
    for (int step = 0; step < N / WKA_FFT_PAR_ROWS; step++) {
        int base_row = step * WKA_FFT_PAR_ROWS;
        if (fused_stolt) {
            load_bulk_stolt_block(in, buffer_in, base_row);
        } else {
            load_transform_block(in, buffer_in, base_row, load_shift, apply_window);
        }
        compute_fft_block(buffer_in, buffer_out, inverse);
        store_transform_block(buffer_out, out, base_row, store_shift, inverse);
    }
}

void run_row_transform(bus_t *__restrict in, bus_t *__restrict out, row_transform_mode_t mode) {
#pragma HLS INLINE off
    bool inverse = mode != WKA_ROW_FORWARD;
    bool load_shift = inverse && mode != WKA_ROW_BULK_STOLT_WINDOW_INVERSE;
    bool store_shift = !inverse;
    bool apply_window = mode == WKA_ROW_WINDOW_INVERSE;
    bool fused_stolt = mode == WKA_ROW_BULK_STOLT_WINDOW_INVERSE;
    run_row_transform_impl(in, out, inverse, load_shift, store_shift, apply_window, fused_stolt);
}

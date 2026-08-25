//===- fft_core.cpp - Hand-written radix-4 HLS engine ---------------------===//
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

template <bool Inverse>
static void lookup_twiddle(int index, acc_t &wr, acc_t &wi) {
#pragma HLS INLINE
  bool upper_half = index >= N / 2;
  int rom_index = upper_half ? index - N / 2 : index;
  acc_t sign = upper_half ? -1.0f : 1.0f;
  wr = sign * WKA_TWIDDLE_COS_ROM[rom_index];
  acc_t forward_sin = sign * WKA_TWIDDLE_SIN_ROM[rom_index];
  wi = Inverse ? -forward_sin : forward_sin;
}

template <bool LoadShift>
static void load_transform_block(const plane_t *__restrict in_re,
                                 const plane_t *__restrict in_im,
                                 data_t buffer[WKA_FFT_PAR_ROWS][N],
                                 int base_row) {
#pragma HLS INLINE off

LOAD_ROWS:
  for (int row_lane = 0; row_lane < WKA_FFT_PAR_ROWS; row_lane++) {
    int row = base_row + row_lane;
    size_t row_base = static_cast<size_t>(row) * PLANE_WORDS_PER_ROW;
  LOAD_WORDS:
    for (int word = 0; word < PLANE_WORDS_PER_ROW; word++) {
#pragma HLS PIPELINE II = WKA_ROW_LOAD_STORE_II
      plane_t packed_re = in_re[row_base + word];
      plane_t packed_im = in_im[row_base + word];
      for (int lane = 0; lane < PLANE_LANES; lane++) {
#pragma HLS UNROLL
        int column = word * PLANE_LANES + lane;
        int target = LoadShift
                         ? ((column < N / 2) ? column + N / 2 : column - N / 2)
                         : column;
        buffer[row_lane][target] = data_t(packed_re[lane], packed_im[lane]);
      }
    }
  }
}

template <bool LoadShift>
static void load_windowed_transform_block(const plane_t *__restrict in_re,
                                          const plane_t *__restrict in_im,
                                          const real_t window[N],
                                          data_t buffer[WKA_FFT_PAR_ROWS][N],
                                          int base_row) {
#pragma HLS INLINE off

LOAD_WINDOWED_ROWS:
  for (int row_lane = 0; row_lane < WKA_FFT_PAR_ROWS; row_lane++) {
    int row = base_row + row_lane;
    size_t row_base = static_cast<size_t>(row) * PLANE_WORDS_PER_ROW;
  LOAD_WINDOWED_WORDS:
    for (int word = 0; word < PLANE_WORDS_PER_ROW; word++) {
#pragma HLS PIPELINE II = WKA_ROW_LOAD_STORE_II
      plane_t packed_re = in_re[row_base + word];
      plane_t packed_im = in_im[row_base + word];
      for (int lane = 0; lane < PLANE_LANES; lane++) {
#pragma HLS UNROLL
        int column = word * PLANE_LANES + lane;
        int target = LoadShift
                         ? ((column < N / 2) ? column + N / 2 : column - N / 2)
                         : column;
        acc_t coefficient = window[column];
        buffer[row_lane][target] = data_t(packed_re[lane] * coefficient,
                                          packed_im[lane] * coefficient);
      }
    }
  }
}

template <bool StoreShift, bool Inverse>
static void store_transform_block(data_t buffer[WKA_FFT_PAR_ROWS][N],
                                  plane_t *__restrict out_re,
                                  plane_t *__restrict out_im, int base_row) {
#pragma HLS INLINE off

  const acc_t scale = Inverse ? 1.0f / static_cast<float>(N) : 1.0f;
STORE_ROWS:
  for (int row_lane = 0; row_lane < WKA_FFT_PAR_ROWS; row_lane++) {
    int row = base_row + row_lane;
    size_t row_base = static_cast<size_t>(row) * PLANE_WORDS_PER_ROW;
  STORE_WORDS:
    for (int word = 0; word < PLANE_WORDS_PER_ROW; word++) {
#pragma HLS PIPELINE II = WKA_ROW_LOAD_STORE_II
      plane_t packed_re;
      plane_t packed_im;
      for (int lane = 0; lane < PLANE_LANES; lane++) {
#pragma HLS UNROLL
        int column = word * PLANE_LANES + lane;
        int source = StoreShift
                         ? ((column < N / 2) ? column + N / 2 : column - N / 2)
                         : column;
        data_t value = buffer[row_lane][source];
        packed_re[lane] = value.real() * scale;
        packed_im[lane] = value.imag() * scale;
      }
      out_re[row_base + word] = packed_re;
      out_im[row_base + word] = packed_im;
    }
  }
}

template <bool Inverse>
static void compute_fft_block(data_t buf_in[WKA_FFT_PAR_ROWS][N],
                              data_t buf_out[WKA_FFT_PAR_ROWS][N]) {
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable = WKA_TWIDDLE_COS_ROM type = rom_np impl =   \
    bram
#pragma HLS BIND_STORAGE variable = WKA_TWIDDLE_SIN_ROM type = rom_np impl =   \
    bram

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
      lookup_twiddle<Inverse>(twiddle1, w1r, w1i);
      lookup_twiddle<Inverse>(twiddle2, w2r, w2i);
      lookup_twiddle<Inverse>(twiddle3, w3r, w3i);

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

        buf_out[row_lane][index0] =
            data_t(ar + br + cr + dr, ai + bi + ci + di);
        buf_out[row_lane][index2] =
            data_t(ar - br + cr - dr, ai - bi + ci - di);
        if (Inverse) {
          buf_out[row_lane][index1] =
              data_t(ar - bi - cr + di, ai + br - ci - dr);
          buf_out[row_lane][index3] =
              data_t(ar + bi - cr - di, ai - br - ci + dr);
        } else {
          buf_out[row_lane][index1] =
              data_t(ar + bi - cr - di, ai - br - ci + dr);
          buf_out[row_lane][index3] =
              data_t(ar - bi - cr + di, ai + br - ci - dr);
        }
      }
    }
  }
}

template <bool Inverse, bool LoadShift, bool StoreShift>
static void run_regular_row_transform(const plane_t *__restrict in_re,
                                      const plane_t *__restrict in_im,
                                      plane_t *__restrict out_re,
                                      plane_t *__restrict out_im) {
#pragma HLS INLINE off

  static data_t buffer_in[WKA_FFT_PAR_ROWS][N];
  static data_t buffer_out[WKA_FFT_PAR_ROWS][N];
#pragma HLS AGGREGATE variable = buffer_in compact = auto
#pragma HLS AGGREGATE variable = buffer_out compact = auto
#pragma HLS ARRAY_PARTITION variable = buffer_in complete dim = 1
#pragma HLS ARRAY_PARTITION variable = buffer_out complete dim = 1
#pragma HLS ARRAY_PARTITION variable = buffer_in dim = 2 cyclic factor =       \
    WKA_FFT_BLOCK_PART_FACTOR
#pragma HLS ARRAY_PARTITION variable = buffer_out dim = 2 cyclic factor =      \
    WKA_FFT_BLOCK_PART_FACTOR
#pragma HLS BIND_STORAGE variable = buffer_in type = ram_t2p impl =            \
    WKA_FFT_BUF_STORAGE_IMPL
#pragma HLS BIND_STORAGE variable = buffer_out type = ram_t2p impl =           \
    WKA_FFT_BUF_STORAGE_IMPL

REGULAR_ROW_LOOP:
  for (int step = 0; step < N / WKA_FFT_PAR_ROWS; step++) {
    int base_row = step * WKA_FFT_PAR_ROWS;
    load_transform_block<LoadShift>(in_re, in_im, buffer_in, base_row);
    compute_fft_block<Inverse>(buffer_in, buffer_out);
    store_transform_block<StoreShift, Inverse>(buffer_out, out_re, out_im,
                                               base_row);
  }
}

static void load_window(const real_t *__restrict input, real_t output[N]) {
#pragma HLS INLINE off
LOAD_WINDOW:
  for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II = 1
    output[i] = input[i];
  }
}

void run_row_forward(const plane_t *__restrict in_re,
                     const plane_t *__restrict in_im,
                     plane_t *__restrict out_re, plane_t *__restrict out_im) {
#pragma HLS INLINE off
  run_regular_row_transform<false, false, true>(in_re, in_im, out_re, out_im);
}

void run_row_inverse(const plane_t *__restrict in_re,
                     const plane_t *__restrict in_im,
                     plane_t *__restrict out_re, plane_t *__restrict out_im) {
#pragma HLS INLINE off
  run_regular_row_transform<true, true, false>(in_re, in_im, out_re, out_im);
}

void run_row_window_inverse(const plane_t *__restrict in_re,
                            const plane_t *__restrict in_im,
                            const real_t *__restrict window,
                            plane_t *__restrict out_re,
                            plane_t *__restrict out_im) {
#pragma HLS INLINE off

  static real_t window_cache[N];
  static data_t buffer_in[WKA_FFT_PAR_ROWS][N];
  static data_t buffer_out[WKA_FFT_PAR_ROWS][N];
#pragma HLS ARRAY_PARTITION variable = window_cache cyclic factor =            \
    PLANE_LANES dim = 1
#pragma HLS BIND_STORAGE variable = window_cache type = ram_t2p impl = bram
#pragma HLS AGGREGATE variable = buffer_in compact = auto
#pragma HLS AGGREGATE variable = buffer_out compact = auto
#pragma HLS ARRAY_PARTITION variable = buffer_in complete dim = 1
#pragma HLS ARRAY_PARTITION variable = buffer_out complete dim = 1
#pragma HLS ARRAY_PARTITION variable = buffer_in dim = 2 cyclic factor =       \
    WKA_FFT_BLOCK_PART_FACTOR
#pragma HLS ARRAY_PARTITION variable = buffer_out dim = 2 cyclic factor =      \
    WKA_FFT_BLOCK_PART_FACTOR
#pragma HLS BIND_STORAGE variable = buffer_in type = ram_t2p impl =            \
    WKA_FFT_BUF_STORAGE_IMPL
#pragma HLS BIND_STORAGE variable = buffer_out type = ram_t2p impl =           \
    WKA_FFT_BUF_STORAGE_IMPL

  load_window(window, window_cache);
WINDOW_ROW_LOOP:
  for (int step = 0; step < N / WKA_FFT_PAR_ROWS; step++) {
    int base_row = step * WKA_FFT_PAR_ROWS;
    load_windowed_transform_block<true>(in_re, in_im, window_cache, buffer_in,
                                        base_row);
    compute_fft_block<true>(buffer_in, buffer_out);
    store_transform_block<false, true>(buffer_out, out_re, out_im, base_row);
  }
}

void run_row_bulk_stolt_window_inverse(const plane_t *__restrict in_re,
                                       const plane_t *__restrict in_im,
                                       const real_t *__restrict range_window,
                                       plane_t *__restrict out_re,
                                       plane_t *__restrict out_im) {
#pragma HLS INLINE off

  static real_t window_cache[N];
  static data_t buffer_in[WKA_FFT_PAR_ROWS][N];
  static data_t buffer_out[WKA_FFT_PAR_ROWS][N];
#pragma HLS ARRAY_PARTITION variable = window_cache cyclic factor =            \
    WKA_STOLT_OUT_LANES dim = 1
#pragma HLS BIND_STORAGE variable = window_cache type = ram_t2p impl = bram
#pragma HLS AGGREGATE variable = buffer_in compact = auto
#pragma HLS AGGREGATE variable = buffer_out compact = auto
#pragma HLS ARRAY_PARTITION variable = buffer_in complete dim = 1
#pragma HLS ARRAY_PARTITION variable = buffer_out complete dim = 1
#pragma HLS ARRAY_PARTITION variable = buffer_in dim = 2 cyclic factor =       \
    WKA_FFT_BLOCK_PART_FACTOR
#pragma HLS ARRAY_PARTITION variable = buffer_out dim = 2 cyclic factor =      \
    WKA_FFT_BLOCK_PART_FACTOR
#pragma HLS BIND_STORAGE variable = buffer_in type = ram_t2p impl =            \
    WKA_FFT_BUF_STORAGE_IMPL
#pragma HLS BIND_STORAGE variable = buffer_out type = ram_t2p impl =           \
    WKA_FFT_BUF_STORAGE_IMPL

  load_window(range_window, window_cache);
BULK_STOLT_ROW_LOOP:
  for (int step = 0; step < N / WKA_FFT_PAR_ROWS; step++) {
    int base_row = step * WKA_FFT_PAR_ROWS;
    load_bulk_stolt_block(in_re, in_im, window_cache, buffer_in, base_row);
    compute_fft_block<true>(buffer_in, buffer_out);
    store_transform_block<false, true>(buffer_out, out_re, out_im, base_row);
  }
}

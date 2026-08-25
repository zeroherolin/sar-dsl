//===- stolt_interpolation.cpp - Hand-written Stolt HLS kernel ------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "stolt_interpolation.h"
#include "generated/wka_luts.h"

static const calc_t WKA_SINC_TAP_SIGN[WKA_STOLT_WEIGHT_TAP_COUNT] = {
    -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
static const calc_t WKA_SINC_TAP_COS[WKA_STOLT_WEIGHT_TAP_COUNT] = {
    -0.7071067811865476, 0.0, 0.7071067811865476,  1.0,
    0.7071067811865476,  0.0, -0.7071067811865476, -1.0};
static const calc_t WKA_SINC_TAP_SIN[WKA_STOLT_WEIGHT_TAP_COUNT] = {
    -0.7071067811865476, -1.0, -0.7071067811865476, 0.0,
    0.7071067811865476,  1.0,  0.7071067811865476,  0.0};

static void input_phase(int column, real_t &phase_cos, real_t &phase_sin) {
#pragma HLS INLINE
  const int phase_step_bins = WKA_STOLT_TIME_SHIFT_SAMPLES;
  int twiddle_index = (column * phase_step_bins) & (N - 1);
  bool upper_half = twiddle_index >= N / 2;
  int rom_index = upper_half ? twiddle_index - N / 2 : twiddle_index;
  real_t sign = upper_half ? -1.0f : 1.0f;
  phase_cos = sign * WKA_TWIDDLE_COS_ROM[rom_index];
  phase_sin = -sign * WKA_TWIDDLE_SIN_ROM[rom_index];
}

static void read_source_row(const plane_t *__restrict in_re,
                            const plane_t *__restrict in_im,
                            data_t source[WKA_STOLT_CACHE_COPIES][N], int row) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION operation instances = fsqrt limit = WKA_STOLT_OUT_LANES
#pragma HLS ALLOCATION operation instances = fdiv limit = WKA_STOLT_OUT_LANES
#pragma HLS ALLOCATION operation instances = fmul limit =                      \
    WKA_STOLT_READ_FMUL_LIMIT
#pragma HLS BIND_STORAGE variable = WKA_TWIDDLE_COS_ROM type = rom_2p impl =   \
    bram
#pragma HLS BIND_STORAGE variable = WKA_TWIDDLE_SIN_ROM type = rom_2p impl =   \
    bram

  const calc_t df_a = PRF / N;
  const calc_t df_r = FS / N;
  const calc_t fa_start = -PRF / 2.0;
  const calc_t fr_start = -FS / 2.0;
  const calc_t coeff = 4.0 * PI * R0 / C0;
  const calc_t pi_over_kr = PI / KR;
  const calc_t c_over_2v = C0 / (2.0 * VR);

  calc_t fa = fa_start + row * df_a;
  calc_t tmp_a = c_over_2v * fa;
  calc_t term2 = tmp_a * tmp_a;

  size_t row_base = static_cast<size_t>(row) * PLANE_WORDS_PER_ROW;
  static plane_t packed_re[PLANE_WORDS_PER_ROW];
  static plane_t packed_im[PLANE_WORDS_PER_ROW];
#pragma HLS BIND_STORAGE variable = packed_re type = ram_t2p impl =            \
    WKA_STOLT_PACKED_STORAGE_IMPL
#pragma HLS BIND_STORAGE variable = packed_im type = ram_t2p impl =            \
    WKA_STOLT_PACKED_STORAGE_IMPL

READ_PACKED_ROW:
  for (int word = 0; word < PLANE_WORDS_PER_ROW; word++) {
#pragma HLS PIPELINE II = WKA_STOLT_READ_WRITE_II
    packed_re[word] = in_re[row_base + word];
    packed_im[word] = in_im[row_base + word];
  }

PROCESS_GROUPS:
  for (int group_index = 0; group_index < N / WKA_STOLT_OUT_LANES;
       group_index++) {
#pragma HLS PIPELINE II = WKA_STOLT_READ_WRITE_II
    int groups_per_word = PLANE_LANES / WKA_STOLT_OUT_LANES;
    int word = group_index / groups_per_word;
    int group = group_index - word * groups_per_word;
    plane_t word_re = packed_re[word];
    plane_t word_im = packed_im[word];
    for (int lane = 0; lane < WKA_STOLT_OUT_LANES; lane++) {
#pragma HLS UNROLL
      int packed_lane = group * WKA_STOLT_OUT_LANES + lane;
      int column = group_index * WKA_STOLT_OUT_LANES + lane;
      data_t value = data_t(word_re[packed_lane], word_im[packed_lane]);

      calc_t fr = fr_start + column * df_r;
      calc_t x = FC + fr;
      calc_t root_arg = x * x - term2;
      calc_t safe_arg =
          root_arg > WKA_BULK_SAFE_SQRT_EPS ? root_arg : WKA_BULK_SAFE_SQRT_EPS;
      calc_t root = std::sqrt(safe_arg);
      // Stable evaluation of sqrt(x*x - term2) - x.
      calc_t difference = -term2 / (root + x);
      calc_t phase = coeff * difference + fr * fr * pi_over_kr;
      real_t phase_cos = static_cast<real_t>(std::cos(phase));
      real_t phase_sin = static_cast<real_t>(std::sin(phase));
      acc_t rotated_r = value.real() * phase_cos - value.imag() * phase_sin;
      acc_t rotated_i = value.real() * phase_sin + value.imag() * phase_cos;
      value = data_t(rotated_r, rotated_i);

      real_t shift_cos;
      real_t shift_sin;
      input_phase(column, shift_cos, shift_sin);
      acc_t shifted_r = value.real() * shift_cos - value.imag() * shift_sin;
      acc_t shifted_i = value.real() * shift_sin + value.imag() * shift_cos;
      data_t shifted(shifted_r, shifted_i);
      for (int copy = 0; copy < WKA_STOLT_CACHE_COPIES; copy++) {
#pragma HLS UNROLL
        source[copy][column] = shifted;
      }
    }
  }
}

static void compute_stolt_row(data_t source[WKA_STOLT_CACHE_COPIES][N],
                              const real_t range_window[N], data_t output[N],
                              int row) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION operation instances = fsqrt limit = WKA_STOLT_OUT_LANES
#pragma HLS ALLOCATION operation instances = fdiv limit = WKA_STOLT_OUT_LANES
#pragma HLS ALLOCATION operation instances = fmul limit =                      \
    WKA_STOLT_INTERP_FMUL_LIMIT
#pragma HLS ARRAY_PARTITION variable = WKA_SINC_TAP_SIGN complete dim = 1
#pragma HLS ARRAY_PARTITION variable = WKA_SINC_TAP_COS complete dim = 1
#pragma HLS ARRAY_PARTITION variable = WKA_SINC_TAP_SIN complete dim = 1

  const calc_t df_a = PRF / N;
  const calc_t df_r = FS / N;
  const calc_t fa_start = -PRF / 2.0;
  const calc_t fr_start = -FS / 2.0;
  const calc_t inv_df_r = 1.0 / df_r;
  const calc_t c_over_2v = C0 / (2.0 * VR);
  calc_t fa = fa_start + row * df_a;
  calc_t tmp_a = c_over_2v * fa;
  calc_t term2 = tmp_a * tmp_a;

INTERPOLATE:
  for (int column_base = 0; column_base < N;
       column_base += WKA_STOLT_OUT_LANES) {
#pragma HLS PIPELINE II = WKA_STOLT_INTERP_II
  OUTPUT_LANES:
    for (int lane = 0; lane < WKA_STOLT_OUT_LANES; lane++) {
#pragma HLS UNROLL
      int column = column_base + lane;
      const int cache_copy = lane % WKA_STOLT_CACHE_COPIES;
      calc_t fy = fr_start + column * df_r;
      calc_t x = FC + fy;
      calc_t root = std::sqrt(x * x + term2);
      // Stable evaluation of sqrt(x*x + term2) - x.
      calc_t delta = term2 / (root + x);
      calc_t idx_float = column + delta * inv_df_r;
      int idx_int = wka_floor_to_int(idx_float);
      calc_t frac = idx_float - idx_int;
      calc_t sin_pi_frac = std::sin(PI * frac);
      calc_t window_angle = PI * frac * 0.25;
      calc_t sin_window = std::sin(window_angle);
      calc_t cos_window = std::cos(window_angle);

      calc_t acc_r = 0;
      calc_t acc_i = 0;

    TAPS:
      for (int tap = 0; tap < WKA_STOLT_WEIGHT_TAP_COUNT; tap++) {
#pragma HLS UNROLL
        int k = tap + WKA_STOLT_WEIGHT_TAP_START;
        int idx_k = idx_int + k;
        if (idx_k >= 0 && idx_k < N) {
          data_t sample = source[cache_copy][idx_k];
          calc_t distance = frac - k;
          calc_t denominator = PI * distance;
          calc_t sinc =
              std::abs(distance) < 1.0e-12
                  ? 1.0
                  : WKA_SINC_TAP_SIGN[tap] * sin_pi_frac / denominator;
          calc_t window = 0.5 + 0.5 * (cos_window * WKA_SINC_TAP_COS[tap] +
                                       sin_window * WKA_SINC_TAP_SIN[tap]);
          calc_t weight = sinc * window;
          acc_r += sample.real() * weight;
          acc_i += sample.imag() * weight;
        }
      }

      real_t phase_out = static_cast<real_t>(-2.0 * PI * fy * STOLT_TIME_SHIFT);
      real_t restore_cr = wka_cos_real(phase_out);
      real_t restore_ci = wka_sin_real(phase_out);
      calc_t out_r = acc_r * restore_cr - acc_i * restore_ci;
      calc_t out_i = acc_r * restore_ci + acc_i * restore_cr;
      calc_t coefficient = range_window[column];
      out_r *= coefficient;
      out_i *= coefficient;
      int target = (column < N / 2) ? column + N / 2 : column - N / 2;
      output[target] = data_t(out_r, out_i);
    }
  }
}

void load_bulk_stolt_block(const plane_t *__restrict in_re,
                           const plane_t *__restrict in_im,
                           const real_t range_window[N],
                           data_t fft_input[WKA_FFT_PAR_ROWS][N],
                           int base_row) {
#pragma HLS INLINE off
  static data_t source[WKA_STOLT_CACHE_COPIES][N];
#pragma HLS ARRAY_PARTITION variable = source complete dim = 1
#pragma HLS ARRAY_PARTITION variable = source cyclic factor =                  \
    WKA_STOLT_ROW_PART_FACTOR dim = 2
#pragma HLS BIND_STORAGE variable = source type = ram_t2p impl =               \
    WKA_STOLT_ROW_STORAGE_IMPL

BLOCK_ROWS:
  for (int row_lane = 0; row_lane < WKA_FFT_PAR_ROWS; row_lane++) {
    int row = base_row + row_lane;
    read_source_row(in_re, in_im, source, row);
    compute_stolt_row(source, range_window, fft_input[row_lane], row);
  }
}

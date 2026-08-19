//===- stolt_interpolation.cpp - Hand-written Stolt HLS kernel ------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "stolt_interpolation.h"
#include "generated/wka_luts.h"

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

static void read_source_row(bus_t *__restrict in, data_t source[N], int row) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION operation instances = fsqrt limit = WKA_STOLT_OUT_LANES
#pragma HLS ALLOCATION operation instances = fdiv limit = WKA_STOLT_OUT_LANES
#pragma HLS ALLOCATION operation instances = fmul limit =                      \
    WKA_STOLT_READ_FMUL_LIMIT
#pragma HLS BIND_STORAGE variable = WKA_TWIDDLE_COS_ROM type = rom_2p impl =   \
    bram
#pragma HLS BIND_STORAGE variable = WKA_TWIDDLE_SIN_ROM type = rom_2p impl =   \
    bram

  const real_t df_a = PRF / N;
  const real_t df_r = FS / N;
  const real_t fa_start = -PRF / 2.0f;
  const real_t fr_start = -FS / 2.0f;
  const real_t coeff = 4.0f * PI * R0 / C0;
  const real_t pi_over_kr = PI / KR;
  const real_t c_over_2v = C0 / (2.0f * VR);

  real_t fa = fa_start + row * df_a;
  real_t tmp_a = c_over_2v * fa;
  real_t term2 = tmp_a * tmp_a;

  size_t row_base = (static_cast<size_t>(row) * N) / BUS_LANES;
  static bus_t packed_row[N / BUS_LANES];
#pragma HLS BIND_STORAGE variable = packed_row type = ram_t2p impl =           \
    WKA_STOLT_PACKED_STORAGE_IMPL

READ_PACKED_ROW:
  for (int word = 0; word < N / BUS_LANES; word++) {
#pragma HLS PIPELINE II = WKA_STOLT_READ_WRITE_II
    packed_row[word] = in[row_base + word];
  }

PROCESS_GROUPS:
  for (int group_index = 0; group_index < N / WKA_STOLT_OUT_LANES;
       group_index++) {
#pragma HLS PIPELINE II = WKA_STOLT_READ_WRITE_II
    int groups_per_word = BUS_LANES / WKA_STOLT_OUT_LANES;
    int word = group_index / groups_per_word;
    int group = group_index - word * groups_per_word;
    bus_t packed = packed_row[word];
    for (int lane = 0; lane < WKA_STOLT_OUT_LANES; lane++) {
#pragma HLS UNROLL
      int packed_lane = group * WKA_STOLT_OUT_LANES + lane;
      int column = group_index * WKA_STOLT_OUT_LANES + lane;
      data_t value = unpack_data(static_cast<uint64_t>(
          packed.range((packed_lane + 1) * WKA_COMPLEX_SAMPLE_BITS - 1,
                       packed_lane * WKA_COMPLEX_SAMPLE_BITS)));

      real_t fr = fr_start + column * df_r;
      real_t x = FC + fr;
      real_t root_arg = x * x - term2;
      real_t safe_arg =
          root_arg > WKA_BULK_SAFE_SQRT_EPS ? root_arg : WKA_BULK_SAFE_SQRT_EPS;
      real_t root = wka_sqrt_real(safe_arg);
      // Stable evaluation of sqrt(x*x - term2) - x.
      real_t difference = -term2 / (root + x);
      real_t phase = coeff * difference + fr * fr * pi_over_kr;
      real_t phase_cos = wka_cos_real(phase);
      real_t phase_sin = wka_sin_real(phase);
      acc_t rotated_r = value.real() * phase_cos - value.imag() * phase_sin;
      acc_t rotated_i = value.real() * phase_sin + value.imag() * phase_cos;
      value = data_t(rotated_r, rotated_i);

      real_t shift_cos;
      real_t shift_sin;
      input_phase(column, shift_cos, shift_sin);
      acc_t shifted_r = value.real() * shift_cos - value.imag() * shift_sin;
      acc_t shifted_i = value.real() * shift_sin + value.imag() * shift_cos;
      source[column] = data_t(shifted_r, shifted_i);
    }
  }
}

static void compute_stolt_row(data_t source[N], data_t output[N], int row) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION operation instances = fsqrt limit = WKA_STOLT_OUT_LANES
#pragma HLS ALLOCATION operation instances = fdiv limit = WKA_STOLT_OUT_LANES
#pragma HLS ALLOCATION operation instances = fmul limit =                      \
    WKA_STOLT_INTERP_FMUL_LIMIT
#pragma HLS ARRAY_PARTITION variable = WKA_STOLT_WEIGHT_ROM complete dim = 1
#pragma HLS BIND_STORAGE variable = WKA_STOLT_WEIGHT_ROM type = rom_2p impl =  \
    bram
#pragma HLS ARRAY_PARTITION variable = WKA_RANGE_WINDOW_ROM cyclic factor =    \
    WKA_STOLT_OUT_LANES dim = 1
#pragma HLS BIND_STORAGE variable = WKA_RANGE_WINDOW_ROM type = rom_2p impl =  \
    bram

  const real_t df_a = PRF / N;
  const real_t df_r = FS / N;
  const real_t fa_start = -PRF / 2.0f;
  const real_t fr_start = -FS / 2.0f;
  const real_t inv_df_r = 1.0f / df_r;
  const real_t c_over_2v = C0 / (2.0f * VR);
  real_t fa = fa_start + row * df_a;
  real_t tmp_a = c_over_2v * fa;
  real_t term2 = tmp_a * tmp_a;

INTERPOLATE:
  for (int column_base = 0; column_base < N;
       column_base += WKA_STOLT_OUT_LANES) {
#pragma HLS PIPELINE II = WKA_STOLT_INTERP_II
  OUTPUT_LANES:
    for (int lane = 0; lane < WKA_STOLT_OUT_LANES; lane++) {
#pragma HLS UNROLL
      int column = column_base + lane;
      real_t fy = fr_start + column * df_r;
      real_t x = FC + fy;
      real_t root = wka_sqrt_real(x * x + term2);
      // Stable evaluation of sqrt(x*x + term2) - x.
      real_t delta = term2 / (root + x);
      real_t idx_float = column + delta * inv_df_r;
      int idx_int = wka_floor_to_int(idx_float);
      real_t frac = idx_float - idx_int;
      int lut_idx = static_cast<int>(frac * WKA_STOLT_WEIGHT_LUT_SIZE);
      if (lut_idx < 0) {
        lut_idx = 0;
      } else if (lut_idx >= WKA_STOLT_WEIGHT_LUT_SIZE) {
        lut_idx = WKA_STOLT_WEIGHT_LUT_SIZE - 1;
      }

      acc_t acc_r = 0;
      acc_t acc_i = 0;

    TAPS:
      for (int tap = 0; tap < WKA_STOLT_WEIGHT_TAP_COUNT; tap++) {
#pragma HLS UNROLL
        int k = tap + WKA_STOLT_WEIGHT_TAP_START;
        int idx_k = idx_int + k;
        if (idx_k >= 0 && idx_k < N) {
          data_t sample = source[idx_k];
          acc_t weight = WKA_STOLT_WEIGHT_ROM[tap][lut_idx];
          acc_r += sample.real() * weight;
          acc_i += sample.imag() * weight;
        }
      }

      real_t phase_out = -2.0f * PI * fy * STOLT_TIME_SHIFT;
      real_t restore_cr = wka_cos_real(phase_out);
      real_t restore_ci = wka_sin_real(phase_out);
      acc_t out_r = acc_r * restore_cr - acc_i * restore_ci;
      acc_t out_i = acc_r * restore_ci + acc_i * restore_cr;
      acc_t coefficient = WKA_RANGE_WINDOW_ROM[column];
      out_r *= coefficient;
      out_i *= coefficient;
      int target = (column < N / 2) ? column + N / 2 : column - N / 2;
      output[target] = data_t(out_r, out_i);
    }
  }
}

void load_bulk_stolt_block(bus_t *__restrict in,
                           data_t fft_input[WKA_FFT_PAR_ROWS][N],
                           int base_row) {
#pragma HLS INLINE off
  static data_t source[N];
#pragma HLS ARRAY_PARTITION variable = source cyclic factor =                  \
    WKA_STOLT_ROW_PART_FACTOR dim = 1
#pragma HLS BIND_STORAGE variable = source type = ram_t2p impl =               \
    WKA_STOLT_ROW_STORAGE_IMPL

BLOCK_ROWS:
  for (int row_lane = 0; row_lane < WKA_FFT_PAR_ROWS; row_lane++) {
    int row = base_row + row_lane;
    read_source_row(in, source, row);
    compute_stolt_row(source, fft_input[row_lane], row);
  }
}

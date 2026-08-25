//===- corner_turn.cpp - Hand-written HLS corner turn ---------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "corner_turn.h"

static void load_tile_corner_turn(
    const plane_t *__restrict in_re, const plane_t *__restrict in_im,
    bus_t tile[TILE_SIZE][TILE_SIZE / PLANE_LANES], int ti, int tj) {
#pragma HLS INLINE off

  const int row_base = ti * TILE_SIZE;
  const int col_base = tj * TILE_SIZE;

LOAD_TILE:
  for (int r = 0; r < TILE_SIZE; r++) {
    size_t offset = static_cast<size_t>(row_base + r) * PLANE_WORDS_PER_ROW +
                    col_base / PLANE_LANES;
    for (int word = 0; word < TILE_SIZE / PLANE_LANES; word++) {
#pragma HLS PIPELINE II = WKA_CORNER_LOAD_STORE_II
      plane_t packed_re = in_re[offset + word];
      plane_t packed_im = in_im[offset + word];
      bus_t packed = 0;
      for (int lane = 0; lane < PLANE_LANES; lane++) {
#pragma HLS UNROLL
        packed.range((lane + 1) * WKA_COMPLEX_SAMPLE_BITS - 1,
                     lane * WKA_COMPLEX_SAMPLE_BITS) =
            pack_data(data_t(packed_re[lane], packed_im[lane]));
      }
      tile[r][word] = packed;
    }
  }
}

static void store_tile_complex(bus_t tile[TILE_SIZE][TILE_SIZE / PLANE_LANES],
                               plane_t *__restrict out_re,
                               plane_t *__restrict out_im, int ti, int tj) {
#pragma HLS INLINE off

  const int row_base = tj * TILE_SIZE;
  const int col_base = ti * TILE_SIZE;

STORE_COMPLEX_TILE:
  for (int c = 0; c < TILE_SIZE; c++) {
    size_t offset = static_cast<size_t>(row_base + c) * PLANE_WORDS_PER_ROW +
                    col_base / PLANE_LANES;
    int source_word = c / PLANE_LANES;
    int source_lane = c % PLANE_LANES;
    for (int output_word = 0; output_word < TILE_SIZE / PLANE_LANES;
         output_word++) {
#pragma HLS PIPELINE II = WKA_CORNER_LOAD_STORE_II
      plane_t packed_re;
      plane_t packed_im;
      for (int lane = 0; lane < PLANE_LANES; lane++) {
#pragma HLS UNROLL
        int source_row = output_word * PLANE_LANES + lane;
        bus_t source = tile[source_row][source_word];
        data_t value = unpack_data(static_cast<uint64_t>(
            source.range((source_lane + 1) * WKA_COMPLEX_SAMPLE_BITS - 1,
                         source_lane * WKA_COMPLEX_SAMPLE_BITS)));
        packed_re[lane] = value.real();
        packed_im[lane] = value.imag();
      }
      out_re[offset + output_word] = packed_re;
      out_im[offset + output_word] = packed_im;
    }
  }
}

static void store_tile_magnitude(bus_t tile[TILE_SIZE][TILE_SIZE / PLANE_LANES],
                                 plane_t *__restrict out, int ti, int tj) {
#pragma HLS INLINE off

  const int row_base = tj * TILE_SIZE;
  const int col_base = ti * TILE_SIZE;

STORE_MAGNITUDE_TILE:
  for (int c = 0; c < TILE_SIZE; c++) {
    size_t offset = static_cast<size_t>(row_base + c) * PLANE_WORDS_PER_ROW +
                    col_base / PLANE_LANES;
    int source_word = c / PLANE_LANES;
    int source_lane = c % PLANE_LANES;
    for (int output_word = 0; output_word < TILE_SIZE / PLANE_LANES;
         output_word++) {
#pragma HLS PIPELINE II = WKA_CORNER_LOAD_STORE_II
      plane_t packed;
      for (int lane = 0; lane < PLANE_LANES; lane++) {
#pragma HLS UNROLL
        int source_row = output_word * PLANE_LANES + lane;
        bus_t source = tile[source_row][source_word];
        data_t value = unpack_data(static_cast<uint64_t>(
            source.range((source_lane + 1) * WKA_COMPLEX_SAMPLE_BITS - 1,
                         source_lane * WKA_COMPLEX_SAMPLE_BITS)));
        packed[lane] = wka_sqrt_real(value.real() * value.real() +
                                     value.imag() * value.imag());
      }
      out[offset + output_word] = packed;
    }
  }
}

static void process_complex_ping_to_pong(
    const plane_t *__restrict in_re, const plane_t *__restrict in_im,
    plane_t *__restrict out_re, plane_t *__restrict out_im,
    bus_t tile_ping[TILE_SIZE][TILE_SIZE / PLANE_LANES],
    bus_t tile_pong[TILE_SIZE][TILE_SIZE / PLANE_LANES], int prev_ti,
    int prev_tj, int next_ti, int next_tj) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  load_tile_corner_turn(in_re, in_im, tile_pong, next_ti, next_tj);
  store_tile_complex(tile_ping, out_re, out_im, prev_ti, prev_tj);
}

static void process_complex_pong_to_ping(
    const plane_t *__restrict in_re, const plane_t *__restrict in_im,
    plane_t *__restrict out_re, plane_t *__restrict out_im,
    bus_t tile_ping[TILE_SIZE][TILE_SIZE / PLANE_LANES],
    bus_t tile_pong[TILE_SIZE][TILE_SIZE / PLANE_LANES], int prev_ti,
    int prev_tj, int next_ti, int next_tj) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  load_tile_corner_turn(in_re, in_im, tile_ping, next_ti, next_tj);
  store_tile_complex(tile_pong, out_re, out_im, prev_ti, prev_tj);
}

static void process_magnitude_ping_to_pong(
    const plane_t *__restrict in_re, const plane_t *__restrict in_im,
    plane_t *__restrict out,
    bus_t tile_ping[TILE_SIZE][TILE_SIZE / PLANE_LANES],
    bus_t tile_pong[TILE_SIZE][TILE_SIZE / PLANE_LANES], int prev_ti,
    int prev_tj, int next_ti, int next_tj) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  load_tile_corner_turn(in_re, in_im, tile_pong, next_ti, next_tj);
  store_tile_magnitude(tile_ping, out, prev_ti, prev_tj);
}

static void process_magnitude_pong_to_ping(
    const plane_t *__restrict in_re, const plane_t *__restrict in_im,
    plane_t *__restrict out,
    bus_t tile_ping[TILE_SIZE][TILE_SIZE / PLANE_LANES],
    bus_t tile_pong[TILE_SIZE][TILE_SIZE / PLANE_LANES], int prev_ti,
    int prev_tj, int next_ti, int next_tj) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  load_tile_corner_turn(in_re, in_im, tile_ping, next_ti, next_tj);
  store_tile_magnitude(tile_pong, out, prev_ti, prev_tj);
}

void corner_turn_complex(const plane_t *__restrict in_re,
                         const plane_t *__restrict in_im,
                         plane_t *__restrict out_re,
                         plane_t *__restrict out_im) {
#pragma HLS INLINE off

  bus_t tile_ping[TILE_SIZE][TILE_SIZE / PLANE_LANES];
  bus_t tile_pong[TILE_SIZE][TILE_SIZE / PLANE_LANES];
#pragma HLS ARRAY_PARTITION variable = tile_ping cyclic factor =               \
    PLANE_LANES dim = 1
#pragma HLS ARRAY_PARTITION variable = tile_pong cyclic factor =               \
    PLANE_LANES dim = 1
#pragma HLS BIND_STORAGE variable = tile_ping type = ram_t2p impl = bram
#pragma HLS BIND_STORAGE variable = tile_pong type = ram_t2p impl = bram

  const int num_tiles = N / TILE_SIZE;
  const int total_tiles = num_tiles * num_tiles;
  load_tile_corner_turn(in_re, in_im, tile_ping, 0, 0);

COMPLEX_TILE_PIPE:
  for (int tile_idx = 1; tile_idx < total_tiles; tile_idx++) {
#pragma HLS LOOP_FLATTEN off
    int prev_idx = tile_idx - 1;
    int prev_ti = prev_idx / num_tiles;
    int prev_tj = prev_idx - prev_ti * num_tiles;
    int next_ti = tile_idx / num_tiles;
    int next_tj = tile_idx - next_ti * num_tiles;

    if (tile_idx & 1) {
      process_complex_ping_to_pong(in_re, in_im, out_re, out_im, tile_ping,
                                   tile_pong, prev_ti, prev_tj, next_ti,
                                   next_tj);
    } else {
      process_complex_pong_to_ping(in_re, in_im, out_re, out_im, tile_ping,
                                   tile_pong, prev_ti, prev_tj, next_ti,
                                   next_tj);
    }
  }

  int last_idx = total_tiles - 1;
  int last_ti = last_idx / num_tiles;
  int last_tj = last_idx - last_ti * num_tiles;
  if (total_tiles & 1) {
    store_tile_complex(tile_ping, out_re, out_im, last_ti, last_tj);
  } else {
    store_tile_complex(tile_pong, out_re, out_im, last_ti, last_tj);
  }
}

void corner_turn_magnitude(const plane_t *__restrict in_re,
                           const plane_t *__restrict in_im,
                           plane_t *__restrict out) {
#pragma HLS INLINE off

  bus_t tile_ping[TILE_SIZE][TILE_SIZE / PLANE_LANES];
  bus_t tile_pong[TILE_SIZE][TILE_SIZE / PLANE_LANES];
#pragma HLS ARRAY_PARTITION variable = tile_ping cyclic factor =               \
    PLANE_LANES dim = 1
#pragma HLS ARRAY_PARTITION variable = tile_pong cyclic factor =               \
    PLANE_LANES dim = 1
#pragma HLS BIND_STORAGE variable = tile_ping type = ram_t2p impl = bram
#pragma HLS BIND_STORAGE variable = tile_pong type = ram_t2p impl = bram

  const int num_tiles = N / TILE_SIZE;
  const int total_tiles = num_tiles * num_tiles;
  load_tile_corner_turn(in_re, in_im, tile_ping, 0, 0);

MAGNITUDE_TILE_PIPE:
  for (int tile_idx = 1; tile_idx < total_tiles; tile_idx++) {
#pragma HLS LOOP_FLATTEN off
    int prev_idx = tile_idx - 1;
    int prev_ti = prev_idx / num_tiles;
    int prev_tj = prev_idx - prev_ti * num_tiles;
    int next_ti = tile_idx / num_tiles;
    int next_tj = tile_idx - next_ti * num_tiles;

    if (tile_idx & 1) {
      process_magnitude_ping_to_pong(in_re, in_im, out, tile_ping, tile_pong,
                                     prev_ti, prev_tj, next_ti, next_tj);
    } else {
      process_magnitude_pong_to_ping(in_re, in_im, out, tile_ping, tile_pong,
                                     prev_ti, prev_tj, next_ti, next_tj);
    }
  }

  int last_idx = total_tiles - 1;
  int last_ti = last_idx / num_tiles;
  int last_tj = last_idx - last_ti * num_tiles;
  if (total_tiles & 1) {
    store_tile_magnitude(tile_ping, out, last_ti, last_tj);
  } else {
    store_tile_magnitude(tile_pong, out, last_ti, last_tj);
  }
}

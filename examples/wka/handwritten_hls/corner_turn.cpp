//===- corner_turn.cpp - Hand-written HLS corner turn --------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "corner_turn.h"

static void load_tile_corner_turn(bus_t *__restrict in,
                                  bus_t tile[TILE_SIZE][TILE_SIZE / BUS_LANES],
                                  int ti, int tj) {
#pragma HLS INLINE off

  const int row_base = ti * TILE_SIZE;
  const int col_base = tj * TILE_SIZE;

LOAD_TILE:
  for (int r = 0; r < TILE_SIZE; r++) {
    size_t offset =
        (static_cast<size_t>(row_base + r) * N + col_base) / BUS_LANES;
    for (int word = 0; word < TILE_SIZE / BUS_LANES; word++) {
#pragma HLS PIPELINE II = WKA_CORNER_LOAD_STORE_II
      tile[r][word] = in[offset + word];
    }
  }
}

static void store_tile_corner_turn(bus_t tile[TILE_SIZE][TILE_SIZE / BUS_LANES],
                                   bus_t *__restrict out, int ti, int tj,
                                   bool magnitude_output) {
#pragma HLS INLINE off

  const int row_base = tj * TILE_SIZE;
  const int col_base = ti * TILE_SIZE;

  if (magnitude_output) {
  STORE_MAGNITUDE_TILE:
    for (int c = 0; c < TILE_SIZE; c++) {
      size_t offset =
          (static_cast<size_t>(row_base + c) * N + col_base) / REAL_BUS_LANES;
      int source_word = c / BUS_LANES;
      int source_lane = c % BUS_LANES;
      for (int output_word = 0; output_word < TILE_SIZE / REAL_BUS_LANES;
           output_word++) {
#pragma HLS PIPELINE II = WKA_CORNER_LOAD_STORE_II
        bus_t packed = 0;
        for (int lane = 0; lane < REAL_BUS_LANES; lane++) {
#pragma HLS UNROLL
          int source_row = output_word * REAL_BUS_LANES + lane;
          bus_t source = tile[source_row][source_word];
          data_t value = unpack_data(static_cast<uint64_t>(
              source.range((source_lane + 1) * WKA_COMPLEX_SAMPLE_BITS - 1,
                           source_lane * WKA_COMPLEX_SAMPLE_BITS)));
          float magnitude = std::sqrt(value.real() * value.real() +
                                      value.imag() * value.imag());
          packed.range((lane + 1) * WKA_IO_SCALAR_BITS - 1,
                       lane * WKA_IO_SCALAR_BITS) = pack_real(magnitude);
        }
        out[offset + output_word] = packed;
      }
    }
    return;
  }

STORE_COMPLEX_TILE:
  for (int c = 0; c < TILE_SIZE; c++) {
    size_t offset =
        (static_cast<size_t>(row_base + c) * N + col_base) / BUS_LANES;
    int source_word = c / BUS_LANES;
    int source_lane = c % BUS_LANES;
    for (int output_word = 0; output_word < TILE_SIZE / BUS_LANES;
         output_word++) {
#pragma HLS PIPELINE II = WKA_CORNER_LOAD_STORE_II
      bus_t packed = 0;
      for (int lane = 0; lane < BUS_LANES; lane++) {
#pragma HLS UNROLL
        int source_row = output_word * BUS_LANES + lane;
        bus_t source = tile[source_row][source_word];
        packed.range((lane + 1) * WKA_COMPLEX_SAMPLE_BITS - 1,
                     lane * WKA_COMPLEX_SAMPLE_BITS) =
            source.range((source_lane + 1) * WKA_COMPLEX_SAMPLE_BITS - 1,
                         source_lane * WKA_COMPLEX_SAMPLE_BITS);
      }
      out[offset + output_word] = packed;
    }
  }
}

static void process_tile_pair_ping_to_pong(
    bus_t *__restrict in, bus_t *__restrict out,
    bus_t tile_ping[TILE_SIZE][TILE_SIZE / BUS_LANES],
    bus_t tile_pong[TILE_SIZE][TILE_SIZE / BUS_LANES], int prev_ti, int prev_tj,
    int next_ti, int next_tj, bool magnitude_output) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  load_tile_corner_turn(in, tile_pong, next_ti, next_tj);
  store_tile_corner_turn(tile_ping, out, prev_ti, prev_tj, magnitude_output);
}

static void process_tile_pair_pong_to_ping(
    bus_t *__restrict in, bus_t *__restrict out,
    bus_t tile_ping[TILE_SIZE][TILE_SIZE / BUS_LANES],
    bus_t tile_pong[TILE_SIZE][TILE_SIZE / BUS_LANES], int prev_ti, int prev_tj,
    int next_ti, int next_tj, bool magnitude_output) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  load_tile_corner_turn(in, tile_ping, next_ti, next_tj);
  store_tile_corner_turn(tile_pong, out, prev_ti, prev_tj, magnitude_output);
}

void corner_turn(bus_t *__restrict in, bus_t *__restrict out,
                 bool magnitude_output) {
#pragma HLS INLINE off

  // Packed rows preserve AXI bursts; row banking supplies the transpose reads.
  bus_t tile_ping[TILE_SIZE][TILE_SIZE / BUS_LANES];
  bus_t tile_pong[TILE_SIZE][TILE_SIZE / BUS_LANES];
#pragma HLS ARRAY_PARTITION variable = tile_ping cyclic factor =               \
    BUS_LANES dim = 1
#pragma HLS ARRAY_PARTITION variable = tile_pong cyclic factor =               \
    BUS_LANES dim = 1
#pragma HLS BIND_STORAGE variable = tile_ping type = ram_t2p impl = bram
#pragma HLS BIND_STORAGE variable = tile_pong type = ram_t2p impl = bram

  const int num_tiles = N / TILE_SIZE;
  const int total_tiles = num_tiles * num_tiles;

  load_tile_corner_turn(in, tile_ping, 0, 0);

TILE_PIPE:
  for (int tile_idx = 1; tile_idx < total_tiles; tile_idx++) {
#pragma HLS LOOP_FLATTEN off
    int prev_idx = tile_idx - 1;
    int prev_ti = prev_idx / num_tiles;
    int prev_tj = prev_idx - prev_ti * num_tiles;
    int next_ti = tile_idx / num_tiles;
    int next_tj = tile_idx - next_ti * num_tiles;

    if (tile_idx & 1) {
      process_tile_pair_ping_to_pong(in, out, tile_ping, tile_pong, prev_ti,
                                     prev_tj, next_ti, next_tj,
                                     magnitude_output);
    } else {
      process_tile_pair_pong_to_ping(in, out, tile_ping, tile_pong, prev_ti,
                                     prev_tj, next_ti, next_tj,
                                     magnitude_output);
    }
  }

  int last_idx = total_tiles - 1;
  int last_ti = last_idx / num_tiles;
  int last_tj = last_idx - last_ti * num_tiles;
  if (total_tiles & 1) {
    store_tile_corner_turn(tile_ping, out, last_ti, last_tj, magnitude_output);
  } else {
    store_tile_corner_turn(tile_pong, out, last_ti, last_tj, magnitude_output);
  }
}

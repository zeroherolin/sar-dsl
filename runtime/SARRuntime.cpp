//===- SARRuntime.cpp - SAR runtime library --------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
// C-ABI kernels backing the sar.fft / sar.ifft / sar.stolt_interp operations
// on CPU targets. Symbols follow the MLIR C interface convention
// (_mlir_ciface_<name>): memrefs are passed as pointers to strided
// descriptors, scalars by value.
//
// Numerical conventions match numpy: the forward DFT is unscaled, the
// inverse DFT is scaled by 1/N. FFT sizes must be powers of two (enforced by
// the sar dialect verifier). All transforms are computed in double precision
// regardless of the storage precision.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace {

//===----------------------------------------------------------------------===//
// MLIR memref descriptors
//===----------------------------------------------------------------------===//

template <typename T, int Rank>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];

  T &at(int64_t i) const { return aligned[offset + i * strides[0]]; }
  T &at(int64_t i, int64_t j) const {
    return aligned[offset + i * strides[0] + j * strides[1]];
  }
};

//===----------------------------------------------------------------------===//
// Parallel helpers
//===----------------------------------------------------------------------===//

/// Runs fn(begin, end) over [0, total) partitioned across hardware threads.
void parallelFor(int64_t total, const std::function<void(int64_t, int64_t)> &fn) {
  unsigned workers = std::thread::hardware_concurrency();
  if (workers <= 1 || total < 2) {
    fn(0, total);
    return;
  }
  workers = static_cast<unsigned>(
      std::min<int64_t>(total, static_cast<int64_t>(workers)));
  std::vector<std::thread> threads;
  threads.reserve(workers);
  int64_t chunk = (total + workers - 1) / workers;
  for (unsigned w = 0; w < workers; ++w) {
    int64_t begin = static_cast<int64_t>(w) * chunk;
    int64_t end = std::min<int64_t>(begin + chunk, total);
    if (begin >= end)
      break;
    threads.emplace_back(fn, begin, end);
  }
  for (auto &t : threads)
    t.join();
}

//===----------------------------------------------------------------------===//
// Radix-2 FFT (double precision workspace)
//===----------------------------------------------------------------------===//

struct FFTPlan {
  int64_t n;
  std::vector<int64_t> bitrev;
  // Twiddles for each butterfly stage, forward direction.
  std::vector<std::complex<double>> twiddles;

  explicit FFTPlan(int64_t size) : n(size) {
    assert(size >= 2 && (size & (size - 1)) == 0 && "size must be 2^k");
    int log2n = 0;
    while ((int64_t{1} << log2n) < n)
      ++log2n;

    bitrev.resize(n);
    for (int64_t i = 0; i < n; ++i) {
      int64_t r = 0;
      for (int b = 0; b < log2n; ++b)
        if (i & (int64_t{1} << b))
          r |= int64_t{1} << (log2n - 1 - b);
      bitrev[i] = r;
    }

    // Stage `s` uses n/2 twiddles maximum; store per-stage tables compactly.
    twiddles.reserve(n - 1);
    for (int64_t len = 2; len <= n; len <<= 1) {
      for (int64_t k = 0; k < len / 2; ++k) {
        double angle = -2.0 * M_PI * static_cast<double>(k) /
                       static_cast<double>(len);
        twiddles.emplace_back(std::cos(angle), std::sin(angle));
      }
    }
  }

  /// In-place transform of a contiguous line (double precision).
  void run(std::complex<double> *line, bool inverse) const {
    for (int64_t i = 0; i < n; ++i) {
      int64_t j = bitrev[i];
      if (i < j)
        std::swap(line[i], line[j]);
    }
    const std::complex<double> *stageTwiddles = twiddles.data();
    for (int64_t len = 2; len <= n; len <<= 1) {
      int64_t half = len / 2;
      for (int64_t base = 0; base < n; base += len) {
        for (int64_t k = 0; k < half; ++k) {
          std::complex<double> w = stageTwiddles[k];
          if (inverse)
            w = std::conj(w);
          std::complex<double> even = line[base + k];
          std::complex<double> odd = line[base + k + half] * w;
          line[base + k] = even + odd;
          line[base + k + half] = even - odd;
        }
      }
      stageTwiddles += half;
    }
    if (inverse) {
      double scale = 1.0 / static_cast<double>(n);
      for (int64_t i = 0; i < n; ++i)
        line[i] *= scale;
    }
  }
};

template <typename Scalar>
void fft1d(const MemRefDescriptor<std::complex<Scalar>, 1> *in,
           MemRefDescriptor<std::complex<Scalar>, 1> *out, bool inverse) {
  int64_t n = in->sizes[0];
  FFTPlan plan(n);
  std::vector<std::complex<double>> line(n);
  for (int64_t i = 0; i < n; ++i)
    line[i] = std::complex<double>(in->at(i).real(), in->at(i).imag());
  plan.run(line.data(), inverse);
  for (int64_t i = 0; i < n; ++i)
    out->at(i) = std::complex<Scalar>(static_cast<Scalar>(line[i].real()),
                                      static_cast<Scalar>(line[i].imag()));
}

template <typename Scalar>
void fft2d(const MemRefDescriptor<std::complex<Scalar>, 2> *in,
           MemRefDescriptor<std::complex<Scalar>, 2> *out, int64_t dim,
           bool inverse) {
  int64_t rows = in->sizes[0];
  int64_t cols = in->sizes[1];
  int64_t numLines = (dim == 1) ? rows : cols;
  int64_t lineLen = (dim == 1) ? cols : rows;
  FFTPlan plan(lineLen);

  parallelFor(numLines, [&](int64_t begin, int64_t end) {
    std::vector<std::complex<double>> line(lineLen);
    for (int64_t l = begin; l < end; ++l) {
      for (int64_t k = 0; k < lineLen; ++k) {
        const std::complex<Scalar> &v =
            (dim == 1) ? in->at(l, k) : in->at(k, l);
        line[k] = std::complex<double>(v.real(), v.imag());
      }
      plan.run(line.data(), inverse);
      for (int64_t k = 0; k < lineLen; ++k) {
        std::complex<Scalar> v(static_cast<Scalar>(line[k].real()),
                               static_cast<Scalar>(line[k].imag()));
        if (dim == 1)
          out->at(l, k) = v;
        else
          out->at(k, l) = v;
      }
    }
  });
}

//===----------------------------------------------------------------------===//
// Stolt interpolation
//===----------------------------------------------------------------------===//

/// Normalized sinc, numpy convention: sinc(x) = sin(pi x) / (pi x).
double sinc(double x) {
  if (std::abs(x) < 1e-12)
    return 1.0;
  double px = M_PI * x;
  return std::sin(px) / px;
}

/// 8-tap windowed-sinc sample of a (double-precision) row at a fractional
/// position; out-of-range taps contribute zero.
inline std::complex<double> sampleWindowedSinc(
    const std::complex<double> *row, int64_t cols, double position) {
  int64_t idxInt = static_cast<int64_t>(std::floor(position));
  std::complex<double> acc(0.0, 0.0);
  for (int64_t k = -3; k <= 4; ++k) {
    int64_t idx = idxInt + k;
    if (idx < 0 || idx >= cols)
      continue;
    double dist = position - static_cast<double>(idx);
    double weight = sinc(dist) * (0.5 + 0.5 * std::cos(M_PI * dist / 4.0));
    acc += row[idx] * weight;
  }
  return acc;
}

/// Generic per-row windowed-sinc resampling (the sar.interp1d kernel).
template <typename Scalar>
void interp1d2d(const MemRefDescriptor<std::complex<Scalar>, 2> *data,
                const MemRefDescriptor<double, 2> *positions,
                MemRefDescriptor<std::complex<Scalar>, 2> *out) {
  int64_t rows = data->sizes[0];
  int64_t cols = data->sizes[1];
  parallelFor(rows, [&](int64_t begin, int64_t end) {
    std::vector<std::complex<double>> row(cols);
    for (int64_t i = begin; i < end; ++i) {
      for (int64_t j = 0; j < cols; ++j) {
        const std::complex<Scalar> &v = data->at(i, j);
        row[j] = std::complex<double>(v.real(), v.imag());
      }
      for (int64_t j = 0; j < cols; ++j) {
        std::complex<double> acc =
            sampleWindowedSinc(row.data(), cols, positions->at(i, j));
        out->at(i, j) =
            std::complex<Scalar>(static_cast<Scalar>(acc.real()),
                                 static_cast<Scalar>(acc.imag()));
      }
    }
  });
}

/// Windowed-sinc Stolt remapping. See the sar.stolt_interp op documentation
/// for the exact formula; this mirrors the omega-K numpy reference
/// implementation (taps k in [-3, 4], cosine window of half-width 4).
template <typename Scalar>
void stolt2d(const MemRefDescriptor<std::complex<Scalar>, 2> *data,
             const MemRefDescriptor<double, 1> *fa,
             const MemRefDescriptor<double, 1> *fr,
             MemRefDescriptor<std::complex<Scalar>, 2> *out, double c,
             double fc, double vr, double tShift) {
  int64_t rows = data->sizes[0];
  int64_t cols = data->sizes[1];
  double fStart = fr->at(0);
  double df = fr->at(1) - fr->at(0);

  // Phase ramp applied to the input spectrum before interpolation.
  std::vector<std::complex<double>> smooth(cols);
  for (int64_t j = 0; j < cols; ++j) {
    double phase = 2.0 * M_PI * fr->at(j) * tShift;
    smooth[j] = std::complex<double>(std::cos(phase), std::sin(phase));
  }

  parallelFor(rows, [&](int64_t begin, int64_t end) {
    std::vector<std::complex<double>> row(cols);
    for (int64_t i = begin; i < end; ++i) {
      for (int64_t j = 0; j < cols; ++j) {
        const std::complex<Scalar> &v = data->at(i, j);
        row[j] = std::complex<double>(v.real(), v.imag()) * smooth[j];
      }
      double faTerm = c * fa->at(i) / (2.0 * vr);
      for (int64_t j = 0; j < cols; ++j) {
        double frj = fr->at(j);
        double term = (frj + fc) * (frj + fc) + faTerm * faTerm;
        double frQuery = std::sqrt(std::max(term, 1e-10)) - fc;
        double idxFloat = (frQuery - fStart) / df;

        std::complex<double> acc =
            sampleWindowedSinc(row.data(), cols, idxFloat);
        double phase = -2.0 * M_PI * frQuery * tShift;
        acc *= std::complex<double>(std::cos(phase), std::sin(phase));
        out->at(i, j) =
            std::complex<Scalar>(static_cast<Scalar>(acc.real()),
                                 static_cast<Scalar>(acc.imag()));
      }
    }
  });
}

} // namespace

//===----------------------------------------------------------------------===//
// Exported C interface
//===----------------------------------------------------------------------===//

extern "C" {

void _mlir_ciface_sar_rt_fft_1d_c64(MemRefDescriptor<std::complex<float>, 1> *in,
                                    MemRefDescriptor<std::complex<float>, 1> *out,
                                    int64_t /*dim*/, bool inverse) {
  fft1d<float>(in, out, inverse);
}

void _mlir_ciface_sar_rt_fft_1d_c128(
    MemRefDescriptor<std::complex<double>, 1> *in,
    MemRefDescriptor<std::complex<double>, 1> *out, int64_t /*dim*/,
    bool inverse) {
  fft1d<double>(in, out, inverse);
}

void _mlir_ciface_sar_rt_fft_2d_c64(MemRefDescriptor<std::complex<float>, 2> *in,
                                    MemRefDescriptor<std::complex<float>, 2> *out,
                                    int64_t dim, bool inverse) {
  fft2d<float>(in, out, dim, inverse);
}

void _mlir_ciface_sar_rt_fft_2d_c128(
    MemRefDescriptor<std::complex<double>, 2> *in,
    MemRefDescriptor<std::complex<double>, 2> *out, int64_t dim, bool inverse) {
  fft2d<double>(in, out, dim, inverse);
}

void _mlir_ciface_sar_rt_interp1d_2d_c64(
    MemRefDescriptor<std::complex<float>, 2> *data,
    MemRefDescriptor<double, 2> *positions,
    MemRefDescriptor<std::complex<float>, 2> *out) {
  interp1d2d<float>(data, positions, out);
}

void _mlir_ciface_sar_rt_interp1d_2d_c128(
    MemRefDescriptor<std::complex<double>, 2> *data,
    MemRefDescriptor<double, 2> *positions,
    MemRefDescriptor<std::complex<double>, 2> *out) {
  interp1d2d<double>(data, positions, out);
}

void _mlir_ciface_sar_rt_stolt_2d_c64(
    MemRefDescriptor<std::complex<float>, 2> *data,
    MemRefDescriptor<double, 1> *fa, MemRefDescriptor<double, 1> *fr,
    MemRefDescriptor<std::complex<float>, 2> *out, double c, double fc,
    double vr, double tShift) {
  stolt2d<float>(data, fa, fr, out, c, fc, vr, tShift);
}

void _mlir_ciface_sar_rt_stolt_2d_c128(
    MemRefDescriptor<std::complex<double>, 2> *data,
    MemRefDescriptor<double, 1> *fa, MemRefDescriptor<double, 1> *fr,
    MemRefDescriptor<std::complex<double>, 2> *out, double c, double fc,
    double vr, double tShift) {
  stolt2d<double>(data, fa, fr, out, c, fc, vr, tShift);
}

} // extern "C"

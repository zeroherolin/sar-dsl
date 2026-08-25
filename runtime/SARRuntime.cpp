//===- SARRuntime.cpp - SAR runtime library -------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
// C-ABI kernels backing the sar.fft / sar.ifft / sar.interp1d operations
// on CPU targets. Symbols follow the MLIR C interface convention
// (_mlir_ciface_<name>): memrefs are passed as pointers to strided
// descriptors, scalars by value.
//
// Numerical conventions match numpy: the forward DFT is unscaled, the
// inverse DFT is scaled by 1/N. Power-of-two sizes run the radix-2 kernel
// directly; other sizes go through Bluestein's chirp-z reduction. All
// transforms are computed in double precision regardless of the storage
// precision.
//
// The library also backs the compiled kernel's buffer allocation, through
// the `_mlir_memref_to_llvm_*` hooks the CPU pipeline lowers `memref.alloc`
// onto (see the plane pool below).
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include "sar/Runtime/RuntimeEnums.h"

namespace {

//===----------------------------------------------------------------------===//
// Plane pool
//===----------------------------------------------------------------------===//
//
// Large intermediate planes are reused by size to avoid repeated mmap and
// page-fault costs. Retention is bounded by the peak live allocation;
// `SAR_RT_POOL_MAX_BYTES=0` disables pooling. Small blocks remain with libc.

class PlanePool {
public:
  static constexpr size_t kPoolMinBytes = 1u << 20;

  static PlanePool &instance() {
    static PlanePool pool;
    return pool;
  }

  void *allocate(size_t size, size_t alignment) {
    if (size == 0)
      return nullptr;
    alignment = std::max(alignment, sizeof(void *));
    if ((alignment & (alignment - 1)) != 0) {
      std::fprintf(stderr, "sar_runtime: invalid allocation alignment %zu\n",
                   alignment);
      std::abort();
    }
    bool pooled = enabled && size >= kPoolMinBytes;
    if (!pooled)
      return allocateRaw(size, alignment);

    std::lock_guard<std::mutex> guard(mutex);
    for (size_t i = 0; i < cache.size(); ++i) {
      if (cache[i].size != size || cache[i].alignment < alignment)
        continue;
      void *ptr = cache[i].ptr;
      cachedBytes -= cache[i].size;
      cache[i] = cache.back();
      cache.pop_back();
      liveBytes += size;
      peakLiveBytes = std::max(peakLiveBytes, liveBytes);
      return ptr;
    }

    evictCachedToFit(size);
    void *ptr = allocateRaw(size, alignment);
    blocks.push_back({ptr, size, alignment});
    liveBytes += size;
    peakLiveBytes = std::max(peakLiveBytes, liveBytes);
    return ptr;
  }

  void release(void *ptr) {
    if (!ptr)
      return;
    if (enabled) {
      std::lock_guard<std::mutex> guard(mutex);
      // Blocks libc served directly are not listed here and fall through.
      for (size_t i = 0; i < blocks.size(); ++i) {
        if (blocks[i].ptr != ptr)
          continue;
        Block block = blocks[i];
        liveBytes -= block.size;
        size_t bound = limit ? limit : peakLiveBytes;
        if (liveBytes + cachedBytes + block.size <= bound) {
          cache.push_back(block);
          cachedBytes += block.size;
          return;
        }
        blocks[i] = blocks.back();
        blocks.pop_back();
        free(ptr);
        return;
      }
    }
    free(ptr);
  }

private:
  struct Block {
    void *ptr;
    size_t size;
    size_t alignment;
  };

  static void *allocateRaw(size_t size, size_t alignment) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) == 0)
      return ptr;
    std::fprintf(stderr,
                 "sar_runtime: failed to allocate %zu bytes "
                 "(alignment %zu)\n",
                 size, alignment);
    std::abort();
  }

  void evictCachedToFit(size_t request) {
    size_t requiredLive = liveBytes + request;
    size_t bound = limit ? std::max(limit, requiredLive)
                         : std::max(peakLiveBytes, requiredLive);
    while (!cache.empty() && liveBytes + cachedBytes + request > bound) {
      Block victim = cache.back();
      cache.pop_back();
      cachedBytes -= victim.size;
      for (size_t i = 0; i < blocks.size(); ++i) {
        if (blocks[i].ptr != victim.ptr)
          continue;
        blocks[i] = blocks.back();
        blocks.pop_back();
        break;
      }
      free(victim.ptr);
    }
  }

  PlanePool() {
    if (const char *env = std::getenv("SAR_RT_POOL_MAX_BYTES")) {
      char *end = nullptr;
      long long value = std::strtoll(env, &end, 10);
      if (end != env && *end == '\0' && value >= 0) {
        limit = static_cast<size_t>(value);
        enabled = value != 0;
      }
    }
  }

  std::mutex mutex;
  std::vector<Block> blocks; // every live block the pool handed out
  std::vector<Block> cache;  // freed blocks kept mapped for reuse
  size_t liveBytes = 0;
  size_t cachedBytes = 0;
  size_t peakLiveBytes = 0;
  size_t limit = 0;
  bool enabled = true;
};

//===----------------------------------------------------------------------===//
// MLIR memref descriptors
//===----------------------------------------------------------------------===//

template <typename T, int Rank> struct MemRefDescriptor {
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

static unsigned availableWorkerCount() {
#if defined(__linux__)
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
    return std::max(1, CPU_COUNT(&affinity));
#endif
  return std::max(1u, std::thread::hardware_concurrency());
}

/// Worker count for runtime parallelism. Explicit requests are capped by the
/// process affinity, so a container cannot accidentally create host-wide
/// thread counts.
unsigned runtimeWorkerCount() {
  static const unsigned cached = [] {
    unsigned available = availableWorkerCount();
    for (const char *var : {"SAR_RT_NUM_THREADS", "OMP_NUM_THREADS"}) {
      if (const char *env = std::getenv(var)) {
        char *end = nullptr;
        long value = std::strtol(env, &end, 10);
        if (end != env && *end == '\0' && value > 0)
          return static_cast<unsigned>(std::min<unsigned long>(
              static_cast<unsigned long>(value), available));
      }
    }
    return std::min(available, 32u);
  }();
  return cached;
}

class RuntimeThreadPool {
public:
  RuntimeThreadPool() : workerCount(runtimeWorkerCount()) {
    for (unsigned id = 1; id < workerCount; ++id)
      threads.emplace_back([this, id] { workerLoop(id); });
  }

  ~RuntimeThreadPool() {
    {
      std::lock_guard<std::mutex> guard(mutex);
      stopping = true;
      ++generation;
    }
    ready.notify_all();
    for (std::thread &thread : threads)
      thread.join();
  }

  void run(int64_t total,
           const std::function<void(int64_t, int64_t)> &function) {
    unsigned active = static_cast<unsigned>(
        std::min<int64_t>(total, static_cast<int64_t>(workerCount)));
    if (active <= 1) {
      function(0, total);
      return;
    }

    std::lock_guard<std::mutex> runGuard(runMutex);
    {
      std::lock_guard<std::mutex> guard(mutex);
      task = function;
      taskTotal = total;
      activeWorkers = active;
      completed = 0;
      ++generation;
    }
    ready.notify_all();
    runChunk(0, function, total, active);

    std::unique_lock<std::mutex> lock(mutex);
    done.wait(lock, [&] { return completed == threads.size(); });
  }

private:
  static void runChunk(unsigned id,
                       const std::function<void(int64_t, int64_t)> &function,
                       int64_t total, unsigned active) {
    if (id >= active)
      return;
    int64_t chunk = (total + active - 1) / active;
    int64_t begin = static_cast<int64_t>(id) * chunk;
    int64_t end = std::min<int64_t>(begin + chunk, total);
    if (begin < end)
      function(begin, end);
  }

  void workerLoop(unsigned id) {
    uint64_t observed = 0;
    while (true) {
      std::function<void(int64_t, int64_t)> function;
      int64_t total = 0;
      unsigned active = 0;
      {
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock, [&] { return stopping || generation != observed; });
        if (stopping)
          return;
        observed = generation;
        function = task;
        total = taskTotal;
        active = activeWorkers;
      }
      runChunk(id, function, total, active);
      {
        std::lock_guard<std::mutex> guard(mutex);
        ++completed;
        if (completed == threads.size())
          done.notify_one();
      }
    }
  }

  unsigned workerCount;
  std::vector<std::thread> threads;
  std::mutex runMutex;
  std::mutex mutex;
  std::condition_variable ready;
  std::condition_variable done;
  std::function<void(int64_t, int64_t)> task;
  int64_t taskTotal = 0;
  unsigned activeWorkers = 0;
  size_t completed = 0;
  uint64_t generation = 0;
  bool stopping = false;
};

/// Runs fn(begin, end) over [0, total) on a process-wide reusable pool.
void parallelFor(int64_t total,
                 const std::function<void(int64_t, int64_t)> &fn) {
  static RuntimeThreadPool pool;
  pool.run(total, fn);
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
        double angle =
            -2.0 * M_PI * static_cast<double>(k) / static_cast<double>(len);
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

//===----------------------------------------------------------------------===//
// Arbitrary-length lines: Bluestein's chirp-z reduction
//===----------------------------------------------------------------------===//

/// DFT plan for any line length >= 2. Powers of two run the radix-2 kernel
/// directly; other sizes convolve with a chirp at the next power of two
/// >= 2n - 1 (Bluestein), reusing the radix-2 kernel for the convolution.
class LinePlan {
public:
  explicit LinePlan(int64_t size) : n(size) {
    if ((n & (n - 1)) == 0) {
      pow2 = std::make_unique<FFTPlan>(n);
      return;
    }
    m = 2;
    while (m < 2 * n - 1)
      m <<= 1;
    pow2 = std::make_unique<FFTPlan>(m);

    // chirp[k] = exp(-j pi k^2 / n). exp has period 2 pi i, so k^2 is
    // reduced modulo 2n before entering the angle; the int64 product k*k
    // cannot overflow for any realistic transform (k < n << 2^31).
    chirp.resize(n);
    for (int64_t k = 0; k < n; ++k) {
      int64_t sq = ((k % (2 * n)) * k) % (2 * n);
      double angle = -M_PI * static_cast<double>(sq) / static_cast<double>(n);
      chirp[k] = std::complex<double>(std::cos(angle), std::sin(angle));
    }

    std::vector<std::complex<double>> b(m, std::complex<double>(0.0, 0.0));
    b[0] = std::conj(chirp[0]);
    for (int64_t k = 1; k < n; ++k)
      b[k] = b[m - k] = std::conj(chirp[k]);
    pow2->run(b.data(), /*inverse=*/false);
    bSpec = std::move(b);
  }

  void run(std::complex<double> *line, bool inverse) const {
    if (chirp.empty()) {
      pow2->run(line, inverse);
      return;
    }
    if (inverse) {
      // ifft(x) = conj(fft(conj(x))) / n
      for (int64_t k = 0; k < n; ++k)
        line[k] = std::conj(line[k]);
      forward(line);
      double scale = 1.0 / static_cast<double>(n);
      for (int64_t k = 0; k < n; ++k)
        line[k] = std::conj(line[k]) * scale;
    } else {
      forward(line);
    }
  }

private:
  void forward(std::complex<double> *line) const {
    std::vector<std::complex<double>> a(m, std::complex<double>(0.0, 0.0));
    for (int64_t k = 0; k < n; ++k)
      a[k] = line[k] * chirp[k];
    pow2->run(a.data(), /*inverse=*/false);
    for (int64_t i = 0; i < m; ++i)
      a[i] *= bSpec[i];
    pow2->run(a.data(), /*inverse=*/true);
    for (int64_t k = 0; k < n; ++k)
      line[k] = a[k] * chirp[k];
  }

  int64_t n;
  int64_t m = 0;
  std::unique_ptr<FFTPlan> pow2;
  std::vector<std::complex<double>> chirp;
  std::vector<std::complex<double>> bSpec;
};

template <typename Scalar>
void fft1d(const MemRefDescriptor<std::complex<Scalar>, 1> *in,
           MemRefDescriptor<std::complex<Scalar>, 1> *out, bool inverse) {
  int64_t n = in->sizes[0];
  LinePlan plan(n);
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
  LinePlan plan(lineLen);

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
// Windowed-sinc interpolation
//===----------------------------------------------------------------------===//

/// Normalized sinc, numpy convention: sinc(x) = sin(pi x) / (pi x).
double sinc(double x) {
  if (std::abs(x) < 1e-12)
    return 1.0;
  double px = M_PI * x;
  return std::sin(px) / px;
}

// Kernel / window / boundary encodings shared with the lowering pass.
using sar_rt::kCubic;
using sar_rt::kEdge;
using sar_rt::kHamming;
using sar_rt::kHann;
using sar_rt::kKaiser;
using sar_rt::kLinear;
using sar_rt::kNearest;
using sar_rt::kRect;
using sar_rt::kReflect;
using sar_rt::kSinc;
using sar_rt::kZero;

/// Modified Bessel function of the first kind, order zero (power series).
/// The term test exits well before the bound over the beta range the
/// verifier admits, so the two backends agree to f64 rounding even though
/// the HLS lowering runs a shorter, straight-line series instead.
double besselI0(double x) {
  double sum = 1.0, term = 1.0;
  double halfX = x / 2.0;
  for (int m = 1; m <= 40; ++m) {
    term *= (halfX / m) * (halfX / m);
    sum += term;
    if (term < 1e-16 * sum)
      break;
  }
  return sum;
}

/// Applies the boundary policy to an out-of-range index. Returns a valid
/// in-bounds index or -1 for kZero when the index is unmappable.
inline int64_t applyBoundary(int64_t idx, int64_t cols, int64_t boundary) {
  if (idx >= 0 && idx < cols)
    return idx;
  if (boundary == kZero)
    return -1;
  if (boundary == kEdge)
    return std::max<int64_t>(0, std::min<int64_t>(cols - 1, idx));
  if (cols == 1)
    return 0;
  int64_t period = 2 * cols;
  idx %= period;
  if (idx < 0)
    idx += period;
  return idx < cols ? idx : period - idx - 1;
}

/// Window taper evaluated at t = d / (taps/2), |t| <= 1 on the support.
double interpWindow(int64_t window, double t, double beta, double invI0Beta) {
  switch (window) {
  case kRect:
    return 1.0;
  case kHann:
    return 0.5 + 0.5 * std::cos(M_PI * t);
  case kHamming:
    return 0.54 + 0.46 * std::cos(M_PI * t);
  case kKaiser: {
    double s = 1.0 - t * t;
    return besselI0(beta * std::sqrt(s > 0.0 ? s : 0.0)) * invI0Beta;
  }
  }
  return 1.0;
}

/// Kernel-based sample of a (double-precision) row at a fractional
/// position. Boundary policy resolves out-of-range taps: zero, edge-clamp,
/// or reflect.
inline std::complex<double> sampleInterp(const std::complex<double> *row,
                                         int64_t cols, double position,
                                         int64_t kernel, int64_t taps,
                                         int64_t window, double beta,
                                         double invI0Beta, int64_t boundary) {
  constexpr double safeIndexLimit = 0x1p62;
  if (!std::isfinite(position) || position <= -safeIndexLimit ||
      position >= safeIndexLimit)
    return {0.0, 0.0};
  if (kernel == kNearest) {
    int64_t idx = static_cast<int64_t>(std::floor(position + 0.5));
    idx = applyBoundary(idx, cols, boundary);
    if (idx < 0)
      return {0.0, 0.0};
    return row[idx];
  }

  int64_t idxInt = static_cast<int64_t>(std::floor(position));
  double frac = position - static_cast<double>(idxInt);
  std::complex<double> acc(0.0, 0.0);

  if (kernel == kLinear) {
    for (int64_t k = 0; k <= 1; ++k) {
      int64_t idx = applyBoundary(idxInt + k, cols, boundary);
      if (idx < 0)
        continue;
      acc += row[idx] * (k == 0 ? 1.0 - frac : frac);
    }
    return acc;
  }

  if (kernel == kCubic) {
    // Keys cubic convolution, a = -0.5.
    double s = frac, s2 = s * s, s3 = s2 * s;
    double w[4] = {-0.5 * s3 + s2 - 0.5 * s, 1.5 * s3 - 2.5 * s2 + 1.0,
                   -1.5 * s3 + 2.0 * s2 + 0.5 * s, 0.5 * s3 - 0.5 * s2};
    for (int64_t k = -1; k <= 2; ++k) {
      int64_t idx = applyBoundary(idxInt + k, cols, boundary);
      if (idx < 0)
        continue;
      acc += row[idx] * w[k + 1];
    }
    return acc;
  }

  // Windowed sinc over `taps` taps centred on the position.
  int64_t half = taps / 2;
  for (int64_t k = 1 - half; k <= half; ++k) {
    int64_t idx = applyBoundary(idxInt + k, cols, boundary);
    if (idx < 0)
      continue;
    double dist = position - static_cast<double>(idxInt + k);
    double weight =
        sinc(dist) *
        interpWindow(window, dist / static_cast<double>(half), beta, invI0Beta);
    acc += row[idx] * weight;
  }
  return acc;
}

/// Generic per-row resampling (the sar.interp1d kernel).
template <typename Scalar>
void interp1d2d(const MemRefDescriptor<std::complex<Scalar>, 2> *data,
                const MemRefDescriptor<double, 2> *positions,
                MemRefDescriptor<std::complex<Scalar>, 2> *out, int64_t kernel,
                int64_t taps, int64_t window, double beta, int64_t boundary) {
  int64_t rows = data->sizes[0];
  int64_t cols = data->sizes[1];
  double invI0Beta = window == kKaiser ? 1.0 / besselI0(beta) : 1.0;
  parallelFor(rows, [&](int64_t begin, int64_t end) {
    std::vector<std::complex<double>> row(cols);
    for (int64_t i = begin; i < end; ++i) {
      for (int64_t j = 0; j < cols; ++j) {
        const std::complex<Scalar> &v = data->at(i, j);
        row[j] = std::complex<double>(v.real(), v.imag());
      }
      for (int64_t j = 0; j < cols; ++j) {
        std::complex<double> acc =
            sampleInterp(row.data(), cols, positions->at(i, j), kernel, taps,
                         window, beta, invI0Beta, boundary);
        out->at(i, j) = std::complex<Scalar>(static_cast<Scalar>(acc.real()),
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

//===----------------------------------------------------------------------===//
// Buffer allocation
//===----------------------------------------------------------------------===//
//
// `finalize-memref-to-llvm` is configured with `use-generic-functions`, so
// a kernel's `memref.alloc` calls these instead of malloc/aligned_alloc,
// and its buffers come from the pool above.

/// ABI-boundary shape check. These entry points form a C ABI that foreign
/// callers can reach directly, so a mismatch aborts with a diagnostic in
/// every build -- an assert would vanish under NDEBUG and turn the mismatch
/// into an out-of-bounds write.
static void requireRuntime(bool ok, const char *what) {
  if (!ok) {
    std::fprintf(stderr, "sar_runtime: %s\n", what);
    std::abort();
  }
}

void *_mlir_memref_to_llvm_alloc(int64_t size) {
  requireRuntime(size >= 0, "allocation size must be non-negative");
  return PlanePool::instance().allocate(static_cast<size_t>(size),
                                        sizeof(void *));
}

void *_mlir_memref_to_llvm_aligned_alloc(int64_t alignment, int64_t size) {
  requireRuntime(size >= 0, "allocation size must be non-negative");
  requireRuntime(alignment > 0 && (alignment & (alignment - 1)) == 0 &&
                     alignment % static_cast<int64_t>(sizeof(void *)) == 0,
                 "allocation alignment must be a pointer-sized power of two");
  return PlanePool::instance().allocate(static_cast<size_t>(size),
                                        static_cast<size_t>(alignment));
}

void _mlir_memref_to_llvm_free(void *ptr) {
  PlanePool::instance().release(ptr);
}

void _mlir_ciface_sar_rt_fft_1d_c64(
    MemRefDescriptor<std::complex<float>, 1> *in,
    MemRefDescriptor<std::complex<float>, 1> *out, int64_t dim, bool inverse) {
  requireRuntime(dim == 0, "fft_1d_c64: dim must be 0");
  requireRuntime(in->sizes[0] >= 2, "fft_1d_c64: extent must be at least 2");
  requireRuntime(in->sizes[0] == out->sizes[0],
                 "fft_1d_c64: output shape must match input shape");
  fft1d<float>(in, out, inverse);
}

void _mlir_ciface_sar_rt_fft_1d_c128(
    MemRefDescriptor<std::complex<double>, 1> *in,
    MemRefDescriptor<std::complex<double>, 1> *out, int64_t dim, bool inverse) {
  requireRuntime(dim == 0, "fft_1d_c128: dim must be 0");
  requireRuntime(in->sizes[0] >= 2, "fft_1d_c128: extent must be at least 2");
  requireRuntime(in->sizes[0] == out->sizes[0],
                 "fft_1d_c128: output shape must match input shape");
  fft1d<double>(in, out, inverse);
}

void _mlir_ciface_sar_rt_fft_2d_c64(
    MemRefDescriptor<std::complex<float>, 2> *in,
    MemRefDescriptor<std::complex<float>, 2> *out, int64_t dim, bool inverse) {
  requireRuntime(dim == 0 || dim == 1, "fft_2d_c64: dim must be 0 or 1");
  requireRuntime(in->sizes[0] > 0 && in->sizes[1] > 0 && in->sizes[dim] >= 2,
                 "fft_2d_c64: extents must be positive and transform "
                 "extent at least 2");
  requireRuntime(in->sizes[0] == out->sizes[0] && in->sizes[1] == out->sizes[1],
                 "fft_2d_c64: output shape must match input shape");
  fft2d<float>(in, out, dim, inverse);
}

void _mlir_ciface_sar_rt_fft_2d_c128(
    MemRefDescriptor<std::complex<double>, 2> *in,
    MemRefDescriptor<std::complex<double>, 2> *out, int64_t dim, bool inverse) {
  requireRuntime(dim == 0 || dim == 1, "fft_2d_c128: dim must be 0 or 1");
  requireRuntime(in->sizes[0] > 0 && in->sizes[1] > 0 && in->sizes[dim] >= 2,
                 "fft_2d_c128: extents must be positive and transform "
                 "extent at least 2");
  requireRuntime(in->sizes[0] == out->sizes[0] && in->sizes[1] == out->sizes[1],
                 "fft_2d_c128: output shape must match input shape");
  fft2d<double>(in, out, dim, inverse);
}

static void requireInterpArgs(int64_t rows, int64_t cols, int64_t kernel,
                              int64_t taps, int64_t window, double beta,
                              int64_t boundary, const char *what) {
  requireRuntime(rows > 0 && cols > 0, what);
  requireRuntime(kernel >= kNearest && kernel <= kSinc,
                 "interp1d: unknown kernel");
  requireRuntime(window >= kRect && window <= kKaiser,
                 "interp1d: unknown window");
  requireRuntime(boundary >= kZero && boundary <= kReflect,
                 "interp1d: unknown boundary");
  requireRuntime(kernel != kSinc || (taps >= 4 && taps <= 32 && taps % 2 == 0),
                 "interp1d: sinc taps must be even and in [4, 32]");
  requireRuntime(window != kKaiser ||
                     (std::isfinite(beta) && beta > 0.0 && beta <= 12.0),
                 "interp1d: Kaiser beta must be finite and in (0, 12]");
}

void _mlir_ciface_sar_rt_interp1d_2d_c64(
    MemRefDescriptor<std::complex<float>, 2> *data,
    MemRefDescriptor<double, 2> *positions,
    MemRefDescriptor<std::complex<float>, 2> *out, int64_t kernel, int64_t taps,
    int64_t window, double beta, int64_t boundary) {
  requireInterpArgs(data->sizes[0], data->sizes[1], kernel, taps, window, beta,
                    boundary, "interp1d_2d_c64: extents must be positive");
  requireRuntime(data->sizes[0] == positions->sizes[0] &&
                     data->sizes[1] == positions->sizes[1],
                 "interp1d_2d_c64: positions shape must match data");
  requireRuntime(data->sizes[0] == out->sizes[0] &&
                     data->sizes[1] == out->sizes[1],
                 "interp1d_2d_c64: output shape must match data");
  interp1d2d<float>(data, positions, out, kernel, taps, window, beta, boundary);
}

void _mlir_ciface_sar_rt_interp1d_2d_c128(
    MemRefDescriptor<std::complex<double>, 2> *data,
    MemRefDescriptor<double, 2> *positions,
    MemRefDescriptor<std::complex<double>, 2> *out, int64_t kernel,
    int64_t taps, int64_t window, double beta, int64_t boundary) {
  requireInterpArgs(data->sizes[0], data->sizes[1], kernel, taps, window, beta,
                    boundary, "interp1d_2d_c128: extents must be positive");
  requireRuntime(data->sizes[0] == positions->sizes[0] &&
                     data->sizes[1] == positions->sizes[1],
                 "interp1d_2d_c128: positions shape must match data");
  requireRuntime(data->sizes[0] == out->sizes[0] &&
                     data->sizes[1] == out->sizes[1],
                 "interp1d_2d_c128: output shape must match data");
  interp1d2d<double>(data, positions, out, kernel, taps, window, beta,
                     boundary);
}

} // extern "C"

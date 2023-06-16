// Flat, cache-friendly tree-ensemble evaluator for LightGBM binary models.
//
// Layout: every tree is stored in depth-first order inside ONE contiguous
// array of 16-byte nodes (four nodes per cache line). The left child of an
// internal node is always the next node (i + 1), so only the right child index
// is stored. Leaves reuse the `value` field for the leaf output.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

constexpr uint8_t kMissingMask = 0x3;     // bits 0-1: 0 none, 1 zero, 2 nan
constexpr uint8_t kMissingZero = 1;
constexpr uint8_t kMissingNaN = 2;
constexpr uint8_t kDefaultLeft = 0x4;     // bit 2
constexpr uint8_t kLeaf = 0x8;            // bit 3
constexpr double kZeroThreshold = 1e-35;  // same constant LightGBM uses

struct alignas(16) Node {
  double value;     // split threshold (internal) or leaf output (leaf)
  uint32_t right;   // global index of right child (internal nodes only)
  uint16_t feature; // feature index (internal nodes only)
  uint8_t flags;
  uint8_t pad;
};
static_assert(sizeof(Node) == 16, "Node must stay 16 bytes");

inline bool go_left(double v, const Node& n) {
  const uint8_t mt = n.flags & kMissingMask;
  // LightGBM: NaN is mapped to 0 unless the split was trained with NaN-as-missing.
  if (std::isnan(v) && mt != kMissingNaN) v = 0.0;
  if ((mt == kMissingZero && v >= -kZeroThreshold && v <= kZeroThreshold) ||
      (mt == kMissingNaN && std::isnan(v))) {
    return (n.flags & kDefaultLeft) != 0;
  }
  return v <= n.value;
}

}  // namespace

class FlatEnsemble {
 public:
  FlatEnsemble(py::array_t<double, py::array::c_style | py::array::forcecast> values,
               py::array_t<uint32_t, py::array::c_style | py::array::forcecast> right,
               py::array_t<uint16_t, py::array::c_style | py::array::forcecast> feature,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> flags,
               py::array_t<int64_t, py::array::c_style | py::array::forcecast> roots,
               int n_features, double sigmoid)
      : n_features_(n_features), sigmoid_(sigmoid) {
    const size_t n = static_cast<size_t>(values.size());
    if (right.size() != values.size() || feature.size() != values.size() ||
        flags.size() != values.size())
      throw std::invalid_argument("node arrays must have equal length");
    if (n_features <= 0) throw std::invalid_argument("n_features must be positive");
    nodes_.resize(n);
    for (size_t i = 0; i < n; ++i) {
      Node& nd = nodes_[i];
      nd.value = values.data()[i];
      nd.right = right.data()[i];
      nd.feature = feature.data()[i];
      nd.flags = flags.data()[i];
      nd.pad = 0;
    }
    const int64_t* r = roots.data();
    roots_.assign(r, r + roots.size());
    validate();
  }

  size_t n_nodes() const { return nodes_.size(); }
  size_t n_trees() const { return roots_.size(); }
  int n_features() const { return n_features_; }

  double predict_one(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                     bool raw) const {
    if (x.ndim() != 1 || x.shape(0) != n_features_)
      throw std::invalid_argument("expected a 1-D row with " + std::to_string(n_features_) +
                                  " features");
    const double s = raw_row(x.data());
    return raw ? s : transform(s);
  }

  py::array_t<double> predict_batch(
      py::array_t<double, py::array::c_style | py::array::forcecast> X, bool raw,
      int n_threads) const {
    if (X.ndim() != 2 || X.shape(1) != n_features_)
      throw std::invalid_argument("expected a 2-D matrix with " + std::to_string(n_features_) +
                                  " columns");
    const size_t rows = static_cast<size_t>(X.shape(0));
    py::array_t<double> out(static_cast<py::ssize_t>(rows));
    const double* in = X.data();
    double* o = out.mutable_data();
    if (n_threads <= 0) n_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    // Threads only pay off once there is enough work to amortise their start-up cost.
    constexpr size_t kMinRowsPerThread = 256;
    const size_t max_useful = std::max<size_t>(1, rows / kMinRowsPerThread);
    const size_t workers = std::min<size_t>(static_cast<size_t>(n_threads), max_useful);
    {
      py::gil_scoped_release release;
      if (workers <= 1) {
        run_range(in, o, 0, rows, raw);
      } else {
        std::vector<std::thread> pool;
        pool.reserve(workers - 1);
        const size_t chunk = (rows + workers - 1) / workers;
        for (size_t w = 1; w < workers; ++w) {
          const size_t b = w * chunk, e = std::min(rows, b + chunk);
          if (b >= e) break;
          pool.emplace_back([=] { run_range(in, o, b, e, raw); });
        }
        run_range(in, o, 0, std::min(rows, chunk), raw);
        for (auto& t : pool) t.join();
      }
    }
    return out;
  }

 private:
  void validate() const {
    const int64_t n = static_cast<int64_t>(nodes_.size());
    for (int64_t i = 0; i < n; ++i) {
      const Node& nd = nodes_[i];
      if (nd.flags & kLeaf) continue;
      // Children must exist and lie strictly after the parent: guarantees
      // termination and in-bounds reads even for a corrupted model file.
      if (i + 1 >= n || nd.right <= i || nd.right >= n)
        throw std::invalid_argument("corrupt model: bad child index at node " + std::to_string(i));
      if (nd.feature >= n_features_)
        throw std::invalid_argument("corrupt model: feature index out of range");
    }
    for (int64_t r : roots_)
      if (r < 0 || r >= n) throw std::invalid_argument("corrupt model: bad tree root");
    if (!nodes_.empty() && !(nodes_.back().flags & kLeaf))
      throw std::invalid_argument("corrupt model: last node must be a leaf");
  }

  inline double raw_row(const double* x) const {
    const Node* base = nodes_.data();
    double sum = 0.0;
    for (int64_t root : roots_) {
      const Node* n = base + root;
      while (!(n->flags & kLeaf)) n = go_left(x[n->feature], *n) ? n + 1 : base + n->right;
      sum += n->value;
    }
    return sum;
  }

  inline double transform(double raw) const { return 1.0 / (1.0 + std::exp(-sigmoid_ * raw)); }

  // Batch path: walk kBlock rows through each tree in lock-step. The traversals are
  // independent dependency chains, so the CPU overlaps their cache misses and
  // branch resolution. Per-row summation order is unchanged, so results are
  // bit-identical to the single-row path.
  static constexpr size_t kBlock = 8;

  void run_range(const double* in, double* out, size_t b, size_t e, bool raw) const {
    const size_t nf = static_cast<size_t>(n_features_);
    const Node* base = nodes_.data();
    size_t i = b;
    for (; i + kBlock <= e; i += kBlock) {
      double acc[kBlock] = {};
      const double* rows[kBlock];
      for (size_t j = 0; j < kBlock; ++j) rows[j] = in + (i + j) * nf;
      for (int64_t root : roots_) {
        const Node* cur[kBlock];
        for (size_t j = 0; j < kBlock; ++j) cur[j] = base + root;
        bool active = true;
        while (active) {
          active = false;
          for (size_t j = 0; j < kBlock; ++j) {
            const Node* n = cur[j];
            if (!(n->flags & kLeaf)) {
              cur[j] = go_left(rows[j][n->feature], *n) ? n + 1 : base + n->right;
              active = true;
            }
          }
        }
        for (size_t j = 0; j < kBlock; ++j) acc[j] += cur[j]->value;
      }
      for (size_t j = 0; j < kBlock; ++j) out[i + j] = raw ? acc[j] : transform(acc[j]);
    }
    for (; i < e; ++i) {
      const double s = raw_row(in + i * nf);
      out[i] = raw ? s : transform(s);
    }
  }

  std::vector<Node> nodes_;
  std::vector<int64_t> roots_;
  int n_features_;
  double sigmoid_;
};

PYBIND11_MODULE(_treeeval, m) {
  m.doc() = "Flat tree-ensemble evaluator (C++17)";
  py::class_<FlatEnsemble>(m, "FlatEnsemble")
      .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<uint32_t, py::array::c_style | py::array::forcecast>,
                    py::array_t<uint16_t, py::array::c_style | py::array::forcecast>,
                    py::array_t<uint8_t, py::array::c_style | py::array::forcecast>,
                    py::array_t<int64_t, py::array::c_style | py::array::forcecast>, int, double>(),
           py::arg("values"), py::arg("right"), py::arg("feature"), py::arg("flags"),
           py::arg("roots"), py::arg("n_features"), py::arg("sigmoid") = 1.0)
      .def_property_readonly("n_nodes", &FlatEnsemble::n_nodes)
      .def_property_readonly("n_trees", &FlatEnsemble::n_trees)
      .def_property_readonly("n_features", &FlatEnsemble::n_features)
      .def("predict_one", &FlatEnsemble::predict_one, py::arg("x"), py::arg("raw") = false)
      .def("predict_batch", &FlatEnsemble::predict_batch, py::arg("X"), py::arg("raw") = false,
           py::arg("n_threads") = 1);
}

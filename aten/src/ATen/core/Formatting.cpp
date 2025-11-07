#include <ATen/core/Formatting.h>
#include <c10/util/irange.h>
<<<<<<< HEAD

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <tuple>

namespace c10 {
std::ostream& operator<<(std::ostream & out, Backend b) {
  return out << toString(b);
}

std::ostream& operator<<(std::ostream & out, const Scalar& s) {
=======
#include <fmt/compile.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>

namespace c10 {
std::ostream& operator<<(std::ostream& out, Backend b) {
  return out << toString(b);
}

std::ostream& operator<<(std::ostream& out, const Scalar& s) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  if (s.isFloatingPoint()) {
    return out << s.toDouble();
  }
  if (s.isComplex()) {
    return out << s.toComplexDouble();
  }
  if (s.isBoolean()) {
    return out << (s.toBool() ? "true" : "false");
  }
  if (s.isSymInt()) {
    return out << (s.toSymInt());
  }
  if (s.isSymFloat()) {
    return out << (s.toSymFloat());
  }
  if (s.isIntegral(false)) {
    return out << s.toLong();
  }
  throw std::logic_error("Unknown type in Scalar");
}

std::string toString(const Scalar& s) {
<<<<<<< HEAD
  std::stringstream out;
  out << s;
  return std::move(out).str();
}
}
namespace at {

//not all C++ compilers have default float so we define our own here
inline std::ios_base& defaultfloat(std::ios_base& __base) {
  __base.unsetf(std::ios_base::floatfield);
  return __base;
}
//saves/restores number formatting inside scope
struct FormatGuard {
  FormatGuard(std::ostream & out)
  : out(out) {
    saved.copyfmt(out);
  }
  ~FormatGuard() {
    out.copyfmt(saved);
  }
  FormatGuard(const FormatGuard&) = delete;
  FormatGuard(FormatGuard&&) = delete;
  FormatGuard& operator=(const FormatGuard&) = delete;
  FormatGuard& operator=(FormatGuard&&) = delete;
private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  std::ostream & out;
  std::ios saved{nullptr};
};

std::ostream& operator<<(std::ostream & out, const DeprecatedTypeProperties& t) {
  return out << t.toString();
}

static std::tuple<double, int> __printFormat(std::ostream& stream, const Tensor& self) {
  auto size = self.numel();
  if(size == 0) {
    return std::make_tuple(1., 0);
  }
=======
  return fmt::format("{}", fmt::streamed(s));
}
} // namespace c10

namespace at {

std::ostream& operator<<(std::ostream& out, const DeprecatedTypeProperties& t) {
  return out << t.toString();
}

enum class FormatType {
  Default, // 'g' format (defaultfloat equivalent)
  Scientific, // 'e' format with precision 4
  Fixed // 'f' format with precision 4
};

struct PrintFormat {
  double scale;
  int width;
  FormatType type;

  PrintFormat(double s, int w, FormatType t = FormatType::Default)
      : scale(s), width(w), type(t) {}
};

static PrintFormat __printFormat(const Tensor& self) {
  auto size = self.numel();
  if (size == 0) {
    return PrintFormat(1., 0);
  }

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  bool intMode = true;
  auto self_p = self.const_data_ptr<double>();
  for (const auto i : c10::irange(size)) {
    auto z = self_p[i];
<<<<<<< HEAD
    if(std::isfinite(z)) {
      if(z != std::ceil(z)) {
=======
    if (std::isfinite(z)) {
      if (z != std::ceil(z)) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        intMode = false;
        break;
      }
    }
  }
<<<<<<< HEAD
  int64_t offset = 0;
  while(!std::isfinite(self_p[offset])) {
    offset = offset + 1;
    if(offset == size) {
      break;
    }
  }
  double expMin = 1;
  double expMax = 1;
  if(offset != size) {
    expMin = fabs(self_p[offset]);
    expMax = fabs(self_p[offset]);
    for (const auto i : c10::irange(offset, size)) {
      double z = fabs(self_p[i]);
      if(std::isfinite(z)) {
        if(z < expMin) {
          expMin = z;
        }
        if(self_p[i] > expMax) {
          expMax = z;
        }
      }
    }
    if(expMin != 0) {
=======

  int64_t offset = 0;
  while (offset < size && !std::isfinite(self_p[offset])) {
    offset = offset + 1;
  }

  double expMin = 1;
  double expMax = 1;
  if (offset != size) {
    expMin = std::fabs(self_p[offset]);
    expMax = std::fabs(self_p[offset]);
    for (const auto i : c10::irange(offset, size)) {
      double z = std::fabs(self_p[i]);
      if (std::isfinite(z)) {
        expMin = std::min(expMin, z);
        expMax = std::max(expMax, z);
      }
    }
    if (expMin != 0) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
      expMin = std::floor(std::log10(expMin)) + 1;
    } else {
      expMin = 1;
    }
<<<<<<< HEAD
    if(expMax != 0) {
=======
    if (expMax != 0) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
      expMax = std::floor(std::log10(expMax)) + 1;
    } else {
      expMax = 1;
    }
  }
<<<<<<< HEAD
  double scale = 1;
  int sz = 11;
  if(intMode) {
    if(expMax > 9) {
      sz = 11;
      stream << std::scientific << std::setprecision(4);
    } else {
      sz = static_cast<int>(expMax) + 1;
      stream << defaultfloat;
    }
  } else {
    if(expMax-expMin > 4) {
      sz = 11;
      if(std::fabs(expMax) > 99 || std::fabs(expMin) > 99) {
        sz = sz + 1;
      }
      stream << std::scientific << std::setprecision(4);
    } else {
      if(expMax > 5 || expMax < 0) {
        sz = 7;
        scale = std::pow(10, expMax-1);
        stream << std::fixed << std::setprecision(4);
      } else {
        if(expMax == 0) {
=======

  double scale = 1;
  int sz = 11;

  if (intMode) {
    if (expMax > 9) {
      sz = 11;
      return PrintFormat(scale, sz, FormatType::Scientific);
    } else {
      sz = static_cast<int>(expMax) + 1;
      return PrintFormat(scale, sz, FormatType::Default);
    }
  } else {
    if (expMax - expMin > 4) {
      sz = 11;
      if (std::fabs(expMax) > 99 || std::fabs(expMin) > 99) {
        sz = sz + 1;
      }
      return PrintFormat(scale, sz, FormatType::Scientific);
    } else {
      if (expMax > 5 || expMax < 0) {
        sz = 7;
        scale = std::pow(10, expMax - 1);
        return PrintFormat(scale, sz, FormatType::Fixed);
      } else {
        if (expMax == 0) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
          sz = 7;
        } else {
          sz = static_cast<int>(expMax) + 6;
        }
<<<<<<< HEAD
        stream << std::fixed << std::setprecision(4);
      }
    }
  }
  return std::make_tuple(scale, sz);
}

static void __printIndent(std::ostream &stream, int64_t indent)
{
  for ([[maybe_unused]] const auto i : c10::irange(indent)) {
    stream << " ";
  }
}

static void printScale(std::ostream & stream, double scale) {
  FormatGuard guard(stream);
  stream << defaultfloat << scale << " *" << '\n';
}
static void __printMatrix(std::ostream& stream, const Tensor& self, int64_t linesize, int64_t indent)
{
  auto [scale, sz] = __printFormat(stream, self);

  __printIndent(stream, indent);
  int64_t nColumnPerLine = (linesize-indent)/(sz+1);
  int64_t firstColumn = 0;
  int64_t lastColumn = -1;
  while(firstColumn < self.size(1)) {
    if(firstColumn + nColumnPerLine <= self.size(1)) {
=======
        return PrintFormat(scale, sz, FormatType::Fixed);
      }
    }
  }
}

// Precompiled format specs
static constexpr auto FMT_G = FMT_COMPILE("{:>{}g}");
static constexpr auto FMT_E4 = FMT_COMPILE("{:>{}.4e}");
static constexpr auto FMT_F4 = FMT_COMPILE("{:>{}.4f}");

// Print a single value directly into the stream buffer with no temporaries
static void printValue(std::ostream& stream, double v, const PrintFormat& pf) {
  auto out_it = std::ostreambuf_iterator<char>(stream);
  double val = v / pf.scale;
  switch (pf.type) {
    case FormatType::Default:
      fmt::format_to(out_it, FMT_G, val, pf.width);
      break;
    case FormatType::Scientific:
      fmt::format_to(out_it, FMT_E4, val, pf.width);
      break;
    case FormatType::Fixed:
      fmt::format_to(out_it, FMT_F4, val, pf.width);
      break;
  }
}

static void __printMatrix(
    std::ostream& stream,
    const Tensor& self,
    int64_t linesize,
    int64_t indent) {
  auto printFmt = __printFormat(self);

  int64_t nColumnPerLine = (linesize - indent) / (printFmt.width + 1);
  int64_t firstColumn = 0;
  int64_t lastColumn = -1;

  while (firstColumn < self.size(1)) {
    if (firstColumn + nColumnPerLine <= self.size(1)) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
      lastColumn = firstColumn + nColumnPerLine - 1;
    } else {
      lastColumn = self.size(1) - 1;
    }
<<<<<<< HEAD
    if(nColumnPerLine < self.size(1)) {
      if(firstColumn != 0) {
        stream << '\n';
      }
      stream << "Columns " << firstColumn+1 << " to " << lastColumn+1;
      __printIndent(stream, indent);
    }
    if(scale != 1) {
      printScale(stream,scale);
      __printIndent(stream, indent);
    }
    for (const auto l : c10::irange(self.size(0))) {
      Tensor row = self.select(0,l);
      const double *row_ptr = row.const_data_ptr<double>();
      for (const auto c : c10::irange(firstColumn, lastColumn+1)) {
        stream << std::setw(sz) << row_ptr[c]/scale;
        if(c == lastColumn) {
          stream << '\n';
          if(l != self.size(0)-1) {
            if(scale != 1) {
              __printIndent(stream, indent);
              stream << " ";
            } else {
              __printIndent(stream, indent);
            }
          }
        } else {
          stream << " ";
=======

    if (nColumnPerLine < self.size(1)) {
      if (firstColumn != 0) {
        stream.put('\n');
      }
      fmt::print(
          stream,
          "Columns {} to {}{:>{}s}",
          firstColumn + 1,
          lastColumn + 1,
          "", // empty string to pad
          indent // width to pad to
      );
    }

    if (printFmt.scale != 1) {
      fmt::print(stream, "{} *\n{:>{}s}", printFmt.scale, "", indent);
    }

    for (const auto l : c10::irange(self.size(0))) {
      Tensor row = self.select(0, l);
      const double* row_ptr = row.const_data_ptr<double>();

      for (const auto c : c10::irange(firstColumn, lastColumn + 1)) {
        printValue(stream, row_ptr[c], printFmt);

        if (c == lastColumn) {
          stream.put('\n');
          if (l != self.size(0) - 1) {
            if (printFmt.scale != 1) {
              fmt::print(stream, "{:>{}s} ", "", indent);
            } else {
              fmt::print(stream, "{:>{}s}", "", indent);
            }
          }
        } else {
          stream.put(' ');
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        }
      }
    }
    firstColumn = lastColumn + 1;
  }
}

<<<<<<< HEAD
static void __printTensor(std::ostream& stream, Tensor& self, int64_t linesize)
{
  std::vector<int64_t> counter(self.ndimension()-2);
  bool start = true;
  bool finished = false;
  counter[0] = -1;
  for (const auto i : c10::irange(1, counter.size())) {
    counter[i] = 0;
  }
  while(true) {
    for(int64_t i = 0; self.ndimension()-2; i++) {
      counter[i] = counter[i] + 1;
      if(counter[i] >= self.size(i)) {
        if(i == self.ndimension()-3) {
=======
static void __printTensor(
    std::ostream& stream,
    Tensor& self,
    int64_t linesize) {
  std::vector<int64_t> counter(self.ndimension() - 2, 0);
  counter[0] = -1;

  bool start = true;
  bool finished = false;

  while (true) {
    for (int64_t i = 0; self.ndimension() - 2; i++) {
      counter[i] = counter[i] + 1;
      if (counter[i] >= self.size(i)) {
        if (i == self.ndimension() - 3) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
          finished = true;
          break;
        }
        counter[i] = 0;
      } else {
        break;
      }
    }
<<<<<<< HEAD
    if(finished) {
      break;
    }
    if(start) {
      start = false;
    } else {
      stream << '\n';
    }
    stream << "(";
    Tensor tensor = self;
    for (const auto i : c10::irange(self.ndimension()-2)) {
      tensor = tensor.select(0, counter[i]);
      stream << counter[i]+1 << ",";
    }
    stream << ".,.) = " << '\n';
=======
    if (finished) {
      break;
    }
    if (start) {
      start = false;
    } else {
      stream.put('\n');
    }

    stream.put('(');
    Tensor tensor = self;
    for (const auto i : c10::irange(self.ndimension() - 2)) {
      tensor = tensor.select(0, counter[i]);
      fmt::print(stream, "{},", counter[i] + 1);
    }
    fmt::print(stream, ".,.) = \n");
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    __printMatrix(stream, tensor, linesize, 1);
  }
}

<<<<<<< HEAD
void print(const Tensor & t, int64_t linesize) {
  print(std::cout,t,linesize);
}
std::ostream& print(std::ostream& stream, const Tensor & tensor_, int64_t linesize) {
  FormatGuard guard(stream);
  if(!tensor_.defined()) {
    stream << "[ Tensor (undefined) ]";
  } else if (tensor_.is_sparse()) {
    stream << "[ " << tensor_.toString() << "{}\n";
    stream << "indices:\n" << tensor_._indices() << "\n";
    stream << "values:\n" << tensor_._values() << "\n";
    stream << "size:\n" << tensor_.sizes() << "\n";
    stream << "]";
  } else {
    Tensor tensor;
    if (tensor_.is_quantized()) {
      tensor = tensor_.dequantize().to(kCPU, kDouble).contiguous();
    } else if (tensor_.is_mkldnn()) {
      stream << "MKLDNN Tensor: ";
      tensor = tensor_.to_dense().to(kCPU, kDouble).contiguous();
    } else if (tensor_.is_mps()) {
      // MPS does not support double tensors, so first copy then convert
      tensor = tensor_.to(kCPU).to(kDouble).contiguous();
    } else {
      tensor = tensor_.to(kCPU, kDouble).contiguous();
    }
    if(tensor.ndimension() == 0) {
      stream << defaultfloat << tensor.const_data_ptr<double>()[0] << '\n';
      stream << "[ " << tensor_.toString() << "{}";
    } else if(tensor.ndimension() == 1) {
      if (tensor.numel() > 0) {
        auto [scale, sz] = __printFormat(stream, tensor);
        if(scale != 1) {
          printScale(stream, scale);
        }
        const double* tensor_p = tensor.const_data_ptr<double>();
        for (const auto i : c10::irange(tensor.size(0))) {
          stream << std::setw(sz) << tensor_p[i]/scale << '\n';
        }
      }
      stream << "[ " << tensor_.toString() << "{" << tensor.size(0) << "}";
    } else if(tensor.ndimension() == 2) {
      if (tensor.numel() > 0) {
        __printMatrix(stream, tensor, linesize, 0);
      }
      stream << "[ " << tensor_.toString() << "{" << tensor.size(0) << "," <<  tensor.size(1) << "}";
    } else {
      if (tensor.numel() > 0) {
        __printTensor(stream, tensor, linesize);
      }
      stream << "[ " << tensor_.toString() << "{" << tensor.size(0);
      for (const auto i : c10::irange(1, tensor.ndimension())) {
        stream << "," << tensor.size(i);
      }
      stream << "}";
    }
    if (tensor_.is_quantized()) {
      stream << ", qscheme: " << toString(tensor_.qscheme());
      if (tensor_.qscheme() == c10::kPerTensorAffine) {
        stream << ", scale: " << tensor_.q_scale();
        stream << ", zero_point: " << tensor_.q_zero_point();
      } else if (tensor_.qscheme() == c10::kPerChannelAffine ||
          tensor_.qscheme() == c10::kPerChannelAffineFloatQParams) {
        stream << ", scales: ";
        Tensor scales = tensor_.q_per_channel_scales();
        print(stream, scales, linesize);
        stream << ", zero_points: ";
        Tensor zero_points = tensor_.q_per_channel_zero_points();
        print(stream, zero_points, linesize);
        stream << ", axis: " << tensor_.q_per_channel_axis();
      }
    }

    // Proxy check for if autograd was built
    if (tensor.getIntrusivePtr()->autograd_meta()) {
      auto& fw_grad = tensor._fw_grad(/* level */ 0);
      if (fw_grad.defined()) {
        stream << ", tangent:" << '\n' << fw_grad;
      }
    }
    stream << " ]";
  }
  return stream;
}

}
=======
void print(const Tensor& t, int64_t linesize) {
  print(std::cout, t, linesize);
}

std::ostream& print(
    std::ostream& stream,
    const Tensor& tensor_,
    int64_t linesize) {
  if (!tensor_.defined()) {
    fmt::print(stream, "[ Tensor (undefined) ]");
    return stream;
  }

  if (tensor_.is_sparse()) {
    fmt::print(stream, "[ {}{{}}\nindices:\n", tensor_.toString());
    print(stream, tensor_._indices(), linesize);
    fmt::print(stream, "\nvalues:\n");
    print(stream, tensor_._values(), linesize);
    fmt::print(stream, "\nsize:\n{}\n]", fmt::streamed(tensor_.sizes()));
    return stream;
  }

  Tensor tensor;

  if (tensor_.is_quantized()) {
    tensor = tensor_.dequantize().to(kCPU, kDouble).contiguous();
  } else if (tensor_.is_mkldnn()) {
    fmt::print(stream, "MKLDNN Tensor: ");
    tensor = tensor_.to_dense().to(kCPU, kDouble).contiguous();
  } else if (tensor_.is_mps()) {
    // MPS does not support double tensors, so first copy then convert
    tensor = tensor_.to(kCPU).to(kDouble).contiguous();
  } else {
    tensor = tensor_.to(kCPU, kDouble).contiguous();
  }

  if (tensor.ndimension() == 0) {
    fmt::print(
        stream,
        "{}\n[ {}{{}}",
        tensor.const_data_ptr<double>()[0],
        tensor_.toString());
  } else if (tensor.ndimension() == 1) {
    if (tensor.numel() > 0) {
      auto printFmt = __printFormat(tensor);
      if (printFmt.scale != 1) {
        fmt::print(stream, "{} *\n", printFmt.scale);
      }
      const double* tensor_p = tensor.const_data_ptr<double>();
      for (const auto i : c10::irange(tensor.size(0))) {
        printValue(stream, tensor_p[i], printFmt);
        stream.put('\n');
      }
    }
    fmt::print(stream, "[ {}{{{}}}", tensor_.toString(), tensor.size(0));
  } else if (tensor.ndimension() == 2) {
    if (tensor.numel() > 0) {
      __printMatrix(stream, tensor, linesize, 0);
    }
    fmt::print(
        stream,
        "[ {}{{{},{}}}",
        tensor_.toString(),
        tensor.size(0),
        tensor.size(1));
  } else {
    if (tensor.numel() > 0) {
      __printTensor(stream, tensor, linesize);
    }
    fmt::print(stream, "[ {}{{{}", tensor_.toString(), tensor.size(0));
    for (const auto i : c10::irange(1, tensor.ndimension())) {
      fmt::print(stream, ",{}", tensor.size(i));
    }
    fmt::print(stream, "}}");
  }

  // Add quantization info
  if (tensor_.is_quantized()) {
    fmt::print(stream, ", qscheme: {}", toString(tensor_.qscheme()));
    if (tensor_.qscheme() == c10::kPerTensorAffine) {
      fmt::print(
          stream,
          ", scale: {}, zero_point: {}",
          tensor_.q_scale(),
          tensor_.q_zero_point());
    } else if (
        tensor_.qscheme() == c10::kPerChannelAffine ||
        tensor_.qscheme() == c10::kPerChannelAffineFloatQParams) {
      fmt::print(stream, ", scales: ");
      print(stream, tensor_.q_per_channel_scales(), linesize);
      fmt::print(stream, ", zero_points: ");
      print(stream, tensor_.q_per_channel_zero_points(), linesize);
      fmt::print(stream, ", axis: {}", tensor_.q_per_channel_axis());
    }
  }

  // Proxy check for if autograd was built
  if (tensor.getIntrusivePtr()->autograd_meta()) {
    auto& fw_grad = tensor._fw_grad(/* level */ 0);
    if (fw_grad.defined()) {
      fmt::print(stream, ", tangent:\n");
      print(stream, fw_grad, linesize);
    }
  }

  fmt::print(stream, " ]");
  return stream;
}

} // namespace at
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

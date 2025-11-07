<<<<<<< HEAD
=======
#include <c10/metal/indexing.h>
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <c10/metal/special_math.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;

<<<<<<< HEAD
=======
struct add_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(a + b);
  }
};

struct sub_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(a - b);
  }
};

struct add_alpha_functor {
  template <typename T>
  inline T operator()(const T a, const T b, const T alpha) {
    return static_cast<T>(a + c10::metal::mul(alpha, b));
  }
};

struct sub_alpha_functor {
  template <typename T>
  inline T operator()(const T a, const T b, const T alpha) {
    return static_cast<T>(a - c10::metal::mul(alpha, b));
  }
};

struct lerp_alpha_functor {
  template <typename T>
  inline T operator()(const T a, const T b, const T alpha) {
    return static_cast<T>(a + c10::metal::mul(alpha, b - a));
  }
};

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
struct fmax_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(::metal::fmax(a, b));
  }
};

struct fmin_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(::metal::fmin(a, b));
  }
};

struct copysign_functor {
  template <typename T>
  inline enable_if_t<is_floating_point_v<T>, T> operator()(
      const T a,
      const T b) {
    return static_cast<T>(::metal::copysign(a, b));
  }
  template <typename T>
  inline enable_if_t<!is_floating_point_v<T>, float> operator()(
      const T a,
      const T b) {
    return ::metal::copysign(static_cast<float>(a), static_cast<float>(b));
  }
};

struct zeta_functor {
<<<<<<< HEAD
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::zeta(a, b));
  }
};

struct xlog1py_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::xlog1py(a, b));
  }
=======
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::zeta(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::zeta(float(a), float(b));
  }
};

struct xlog1py_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::xlog1py(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::xlog1py(float(a), float(b));
  }
};

struct chebyshev_polynomial_t_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::chebyshev_polynomial_t_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::chebyshev_polynomial_t_forward(float(a), float(b));
  }
};

struct chebyshev_polynomial_u_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::chebyshev_polynomial_u_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::chebyshev_polynomial_u_forward(float(a), float(b));
  }
};

struct chebyshev_polynomial_v_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::chebyshev_polynomial_v_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::chebyshev_polynomial_v_forward(float(a), float(b));
  }
};

struct chebyshev_polynomial_w_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::chebyshev_polynomial_w_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::chebyshev_polynomial_w_forward(float(a), float(b));
  }
};

struct hermite_polynomial_h_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::hermite_polynomial_h_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::hermite_polynomial_h_forward(float(a), float(b));
  }
};

struct hermite_polynomial_he_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::hermite_polynomial_he_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::hermite_polynomial_he_forward(float(a), float(b));
  }
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
};

struct nextafter_functor {
#if __METAL_VERSION__ < 310
  template <typename U>
  struct bit_type {};
  template <>
  struct bit_type<float> {
    using type = int;
  };
  template <>
  struct bit_type<half> {
    using type = short;
  };
#endif
  template <typename T>
  inline T operator()(const T a, const T b) {
#if __METAL_VERSION__ >= 310
    return static_cast<T>(::metal::nextafter(a, b));
#else
    using U = typename bit_type<T>::type;
    if (a == b) {
      return a;
    }
    if (::metal::isunordered(a, b)) {
      return NAN;
    }
    if (a == 0) {
      constexpr auto eps = as_type<T>(static_cast<U>(1));
      return b > 0 ? eps : -eps;
    }
    auto bits = as_type<U>(a);
    (a > 0) ^ (a > b) ? bits++ : bits--;
    return as_type<T>(bits);
#endif
  }
};

<<<<<<< HEAD
=======
// Complex binary functors
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
struct polar_functor {
  template <typename U>
  using ret_type = c10::metal::vec2type_t<U>;
  template <typename T>
  inline ret_type<T> operator()(const T a, const T b) {
    return ret_type<T>(a * cos(b), a * sin(b));
  }
};

<<<<<<< HEAD
// Future BinaryTensorIterator
template <typename T, typename F>
using result_of = decltype(::metal::declval<F>()(
    ::metal::declval<T>(),
    ::metal::declval<T>()));

template <typename T, typename F>
kernel void binary_indexing(
    constant void* input_ [[buffer(0)]],
    constant void* other_ [[buffer(1)]],
    device void* out_ [[buffer(2)]],
    constant uint3* offsets [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto out = (device result_of<T, F>*)((device uint8_t*)out_ + offsets[tid].x);
  auto input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  auto other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);
  F f;
  *out = f(*input, *other);
}

template <typename T, typename F>
kernel void binary_dense(
    constant T* input [[buffer(0)]],
    constant T* other [[buffer(1)]],
    device result_of<T, F>* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  out[tid] = f(input[tid], other[tid]);
}

#define REGISTER_BINARY_INDEXING_OP(NAME, DTYPE)             \
  template [[host_name(#NAME "_" #DTYPE)]] kernel void       \
  binary_indexing<DTYPE, NAME##_functor>(                    \
      constant void* input_,                                 \
      constant void* other_,                                 \
      device void* out_,                                     \
      constant uint3* offsets,                               \
      uint tid);                                             \
  template [[host_name(#NAME "_dense_" #DTYPE)]] kernel void \
  binary_dense<DTYPE, NAME##_functor>(                       \
      constant DTYPE * input_,                               \
      constant DTYPE * other_,                               \
      device result_of<DTYPE, NAME##_functor> * out_,        \
      uint tid)

#define REGISTER_BINARY_OP(NAME, DTYPE)                             \
  template [[host_name(#NAME "_" #DTYPE)]] kernel void NAME<DTYPE>( \
      constant void* input_,                                        \
      constant void* other_,                                        \
      device void* out_,                                            \
      constant uint3* offsets,                                      \
      uint tid)

REGISTER_BINARY_INDEXING_OP(copysign, long);
REGISTER_BINARY_INDEXING_OP(copysign, int);
REGISTER_BINARY_INDEXING_OP(copysign, float);
REGISTER_BINARY_INDEXING_OP(copysign, half);
REGISTER_BINARY_INDEXING_OP(copysign, short);
REGISTER_BINARY_INDEXING_OP(copysign, uchar);
REGISTER_BINARY_INDEXING_OP(copysign, char);
REGISTER_BINARY_INDEXING_OP(copysign, bool);
REGISTER_BINARY_INDEXING_OP(fmax, float);
REGISTER_BINARY_INDEXING_OP(fmax, half);
REGISTER_BINARY_INDEXING_OP(fmin, float);
REGISTER_BINARY_INDEXING_OP(fmin, half);
REGISTER_BINARY_INDEXING_OP(nextafter, float);
REGISTER_BINARY_INDEXING_OP(nextafter, half);
REGISTER_BINARY_INDEXING_OP(zeta, float);
REGISTER_BINARY_INDEXING_OP(zeta, half);
REGISTER_BINARY_INDEXING_OP(xlog1py, float);
REGISTER_BINARY_INDEXING_OP(xlog1py, half);

#if __METAL_VERSION__ >= 310
REGISTER_BINARY_INDEXING_OP(copysign, bfloat);
REGISTER_BINARY_INDEXING_OP(fmax, bfloat);
REGISTER_BINARY_INDEXING_OP(fmin, bfloat);
REGISTER_BINARY_INDEXING_OP(nextafter, bfloat);
REGISTER_BINARY_INDEXING_OP(zeta, bfloat);
REGISTER_BINARY_INDEXING_OP(xlog1py, bfloat);
#endif

// Complex binary functions
REGISTER_BINARY_INDEXING_OP(polar, float);
REGISTER_BINARY_INDEXING_OP(polar, half);

template <typename T>
kernel void complex_mul(
    constant void* input_ [[buffer(0)]],
    constant void* other_ [[buffer(1)]],
    device void* out_ [[buffer(2)]],
    constant uint3* offsets [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  device T* out = (device T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);
  out[0] = input[0] * other[0] - input[1] * other[1];
  out[1] = input[0] * other[1] + input[1] * other[0];
}

REGISTER_BINARY_OP(complex_mul, float);
REGISTER_BINARY_OP(complex_mul, half);

template <typename T>
kernel void complex_kernel(
    constant void* real_ [[buffer(0)]],
    constant void* imag_ [[buffer(1)]],
    device void* out_ [[buffer(2)]],
    constant uint3* offsets [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  device T* out = (device T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* real = (constant T*)((constant uint8_t*)real_ + offsets[tid].y);
  constant T* imag = (constant T*)((constant uint8_t*)imag_ + offsets[tid].z);
  out[0] = real[0];
  out[1] = imag[0];
}

REGISTER_BINARY_OP(complex_kernel, float);
REGISTER_BINARY_OP(complex_kernel, half);
=======
// Constructs complex tensor from real and imaginary planes
struct make_complex_functor {
  template <typename U>
  using ret_type = c10::metal::vec2type_t<U>;
  template <typename T>
  inline ret_type<T> operator()(const T a, const T b) {
    return ret_type<T>(a, b);
  }
};

struct mul_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return c10::metal::mul(a, b);
  }
};

struct div_true_functor {
  template <
      typename T,
      ::metal::enable_if_t<!::metal::is_integral_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return c10::metal::div(a, b);
  }
  template <
      typename T,
      ::metal::enable_if_t<::metal::is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::div(float(a), float(b));
  }
};

struct div_floor_functor {
  template <
      typename T,
      ::metal::enable_if_t<!::metal::is_integral_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return metal::floor(c10::metal::div(a, b));
  }
  template <
      typename T,
      ::metal::enable_if_t<
          ::metal::is_integral_v<T>&& ::metal::is_signed_v<T>,
          bool> = true>
  inline T operator()(const T a, const T b) {
    const auto quot = a / b;
    if ((a < 0) == (b < 0)) {
      return quot;
    }
    return a % b != 0 ? quot - 1 : quot;
  }
  template <
      typename T,
      ::metal::enable_if_t<
          ::metal::is_integral_v<T> && !::metal::is_signed_v<T>,
          bool> = true>
  inline T operator()(const T a, const T b) {
    return a / b;
  }
};

struct div_trunc_functor {
  template <
      typename T,
      ::metal::enable_if_t<!::metal::is_integral_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return T(metal::trunc(c10::metal::div(a, b)));
  }
  template <
      typename T,
      ::metal::enable_if_t<::metal::is_integral_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return a / b;
  }
};

struct remainder_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return T(c10::metal::remainder(a, b));
  }
};

struct fmod_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return c10::metal::fmod(a, b);
  }
};

// Some helper defines
#if __METAL_VERSION__ >= 310
#define _METAL_310_PLUS(x) x
#else
#define _METAL_310_PLUS(x)
#endif

#define REGISTER_INTEGER_BINARY_OP(NAME)  \
  REGISTER_BINARY_OP(NAME, long, long);   \
  REGISTER_BINARY_OP(NAME, int, int);     \
  REGISTER_BINARY_OP(NAME, short, short); \
  REGISTER_BINARY_OP(NAME, uchar, uchar); \
  REGISTER_BINARY_OP(NAME, char, char);   \
  REGISTER_BINARY_OP(NAME, bool, bool)

#define REGISTER_INT2FLOAT_BINARY_OP(NAME) \
  REGISTER_BINARY_OP(NAME, long, float);   \
  REGISTER_BINARY_OP(NAME, int, float);    \
  REGISTER_BINARY_OP(NAME, short, float);  \
  REGISTER_BINARY_OP(NAME, uchar, float);  \
  REGISTER_BINARY_OP(NAME, char, float);   \
  REGISTER_BINARY_OP(NAME, bool, float)

#define REGISTER_FLOAT_BINARY_OP(NAME)    \
  REGISTER_BINARY_OP(NAME, float, float); \
  REGISTER_BINARY_OP(NAME, half, half);   \
  _METAL_310_PLUS(REGISTER_BINARY_OP(NAME, bfloat, bfloat))

#define REGISTER_OPMATH_FLOAT_BINARY_OP(NAME)    \
  REGISTER_OPMATH_BINARY_OP(NAME, float, float); \
  REGISTER_OPMATH_BINARY_OP(NAME, half, half);   \
  _METAL_310_PLUS(REGISTER_OPMATH_BINARY_OP(NAME, bfloat, bfloat))

REGISTER_FLOAT_BINARY_OP(copysign);
REGISTER_INT2FLOAT_BINARY_OP(copysign);
REGISTER_FLOAT_BINARY_OP(fmax);
REGISTER_FLOAT_BINARY_OP(fmin);
REGISTER_FLOAT_BINARY_OP(nextafter);
REGISTER_FLOAT_BINARY_OP(zeta);
REGISTER_INT2FLOAT_BINARY_OP(zeta);
REGISTER_FLOAT_BINARY_OP(xlog1py);
REGISTER_INT2FLOAT_BINARY_OP(xlog1py);
REGISTER_FLOAT_BINARY_OP(chebyshev_polynomial_t);
REGISTER_INT2FLOAT_BINARY_OP(chebyshev_polynomial_t);
REGISTER_FLOAT_BINARY_OP(chebyshev_polynomial_u);
REGISTER_INT2FLOAT_BINARY_OP(chebyshev_polynomial_u);
REGISTER_FLOAT_BINARY_OP(chebyshev_polynomial_v);
REGISTER_INT2FLOAT_BINARY_OP(chebyshev_polynomial_w);
REGISTER_FLOAT_BINARY_OP(chebyshev_polynomial_w);
REGISTER_INT2FLOAT_BINARY_OP(chebyshev_polynomial_v);
REGISTER_FLOAT_BINARY_OP(hermite_polynomial_h);
REGISTER_INT2FLOAT_BINARY_OP(hermite_polynomial_h);
REGISTER_FLOAT_BINARY_OP(hermite_polynomial_he);
REGISTER_INT2FLOAT_BINARY_OP(hermite_polynomial_he);
REGISTER_FLOAT_BINARY_OP(add);
REGISTER_INTEGER_BINARY_OP(add);
REGISTER_OPMATH_FLOAT_BINARY_OP(mul);
REGISTER_INTEGER_BINARY_OP(mul);
REGISTER_FLOAT_BINARY_OP(sub);
REGISTER_INTEGER_BINARY_OP(sub);
REGISTER_OPMATH_FLOAT_BINARY_OP(div_floor);
REGISTER_INTEGER_BINARY_OP(div_floor);
REGISTER_FLOAT_BINARY_OP(div_trunc);
REGISTER_INTEGER_BINARY_OP(div_trunc);
REGISTER_OPMATH_FLOAT_BINARY_OP(div_true);
REGISTER_INT2FLOAT_BINARY_OP(div_true);
REGISTER_OPMATH_FLOAT_BINARY_OP(remainder);
REGISTER_INTEGER_BINARY_OP(remainder);
REGISTER_OPMATH_FLOAT_BINARY_OP(fmod);
REGISTER_INTEGER_BINARY_OP(fmod);
REGISTER_BINARY_ALPHA_OP(add_alpha, long, long, long);
REGISTER_BINARY_ALPHA_OP(add_alpha, int, int, int);
REGISTER_BINARY_ALPHA_OP(add_alpha, float, float, float);
REGISTER_BINARY_ALPHA_OP(add_alpha, half, half, half);
REGISTER_BINARY_ALPHA_OP(add_alpha, short, short, short);
REGISTER_BINARY_ALPHA_OP(add_alpha, uchar, uchar, uchar);
REGISTER_BINARY_ALPHA_OP(add_alpha, char, char, char);
REGISTER_BINARY_ALPHA_OP(add_alpha, bool, bool, bool);
REGISTER_BINARY_ALPHA_OP(sub_alpha, long, long, long);
REGISTER_BINARY_ALPHA_OP(sub_alpha, int, int, int);
REGISTER_BINARY_ALPHA_OP(sub_alpha, float, float, float);
REGISTER_BINARY_ALPHA_OP(sub_alpha, half, half, half);
REGISTER_BINARY_ALPHA_OP(sub_alpha, short, short, short);
REGISTER_BINARY_ALPHA_OP(sub_alpha, uchar, uchar, uchar);
REGISTER_BINARY_ALPHA_OP(sub_alpha, char, char, char);
REGISTER_BINARY_ALPHA_OP(sub_alpha, bool, bool, bool);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, long, long, long);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, int, int, int);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, float, float, float);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, half, half, half);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, short, short, short);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, uchar, uchar, uchar);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, char, char, char);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, bool, bool, bool);

#if __METAL_VERSION__ >= 310
REGISTER_BINARY_ALPHA_OP(add_alpha, bfloat, bfloat, bfloat);
REGISTER_BINARY_ALPHA_OP(sub_alpha, bfloat, bfloat, bfloat);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, bfloat, bfloat, bfloat);
#endif

// Complex binary functions
REGISTER_BINARY_OP(polar, float, float2);
REGISTER_BINARY_OP(polar, half, half2);
REGISTER_BINARY_OP(make_complex, float, float2);
REGISTER_BINARY_OP(make_complex, half, half2);
REGISTER_OPMATH_BINARY_OP(mul, float2, float2);
REGISTER_OPMATH_BINARY_OP(mul, half2, half2);
REGISTER_OPMATH_BINARY_OP(div_true, float2, float2);
REGISTER_OPMATH_BINARY_OP(div_true, half2, half2);
REGISTER_BINARY_OP(add, float2, float2);
REGISTER_BINARY_OP(add, half2, half2);
REGISTER_BINARY_OP(sub, float2, float2);
REGISTER_BINARY_OP(sub, half2, half2);
REGISTER_BINARY_ALPHA_OP(add_alpha, float2, float2, float2);
REGISTER_BINARY_ALPHA_OP(add_alpha, half2, half2, half2);
REGISTER_BINARY_ALPHA_OP(sub_alpha, float2, float2, float2);
REGISTER_BINARY_ALPHA_OP(sub_alpha, half2, half2, half2);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, float2, float2, float2);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, half2, half2, half2);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

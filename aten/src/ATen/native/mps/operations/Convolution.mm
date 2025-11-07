//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ConvUtils.h>
<<<<<<< HEAD
=======
#include <ATen/native/mps/MPSGraphVenturaOps.h>
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/_mps_convolution_native.h>
#include <ATen/ops/_mps_convolution_transpose_native.h>
#include <ATen/ops/mps_convolution_backward_native.h>
#include <ATen/ops/mps_convolution_transpose_backward_native.h>
<<<<<<< HEAD
#include <fmt/format.h>
=======

#if !defined(__MAC_13_2) && (!defined(MAC_OS_X_VERSION_13_2) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_13_2))

@implementation FakeMPSGraphConvolution3DOpDescriptor
- (nonnull id)copyWithZone:(nullable NSZone*)zone {
  return self;
}

@end

#endif
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

namespace at::native {

// Create 3D convolution descriptor
static void fill_conv3d_desc(MPSGraphConvolution3DOpDescriptor* descriptor_,
                             NSUInteger strideInX,
                             NSUInteger strideInY,
                             NSUInteger strideInZ,
                             NSUInteger dilationRateInX,
                             NSUInteger dilationRateInY,
                             NSUInteger dilationRateInZ,
                             NSUInteger paddingHorizontal,
                             NSUInteger paddingVertical,
                             NSUInteger paddingDepth,
                             NSUInteger groups) {
  descriptor_.strideInX = strideInX;
  descriptor_.strideInY = strideInY;
  descriptor_.strideInZ = strideInZ;
  descriptor_.dilationRateInX = dilationRateInX;
  descriptor_.dilationRateInY = dilationRateInY;
  descriptor_.dilationRateInZ = dilationRateInZ;

  // TODO: Program the padding style
  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;

  descriptor_.paddingLeft = paddingHorizontal;
  descriptor_.paddingRight = paddingHorizontal;
  descriptor_.paddingTop = paddingVertical;
  descriptor_.paddingBottom = paddingVertical;
  descriptor_.paddingFront = paddingDepth;
  descriptor_.paddingBack = paddingDepth;

<<<<<<< HEAD
  descriptor_.dataLayout = MPSGraphTensorNamedDataLayoutNCDHW;

  descriptor_.weightsLayout = MPSGraphTensorNamedDataLayoutOIDHW;
=======
  // PyTorch always uses NCDHW memory layout for 3D tensors
  descriptor_.dataLayout = (MPSGraphTensorNamedDataLayout)7L; // MPSGraphTensorNamedDataLayoutNCDHW;

  // PyTorch always uses OIDHW memory layout for 3D weights
  descriptor_.weightsLayout = (MPSGraphTensorNamedDataLayout)9L; // MPSGraphTensorNamedDataLayoutOIDHW;
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

  descriptor_.groups = groups; // not yet tested in Xcode/C++
}

static void fill_depthwise_conv_desc(MPSGraphDepthwiseConvolution3DOpDescriptor* descriptor_,
                                     NSUInteger strideInX,
                                     NSUInteger strideInY,
                                     NSUInteger dilationRateInX,
                                     NSUInteger dilationRateInY,
                                     NSUInteger paddingHorizontal,
<<<<<<< HEAD
                                     NSUInteger paddingVertical) {
=======
                                     NSUInteger paddingVertical,
                                     c10::MemoryFormat memory_format,
                                     NSUInteger groups) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  descriptor_.strides =
      @[ @1, [[NSNumber alloc] initWithInteger:strideInY], [[NSNumber alloc] initWithInteger:strideInX] ];
  descriptor_.dilationRates =
      @[ @1, [[NSNumber alloc] initWithInteger:dilationRateInY], [[NSNumber alloc] initWithInteger:dilationRateInX] ];

  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;
  descriptor_.paddingValues = @[
    @0,
    @0,
    [[NSNumber alloc] initWithInteger:paddingVertical],
    [[NSNumber alloc] initWithInteger:paddingVertical],
    [[NSNumber alloc] initWithInteger:paddingHorizontal],
    [[NSNumber alloc] initWithInteger:paddingHorizontal]
  ];
  descriptor_.channelDimensionIndex = -3LL;
}

// Create convolution descriptor
static void fill_conv_desc(MPSGraphConvolution2DOpDescriptor* descriptor_,
                           NSUInteger strideInX,
                           NSUInteger strideInY,
                           NSUInteger dilationRateInX,
                           NSUInteger dilationRateInY,
                           NSUInteger paddingHorizontal,
                           NSUInteger paddingVertical,
                           c10::MemoryFormat memory_format,
                           NSUInteger groups) {
  descriptor_.strideInX = strideInX;
  descriptor_.strideInY = strideInY;
  descriptor_.dilationRateInX = dilationRateInX;
  descriptor_.dilationRateInY = dilationRateInY;

  // TODO: Program the padding style
  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;

  descriptor_.paddingLeft = paddingHorizontal;
  descriptor_.paddingRight = paddingHorizontal;
  descriptor_.paddingTop = paddingVertical;
  descriptor_.paddingBottom = paddingVertical;

  descriptor_.dataLayout = (memory_format == at::MemoryFormat::Contiguous) ? MPSGraphTensorNamedDataLayoutNCHW
                                                                           : MPSGraphTensorNamedDataLayoutNHWC;

  // PyTorch always uses OIHW memory layout for weights
  descriptor_.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
  descriptor_.groups = groups;
}

<<<<<<< HEAD
static Tensor _mps_convolution_impl(const Tensor& input_t,
=======
static Tensor _mps_convolution_impl(const Tensor& input_t_,
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
                                    const Tensor& weight_t,
                                    const std::optional<Tensor>& bias_opt,
                                    IntArrayRef padding,
                                    IntArrayRef stride,
                                    IntArrayRef dilation,
                                    int64_t groups,
                                    std::optional<IntArrayRef> input_shape) {
<<<<<<< HEAD
  constexpr auto kChannelsLast = MemoryFormat::ChannelsLast;
  constexpr auto kContiguous = MemoryFormat::Contiguous;
  const bool is_macos_15_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);

  const bool is3DConv = input_t.dim() == 5;
  const auto memory_format = input_t.suggest_memory_format();
  const auto input_suggested_layout = memory_format == kChannelsLast && is_macos_15_plus ? kChannelsLast : kContiguous;
  const bool is_channels_last = mps_conv_use_channels_last(input_t, weight_t) && !is3DConv;
  const bool bias_defined = bias_opt ? bias_opt->defined() : false;
=======
  const bool is_macOS_13_2_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_2_PLUS);
  const bool is_macOS_15_0_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);
  Tensor input_t = input_t_;
  bool is3DConv = input_t.dim() == 5;
  if (!is_macOS_15_0_or_newer || is3DConv) {
    input_t = input_t.contiguous();
  }

  TORCH_CHECK(((input_t.dim() < 5) || is_macOS_13_2_or_newer),
              "Conv3D is only supported on MPS for MacOS_13_2 or newer");
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

  TORCH_CHECK(isFloatingType(input_t.scalar_type()), "Convolution is supported only for Floating types");

  using namespace at::native::mps;
  CheckedFrom c = "mps_convolution";
  TensorArg input{input_t, "input", 1}, weight{weight_t, "weight", 2};
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

<<<<<<< HEAD
=======
  bool bias_defined;

  if (bias_opt == std::nullopt)
    bias_defined = false;
  else
    bias_defined = bias_opt->defined();

  auto memory_format = input_t.suggest_memory_format();
  bool is_channels_last = (memory_format == at::MemoryFormat::ChannelsLast) && !is3DConv;
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  auto output_t =
      at::empty(input_shape.has_value() ? input_shape.value()
                                        : conv_output_size(input->sizes(), weight->sizes(), padding, stride, dilation),
                input->scalar_type(),
                std::nullopt,
                kMPS,
                std::nullopt,
<<<<<<< HEAD
                is_channels_last ? kChannelsLast : kContiguous);
=======
                is_macOS_15_0_or_newer ? memory_format : MemoryFormat::Contiguous);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  if (output_t.numel() == 0) {
    return output_t;
  }
  TensorArg output{output_t, "result", 0};

<<<<<<< HEAD
  // TODO: Remove me when MacOS-14 is no longer supported
  std::optional<Tensor> output_c;
  if (!is_macos_15_plus && is_channels_last) {
    output_c = at::empty_like(output_t, output_t.options().memory_format(kContiguous));
  }

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  if (!is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_1_PLUS)) {
    // On macOS < 15.1, MPS convolution kernel does not support output channels > 2^16
    for (auto elem : output_t.sizes()) {
      TORCH_CHECK_NOT_IMPLEMENTED(elem <= (1 << 16), "Output channels > 65536 not supported at the MPS device. ");
    }
  }

  convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* biasTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    IntArrayRef bias_shape;
    if (bias_defined)
      bias_shape = bias_opt.value().sizes();

<<<<<<< HEAD
=======
    std::string mem_format_key;
    switch (memory_format) {
      case at::MemoryFormat::Contiguous:
        mem_format_key = "Contiguous";
        break;
      case at::MemoryFormat::ChannelsLast:
        mem_format_key = "ChannelsLast";
        break;
      default:
        assert(0 && "Check should have been done earlier\n");
    }

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    std::string bias_shape_key;
    if (bias_defined) {
      bias_shape_key = std::to_string(bias_shape[0]);
    } else {
      bias_shape_key = "nobias";
    }

<<<<<<< HEAD
    std::string key = fmt::format("mps_{}convolution:{}:{}:{}:{}:{}:{}:{}:{}",
                                  is3DConv ? "3d_" : "",
                                  getArrayRefString(stride),
                                  getArrayRefString(dilation),
                                  getArrayRefString(padding),
                                  groups,
                                  input_suggested_layout == kChannelsLast,
                                  mps::getTensorsStringKey({input_t, weight_t}),
                                  bias_defined,
                                  bias_shape_key);

    auto inputShape = mps::getMPSShape(input_t, input_suggested_layout);
    auto outputShape = mps::getMPSShape(output_t, input_suggested_layout);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      bool isDepthwiseConv =
          (groups > 1 && weight_t.size(1) == 1) && input_t.dim() >= 4 && weight_t.dim() >= 4 && !is_channels_last;

      auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(input_t), inputShape);
      auto weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_t);
      MPSGraphTensor* outputTensor = nil;
      if (is3DConv) {
        auto conv3dDescriptor_ = [[MPSGraphConvolution3DOpDescriptor new] autorelease];
=======
    std::string key;
    if (is3DConv) {
      key = "mps_3d_convolution:" + std::to_string(stride[0]) + ":" + std::to_string(stride[1]) + ":" +
          std::to_string(stride[2]) + ":" + std::to_string(dilation[0]) + ":" + std::to_string(dilation[1]) + ":" +
          std::to_string(dilation[2]) + ":" + std::to_string(padding[0]) + ":" + std::to_string(padding[1]) + ":" +
          std::to_string(padding[2]) + ":" + std::to_string(groups) + ":" + mem_format_key +
          mps::getTensorsStringKey({input_t, weight_t}) + ":" + std::to_string(bias_defined) + ":" + bias_shape_key;

    } else {
      key = "mps_convolution:" + std::to_string(stride[0]) + ":" + std::to_string(stride[1]) + ":" +
          std::to_string(dilation[0]) + ":" + std::to_string(dilation[1]) + ":" + std::to_string(padding[0]) + ":" +
          std::to_string(padding[1]) + ":" + std::to_string(groups) + ":" + mem_format_key +
          mps::getTensorsStringKey({input_t, weight_t}) + ":" + std::to_string(bias_defined) + ":" + bias_shape_key;
    }

    MPSShape* inputShape = mps::getMPSShape(input_t, memory_format);
    MPSShape* outputShape = mps::getMPSShape(output_t, memory_format);
    MPSNDArray* inputNDArray = nil;
    MPSNDArray* outputNDArray = nil;

    if (input_t.is_contiguous(memory_format) && output_t.is_contiguous(memory_format) && is_macOS_15_0_or_newer) {
      inputNDArray = getMPSNDArray(input_t, inputShape);
      outputNDArray = getMPSNDArray(*output, outputShape);
    }

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSShape* weightShape = mps::getMPSShape(weight_t);
      bool isDepthwiseConv = ((groups > 1 && (weightShape[1].intValue == 1)) && inputShape.count >= 4 &&
                              weightShape.count >= 4 && !is_channels_last);

      MPSGraphTensor* inputTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(input_t.scalar_type()), inputShape);
      MPSGraphTensor* weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_t);
      MPSGraphTensor* outputTensor;
      if (is3DConv) {
        MPSGraphConvolution3DOpDescriptor* conv3dDescriptor_ = [[MPSGraphConvolution3DOpDescriptor new] autorelease];
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        fill_conv3d_desc(conv3dDescriptor_,
                         stride[2],
                         stride[1],
                         stride[0],
                         dilation[2],
                         dilation[1],
                         dilation[0],
                         padding[2],
                         padding[1],
                         padding[0],
                         groups);

        outputTensor = [mpsGraph convolution3DWithSourceTensor:inputTensor
                                                 weightsTensor:weightTensor
                                                    descriptor:conv3dDescriptor_
                                                          name:nil];
      } else if (isDepthwiseConv) {
<<<<<<< HEAD
        auto depthWiseConv3dDescriptor_ = [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
        fill_depthwise_conv_desc(
            depthWiseConv3dDescriptor_, stride[1], stride[0], dilation[1], dilation[0], padding[1], padding[0]);
=======
        MPSGraphDepthwiseConvolution3DOpDescriptor* depthWiseConv3dDescriptor_ =
            [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
        fill_depthwise_conv_desc(depthWiseConv3dDescriptor_,
                                 stride[1],
                                 stride[0],
                                 dilation[1],
                                 dilation[0],
                                 padding[1],
                                 padding[0],
                                 memory_format,
                                 groups);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

        MPSGraphTensor* weightTransposeTensor = [mpsGraph transposeTensor:weightTensor
                                                                dimension:-3
                                                            withDimension:-4
                                                                     name:nil];
        outputTensor = [mpsGraph depthwiseConvolution3DWithSourceTensor:inputTensor
                                                          weightsTensor:weightTransposeTensor
                                                             descriptor:depthWiseConv3dDescriptor_
                                                                   name:nil];
      } else {
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
        fill_conv_desc(conv2dDescriptor_,
                       stride[1],
                       stride[0],
                       dilation[1],
                       dilation[0],
                       padding[1],
                       padding[0],
<<<<<<< HEAD
                       input_suggested_layout,
=======
                       memory_format,
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
                       groups);

        outputTensor = [mpsGraph convolution2DWithSourceTensor:inputTensor
                                                 weightsTensor:weightTensor
                                                    descriptor:conv2dDescriptor_
                                                          name:nil];
      }

      MPSGraphTensor* biasTensor = nil;
      if (bias_defined) {
        biasTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(bias_opt.value()));
<<<<<<< HEAD
=======
      }

      if (is_channels_last && !is_macOS_15_0_or_newer) {
        outputTensor = mps::convertNHWCtoNCHW(mpsGraph, outputTensor);
      }

      if (bias_defined) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        outputTensor = [mpsGraph additionWithPrimaryTensor:outputTensor secondaryTensor:biasTensor name:nil];
      }
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->biasTensor_ = biasTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

<<<<<<< HEAD
    auto inputPlaceholder = input_suggested_layout == kContiguous
        ? Placeholder(cachedGraph->inputTensor_, output_c || is3DConv ? input_t.contiguous() : input_t)
        : Placeholder(cachedGraph->inputTensor_, getMPSNDArray(input_t, inputShape));
    auto outputPlaceholder = input_suggested_layout == kContiguous
        ? Placeholder(cachedGraph->outputTensor_, output_c ? *output_c : output_t)
        : Placeholder(cachedGraph->outputTensor_, getMPSNDArray(output_t, outputShape));
    auto weightsPlaceholder = Placeholder(cachedGraph->weightTensor_, output_c ? weight_t.contiguous() : weight_t);
=======
    auto inputPlaceholder = inputNDArray ? Placeholder(cachedGraph->inputTensor_, inputNDArray)
                                         : Placeholder(cachedGraph->inputTensor_, input_t, inputShape);
    auto weightsPlaceholder = Placeholder(cachedGraph->weightTensor_, weight_t);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    auto biasPlaceholder = Placeholder();
    // Reshape the bias to be broadcastable with output of conv2d or conv3d
    if (bias_defined) {
      if (is3DConv) {
<<<<<<< HEAD
        biasPlaceholder = Placeholder(cachedGraph->biasTensor_, bias_opt->view({1, bias_shape[0], 1, 1, 1}));
      } else if (input_suggested_layout == kChannelsLast) {
        biasPlaceholder = Placeholder(cachedGraph->biasTensor_, bias_opt->view({1, 1, 1, bias_shape[0]}));
      } else {
        biasPlaceholder = Placeholder(cachedGraph->biasTensor_, bias_opt->view({1, bias_shape[0], 1, 1}));
      }
    }

    auto feeds = [[[NSMutableDictionary alloc] initWithCapacity:3] autorelease];
=======
        biasPlaceholder = Placeholder(cachedGraph->biasTensor_, (bias_opt.value()).view({1, bias_shape[0], 1, 1, 1}));
      } else {
        if (is_channels_last && is_macOS_15_0_or_newer) {
          biasPlaceholder = Placeholder(cachedGraph->biasTensor_, (bias_opt.value()).view({1, 1, 1, bias_shape[0]}));
        } else {
          biasPlaceholder = Placeholder(cachedGraph->biasTensor_, (bias_opt.value()).view({1, bias_shape[0], 1, 1}));
        }
      }
    }
    auto outputPlaceholder = outputNDArray ? Placeholder(cachedGraph->outputTensor_, outputNDArray)
                                           : Placeholder(cachedGraph->outputTensor_, *output);

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
        [[[NSMutableDictionary alloc] initWithCapacity:3] autorelease];
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[weightsPlaceholder.getMPSGraphTensor()] = weightsPlaceholder.getMPSGraphTensorData();
    if (bias_defined) {
      feeds[biasPlaceholder.getMPSGraphTensor()] = biasPlaceholder.getMPSGraphTensorData();
    }

    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

<<<<<<< HEAD
  if (output_c) {
    output_t.copy_(*output_c);
  }

  return output_t;
=======
  return *output;
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

Tensor _mps_convolution(const Tensor& input_t,
                        const Tensor& weight_t,
                        const std::optional<Tensor>& bias_opt,
                        IntArrayRef padding,
                        IntArrayRef stride,
                        IntArrayRef dilation,
                        int64_t groups) {
  return _mps_convolution_impl(input_t, weight_t, bias_opt, padding, stride, dilation, groups, std::nullopt);
}

static Tensor mps_convolution_backward_input(IntArrayRef input_size,
                                             const Tensor& grad_output_t,
                                             const Tensor& weight_t,
                                             IntArrayRef padding,
                                             IntArrayRef stride,
                                             IntArrayRef dilation,
                                             int64_t groups,
                                             bool bias_defined) {
  using namespace at::native::mps;
  using namespace mps;
  bool is3DConv = grad_output_t.dim() == 5;
  if (!is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_1_PLUS)) {
    // On macOS < 15.1, MPS convolution kernel does not support output channels > 2^16
    for (auto elem : grad_output_t.sizes()) {
      TORCH_CHECK_NOT_IMPLEMENTED(elem <= (1 << 16), "Output channels > 65536 not supported at the MPS device. ");
    }
  }

  TORCH_CHECK(isFloatingType(grad_output_t.scalar_type()), "Convolution is supported only for Floating types");
  CheckedFrom c = "mps_convolution_backward_input";
  TensorArg grad_output{grad_output_t, "grad_output", 1}, weight{weight_t, "weight", 2};
  checkAllSameType(c, {grad_output, weight});
  checkAllSameGPU(c, {grad_output, weight});
<<<<<<< HEAD
  constexpr auto kChannelsLast = at::MemoryFormat::ChannelsLast;
  bool is_channels_last = mps_conv_use_channels_last(grad_output_t, weight_t) && !is3DConv;
  auto grad_input_t =
      at::empty(input_size, grad_output_t.options(), is_channels_last ? std::optional(kChannelsLast) : std::nullopt);
=======
  auto memory_format = grad_output_t.suggest_memory_format();
  bool is_channels_last = (memory_format == at::MemoryFormat::ChannelsLast) && !is3DConv;
  auto grad_input_t = at::empty(input_size, grad_output_t.options(), std::nullopt);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

  // Avoid "grad_input" when this is being used as transposed convolution
  TensorArg grad_input{grad_input_t, "result", 0};
  convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

<<<<<<< HEAD
  // TODO: Remove me when MacOS-14 is no longer supported
  std::optional<Tensor> grad_input_c;
  if (!is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS) && is_channels_last) {
    grad_input_c = at::empty_like(grad_input_t, grad_input_t.options().memory_format(MemoryFormat::Contiguous));
  }

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* gradInputTensor_ = nil;
  };

  // Add backward with input
  @autoreleasepool {
    MPSStream* stream = getCurrentMPSStream();
<<<<<<< HEAD
    MPSShape* mps_input_shape = getMPSShape(input_size);
    std::string key = fmt::format("mps_{}_convolution_backward_input:{}:{}:{}:{}:{}:{}",
                                  is3DConv ? "3d_" : "",
                                  getArrayRefString(stride),
                                  getArrayRefString(dilation),
                                  getArrayRefString(padding),
                                  groups,
                                  is_channels_last,
                                  getTensorsStringKey({grad_output_t, weight_t}));
=======

    std::string mem_format_key;
    switch (memory_format) {
      case at::MemoryFormat::Contiguous:
        mem_format_key = "Contiguous";
        break;
      case at::MemoryFormat::ChannelsLast:
        mem_format_key = "ChannelsLast";
        break;
      default:
        assert(0 && "Check should have been done earlier\n");
    }

    MPSShape* mps_input_shape = getMPSShape(input_size);
    std::string key;
    if (is3DConv) {
      key = "mps_3d_convolution_backward_input:" + std::to_string(stride[0]) + ":" + std::to_string(stride[1]) + ":" +
          ":" + std::to_string(stride[2]) + std::to_string(dilation[0]) + ":" + std::to_string(dilation[1]) + ":" +
          std::to_string(dilation[2]) + ":" + std::to_string(padding[0]) + ":" + std::to_string(padding[1]) + ":" +
          std::to_string(padding[2]) + ":" + std::to_string(groups) + ":" + mem_format_key +
          getTensorsStringKey({grad_output_t, weight_t});

    } else {
      key = "mps_convolution_backward_input:" + std::to_string(stride[0]) + ":" + std::to_string(stride[1]) + ":" +
          std::to_string(dilation[0]) + ":" + std::to_string(dilation[1]) + ":" + std::to_string(padding[0]) + ":" +
          std::to_string(padding[1]) + ":" + std::to_string(groups) + ":" + mem_format_key +
          getTensorsStringKey({grad_output_t, weight_t});
    }
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output_t);
      auto weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_t);

      MPSGraphTensor* gradInputTensor;
      MPSShape* weightOutputShape = mps::getMPSShape(weight_t);
      // Depthwise conv is input feature channels = groups. So I in OIHW has to be 1.
      bool isDepthwiseConv = ((groups > 1 && (weightOutputShape[1].intValue == 1)) && grad_output_t.ndimension() >= 4 &&
                              weightOutputShape.count >= 4 && !is_channels_last);

      if (is3DConv) {
        MPSGraphConvolution3DOpDescriptor* conv3dDescriptor_ = [[MPSGraphConvolution3DOpDescriptor new] autorelease];
        fill_conv3d_desc(conv3dDescriptor_,
                         stride[2],
                         stride[1],
                         stride[0],
                         dilation[2],
                         dilation[1],
                         dilation[0],
                         padding[2],
                         padding[1],
                         padding[0],
                         groups);
        gradInputTensor = [mpsGraph convolution3DDataGradientWithIncomingGradientTensor:gradOutputTensor
                                                                          weightsTensor:weightTensor
                                                                            outputShape:mps_input_shape
                                                           forwardConvolutionDescriptor:conv3dDescriptor_
                                                                                   name:nil];
      } else if (isDepthwiseConv) {
        MPSGraphDepthwiseConvolution3DOpDescriptor* depthWiseConv3dDescriptor_ =
            [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
<<<<<<< HEAD
        fill_depthwise_conv_desc(
            depthWiseConv3dDescriptor_, stride[1], stride[0], dilation[1], dilation[0], padding[1], padding[0]);
=======
        fill_depthwise_conv_desc(depthWiseConv3dDescriptor_,
                                 stride[1],
                                 stride[0],
                                 dilation[1],
                                 dilation[0],
                                 padding[1],
                                 padding[0],
                                 at::MemoryFormat::Contiguous,
                                 groups);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        MPSGraphTensor* weightTransposeTensor = [mpsGraph transposeTensor:weightTensor
                                                                dimension:-3
                                                            withDimension:-4
                                                                     name:nil];
        gradInputTensor =
            [mpsGraph depthwiseConvolution3DDataGradientWithIncomingGradientTensor:gradOutputTensor
                                                                     weightsTensor:weightTransposeTensor
                                                                       outputShape:mps_input_shape
                                                                        descriptor:depthWiseConv3dDescriptor_
                                                                              name:nil];
      } else {
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
        fill_conv_desc(conv2dDescriptor_,
                       stride[1],
                       stride[0],
                       dilation[1],
                       dilation[0],
                       padding[1],
                       padding[0],
                       at::MemoryFormat::Contiguous,
                       groups);

        gradInputTensor = [mpsGraph convolution2DDataGradientWithIncomingGradientTensor:gradOutputTensor
                                                                          weightsTensor:weightTensor
                                                                            outputShape:mps_input_shape
                                                           forwardConvolutionDescriptor:conv2dDescriptor_
                                                                                   name:nil];
      }

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

<<<<<<< HEAD
    auto gradOutputPlaceholder =
        Placeholder(cachedGraph->gradOutputTensor_, grad_input_c ? grad_output_t.contiguous() : grad_output_t);
    auto weightsPlaceholder = Placeholder(cachedGraph->weightTensor_, grad_input_c ? weight_t.contiguous() : weight_t);
    auto outputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input_c ? *grad_input_c : grad_input_t);
=======
    auto gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output_t);
    auto weightsPlaceholder = Placeholder(cachedGraph->weightTensor_, weight_t);
    auto outputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, *grad_input);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, weightsPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
<<<<<<< HEAD
  if (grad_input_c) {
    grad_input_t.copy_(*grad_input_c);
  }
  return grad_input_t;
=======
  return *grad_input;
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

static Tensor mps_convolution_backward_weights(IntArrayRef weight_size,
                                               const Tensor& grad_output_t,
                                               const Tensor& input_t,
                                               IntArrayRef padding,
                                               IntArrayRef stride,
                                               IntArrayRef dilation,
                                               int64_t groups,
                                               bool bias_defined) {
  using namespace at::native::mps;
  using namespace mps;
<<<<<<< HEAD
  const bool is3DConv = input_t.dim() == 5;
  TORCH_CHECK(isFloatingType(grad_output_t.scalar_type()), "Convolution is supported only for Floating types");
  CheckedFrom c = "mps_convolution_backward_weights";
  constexpr auto kChannelsLast = at::MemoryFormat::ChannelsLast;
  bool is_channels_last = mps_conv_use_channels_last(input_t, grad_output_t) && !is3DConv;
=======
  bool is3DConv = input_t.dim() == 5;
  TORCH_CHECK(isFloatingType(grad_output_t.scalar_type()), "Convolution is supported only for Floating types");
  CheckedFrom c = "mps_convolution_backward_weights";
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_output{grad_output_t, "grad_output", 1};
  TensorArg input{input_t, "input", 2};

  checkAllSameType(c, {grad_output, input});
  checkAllSameGPU(c, {grad_output, input});

  auto grad_weight_t =
<<<<<<< HEAD
      at::empty(weight_size, grad_output_t.options(), is_channels_last ? std::optional(kChannelsLast) : std::nullopt);

=======
      at::empty(weight_size, grad_output_t.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  TensorArg grad_weight{grad_weight_t, "result", 0};

  convolution_shape_check(c, input, grad_weight, grad_output, padding, stride, dilation, groups);

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* gradWeightTensor_ = nil;
  };

<<<<<<< HEAD
  // TODO: Remove me when MacOS-14 is no longer supported
  std::optional<Tensor> grad_weight_c;
  if (!is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS) && is_channels_last) {
    grad_weight_c = at::empty_like(grad_weight_t, grad_weight_t.options().memory_format(MemoryFormat::Contiguous));
  }

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  @autoreleasepool {
    MPSStream* stream = getCurrentMPSStream();

    MPSShape* mps_weight_shape = getMPSShape(weight_size);
<<<<<<< HEAD
    std::string key = fmt::format("mps_{}convolution_backward_weights:{}:{}:{}:{}:{}:{}",
                                  is3DConv ? "3d_" : "",
                                  getArrayRefString(stride),
                                  getArrayRefString(dilation),
                                  getArrayRefString(padding),
                                  groups,
                                  is_channels_last,
                                  getTensorsStringKey({grad_output_t, input_t, grad_weight_t}));
=======
    std::string key;
    if (is3DConv) {
      key = "mps_3d_convolution_backward_weights:" + std::to_string(stride[0]) + ":" + std::to_string(stride[1]) + ":" +
          std::to_string(stride[2]) + ":" + std::to_string(dilation[0]) + ":" + std::to_string(dilation[1]) + ":" +
          std::to_string(dilation[2]) + ":" + std::to_string(padding[0]) + ":" + std::to_string(padding[1]) + ":" +
          std::to_string(padding[2]) + ":" + std::to_string(groups) + ":" +
          getTensorsStringKey({grad_output_t, input_t, grad_weight_t});
    } else {
      key = "mps_convolution_backward_weights:" + std::to_string(stride[0]) + ":" + std::to_string(stride[1]) + ":" +
          std::to_string(dilation[0]) + ":" + std::to_string(dilation[1]) + ":" + std::to_string(padding[0]) + ":" +
          std::to_string(padding[1]) + ":" + std::to_string(groups) + ":" +
          getTensorsStringKey({grad_output_t, input_t, grad_weight_t});
    }
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSShape* inputShape = getMPSShape(input_t);
      bool isDepthwiseConv =
          ((groups > 1 && (mps_weight_shape[1].intValue == 1)) && inputShape.count >= 4 && mps_weight_shape.count >= 4);

      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output_t);
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_t);

      MPSGraphTensor* gradWeightTensor;
      if (is3DConv) {
        MPSGraphConvolution3DOpDescriptor* conv3dDescriptor_ = [[MPSGraphConvolution3DOpDescriptor new] autorelease];
        fill_conv3d_desc(conv3dDescriptor_,
                         stride[2],
                         stride[1],
                         stride[0],
                         dilation[2],
                         dilation[1],
                         dilation[0],
                         padding[2],
                         padding[1],
                         padding[0],
                         groups);
        gradWeightTensor = [mpsGraph convolution3DWeightsGradientWithIncomingGradientTensor:gradOutputTensor
                                                                               sourceTensor:inputTensor
                                                                                outputShape:mps_weight_shape
                                                               forwardConvolutionDescriptor:conv3dDescriptor_
                                                                                       name:nil];
      } else if (isDepthwiseConv) {
        MPSGraphDepthwiseConvolution3DOpDescriptor* depthWiseConv3dDescriptor_ =
            [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
<<<<<<< HEAD
        fill_depthwise_conv_desc(
            depthWiseConv3dDescriptor_, stride[1], stride[0], dilation[1], dilation[0], padding[1], padding[0]);
=======
        fill_depthwise_conv_desc(depthWiseConv3dDescriptor_,
                                 stride[1],
                                 stride[0],
                                 dilation[1],
                                 dilation[0],
                                 padding[1],
                                 padding[0],
                                 at::MemoryFormat::Contiguous,
                                 groups);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        NSNumber* outputFeatChannelDim = mps_weight_shape[0];
        MPSShape* weightShapeTranspose = @[ @1, outputFeatChannelDim, mps_weight_shape[2], mps_weight_shape[3] ];
        MPSGraphTensor* gradWeightTensorTranspose =
            [mpsGraph depthwiseConvolution3DWeightsGradientWithIncomingGradientTensor:gradOutputTensor
                                                                         sourceTensor:inputTensor
                                                                          outputShape:weightShapeTranspose
                                                                           descriptor:depthWiseConv3dDescriptor_
                                                                                 name:nil];
        gradWeightTensor = [mpsGraph transposeTensor:gradWeightTensorTranspose dimension:-3 withDimension:-4 name:nil];
      } else {
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
        fill_conv_desc(conv2dDescriptor_,
                       stride[1],
                       stride[0],
                       dilation[1],
                       dilation[0],
                       padding[1],
                       padding[0],
                       at::MemoryFormat::Contiguous,
                       groups);

        gradWeightTensor = [mpsGraph convolution2DWeightsGradientWithIncomingGradientTensor:gradOutputTensor
                                                                               sourceTensor:inputTensor
                                                                                outputShape:mps_weight_shape
                                                               forwardConvolutionDescriptor:conv2dDescriptor_
                                                                                       name:nil];
      }

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradWeightTensor_ = gradWeightTensor;
    });

<<<<<<< HEAD
    auto gradOutputPlaceholder =
        Placeholder(cachedGraph->gradOutputTensor_, grad_weight_c ? grad_output_t.contiguous() : grad_output_t);
    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, grad_weight_c ? input_t.contiguous() : input_t);
    auto outputPlaceholder =
        Placeholder(cachedGraph->gradWeightTensor_, grad_weight_c ? *grad_weight_c : grad_weight_t);
=======
    auto gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output_t);
    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = Placeholder(cachedGraph->gradWeightTensor_, grad_weight_t);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, inputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

<<<<<<< HEAD
  if (grad_weight_c) {
    grad_weight_t.copy_(*grad_weight_c);
  }
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  return grad_weight_t;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mps_convolution_backward(const at::Tensor& input,
                                                                        const at::Tensor& grad_output,
                                                                        const at::Tensor& weight,
                                                                        IntArrayRef padding,
                                                                        IntArrayRef stride,
                                                                        IntArrayRef dilation,
                                                                        int64_t groups,
                                                                        std::array<bool, 3> output_mask) {
  Tensor grad_input, grad_weight, grad_bias;
  if (input.numel() == 0) {
    if (output_mask[0]) {
      grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (output_mask[1]) {
      grad_weight = at::zeros_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
  } else {
    if (output_mask[0]) {
      grad_input = mps_convolution_backward_input(
          input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
    }
    if (output_mask[1]) {
      grad_weight = mps_convolution_backward_weights(
          weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2]);
    }
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

static Tensor mps_convolution_transpose_forward(const Tensor& grad_output,
                                                const Tensor& weight,
                                                IntArrayRef padding,
                                                IntArrayRef output_padding,
                                                IntArrayRef stride,
                                                IntArrayRef dilation,
                                                int64_t groups) {
  auto input_size =
      conv_input_size(grad_output.sizes(), weight.sizes(), padding, output_padding, stride, dilation, groups);
  return mps_convolution_backward_input(input_size, grad_output, weight, padding, stride, dilation, groups, false);
}

Tensor _mps_convolution_transpose(const Tensor& input_t,
                                  const Tensor& weight_t,
                                  IntArrayRef padding,
                                  IntArrayRef output_padding,
                                  IntArrayRef stride,
                                  IntArrayRef dilation,
                                  int64_t groups) {
  bool is_unsupported_3d_dtype =
      (input_t.dim() == 5 && (input_t.scalar_type() == kHalf || input_t.scalar_type() == kBFloat16));
  TORCH_CHECK(!is_unsupported_3d_dtype, "ConvTranspose 3D with BF16 or FP16 types is not supported on MPS");

  auto output_t =
      mps_convolution_transpose_forward(input_t, weight_t, padding, output_padding, stride, dilation, groups);
  return output_t;
}

static Tensor mps_convolution_transpose_backward_input(const Tensor& grad_output_t,
                                                       const Tensor& weight_t,
                                                       IntArrayRef padding,
                                                       IntArrayRef stride,
                                                       IntArrayRef dilation,
                                                       int64_t groups,
                                                       IntArrayRef input_shape) {
  return _mps_convolution_impl(grad_output_t, weight_t, std::nullopt, padding, stride, dilation, groups, input_shape);
}

static Tensor mps_convolution_transpose_backward_weight(IntArrayRef weight_size,
                                                        const Tensor& grad_output_t,
                                                        const Tensor& input_t,
                                                        IntArrayRef padding,
                                                        IntArrayRef stride,
                                                        IntArrayRef dilation,
                                                        int64_t groups) {
  return mps_convolution_backward_weights(
      weight_size, input_t, grad_output_t, padding, stride, dilation, groups, false);
}

std::tuple<Tensor, Tensor> mps_convolution_transpose_backward(const Tensor& input,
                                                              const Tensor& grad_output,
                                                              const Tensor& weight,
                                                              IntArrayRef padding,
                                                              IntArrayRef output_padding,
                                                              IntArrayRef stride,
                                                              IntArrayRef dilation,
                                                              int64_t groups,
                                                              std::array<bool, 2> output_mask) {
  Tensor grad_input, grad_weight;
  if (output_mask[0]) {
    grad_input =
        mps_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, input.sizes());
  }
  if (output_mask[1]) {
    grad_weight = mps_convolution_transpose_backward_weight(
        weight.sizes(), grad_output, input, padding, stride, dilation, groups);
  }

  return std::tuple<Tensor, Tensor>{grad_input, grad_weight};
}

} // namespace at::native

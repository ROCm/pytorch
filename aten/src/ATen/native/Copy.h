#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/core/Device.h>
#include <c10/util/Optional.h>

namespace at {

class Tensor;
struct TensorIterator;
class TensorBase;

namespace native {

using copy_fn = void (*)(TensorIterator&, bool non_blocking);
using move_fn = void (*)(TensorIterator&, c10::optional<Device> dst_device, bool non_blocking);

DECLARE_DISPATCH(copy_fn, copy_stub);
DECLARE_DISPATCH(move_fn, move_stub);

TORCH_API void copy_ignoring_overlaps(const TensorBase &dst, const TensorBase &src);

} // namespace native
} // namespace at

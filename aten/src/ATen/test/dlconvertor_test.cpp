#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>

<<<<<<< HEAD
using namespace at;

=======
#include <iostream>
// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <string.h>
#include <sstream>

using namespace at;
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
TEST(TestDlconvertor, TestDlconvertor) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensor* dlMTensor = toDLPack(a);

  Tensor b = fromDLPack(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertor, TestDlconvertorNoStrides) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensor* dlMTensor = toDLPack(a);
  dlMTensor->dl_tensor.strides = nullptr;

  Tensor b = fromDLPack(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}
<<<<<<< HEAD

TEST(TestDlconvertorUnversioned, TestDlconvertor) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensorVersioned* dlMTensor = toDLPackVersioned(a);

  Tensor b = fromDLPackVersioned(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertorUnversioned, TestDlconvertorNoStrides) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensorVersioned* dlMTensor = toDLPackVersioned(a);
  dlMTensor->dl_tensor.strides = nullptr;

  Tensor b = fromDLPackVersioned(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

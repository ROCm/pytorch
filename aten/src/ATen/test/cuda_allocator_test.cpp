#include <gtest/gtest.h>

#include <ATen/ATen.h>
<<<<<<< HEAD
#include <ATen/cuda/CUDAContext.h>
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/test/allocator_clone_test.h>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

<<<<<<< HEAD
std::unordered_map<void*, size_t> allocation_sizes;

void* logging_malloc(size_t size, int device, cudaStream_t stream) {
    void* ptr;
    cudaMalloc(&ptr, size);
    allocation_sizes[ptr] = size;
    return ptr;
}

void logging_free(void* ptr, size_t size, int device, cudaStream_t stream) {
    if (allocation_sizes.find(ptr) != allocation_sizes.end()) {
        if (allocation_sizes[ptr] != size) {
          throw std::runtime_error("free mismatch");
        }
    } else {
      throw std::runtime_error("free of unknown ptr");
    }
    cudaFree(ptr);
    allocation_sizes.erase(ptr);
}

TEST(TestTorchUnique, UniqueComparisonTest) {
  if (!at::cuda::is_available()) return;
  auto custom_allocator =
      torch::cuda::CUDAPluggableAllocator::createCustomAllocator(logging_malloc, logging_free);
  torch::cuda::CUDAPluggableAllocator::changeCurrentAllocator(custom_allocator);
  // Run the command 3 times; the first 2 will pass and the third invocation will have
  // different sizes in alloc and free if the test fails.
  for (int i = 0; i < 3; ++i) {
    // Initialize simple sorted tensor with repeats
    at::Tensor sorted_tensor =
        at::tensor({0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 5},
                      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

    // This operation will call malloc/free with different sizes on the same pointer
    auto unique_dim_result = at::unique_consecutive(sorted_tensor, false, true, 0);

    // Everything below is only there to validate correct results
    auto unique_dim_values = std::get<0>(unique_dim_result);
    auto unique_dim_counts = std::get<2>(unique_dim_result);

    // Check tensor sizes
    EXPECT_EQ(unique_dim_values.size(0), 5);
    EXPECT_EQ(unique_dim_counts.size(0), 5);

    // Copy to CPU before accessing elements
    at::Tensor cpu_values = unique_dim_values.cpu();
    at::Tensor cpu_counts = unique_dim_counts.cpu();

    // Use accessors on the CPU tensors
    auto values_accessor = cpu_values.accessor<float, 1>();
    auto counts_accessor = cpu_counts.accessor<int64_t, 1>();

    // Check individual values using accessors
    EXPECT_EQ(values_accessor[0], 0.0f);
    EXPECT_EQ(values_accessor[1], 1.0f);
    EXPECT_EQ(values_accessor[2], 2.0f);
    EXPECT_EQ(values_accessor[3], 3.0f);
    EXPECT_EQ(values_accessor[4], 5.0f);

    // Check count values using accessors
    EXPECT_EQ(counts_accessor[0], 3);
    EXPECT_EQ(counts_accessor[1], 2);
    EXPECT_EQ(counts_accessor[2], 1);
    EXPECT_EQ(counts_accessor[3], 4);
    EXPECT_EQ(counts_accessor[4], 1);
  }
}

TEST(AllocatorTestCUDA, test_clone) {
  if (!at::cuda::is_available()) return;
  test_allocator_clone(c10::cuda::CUDACachingAllocator::get());
}
=======
TEST(AllocatorTestCUDA, test_clone) {
  test_allocator_clone(c10::cuda::CUDACachingAllocator::get());
}

static int called_dummy_free_0 = 0;
static int called_dummy_free_1 = 0;

void* dummy_alloc_0(size_t size, int device, void* stream) {return nullptr;}
void dummy_free_0(void* data, size_t size, int device, void* stream) {
  called_dummy_free_0++;
}
void dummy_free_1(void* data, size_t size, int device, void* stream) {
  called_dummy_free_1++;
}

// Tests that data_ptrs have their respective deleters
// when mixing allocators
TEST(AllocatorTestCUDA, test_pluggable_allocator_deleters) {
  // Create a tensor with dummy_allocator_0, where dummy_free_0 is the deleter
  auto dummy_allocator_0 = torch::cuda::CUDAPluggableAllocator::createCustomAllocator(dummy_alloc_0, dummy_free_0);
  c10::cuda::CUDACachingAllocator::allocator.store(dummy_allocator_0.get());
  at::Tensor a = at::empty({0}, at::TensorOptions().device(at::kCUDA));

  // Create a tensor with dummy_allocator_1, where dummy_free_1 is the deleter
  auto dummy_allocator_1 = torch::cuda::CUDAPluggableAllocator::createCustomAllocator(dummy_alloc_0, dummy_free_1);
  c10::cuda::CUDACachingAllocator::allocator.store(dummy_allocator_1.get());
  at::Tensor b = at::empty({0}, at::TensorOptions().device(at::kCUDA));

  // Manually use a's deleter
  auto* ctx = a.storage().data_ptr().get_context();
  a.storage().data_ptr().get_deleter()(ctx);
  a.storage().mutable_data_ptr().release_context();

  // a's deleter is dummy_free_0
  // dummy_free_0 should be called above, so called_dummy_free_0 should be 1
  ASSERT_TRUE(called_dummy_free_0 == 1);

  // Manually use b's deleter
  ctx = b.storage().data_ptr().get_context();
  b.storage().data_ptr().get_deleter()(ctx);
  b.storage().mutable_data_ptr().release_context();

  // b's deleter is dummy_free_1
  // dummy_free_1 should be called above, so called_dummy_free_1 should be 1
  ASSERT_TRUE(called_dummy_free_1 == 1);
}
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

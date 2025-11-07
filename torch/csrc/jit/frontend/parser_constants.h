#pragma once

namespace torch::jit {
<<<<<<< HEAD
static const char* valid_single_char_tokens = "+-*/%@()[]:,={}><.?!&^|~";
=======
static constexpr const char* valid_single_char_tokens =
    "+-*/%@()[]:,={}><.?!&^|~";
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
} // namespace torch::jit

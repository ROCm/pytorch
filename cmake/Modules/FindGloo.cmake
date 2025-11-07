# Try to find the Gloo library and headers.
#  Gloo_FOUND        - system has Gloo lib
#  Gloo_INCLUDE_DIRS - the Gloo include directory
<<<<<<< HEAD
#  Gloo_NATIVE_LIBRARY - base gloo library, needs to be linked
#  Gloo_CUDA_LIBRARY/Gloo_HIP_LIBRARY - CUDA/HIP support library in Gloo
=======
#  Gloo_LIBRARY/Gloo_NATIVE_LIBRARY    - libraries needed to use Gloo
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

find_path(Gloo_INCLUDE_DIR
  NAMES gloo/common/common.h
  DOC "The directory where Gloo includes reside"
)

find_library(Gloo_NATIVE_LIBRARY
  NAMES gloo
<<<<<<< HEAD
  DOC "The Gloo library"
)

# Gloo has optional CUDA support
# if Gloo + CUDA is desired, Gloo_CUDA_LIBRARY
# needs to be linked into desired target
find_library(Gloo_CUDA_LIBRARY
  NAMES gloo_cuda
  DOC "Gloo's CUDA support/code"
)

# Gloo has optional HIP support
# if Gloo + HIP is desired, Gloo_HIP_LIBRARY
# needs to be linked to desired target
find_library(Gloo_HIP_LIBRARY
  NAMES gloo_hiop
  DOC "Gloo's HIP support/code"
=======
  DOC "The Gloo library (without CUDA)"
)

find_library(Gloo_CUDA_LIBRARY
  NAMES gloo_cuda
  DOC "The Gloo library (with CUDA)"
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
)

set(Gloo_INCLUDE_DIRS ${Gloo_INCLUDE_DIR})

<<<<<<< HEAD
=======
# use the CUDA library depending on the Gloo_USE_CUDA variable
if (DEFINED Gloo_USE_CUDA)
  if (${Gloo_USE_CUDA})
    set(Gloo_LIBRARY ${Gloo_CUDA_LIBRARY})
    set(Gloo_NATIVE_LIBRARY ${Gloo_NATIVE_LIBRARY})
  else()
    set(Gloo_LIBRARY ${Gloo_NATIVE_LIBRARY})
    set(Gloo_NATIVE_LIBRARY ${Gloo_NATIVE_LIBRARY})
  endif()
else()
  # else try to use the CUDA library if found
  if (${Gloo_CUDA_LIBRARY} STREQUAL "Gloo_CUDA_LIBRARY-NOTFOUND")
    set(Gloo_LIBRARY ${Gloo_NATIVE_LIBRARY})
    set(Gloo_NATIVE_LIBRARY ${Gloo_NATIVE_LIBRARY})
  else()
    set(Gloo_LIBRARY ${Gloo_CUDA_LIBRARY})
    set(Gloo_NATIVE_LIBRARY ${Gloo_NATIVE_LIBRARY})
  endif()
endif()
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gloo
  FOUND_VAR Gloo_FOUND
<<<<<<< HEAD
  REQUIRED_VARS Gloo_INCLUDE_DIR Gloo_NATIVE_LIBRARY
=======
  REQUIRED_VARS Gloo_INCLUDE_DIR Gloo_LIBRARY
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
)

mark_as_advanced(Gloo_FOUND)

#
# create INTERFACE target for CK library
#
if(NOT __CK_KERNELS_INCLUDED)
  set(__CK_KERNELS_INCLUDED TRUE)

  set(CK_KERNELS_VERSION 0.1)

  # get CK commit hash
  execute_process(
      COMMAND git -C ${CMAKE_SOURCE_DIR}/third_party submodule status composable_kernel
      RESULT_VARIABLE result
      OUTPUT_VARIABLE submodule_status
      ERROR_VARIABLE submodule_status_error
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
  if(result EQUAL 0)
      string(REGEX REPLACE "^[ \t]" "" submodule_status ${submodule_status})
      # extract first 8 characters of the commit hash
      string(SUBSTRING "${submodule_status}" 0 8 ck_commit_hash)
  else()
      message(FATAL_ERROR "Failed to get submodule status for composable_kernel.")
  endif()
  
  # full path for CK library on compute-artifactory.amd.com
  # set(CK_KERNELS_PACKAGE_BASE_URL $ENV{CK_KERNELS_PACKAGE_BASE_URL})
  set(CK_KERNELS_PACKAGE_BASE_URL "https://compute-artifactory.amd.com/artifactory/rocm-generic-local")
  if (NOT CK_KERNELS_PACKAGE_BASE_URL)
      message("CK_KERNELS_PACKAGE_BASE_URL env var is not defined. Not installing prebuilt CK_kernels.")
      return()
  else()
      set(ck_lib_full_path "${CK_KERNELS_PACKAGE_BASE_URL}/torch_ck_gen_lib/ck_${ck_commit_hash}/rocm_${ROCM_VERSION_DEV}/libck_kernels.so")
  endif() 
  
  # set destination
  set(destination "${CMAKE_SOURCE_DIR}/torch/lib/libck_kernels.so")
  
  # download CK library
  file(DOWNLOAD ${ck_lib_full_path} ${destination} SHOW_PROGRESS RESULT_VARIABLE download_status)
  if(NOT download_status)
      message(STATUS "Downloaded CK library successfully.")
  else()
      message(FATAL_ERROR "Failed to download the CK library from ${SOURCE_URL}.")
  endif()
  
  # create INTERFACE target
  add_library(__ck_lib INTERFACE)
  
  # specify path to CK library
  target_link_libraries(__ck_lib INTERFACE ${destination})
endif() # __CK_KERNELS_INCLUDED

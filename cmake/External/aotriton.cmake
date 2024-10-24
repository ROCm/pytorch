if(NOT __AOTRITON_INCLUDED)
  set(__AOTRITON_INCLUDED TRUE)

  set(__AOTRITON_EXTERN_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/aotriton")
  set(__AOTRITON_INSTALL_DIR "${PROJECT_SOURCE_DIR}/torch")
  add_library(__caffe2_aotriton INTERFACE)

  # AOTriton package information from GitHub Release Pages
  # Replaces .ci/docker/aotriton_version.txt
  set(__AOTRITON_VER "0.7.1b")
  set(__AOTRITON_MANYLINUX_LIST
      "manylinux_2_17"  # rocm6.2
      "manylinux_2_28"  # rocm6.3
     )
  set(__AOTRITON_ROCM_LIST
      "rocm6.2"
      "rocm6.3"
     )
  message("JN: ${__AOTRITON_ROCM_LIST}")
  set(__AOTRITON_CI_COMMIT "99f540a954e80f446ec6980f108e8c25408f1823")
  set(__AOTRITON_SHA256_LIST
      "df00412ae36fe5732d0a4601802bd3622b5dec12df7ec86027c5147adeb54c25"  # rocm6.2
      "e4e3b06d2431e68e0096fcc8d3668cd5034ca0fd6fe236fb3b96774427d934b8"  # rocm6.3
     )
  set(__AOTRITON_Z "gz")

  # Note it is INSTALL"ED"
  if(DEFINED ENV{AOTRITON_INSTALLED_PREFIX})
    install(DIRECTORY
            $ENV{AOTRITON_INSTALLED_PREFIX}/lib
            $ENV{AOTRITON_INSTALLED_PREFIX}/include
            DESTINATION ${__AOTRITON_INSTALL_DIR})
    set(__AOTRITON_INSTALL_DIR "$ENV{AOTRITON_INSTALLED_PREFIX}")
    message(STATUS "Using Preinstalled AOTriton at ${__AOTRITON_INSTALL_DIR}")
  else()
    list(GET __AOTRITON_ROCM_LIST 0 __AOTRITON_ROCM_LOW_STR)
    list(GET __AOTRITON_ROCM_LIST -1 __AOTRITON_ROCM_HIGH_STR)
    # len("rocm") == 4
    string(SUBSTRING ${__AOTRITON_ROCM_LOW_STR} 4 -1 __AOTRITON_ROCM_LOW)
    string(SUBSTRING ${__AOTRITON_ROCM_HIGH_STR} 4 -1 __AOTRITON_ROCM_HIGH)
    set(__AOTRITON_SYSTEM_ROCM "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}")
    if(__AOTRITON_SYSTEM_ROCM VERSION_LESS __AOTRITON_ROCM_LOW)
      set(__AOTRITON_ROCM ${__AOTRITON_ROCM_LOW})
    elseif(__AOTRITON_SYSTEM_ROCM VERSION_GREATER __AOTRITON_ROCM_HIGH)
      set(__AOTRITON_ROCM ${__AOTRITON_ROCM_HIGH})
    else()
      set(__AOTRITON_ROCM ${__AOTRITON_SYSTEM_ROCM})
    endif()
    list(FIND __AOTRITON_ROCM_LIST "rocm${__AOTRITON_ROCM}" __AOTRITON_ROCM_INDEX)
    list(GET __AOTRITON_SHA256_LIST ${__AOTRITON_ROCM_INDEX} __AOTRITON_SHA256)
    list(GET __AOTRITON_MANYLINUX_LIST ${__AOTRITON_ROCM_INDEX} __AOTRITON_MANYLINUX)
    set(__AOTRITON_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
    string(CONCAT __AOTRITON_FILE "aotriton-"
                                  "${__AOTRITON_VER}-${__AOTRITON_MANYLINUX}"
                                  "_${__AOTRITON_ARCH}-rocm${__AOTRITON_ROCM}"
                                  "-shared.tar.${__AOTRITON_Z}")
    string(CONCAT __AOTRITON_URL "https://github.com/ROCm/aotriton/releases/download/"
                                 "${__AOTRITON_VER}/${__AOTRITON_FILE}")
    ExternalProject_Add(aotriton_external
      URL "${__AOTRITON_URL}"
      URL_HASH SHA256=${__AOTRITON_SHA256}
      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton_tarball
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/aotriton_tarball"
      "${__AOTRITON_INSTALL_DIR}"
      BUILD_BYPRODUCTS "${__AOTRITON_INSTALL_DIR}/lib/libaotriton_v2.so"
    )
    add_dependencies(__caffe2_aotriton aotriton_external)
    message(STATUS "Using AOTriton from pre-compiled binary ${__AOTRITON_URL}")
  endif()
  target_link_libraries(__caffe2_aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/lib/libaotriton_v2.so)
  target_include_directories(__caffe2_aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/include)
  set(AOTRITON_FOUND TRUE)
endif() # __AOTRITON_INCLUDED

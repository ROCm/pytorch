if(NOT __AOTRITON_INCLUDED)
  set(__AOTRITON_INCLUDED TRUE)

  set(__AOTRITON_EXTERN_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/aotriton")
  set(__AOTRITON_INSTALL_DIR "${PROJECT_SOURCE_DIR}/torch")
  add_library(__caffe2_aotriton INTERFACE)

  # AOTriton package information from GitHub Release Pages
  # Replaces .ci/docker/aotriton_version.txt
  # Note packages information may have versions skipped (due to no ABI breaks)
  # But they must be listed from lower version to higher version
  set(__AOTRITON_VER "0.14.50tp")
  set(__AOTRITON_MANYLINUX_LIST
      "manylinux_2_28"  # rocm6.4
      "manylinux_2_28"  # rocm7.0
      "manylinux_2_28"  # rocm7.1
      "manylinux_2_28"  # rocm7.2
      "manylinux_2_28"  # rocm7.14
      "manylinux_2_28"  # rocm7.15
      "manylinux_2_28"  # rocm7.16
      )
  set(__AOTRITON_ROCM_LIST
      "rocm6.4"
      "rocm7.0"
      "rocm7.1"
      "rocm7.2"
      "rocm7.14"
      "rocm7.15"
      "rocm7.16"
      )
  if(DEFINED ENV{PYTORCH_AOTRITON_COMMIT})
    set(__AOTRITON_CI_COMMIT "$ENV{PYTORCH_AOTRITON_COMMIT}")
  else()
    set(__AOTRITON_CI_COMMIT "7a6963a7265d2cf0fc48c6a63464386134d32483")
  endif()
  set(__AOTRITON_SHA256_LIST
      "49352b21e7c6cb029fa79e500adb583619227d1c9f856e3b5d41daf020475f5d"  # rocm6.4
      "6bfc5a951c3b8d415cd14947f51506e18490eaba1eb6e9fb14c29f232487f801"  # rocm7.0
      "0193395680a4b6d68151ca9093c1cdb91d169906d13f478924466eb328550647"  # rocm7.1
      "732de2de06889f80284c18e7be4bc45905b0951f39885b07062309667cfa6d96"  # rocm7.2
      "df3ecc66faccc76e33ace6b7140edeb28fa3a4667d3f073d1efdb0e147fbd37d"  # rocm7.14
      "537c329217bb7d69c7b7827fc513380e0814432ebe46219077b83c33e7344945"  # rocm7.15
      "8dc955b0c1d65b496211805ee3ea82a93afa54bd69f9a1aed212752baf196105"  # rocm7.16
      )
  set(__AOTRITON_IMAGE_LIST
      "amd-gfx90a"
      "amd-gfx942"
      "amd-gfx950"
      "amd-gfx110x"
      "amd-gfx115x"
      "amd-gfx120x"
      "amd-gfx1250"
     )
  # gfx1250-strict is not a separate AOTriton image; the amd-gfx1250 regex
  # also matches gfx1250-strict in PYTORCH_ROCM_ARCH.
  set(__AOTRITON_IMAGE_SHA256_LIST
     "7cc293803aa73bd223554d51a77fab50fd37b0bdb9d5e874f7e2997f4390b9c5" # amd-gfx90a
     "eae5d36c974d974f085cfbdecf77760db71324fa9814ed50d8a3fc42bfbe3e41" # amd-gfx942
     "23716cbf855df9df1ca7d480a90e0fb29b0356bb424015c426fc0e5a973f682b" # amd-gfx950
     "a54fecf9b4e16425681785eba475912eb9ae761e35a4df3e3579bdbd7c6fe4db" # amd-gfx110x
     "e4bda76fd6857a733cd7b4b3eb9ccb21c0f47b565ce304777a61bbb13d12ad6e" # amd-gfx115x
     "f173671cc5112897e134f1d209bd6f572ecd507f4d68dc84cb6f07b67e13e851" # amd-gfx120x
     "5e01ab32229ad39568071855e05a132771940648821a542601031c4f8718c1a8" # amd-gfx1250
     )
  set(__AOTRITON_BASE_URL "$ENV{PYTORCH_AOTRITON_BASE_URL}")
  if(NOT __AOTRITON_BASE_URL)
    set(__AOTRITON_BASE_URL "https://github.com/ROCm/aotriton/releases/download/")  # @lint-ignore
  endif()
  set(__AOTRITON_Z "gz")
  # Set the default __AOTRITON_LIB path
  if(NOT WIN32)
    set(__AOTRITON_LIB "lib/libaotriton_v2.so")
  else()
    set(__AOTRITON_LIB "lib/aotriton_v2.lib")
  endif()

  function(aotriton_build_windows_dependencies dlfcn-win32_external xz_external dlfcn-win32_DIR liblzma_DIR)
    # Windows-specific dependencies - build these first
    if(NOT noimage)
      message(FATAL_ERROR "noimage must be ON for Windows builds")
    endif()
    # Build dlfcn-win32
    set(__DLFCN_WIN32_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/dlfcn-win32")
    set(__DLFCN_WIN32_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/dlfcn-win32-install")

    ExternalProject_Add(${dlfcn-win32_external}
      GIT_REPOSITORY https://github.com/dlfcn-win32/dlfcn-win32.git
      GIT_TAG v1.4.2
      PREFIX ${__DLFCN_WIN32_PREFIX}
      INSTALL_DIR ${__DLFCN_WIN32_INSTALL_DIR}
      CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${__DLFCN_WIN32_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=cl
        -DCMAKE_CXX_COMPILER=cl
        -DBUILD_SHARED_LIBS=ON
        -DBUILD_TESTS=OFF
      BUILD_BYPRODUCTS
        "${__DLFCN_WIN32_INSTALL_DIR}/lib/dl.lib"
        "${__DLFCN_WIN32_INSTALL_DIR}/bin/dl.dll"
    )
    ExternalProject_Add_Step(${dlfcn-win32_external} copy_to_aotriton
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${__DLFCN_WIN32_INSTALL_DIR}/bin/dl.dll"
        "${__AOTRITON_INSTALL_DIR}/lib/"
      DEPENDEES install
    )
    set(${dlfcn-win32_DIR} "${__DLFCN_WIN32_INSTALL_DIR}/share/dlfcn-win32" CACHE PATH "Path to dlfcn-win32 CMake config" FORCE)

    # Build xz/liblzma
    set(__XZ_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/xz")
    set(__XZ_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/xz-install")

    ExternalProject_Add(${xz_external}
      GIT_REPOSITORY https://github.com/tukaani-project/xz.git
      GIT_TAG v5.8.1
      PREFIX ${__XZ_PREFIX}
      INSTALL_DIR ${__XZ_INSTALL_DIR}
      CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${__XZ_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=ON
        -DENABLE_NLS=OFF
        -DXZ_TOOL_LZMAINFO=OFF
        -DXZ_TOOL_XZ=OFF
        -DXZ_TOOL_XZDEC=OFF
        -DXZ_TOOL_LZMADEC=OFF
      BUILD_BYPRODUCTS
        "${__XZ_INSTALL_DIR}/lib/lzma.lib"
        "${__XZ_INSTALL_DIR}/bin/liblzma.dll"
    )
    ExternalProject_Add_Step(${xz_external} copy_to_aotriton
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${__XZ_INSTALL_DIR}/bin/liblzma.dll"
        "${__AOTRITON_INSTALL_DIR}/lib/"
      DEPENDEES install
    )
    set(${liblzma_DIR} "${__XZ_INSTALL_DIR}/lib/cmake/liblzma" CACHE PATH "Path to xz/liblzma CMake config" FORCE)
  endfunction()

  function(aotriton_build_from_source noimage project)
    if(noimage)
      SET(RECURSIVE "OFF")
    else()
      SET(RECURSIVE "ON")
    endif()
    if(WIN32)
      message(STATUS "Building AOTriton Windows dependencies")
      aotriton_build_windows_dependencies(dlfcn-win32_external xz_external dlfcn-win32_DIR liblzma_DIR)
    endif()
    message(STATUS "PYTORCH_ROCM_ARCH ${PYTORCH_ROCM_ARCH}")

    ExternalProject_Add(${project}
      GIT_REPOSITORY https://github.com/ROCm/aotriton.git
      GIT_SUBMODULES_RECURSE ${RECURSIVE}
      GIT_TAG ${__AOTRITON_CI_COMMIT}
      PREFIX ${__AOTRITON_EXTERN_PREFIX}
      CMAKE_CACHE_ARGS
      -DAOTRITON_TARGET_ARCH:STRING=${PYTORCH_ROCM_ARCH}
      -DCMAKE_INSTALL_PREFIX:FILEPATH=${__AOTRITON_INSTALL_DIR}
      CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DAOTRITON_GPU_BUILD_TIMEOUT=0
      -DAOTRITON_NO_PYTHON=ON
      -DAOTRITON_NOIMAGE_MODE=${noimage}
      -DHIP_PLATFORM=amd
      $<$<BOOL:${WIN32}>:-Ddlfcn-win32_DIR=${dlfcn-win32_DIR}>
      $<$<BOOL:${WIN32}>:-Dliblzma_DIR=${liblzma_DIR}>
      BUILD_BYPRODUCTS "${__AOTRITON_INSTALL_DIR}/${__AOTRITON_LIB}"
      USES_TERMINAL_DOWNLOAD TRUE
      USES_TERMINAL_CONFIGURE TRUE
      USES_TERMINAL_BUILD TRUE
      USES_TERMINAL_INSTALL TRUE
    )
    if(WIN32)
      add_dependencies(${project} dlfcn-win32_external xz_external)
    endif()
  endfunction()

  set(__AOTRITON_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
  function(aotriton_download_runtime index project)
    list(GET __AOTRITON_ROCM_LIST ${index} __AOTRITON_ROCM)
    list(GET __AOTRITON_MANYLINUX_LIST ${index} __AOTRITON_MANYLINUX)
    list(GET __AOTRITON_SHA256_LIST ${index} __AOTRITON_SHA256)

    string(CONCAT __AOTRITON_FILE "aotriton-"
                                  "${__AOTRITON_VER}-${__AOTRITON_MANYLINUX}"
                                  "_${__AOTRITON_ARCH}-${__AOTRITON_ROCM}"
                                  "-shared.tar.${__AOTRITON_Z}")
    string(CONCAT __AOTRITON_URL
           "${__AOTRITON_BASE_URL}"
           "${__AOTRITON_VER}/${__AOTRITON_FILE}")
    ExternalProject_Add(${project}
      URL "${__AOTRITON_URL}"
      URL_HASH SHA256=${__AOTRITON_SHA256}
      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton_runtime
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/aotriton_runtime"
      "${__AOTRITON_INSTALL_DIR}"
      BUILD_BYPRODUCTS "${__AOTRITON_INSTALL_DIR}/${__AOTRITON_LIB}"
    )
    message(STATUS "Using AOTriton Runtime from pre-compiled binary ${__AOTRITON_URL}.\
    Set env variables AOTRITON_INSTALL_FROM_SOURCE=1 to build from source.")
  endfunction()

  function(aotriton_download_image image project)
    list(FIND __AOTRITON_IMAGE_LIST ${image} index)
    list(GET __AOTRITON_IMAGE_SHA256_LIST ${index} __AOTRITON_SHA256)

    string(CONCAT __AOTRITON_FILE
           "aotriton-${__AOTRITON_VER}-images-"
           "${image}.tar.${__AOTRITON_Z}")
    string(CONCAT __AOTRITON_URL
           "${__AOTRITON_BASE_URL}"
           "${__AOTRITON_VER}/${__AOTRITON_FILE}")

    # Set up directories
    set(__AOTRITON_DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton_download-${image})
    set(__AOTRITON_EXTRACT_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton_image-${image})
    set(__AOTRITON_INSTALL_SOURCE_DIR ${__AOTRITON_EXTRACT_DIR})
    set(__DOWNLOAD_NO_EXTRACT "")
    set(__BUILD_COMMANDS "")

    # On Windows, we need custom tar extraction with UTF-8 support
    if(WIN32)
      set(__DOWNLOAD_NO_EXTRACT "DOWNLOAD_NO_EXTRACT;TRUE")
      set(__BUILD_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E make_directory "${__AOTRITON_EXTRACT_DIR}"
        COMMAND tar --options hdrcharset=UTF-8 -xf "${__AOTRITON_DOWNLOAD_DIR}/${__AOTRITON_FILE}" -C "${__AOTRITON_EXTRACT_DIR}"
      )
      set(__AOTRITON_INSTALL_SOURCE_DIR ${__AOTRITON_EXTRACT_DIR}/aotriton)
    endif()

    ExternalProject_Add(${project}
      URL "${__AOTRITON_URL}"
      URL_HASH SHA256=${__AOTRITON_SHA256}
      DOWNLOAD_DIR ${__AOTRITON_DOWNLOAD_DIR}
      ${__DOWNLOAD_NO_EXTRACT}
      SOURCE_DIR ${__AOTRITON_EXTRACT_DIR}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      ${__BUILD_COMMANDS}
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${__AOTRITON_INSTALL_SOURCE_DIR}"
      "${__AOTRITON_INSTALL_DIR}"
      BUILD_BYPRODUCTS
      "${__AOTRITON_INSTALL_DIR}/lib/aotriton.images/${image}/__signature__"
    )
    message(STATUS "Download AOTriton pre-compiled GPU images from ${__AOTRITON_URL}.")
  endfunction()

  # Note it is INSTALL"ED"
  if(DEFINED ENV{AOTRITON_INSTALLED_PREFIX})
    install(DIRECTORY
            $ENV{AOTRITON_INSTALLED_PREFIX}/lib
            $ENV{AOTRITON_INSTALLED_PREFIX}/include
            DESTINATION ${__AOTRITON_INSTALL_DIR})
    set(__AOTRITON_INSTALL_DIR "$ENV{AOTRITON_INSTALLED_PREFIX}")
    message(STATUS "Using Preinstalled AOTriton at ${__AOTRITON_INSTALL_DIR}")
  elseif(DEFINED ENV{AOTRITON_INSTALL_FROM_SOURCE})
    aotriton_build_from_source(OFF aotriton_external)
    add_dependencies(__caffe2_aotriton aotriton_external)
    message(STATUS "Using AOTriton compiled from source directory ${__AOTRITON_EXTERN_PREFIX}")
  else()
    set(__AOTRITON_SYSTEM_ROCM "${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}")
    list(FIND __AOTRITON_ROCM_LIST "rocm${__AOTRITON_SYSTEM_ROCM}" __AOTRITON_RUNTIME_INDEX)
    # Always build aotriton runtime from source on Windows due to lack of pre-built binaries
    if(${__AOTRITON_RUNTIME_INDEX} LESS 0 OR WIN32)
      message(STATUS "Cannot find AOTriton runtime for ROCM ${__AOTRITON_SYSTEM_ROCM}. \
      Build runtime from source")
      aotriton_build_from_source(ON aotriton_runtime)
    else()
      aotriton_download_runtime(${__AOTRITON_RUNTIME_INDEX} aotriton_runtime)
    endif()
    add_dependencies(__caffe2_aotriton aotriton_runtime)
    set(__AOTRITON_CHAINED_IMAGE "aotriton_runtime")
    foreach(image ${__AOTRITON_IMAGE_LIST})
      string(SUBSTRING ${image} 7 -1 gfx_pattern)
      string(REPLACE "x" "." gfx_regex ${gfx_pattern})
      foreach(target ${PYTORCH_ROCM_ARCH})
        if(target MATCHES ${gfx_regex})
          set(__AOTRITON_DOWNLOAD_TARGET aotriton_image_${gfx_pattern})
          aotriton_download_image(${image} ${__AOTRITON_DOWNLOAD_TARGET})
          add_dependencies(${__AOTRITON_CHAINED_IMAGE} ${__AOTRITON_DOWNLOAD_TARGET})
          set(__AOTRITON_CHAINED_IMAGE ${__AOTRITON_DOWNLOAD_TARGET})
          break()
        endif()
      endforeach()
    endforeach()
  endif()
  target_link_libraries(__caffe2_aotriton INTERFACE "${__AOTRITON_INSTALL_DIR}/${__AOTRITON_LIB}")
  target_include_directories(__caffe2_aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/include)
  set(AOTRITON_FOUND TRUE)
  # Install libaotriton_v2.so into the cmake install tree so it ends up in
  # site-packages/torch/lib/ when building with scikit-build-core.
  # aotriton's ExternalProject puts the library directly in the source tree
  # (${PROJECT_SOURCE_DIR}/torch/lib/) without a cmake install() rule, so it
  # is absent from the installed wheel and causes link failures in downstream
  # cmake builds (e.g., custom op builds) that link against installed torch.
  install(DIRECTORY "${__AOTRITON_INSTALL_DIR}/lib/"
    DESTINATION "lib"
    FILES_MATCHING PATTERN "libaotriton_v2*.so*"
  )
  # Install aotriton GPU kernel images (compressed ISA blobs) into the wheel.
  install(DIRECTORY "${__AOTRITON_INSTALL_DIR}/lib/aotriton.images"
    DESTINATION "lib"
    OPTIONAL
  )
endif() # __AOTRITON_INCLUDED

set(ATen_HIP_INCLUDE ${ATen_HIP_INCLUDE}
  "${CMAKE_CURRENT_SOURCE_DIR}"
PARENT_SCOPE)

install(FILES
          THHAtomics.cuh
          THHDeviceUtils.cuh
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THH")

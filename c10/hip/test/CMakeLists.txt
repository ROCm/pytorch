# ---[ Test binaries.

set(C10_HIP_ALL_TEST_FILES
    impl/HIPAssertionsTest_1_var_test.hip
    impl/HIPAssertionsTest_catches_stream.hip
    impl/HIPAssertionsTest_catches_thread_and_block_and_device.hip
    impl/HIPAssertionsTest_from_2_processes.hip
    impl/HIPAssertionsTest_multiple_writes_from_blocks_and_threads.hip
    impl/HIPAssertionsTest_multiple_writes_from_multiple_blocks.hip
    impl/HIPAssertionsTest_multiple_writes_from_same_block.hip
    impl/HIPTest.cpp
    impl/HIPCachingAllocatorReserveTest.cpp
    impl/HIPCachingAllocatorReserveDeviceTest.cpp
)
if(BUILD_TEST)
  foreach(test_src ${C10_HIP_ALL_TEST_FILES})
    get_filename_component(test_file_name ${test_src} NAME_WE)
    set(test_name "c10_cuda_${test_file_name}")
    if(test_src MATCHES "^.*\.hip$")
      set_source_files_properties(${test_src} PROPERTIES LANGUAGE HIP)
      add_executable(${test_name} "${test_src}")
      set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE CXX HIP_ARCHITECTURES "${PYTORCH_ROCM_ARCH}")
    else()
      add_executable(${test_name} "${test_src}")
    endif()
    target_link_libraries(${test_name} ${C10_HIP_LIB} ${C10_LIB} gmock gtest gtest_main)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    if(INSTALL_TEST)
      set_target_properties(${test_name} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
      install(TARGETS ${test_name} DESTINATION test)
    endif()
  endforeach()
endif()

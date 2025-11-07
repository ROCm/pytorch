call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat
if errorlevel 1 exit /b 1

:: Save the current working directory so that we can go back there
set CWD=%cd%

set CPP_TESTS_DIR=%TMP_DIR_WIN%\build\torch\bin
set PATH=%TMP_DIR_WIN%\build\torch\lib;%PATH%

set TORCH_CPP_TEST_MNIST_PATH=%CWD%\test\cpp\api\mnist
python tools\download_mnist.py --quiet -d %TORCH_CPP_TEST_MNIST_PATH%

python test\run_test.py --cpp --verbose -i cpp/test_api
if errorlevel 1 exit /b 1
if not errorlevel 0 exit /b 1

cd %TMP_DIR_WIN%\build\torch\test
<<<<<<< HEAD

:: Enable delayed variable expansion to make the list
setlocal enabledelayedexpansion
set EXE_LIST=
for /r "." %%a in (*.exe) do (
  if "%%~na" == "c10_intrusive_ptr_benchmark" (
    @REM NB: This is not a gtest executable file, thus couldn't be handled by
    @REM pytest-cpp and is excluded from test discovery by run_test
    call "%%~fa"
    if errorlevel 1 goto fail
    if not errorlevel 0 goto fail
  ) else (
    if "%%~na" == "verify_api_visibility" (
      @REM Skip verify_api_visibility as it is a compile-level test
    ) else (
      set EXE_LIST=!EXE_LIST! cpp/%%~na
    )
  )
)

cd %CWD%
set CPP_TESTS_DIR=%TMP_DIR_WIN%\build\torch\test

:: Run python test\run_test.py on the list
set NO_TD=True && python test\run_test.py --cpp --verbose -i !EXE_LIST!
if errorlevel 1 goto fail
if not errorlevel 0 goto fail

goto :eof
=======
for /r "." %%a in (*.exe) do (
    call :libtorch_check "%%~na" "%%~fa"
    if errorlevel 1 goto fail
)

goto :eof

:libtorch_check

cd %CWD%
set CPP_TESTS_DIR=%TMP_DIR_WIN%\build\torch\test

:: Skip verify_api_visibility as it a compile level test
if "%~1" == "verify_api_visibility" goto :eof

echo Running "%~2"
if "%~1" == "c10_intrusive_ptr_benchmark" (
  :: NB: This is not a gtest executable file, thus couldn't be handled by pytest-cpp
  call "%~2"
  goto :eof
)

python test\run_test.py --cpp --verbose -i "cpp/%~1"
if errorlevel 1 (
  echo %1 failed with exit code %errorlevel%
  goto fail
)
if not errorlevel 0 (
  echo %1 failed with exit code %errorlevel%
  goto fail
)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

:eof
exit /b 0

:fail
exit /b 1

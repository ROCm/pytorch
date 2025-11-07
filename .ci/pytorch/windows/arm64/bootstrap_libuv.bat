@echo off

echo Dependency libuv installation started.

:: Pre-check for downloads and dependencies folders
if not exist "%DOWNLOADS_DIR%" mkdir %DOWNLOADS_DIR%
if not exist "%DEPENDENCIES_DIR%" mkdir %DEPENDENCIES_DIR%

:: activate visual studio
<<<<<<< HEAD
call "%DEPENDENCIES_DIR%\VSBuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64
=======
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" arm64
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
where cl.exe

cd %DEPENDENCIES_DIR%
git clone https://github.com/libuv/libuv.git -b v1.39.0

echo Configuring libuv...
mkdir libuv\build
cd libuv\build
cmake .. -DBUILD_TESTING=OFF

echo Building libuv...
cmake --build . --config Release

echo Installing libuv...
cmake --install . --prefix ../install

:: Check if installation was successful
if %errorlevel% neq 0 (
    echo "Failed to install libuv. (exitcode = %errorlevel%)"
    exit /b 1
)

echo Dependency libuv installation finished.
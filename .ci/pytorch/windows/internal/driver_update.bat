<<<<<<< HEAD
set WIN_DRIVER_VN=580.88
set "DRIVER_DOWNLOAD_LINK=https://ossci-windows.s3.amazonaws.com/%WIN_DRIVER_VN%-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe" & REM @lint-ignore
curl --retry 3 -kL %DRIVER_DOWNLOAD_LINK% --output %WIN_DRIVER_VN%-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe
if errorlevel 1 exit /b 1

start /wait %WIN_DRIVER_VN%-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe -s -noreboot
if errorlevel 1 exit /b 1

del %WIN_DRIVER_VN%-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe || ver > NUL
=======
set WIN_DRIVER_VN=528.89
set "DRIVER_DOWNLOAD_LINK=https://ossci-windows.s3.amazonaws.com/%WIN_DRIVER_VN%-data-center-tesla-desktop-winserver-2016-2019-2022-dch-international.exe" & REM @lint-ignore
curl --retry 3 -kL %DRIVER_DOWNLOAD_LINK% --output %WIN_DRIVER_VN%-data-center-tesla-desktop-winserver-2016-2019-2022-dch-international.exe
if errorlevel 1 exit /b 1

start /wait %WIN_DRIVER_VN%-data-center-tesla-desktop-winserver-2016-2019-2022-dch-international.exe -s -noreboot
if errorlevel 1 exit /b 1

del %WIN_DRIVER_VN%-data-center-tesla-desktop-winserver-2016-2019-2022-dch-international.exe || ver > NUL
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

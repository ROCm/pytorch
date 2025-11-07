@echo off

<<<<<<< HEAD
curl -k https://www.7-zip.org/a/7z1805-x64.exe -O
=======
curl -k -L "https://sourceforge.net/projects/sevenzip/files/7-Zip/18.05/7z1805-x64.exe/download" -o 7z1805-x64.exe
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
if errorlevel 1 exit /b 1

start /wait 7z1805-x64.exe /S
if errorlevel 1 exit /b 1

set "PATH=%ProgramFiles%\7-Zip;%PATH%"

@echo off
setlocal
cd /d "%~dp0"
if not exist build mkdir build

REM ---------------------------------------------------------------------------
REM  POINT TEMP INTO THE WORKSPACE.
REM
REM  nvcc writes its intermediate files (tmpxft_*) to TEMP, and cmake probes the
REM  compiler by invoking nvcc. Where TEMP is not writable for the account running
REM  the build, that probe fails with
REM      nvcc fatal : Could not open output file
REM  and cmake reports Configuring incomplete -- which reads like a broken
REM  toolchain rather than a permissions problem.
REM
REM  This was masked for a long time because the shell I built from had a writable
REM  TEMP. A plain git clone followed by build.bat on a fresh account failed, and
REM  the fix belongs here rather than in the caller: a build script that works only
REM  under one shell environment is not a build script.
REM ---------------------------------------------------------------------------
set "TEMP=%CD%\build\tmp"
set "TMP=%CD%\build\tmp"
if not exist "%TEMP%" mkdir "%TEMP%"

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
if errorlevel 1 ( echo [FAIL] vcvarsall & exit /b 1 )
set "NINJA=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if exist build\CMakeCache.txt del /q build\CMakeCache.txt
cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release >build\cmake.log 2>&1
if errorlevel 1 ( echo [FAIL] cmake configure -- see build\cmake.log & exit /b 2 )
cmake --build build >build\build.log 2>&1
if errorlevel 1 ( echo [FAIL] build -- see build\build.log & exit /b 3 )
if not exist build\canticle.exe ( echo [FAIL] exe missing & exit /b 4 )
echo [OK] build\canticle.exe
endlocal

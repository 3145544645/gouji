@echo off
setlocal
cd /d "%~dp0"
if not exist build mkdir build
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

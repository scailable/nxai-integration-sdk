@echo off
setlocal enabledelayedexpansion

rem The Windows verify entry point for the offline SHM harness.
rem
rem Configures, builds and runs the harness suite with MSVC and exits non-zero on any failing
rem case. It enters the MSVC developer environment itself, so it can be started from any shell
rem -- do not wrap it in vcvars first.
rem
rem Usage: verify-windows.cmd [build-dir]        (default: build-windows)
rem
rem Only the C++ targets are built. A bare `cmake --build` would also package every Python
rem processor with Nuitka, which the suite does not use: the Windows cases are the C++
rem scenarios and the launcher's own tests.

set "REPO_ROOT=%~dp0"
set "BUILD_DIR=%~1"
if "%BUILD_DIR%"=="" set "BUILD_DIR=%REPO_ROOT%build-windows"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo verify-windows: vswhere.exe not found -- is Visual Studio installed?
    exit /b 2
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_PATH=%%i"
if not defined VS_PATH (
    echo verify-windows: no Visual Studio installation with the C++ toolset was found.
    exit /b 2
)

call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul || (
    echo verify-windows: could not enter the MSVC environment.
    exit /b 2
)

rem Ninja and CMake both ship with the VS C++ toolset, so vcvars64 above is enough to have
rem them on PATH even without a separate CMake installation.
cmake -S "%REPO_ROOT%." -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Debug || exit /b 1
cmake --build "%BUILD_DIR%" --target e2e-postprocessor-harness e2e-child-process-test e2e-ipc-sync-obj-test e2e-ipc-failure-paths-test e2e-ipc-exit-signal-test e2e-processor-loop-test postprocessor-cpp-objects-example postprocessor-cpp-tensors-example || exit /b 1
ctest --test-dir "%BUILD_DIR%" --output-on-failure || exit /b 1

echo verify-windows: PASS
exit /b 0

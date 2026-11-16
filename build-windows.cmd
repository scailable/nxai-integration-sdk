@echo off

rem The Windows build entry point.
rem
rem Enters the MSVC developer environment, configures build-windows if it has no CMake cache
rem yet, and builds the targets given as arguments. It can be started from any shell -- do not
rem wrap it in vcvars first.
rem
rem There is no setlocal: a caller that uses `call` keeps the MSVC environment, so
rem verify-windows.cmd can run ctest after it.
rem
rem Usage: build-windows.cmd [target ...]
rem
rem With no target it only configures. Always name the targets: a bare `cmake --build` would
rem package every Python processor with Nuitka.

set "REPO_ROOT=%~dp0"
set "BUILD_DIR=%REPO_ROOT%build-windows"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo build-windows: vswhere.exe not found -- is Visual Studio installed?
    exit /b 2
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_PATH=%%i"
if not defined VS_PATH (
    echo build-windows: no Visual Studio installation with the C++ toolset was found.
    exit /b 2
)

call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul || (
    echo build-windows: could not enter the MSVC environment.
    exit /b 2
)

rem Ninja and CMake both ship with the VS C++ toolset, so vcvars64 above is enough to have
rem them on PATH even without a separate CMake installation.
if not exist "%BUILD_DIR%\CMakeCache.txt" (
    cmake -S "%REPO_ROOT%." -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Debug || exit /b 1
)

if "%~1"=="" exit /b 0
cmake --build "%BUILD_DIR%" --target %* || exit /b 1
exit /b 0

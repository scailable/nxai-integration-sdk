@echo off
setlocal

rem The Windows verify entry point for the offline SHM harness.
rem
rem Builds the harness suite with build-windows.cmd, runs it and exits non-zero on any failing
rem case. It can be started from any shell -- do not wrap it in vcvars first.
rem
rem Usage: verify-windows.cmd
rem
rem Only the C++ targets are built. The Windows cases are the C++ scenarios and the launcher's
rem own tests; no case uses a Nuitka-packaged Python processor.

set "REPO_ROOT=%~dp0"

call "%REPO_ROOT%build-windows.cmd" e2e-postprocessor-harness e2e-child-process-test e2e-ipc-sync-obj-test e2e-ipc-failure-paths-test e2e-ipc-exit-signal-test e2e-processor-loop-test postprocessor-cpp-objects-example postprocessor-cpp-tensors-example || exit /b 1
ctest --test-dir "%REPO_ROOT%build-windows" --output-on-failure || exit /b 1

echo verify-windows: PASS
exit /b 0

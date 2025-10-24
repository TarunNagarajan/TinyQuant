@echo off
REM Ensure we are in the project root directory
pushd "%~dp0"
build\Release\model-inspection.exe llama\llama.cpp\build\bin\Release\tinyllama.gguf
popd


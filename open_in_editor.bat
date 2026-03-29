@echo off
setlocal

if "%~1"=="" (
  echo Usage:
  echo   open_in_editor.bat ^<file1^> [file2] [file3] ...
  echo Example:
  echo   open_in_editor.bat docs\PROJECT_SCAN_SUMMARY_2026-03-30.md
  exit /b 1
)

:loop
if "%~1"=="" goto done
code -r "%~1"
timeout /t 1 /nobreak >nul
shift
goto loop

:done
exit /b 0

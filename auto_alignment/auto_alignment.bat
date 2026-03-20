@echo off
set "SCRIPT_DIR=%~dp0"
set "ENV_DIR=%SCRIPT_DIR%env"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "ENTRY_SCRIPT=%PROJECT_DIR%\phonetic_toolbox\gui\dialogs\mfa_auto_alignment_standalone.py"

echo Initializing Environment... Please wait...
call "%ENV_DIR%\Scripts\activate.bat"

echo Starting AutoAnnotation...
python "%ENTRY_SCRIPT%"

if %ERRORLEVEL% NEQ 0 (
    echo Application exited with error code %ERRORLEVEL%
    pause
)
